# Minimal Telegram Quiz Bot – stable, low-traffic, admin-only
from __future__ import annotations
import os, json, time, random, html, logging, threading, uuid, asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut, NetworkError, BadRequest
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("quiz-min")

# ---------- Config ----------
QUESTION_TIME = int(os.getenv("QUESTION_TIME", "10"))    # seconds
DELAY_NEXT    = int(os.getenv("DELAY_NEXT", "0"))        # seconds (keep 0 for speed)
QUESTIONS_FILE = os.getenv("QUESTIONS_FILE", "questions.json")

MODES = ("beginner", "standard", "expert")
ALLOWED_SESSION_SIZES = (10, 20, 30, 40, 50)
ADMINS_ONLY = True
INSTANCE_ID = os.getenv("RAILWAY_REPLICA_ID") or str(uuid.uuid4())[:8]

def esc(s: str) -> str:
    return html.escape(str(s), quote=True)

# ---------- Safe Telegram call ----------
async def tg(desc: str, coro, *args, **kwargs):
    """Retry a few times on common transient errors; otherwise swallow & continue."""
    for attempt in range(1, 4):
        try:
            return await coro(*args, **kwargs)
        except RetryAfter as e:
            wait = float(getattr(e, "retry_after", 1.0)) + 0.5
            log.warning("%s: RetryAfter %.2fs (attempt %d/3)", desc, wait, attempt)
            await asyncio.sleep(wait)
        except (TimedOut, NetworkError) as e:
            wait = 1.5 * attempt
            log.warning("%s: %s; backoff %.1fs (attempt %d/3)", desc, type(e).__name__, wait, attempt)
            await asyncio.sleep(wait)
        except BadRequest as e:
            log.debug("%s: BadRequest ignored: %s", desc, e); return None
        except Exception as e:
            log.error("%s: unexpected %s: %s", desc, type(e).__name__, e); return None
    return None

# ---------- Data ----------
@dataclass
class QItem:
    text: str
    options: List[str]
    correct: int
    mode: str

@dataclass
class GameState:
    chat_id: int
    started_by: int
    questions: List[QItem]
    limit: int
    mode: str

    q_index: int = 0
    q_msg_id: Optional[int] = None
    q_start_ts: Optional[float] = None
    locked: bool = False

    players: Dict[int, str] = field(default_factory=dict)
    totals: Dict[int, int] = field(default_factory=dict)      # simple +1 per correct
    per_q_answers: Dict[int, Dict[int, int]] = field(default_factory=dict)  # uid -> option
    answered_now: Dict[int, Set[int]] = field(default_factory=dict)         # uid set

GAMES: Dict[int, GameState] = {}
SETTINGS: Dict[int, Dict[str, int | str]] = {}

# ---------- Load questions ----------
def load_questions() -> List[QItem]:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[QItem] = []
    for i, q in enumerate(data, start=1):
        if not all(k in q for k in ("text","options","correct","mode")):
            raise ValueError(f"Question {i} missing fields")
        if q["mode"] not in MODES:
            raise ValueError(f"Question {i} invalid mode {q['mode']}")
        if len(q["options"]) != 4:
            raise ValueError(f"Question {i} must have exactly 4 options")
        c = int(q["correct"])
        if not 0 <= c < 4:
            raise ValueError(f"Question {i} invalid correct index {c}")
        out.append(QItem(q["text"], list(q["options"]), c, q["mode"]))
    return out

def filter_by_mode(qs: List[QItem], mode: str) -> List[QItem]:
    return [q for q in qs if q.mode == mode]

def shuffle_qs(qs: List[QItem]) -> List[QItem]:
    qs = qs.copy(); random.shuffle(qs); out=[]
    for q in qs:
        pairs=list(enumerate(q.options)); random.shuffle(pairs)
        new_opts=[t for _,t in pairs]; new_correct=next(i for i,(orig,_) in enumerate(pairs) if orig==q.correct)
        out.append(QItem(q.text,new_opts,new_correct,q.mode))
    return out

def kb_answers(q: QItem, qnum: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(opt, callback_data=f"ans:{qnum}:{i}")]
                                 for i, opt in enumerate(q.options)])

def fmt_q(qnum: int, total: int, q: QItem) -> str:
    return f"❓ <b>Question {qnum}/{total}</b>\n{esc(q.text)}\n\n⏱ <b>{QUESTION_TIME}s</b> to answer"

# ---------- Jobs ----------
async def close_question(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id)
    if not st:
        return
    st.locked = True

    # Freeze the question message (remove keyboard)
    try:
        await context.bot.edit_message_reply_markup(chat_id=chat_id, message_id=st.q_msg_id, reply_markup=None)
    except Exception:
        pass

    # Tally & recap
    qidx = st.q_index
    q = st.questions[qidx]
    per = st.per_q_answers.get(qidx, {})
    correcters = []
    for uid, choice in per.items():
        if choice == q.correct:
            st.totals[uid] = st.totals.get(uid, 0) + 1
            correcters.append(uid)

    if correcters:
        names = [esc(st.players.get(uid, str(uid))) for uid in correcters[:10]]
        recap = f"✅ Correct: <b>{esc(q.options[q.correct])}</b>\n🏁 Correct answers: " + ", ".join(names)
    else:
        recap = f"✅ Correct: <b>{esc(q.options[q.correct])}</b>\n😶 Nobody got it."

    await tg("recap.send", context.bot.send_message, chat_id=chat_id, text=recap, parse_mode=ParseMode.HTML)

    # Next step
    st.q_index += 1
    if st.q_index >= st.limit:
        await finish_quiz(context, st)
    else:
        await asyncio.sleep(max(0, DELAY
