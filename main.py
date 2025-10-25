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
        await asyncio.sleep(max(0, DELAY_NEXT))
        await ask_question(context, st)

async def ask_question(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    q = st.questions[st.q_index]
    m = await tg("ask.send", context.bot.send_message, chat_id=st.chat_id,
                 text=fmt_q(st.q_index+1, st.limit, q), reply_markup=kb_answers(q, st.q_index+1),
                 parse_mode=ParseMode.HTML)
    if not m:
        log.error("Failed to send question. Ending session.")
        await finish_quiz(context, st)
        return
    st.q_msg_id = m.message_id
    st.q_start_ts = time.time()
    st.locked = False
    st.answered_now[st.q_index] = set()
    # Close this question after QUESTION_TIME
    context.job_queue.run_once(close_question, when=QUESTION_TIME, data={"chat_id": st.chat_id}, name=f"close:{st.chat_id}:{st.q_index}")

async def finish_quiz(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    # Simple leaderboard
    if not st.totals:
        await tg("finish.none", context.bot.send_message, chat_id=st.chat_id, text="Quiz finished. No correct answers recorded.")
    else:
        top = sorted(st.totals.items(), key=lambda x: x[1], reverse=True)
        lines = ["🏁 <b>Final Results</b>"]
        for rank, (uid, score) in enumerate(top[:10], start=1):
            name = esc(st.players.get(uid, str(uid)))
            medal = ("🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}.")
            lines.append(f"{medal} {name} — {score} correct")
        await tg("finish.board", context.bot.send_message, chat_id=st.chat_id, text="\n".join(lines), parse_mode=ParseMode.HTML)
    GAMES.pop(st.chat_id, None)

# ---------- Admin helper ----------
async def is_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
    try:
        m = await context.bot.get_chat_member(chat_id, user_id)
        return m.status in ("administrator", "creator")
    except Exception:
        return False

# ---------- Commands ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✨ <b>Quiz Bot</b>\n"
        "1) /menu — choose Mode & Length\n"
        "2) Tap ▶️ Start or use /startquiz <mode> <length>\n"
        "Admin-only in groups. Try /ping to check health.",
        parse_mode=ParseMode.HTML
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"pong from {INSTANCE_ID}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, update.effective_chat.id, update.effective_user.id):
            await update.message.reply_text("Only group admins can configure the quiz here.")
            return
    rows = [[InlineKeyboardButton("Beginner", callback_data="cfg:mode:beginner"),
             InlineKeyboardButton("Standard", callback_data="cfg:mode:standard"),
             InlineKeyboardButton("Expert",   callback_data="cfg:mode:expert")]]
    await update.message.reply_text("<b>Step 1/2</b>: Choose a <b>Mode</b>.", reply_markup=InlineKeyboardMarkup(rows), parse_mode=ParseMode.HTML)

async def cfg_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    if not q.data.startswith("cfg:"): return
    chat_id = q.message.chat.id

    if q.message.chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, chat_id, q.from_user.id):
            try: await q.answer("Only group admins can change quiz settings here.", show_alert=True)
            except Exception: pass
            return

    _, kind, val = q.data.split(":")
    if kind == "mode":
        if val not in MODES:
            await q.edit_message_text("Invalid mode."); return
        SETTINGS.setdefault(chat_id, {})["mode"] = val
        try: await q.edit_message_text(f"Mode set to <b>{esc(val.title())}</b> ✅", parse_mode=ParseMode.HTML)
        except Exception: pass
        # show lengths
        rows = [[InlineKeyboardButton(str(n), callback_data=f"cfg:len:{n}") for n in (10,20,30)],
                [InlineKeyboardButton(str(n), callback_data=f"cfg:len:{n}") for n in (40,50)]]
        await tg("len.send", context.bot.send_message, chat_id=chat_id,
                 text="<b>Step 2/2</b>: How many questions?", reply_markup=InlineKeyboardMarkup(rows), parse_mode=ParseMode.HTML)
        return

    if kind == "len":
        try: length = int(val)
        except: await q.edit_message_text("Invalid length."); return
        if length not in ALLOWED_SESSION_SIZES:
            await q.edit_message_text("Invalid length."); return
        SETTINGS.setdefault(chat_id, {})["length"] = length
        mode = SETTINGS.get(chat_id, {}).get("mode")
        await q.edit_message_text(f"Length set to <b>{length}</b> ✅", parse_mode=ParseMode.HTML)
        if mode in MODES:
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("▶️ Start quiz", callback_data=f"start:{mode}:{length}")]])
            await tg("start.btn", context.bot.send_message, chat_id=chat_id,
                     text=f"Ready to start <b>{esc(mode.title())}</b> • {length} Qs?",
                     reply_markup=kb, parse_mode=ParseMode.HTML)

async def start_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    try:
        _, mode, length_s = q.data.split(":"); length = int(length_s)
    except Exception:
        return
    chat_id = q.message.chat.id
    if q.message.chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, chat_id, q.from_user.id):
            try: await q.answer("Only group admins can start here.", show_alert=True)
            except Exception: pass
            return
    await do_startquiz(context, chat_id, q.from_user.id, mode, length)

async def cmd_startquiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id; user = update.effective_user
    if update.effective_chat.type != "private":
        if not await is_admin(context, chat_id, user.id):
            await update.message.reply_text("Only group admins can start a quiz here."); return
    mode = None; length = None
    if context.args and len(context.args) >= 2:
        mode = str(context.args[0]).lower()
        try: length = int(context.args[1])
        except: pass
    if mode not in MODES or length not in ALLOWED_SESSION_SIZES:
        cfg = SETTINGS.get(chat_id, {})
        mode = mode or cfg.get("mode")
        length = length or cfg.get("length")
    if mode not in MODES or length not in ALLOWED_SESSION_SIZES:
        await update.message.reply_text("Use /menu first, or /startquiz <mode> <length> (e.g. /startquiz beginner 10)")
        return
    await do_startquiz(context, chat_id, user.id, mode, length)

async def do_startquiz(context: ContextTypes.DEFAULT_TYPE, chat_id: int, starter_user_id: int, mode: str, length: int):
    pool = filter_by_mode(load_questions(), mode)
    if len(pool) < length:
        await tg("start.notenough", context.bot.send_message, chat_id=chat_id,
                 text=f"Not enough questions for <b>{esc(mode)}</b> (need {length}).", parse_mode=ParseMode.HTML)
        return
    qs = shuffle_qs(pool)[:length]
    st = GameState(chat_id=chat_id, started_by=starter_user_id, questions=qs, limit=length, mode=mode)
    GAMES[chat_id] = st
    await tg("intro", context.bot.send_message, chat_id=chat_id,
             text=f"🎯 <b>{esc(mode.title())}</b> • {length} questions\n⏱ {QUESTION_TIME}s per question",
             parse_mode=ParseMode.HTML)
    await ask_question(context, st)

async def on_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; chat_id = q.message.chat.id; user = q.from_user
    try:
        _, qnum_s, opt_s = q.data.split(":"); qnum = int(qnum_s); opt = int(opt_s)
    except Exception:
        return
    st = GAMES.get(chat_id)
    if not st: 
        try: await q.answer("No active quiz here.")
        except: pass
        return
    if st.locked or st.q_index + 1 != qnum:
        try: await q.answer("Too late for this one."); 
        except: pass
        return
    st.answered_now.setdefault(st.q_index, set())
    if user.id in st.answered_now[st.q_index]:
        try: await q.answer("Only your first answer counts.")
        except: pass
        return
    st.answered_now[st.q_index].add(user.id)
    st.players[user.id] = (user.full_name or user.username or str(user.id))[:64]
    st.per_q_answers.setdefault(st.q_index, {})[user.id] = opt
    try: await q.answer("Locked in!")
    except: pass

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if update.effective_chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, chat_id, update.effective_user.id):
            await update.message.reply_text("Only group admins can stop here."); return
    st = GAMES.get(chat_id)
    if not st:
        await update.message.reply_text("No active quiz.")
        return
    st.limit = st.q_index + 1
    st.locked = True
    await finish_quiz(context, st)

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if update.effective_chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, chat_id, update.effective_user.id):
            await update.message.reply_text("Only group admins can reset here."); return
    GAMES.pop(chat_id, None); SETTINGS.pop(chat_id, None)
    await update.message.reply_text("✅ Reset. Use /menu then /startquiz.")

# ---------- Keepalive for Web services ----------
def start_keepalive_server():
    port = os.getenv("PORT")
    if not port: 
        log.info("No $PORT detected — assuming Worker mode.")
        return
    try: port = int(port)
    except: 
        log.warning("Invalid PORT=%r; skipping keepalive", os.getenv("PORT"))
        return
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type","text/plain; charset=utf-8")
            self.end_headers(); self.wfile.write(b"ok")
        def log_message(self, *a, **k): return
    threading.Thread(target=lambda: HTTPServer(("0.0.0.0", port), H).serve_forever(), daemon=True).start()

# ---------- Error handler ----------
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("Unhandled error: %s", context.error)

# ---------- Build/Run ----------
def build_app() -> Application:
    token = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("Set BOT_TOKEN (or TELEGRAM_BOT_TOKEN / TELEGRAM_TOKEN) env var.")

    builder = Application.builder().token(token)
    # Make rate limiter optional (won't crash if extra not installed)
    try:
        from telegram.ext import AIORateLimiter
        builder = builder.rate_limiter(AIORateLimiter())
    except Exception as e:
        log.info("Rate limiter unavailable; continuing without it: %s", e)

    app = builder.build()
    app.add_error_handler(on_error)

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("ping",   cmd_ping))
    app.add_handler(CommandHandler("menu",   cmd_menu))
    app.add_handler(CommandHandler("startquiz", cmd_startquiz))
    app.add_handler(CommandHandler("stop",   cmd_stop))
    app.add_handler(CommandHandler("reset",  cmd_reset))

    app.add_handler(CallbackQueryHandler(cfg_click,   pattern=r"^cfg:"))
    app.add_handler(CallbackQueryHandler(start_click, pattern=r"^start:(beginner|standard|expert):\d+$"))
    app.add_handler(CallbackQueryHandler(on_answer,   pattern=r"^ans:\d+:\d$"))
    return app

if __name__ == "__main__":
    log.info("Starting quiz bot…")
    start_keepalive_server()
    app = build_app()

    # No post_init here (to be compatible with your PTB build)
    app.run_polling(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True,
        close_loop=False
    )
