# Telegram Quiz Bot — precise timing (monotonic), scoped close job, admin-only controls
from __future__ import annotations
import os, json, time, math, html, logging, threading, uuid, asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut, NetworkError, BadRequest
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, ContextTypes, filters

# Optional: psutil for memory logging (install if you want: pip install psutil)
try:
    import psutil
except Exception:
    psutil = None

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("quizbot")

def log_mem(tag=""):
    try:
        if psutil:
            p = psutil.Process(os.getpid())
            rss_mb = p.memory_info().rss / 1024 / 1024
            log.info("%s: memory=%.1fMB", tag, rss_mb)
        else:
            # fallback: only log pid
            log.info("%s: pid=%s", tag, os.getpid())
    except Exception:
        log.exception("mem logging failed")

# ---------- env helpers ----------
def _int(name, d, lo, hi):
    v = os.getenv(name)
    if v is None: return d
    try:
        x = int(v); return max(lo, min(hi, x))
    except Exception: return d

def _float(name, d, lo, hi):
    v = os.getenv(name)
    if v is None: return d
    try:
        x = float(v); return max(lo, min(hi, x))
    except Exception: return d

def _bool(name, d):
    v = os.getenv(name)
    if v is None: return d
    return v.strip().lower() in ("1","true","yes","y","on")

TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("TOKEN")
QUESTION_TIME = _int("QUESTION_TIME", 10, 3, 120)
DELAY_NEXT = _int("DELAY_NEXT", 3, 1, 60)
TICK_SECONDS = _float("TICK_SECONDS", 1.0, 0.5, 5.0)
POINTS_MAX = _int("POINTS_MAX", 100, 10, 1000)
QUESTIONS_FILE = os.getenv("QUESTIONS_FILE", "questions.json")
ADMINS_ONLY = _bool("ADMINS_ONLY", True)
ALLOWED_SESSION_SIZES = (10,20,30,40,50)
MODES = ("beginner","standard","expert")
INSTANCE_ID = os.getenv("RAILWAY_REPLICA_ID") or str(uuid.uuid4())[:8]

log.info("CONFIG: TOKEN=%s QUESTION_TIME=%s DELAY_NEXT=%s TICK_SECONDS=%s ADMINS_ONLY=%s",
         ("SET" if TOKEN else "MISSING"), QUESTION_TIME, DELAY_NEXT, TICK_SECONDS, ADMINS_ONLY)
log_mem("startup")

def esc(s: str) -> str:
    return html.escape(str(s), quote=True)

# ---------- _tg safe wrapper ----------
async def _tg(desc: str, coro, *args, **kwargs):
    attempts = 0
    while True:
        attempts += 1
        try:
            return await coro(*args, **kwargs)
        except RetryAfter as e:
            wait = float(getattr(e, "retry_after", 1.0)) + 0.5
            log.warning("%s: RetryAfter %.2fs (attempt %d)", desc, wait, attempts)
            if attempts >= 3:
                log.warning("%s: giving up after RetryAfter", desc)
                return None
            await asyncio.sleep(wait)
        except (TimedOut, NetworkError) as e:
            wait = min(2.0 * attempts, 6.0)
            log.warning("%s: %s; backoff %.2fs (attempt %d)", desc, type(e).__name__, wait, attempts)
            if attempts >= 3:
                log.warning("%s: giving up after network errors", desc); return None
            await asyncio.sleep(wait)
        except BadRequest as e:
            log.debug("%s: BadRequest ignored: %s", desc, e); return None
        except Exception:
            log.exception("%s: unexpected exception", desc); return None

# ---------- data ----------
@dataclass
class QItem:
    text: str
    options: List[str]
    correct: int
    mode: str

@dataclass
class AnswerRec:
    choice: int
    is_correct: bool
    elapsed: float
    points: int

@dataclass
class GameState:
    chat_id: int
    started_by: int
    questions: List[QItem]
    limit: int
    mode: str
    q_index: int = 0
    q_msg_id: Optional[int] = None
    q_start_mono: Optional[float] = None
    q_end_mono: Optional[float] = None
    locked: bool = False
    per_q_answers: Dict[int, Dict[int, AnswerRec]] = field(default_factory=dict)
    answered_now: Dict[int, Set[int]] = field(default_factory=dict)
    players: Dict[int, str] = field(default_factory=dict)
    totals: Dict[int, int] = field(default_factory=dict)
    corrects: Dict[int, int] = field(default_factory=dict)

GAMES: Dict[int, GameState] = {}

def points_for_elapsed(elapsed: float) -> int:
    if elapsed >= QUESTION_TIME: return 0
    return int(round((max(0.0, QUESTION_TIME - elapsed) / QUESTION_TIME) * POINTS_MAX))

def answer_kb(q: QItem, qnum: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(opt, callback_data=f"ans:{qnum}:{i}")]
                                 for i, opt in enumerate(q.options)])

def fmt_q(qnum: int, total: int, q: QItem, left: Optional[int] = None, locked_count: Optional[int] = None) -> str:
    s = f"❓ <b>Question {qnum}/{total}</b>\n{esc(q.text)}"
    if left is not None: s += f"\n\n⏱ <b>{int(left)}s left</b>"
    if locked_count is not None: s += f"\n🗳 <b>Answers locked:</b> {locked_count}"
    return s

def load_questions() -> List[QItem]:
    if not os.path.exists(QUESTIONS_FILE):
        raise FileNotFoundError(f"{QUESTIONS_FILE} missing")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for i,q in enumerate(data, start=1):
        if not all(k in q for k in ("text","options","correct","mode")):
            raise ValueError(f"Q{i} missing fields")
        if q["mode"] not in MODES: raise ValueError(f"Q{i} invalid mode")
        if not isinstance(q["options"], list) or len(q["options"])!=4:
            raise ValueError(f"Q{i} must have 4 options")
        c = int(q["correct"])
        if not 0<=c<4: raise ValueError(f"Q{i} invalid correct")
        out.append(QItem(q["text"], list(q["options"]), c, q["mode"]))
    return out

def by_mode(all_qs: List[QItem], mode: str) -> List[QItem]:
    return [q for q in all_qs if q.mode == mode]

import random
def shuffle_qs(qs: List[QItem]) -> List[QItem]:
    qs = qs.copy(); random.shuffle(qs)
    out = []
    for q in qs:
        pairs = list(enumerate(q.options)); random.shuffle(pairs)
        new_opts = [t for _,t in pairs]
        new_correct = next(i for i,(orig,_) in enumerate(pairs) if orig == q.correct)
        out.append(QItem(q.text, new_opts, new_correct, q.mode))
    return out

def cancel_jobs(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    try:
        for job in context.job_queue.jobs():
            name = getattr(job, "name", "") or ""
            if name.startswith(f"tick:{chat_id}:") or name.startswith(f"close:{chat_id}:") or name.startswith(f"gap:{chat_id}:"):
                job.schedule_removal()
                log.info("cancel_jobs: removed job %s", name)
    except Exception:
        log.exception("job cleanup error")

# ---------- jobs & lifecycle (with logs) ----------
async def tick_edit(context: ContextTypes.DEFAULT_TYPE):
    d = context.job.data
    chat_id = d["chat_id"]; msg_id = d["msg_id"]; qidx = d["qidx"]
    st = GAMES.get(chat_id)
    if not st or st.q_index != qidx or st.q_end_mono is None:
        context.job.schedule_removal(); return
    left = max(0, int(math.ceil(st.q_end_mono - time.monotonic())))
    if left <= 0:
        context.job.schedule_removal(); return
    log.debug("tick_edit: chat=%s qidx=%d left=%d msg=%s", chat_id, qidx, left, msg_id)
    # IMPORTANT: do NOT pass reply_markup here -> keyboard remains
    await _tg("tick.edit", context.bot.edit_message_text,
              chat_id=chat_id, message_id=msg_id,
              text=fmt_q(st.q_index+1, st.limit, st.questions[st.q_index], left,
                         len(st.per_q_answers.get(st.q_index, {}))),
              parse_mode=ParseMode.HTML)

async def gap_tick(context: ContextTypes.DEFAULT_TYPE):
    d = context.job.data
    chat_id = d["chat_id"]; msg_id = d["msg_id"]; end_mono = d["end_mono"]
    st = GAMES.get(chat_id)
    if not st: context.job.schedule_removal(); return
    left = max(0, int(math.ceil(end_mono - time.monotonic())))
    if left <= 0:
        context.job.schedule_removal()
        await _tg("gap.final", context.bot.edit_message_text, chat_id=chat_id, message_id=msg_id, text="🚀 Starting next question…")
        return
    await _tg("gap.tick", context.bot.edit_message_text, chat_id=chat_id, message_id=msg_id, text=f"⏭️ Next question in {left}s…")

async def post_round_recap(context: ContextTypes.DEFAULT_TYPE, st: GameState, qidx: int):
    q = st.questions[qidx]
    per = st.per_q_answers.get(qidx, {})
    scorers = sorted(((uid, rec.points) for uid, rec in per.items() if rec.is_correct), key=lambda t:t[1], reverse=True)
    scores, corrects = st.totals, st.corrects
    ranking = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    lines = [f"📘 <b>Q{qidx+1} Result</b>", f"✅ Correct answer: <b>{esc(q.options[q.correct])}</b>"]
    if scorers:
        names = [f"{esc(st.players.get(uid, str(uid)))} (+{pts})" for uid, pts in scorers[:5]]
        extra = len(scorers) - 5
        lines.append("🏎️ Fastest correct: " + ", ".join(names) + (f" … +{extra} more" if extra > 0 else ""))
    else:
        lines.append("😶 No correct answers this round.")
    if ranking:
        lines.append("\n🏁 <b>Current Leaderboard</b> (Top 5)")
        for rank,(uid,total) in enumerate(ranking[:5], start=1):
            name = esc(st.players.get(uid, str(uid))); corr = corrects.get(uid, 0)
            medal = ("🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}.")
            lines.append(f"{medal} {name} — {corr} correct — {total} pts")
    await _tg("recap.send", context.bot.send_message, chat_id=st.chat_id, text="\n".join(lines), parse_mode=ParseMode.HTML)

async def close_question(context: ContextTypes.DEFAULT_TYPE):
    d = context.job.data
    chat_id = d.get("chat_id"); qidx_job = d.get("qidx")
    log.info("close_question called: chat=%s qidx_job=%s", chat_id, qidx_job)
    st = GAMES.get(chat_id)
    if not st or st.q_end_mono is None:
        log.info("close_question: no state or q_end_mono None for chat=%s", chat_id); return
    if qidx_job is None or st.q_index != qidx_job:
        log.info("close_question: stale job: st.q_index=%s qidx_job=%s -> remove", st.q_index, qidx_job)
        context.job.schedule_removal(); return
    now = time.monotonic(); remaining = st.q_end_mono - now
    log.info("close_question: now=%.3f end=%.3f remaining=%.3f chat=%s", now, st.q_end_mono, remaining, chat_id)
    if remaining > 0.05:
        context.job_queue.run_once(close_question, when=remaining, data={"chat_id": chat_id, "qidx": qidx_job}, name=f"close:{chat_id}:{qidx_job}")
        log.info("close_question: re-armed for chat=%s qidx=%s remaining=%.3f", chat_id, qidx_job, remaining); return
    if st.locked: return
    st.locked = True
    await _tg("close.freeze", context.bot.edit_message_text,
              chat_id=chat_id, message_id=st.q_msg_id,
              text=fmt_q(st.q_index+1, st.limit, st.questions[st.q_index], 0,
                         len(st.per_q_answers.get(st.q_index, {}))),
              reply_markup=None, parse_mode=ParseMode.HTML)
    await post_round_recap(context, st, st.q_index)
    st.per_q_answers.pop(st.q_index, None); st.answered_now.pop(st.q_index, None)
    gap_end = time.monotonic() + DELAY_NEXT
    m = await _tg("gap.send", context.bot.send_message, chat_id=chat_id, text=f"⏭️ Next question in {DELAY_NEXT}s…")
    if m and getattr(m, "message_id", None):
        context.job_queue.run_repeating(gap_tick, interval=max(1.0, TICK_SECONDS), first=max(1.0, TICK_SECONDS),
                                        data={"chat_id": chat_id, "msg_id": m.message_id, "end_mono": gap_end},
                                        name=f"gap:{chat_id}:{st.q_index}")
    context.job_queue.run_once(next_question, when=DELAY_NEXT, data={"chat_id": chat_id})

async def next_question(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id); 
    if not st: return
    st.q_index += 1
    if st.q_index >= st.limit:
        await finish_quiz(context, st); return
    await ask_question(context, st)

async def ask_question(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    q = st.questions[st.q_index]
    m = await _tg("ask.send", context.bot.send_message, chat_id=st.chat_id,
                  text=fmt_q(st.q_index+1, st.limit, q, QUESTION_TIME, 0),
                  reply_markup=answer_kb(q, st.q_index),
                  parse_mode=ParseMode.HTML)
    if not m:
        log.error("ask_question: failed to send; finishing")
        await finish_quiz(context, st); return
    st.q_msg_id = getattr(m, "message_id", None)
    st.q_start_mono = time.monotonic()
    st.q_end_mono = st.q_start_mono + QUESTION_TIME
    log.info("ask_question: chat=%s qidx=%d start=%.3f end=%.3f msg=%s", st.chat_id, st.q_index, st.q_start_mono, st.q_end_mono, st.q_msg_id)
    log_mem(f"ask_question chat={st.chat_id} qidx={st.q_index}")
    st.locked = False; st.answered_now[st.q_index] = set()
    context.job_queue.run_repeating(tick_edit, interval=max(0.5, TICK_SECONDS), first=max(0.5, TICK_SECONDS),
                                    data={"chat_id": st.chat_id, "msg_id": st.q_msg_id, "qidx": st.q_index, "end_mono": st.q_end_mono},
                                    name=f"tick:{st.chat_id}:{st.q_index}")
    context.job_queue.run_once(close_question, when=QUESTION_TIME, data={"chat_id": st.chat_id, "qidx": st.q_index}, name=f"close:{st.chat_id}:{st.q_index}")
    log.info("ask_question: scheduled tick and close jobs chat=%s qidx=%d", st.chat_id, st.q_index)

async def finish_quiz(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    scores = st.totals
    ranking = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    lines = [f"🏁 <b>Quiz Finished</b> — Played by {len(st.players)} players"]
    if ranking:
        lines.append("\n🏆 Final leaderboard (Top 10):")
        for rank,(uid,total) in enumerate(ranking[:10], start=1):
            name = esc(st.players.get(uid, str(uid))); corr = st.corrects.get(uid, 0)
            medal = ("🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}.")
            lines.append(f"{medal} {name} — {corr} correct — {total} pts")
    else:
        lines.append("No scoring data.")
    await _tg("finish.send", context.bot.send_message, chat_id=st.chat_id, text="\n".join(lines), parse_mode=ParseMode.HTML)
    cancel_jobs(context, st.chat_id)
    GAMES.pop(st.chat_id, None)
    log.info("finish_quiz: cleaned game for chat=%s", st.chat_id)
    log_mem("after finish")

# ---------- handlers ----------
async def answer_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query: return
    await context.bot.answer_callback_query(query.id)
    data = query.data or ""
    try:
        _, qidx_s, choice_s = data.split(":")
        qidx = int(qidx_s); choice = int(choice_s)
    except Exception:
        log.debug("bad callback data: %r", data); return
    chat_id = query.message.chat_id
    st = GAMES.get(chat_id)
    if not st:
        log.info("answer_cb: no game for chat=%s", chat_id); return
    if st.q_index != qidx:
        log.info("answer_cb: stale answer chat=%s qidx=%s current=%s", chat_id, qidx, st.q_index); return
    if st.locked or st.q_end_mono is None:
        log.info("answer_cb: locked/no end chat=%s", chat_id); return
    user = update.effective_user; uid = user.id
    if uid in st.answered_now.get(qidx, set()):
        log.info("answer_cb: duplicate uid=%s qidx=%s", uid, qidx); return
    st.answered_now.setdefault(qidx, set()).add(uid)
    elapsed = time.monotonic() - (st.q_start_mono or time.monotonic()); elapsed = max(0.0, elapsed)
    is_correct = (choice == st.questions[qidx].correct)
    pts = points_for_elapsed(elapsed) if is_correct else 0
    st.per_q_answers.setdefault(qidx, {})[uid] = AnswerRec(choice, is_correct, elapsed, pts)
    st.players.setdefault(uid, user.full_name or user.username or str(uid))
    st.totals[uid] = st.totals.get(uid, 0) + pts
    if is_correct: st.corrects[uid] = st.corrects.get(uid, 0) + 1
    log.info("answer_cb: chat=%s qidx=%d uid=%s choice=%d elapsed=%.3f pts=%d correct=%s", chat_id, qidx, uid, choice, elapsed, pts, is_correct)

async def is_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
    try:
        mem = await context.bot.get_chat_member(chat_id, user_id)
        return mem.status in ("administrator","creator")
    except Exception:
        log.exception("is_admin failed"); return False

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, update.effective_chat.id, update.effective_user.id): return
    await update.message.reply_text(f"pong from {INSTANCE_ID}")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("/startquiz <mode> <size> — start\n/stopquiz — stop\n/ping — ping")

async def cmd_startquiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, update.effective_chat.id, update.effective_user.id):
            await update.message.reply_text("Only admins may start quizzes here."); return
    args = context.args or []
    if len(args) < 2:
        await update.message.reply_text("Usage: /startquiz <mode> <size>"); return
    mode = args[0]
    if mode not in MODES: await update.message.reply_text(f"Mode one of: {', '.join(MODES)}"); return
    try: size = int(args[1])
    except: await update.message.reply_text("Size must be a number"); return
    if size not in ALLOWED_SESSION_SIZES:
        await update.message.reply_text(f"Size must be one of: {', '.join(map(str, ALLOWED_SESSION_SIZES))}"); return
    chat_id = update.effective_chat.id; user = update.effective_user
    await do_startquiz(context, chat_id, user.id, mode, int(size))

async def do_startquiz(context: ContextTypes.DEFAULT_TYPE, chat_id: int, starter_user_id: int, mode: str, length: int):
    cancel_jobs(context, chat_id)
    try:
        pool = by_mode(load_questions(), str(mode))
    except Exception as e:
        log.exception("Failed to load questions"); await _tg("start.err", context.bot.send_message, chat_id=chat_id, text=f"Failed to load questions: {e}"); return
    if len(pool) < int(length):
        await _tg("start.notenough", context.bot.send_message, chat_id=chat_id, text="Not enough questions for that mode."); return
    qs = shuffle_qs(pool)[:int(length)]
    st = GameState(chat_id=chat_id, started_by=starter_user_id, questions=qs, limit=int(length), mode=mode)
    GAMES[chat_id] = st
    await _tg("start.ack", context.bot.send_message, chat_id=chat_id, text=f"Quiz starting ({mode}, {length} questions)…")
    log.info("do_startquiz: started chat=%s mode=%s length=%s", chat_id, mode, length)
    await ask_question(context, st)

async def cmd_stopquiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, update.effective_chat.id, update.effective_user.id):
            await update.message.reply_text("Only admins may stop quizzes here."); return
    chat_id = update.effective_chat.id; st = GAMES.get(chat_id)
    if not st: await update.message.reply_text("No active quiz."); return
    await finish_quiz(context, st)

def build_app() -> Application:
    if not TOKEN: raise RuntimeError("No TELEGRAM TOKEN in env")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("startquiz", cmd_startquiz))
    app.add_handler(CommandHandler("stopquiz", cmd_stopquiz))
    app.add_handler(CallbackQueryHandler(answer_cb, pattern=r"^ans:"))
    return app

# Keepalive server (daemon thread)
def start_keepalive_if_needed():
    port_env = os.getenv("PORT")
    if not port_env:
        log.info("No $PORT detected — assuming worker mode."); return
    try:
        port = int(port_env)
    except:
        log.warning("Invalid PORT=%r; skipping keep-alive", port_env); return
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type","text/plain; charset=utf-8"); self.end_headers()
            self.wfile.write(b"ok")
        def log_message(self,*a,**k): return
    def run():
        try:
            log.info("Keep-alive HTTP listening on 0.0.0.0:%d", port)
            HTTPServer(("0.0.0.0", port), H).serve_forever()
        except Exception:
            log.exception("keepalive crashed")
    t = threading.Thread(target=run, daemon=True, name="keepalive")
    t.start()

if __name__ == "__main__":
    try:
        log.info("Starting quiz bot…")
        start_keepalive_if_needed()
        app = build_app()
        app.run_polling()
    except Exception:
        log.exception("Fatal error in main")
