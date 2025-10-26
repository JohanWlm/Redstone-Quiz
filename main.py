# Telegram Quiz Bot — precise timing (monotonic), keyboard stays until timeout
from __future__ import annotations
import os, json, time, random, math, html, logging, threading, uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut, NetworkError, BadRequest
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, MessageHandler,
    ContextTypes, filters
)

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("quizbot")

# --------------- ENV ---------------------
def _int(name, d, lo, hi):
    v = os.getenv(name)
    if v is None: return d
    try:
        x=int(v); return max(lo, min(hi, x))
    except: return d

def _float(name, d, lo, hi):
    v = os.getenv(name)
    if v is None: return d
    try:
        x=float(v); return max(lo, min(hi, x))
    except: return d

def _bool(name, d):
    v = os.getenv(name)
    if v is None: return d
    return v.strip().lower() in ("1","true","yes","y","on")

QUESTION_TIME  = _int("QUESTION_TIME", 10, 3, 120)      # full time buttons remain
DELAY_NEXT     = _int("DELAY_NEXT",     3, 1, 60)       # gap between questions
TICK_SECONDS   = _float("TICK_SECONDS", 2.0, 0.5, 5.0)  # countdown edit cadence
POINTS_MAX     = _int("POINTS_MAX",   100, 10, 1000)
QUESTIONS_FILE = os.getenv("QUESTIONS_FILE", "questions.json")
ADMINS_ONLY    = _bool("ADMINS_ONLY", True)

ALLOWED_SESSION_SIZES = (10, 20, 30, 40, 50)
MODES = ("beginner", "standard", "expert")

INSTANCE_ID = os.getenv("RAILWAY_REPLICA_ID") or str(uuid.uuid4())[:8]

def esc(s: str) -> str:
    return html.escape(str(s), quote=True)

# ----------- SAFE TELEGRAM CALL ----------
async def _tg(desc: str, coro, *args, **kwargs):
    try:
        return await coro(*args, **kwargs)
    except RetryAfter as e:
        wait = float(getattr(e, "retry_after", 1.0)) + 0.5
        log.warning("%s: RetryAfter %.2fs", desc, wait)
        await asyncio.sleep(wait)
        try: return await coro(*args, **kwargs)
        except Exception as e2:
            log.warning("%s: giving up after retry: %s", desc, e2); return None
    except (TimedOut, NetworkError) as e:
        log.warning("%s: %s; mild backoff", desc, type(e).__name__)
        await asyncio.sleep(1.0)
        try: return await coro(*args, **kwargs)
        except Exception as e2:
            log.warning("%s: giving up after retry: %s", desc, e2); return None
    except BadRequest as e:
        log.debug("%s: BadRequest ignored: %s", desc, e); return None
    except Exception as e:
        log.error("%s: unexpected %s: %s", desc, type(e).__name__, e); return None

# -------------- DATA ---------------------
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
LAST: Dict[int, dict] = {}
SETTINGS: Dict[int, Dict[str, int | str]] = {}

# -------------- HELPERS ------------------
def points(elapsed: float) -> int:
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
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[QItem] = []
    for i, q in enumerate(data, start=1):
        if not all(k in q for k in ("text","options","correct","mode")):
            raise ValueError(f"Q{i} missing fields")
        if q["mode"] not in MODES:
            raise ValueError(f"Q{i} invalid mode {q['mode']}")
        if len(q["options"]) != 4:
            raise ValueError(f"Q{i} must have 4 options")
        c = int(q["correct"])
        if not 0 <= c < 4:
            raise ValueError(f"Q{i} invalid correct index {c}")
        out.append(QItem(q["text"], list(q["options"]), c, q["mode"]))
    return out

def by_mode(all_qs: List[QItem], mode: str) -> List[QItem]:
    return [q for q in all_qs if q.mode == mode]

def shuffle_qs(qs: List[QItem]) -> List[QItem]:
    qs = qs.copy(); random.shuffle(qs)
    out: List[QItem] = []
    for q in qs:
        pairs = list(enumerate(q.options)); random.shuffle(pairs)
        new_opts = [t for _, t in pairs]
        new_correct = next(i for i, (orig, _) in enumerate(pairs) if orig == q.correct)
        out.append(QItem(q.text, new_opts, new_correct, q.mode))
    return out

def compute_totals(st: GameState) -> Tuple[Dict[int, int], Dict[int, int]]:
    return st.totals, st.corrects

def cancel_jobs(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    try:
        for job in context.job_queue.jobs():
            name = getattr(job, "name", "") or ""
            if name.startswith(f"tick:{chat_id}:") or name.startswith(f"close:{chat_id}:") or name.startswith(f"gap:{chat_id}:"):
                job.schedule_removal()
    except Exception as e:
        log.warning("job cleanup error: %s", e)

# -------------- JOBS ---------------------
import asyncio

async def tick_edit(context: ContextTypes.DEFAULT_TYPE):
    d = context.job.data
    chat_id = d["chat_id"]; msg_id = d["msg_id"]; qidx = d["qidx"]
    st = GAMES.get(chat_id)
    if not st or st.q_index != qidx or st.q_end_mono is None:
        context.job.schedule_removal(); return
    left = max(0, int(math.ceil(st.q_end_mono - time.monotonic())))
    if left <= 0:
        context.job.schedule_removal(); return
    # IMPORTANT: do NOT pass reply_markup here -> keyboard stays
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
        await _tg("gap.final", context.bot.edit_message_text,
                  chat_id=chat_id, message_id=msg_id, text="🚀 Starting next question…")
        return
    await _tg("gap.tick", context.bot.edit_message_text,
              chat_id=chat_id, message_id=msg_id, text=f"⏭️ Next question in {left}s…")

async def post_round_recap(context: ContextTypes.DEFAULT_TYPE, st: GameState, qidx: int):
    q = st.questions[qidx]
    per = st.per_q_answers.get(qidx, {})
    scorers = sorted(((uid, rec.points) for uid, rec in per.items() if rec.is_correct),
                     key=lambda t:t[1], reverse=True)
    scores, corrects = compute_totals(st)
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
        for rank, (uid, total) in enumerate(ranking[:5], start=1):
            name = esc(st.players.get(uid, str(uid)))
            corr = corrects.get(uid, 0)
            medal = ("🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}.")
            lines.append(f"{medal} {name} — {corr} correct — {total} pts")
    await _tg("recap.send", context.bot.send_message, chat_id=st.chat_id,
              text="\n".join(lines), parse_mode=ParseMode.HTML)

async def close_question(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id)
    if not st or st.q_end_mono is None:
        return
    # Guard against early wake-up: rearm to exact time if needed
    now = time.monotonic()
    if now < st.q_end_mono - 0.05:
        context.job_queue.run_once(close_question, when=st.q_end_mono - now, data={"chat_id": chat_id})
        return
    if st.locked: return
    st.locked = True

    # Remove keyboard exactly at timeout
    await _tg("close.freeze", context.bot.edit_message_text,
              chat_id=chat_id, message_id=st.q_msg_id,
              text=fmt_q(st.q_index+1, st.limit, st.questions[st.q_index], 0,
                         len(st.per_q_answers.get(st.q_index, {}))),
              reply_markup=None, parse_mode=ParseMode.HTML)

    # Recap + free per-question memory
    await post_round_recap(context, st, st.q_index)
    st.per_q_answers.pop(st.q_index, None)
    st.answered_now.pop(st.q_index, None)

    # Gap and next question
    gap_end = time.monotonic() + DELAY_NEXT
    m = await _tg("gap.send", context.bot.send_message, chat_id=chat_id,
                  text=f"⏭️ Next question in {DELAY_NEXT}s…")
    if m and getattr(m, "message_id", None):
        context.job_queue.run_repeating(
            gap_tick, interval=max(1.0, TICK_SECONDS), first=max(1.0, TICK_SECONDS),
            data={"chat_id": chat_id, "msg_id": m.message_id, "end_mono": gap_end},
            name=f"gap:{chat_id}:{st.q_index}"
        )
    context.job_queue.run_once(next_question, when=DELAY_NEXT, data={"chat_id": chat_id})

async def next_question(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id)
    if not st: return
    st.q_index += 1
    if st.q_index >= st.limit:
        await finish_quiz(context, st); return
    await ask_question(context, st)

async def ask_question(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    q = st.questions[st.q_index]
    m = await _tg("ask.send", context.bot.send_message, chat_id=st.chat_id,
                  text=fmt_q(st.q_index+1, st.limit, q, QUESTION_TIME, 0),
                  reply_markup=answer_kb(q, st.q_index+1), parse_mode=ParseMode.HTML)
    if not m:
        await finish_quiz(context, st); return
    st.q_msg_id = m.message_id
    st.q_start_mono = time.monotonic()
    st.q_end_mono   = st.q_start_mono + QUESTION_TIME
    st.locked       = False
    st.answered_now[st.q_index] = set()

    # One ticker (text only), one closer (removes keyboard exactly at deadline)
    context.job_queue.run_repeating(
        tick_edit, interval=TICK_SECONDS, first=TICK_SECONDS,
        data={"chat_id": st.chat_id, "msg_id": st.q_msg_id, "qidx": st.q_index},
        name=f"tick:{st.chat_id}:{st.q_index}"
    )
    context.job_queue.run_once(
        close_question, when=QUESTION_TIME, data={"chat_id": st.chat_id},
        name=f"close:{st.chat_id}:{st.q_index}"
    )

# -------------- ADMIN --------------------
async def is_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
    try:
        m = await context.bot.get_chat_member(chat_id, user_id)
        return m.status in ("administrator", "creator")
    except Exception:
        return False

# -------------- COMMANDS -----------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = SETTINGS.get(update.effective_chat.id, {})
    mode = cfg.get("mode"); length = cfg.get("length")
    await update.message.reply_text(
        "✨ <b>Quiz Bot</b>\n────────────\n"
        "Step 1: /menu — choose <b>Mode</b>\n"
        "Step 2: choose <b>How many questions</b>\n"
        "Then: /startquiz — start (admin-only in groups)\n\n"
        "During play: recap + leaderboard each round.\n"
        "Tools: /leaderboard • /answer • /stop • /reset • /ping\n\n"
        f"Current: Mode={esc(mode) if mode else '—'} • Length={esc(length) if length else '—'}",
        parse_mode=ParseMode.HTML
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n/start • /help • /menu • /startquiz\n"
        "/leaderboard • /answer • /stop • /reset\n/ping",
        parse_mode=ParseMode.HTML
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private":
        if not await is_admin(context, update.effective_chat.id, update.effective_user.id):
            return
    await update.message.reply_text(f"pong from {INSTANCE_ID}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, update.effective_chat.id, update.effective_user.id):
            await update.message.reply_text("Only group admins can configure the quiz here."); return
    rows = [[InlineKeyboardButton("Beginner", callback_data="cfg:mode:beginner"),
             InlineKeyboardButton("Standard", callback_data="cfg:mode:standard"),
             InlineKeyboardButton("Expert",   callback_data="cfg:mode:expert")]]
    await _tg("menu.send", context.bot.send_message, chat_id=update.effective_chat.id,
              text="<b>Step 1/2</b>: Choose a <b>Mode</b>.",
              reply_markup=InlineKeyboardMarkup(rows), parse_mode=ParseMode.HTML)

async def _send_length_menu(update_or_query, context):
    rows = [[InlineKeyboardButton("10", callback_data="cfg:len:10"),
             InlineKeyboardButton("20", callback_data="cfg:len:20"),
             InlineKeyboardButton("30", callback_data="cfg:len:30")],
            [InlineKeyboardButton("40", callback_data="cfg:len:40"),
             InlineKeyboardButton("50", callback_data="cfg:len:50")]]
    chat_id = update_or_query.effective_chat.id if isinstance(update_or_query, Update) else update_or_query.message.chat.id
    await _tg("length.send", context.bot.send_message, chat_id=chat_id,
              text="<b>Step 2/2</b>: How many questions will be held?",
              reply_markup=InlineKeyboardMarkup(rows), parse_mode=ParseMode.HTML)

async def cfg_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data
    if not data.startswith("cfg:"): return
    chat_id = q.message.chat.id
    if q.message.chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, chat_id, q.from_user.id):
            try: await q.answer("Only group admins can change quiz settings here.", show_alert=True)
            except Exception: pass
            return
    _, kind, val = data.split(":")
    GAMES.pop(chat_id, None); LAST.pop(chat_id, None)
    if kind == "mode":
        if val not in MODES: await q.edit_message_text("Invalid mode."); return
        SETTINGS.setdefault(chat_id, {})["mode"] = val
        try: await q.edit_message_text(f"Mode set to <b>{esc(val.title())}</b> ✅", parse_mode=ParseMode.HTML)
        except Exception: pass
        await _send_length_menu(update, context); return
    if kind == "len":
        try: length=int(val)
        except: await q.edit_message_text("Invalid length."); return
        if length not in ALLOWED_SESSION_SIZES: await q.edit_message_text("Invalid length."); return
        SETTINGS.setdefault(chat_id, {})["length"]=length
        mode = SETTINGS.get(chat_id, {}).get("mode")
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("▶️ Start quiz", callback_data=f"start:{mode}:{length}")]]) if mode in MODES else None
        await q.edit_message_text(f"Length set to <b>{length}</b> ✅\nUse /startquiz to begin, or tap ▶️ Start.", parse_mode=ParseMode.HTML)
        if kb:
            await _tg("start.button", context.bot.send_message, chat_id=chat_id,
                      text=f"Ready to start <b>{esc(str(mode).title())}</b> • {length} questions?",
                      reply_markup=kb, parse_mode=ParseMode.HTML)

async def start_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    try:
        _, mode, length_s = q.data.split(":"); length = int(length_s)
    except Exception: return
    chat_id = q.message.chat.id
    if q.message.chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, chat_id, q.from_user.id):
            try: await q.answer("Only group admins can start the quiz here.", show_alert=True)
            except Exception: pass
            return
    await do_startquiz(context, chat_id, q.from_user.id, mode, length)

async def cmd_startquiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id=update.effective_chat.id; user=update.effective_user
    if update.effective_chat.type!="private":
        if not await is_admin(context, chat_id, user.id):
            await update.message.reply_text("Only group admins can start a quiz."); return
    mode=None; length=None
    if context.args and len(context.args)>=2:
        mode = str(context.args[0]).lower()
        try: length = int(context.args[1])
        except: length = None
    if mode not in MODES or length not in ALLOWED_SESSION_SIZES:
        cfg=SETTINGS.get(chat_id,{}); mode = mode or cfg.get("mode"); length = length or cfg.get("length")
    if mode not in MODES or length not in ALLOWED_SESSION_SIZES:
        await update.message.reply_text(
            "Please run /menu and complete both steps first, or use:\n"
            "/startquiz <mode> <length>\nExample: /startquiz beginner 10"
        ); return
    await do_startquiz(context, chat_id, user.id, mode, int(length))

async def do_startquiz(context: ContextTypes.DEFAULT_TYPE, chat_id: int, starter_user_id: int, mode: str, length: int):
    pool = by_mode(load_questions(), str(mode))
    if len(pool) < int(length):
        await _tg("start.notenough", context.bot.send_message, chat_id=chat_id,
                  text=f"Not enough questions in <b>{esc(mode)}</b> (need {length}).", parse_mode=ParseMode.HTML)
        return
    qs = shuffle_qs(pool)[: int(length)]
    st = GameState(chat_id=chat_id, started_by=starter_user_id, questions=qs, limit=int(length), mode=str(mode))
    GAMES[chat_id]=st; LAST.pop(chat_id, None)
    await _tg("intro.send", context.bot.send_message, chat_id=chat_id,
              text=(f"🎯 <b>{esc(str(mode).title())}</b> mode • {length} questions\n"
                    f"⏱ {QUESTION_TIME}s per question • Next question in {DELAY_NEXT}s after each."),
              parse_mode=ParseMode.HTML)
    await ask_question(context, st)

# --------- ANSWERS/RESULTS ----------
async def on_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q=update.callback_query; user=q.from_user; chat_id=q.message.chat.id
    try:
        _, qnum_str, opt_str = q.data.split(":"); qnum=int(qnum_str); opt=int(opt_str)
    except Exception: return
    st=GAMES.get(chat_id)
    if not st:
        await q.answer("No active quiz here.", show_alert=False); return
    if st.q_index+1!=qnum:
        await q.answer("This question is already closed.", show_alert=False); return
    if st.locked or st.q_start_mono is None:
        await q.answer("Time is up!", show_alert=False); return

    st.answered_now.setdefault(st.q_index,set())
    if user.id in st.answered_now[st.q_index]:
        await q.answer("Only your first answer counts.", show_alert=False); return
    st.answered_now[st.q_index].add(user.id)

    st.players[user.id] = (user.full_name or user.username or str(user.id))[:64]
    st.per_q_answers.setdefault(st.q_index,{})
    if user.id in st.per_q_answers[st.q_index]:
        await q.answer("Only your first answer counts.", show_alert=False); return

    elapsed=max(0.0, time.monotonic()-st.q_start_mono); is_correct=(opt==st.questions[st.q_index].correct)
    pts=points(elapsed) if is_correct else 0
    st.per_q_answers[st.q_index][user.id]=AnswerRec(opt,is_correct,elapsed,pts)
    if is_correct: st.corrects[user.id]=st.corrects.get(user.id,0)+1
    st.totals[user.id]=st.totals.get(user.id,0)+pts

    await q.answer(
        f"You chose: {st.questions[st.q_index].options[opt]}\n" +
        ("✅ Correct" if is_correct else "❌ Locked"),
        show_alert=False
    )

async def cmd_leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id=update.effective_chat.id; st=GAMES.get(chat_id)
    if st:
        scores,corrects=compute_totals(st); source=f"Current session — {st.q_index+1}/{st.limit} asked"; name_of=st.players
    else:
        snap=LAST.get(chat_id)
        if not snap:
            await update.message.reply_text("No session found here yet."); return
        scores=snap["scores"]; corrects=snap["corrects"]; source=f"Last finished — {snap['limit']} questions"; name_of=snap["players"]
    if not scores:
        await update.message.reply_text("No participants yet."); return
    ranking=sorted(scores.items(), key=lambda x:x[1], reverse=True)
    lines=[f"🏁 <b>Leaderboard</b> ({esc(source)})"]
    for rank,(uid,pts_) in enumerate(ranking[:10],start=1):
        name=esc(name_of.get(uid,str(uid))); corr=corrects.get(uid,0); medal=("🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}.")
        lines.append(f"{medal} {name} — {corr} correct — {pts_} pts")
    lines.append("\nGG! Thanks for participating 🎉")
    await _tg("leaderboard.send", context.bot.send_message, chat_id=chat_id, text="\n".join(lines), parse_mode=ParseMode.HTML)

async def cmd_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id=update.effective_chat.id; snap=LAST.get(chat_id)
    if not snap:
        await update.message.reply_text("No finished quiz found here yet."); return
    qs: List[QItem]=snap["questions"]
    lines=["📘 <b>All Correct Answers</b>"]
    for i,q in enumerate(qs, start=1):
        lines.append(f"Q{i}: <b>{esc(q.options[q.correct])}</b>")
        if len("\n".join(lines))>3500:
            await _tg("answer.chunk", context.bot.send_message, chat_id=chat_id, text="\n".join(lines), parse_mode=ParseMode.HTML); lines=[]
    if lines:
        await _tg("answer.final", context.bot.send_message, chat_id=chat_id, text="\n".join(lines), parse_mode=ParseMode.HTML)

async def finish_quiz(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    cancel_jobs(context, st.chat_id)
    scores,corrects=compute_totals(st)
    if not scores:
        await _tg("finish.empty", context.bot.send_message, chat_id=st.chat_id, text="Quiz ended. No participants 😅")
        GAMES.pop(st.chat_id, None); return
    ranking=sorted(scores.items(), key=lambda x:x[1], reverse=True)
    top_uid, top_pts = ranking[0]; top_name=esc(st.players.get(top_uid,str(top_uid)))
    await _tg("finish.win", context.bot.send_message, chat_id=st.chat_id, text=f"🎉 Congrats, <b>{top_name}</b>! You topped the quiz with <b>{top_pts} pts</b>!", parse_mode=ParseMode.HTML)
    lines=["🏁 <b>Final Results</b> — Top 10"]
    for rank,(uid,pts_) in enumerate(ranking[:10], start=1):
        name=esc(st.players.get(uid,str(uid))); corr=corrects.get(uid,0); medal=("🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}.")
        lines.append(f"{medal} {name} — {corr}/{st.limit} correct — {pts_} pts")
    lines.append("\nUse /leaderboard anytime. Use /answer for all correct answers. GG! 🎉")
    await _tg("finish.board", context.bot.send_message, chat_id=st.chat_id, text="\n".join(lines), parse_mode=ParseMode.HTML)
    LAST[st.chat_id]={"questions":st.questions,"limit":st.limit,"scores":scores,"corrects":corrects,"players":st.players,"mode":st.mode}
    GAMES.pop(st.chat_id, None)

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user=update.effective_user; here_id=update.effective_chat.id
    if update.effective_chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, here_id, user.id):
            await update.message.reply_text("Only group admins can stop the quiz here."); return
    st=GAMES.get(here_id)
    if not st:
        await update.message.reply_text("No active quiz to stop."); return
    st.limit=st.q_index+1; st.locked=True
    await update.message.reply_text("Stopping quiz…"); cancel_jobs(context, st.chat_id); await finish_quiz(context, st)

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id=update.effective_chat.id
    if update.effective_chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, chat_id, update.effective_user.id):
            await update.message.reply_text("Only group admins can reset this chat."); return
    GAMES.pop(chat_id, None); LAST.pop(chat_id, None); SETTINGS.pop(chat_id, None); cancel_jobs(context, chat_id)
    await update.message.reply_text("✅ Reset complete. Use /menu to choose Mode, then Length, then /startquiz.")

# Unknown commands -> help
async def on_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text and update.message.text.startswith("/"):
        await update.message.reply_text("Unknown command. Try /help")

# -------------- KEEP-ALIVE (for Web) --------------
def start_keepalive_if_needed():
    port_env = os.getenv("PORT")
    if not port_env:
        log.info("No $PORT detected — assuming Worker service.")
        return
    try: port = int(port_env)
    except:
        log.warning("Invalid PORT=%r; skipping keep-alive", port_env); return
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type","text/plain; charset=utf-8")
            self.end_headers(); self.wfile.write(b"ok")
        def log_message(self, *a, **k): return
    threading.Thread(target=lambda: HTTPServer(("0.0.0.0", port), H).serve_forever(), daemon=True).start()
    log.info("Keep-alive HTTP listening on 0.0.0.0:%d", port)

# -------------- BUILD --------------------
def build_app() -> Application:
    token = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
    if not token: raise RuntimeError("Set BOT_TOKEN (or TELEGRAM_BOT_TOKEN / TELEGRAM_TOKEN) env var.")
    builder = Application.builder().token(token)
    try:
        from telegram.ext import AIORateLimiter
        builder = builder.rate_limiter(AIORateLimiter())
        log.info("AIORateLimiter enabled.")
    except Exception as e:
        log.info("Rate limiter not available; continuing: %s", e)
    app = builder.build()

    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("help",        cmd_help))
    app.add_handler(CommandHandler("menu",        cmd_menu))
    app.add_handler(CallbackQueryHandler(cfg_click, pattern=r"^cfg:"))
    app.add_handler(CallbackQueryHandler(start_click, pattern=r"^start:(beginner|standard|expert):\d+$"))
    app.add_handler(CommandHandler("startquiz",   cmd_startquiz))
    app.add_handler(CommandHandler("leaderboard", cmd_leaderboard))
    app.add_handler(CommandHandler("answer",      cmd_answer))
    app.add_handler(CommandHandler("stopquiz",    cmd_stop))
    app.add_handler(CommandHandler("stop",        cmd_stop))
    app.add_handler(CommandHandler("cancel",      cmd_stop))
    app.add_handler(CommandHandler("reset",       cmd_reset))
    app.add_handler(CommandHandler("ping",        cmd_ping))
    app.add_handler(CallbackQueryHandler(on_answer, pattern=r"^ans:\d+:\d$"))
    app.add_handler(MessageHandler(filters.COMMAND, on_unknown))
    return app

# -------------- START --------------------
if __name__ == "__main__":
    import asyncio
    log.info("Starting quiz bot…")
    start_keepalive_if_needed()
    app = build_app()
    app.run_polling(drop_pending_updates=True)
