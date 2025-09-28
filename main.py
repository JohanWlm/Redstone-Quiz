# Quiz Bot — Modes, Admin-only, 20s gap, Answers via /answer, Clean/Fun themes
# deps: python-telegram-bot[job-queue]==21.*

from __future__ import annotations
import os, json, time, random
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ---------- CONFIG ----------
QUESTION_TIME = 15            # seconds to answer
DELAY_NEXT    = 20            # 20s gap after each question (no per-question reveal)
POINTS_MAX    = 100
QUESTIONS_FILE = "questions.json"

ALLOWED_SESSION_SIZES = (10, 20, 30, 40, 50)
MODES = ("beginner", "standard", "expert")
DEFAULT_THEME = "fun"         # "clean" or "fun"
# ----------------------------

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
    theme: str
    q_index: int = 0
    q_msg_id: Optional[int] = None
    q_start_ts: Optional[float] = None
    locked: bool = False
    per_q_answers: Dict[int, Dict[int, AnswerRec]] = field(default_factory=dict)
    players: Dict[int, str] = field(default_factory=dict)

GAMES: Dict[int, GameState] = {}  # active per chat
LAST: Dict[int, dict] = {}        # last finished snapshot per chat (for /answer, /leaderboard)

# ---------- Styles ----------
def style_from_theme(theme: str) -> Dict[str, str]:
    if theme == "clean":
        return {"q":"❓","timer":"⏱","ok":"✅","no":"❌","flag":"🏁","dot":"•","rule":"──────────"}
    return {"q":"✨","timer":"⏱️","ok":"✅","no":"❌","flag":"🏁","dot":"•","rule":"━━━━━━━━━━"}

def points(elapsed: float) -> int:
    if elapsed >= QUESTION_TIME: return 0
    return int(round((max(0.0, QUESTION_TIME - elapsed)/QUESTION_TIME)*POINTS_MAX))

def answer_kb(q: QItem, qnum: int) -> InlineKeyboardMarkup:
    # No A/B/C/D prefixes — buttons are just the option text
    return InlineKeyboardMarkup([[InlineKeyboardButton(opt, callback_data=f"ans:{qnum}:{i}")]
                                for i,opt in enumerate(q.options)])

def fmt_question(theme: str, qnum: int, total: int, q: QItem, left: Optional[int]=None) -> str:
    s = style_from_theme(theme)
    head = f"{s['q']} *Question {qnum}/{total}* — *{q.mode.title()}*\n{q.text}"
    if left is not None:
        head += f"\n\n{s['timer']} *{int(left)}s left*"
    return head

# ---------- Load ----------
def load_questions() -> List[QItem]:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    out=[]
    for i,q in enumerate(data, start=1):
        if not all(k in q for k in ("text","options","correct","mode")): raise ValueError(f"Q{i} missing fields")
        if q["mode"] not in MODES: raise ValueError(f"Q{i} invalid mode {q['mode']}")
        if len(q["options"])!=4: raise ValueError(f"Q{i} must have 4 options")
        c=int(q["correct"])
        if not 0<=c<4: raise ValueError(f"Q{i} invalid correct index {c}")
        out.append(QItem(text=q["text"], options=list(q["options"]), correct=c, mode=q["mode"]))
    return out

def filter_by_mode(all_qs: List[QItem], mode: str) -> List[QItem]:
    return [q for q in all_qs if q.mode==mode]

def shuffle_qs(qs: List[QItem]) -> List[QItem]:
    qs = qs.copy(); random.shuffle(qs)
    out=[]
    for q in qs:
        pairs=list(enumerate(q.options)); random.shuffle(pairs)
        new_opts=[t for _,t in pairs]
        new_correct=next(i for i,(orig,_) in enumerate(pairs) if orig==q.correct)
        out.append(QItem(text=q.text, options=new_opts, correct=new_correct, mode=q.mode))
    return out

# ---------- Jobs / Flow ----------
async def tick_edit(context: ContextTypes.DEFAULT_TYPE):
    """Edit the same message every second to update the numeric countdown."""
    data = context.job.data
    chat_id = data["chat_id"]; msg_id = data["msg_id"]; end_ts = data["end_ts"]
    st = GAMES.get(chat_id)
    if not st: 
        context.job.schedule_removal(); 
        return
    left = int(round(end_ts - time.time()))
    if left <= 0:
        context.job.schedule_removal()
        left = 0
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=msg_id,
            text=fmt_question(st.theme, st.q_index+1, st.limit, st.questions[st.q_index], left),
            reply_markup=answer_kb(st.questions[st.q_index], st.q_index+1),
            parse_mode="Markdown"
        )
    except Exception:
        pass  # benign edit conflicts are fine

async def close_question(context: ContextTypes.DEFAULT_TYPE):
    """Close the question, then wait DELAY_NEXT seconds and move on."""
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id)
    if not st: return
    st.locked = True
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=st.q_msg_id,
            text=fmt_question(st.theme, st.q_index+1, st.limit, st.questions[st.q_index], 0),
            parse_mode="Markdown"
        )
    except Exception:
        pass
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
    msg = await context.bot.send_message(
        chat_id=st.chat_id,
        text=fmt_question(st.theme, st.q_index+1, st.limit, q, QUESTION_TIME),
        reply_markup=answer_kb(q, st.q_index+1),
        parse_mode="Markdown"
    )
    st.q_msg_id = msg.message_id
    st.q_start_ts = time.time()
    st.locked = False
    end_ts = st.q_start_ts + QUESTION_TIME
    # repeat: tick every 1s on THIS message id
    context.job_queue.run_repeating(
        tick_edit, interval=1.0, first=1.0,
        data={"chat_id": st.chat_id, "msg_id": st.q_msg_id, "end_ts": end_ts},
        name=f"tick:{st.chat_id}:{st.q_index}"
    )
    # auto-close after QUESTION_TIME
    context.job_queue.run_once(
        close_question, when=QUESTION_TIME, data={"chat_id": st.chat_id},
        name=f"close:{st.chat_id}:{st.q_index}"
    )

# ---------- Helpers ----------
async def is_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
    try:
        m = await context.bot.get_chat_member(chat_id, user_id)
        return m.status in ("administrator", "creator")
    except Exception:
        return False

def compute_scores(st: GameState):
    scores: Dict[int,int] = {}; corrects: Dict[int,int] = {}
    for qidx in range(st.limit):
        for uid, rec in st.per_q_answers.get(qidx, {}).items():
            scores[uid] = scores.get(uid, 0) + rec.points
            if rec.is_correct: corrects[uid] = corrects.get(uid, 0) + 1
    return scores, corrects

# ---------- Commands ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    theme = context.chat_data.get("theme", DEFAULT_THEME)
    mode  = context.chat_data.get("mode")
    length= context.chat_data.get("length")
    s = style_from_theme(theme)
    await update.message.reply_text(
        f"{s['q']} *Quiz Bot*\n{s['rule']}\n"
        f"1) /menu — choose *Mode* (Beginner/Standard/Expert), *Length* (10–50), Theme\n"
        f"2) /startquiz — start (admin-only in groups)\n"
        f"3) /leaderboard — show ranks (current or last)\n"
        f"4) /answer — reveal all correct answers (after quiz ends)\n"
        f"5) /stopquiz — end early\n"
        f"6) /reset — clear previous session & settings\n\n"
        f"Current: Mode={mode or '—'} • Length={length or '—'} • Theme={theme.title()}",
        parse_mode="Markdown"
    )

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    theme = context.chat_data.get("theme", DEFAULT_THEME)
    length= context.chat_data.get("length") or 10
    mode  = context.chat_data.get("mode") or "—"
    rows = [
        [InlineKeyboardButton("Beginner", callback_data="cfg:mode:beginner"),
         InlineKeyboardButton("Standard", callback_data="cfg:mode:standard"),
         InlineKeyboardButton("Expert", callback_data="cfg:mode:expert")],
        [InlineKeyboardButton("10", callback_data="cfg:len:10"),
         InlineKeyboardButton("20", callback_data="cfg:len:20"),
         InlineKeyboardButton("30", callback_data="cfg:len:30")],
        [InlineKeyboardButton("40", callback_data="cfg:len:40"),
         InlineKeyboardButton("50", callback_data="cfg:len:50")],
        [InlineKeyboardButton(f"Theme: {theme.title()} (toggle)", callback_data="cfg:theme:toggle")],
    ]
    await update.message.reply_text(
        f"*Setup Required*: Choose a *Mode* and *Length*.\nCurrent → Mode: *{mode}*, Length: *{length}*, Theme: *{theme.title()}*",
        reply_markup=InlineKeyboardMarkup(rows), parse_mode="Markdown"
    )

async def cfg_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data
    if not data.startswith("cfg:"): return
    _, kind, val = data.split(":")
    chat_id = q.message.chat.id

    # Changing settings wipes old session/logs
    GAMES.pop(chat_id, None)
    LAST.pop(chat_id, None)

    if kind == "mode":
        if val not in MODES:
            await q.edit_message_text("Invalid mode."); return
        context.chat_data["mode"] = val
        await q.edit_message_text(f"Mode set to *{val.title()}*. Use /menu to adjust or /startquiz to begin.", parse_mode="Markdown")
        return
    if kind == "len":
        try:
            length = int(val)
            if length not in ALLOWED_SESSION_SIZES: raise ValueError()
        except Exception:
            await q.edit_message_text("Invalid length."); return
        context.chat_data["length"] = length
        await q.edit_message_text(f"Length set to *{length}*. Use /menu to adjust or /startquiz to begin.", parse_mode="Markdown")
        return
    if kind == "theme" and val == "toggle":
        cur = context.chat_data.get("theme", DEFAULT_THEME)
        context.chat_data["theme"] = "clean" if cur=="fun" else "fun"
        await q.edit_message_text(f"Theme set to *{context.chat_data['theme'].title()}*.", parse_mode="Markdown")
        return

async def cmd_startquiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user

    # admin-only in groups
    if update.effective_chat.type != "private":
        if not await is_admin(context, chat_id, user.id):
            await update.message.reply_text("Only group admins can start a quiz.")
            return

    mode = context.chat_data.get("mode")
    length = context.chat_data.get("length")
    theme = context.chat_data.get("theme", DEFAULT_THEME)
    if mode not in MODES or length not in ALLOWED_SESSION_SIZES:
        await update.message.reply_text("Please run /menu and choose a *Mode* and *Length* first.", parse_mode="Markdown")
        return

    # start new session—clear previous logs/snapshots
    GAMES.pop(chat_id, None)
    LAST.pop(chat_id, None)

    all_qs = load_questions()
    pool = filter_by_mode(all_qs, mode)
    if len(pool) < length:
        await update.message.reply_text(f"Not enough questions in *{mode}* (need {length}).")
        return

    qs = shuffle_qs(pool)[:length]
    st = GameState(chat_id=chat_id, started_by=user.id, questions=qs, limit=length, mode=mode, theme=theme)
    GAMES[chat_id] = st

    await update.message.reply_text(
        f"🎯 *{mode.title()}* mode • {length} questions\n"
        f"⏱ {QUESTION_TIME}s per question • Next question in {DELAY_NEXT}s after each.",
        parse_mode="Markdown"
    )
    await ask_question(context, st)

async def on_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    user = q.from_user
    chat_id = q.message.chat.id
    try:
        _, qnum_str, opt_str = q.data.split(":")
        qnum = int(qnum_str); opt = int(opt_str)
    except Exception:
        return

    st = GAMES.get(chat_id)
    if not st: return
    if st.q_index + 1 != qnum:
        try: await q.answer("This question is already closed.", show_alert=False)
        except Exception: pass
        return
    if st.locked or st.q_start_ts is None:
        try: await q.answer("Time is up!", show_alert=False)
        except Exception: pass
        return

    st.players[user.id] = (user.full_name or user.username or str(user.id))[:64]
    st.per_q_answers.setdefault(st.q_index, {})
    if user.id in st.per_q_answers[st.q_index]:
        try: await q.answer("Only your first answer counts.", show_alert=False)
        except Exception: pass
        return

    elapsed = max(0.0, time.time() - st.q_start_ts)
    is_correct = (opt == st.questions[st.q_index].correct)
    pts = points(elapsed) if is_correct else 0
    st.per_q_answers[st.q_index][user.id] = AnswerRec(choice=opt, is_correct=is_correct, elapsed=elapsed, points=pts)

    try:
        await q.answer(f"{'✅' if is_correct else '❌'} {pts} pts", show_alert=False)
    except Exception:
        pass

async def cmd_leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = GAMES.get(chat_id)
    theme = (st.theme if st else context.chat_data.get("theme", DEFAULT_THEME))
    s = style_from_theme(theme)

    if st:
        scores, corrects = compute_scores(st)
        source = f"Current session — {st.q_index+1}/{st.limit} asked"
        name_of = st.players
    else:
        snap = LAST.get(chat_id)
        if not snap:
            await update.message.reply_text("No session found here yet."); return
        scores = snap["scores"]; corrects = snap["corrects"]
        source = f"Last finished — {snap['limit']} questions"
        name_of = snap["players"]

    if not scores:
        await update.message.reply_text("No participants yet."); return

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    lines = [f"{s['flag']} *Leaderboard* ({source})"]
    for rank,(uid,pts_) in enumerate(ranking[:10], start=1):
        name = name_of.get(uid, str(uid))
        corr = corrects.get(uid, 0)
        medal = ("🥇" if rank == 1 else
                 "🥈" if rank == 2 else
                 "🥉" if rank == 3 else
                 f"{rank}.")
        lines.append(f"{medal} {name} — {corr} correct — {pts_} pts")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    snap = LAST.get(chat_id)
    if not snap:
        await update.message.reply_text("No finished quiz found here yet."); return
    qs: List[QItem] = snap["questions"]
    lines = ["📘 *All Correct Answers*"]
    for i,q in enumerate(qs, start=1):
        lines.append(f"Q{i}: *{q.options[q.correct]}*")
        if len("\n".join(lines)) > 3500:  # chunk to respect Telegram limits
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
            lines = []
    if lines:
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def finish_quiz(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    scores, corrects = compute_scores(st)
    if not scores:
        await context.bot.send_message(chat_id=st.chat_id, text="Quiz ended. No participants 😅")
        GAMES.pop(st.chat_id, None); return

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Winner shoutout
    top_uid, top_pts = ranking[0]
    top_name = st.players.get(top_uid, str(top_uid))
    await context.bot.send_message(
        chat_id=st.chat_id,
        text=f"🎉 Congrats, *{top_name}!* You topped the quiz with *{top_pts} pts*!",
        parse_mode="Markdown"
    )

    # Final results (answers via /answer)
    s = style_from_theme(st.theme)
    lines=[f"{s['flag']} *Final Results* — Top 10"]
    for rank,(uid,pts_) in enumerate(ranking[:10], start=1):
        name = st.players.get(uid, str(uid))
        corr = corrects.get(uid, 0)
        medal = ("🥇" if rank == 1 else
                 "🥈" if rank == 2 else
                 "🥉" if rank == 3 else
                 f"{rank}.")
        lines.append(f"{medal} {name} — {corr}/{st.limit} correct — {pts_} pts")
    lines.append("\nUse /leaderboard anytime. Use /answer to reveal all correct answers.")
    await context.bot.send_message(chat_id=st.chat_id, text="\n".join(lines), parse_mode="Markdown")

    # Snapshot for /answer and /leaderboard
    LAST[st.chat_id] = {
        "questions": st.questions,
        "limit": st.limit,
        "scores": scores,
        "corrects": corrects,
        "players": st.players,
        "mode": st.mode,
        "theme": st.theme,
    }

    # Clear active session
    GAMES.pop(st.chat_id, None)

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    # admin-only in groups
    if update.effective_chat.type != "private":
        if not await is_admin(context, chat_id, update.effective_user.id):
            await update.message.reply_text("Only group admins can stop the quiz.")
            return
    st = GAMES.get(chat_id)
    if not st:
        await update.message.reply_text("No active quiz to stop."); return
    st.limit = st.q_index + 1
    await finish_quiz(context, st)

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    GAMES.pop(chat_id, None)
    LAST.pop(chat_id, None)
    context.chat_data.clear()
    await update.message.reply_text("✅ Reset complete. Use /menu to set Mode & Length, then /startquiz.")

def build_app() -> Application:
    token = os.getenv("BOT_TOKEN")
    if not token: raise RuntimeError("Set BOT_TOKEN environment variable.")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("menu",        cmd_menu))
    app.add_handler(CallbackQueryHandler(cfg_click, pattern=r"^cfg:"))
    app.add_handler(CommandHandler("startquiz",   cmd_startquiz))
    app.add_handler(CommandHandler("leaderboard", cmd_leaderboard))
    app.add_handler(CommandHandler("answer",      cmd_answer))
    app.add_handler(CommandHandler("stopquiz",    cmd_stop))
    app.add_handler(CommandHandler("reset",       cmd_reset))
    app.add_handler(CallbackQueryHandler(on_answer, pattern=r"^ans:\d+:\d$"))
    return app

if __name__ == "__main__":
    print("Starting quiz bot…")
    build_app().run_polling(close_loop=False)
