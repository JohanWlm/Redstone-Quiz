# Quiz Bot — Modes (Beginner/Standard/Expert), Admin-Only in Groups, Clean/Fun Themes
# deps: python-telegram-bot[job-queue]==21.*

from __future__ import annotations
import os, json, time, random, math
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ---------- CONFIG ----------
QUESTION_TIME = 15
DELAY_REVEAL = 5
DELAY_NEXT   = 10
POINTS_MAX   = 100
QUESTIONS_FILE = "questions.json"

ALLOWED_SESSION_SIZES = (10, 20, 30, 40, 50)
MODES = ("beginner", "standard", "expert")
DEFAULT_THEME = "fun"  # "clean" or "fun"
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

GAMES: Dict[int, GameState] = {}  # per chat

# ---------- Styles ----------
def style_from_theme(theme: str) -> Dict[str, str]:
    if theme == "clean":
        return {"q":"❓","timer":"⏱","ok":"✅","no":"❌","fast":"⟶","flag":"🏁","dot":"•","rule":"──────────"}
    return {"q":"✨","timer":"⏱️","ok":"✅","no":"❌","fast":"🏃","flag":"🏁","dot":"•","rule":"━━━━━━━━━━"}

def progress_bar_total(cur: int, total: int) -> str:
    # overall progress (10 blocks)
    blocks = 10
    filled = min(blocks, math.floor(((cur-1)/max(1,total))*blocks))
    return "▰"*filled + "▱"*(blocks-filled)

def progress_bar_time(seconds_left: int) -> str:
    # 15→10→5→0 in 3 blocks
    blocks = max(0, math.ceil(seconds_left/5))
    return "▰"*blocks + "▱"*(3-blocks)

def points(elapsed: float) -> int:
    if elapsed >= QUESTION_TIME: return 0
    return int(round((max(0.0, QUESTION_TIME - elapsed)/QUESTION_TIME)*POINTS_MAX))

def answer_kb(q: QItem, qnum: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(f"{chr(65+i)}. {opt}", callback_data=f"ans:{qnum}:{i}")]
                                for i,opt in enumerate(q.options)])

def fmt_question(theme: str, qnum: int, total: int, q: QItem, seconds_left: Optional[int]=None) -> str:
    s = style_from_theme(theme)
    head = f"{s['q']} *Question {qnum}/{total}* — *{q.mode.title()}*\n{q.text}"
    pb_total = progress_bar_total(qnum, total)
    head += f"\n`{pb_total}`  *(completed: {qnum-1}/{total})*"
    if seconds_left is not None:
        head += f"\n\n{s['timer']} *{seconds_left}s*  `{progress_bar_time(seconds_left)}`"
    return head

def fmt_correct(theme: str, q: QItem) -> str:
    s = style_from_theme(theme)
    return f"{s['ok']} Correct: *{chr(65+q.correct)}. {q.options[q.correct]}*"

# ---------- Load ----------
def load_questions() -> List[QItem]:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    out=[]
    for i,q in enumerate(data, start=1):
        if not all(k in q for k in ("text","options","correct","mode")): raise ValueError(f"Q{i} missing fields")
        if q["mode"] not in MODES: raise ValueError(f"Q{i} has invalid mode {q['mode']}")
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

# ---------- Jobs ----------
async def tick_edit(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id)
    if not st: return
    left = int(round(context.job.data["end_ts"] - time.time()))
    left = max(0, (left//5)*5)  # edit every 5s window
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=st.q_msg_id,
            text=fmt_question(st.theme, st.q_index+1, st.limit, st.questions[st.q_index], left),
            reply_markup=answer_kb(st.questions[st.q_index], st.q_index+1),
            parse_mode="Markdown"
        )
    except Exception:
        pass

async def close_question(context: ContextTypes.DEFAULT_TYPE):
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
    # reveal then next
    context.job_queue.run_once(reveal_answer, when=DELAY_REVEAL, data={"chat_id": chat_id})

async def reveal_answer(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id)
    if not st: return
    q = st.questions[st.q_index]
    # Fastest 3
    pairs=list(st.per_q_answers.get(st.q_index, {}).items())
    fastest=sorted((p for p in pairs if p[1].is_correct), key=lambda x: x[1].elapsed)[:3]
    lines=[fmt_correct(st.theme, q)]
    if fastest:
        s=style_from_theme(st.theme)
        lines.append(f"\n{s['fast']} Fastest:")
        for uid, rec in fastest:
            name = st.players.get(uid, str(uid))
            lines.append(f"{s['dot']} {name} — {rec.points} pts ({rec.elapsed:.2f}s)")
    await context.bot.send_message(chat_id=chat_id, text="\n".join(lines), parse_mode="Markdown")
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
    context.job_queue.run_repeating(
        tick_edit, interval=5.0, first=5.0, data={"chat_id": st.chat_id, "end_ts": end_ts},
        name=f"tick:{st.chat_id}:{st.q_index}"
    )
    context.job_queue.run_once(
        close_question, when=QUESTION_TIME, data={"chat_id": st.chat_id},
        name=f"close:{st.chat_id}:{st.q_index}"
    )

# ---------- Admin helper ----------
async def is_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
    try:
        m = await context.bot.get_chat_member(chat_id, user_id)
        return m.status in ("administrator", "creator")
    except Exception:
        return False

# ---------- Commands ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    theme = context.chat_data.get("theme", DEFAULT_THEME)
    mode  = context.chat_data.get("mode")  # required before /startquiz
    length= context.chat_data.get("length")
    s = style_from_theme(theme)
    await update.message.reply_text(
        f"{s['q']} *Quiz Bot*\n{s['rule']}\n"
        f"1) /menu — pick *Mode* (Beginner/Standard/Expert) & *Length* (10/20/30/40/50) & Theme\n"
        f"2) /startquiz — start after selecting mode (admin-only in groups)\n"
        f"3) /stopquiz — end early\n"
        f"4) /myresults — personal breakdown after\n"
        f"5) /reset — clear previous session/config\n\n"
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
        f"*Setup Required*: choose a *Mode* and *Length*.\nCurrent → Mode: *{mode}*, Length: *{length}*, Theme: *{theme.title()}*",
        reply_markup=InlineKeyboardMarkup(rows), parse_mode="Markdown"
    )

async def cfg_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data  # cfg:mode:beginner | cfg:len:20 | cfg:theme:toggle
    if not data.startswith("cfg:"): return
    _, kind, val = data.split(":")
    chat_id = q.message.chat.id

    # If a game exists, wipe it when settings change (auto "delete logs")
    if chat_id in GAMES:
        GAMES.pop(chat_id, None)

    if kind == "mode":
        if val not in MODES:
            await q.edit_message_text("Invalid mode."); return
        context.chat_data["mode"] = val
        await q.edit_message_text(f"Mode set to *{val.title()}*. Use /menu to change other settings or /startquiz to begin.", parse_mode="Markdown")
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

    # admin-only for groups
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

    # starting a new session clears any previous (auto 'delete logs')
    GAMES.pop(chat_id, None)

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
        f"⏱ {QUESTION_TIME}s each, faster taps = more points.",
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

async def cmd_myresults(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user
    st = GAMES.get(chat_id)
    if not st or st.q_index < st.limit - 1:
        await update.message.reply_text("No finished quiz found here yet."); return

    total_pts = 0; correct_cnt = 0
    lines = ["📊 *Your Results*"]
    for i, q in enumerate(st.questions, start=1):
        rec = st.per_q_answers.get(i-1, {}).get(user.id)
        if not rec:
            lines.append(f"Q{i}: No answer ❌ — 0 pts")
        else:
            total_pts += rec.points
            if rec.is_correct: correct_cnt += 1
            lines.append(f"Q{i}: {'Correct ✅' if rec.is_correct else 'Wrong ❌'} — {rec.points} pts")
    lines.append(f"\nTotal: {correct_cnt}/{st.limit} correct — {total_pts} pts")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def finish_quiz(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    s = style_from_theme(st.theme)
    scores: Dict[int,int] = {}; corrects: Dict[int,int] = {}
    for qidx in range(st.limit):
        for uid, rec in st.per_q_answers.get(qidx, {}).items():
            scores[uid] = scores.get(uid, 0) + rec.points
            if rec.is_correct: corrects[uid] = corrects.get(uid, 0) + 1

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

    lines=[f"{s['flag']} *Final Results* — Top 10"]
    for rank,(uid,pts_) in enumerate(ranking[:10], start=1):
        name = st.players.get(uid, str(uid))
        corr = corrects.get(uid, 0)
        medal = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}."
        lines.append(f"{medal} {name} — {corr}/{st.limit} correct — {pts_} pts")
    lines.append("\nUse /myresults to see your breakdown.")
    await context.bot.send_message(chat_id=st.chat_id, text="\n".join(lines), parse_mode="Markdown")

    GAMES.pop(st.chat_id, None)  # clear logs/state

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
    context.chat_data.clear()
    await update.message.reply_text("✅ Reset done. Use /menu to set Mode & Length, then /startquiz.")

def build_app() -> Application:
    token = os.getenv("BOT_TOKEN")
    if not token: raise RuntimeError("Set BOT_TOKEN environment variable.")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("menu",  cmd_menu))
    app.add_handler(CallbackQueryHandler(cfg_click, pattern=r"^cfg:"))
    app.add_handler(CommandHandler("startquiz", cmd_startquiz))
    app.add_handler(CommandHandler("myresults", cmd_myresults))
    app.add_handler(CommandHandler("stopquiz", cmd_stop))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CallbackQueryHandler(on_answer, pattern=r"^ans:\d+:\d$"))
    return app

if __name__ == "__main__":
    print("Starting quiz bot…")
    build_app().run_polling(close_loop=False)
