# Telegram Quiz Bot — simple & aesthetic, mobile-first
# pip deps: python-telegram-bot==21.*

from __future__ import annotations
import os, json, time, random, math
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ====== CONFIG ======
QUESTION_TIME = 15           # seconds to answer
DELAY_REVEAL = 5             # wait 5s after time's up, then show answer
DELAY_NEXT   = 10            # wait 10s after reveal, then next question
POINTS_MAX   = 100           # max points if answered instantly
QUESTIONS_FILE = "questions.json"
# =====================

@dataclass
class QItem:
    text: str
    options: List[str]
    correct: int  # index 0..3

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
    questions: List[QItem]                # already shuffled order
    q_index: int = 0                       # 0-based pointer
    q_msg_id: Optional[int] = None         # message id of question (for countdown edits)
    q_start_ts: Optional[float] = None
    locked: bool = False                   # question closed to new answers
    per_q_answers: Dict[int, Dict[int, AnswerRec]] = field(default_factory=dict)  # q-> user_id -> AnswerRec
    players: Dict[int, str] = field(default_factory=dict)         # user_id -> display name

# In-memory state: one game per chat
GAMES: Dict[int, GameState] = {}

# ---------- Utilities ----------
def load_questions() -> List[QItem]:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for i, q in enumerate(data, start=1):
        if "text" not in q or "options" not in q or "correct" not in q:
            raise ValueError(f"Question {i} missing fields")
        if len(q["options"]) != 4:
            raise ValueError(f"Question {i} must have exactly 4 options")
        c = int(q["correct"])
        if not 0 <= c < 4:
            raise ValueError(f"Question {i} has invalid correct index {c}")
        out.append(QItem(text=q["text"], options=list(q["options"]), correct=c))
    return out

def build_keyboard(q: QItem, q_number: int) -> InlineKeyboardMarkup:
    # callback carries current q_number and option index so late taps are ignored
    rows = [[InlineKeyboardButton(f"{chr(65+i)}. {opt}", callback_data=f"ans:{q_number}:{i}")]
            for i, opt in enumerate(q.options)]
    return InlineKeyboardMarkup(rows)

def points_from_elapsed(elapsed: float) -> int:
    """Linear: instant -> 100, at 15s -> 0 (or below)"""
    if elapsed >= QUESTION_TIME:
        return 0
    remaining = max(0.0, QUESTION_TIME - elapsed)
    pts = remaining / QUESTION_TIME * POINTS_MAX
    return int(round(pts))  # 0..100

def fmt_question(qnum: int, q: QItem, remaining: Optional[int] = None) -> str:
    head = f"❓ *Question {qnum}*\n{q.text}"
    if remaining is not None:
        head += f"\n\n⏱ *{remaining}s left*"
    return head

def fmt_correct_line(q: QItem) -> str:
    letter = chr(65 + q.correct)
    return f"✅ The correct answer is **{letter}. {q.options[q.correct]}**"

async def send_countdown_tick(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data
    chat_id = data["chat_id"]
    msg_id = data["msg_id"]
    qnum = data["qnum"]
    q: QItem = data["q"]
    end_ts = data["end_ts"]

    remaining = int(round(end_ts - time.time()))
    if remaining < 0:
        # stop repeating
        context.job.schedule_removal()
        return
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg_id,
            text=fmt_question(qnum, q, remaining),
            reply_markup=build_keyboard(q, qnum),
            parse_mode="Markdown"
        )
    except Exception:
        # edits can fail due to flood limits; it's fine to skip a tick
        pass

async def close_question(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data
    chat_id = data["chat_id"]
    st = GAMES.get(chat_id)
    if not st: return
    st.locked = True
    # optionally mark "Time's up" by editing once more without buttons
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=st.q_msg_id,
            text=fmt_question(st.q_index + 1, st.questions[st.q_index], 0),
            parse_mode="Markdown"
        )
    except Exception:
        pass
    # schedule reveal after DELAY_REVEAL
    context.job_queue.run_once(reveal_answer, when=DELAY_REVEAL, data={"chat_id": chat_id})

async def reveal_answer(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id)
    if not st: return
    q = st.questions[st.q_index]
    # Build fastest correct (top 3)
    records = list((st.per_q_answers.get(st.q_index, {})).items())  # (uid, AnswerRec)
    fastest = sorted([r for r in records if r[1].is_correct], key=lambda x: x[1].elapsed)[:3]
    lines = [fmt_correct_line(q)]
    if fastest:
        lines.append("\n🏃 Fastest correct:")
        for uid, rec in fastest:
            name = st.players.get(uid, str(uid))
            lines.append(f"• {name} — {rec.points} pts ({rec.elapsed:.2f}s)")
    await context.bot.send_message(chat_id=chat_id, text="\n".join(lines), parse_mode="Markdown")
    # schedule next question after DELAY_NEXT
    context.job_queue.run_once(next_question, when=DELAY_NEXT, data={"chat_id": chat_id})

async def next_question(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id)
    if not st: return
    st.q_index += 1
    if st.q_index >= len(st.questions):
        await finish_quiz(context, st)
        return
    await ask_question(context, st)

async def ask_question(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    chat_id = st.chat_id
    q = st.questions[st.q_index]

    # send initial message with full 15s left
    msg = await context.bot.send_message(
        chat_id=chat_id,
        text=fmt_question(st.q_index + 1, q, QUESTION_TIME),
        reply_markup=build_keyboard(q, st.q_index + 1),
        parse_mode="Markdown"
    )
    st.q_msg_id = msg.message_id
    st.q_start_ts = time.time()
    st.locked = False

    # schedule countdown ticks each second for QUESTION_TIME seconds
    end_ts = st.q_start_ts + QUESTION_TIME
    context.job_queue.run_repeating(
        send_countdown_tick,
        interval=1.0,
        first=1.0,
        data={"chat_id": chat_id, "msg_id": st.q_msg_id, "qnum": st.q_index + 1, "q": q, "end_ts": end_ts},
        name=f"cd:{chat_id}:{st.q_index}"
    )
    # schedule closure at end
    context.job_queue.run_once(
        close_question,
        when=QUESTION_TIME,
        data={"chat_id": chat_id},
        name=f"close:{chat_id}:{st.q_index}"
    )

# ---------- Handlers ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Quiz bot ready!\n"
        f"• 50 questions (random order)\n"
        f"• {QUESTION_TIME}s per question with a live countdown\n"
        "• Speed scoring (max 100 points)\n\n"
        "Commands:\n"
        "/startquiz — begin the quiz\n"
        "/myresults — your personal breakdown (after the quiz)"
    )

async def cmd_startquiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user

    # Load & shuffle questions
    all_qs = load_questions()
    if len(all_qs) < 50:
        await update.message.reply_text("Need at least 50 questions in questions.json.")
        return
    qs = all_qs.copy()
    random.shuffle(qs)
    qs = qs[:50]  # take first 50 after shuffle

    # Shuffle options per question (and re-map correct)
    for i, q in enumerate(qs):
        pairs = list(enumerate(q.options))
        random.shuffle(pairs)
        new_opts = [txt for _, txt in pairs]
        new_correct = next(idx for idx, (orig_idx, _) in enumerate(pairs) if orig_idx == q.correct)
        qs[i] = QItem(text=q.text, options=new_opts, correct=new_correct)

    st = GameState(chat_id=chat_id, started_by=user.id, questions=qs)
    GAMES[chat_id] = st

    await update.message.reply_text(
        f"🎉 Starting quiz! 50 questions.\n"
        f"You have {QUESTION_TIME}s per question. Faster answers = more points."
    )
    await ask_question(context, st)

async def on_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # quick, to remove loading spinner

    user = query.from_user
    chat_id = query.message.chat.id
    data = query.data  # ans:<qnum>:<opt>
    try:
        _, qnum_str, opt_str = data.split(":")
        qnum = int(qnum_str)           # 1-based
        opt = int(opt_str)
    except Exception:
        return

    st = GAMES.get(chat_id)
    if not st:
        return

    # Only accept answers for the current open question
    if st.q_index + 1 != qnum:
        try:
            await query.answer("This question is already closed.", show_alert=False)
        except Exception:
            pass
        return
    if st.locked or st.q_start_ts is None:
        try:
            await query.answer("Time is up!", show_alert=False)
        except Exception:
            pass
        return

    # Register player name
    st.players[user.id] = (user.full_name or user.username or str(user.id))[:64]

    # Ensure per-q dict
    if st.q_index not in st.per_q_answers:
        st.per_q_answers[st.q_index] = {}

    # First answer only
    if user.id in st.per_q_answers[st.q_index]:
        try:
            await query.answer("Only your first answer counts.", show_alert=False)
        except Exception:
            pass
        return

    elapsed = max(0.0, time.time() - st.q_start_ts)
    is_correct = (opt == st.questions[st.q_index].correct)
    pts = points_from_elapsed(elapsed) if is_correct else 0

    st.per_q_answers[st.q_index][user.id] = AnswerRec(choice=opt, is_correct=is_correct, elapsed=elapsed, points=pts)

    # Tiny feedback toast
    try:
        await query.answer(f"{'✅' if is_correct else '❌'} {pts} pts", show_alert=False)
    except Exception:
        pass

async def cmd_myresults(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user
    st = GAMES.get(chat_id)
    if not st or st.q_index < len(st.questions) - 1:
        await update.message.reply_text("No finished quiz found here yet.")
        return

    # Build personal breakdown
    total_pts = 0
    correct_cnt = 0
    lines = [f"📊 *Your Results*"]
    for i, q in enumerate(st.questions, start=1):
        rec = st.per_q_answers.get(i-1, {}).get(user.id)
        if not rec:
            lines.append(f"Q{i}: — No answer ❌ — 0 pts")
        else:
            total_pts += rec.points
            if rec.is_correct:
                correct_cnt += 1
            status = "Correct ✅" if rec.is_correct else "Wrong ❌"
            lines.append(f"Q{i}: {status} — {rec.points} pts")
    lines.append(f"\nTotal: {correct_cnt}/50 correct — {total_pts} pts")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def finish_quiz(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    # Leaderboard (sum points)
    scores: Dict[int, int] = {}
    corrects: Dict[int, int] = {}
    for qidx in range(len(st.questions)):
        for uid, rec in st.per_q_answers.get(qidx, {}).items():
            scores[uid] = scores.get(uid, 0) + rec.points
            if rec.is_correct:
                corrects[uid] = corrects.get(uid, 0) + 1

    if not scores:
        await context.bot.send_message(chat_id=st.chat_id, text="Quiz ended. No participants 😅")
        GAMES.pop(st.chat_id, None)
        return

    # Sort and print top
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    lines = ["🏁 *Final Results* (Top 10)"]
    for rank, (uid, pts) in enumerate(ranking[:10], start=1):
        name = st.players.get(uid, str(uid))
        corr = corrects.get(uid, 0)
        lines.append(f"{rank}. {name} — {corr}/50 correct — {pts} pts")
    lines.append("\nUse /myresults to see your question-by-question breakdown.")
    await context.bot.send_message(chat_id=st.chat_id, text="\n".join(lines), parse_mode="Markdown")

    # Optional: auto-send detailed results to starter (if they participated)
    if st.started_by in scores:
        # build detailed list for starter
        total_pts = scores[st.started_by]
        corr = corrects.get(st.started_by, 0)
        detail = [f"📬 *Your Final Result* — {corr}/50 correct — {total_pts} pts"]
        for i, q in enumerate(st.questions, start=1):
            rec = st.per_q_answers.get(i-1, {}).get(st.started_by)
            if not rec:
                detail.append(f"Q{i}: No answer ❌ — 0 pts")
            else:
                status = "Correct ✅" if rec.is_correct else "Wrong ❌"
                detail.append(f"Q{i}: {status} — {rec.points} pts")
        try:
            # If the user has never started the bot in DM, this may fail; that's okay.
            await context.bot.send_message(chat_id=st.started_by, text="\n".join(detail), parse_mode="Markdown")
        except Exception:
            pass

    # cleanup
    GAMES.pop(st.chat_id, None)

def build_app() -> Application:
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Set BOT_TOKEN environment variable (from @BotFather).")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("startquiz", cmd_startquiz))
    app.add_handler(CommandHandler("myresults", cmd_myresults))
    app.add_handler(CallbackQueryHandler(on_answer, pattern=r"^ans:\d+:\d$"))
    return app

if __name__ == "__main__":
    print("Starting quiz bot…")
    build_app().run_polling(close_loop=False)
