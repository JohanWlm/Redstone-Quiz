# Telegram Quiz Bot — sessions, silent mode, taps feedback
# deps: python-telegram-bot[job-queue]==21.*

from __future__ import annotations
import os, json, time, random
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes
)

# ===== CONFIG =====
QUESTION_TIME = 15     # seconds to answer
DELAY_REVEAL = 5       # wait 5s after time's up, then show answer
DELAY_NEXT   = 10      # wait 10s after reveal/silent, then next question
POINTS_MAX   = 100     # max points if answered instantly
QUESTIONS_FILE = "questions.json"
DEFAULT_SESSION_SIZE = 50  # fallback if user doesn't choose
ALLOWED_SESSION_SIZES = (10, 20, 30, 40, 50)
# ===================

@dataclass
class QItem:
    text: str
    options: List[str]
    correct: int  # 0..3

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
    limit: int                          # how many questions this session (10/20/..)
    reveal_enabled: bool = True         # if False => no per-question “correct answer” message
    q_index: int = 0                    # 0-based pointer
    q_msg_id: Optional[int] = None      # question message id (for countdown edits)
    q_start_ts: Optional[float] = None
    locked: bool = False                # question closed to new answers
    per_q_answers: Dict[int, Dict[int, AnswerRec]] = field(default_factory=dict)  # q-> user_id -> AnswerRec
    players: Dict[int, str] = field(default_factory=dict)  # user_id -> name

GAMES: Dict[int, GameState] = {}  # one game per chat

# ---------- load & utils ----------
def load_questions() -> List[QItem]:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[QItem] = []
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

def compute_points(elapsed: float) -> int:
    if elapsed >= QUESTION_TIME: return 0
    remaining = max(0.0, QUESTION_TIME - elapsed)
    return int(round((remaining / QUESTION_TIME) * POINTS_MAX))  # 0..100

def build_answer_kb(q: QItem, qnum: int) -> InlineKeyboardMarkup:
    # include qnum so late taps are ignored
    rows = [[InlineKeyboardButton(f"{chr(65+i)}. {opt}", callback_data=f"ans:{qnum}:{i}")]
            for i, opt in enumerate(q.options)]
    return InlineKeyboardMarkup(rows)

def fmt_question(qnum: int, q: QItem, seconds_left: Optional[int]=None) -> str:
    head = f"❓ *Question {qnum}*\n{q.text}"
    if seconds_left is not None:
        head += f"\n\n⏱ *{seconds_left}s left*"
    return head

def fmt_correct(q: QItem) -> str:
    return f"✅ The correct answer is *{chr(65+q.correct)}. {q.options[q.correct]}*"

# ---------- countdown ticks ----------
async def tick_edit(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data
    chat_id = data["chat_id"]; msg_id = data["msg_id"]
    qnum = data["qnum"]; q: QItem = data["q"]; end_ts = data["end_ts"]
    left = int(round(end_ts - time.time()))
    if left < 0:
        context.job.schedule_removal(); return
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=msg_id,
            text=fmt_question(qnum, q, left),
            reply_markup=build_answer_kb(q, qnum),
            parse_mode="Markdown"
        )
    except Exception:
        pass  # safe to ignore occasional edit failures

async def close_question(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id)
    if not st: return
    st.locked = True
    # final edit with 0s left (no buttons)
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=st.q_msg_id,
            text=fmt_question(st.q_index + 1, st.questions[st.q_index], 0),
            parse_mode="Markdown"
        )
    except Exception:
        pass
    # reveal or go straight to next
    if st.reveal_enabled:
        context.job_queue.run_once(reveal_answer, when=DELAY_REVEAL, data={"chat_id": chat_id})
    else:
        context.job_queue.run_once(next_question, when=DELAY_NEXT, data={"chat_id": chat_id})

async def reveal_answer(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id)
    if not st: return
    q = st.questions[st.q_index]

    # fastest 3 correct
    pairs = list(st.per_q_answers.get(st.q_index, {}).items())
    fastest = sorted((p for p in pairs if p[1].is_correct), key=lambda x: x[1].elapsed)[:3]
    lines = [fmt_correct(q)]
    if fastest:
        lines.append("\n🏃 Fastest correct:")
        for uid, rec in fastest:
            name = st.players.get(uid, str(uid))
            lines.append(f"• {name} — {rec.points} pts ({rec.elapsed:.2f}s)")
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
    chat_id = st.chat_id
    q = st.questions[st.q_index]
    # send with full 15s left
    msg = await context.bot.send_message(
        chat_id=chat_id,
        text=fmt_question(st.q_index + 1, q, QUESTION_TIME),
        reply_markup=build_answer_kb(q, st.q_index + 1),
        parse_mode="Markdown"
    )
    st.q_msg_id = msg.message_id
    st.q_start_ts = time.time()
    st.locked = False

    # schedule ticks + close
    end_ts = st.q_start_ts + QUESTION_TIME
    context.job_queue.run_repeating(
        tick_edit, interval=1.0, first=1.0,
        data={"chat_id": chat_id, "msg_id": st.q_msg_id, "qnum": st.q_index + 1, "q": q, "end_ts": end_ts},
        name=f"tick:{chat_id}:{st.q_index}"
    )
    context.job_queue.run_once(
        close_question, when=QUESTION_TIME,
        data={"chat_id": chat_id}, name=f"close:{chat_id}:{st.q_index}"
    )

# ---------- commands ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Quiz bot ready!\n"
        f"• Choose a session length: /menu (10/20/30/40/50)\n"
        f"• Start immediately: /startquiz 10  (or 20/30/40/50)\n"
        f"• Stop anytime: /stopquiz\n"
        f"• Per-question reveal: on by default; toggle in /menu\n"
        f"• Your results after: /myresults"
    )

def _shuffle_questions(src: List[QItem]) -> List[QItem]:
    qs = src.copy()
    random.shuffle(qs)
    # also shuffle answer options per question
    out: List[QItem] = []
    for q in qs:
        pairs = list(enumerate(q.options))
        random.shuffle(pairs)
        new_opts = [txt for _, txt in pairs]
        new_correct = next(idx for idx, (orig_i, _) in enumerate(pairs) if orig_i == q.correct)
        out.append(QItem(text=q.text, options=new_opts, correct=new_correct))
    return out

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = [
        [InlineKeyboardButton(f"{n} questions", callback_data=f"cfg:len:{n}")]
        for n in ALLOWED_SESSION_SIZES
    ]
    rows.append([InlineKeyboardButton("Toggle reveal on/off", callback_data="cfg:reveal:toggle")])
    await update.message.reply_text("Choose session length & settings:", reply_markup=InlineKeyboardMarkup(rows))

async def cfg_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data  # cfg:len:20 or cfg:reveal:toggle
    if not data.startswith("cfg:"): return
    _, kind, val = data.split(":")
    chat_id = q.message.chat.id
    st = GAMES.get(chat_id)

    if kind == "len":
        try:
            chosen = int(val)
            if chosen not in ALLOWED_SESSION_SIZES:
                raise ValueError()
        except Exception:
            await q.edit_message_text("Invalid length. Use /menu again."); return
        # store hint on the chat by creating a lightweight pending state
        # (we'll read it when /startquiz runs without an explicit number)
        context.chat_data["preferred_len"] = chosen
        await q.edit_message_text(f"Session length set to {chosen}. Use /startquiz to begin.")
        return

    if kind == "reveal" and val == "toggle":
        # flip default preference in chat_data
        pref = context.chat_data.get("reveal_enabled", True)
        pref = not pref
        context.chat_data["reveal_enabled"] = pref
        await q.edit_message_text(f"Per-question reveal is now {'ON' if pref else 'OFF'}.")
        return

async def cmd_startquiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user

    # parse desired count from argument or chat preference
    limit = None
    if context.args:
        try:
            limit = int(context.args[0])
        except Exception:
            pass
    if limit not in ALLOWED_SESSION_SIZES:
        limit = context.chat_data.get("preferred_len", DEFAULT_SESSION_SIZE)
        if limit not in ALLOWED_SESSION_SIZES:
            limit = DEFAULT_SESSION_SIZE

    all_qs = load_questions()
    if len(all_qs) < 50:
        await update.message.reply_text("Need at least 50 questions in questions.json.")
        return

    qs = _shuffle_questions(all_qs)[:limit]

    reveal = context.chat_data.get("reveal_enabled", True)
    st = GameState(chat_id=chat_id, started_by=user.id, questions=qs, limit=limit, reveal_enabled=reveal)
    GAMES[chat_id] = st

    await update.message.reply_text(
        f"🎉 Starting quiz! {limit} questions.\n"
        f"You have {QUESTION_TIME}s per question. Faster answers = more points.\n"
        f"{'Answers will be revealed each round.' if reveal else 'Reveal is OFF (silent mode).'}"
    )
    await ask_question(context, st)

async def on_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # quick remove spinner

    user = query.from_user
    chat_id = query.message.chat.id
    data = query.data  # ans:<qnum>:<opt>
    try:
        _, qnum_str, opt_str = data.split(":")
        qnum = int(qnum_str)  # 1-based
        opt = int(opt_str)
    except Exception:
        return

    st = GAMES.get(chat_id)
    if not st: return

    if st.q_index + 1 != qnum:
        try: await query.answer("This question is already closed.", show_alert=False)
        except Exception: pass
        return
    if st.locked or st.q_start_ts is None:
        try: await query.answer("Time is up!", show_alert=False)
        except Exception: pass
        return

    # register player name
    st.players[user.id] = (user.full_name or user.username or str(user.id))[:64]
    # ensure dict
    st.per_q_answers.setdefault(st.q_index, {})

    # first answer only
    if user.id in st.per_q_answers[st.q_index]:
        try: await query.answer("Only your first answer counts.", show_alert=False)
        except Exception: pass
        return

    elapsed = max(0.0, time.time() - st.q_start_ts)
    is_correct = (opt == st.questions[st.q_index].correct)
    pts = compute_points(elapsed) if is_correct else 0
    st.per_q_answers[st.q_index][user.id] = AnswerRec(choice=opt, is_correct=is_correct, elapsed=elapsed, points=pts)

    # show toast + (in groups) try to DM the user a short ack
    try:
        await query.answer(f"{'✅' if is_correct else '❌'} {pts} pts", show_alert=False)
    except Exception:
        pass
    try:
        # DM might fail if the user never opened the bot in private
        await context.bot.send_message(
            chat_id=user.id,
            text=f"Q{qnum}: {'Correct ✅' if is_correct else 'Wrong ❌'} — {pts} pts (in {elapsed:.2f}s)"
        )
    except Exception:
        pass

async def cmd_myresults(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user
    st = GAMES.get(chat_id)
    if not st or st.q_index < st.limit - 1:
        await update.message.reply_text("No finished quiz found here yet.")
        return

    total_pts = 0; correct_cnt = 0
    lines = [f"📊 *Your Results*"]
    for i, q in enumerate(st.questions, start=1):
        rec = st.per_q_answers.get(i-1, {}).get(user.id)
        if not rec:
            lines.append(f"Q{i}: No answer ❌ — 0 pts")
        else:
            total_pts += rec.points
            correct_cnt += 1 if rec.is_correct else 0
            lines.append(f"Q{i}: {'Correct ✅' if rec.is_correct else 'Wrong ❌'} — {rec.points} pts")
    lines.append(f"\nTotal: {correct_cnt}/{st.limit} correct — {total_pts} pts")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def finish_quiz(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    # aggregate
    scores: Dict[int, int] = {}
    corrects: Dict[int, int] = {}
    for qidx in range(st.limit):
        for uid, rec in st.per_q_answers.get(qidx, {}).items():
            scores[uid] = scores.get(uid, 0) + rec.points
            if rec.is_correct:
                corrects[uid] = corrects.get(uid, 0) + 1

    if not scores:
        await context.bot.send_message(chat_id=st.chat_id, text="Quiz ended. No participants 😅")
        GAMES.pop(st.chat_id, None); return

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    lines = [f"🏁 *Final Results* ({st.limit} questions) — Top 10"]
    for rank, (uid, pts) in enumerate(ranking[:10], start=1):
        name = st.players.get(uid, str(uid))
        corr = corrects.get(uid, 0)
        lines.append(f"{rank}. {name} — {corr}/{st.limit} correct — {pts} pts")
    lines.append("\nUse /myresults to see your per-question breakdown.")
    await context.bot.send_message(chat_id=st.chat_id, text="\n".join(lines), parse_mode="Markdown")

    # cleanup
    GAMES.pop(st.chat_id, None)

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = GAMES.get(chat_id)
    if not st:
        await update.message.reply_text("No active quiz to stop.")
        return
    # set limit to current q_index+1 and finish
    st.limit = st.q_index + 1
    await finish_quiz(context, st)

def build_app() -> Application:
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Set BOT_TOKEN environment variable (from @BotFather).")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CallbackQueryHandler(cfg_click, pattern=r"^cfg:"))
    app.add_handler(CommandHandler("startquiz", cmd_startquiz))
    app.add_handler(CommandHandler("myresults", cmd_myresults))
    app.add_handler(CommandHandler("stopquiz", cmd_stop))
    app.add_handler(CallbackQueryHandler(on_answer, pattern=r"^ans:\d+:\d$"))
    return app

if __name__ == "__main__":
    print("Starting quiz bot…")
    build_app().run_polling(close_loop=False)
