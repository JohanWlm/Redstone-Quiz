
# Telegram Quiz Bot — hardened + admin-only + "Start" callback (fixes config loss)
# See README in message: supports /startquiz <mode> <length> and a Start button with encoded config.
from __future__ import annotations
import os, json, time, random, math, html, logging, threading, uuid, asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut, NetworkError, BadRequest
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("quizbot")

# ---------- CONFIG ----------
QUESTION_TIME = int(os.getenv("QUESTION_TIME", "10"))
DELAY_NEXT    = int(os.getenv("DELAY_NEXT", "5"))
POINTS_MAX    = int(os.getenv("POINTS_MAX", "100"))
QUESTIONS_FILE = os.getenv("QUESTIONS_FILE", "questions.json")

ALLOWED_SESSION_SIZES = (10, 20, 30, 40, 50)
MODES = ("beginner", "standard", "expert")

ADMINS_ONLY = True
TICK_SECONDS = float(os.getenv("TICK_SECONDS", "1.0"))

INSTANCE_ID = os.getenv("RAILWAY_REPLICA_ID") or str(uuid.uuid4())[:8]

def esc(s: str) -> str:
    return html.escape(str(s), quote=True)

# ---------- Retry wrapper ----------
async def _tg_call(desc: str, coro, *args, **kwargs):
    attempts = 0
    while True:
        attempts += 1
        try:
            return await coro(*args, **kwargs)
        except RetryAfter as e:
            wait = float(getattr(e, "retry_after", 1.0)) + 0.5
            log.warning("%s: RetryAfter %.2fs (attempt %d/3)", desc, wait, attempts)
            if attempts >= 3: return None
            await asyncio.sleep(wait)
        except (TimedOut, NetworkError) as e:
            wait = min(2.0 * attempts, 6.0)
            log.warning("%s: %s; backoff %.2fs (attempt %d/3)", desc, type(e).__name__, wait, attempts)
            if attempts >= 3: return None
            await asyncio.sleep(wait)
        except BadRequest as e:
            log.debug("%s: BadRequest ignored: %s", desc, e)
            return None
        except Exception as e:
            log.error("%s: unexpected %s: %s", desc, type(e).__name__, e)
            return None

# ---------- Data ----------
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
    q_start_ts: Optional[float] = None
    locked: bool = False

    per_q_answers: Dict[int, Dict[int, AnswerRec]] = field(default_factory=dict)
    answered_now: Dict[int, Set[int]] = field(default_factory=dict)

    players: Dict[int, str] = field(default_factory=dict)
    totals: Dict[int, int] = field(default_factory=dict)
    corrects: Dict[int, int] = field(default_factory=dict)

GAMES: Dict[int, GameState] = {}
LAST: Dict[int, dict] = {}
SETTINGS: Dict[int, Dict[str, int | str]] = {}

# ---------- Helpers ----------
def points(elapsed: float) -> int:
    if elapsed >= QUESTION_TIME: return 0
    return int(round((max(0.0, QUESTION_TIME - elapsed) / QUESTION_TIME) * POINTS_MAX))

def answer_kb(q: QItem, qnum: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(opt, callback_data=f"ans:{qnum}:{i}")]
                                 for i, opt in enumerate(q.options)])

def fmt_question(qnum: int, total: int, q: QItem, left: Optional[int] = None, locked_count: Optional[int] = None) -> str:
    s = f"❓ <b>Question {qnum}/{total}</b>\n{esc(q.text)}"
    if left is not None: s += f"\\n\\n⏱ <b>{int(left)}s left</b>"
    if locked_count is not None: s += f"\\n🗳 <b>Answers locked:</b> {locked_count}"
    return s

def load_questions() -> List[QItem]:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f: data = json.load(f)
    out = []
    for i, q in enumerate(data, start=1):
        if not all(k in q for k in ("text","options","correct","mode")): raise ValueError(f"Q{i} missing fields")
        if q["mode"] not in MODES: raise ValueError(f"Q{i} invalid mode {q['mode']}")
        if len(q["options"]) != 4: raise ValueError(f"Q{i} must have 4 options")
        c = int(q["correct"]); 
        if not 0 <= c < 4: raise ValueError(f"Q{i} invalid correct index {c}")
        out.append(QItem(q["text"], list(q["options"]), c, q["mode"]))
    return out

def filter_by_mode(all_qs: List[QItem], mode: str) -> List[QItem]: return [q for q in all_qs if q.mode == mode]

def shuffle_qs(qs: List[QItem]) -> List[QItem]:
    qs = qs.copy(); random.shuffle(qs); out=[]
    for q in qs:
        pairs=list(enumerate(q.options)); random.shuffle(pairs)
        new_opts=[t for _,t in pairs]; new_correct=next(i for i,(orig,_) in enumerate(pairs) if orig==q.correct)
        out.append(QItem(q.text,new_opts,new_correct,q.mode))
    return out

def compute_totals(st: GameState): return st.totals, st.corrects

def cancel_jobs_for_chat(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    try:
        for job in context.job_queue.jobs():
            name = getattr(job,"name","") or ""
            if name.startswith(f"tick:{chat_id}:") or name.startswith(f"close:{chat_id}:") or name.startswith(f"gap:{chat_id}:"):
                job.schedule_removal()
    except Exception as e: log.warning("job cleanup error: %s", e)

# ---------- Jobs ----------
async def tick_edit(context: ContextTypes.DEFAULT_TYPE):
    d = context.job.data; chat_id=d["chat_id"]; msg_id=d["msg_id"]; end_ts=d["end_ts"]; qidx=d["qidx"]
    st = GAMES.get(chat_id); 
    if not st or st.q_index != qidx: context.job.schedule_removal(); return
    left = max(0, int(math.ceil(end_ts - time.time()))); 
    if left <= 0: context.job.schedule_removal()
    await _tg_call("tick.edit", context.bot.edit_message_text, chat_id=chat_id, message_id=msg_id,
                   text=fmt_question(st.q_index+1, st.limit, st.questions[st.q_index], left, len(st.per_q_answers.get(st.q_index,{}))),
                   reply_markup=answer_kb(st.questions[st.q_index], st.q_index+1), parse_mode=ParseMode.HTML)

async def gap_tick(context: ContextTypes.DEFAULT_TYPE):
    d = context.job.data; chat_id=d["chat_id"]; msg_id=d["msg_id"]; end_ts=d["end_ts"]
    st = GAMES.get(chat_id); 
    if not st: context.job.schedule_removal(); return
    left = max(0, int(math.ceil(end_ts - time.time())))
    if left <= 0:
        context.job.schedule_removal()
        await _tg_call("gap.final", context.bot.edit_message_text, chat_id=chat_id, message_id=msg_id, text="🚀 Starting next question…"); return
    await _tg_call("gap.tick", context.bot.edit_message_text, chat_id=chat_id, message_id=msg_id,
                   text=f"⏭️ Next question is coming in {left}s…")

async def post_round_recap(context, st: GameState, qidx: int):
    q = st.questions[qidx]; per = st.per_q_answers.get(qidx, {})
    scorers = sorted(((uid, rec.points) for uid, rec in per.items() if rec.is_correct), key=lambda t:t[1], reverse=True)
    scores, corrects = compute_totals(st); ranking = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    lines=[f"📘 <b>Q{qidx+1} Result</b>", f"✅ Correct answer: <b>{esc(q.options[q.correct])}</b>"]
    if scorers:
        names=[f"{esc(st.players.get(uid,str(uid)))} (+{pts})" for uid,pts in scorers[:5]]
        others=max(0,len(scorers)-5); lines.append("🏎️ Fastest correct: "+", ".join(names)+ (f" … +{others} more" if others else ""))
    else: lines.append("😶 No correct answers this round.")
    if ranking:
        lines.append("\\n🏁 <b>Current Leaderboard</b> (Top 5)")
        for rank,(uid,total) in enumerate(ranking[:5], start=1):
            name=esc(st.players.get(uid,str(uid))); corr=corrects.get(uid,0)
            medal=("🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}.")
            lines.append(f"{medal} {name} — {corr} correct — {total} pts")
    await _tg_call("recap.send", context.bot.send_message, chat_id=st.chat_id, text="\\n".join(lines), parse_mode=ParseMode.HTML)

async def close_question(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]; st = GAMES.get(chat_id)
    if not st: return
    st.locked = True
    await _tg_call("close.freeze", context.bot.edit_message_text, chat_id=chat_id, message_id=st.q_msg_id,
                   text=fmt_question(st.q_index+1, st.limit, st.questions[st.q_index], 0, len(st.per_q_answers.get(st.q_index,{}))),
                   reply_markup=None, parse_mode=ParseMode.HTML)
    await post_round_recap(context, st, st.q_index)
    st.per_q_answers.pop(st.q_index, None); st.answered_now.pop(st.q_index, None)

    gap_end = time.time() + DELAY_NEXT
    m = await _tg_call("gap.send", context.bot.send_message, chat_id=chat_id,
                       text=f"⏭️ Next question is coming in {int(math.ceil(DELAY_NEXT))}s…")
    if m and getattr(m,"message_id",None):
        context.job_queue.run_repeating(gap_tick, interval=max(1.0,TICK_SECONDS), first=max(1.0,TICK_SECONDS),
                                        data={"chat_id":chat_id,"msg_id":m.message_id,"end_ts":gap_end}, name=f"gap:{chat_id}:{st.q_index}")
    context.job_queue.run_once(next_question, when=DELAY_NEXT, data={"chat_id": chat_id})

async def next_question(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]; st = GAMES.get(chat_id)
    if not st: return
    st.q_index += 1
    if st.q_index >= st.limit: await finish_quiz(context, st); return
    await ask_question(context, st)

async def ask_question(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    q = st.questions[st.q_index]
    m = await _tg_call("ask.send", context.bot.send_message, chat_id=st.chat_id,
                       text=fmt_question(st.q_index+1, st.limit, q, QUESTION_TIME, 0),
                       reply_markup=answer_kb(q, st.q_index+1), parse_mode=ParseMode.HTML)
    if not m:
        log.error("ask_question: failed to send; ending session"); await finish_quiz(context, st); return
    st.q_msg_id = m.message_id; st.q_start_ts = time.time(); st.locked=False; st.answered_now[st.q_index]=set()
    end_ts = st.q_start_ts + QUESTION_TIME
    context.job_queue.run_repeating(tick_edit, interval=TICK_SECONDS, first=TICK_SECONDS,
                                    data={"chat_id":st.chat_id,"msg_id":st.q_msg_id,"end_ts":end_ts,"qidx":st.q_index},
                                    name=f"tick:{st.chat_id}:{st.q_index}")
    context.job_queue.run_once(close_question, when=QUESTION_TIME, data={"chat_id":st.chat_id}, name=f"close:{st.chat_id}:{st.q_index}")

# ---------- Admin helper ----------
async def is_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
    try: m = await context.bot.get_chat_member(chat_id, user_id); return m.status in ("administrator","creator")
    except Exception: return False

# ---------- Commands ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = SETTINGS.get(update.effective_chat.id, {}); mode = cfg.get("mode"); length = cfg.get("length")
    await update.message.reply_text(
        "✨ <b>Quiz Bot</b>\\n────────────\\n"
        "Step 1: /menu — choose <b>Mode</b>\\n"
        "Step 2: choose <b>How many questions</b>\\n"
        "Then: /startquiz — start (admin-only in groups)\\n"
        "Or tap ▶️ Start in the menu after choosing length.\\n\\n"
        "During play: recap + leaderboard each round.\\n"
        "Tools: /leaderboard • /answer • /stop • /reset\\n\\n"
        f"Current: Mode={esc(mode) if mode else '—'} • Length={esc(length) if length else '—'}",
        parse_mode=ParseMode.HTML)

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, update.effective_chat.id, update.effective_user.id):
            await update.message.reply_text("Only group admins can configure the quiz here."); return
    rows = [[InlineKeyboardButton("Beginner", callback_data="cfg:mode:beginner"),
             InlineKeyboardButton("Standard", callback_data="cfg:mode:standard"),
             InlineKeyboardButton("Expert",   callback_data="cfg:mode:expert")]]
    await update.message.reply_text("<b>Step 1/2</b>: Choose a <b>Mode</b>.",
                                    reply_markup=InlineKeyboardMarkup(rows), parse_mode=ParseMode.HTML)

async def _send_length_menu(update_or_query, context):
    rows = [[InlineKeyboardButton("10", callback_data="cfg:len:10"),
             InlineKeyboardButton("20", callback_data="cfg:len:20"),
             InlineKeyboardButton("30", callback_data="cfg:len:30")],
            [InlineKeyboardButton("40", callback_data="cfg:len:40"),
             InlineKeyboardButton("50", callback_data="cfg:len:50")]]
    chat_id = update_or_query.effective_chat.id if isinstance(update_or_query, Update) else update_or_query.message.chat.id
    await _tg_call("length.send", context.bot.send_message, chat_id=chat_id,
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
        await q.edit_message_text(f"Length set to <b>{length}</b> ✅\\nUse /startquiz to begin, or tap ▶️ Start.", parse_mode=ParseMode.HTML)
        if kb:
            await _tg_call("start.button", context.bot.send_message, chat_id=chat_id,
                           text=f"Ready to start <b>{esc(str(mode).title())}</b> • {length} questions?",
                           reply_markup=kb, parse_mode=ParseMode.HTML)

# Start via callback carrying mode+length
async def start_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    try:
        _, mode, length_s = q.data.split(":")
        length = int(length_s)
    except Exception:
        return
    chat_id = q.message.chat.id
    if q.message.chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, chat_id, q.from_user.id):
            try: await q.answer("Only group admins can start the quiz here.", show_alert=True)
            except Exception: pass
            return
    await do_startquiz(context, chat_id, q.from_user.id, mode, length)

# Allow /startquiz beginner 10 (args)
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
        await update.message.reply_text("Please run /menu and complete both steps first, or use:\\n/startquiz <mode> <length>\\nExample: /startquiz beginner 10"); 
        return
    await do_startquiz(context, chat_id, user.id, mode, int(length))

async def do_startquiz(context: ContextTypes.DEFAULT_TYPE, chat_id: int, starter_user_id: int, mode: str, length: int):
    pool = filter_by_mode(load_questions(), str(mode))
    if len(pool) < int(length):
        await _tg_call("start.notenough", context.bot.send_message, chat_id=chat_id,
                       text=f"Not enough questions in <b>{esc(mode)}</b> (need {length}).", parse_mode=ParseMode.HTML); 
        return
    qs = shuffle_qs(pool)[: int(length)]
    st = GameState(chat_id=chat_id, started_by=starter_user_id, questions=qs, limit=int(length), mode=str(mode))
    GAMES[chat_id]=st; LAST.pop(chat_id, None)
    await _tg_call("intro.send", context.bot.send_message, chat_id=chat_id,
                   text=(f"🎯 <b>{esc(str(mode).title())}</b> mode • {length} questions\\n"
                         f"⏱ {QUESTION_TIME}s per question • Next question in {int(math.ceil(DELAY_NEXT))}s after each."),
                   parse_mode=ParseMode.HTML)
    await ask_question(context, st)

# ---------- Answer / Results ----------
async def on_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q=update.callback_query; user=q.from_user; chat_id=q.message.chat.id
    try: _, qnum_str, opt_str = q.data.split(":"); qnum=int(qnum_str); opt=int(opt_str)
    except Exception: return
    st=GAMES.get(chat_id)
    if not st: 
        try: await q.answer("No active quiz here.", show_alert=False)
        except: pass
        return
    if st.q_index+1!=qnum:
        try: await q.answer("This question is already closed.", show_alert=False)
        except: pass
        return
    if st.locked or st.q_start_ts is None:
        try: await q.answer("Time is up!", show_alert=False)
        except: pass
        return
    st.answered_now.setdefault(st.q_index,set())
    if user.id in st.answered_now[st.q_index]:
        try: await q.answer("Only your first answer counts.", show_alert=False)
        except: pass
        return
    st.answered_now[st.q_index].add(user.id)
    st.players[user.id] = (user.full_name or user.username or str(user.id))[:64]
    st.per_q_answers.setdefault(st.q_index,{})    
    if user.id in st.per_q_answers[st.q_index]:
        try: await q.answer("Only your first answer counts.", show_alert=False)
        except: pass
        return
    elapsed=max(0.0, time.time()-st.q_start_ts); is_correct=(opt==st.questions[st.q_index].correct)
    pts=points(elapsed) if is_correct else 0
    st.per_q_answers[st.q_index][user.id]=AnswerRec(opt,is_correct,elapsed,pts)
    if is_correct: st.corrects[user.id]=st.corrects.get(user.id,0)+1
    st.totals[user.id]=st.totals.get(user.id,0)+pts
    try: await q.answer(f"You chose:\\n\\n{st.questions[st.q_index].options[opt]}\\n\\n"+("✅ Correct" if is_correct else "❌ Locked in")+f" • {pts} pts", show_alert=True)
    except: pass

async def cmd_leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id=update.effective_chat.id; st=GAMES.get(chat_id)
    if st:
        scores,corrects=compute_totals(st); source=f"Current session — {st.q_index+1}/{st.limit} asked"; name_of=st.players
    else:
        snap=LAST.get(chat_id)
        if not snap: await update.message.reply_text("No session found here yet."); return
        scores=snap["scores"]; corrects=snap["corrects"]; source=f"Last finished — {snap['limit']} questions"; name_of=snap["players"]
    if not scores: await update.message.reply_text("No participants yet."); return
    ranking=sorted(scores.items(), key=lambda x:x[1], reverse=True)
    lines=[f"🏁 <b>Leaderboard</b> ({esc(source)})"]
    for rank,(uid,pts_) in enumerate(ranking[:10],start=1):
        name=esc(name_of.get(uid,str(uid))); corr=corrects.get(uid,0); medal=("🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}.")
        lines.append(f"{medal} {name} — {corr} correct — {pts_} pts")
    lines.append("\\nGG! Thanks for participating 🎉")
    await _tg_call("leaderboard.send", context.bot.send_message, chat_id=chat_id, text="\\n".join(lines), parse_mode=ParseMode.HTML)

async def cmd_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id=update.effective_chat.id; snap=LAST.get(chat_id)
    if not snap: await update.message.reply_text("No finished quiz found here yet."); return
    qs: List[QItem]=snap["questions"]
    lines=["📘 <b>All Correct Answers</b>"]
    for i,q in enumerate(qs, start=1):
        lines.append(f"Q{i}: <b>{esc(q.options[q.correct])}</b>")
        if len("\\n".join(lines))>3500:
            await _tg_call("answer.chunk", context.bot.send_message, chat_id=chat_id, text="\\n".join(lines), parse_mode=ParseMode.HTML); lines=[]
    if lines: await _tg_call("answer.final", context.bot.send_message, chat_id=chat_id, text="\\n".join(lines), parse_mode=ParseMode.HTML)

async def finish_quiz(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    cancel_jobs_for_chat(context, st.chat_id)
    scores,corrects=compute_totals(st)
    if not scores:
        await _tg_call("finish.empty", context.bot.send_message, chat_id=st.chat_id, text="Quiz ended. No participants 😅")
        GAMES.pop(st.chat_id, None); return
    ranking=sorted(scores.items(), key=lambda x:x[1], reverse=True)
    top_uid, top_pts = ranking[0]; top_name=esc(st.players.get(top_uid,str(top_uid)))
    await _tg_call("finish.win", context.bot.send_message, chat_id=st.chat_id, text=f"🎉 Congrats, <b>{top_name}</b>! You topped the quiz with <b>{top_pts} pts</b>!", parse_mode=ParseMode.HTML)
    lines=["🏁 <b>Final Results</b> — Top 10"]
    for rank,(uid,pts_) in enumerate(ranking[:10], start=1):
        name=esc(st.players.get(uid,str(uid))); corr=corrects.get(uid,0); medal=("🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}.")
        lines.append(f"{medal} {name} — {corr}/{st.limit} correct — {pts_} pts")
    lines.append("\\nUse /leaderboard anytime. Use /answer for all correct answers. GG! 🎉")
    await _tg_call("finish.board", context.bot.send_message, chat_id=st.chat_id, text="\\n".join(lines), parse_mode=ParseMode.HTML)
    LAST[st.chat_id]={"questions":st.questions,"limit":st.limit,"scores":scores,"corrects":corrects,"players":st.players,"mode":st.mode}
    GAMES.pop(st.chat_id, None)

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user=update.effective_user; here_id=update.effective_chat.id
    if update.effective_chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, here_id, user.id): await update.message.reply_text("Only group admins can stop the quiz here."); return
    st=GAMES.get(here_id)
    if not st:
        for g in list(GAMES.values()):
            try:
                member=await context.bot.get_chat_member(g.chat_id, user.id)
                if member.status in ("administrator","creator"): st=g; break
            except Exception: continue
        if not st: await update.message.reply_text("No active quiz to stop."); return
    st.limit=st.q_index+1; st.locked=True
    await update.message.reply_text("Stopping quiz…"); cancel_jobs_for_chat(context, st.chat_id); await finish_quiz(context, st)

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id=update.effective_chat.id
    if update.effective_chat.type != "private" and ADMINS_ONLY:
        if not await is_admin(context, chat_id, update.effective_user.id): await update.message.reply_text("Only group admins can reset this chat."); return
    GAMES.pop(chat_id, None); LAST.pop(chat_id, None); SETTINGS.pop(chat_id, None); cancel_jobs_for_chat(context, chat_id)
    await update.message.reply_text("✅ Reset complete. Use /menu to choose Mode, then Length, then /startquiz.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Commands:\\n/start • /help • /menu • /startquiz\\n/leaderboard • /answer • /stop • /reset\\n/ping", parse_mode=ParseMode.HTML)

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"pong from {INSTANCE_ID}")

# ---------- Keepalive (for Web services) ----------
def start_keepalive_server():
    port_env = os.getenv("PORT")
    if not port_env: log.info("No $PORT detected — assuming Worker service."); return
    try: port=int(port_env)
    except: log.warning("Invalid PORT=%r; skipping keep-alive", port_env); return
    class H(BaseHTTPRequestHandler):
        def do_GET(self): self.send_response(200); self.send_header("Content-Type","text/plain; charset=utf-8"); self.end_headers(); self.wfile.write(b"ok")
        def log_message(self, *a, **k): return
    def run(): HTTPServer(("0.0.0.0", port), H).serve_forever()
    threading.Thread(target=run, daemon=True).start()

def build_app() -> Application:
    token = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
    if not token: raise RuntimeError("Set BOT_TOKEN (or TELEGRAM_BOT_TOKEN / TELEGRAM_TOKEN) env var.")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("menu",  cmd_menu))
    app.add_handler(CallbackQueryHandler(cfg_click, pattern=r"^cfg:"))
    app.add_handler(CallbackQueryHandler(start_click, pattern=r"^start:(beginner|standard|expert):\d+$"))
    app.add_handler(CommandHandler("startquiz",   cmd_startquiz))
    app.add_handler(CommandHandler("leaderboard", cmd_leaderboard))
    app.add_handler(CommandHandler("answer",      cmd_answer))
    app.add_handler(CommandHandler("stopquiz",    cmd_stop))
    app.add_handler(CommandHandler("stop",        cmd_stop))
    app.add_handler(CommandHandler("reset",       cmd_reset))
    app.add_handler(CommandHandler("ping",        cmd_ping))
    app.add_handler(CallbackQueryHandler(on_answer, pattern=r"^ans:\d+:\d$"))
    return app

if __name__ == "__main__":
    log.info("Starting quiz bot…")
    start_keepalive_server()
    build_app().run_polling(close_loop=False, drop_pending_updates=True)
