import os
import json
import asyncio
import logging
import sqlite3
import random
import requests
from datetime import datetime
from telegram import Update, Poll, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    PollAnswerHandler,
    ChatMemberHandler,
    ContextTypes,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ==========================================
# ⚙️ CONFIGURATION & CREDENTIALS
# ==========================================
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
OWNER_ID = int(os.getenv("OWNER_ID", "123456789"))  # Replace with your Telegram ID

# Force Join & Update Channels
UPDATE_CHANNEL = os.getenv("UPDATE_CHANNEL", "@rajasthani_mannu")
UPDATE_GROUP = os.getenv("UPDATE_GROUP", "@NEET_NURSING_CNET_GROUP")

# AI API Keys (Groq / Gemini)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

DB_NAME = "quiz_sync.db"

# Question subjects pool for AI prompt diversity
AI_SUBJECTS = [
    "Human Physiology and Anatomy",
    "Cell Biology and Genetics",
    "General Chemistry & Organic Reactions",
    "Physics: Optics, Mechanics and Electromagnetism",
    "Medical Laboratory & Diagnostic Microbiology",
    "General Knowledge & Indian History",
    "Current Affairs & General Science",
    "Biochemistry: Enzymes and Vitamins",
]

# In-memory storage for real-time poll interactions
# poll_records[poll_id] = { "question", "options", "correct_id", "start_time", "chat_id", "answers": { user_id: {name, option, correct, time} } }
poll_records = {}

# ==========================================
# 💾 DATABASE LAYER (Groups & Stats)
# ==========================================
def db_connect():
    return sqlite3.connect(DB_NAME, timeout=30)

def init_db():
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS groups (
            group_id BIGINT PRIMARY KEY,
            title TEXT,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS verified_users (
            user_id BIGINT PRIMARY KEY
        )
    """)
    conn.commit()
    conn.close()

init_db()

def track_group(group_id: int, title: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO groups (group_id, title) VALUES (?, ?)", (group_id, title[:100]))
    conn.commit()
    conn.close()

def untrack_group(group_id: int):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM groups WHERE group_id = ?", (group_id,))
    conn.commit()
    conn.close()

def get_all_groups() -> list:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT group_id FROM groups")
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

# ==========================================
# 🤖 AI QUESTION GENERATOR (Groq / Gemini)
# ==========================================
def generate_ai_question_sync() -> dict:
    subject = random.choice(AI_SUBJECTS)
    prompt = (
        f"Generate 1 multiple choice question on the subject: '{subject}'. "
        "Language: English (or bilingual Hindi+English if helpful). "
        "Provide exactly 4 options and mark the correct option index (0 to 3). "
        "Provide a short explanation under 150 characters. "
        "Return ONLY a raw JSON object with keys: "
        "'question' (string), 'options' (array of 4 strings), 'correct_index' (int 0-3), 'explanation' (string). "
        "No markdown blocks, no backticks, no comments."
    )

    # 1. Try Groq API (super fast)
    if GROQ_API_KEY:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                data = json.loads(raw)
                if isinstance(data, dict) and len(data.get("options", [])) == 4:
                    return data
        except Exception as e:
            logger.warning(f"Groq API generation failed: {e}")

    # 2. Try Gemini API
    if GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=15,
            )
            if resp.status_code == 200:
                raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                data = json.loads(raw)
                if isinstance(data, dict) and len(data.get("options", [])) == 4:
                    return data
        except Exception as e:
            logger.warning(f"Gemini API generation failed: {e}")

    # Fallback default question if AI key is missing or network rate-limited
    return {
        "question": f"Which parameter is vital in {subject}?",
        "options": ["Option A (Normal)", "Option B (Standard)", "Option C (Critical)", "Option D (Control)"],
        "correct_index": 1,
        "explanation": "Standard physiological parameter.",
    }

# ==========================================
# 🔐 FORCE JOIN & VERIFICATION
# ==========================================
async def check_user_membership(bot, user_id: int) -> bool:
    if user_id == OWNER_ID:
        return True
    try:
        member_ch = await bot.get_chat_member(chat_id=UPDATE_CHANNEL, user_id=user_id)
        member_gr = await bot.get_chat_member(chat_id=UPDATE_GROUP, user_id=user_id)
        valid = ("creator", "administrator", "member")
        return member_ch.status in valid and member_gr.status in valid
    except Exception as e:
        logger.warning(f"Verification check error: {e}")
        return False

def get_verify_keyboard():
    ch = UPDATE_CHANNEL.replace("@", "")
    gr = UPDATE_GROUP.replace("@", "")
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📢 Join Update Channel", url=f"https://t.me/{ch}")],
        [InlineKeyboardButton("💬 Join Support Group", url=f"https://t.me/{gr}")],
        [InlineKeyboardButton("✅ I Am Verified", callback_data="verify_check")],
    ])

# ==========================================
# 🚀 COMMAND HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user

    # If added or started in a group
    if chat.type in ("group", "supergroup"):
        track_group(chat.id, chat.title or "Group")
        await update.message.reply_text(
            f"⚡ **Group Synced!**\nIs group me har 1 minute me automatic AI Live Quiz Poll aayega.",
            parse_mode="Markdown",
        )
        return

    # Private DM Verification
    if not await check_user_membership(context.bot, user.id):
        await update.message.reply_text(
            f"👋 Namaste **{user.first_name}**!\n\nBot use karne ke liye channel aur group join karke verify karein:",
            reply_markup=get_verify_keyboard(),
            parse_mode="Markdown",
        )
        return

    ch = UPDATE_CHANNEL.replace("@", "")
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📢 Official Channel", url=f"https://t.me/{ch}")],
    ])
    await update.message.reply_text(
        f"✨ **AI Auto Quiz Sync Bot** ✨\n\n"
        f"• Status: 🟢 Live\n"
        f"• Auto-Feed: Har 1 minute me new AI question\n"
        f"• Group Count: `{len(get_all_groups())}` groups synced\n\n"
        f"Apne group me bot ko **Admin** banayein aur `/sync` likhein.",
        reply_markup=keyboard,
        parse_mode="Markdown",
    )

async def sync_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if chat.type in ("group", "supergroup"):
        track_group(chat.id, chat.title or "Group")
        await update.message.reply_text(
            "✅ **Group Linked & Synced!** Ab har 1 minute me AI poll automatic aayega.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text("Ye command group me execute karein.")

async def verify_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = update.effective_user

    if await check_user_membership(context.bot, user.id):
        await query.edit_message_text(
            "✅ **Verification Successful!**\nAb aap bot ko apne kisi bhi group me add karke `/sync` chala sakte hain.",
            parse_mode="Markdown",
        )
    else:
        await query.answer("⚠️ Pehle dono links se Channel aur Group join karein!", show_alert=True)

# Auto track group status when bot is added/promoted
async def chat_member_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if not chat or chat.type not in ("group", "supergroup"):
        return
    status = update.my_chat_member.new_chat_member.status
    if status in ("administrator", "member"):
        track_group(chat.id, chat.title or "Group")
    elif status in ("left", "kicked"):
        untrack_group(chat.id)

# ==========================================
# 📊 LEADERBOARD & GRAPH (VIEW RESULT)
# ==========================================
def render_percentage_bar(percentage: float, total_blocks: int = 10) -> str:
    filled = int(round((percentage / 100) * total_blocks))
    return "▰" * filled + "▱" * (total_blocks - filled)

async def leaderboard_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    poll_id = query.data.replace("poll_res_", "")
    record = poll_records.get(poll_id)

    if not record:
        await query.answer("Is poll ka record expire ho chuka hai.", show_alert=True)
        return

    answers = record["answers"]
    total_answers = len(answers)
    if total_answers == 0:
        await query.answer("Abhi tak kisi member ne answer nahi diya!", show_alert=True)
        return

    options = record["options"]
    correct_id = record["correct_id"]

    # Option counts for % and graph
    counts = [0] * len(options)
    for ans in answers.values():
        if 0 <= ans["option"] < len(options):
            counts[ans["option"]] += 1

    lines = [
        f"📊 **Poll Result Breakdown & Graph**",
        f"❓ **Q:** {record['question'][:120]}",
        f"👥 Total Participants: `{total_answers}`\n",
    ]

    for idx, opt in enumerate(options):
        cnt = counts[idx]
        pct = round((cnt / total_answers) * 100, 1)
        bar = render_percentage_bar(pct, total_blocks=8)
        mark = "✅" if idx == correct_id else "▫️"
        lines.append(f"{mark} **{opt[:20]}**: `{cnt}` ({pct}%)\n   `{bar}`")

    # Member-wise leaderboard with timing
    sorted_players = sorted(
        answers.values(),
        key=lambda x: (not x["correct"], x["time"]),
    )

    lines.append("\n🏆 **Live Member Leaderboard & Timing:**")
    for rank, p in enumerate(sorted_players[:20], start=1):
        status = "✅" if p["correct"] else "❌"
        lines.append(f"{rank}. {status} **{p['name']}** — `{p['time']}s`")

    await query.answer()
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text="\n".join(lines),
        parse_mode="Markdown",
    )

# ==========================================
# 🗳️ POLL ANSWER REAL-TIME TRACKING
# ==========================================
async def poll_answer_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ans = update.poll_answer
    poll_id = ans.poll_id
    if poll_id not in poll_records:
        return

    record = poll_records[poll_id]
    if not ans.option_ids:
        return

    selected_option = ans.option_ids[0]
    is_correct = (selected_option == record["correct_id"])
    elapsed_time = round(datetime.utcnow().timestamp() - record["start_time"], 2)

    user = ans.user
    user_name = user.first_name or "Player"

    record["answers"][user.id] = {
        "name": user_name,
        "option": selected_option,
        "correct": is_correct,
        "time": elapsed_time,
    }

# ==========================================
# ⏱️ 1-MINUTE AUTO FEED (AI QUESTIONS)
# ==========================================
async def auto_feed_job(context: ContextTypes.DEFAULT_TYPE):
    groups = get_all_groups()
    if not groups:
        return

    # Generate fresh AI question off the main loop
    q_data = await asyncio.to_thread(generate_ai_question_sync)

    question_text = f"❓ {q_data['question']}"[:300]
    options = [o[:100] for o in q_data["options"]][:4]
    correct_id = q_data.get("correct_index", 0)
    explanation = (q_data.get("explanation") or "")[:200]

    for gid in groups:
        try:
            # 1. Native Quiz Poll
            msg = await context.bot.send_poll(
                chat_id=gid,
                question=question_text,
                options=options,
                type=Poll.QUIZ,
                correct_option_id=correct_id,
                explanation=explanation if explanation else None,
                is_anonymous=False,
                open_period=55,  # 55 sec open time for 1-minute interval
            )

            poll_id = msg.poll.id
            poll_records[poll_id] = {
                "question": q_data["question"],
                "options": options,
                "correct_id": correct_id,
                "start_time": datetime.utcnow().timestamp(),
                "chat_id": gid,
                "answers": {},
            }

            # 2. View Result & Leaderboard Button
            btn = InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 View Result & Leaderboard", callback_data=f"poll_res_{poll_id}")]
            ])
            await context.bot.send_message(
                chat_id=gid,
                text="👆 Tap below to see live timing, percentage graph & user answers for this poll:",
                reply_markup=btn,
            )
            await asyncio.sleep(0.05)
        except Exception as e:
            logger.warning(f"Auto-feed poll failed for group {gid}: {e}")

# ==========================================
# 📢 OWNER BROADCAST
# ==========================================
async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Sirf Owner hi broadcast kar sakte hain.")
        return

    if not context.args:
        await update.message.reply_text("Usage: `/broadcast <message>`", parse_mode="Markdown")
        return

    msg = " ".join(context.args)
    groups = get_all_groups()
    delivered = 0

    for gid in groups:
        try:
            await context.bot.send_message(chat_id=gid, text=msg, parse_mode="Markdown")
            delivered += 1
            await asyncio.sleep(0.05)
        except Exception:
            pass

    await update.message.reply_text(f"📢 Broadcast {delivered}/{len(groups)} groups me send ho gaya.")

# ==========================================
# 🚀 BOT INITIALIZATION & RUNNER
# ==========================================
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    # Commands & Callbacks
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("sync", sync_command))
    app.add_handler(CommandHandler("broadcast", broadcast_command))
    app.add_handler(CallbackQueryHandler(verify_callback, pattern="^verify_check$"))
    app.add_handler(CallbackQueryHandler(leaderboard_callback, pattern="^poll_res_"))
    app.add_handler(PollAnswerHandler(poll_answer_handler))
    app.add_handler(ChatMemberHandler(chat_member_update, ChatMemberHandler.MY_CHAT_MEMBER))

    # Job Queue: Runs auto_feed_job every 60 seconds (1 minute)
    if app.job_queue:
        app.job_queue.run_repeating(auto_feed_job, interval=60, first=5)

    logger.info("Bot starting...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
