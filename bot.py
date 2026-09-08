import os
import re
import io
import json
import base64
import logging
import sqlite3
import asyncio
import time
import random
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Poll,
)
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ChatMemberHandler,
    ConversationHandler,
    ContextTypes,
    ApplicationHandlerStop,
    filters,
)

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# 🔐 token .env / environment variable se aata hai, code me kabhi mat likhna.
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise SystemExit(
        "BOT_TOKEN environment variable is not set.\n"
        "Copy .env.example to .env, fill in a NEW token from @BotFather, then run again."
    )

OWNER_ID = int(os.getenv("OWNER_ID", "123456789"))
DB_NAME = os.getenv("DB_NAME", "quiz_bot.db")

# 🐘 Optional Postgres support — set DATABASE_URL and the bot stores everything there
# instead of the local SQLite file. Leave it unset (e.g. running in Pydroid 3) and it
# keeps using SQLite exactly like before.
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USE_POSTGRES = bool(DATABASE_URL) and psycopg2 is not None
if bool(DATABASE_URL) and psycopg2 is None:
    raise SystemExit(
        "DATABASE_URL is set but 'psycopg2-binary' isn't installed.\n"
        "Run: pip install psycopg2-binary"
    )
PK_TYPE = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"

# Update channel & group used for the join-verification gate.
UPDATE_CHANNEL = os.getenv("UPDATE_CHANNEL", "@Telegram")
UPDATE_GROUP = os.getenv("UPDATE_GROUP", "@Telegram")

# AI generation — these are the bot OWNER's fallback keys, used automatically for
# every user (there is no per-user /addapi anymore — this bot only uses these).
# Set ANY of these env vars you have; every one becomes an extra fallback, tried in
# OWNER_FALLBACK_ORDER below. Multiple keys for the same provider? Comma-separate
# them in the same env var (e.g. GROQ_API_KEY=key1,key2) — each becomes its own
# fallback entry with its own daily-usage tracking.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "claude-sonnet-5")


def _split_keys(raw: str) -> list:
    return [k.strip() for k in raw.split(",") if k.strip()]


OWNER_PROVIDER_KEYS = {
    "cerebras": {"api_keys": _split_keys(os.getenv("CEREBRAS_API_KEY", ""))},
    "groq": {"api_keys": _split_keys(os.getenv("GROQ_API_KEY", ""))},
    "gemini": {"api_keys": _split_keys(os.getenv("GEMINI_API_KEY", ""))},
    "cloudflare_text": {
        "api_keys": _split_keys(os.getenv("CLOUDFLARE_API_KEY", "")),
        "account_ids": _split_keys(os.getenv("CLOUDFLARE_ACCOUNT_ID", "")),
    },
    "cloudflare_vision": {
        "api_keys": _split_keys(os.getenv("CLOUDFLARE_API_KEY", "")),
        "account_ids": _split_keys(os.getenv("CLOUDFLARE_ACCOUNT_ID", "")),
    },
    "github": {"api_keys": _split_keys(os.getenv("GITHUB_API_KEY", ""))},
    "openrouter_text": {"api_keys": _split_keys(os.getenv("OPENROUTER_API_KEY", ""))},
    "openrouter_vision": {"api_keys": _split_keys(os.getenv("OPENROUTER_API_KEY", ""))},
    "mistral": {"api_keys": _split_keys(os.getenv("MISTRAL_API_KEY", ""))},
    "huggingface": {"api_keys": _split_keys(os.getenv("HUGGINGFACE_API_KEY", ""))},
    "openai": {"api_keys": _split_keys(os.getenv("OPENAI_API_KEY", ""))},
    "anthropic": {"api_keys": _split_keys(ANTHROPIC_API_KEY)},
}
OWNER_FALLBACK_ORDER = [
    "cerebras", "groq", "gemini", "cloudflare_text", "cloudflare_vision",
    "github", "openrouter_text", "openrouter_vision", "mistral", "huggingface", "openai", "anthropic",
]

# ==========================================
# 🔌 MULTI-PROVIDER AI CONFIG
# "style" decides which HTTP call shape is used:
#   - "anthropic": Anthropic Messages API (native image/PDF support)
#   - "openai":    OpenAI-compatible /chat/completions (Groq, OpenAI, and any
#                  open-source/self-hosted server)
#   - "gemini":    Google Generative Language API
# ==========================================
PROVIDER_PRESETS = {
    "groq": {"label": "🟢 Groq", "style": "openai", "api_base": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "gemini": {"label": "🔵 Google Gemini", "style": "gemini", "api_base": "https://generativelanguage.googleapis.com/v1beta", "model": "gemini-2.5-flash"},
    "openai": {"label": "🟣 OpenAI", "style": "openai", "api_base": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "anthropic": {"label": "🟠 Anthropic (Claude)", "style": "anthropic", "api_base": None, "model": "claude-sonnet-5"},
    "github": {"label": "🐙 GitHub Models", "style": "openai", "api_base": "https://models.inference.ai.azure.com", "model": "gpt-4o-mini"},
    "openrouter_vision": {"label": "🌐 OpenRouter (Vision)", "style": "openai", "api_base": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.2-11b-vision-instruct:free"},
    "openrouter_text": {"label": "🌐 OpenRouter (Text/PDF)", "style": "openai", "api_base": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.3-70b-instruct:free"},
    "cloudflare_vision": {"label": "☁️ Cloudflare (Vision)", "style": "openai", "api_base": "", "model": "@cf/meta/llama-3.2-11b-vision-instruct", "needs_base": True},
    "cloudflare_text": {"label": "☁️ Cloudflare (Fast Text)", "style": "openai", "api_base": "", "model": "@cf/meta/llama-3.1-8b-instruct", "needs_base": True},
    "mistral": {"label": "🌊 Mistral (Pixtral OCR)", "style": "openai", "api_base": "https://api.mistral.ai/v1", "model": "pixtral-12b-2409"},
    "cerebras": {"label": "⚡ Cerebras (Ultra-fast)", "style": "openai", "api_base": "https://api.cerebras.ai/v1", "model": "llama-3.3-70b"},
    "huggingface": {"label": "🤗 Hugging Face (Vision)", "style": "openai", "api_base": "https://router.huggingface.co/v1", "model": "meta-llama/Llama-3.2-11B-Vision-Instruct"},
}

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Conversation states — only what the trimmed flow actually needs.
ASK_CONTENT, ASK_COUNT, ASK_VOICE = range(3)

# ==========================================
# 🎨 STYLISH FONT HELPER
# ==========================================
STYLE_MAP = {
    'a': '𝛂', 'b': 'в', 'c': 'ς', 'd': '∂', 'e': 'є', 'f': 'ƒ', 'g': 'g',
    'h': 'н', 'i': 'ι', 'j': 'ʝ', 'k': 'ĸ', 'l': 'ʟ', 'm': 'м', 'n': 'η',
    'o': 'ο', 'p': 'ρ', 'q': 'q', 'r': 'ɤ', 's': '𝛅', 't': 'τ', 'u': '𝛖',
    'v': 'ν', 'w': 'ω', 'x': 'χ', 'y': 'у', 'z': 'ᴢ',
}


def stylize(text: str) -> str:
    return "".join(STYLE_MAP.get(ch.lower(), ch) for ch in text)


FONT_STYLE = stylize("your stay & let's grow together")

# ==========================================
# 💾 DATABASE CONNECTION LAYER (SQLite or Postgres)
# ==========================================
class _PGCursorShim:
    def __init__(self, cursor):
        self._cursor = cursor
        self.lastrowid = None

    def execute(self, sql, params=()):
        self._cursor.execute(sql.replace("?", "%s"), params)
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _PGConnShim:
    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return _PGCursorShim(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()


def db_connect():
    if USE_POSTGRES:
        return _PGConnShim(psycopg2.connect(DATABASE_URL, sslmode="require"))
    conn = sqlite3.connect(DB_NAME, timeout=30, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
    except sqlite3.OperationalError:
        pass
    return conn


def last_insert_id(cursor, conn, table: str, pk: str = "id"):
    if USE_POSTGRES:
        cursor.execute(f"SELECT currval(pg_get_serial_sequence('{table}', '{pk}'))")
        return cursor.fetchone()[0]
    return cursor.lastrowid


def upsert(cursor, table: str, keys: dict, values: dict):
    all_cols = list(keys) + list(values)
    all_vals = list(keys.values()) + list(values.values())
    placeholders = ", ".join(["?"] * len(all_cols))
    if USE_POSTGRES:
        conflict_cols = ", ".join(keys)
        update_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in values) if values else conflict_cols
        sql = (f"INSERT INTO {table} ({', '.join(all_cols)}) VALUES ({placeholders}) "
               f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_clause}")
    else:
        sql = f"INSERT OR REPLACE INTO {table} ({', '.join(all_cols)}) VALUES ({placeholders})"
    cursor.execute(sql, all_vals)


# ==========================================
# 💾 DATABASE INITIALIZATION (minimal — only what this trimmed bot needs)
# ==========================================
def init_db():
    conn = db_connect()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id BIGINT PRIMARY KEY,
            questions_used INTEGER DEFAULT 0,
            question_limit INTEGER DEFAULT 100000,
            is_verified INTEGER DEFAULT 0,
            tag TEXT DEFAULT '',
            description_tag TEXT DEFAULT '',
            default_difficulty TEXT DEFAULT '',
            default_language TEXT DEFAULT '',
            daily_ai_used INTEGER DEFAULT 0,
            daily_ai_date TEXT DEFAULT '',
            ai_unlimited INTEGER DEFAULT 0,
            option_style TEXT DEFAULT 'plain'
        )
    """)

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS quizzes (
            id {PK_TYPE},
            user_id BIGINT,
            title TEXT,
            data TEXT,
            timer_seconds INTEGER DEFAULT 0,
            quiz_type TEXT DEFAULT 'ai_generated',
            creator_name TEXT DEFAULT ''
        )
    """)

    # Per-key daily usage tracking, so the owner's shared free-tier keys don't
    # silently run dry — see record_api_key_usage / is_key_exhausted_today below.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_key_usage (
            scope TEXT,
            usage_date TEXT,
            count INTEGER DEFAULT 0,
            last_notified_pct INTEGER DEFAULT 0,
            PRIMARY KEY (scope, usage_date)
        )
    """)

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS pending_owner_notifications (
            id {PK_TYPE},
            message TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized.")


# ==========================================
# 💾 SELF-BACKUP — Telegram itself as free, permanent off-site storage
# ==========================================
BACKUP_TABLES = ["users", "quizzes"]


def export_all_tables() -> dict:
    conn = db_connect()
    cur = conn.cursor()
    dump = {}
    for table in BACKUP_TABLES:
        cur.execute(f"SELECT * FROM {table}")
        cols = [d[0] for d in cur.description]
        dump[table] = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return dump


def build_backup_file() -> io.BytesIO:
    payload = {"exported_at": datetime.utcnow().isoformat(), "tables": export_all_tables()}
    buf = io.BytesIO(json.dumps(payload, indent=2, default=str).encode("utf-8"))
    buf.name = f"quizbot_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    return buf


async def send_db_backup(bot, note: str = ""):
    if not OWNER_ID:
        return
    buf = build_backup_file()
    try:
        await bot.send_document(
            chat_id=OWNER_ID, document=buf, filename=buf.name,
            caption=f"💾 Auto-backup{(' — ' + note) if note else ''}.",
        )
    except Exception as e:
        logger.warning(f"Auto-backup DM to owner failed: {e}")


async def periodic_backup_job(context: ContextTypes.DEFAULT_TYPE):
    await send_db_backup(context.bot)


async def report_generation_failure(bot, chat_id: int, user_id: int, error) -> None:
    """User-facing side stays generic on purpose. The owner still gets full detail."""
    await bot.send_message(chat_id, "⚠️ Quiz could not be created due to an error. Please try again in a bit.")
    if OWNER_ID:
        try:
            await bot.send_message(OWNER_ID, f"🔧 Quiz generation failed for user {user_id}:\n{error}")
        except Exception as e:
            logger.warning(f"Failed to DM owner about generation failure: {e}")


async def flush_owner_notifications_job(context: ContextTypes.DEFAULT_TYPE):
    if not OWNER_ID:
        return
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT id, message FROM pending_owner_notifications ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return
    for row_id, message in rows:
        try:
            await context.bot.send_message(OWNER_ID, message, parse_mode="Markdown")
        except Exception as e:
            logger.warning(f"Failed to deliver queued owner notification #{row_id}: {e}")
            continue
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM pending_owner_notifications WHERE id=?", (row_id,))
        conn.commit()
        conn.close()


# ==========================================
# 🔐 DATABASE HELPERS — users
# ==========================================
def get_user(user_id: int) -> dict:
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_id, questions_used, question_limit, is_verified, tag, "
        "description_tag, default_difficulty, default_language, daily_ai_used, "
        "daily_ai_date, ai_unlimited, option_style FROM users WHERE user_id = ?",
        (user_id,)
    )
    user = cursor.fetchone()

    if not user:
        cursor.execute(
            "INSERT INTO users (user_id, questions_used, question_limit, is_verified) VALUES (?, 0, 100000, 0)",
            (user_id,)
        )
        conn.commit()
        cursor.execute(
            "SELECT user_id, questions_used, question_limit, is_verified, tag, "
            "description_tag, default_difficulty, default_language, daily_ai_used, "
            "daily_ai_date, ai_unlimited, option_style FROM users WHERE user_id = ?",
            (user_id,)
        )
        user = cursor.fetchone()

    conn.close()
    return {
        "user_id": user[0], "questions_used": user[1], "question_limit": user[2],
        "is_verified": user[3], "tag": user[4] or '', "description_tag": user[5] or '',
        "default_difficulty": user[6] or '', "default_language": user[7] or '',
        "daily_ai_used": user[8] or 0, "daily_ai_date": user[9] or '',
        "ai_unlimited": bool(user[10]), "option_style": user[11] or 'plain',
    }


def set_verified(user_id: int, status: int = 1):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET is_verified = ? WHERE user_id = ?", (status, user_id))
    conn.commit()
    conn.close()


def increment_questions_used(user_id: int, n: int):
    if n <= 0:
        return
    get_user(user_id)
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET questions_used = questions_used + ? WHERE user_id = ?", (n, user_id))
    conn.commit()
    conn.close()


# ==========================================
# 📅 DAILY AI GENERATION QUOTA — protects the owner's free-tier fallback keys
# from being drained by a single user. Resets automatically at UTC midnight.
# ==========================================
DAILY_AI_QUESTION_LIMIT = 300


def get_daily_ai_remaining(user_id: int):
    if user_id == OWNER_ID:
        return None
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT daily_ai_used, daily_ai_date, ai_unlimited FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return DAILY_AI_QUESTION_LIMIT
    used, date_str, unlimited = row
    if unlimited:
        return None
    today = datetime.utcnow().strftime("%Y-%m-%d")
    used = used if date_str == today else 0
    return max(0, DAILY_AI_QUESTION_LIMIT - (used or 0))


def record_daily_ai_usage(user_id: int, n: int):
    if n <= 0 or user_id == OWNER_ID:
        return
    get_user(user_id)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT daily_ai_used, daily_ai_date FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    used = row[0] if row and row[1] == today else 0
    cur.execute("UPDATE users SET daily_ai_used=?, daily_ai_date=? WHERE user_id=?", ((used or 0) + n, today, user_id))
    conn.commit()
    conn.close()


def apply_option_style(options: list, style: str) -> list:
    style = style or 'plain'
    if style == 'abcd':
        return [f"{chr(65 + i)}) {o}"[:100] for i, o in enumerate(options)]
    if style == 'numeric':
        return [f"{i + 1}) {o}"[:100] for i, o in enumerate(options)]
    return [o[:100] for o in options]


def save_quiz(user_id: int, title: str, questions: list, timer_seconds: int = 0,
              quiz_type: str = "ai_generated", creator_name: str = "") -> int:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO quizzes (user_id, title, data, timer_seconds, quiz_type, creator_name) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, (title or "Quiz")[:60], json.dumps(questions), timer_seconds, quiz_type, creator_name[:60])
        )
        conn.commit()
        return last_insert_id(cur, conn, "quizzes")
    finally:
        conn.close()


# ==========================================
# 🔌 PROVIDER CHAIN — owner's shared fallback keys only (no per-user /addapi)
# ==========================================
def build_provider_chain(user_id: int) -> list:
    chain = []
    for provider in OWNER_FALLBACK_ORDER:
        owner_cfg = OWNER_PROVIDER_KEYS.get(provider, {})
        api_keys = owner_cfg.get("api_keys", [])
        if not api_keys:
            continue
        preset = PROVIDER_PRESETS.get(provider, {})
        account_ids = owner_cfg.get("account_ids", [])
        for idx, api_key in enumerate(api_keys):
            api_base = preset.get("api_base")
            if preset.get("needs_base"):
                if not account_ids:
                    continue
                account_id = account_ids[idx] if idx < len(account_ids) else account_ids[-1]
                api_base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
            model = (AI_MODEL if provider == "anthropic" and AI_MODEL else preset.get("model"))
            label = f"Owner's shared {preset.get('label', provider)}"
            if len(api_keys) > 1:
                label += f" #{idx + 1}"
            chain.append({
                "provider": provider, "api_key": api_key, "api_base": api_base, "model": model,
                "label": label, "key_id": None, "owner_key_index": idx, "owner_user_id": OWNER_ID,
            })
    return chain


DEFAULT_KEY_DAILY_LIMIT = int(os.getenv("KEY_DAILY_LIMIT", "1500"))
USAGE_ALERT_THRESHOLDS = [50, 70, 90, 100]
PROVIDER_DAILY_LIMIT_DEFAULTS = {
    "groq": 1000, "gemini": 1500, "openai": None, "anthropic": None, "github": 150,
    "openrouter_vision": 50, "openrouter_text": 50, "cloudflare_vision": 300,
    "cloudflare_text": 1000, "mistral": 500, "cerebras": 14000, "huggingface": 10,
}


def get_provider_daily_limit(provider: str):
    env_val = os.getenv(f"{provider.upper()}_DAILY_LIMIT")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    if provider in PROVIDER_DAILY_LIMIT_DEFAULTS:
        return PROVIDER_DAILY_LIMIT_DEFAULTS[provider]
    return DEFAULT_KEY_DAILY_LIMIT


def _usage_scope(candidate: dict) -> str:
    idx = candidate.get("owner_key_index", 0)
    return f"owner:{candidate['provider']}:{idx}"


def is_key_exhausted_today(candidate: dict) -> bool:
    limit = get_provider_daily_limit(candidate["provider"])
    if not limit:
        return False
    scope = _usage_scope(candidate)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT count FROM api_key_usage WHERE scope=? AND usage_date=?", (scope, today))
    row = cur.fetchone()
    conn.close()
    return bool(row and row[0] >= limit)


def record_api_key_usage(candidate: dict):
    limit = get_provider_daily_limit(candidate["provider"])
    scope = _usage_scope(candidate)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT count, last_notified_pct FROM api_key_usage WHERE scope=? AND usage_date=?", (scope, today))
    row = cur.fetchone()
    count = (row[0] if row else 0) + 1
    last_notified = row[1] if row else 0

    crossed = None
    if limit:
        pct = int(count * 100 / limit)
        for t in USAGE_ALERT_THRESHOLDS:
            if pct >= t and last_notified < t:
                crossed = t
        if crossed is not None:
            last_notified = crossed

    upsert(cur, "api_key_usage", {"scope": scope, "usage_date": today}, {"count": count, "last_notified_pct": last_notified})

    if crossed is not None:
        who = "Owner's shared key"
        if crossed >= 100:
            note = (f"🚨 **Daily limit reached**\n{who} — {candidate['label']} ({candidate['provider']})\n"
                    f"Used: {count}/{limit} calls today (100%+).")
        else:
            note = (f"📊 **Usage alert — {crossed}%**\n{who} — {candidate['label']} ({candidate['provider']})\n"
                    f"Used: {count}/{limit} calls today.")
        cur.execute("INSERT INTO pending_owner_notifications (message) VALUES (?)", (note,))

    conn.commit()
    conn.close()


# ==========================================
# 🎨 KEYBOARD BUILDERS
# ==========================================
def verification_keyboard():
    channel_url = f"https://t.me/{UPDATE_CHANNEL.replace('@', '')}"
    group_url = f"https://t.me/{UPDATE_GROUP.replace('@', '')}"
    keyboard = [
        [InlineKeyboardButton("📢 Join Update Channel", url=channel_url)],
        [InlineKeyboardButton("💬 Join Support Group", url=group_url)],
        [InlineKeyboardButton("✅ I Am Verified", callback_data="check_verification")],
    ]
    return InlineKeyboardMarkup(keyboard)


def main_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("🧠 Topic", callback_data="create_topic"),
         InlineKeyboardButton("📄 PDF", callback_data="create_pdf")],
        [InlineKeyboardButton("🖼 Image", callback_data="create_image"),
         InlineKeyboardButton("🎙️ Voice", callback_data="create_voice")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="open_help")],
    ]
    return InlineKeyboardMarkup(keyboard)


def cancel_keyboard(extra_rows=None):
    rows = list(extra_rows or [])
    rows.append([InlineKeyboardButton("❌ Cancel", callback_data="gcancel")])
    return InlineKeyboardMarkup(rows)


def build_welcome_text(user_name: str, user_data: dict, verified_banner: bool = False) -> str:
    if verified_banner:
        icon, title = "✅", stylize("Verification Successful") + "!"
    else:
        icon, title = "✨", stylize("Welcome") + f", {user_name}!"
    lines = [
        f"{icon} **{title}**",
        f"_{FONT_STYLE}_",
        "",
        "Pick how you want to build your quiz:",
        "🧠 Topic · 📄 PDF · 🖼 Image · 🎙️ Voice",
        "",
        "Language is auto-detected (Hindi/English), timer and shuffle are automatic —",
        "you'll just pick how many questions (25/50/75/100).",
    ]
    return "\n".join(lines)


# ==========================================
# 🤖 AI GENERATION CORE (unchanged — this is the part you asked to keep)
# ==========================================
class ProviderError(RuntimeError):
    def __init__(self, message: str, is_quota: bool = False, status_code: int = None):
        super().__init__(message)
        self.is_quota = is_quota
        self.status_code = status_code


_QUOTA_SIGNALS = (
    "rate limit", "rate_limit", "ratelimit", "429", "quota", "insufficient_quota",
    "resource_exhausted", "resource exhausted", "billing", "credit balance",
    "too many requests", "exceeded your current quota", "plan and billing",
)
_AUTH_SIGNALS = (
    "invalid api key", "invalid_api_key", "unauthorized", "401", "incorrect api key",
    "authentication", "permission_denied", "api key not valid",
)


def _looks_like_quota_error(text: str, exc: Exception = None, status_code: int = None) -> bool:
    if status_code == 429:
        return True
    if exc is not None and exc.__class__.__name__ == "RateLimitError":
        return True
    t = (text or "").lower()
    return any(sig in t for sig in _QUOTA_SIGNALS)


def _looks_like_auth_error(text: str, status_code: int = None) -> bool:
    if status_code in (401, 403):
        return True
    t = (text or "").lower()
    return any(sig in t for sig in _AUTH_SIGNALS)


def _friendly_provider_message(raw: str, is_quota: bool, is_auth: bool) -> str:
    if is_quota:
        return "iski API key ki limit/quota khatam ho gayi hai"
    if is_auth:
        return "iski API key invalid/expired hai"
    return raw[:200]


def _build_system_prompt(count: int, difficulty: str, language: str) -> str:
    if (language or "").strip().lower() == "auto":
        language_clause = (
            "LANGUAGE — read carefully: first detect whether the source material below is "
            "primarily in Hindi or in English. Then write EVERY piece of output — the "
            "question, all 4 options, AND the explanation, for EVERY item — entirely in "
            "that SAME language (Hindi or English, whichever the source is). Never mix "
            "languages within one item, and never leave anything partially translated."
        )
    else:
        language_clause = (
            f"LANGUAGE — read carefully: every single piece of text you output — the "
            f"question, all 4 options, AND the explanation, for EVERY item — must be written "
            f"ENTIRELY in {language}. This applies even if the source material below is in a "
            f"different language: translate the underlying facts/content into {language}, "
            f"don't just copy fragments from the source as-is. Never leave a question, option, "
            f"or explanation partially or fully in a different language, and never mix two "
            f"languages within the same item. Proper nouns (names, places) may stay as "
            f"commonly written, but every other word must be in {language}."
        )
    return (
        "You are an expert quiz writer. Read the supplied source material and write "
        f"exactly {count} multiple-choice questions at {difficulty} difficulty.\n\n"
        f"{language_clause}\n\n"
        "Respond with ONLY a JSON array — no markdown fences, no commentary before or after. "
        "Each array item must be an object with exactly these keys: "
        '"question" (string), "options" (array of exactly 4 short strings), '
        '"correct_index" (integer 0-3, the index of the correct option), '
        '"explanation" (short string under 150 characters explaining the answer).'
    )


def _clean_questions(raw_questions) -> list:
    cleaned = []
    for q in raw_questions:
        try:
            options = [str(o)[:100] for o in q["options"]][:4]
            if len(options) < 2:
                continue
            correct_index = int(q["correct_index"])
            if not (0 <= correct_index < len(options)):
                correct_index = 0
            cleaned.append({
                "question": str(q["question"])[:290],
                "options": options,
                "correct_index": correct_index,
                "explanation": str(q.get("explanation", ""))[:195],
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not cleaned:
        raise RuntimeError("No valid questions came back — try again or use a longer source.")
    return cleaned


def _parse_json_questions(raw_text: str) -> list:
    raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip())
    m = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if m:
        raw_text = m.group(0)
    try:
        raw_questions = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error(f"Bad AI JSON output: {raw_text[:500]}")
        raise RuntimeError("The AI response couldn't be parsed — please try again.")
    return _clean_questions(raw_questions)


def _extract_pdf_text(b64_data: str) -> str:
    if not PdfReader:
        raise RuntimeError("PDF text extraction isn't installed on the server (pip install pypdf).")
    try:
        raw = base64.b64decode(b64_data)
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        raise RuntimeError(f"Couldn't read that PDF: {e}")
    if not text.strip():
        raise RuntimeError("Couldn't extract any text from that PDF (likely scanned/image-only).")
    return text[:15000]


def _generate_via_anthropic(api_key, model, source_parts, count, difficulty, language) -> list:
    if not (Anthropic and api_key):
        raise ProviderError("No Anthropic key available for this step.", is_quota=False)
    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model or AI_MODEL, max_tokens=4096,
            system=_build_system_prompt(count, difficulty, language),
            messages=[{"role": "user", "content": source_parts}],
        )
    except Exception as e:
        status_code = getattr(e, "status_code", None)
        is_quota = _looks_like_quota_error(str(e), exc=e, status_code=status_code)
        is_auth = _looks_like_auth_error(str(e), status_code=status_code)
        logger.error(f"Anthropic call failed: {e}")
        raise ProviderError(_friendly_provider_message(str(e), is_quota, is_auth), is_quota=is_quota, status_code=status_code)
    raw_text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
    return _parse_json_questions(raw_text)


def _build_openai_style_content(source_parts: list) -> list:
    content = []
    for p in source_parts:
        t = p.get("type")
        if t == "text":
            content.append({"type": "text", "text": p["text"][:15000]})
        elif t == "image":
            media_type = p["source"]["media_type"]
            data = p["source"]["data"]
            content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}})
        elif t == "document":
            content.append({"type": "text", "text": _extract_pdf_text(p["source"]["data"])})
    if not content:
        raise RuntimeError("No readable content found in the source.")
    return content


def _generate_via_openai_compatible(api_key, api_base, model, source_parts, count, difficulty, language) -> list:
    if not api_base or not model:
        raise ProviderError("This provider isn't fully configured.", is_quota=False)
    content_blocks = _build_openai_style_content(source_parts)
    message_content = content_blocks[0]["text"] if (len(content_blocks) == 1 and content_blocks[0]["type"] == "text") else content_blocks

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _build_system_prompt(count, difficulty, language)},
            {"role": "user", "content": message_content},
        ],
        "temperature": 0.7, "max_tokens": 4096,
    }
    resp = None
    try:
        resp = requests.post(f"{api_base.rstrip('/')}/chat/completions", headers=headers, json=body, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        status_code = resp.status_code if resp is not None else None
        body_text = resp.text[:300] if resp is not None else str(e)
        is_quota = _looks_like_quota_error(body_text, status_code=status_code)
        is_auth = _looks_like_auth_error(body_text, status_code=status_code)
        logger.error(f"OpenAI-compatible call failed ({status_code}): {body_text}")
        raise ProviderError(_friendly_provider_message(body_text, is_quota, is_auth), is_quota=is_quota, status_code=status_code)
    return _parse_json_questions(raw_text)


def _build_gemini_parts(source_parts: list) -> list:
    parts = []
    for p in source_parts:
        t = p.get("type")
        if t == "text":
            parts.append({"text": p["text"][:15000]})
        elif t == "image":
            parts.append({"inline_data": {"mime_type": p["source"]["media_type"], "data": p["source"]["data"]}})
        elif t == "document":
            parts.append({"inline_data": {"mime_type": "application/pdf", "data": p["source"]["data"]}})
    if not parts:
        raise RuntimeError("No readable content found in the source.")
    return parts


def _generate_via_gemini(api_key, model, source_parts, count, difficulty, language) -> list:
    if not api_key:
        raise ProviderError("No Gemini key available for this step.", is_quota=False)
    parts = [{"text": _build_system_prompt(count, difficulty, language)}] + _build_gemini_parts(source_parts)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {"contents": [{"parts": parts}], "generationConfig": {"maxOutputTokens": 4096}}
    resp = None
    try:
        resp = requests.post(url, json=body, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        status_code = resp.status_code if resp is not None else None
        body_text = resp.text[:300] if resp is not None else str(e)
        is_quota = _looks_like_quota_error(body_text, status_code=status_code)
        is_auth = _looks_like_auth_error(body_text, status_code=status_code)
        logger.error(f"Gemini call failed ({status_code}): {body_text}")
        raise ProviderError(_friendly_provider_message(body_text, is_quota, is_auth), is_quota=is_quota, status_code=status_code)
    return _parse_json_questions(raw_text)


GEN_BATCH_SIZE = 20


def _generate_batch(user_id: int, source_parts: list, batch_count: int, difficulty: str, language: str) -> list:
    chain = build_provider_chain(user_id)
    if not chain:
        raise RuntimeError("⚠️ Koi AI key available nahi hai. Bot owner se ANTHROPIC_API_KEY/GROQ_API_KEY/GEMINI_API_KEY set karne ko bolo.")

    errors = []
    any_non_quota = False
    for cand in chain:
        if is_key_exhausted_today(cand):
            logger.warning(f"Skipping '{cand['label']}' for user {user_id} — local daily limit already reached today.")
            errors.append((cand["label"], "Aaj ki daily limit (tracked) khatam ho chuki hai.", True))
            continue
        try:
            if cand["provider"] == "anthropic":
                result = _generate_via_anthropic(cand["api_key"], cand["model"], source_parts, batch_count, difficulty, language)
            elif cand["provider"] == "gemini":
                result = _generate_via_gemini(cand["api_key"], cand["model"], source_parts, batch_count, difficulty, language)
            else:
                result = _generate_via_openai_compatible(cand["api_key"], cand["api_base"], cand["model"], source_parts, batch_count, difficulty, language)
            record_api_key_usage(cand)
            return result
        except ProviderError as e:
            logger.warning(f"Provider '{cand['label']}' failed for user {user_id} (quota={e.is_quota}): {e}")
            errors.append((cand["label"], str(e), e.is_quota))
            if not e.is_quota:
                any_non_quota = True
            continue
        except RuntimeError as e:
            logger.warning(f"Provider '{cand['label']}' failed for user {user_id} (non-quota): {e}")
            errors.append((cand["label"], str(e), False))
            any_non_quota = True
            continue

    if errors and not any_non_quota:
        lines = "\n".join(f"• {label}" for label, _, _ in errors)
        raise RuntimeError(f"⚠️ **API key(s) ki limit/quota khatam ho gayi hai**:\n{lines}\n\nThodi der baad try karo.")

    lines = "\n".join(f"• {label}: {msg}" for label, msg, _ in errors)
    raise RuntimeError(f"⚠️ Question generate nahi ho paaye, har provider me problem aayi:\n{lines}")


def generate_mcqs_sync(user_id: int, source_parts: list, count: int, difficulty: str, language: str) -> list:
    count = max(1, min(count, 100))
    all_questions = []
    remaining = count
    while remaining > 0:
        batch_count = min(GEN_BATCH_SIZE, remaining)
        batch = _generate_batch(user_id, source_parts, batch_count, difficulty, language)
        all_questions.extend(batch)
        remaining -= batch_count
    return all_questions[:count]


async def generate_mcqs_async(user_id: int, source_parts: list, count: int, difficulty: str, language: str) -> list:
    count = max(1, min(count, 100))
    batch_sizes = []
    remaining = count
    while remaining > 0:
        batch_sizes.append(min(GEN_BATCH_SIZE, remaining))
        remaining -= batch_sizes[-1]

    tasks = [asyncio.to_thread(_generate_batch, user_id, source_parts, n, difficulty, language) for n in batch_sizes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_questions = []
    first_error = None
    for r in results:
        if isinstance(r, Exception):
            if first_error is None:
                first_error = r
            continue
        all_questions.extend(r)

    if not all_questions and first_error is not None:
        raise first_error
    return all_questions[:count]


def shuffle_question_options(q: dict) -> dict:
    order = list(range(len(q["options"])))
    random.shuffle(order)
    new_options = [q["options"][i] for i in order]
    new_correct = order.index(q["correct_index"])
    shuffled = dict(q)
    shuffled["options"] = new_options
    shuffled["correct_index"] = new_correct
    return shuffled


_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


def detect_language(text: str) -> str:
    """Simple auto-detect: any Devanagari script present -> Hindi, else English."""
    if text and _DEVANAGARI_RE.search(text):
        return "Hindi"
    return "English"


# ==========================================
# 🎙️ VOICE TRANSCRIPTION
# ==========================================
def _transcribe_via_openai_whisper(api_key, api_base, model_hint, audio_bytes: bytes) -> str:
    if not api_base:
        raise ProviderError("This provider isn't fully configured for audio.", is_quota=False)
    whisper_model = "whisper-large-v3" if "groq" in api_base else "whisper-1"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    files = {"file": ("voice.ogg", audio_bytes, "audio/ogg")}
    data = {"model": whisper_model}
    resp = None
    try:
        resp = requests.post(f"{api_base.rstrip('/')}/audio/transcriptions", headers=headers, files=files, data=data, timeout=60)
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
    except Exception as e:
        status_code = resp.status_code if resp is not None else None
        body_text = resp.text[:300] if resp is not None else str(e)
        is_quota = _looks_like_quota_error(body_text, status_code=status_code)
        is_auth = _looks_like_auth_error(body_text, status_code=status_code)
        logger.error(f"Whisper transcription failed ({status_code}): {body_text}")
        raise ProviderError(_friendly_provider_message(body_text, is_quota, is_auth), is_quota=is_quota, status_code=status_code)
    if not text:
        raise ProviderError("Audio se koi text nahi mila — dobara, thoda saaf bolke try karo.", is_quota=False)
    return text


def _transcribe_via_gemini(api_key, model, audio_bytes: bytes) -> str:
    if not api_key:
        raise ProviderError("No Gemini key available for this step.", is_quota=False)
    b64 = base64.b64encode(audio_bytes).decode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {"contents": [{"parts": [
        {"text": "Transcribe this audio exactly, in whatever language is spoken. Reply with ONLY the transcript text, nothing else."},
        {"inline_data": {"mime_type": "audio/ogg", "data": b64}},
    ]}]}
    resp = None
    try:
        resp = requests.post(url, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        status_code = resp.status_code if resp is not None else None
        body_text = resp.text[:300] if resp is not None else str(e)
        is_quota = _looks_like_quota_error(body_text, status_code=status_code)
        is_auth = _looks_like_auth_error(body_text, status_code=status_code)
        logger.error(f"Gemini transcription failed ({status_code}): {body_text}")
        raise ProviderError(_friendly_provider_message(body_text, is_quota, is_auth), is_quota=is_quota, status_code=status_code)
    if not text:
        raise ProviderError("Audio se koi text nahi mila — dobara, thoda saaf bolke try karo.", is_quota=False)
    return text


def transcribe_voice_sync(user_id: int, audio_bytes: bytes) -> str:
    chain = build_provider_chain(user_id)
    audio_capable = [c for c in chain if c["provider"] in ("groq", "openai", "gemini")]
    if not audio_capable:
        raise RuntimeError("⚠️ Voice samajhne ke liye Groq, OpenAI, ya Gemini API key chahiye (owner se set karwao).")

    errors = []
    any_non_quota = False
    for cand in audio_capable:
        try:
            if cand["provider"] == "gemini":
                return _transcribe_via_gemini(cand["api_key"], cand["model"], audio_bytes)
            return _transcribe_via_openai_whisper(cand["api_key"], cand["api_base"], cand["model"], audio_bytes)
        except ProviderError as e:
            errors.append((cand["label"], str(e), e.is_quota))
            if not e.is_quota:
                any_non_quota = True
            continue

    if errors and not any_non_quota:
        lines = "\n".join(f"• {label}" for label, _, _ in errors)
        raise RuntimeError(f"⚠️ **API key(s) ki limit/quota khatam ho gayi hai** — voice samajh nahi paya:\n{lines}")
    lines = "\n".join(f"• {label}: {msg}" for label, msg, _ in errors)
    raise RuntimeError(f"⚠️ Voice samajh nahi paya, har provider me problem aayi:\n{lines}")


_VOICE_NUM_WORDS = {
    'ek': 1, 'one': 1, 'do': 2, 'two': 2, 'teen': 3, 'three': 3, 'char': 4, 'chaar': 4, 'four': 4,
    'paanch': 5, 'panch': 5, 'five': 5, 'chhe': 6, 'chhah': 6, 'six': 6, 'saat': 7, 'seven': 7,
    'aath': 8, 'eight': 8, 'nau': 9, 'nine': 9, 'das': 10, 'dus': 10, 'ten': 10,
    'pandrah': 15, 'fifteen': 15, 'bees': 20, 'twenty': 20, 'pachees': 25, 'twenty-five': 25,
}
_VOICE_FILLER_RE = re.compile(
    r'\b(sawal|sawaal|question|questions|quiz|banao|bana\s*do|bnao|bnado|banaiye|banana|mcq|mcqs|'
    r'ke|ka|ki|pe|par|se|topic|please|plz|kripya)\b', re.IGNORECASE
)


def parse_voice_command(transcript: str):
    t = transcript.strip()
    count = None
    m = re.search(r'\b(\d{1,3})\b', t)
    if m:
        count = max(1, min(int(m.group(1)), 100))
        t = t[:m.start()] + " " + t[m.end():]
    else:
        for word, val in _VOICE_NUM_WORDS.items():
            wm = re.search(rf'\b{re.escape(word)}\b', t, re.IGNORECASE)
            if wm:
                count = val
                t = t[:wm.start()] + " " + t[wm.end():]
                break
    topic = _VOICE_FILLER_RE.sub(' ', t)
    topic = re.sub(r'\s{2,}', ' ', topic).strip(" ,.!?-")
    return topic, count


# ==========================================
# 📮 POSTING QUIZZES — native Telegram quiz-polls with an auto per-question timer
# ==========================================
async def post_quiz_polls(bot, chat_id, questions: list, timer_seconds: int = 0, option_style: str = 'plain') -> int:
    sent = 0
    for q in questions:
        question_text = q["question"][:300]
        explanation = (q.get("explanation") or "")[:200] or None
        try:
            await bot.send_poll(
                chat_id=chat_id,
                question=question_text,
                options=apply_option_style(q["options"], option_style),
                type=Poll.QUIZ,
                correct_option_id=q["correct_index"],
                explanation=explanation,
                is_anonymous=False,
                open_period=timer_seconds if timer_seconds else None,
            )
            sent += 1
        except Exception as e:
            logger.warning(f"Failed to send a poll: {e}")
    return sent


# ==========================================
# 🔐 VERIFICATION LOGIC
# ==========================================
async def check_user_membership(bot, user_id: int) -> bool:
    if user_id == OWNER_ID:
        return True
    try:
        member_ch = await bot.get_chat_member(chat_id=UPDATE_CHANNEL, user_id=user_id)
        member_gr = await bot.get_chat_member(chat_id=UPDATE_GROUP, user_id=user_id)
    except Exception as e:
        logger.warning(f"Verification check failed for user {user_id}: {e}")
        return False
    valid_statuses = ('creator', 'administrator', 'member')
    return member_ch.status in valid_statuses and member_gr.status in valid_statuses


async def require_verification(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    if not user or user.id == OWNER_ID:
        return True
    user_data = get_user(user.id)
    if user_data['is_verified']:
        return True
    if await check_user_membership(context.bot, user.id):
        set_verified(user.id, 1)
        return True
    msg = update.effective_message
    if msg:
        await msg.reply_text(
            text=(f"🔐 **{stylize('Verification Required')}**\n_{FONT_STYLE}_\n\n"
                  "Please join our update channel and support group, then tap "
                  "**I Am Verified** below to continue."),
            parse_mode="Markdown", reply_markup=verification_keyboard(),
        )
    return False


async def verification_command_gate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if msg and msg.text and msg.text.split()[0].split('@')[0] == "/start":
        return
    if not await require_verification(update, context):
        raise ApplicationHandlerStop


async def handle_update_channel_group_membership(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cmu = update.chat_member
    if not cmu:
        return
    chat = cmu.chat
    chat_ref = f"@{chat.username}" if chat.username else str(chat.id)
    tracked_refs = {UPDATE_CHANNEL, UPDATE_GROUP, UPDATE_CHANNEL.lstrip('@'), UPDATE_GROUP.lstrip('@')}
    if chat_ref not in tracked_refs and str(chat.id) not in tracked_refs:
        return

    valid_statuses = ('creator', 'administrator', 'member')
    was_in = cmu.old_chat_member.status in valid_statuses
    still_in = cmu.new_chat_member.status in valid_statuses
    if not (was_in and not still_in):
        return

    user = cmu.new_chat_member.user
    if user.is_bot or user.id == OWNER_ID:
        return
    set_verified(user.id, 0)
    try:
        await context.bot.send_message(
            chat_id=user.id,
            text=("⚠️ **You left our update channel/group, so your verification was reset.**\n\n"
                  "The bot won't work again until you rejoin and verify.\n"
                  "Tap the buttons below, then hit **I Am Verified**."),
            parse_mode="Markdown", reply_markup=verification_keyboard(),
        )
    except Exception as e:
        logger.warning(f"Couldn't DM rejoin notice to {user.id} (they may not have started the bot): {e}")


# ==========================================
# 🚀 BASIC COMMANDS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    user_data = get_user(user_id)

    is_verified = await check_user_membership(context.bot, user_id)
    if not is_verified:
        verify_msg = (
            f"🔐 **{stylize('Verification Required')}**\n_{FONT_STYLE}_\n\n"
            f"Hello **{user_name}**, to use this bot, please join our official update channel "
            f"and support group first.\n\nClick **I Am Verified** after joining!"
        )
        await update.message.reply_text(text=verify_msg, parse_mode="Markdown", reply_markup=verification_keyboard())
        return

    set_verified(user_id, 1)
    user_data = get_user(user_id)
    await update.message.reply_text(
        text=build_welcome_text(user_name, user_data),
        parse_mode="Markdown",
        reply_markup=main_menu_keyboard(),
    )


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("❌ Invalid command. Type /help to see the list of valid commands.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    daily_left = get_daily_ai_remaining(update.effective_user.id)
    limit_line = "♾️ Unlimited AI generation" if daily_left is None else f"🎯 {daily_left}/{DAILY_AI_QUESTION_LIMIT} AI questions left today"
    help_text = (
        f"📖 **{stylize('Help')}**\n_{FONT_STYLE}_ · {limit_line}\n\n"
        "📌 **Quiz Creation:**\n"
        "• `/topic` — MCQs from a topic or text\n"
        "• `/pdf` — MCQs from a PDF\n"
        "• `/image` — MCQs from a photo\n"
        "• `/voicequiz` — 🎙️ bolo topic aur count, quiz ban jayegi\n\n"
        "Language (Hindi/English) is auto-detected, options are auto-shuffled, and the "
        "per-question timer (10-30s) is set automatically. You'll just pick the number "
        "of questions: 25 / 50 / 75 / 100.\n\n"
        "• `/cancel` — stop whatever you're in the middle of"
    )
    await update.message.reply_text(text=help_text, parse_mode="Markdown")


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.effective_message.reply_text("❌ Cancelled — nothing was lost.")
    return ConversationHandler.END


async def cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data.clear()
    try:
        await query.edit_message_text("❌ Cancelled — nothing was lost.")
    except Exception:
        pass
    return ConversationHandler.END


# ==========================================
# 🧠 AI QUIZ GENERATION — CONVERSATION FLOW (topic / pdf / image)
# start -> verification -> pick source -> send content -> auto language -> pick
# count (25/50/75/100) -> auto timer (10-30s) + auto shuffle -> polls posted
# ==========================================
async def ai_entry(update: Update, context: ContextTypes.DEFAULT_TYPE, source_type: str):
    context.user_data.clear()
    context.user_data['gen_source_type'] = source_type
    context.user_data['gen_parts'] = []
    prompts = {
        'topic': "Send me the topic or paste the text you want MCQs from.",
        'pdf': "Send me the PDF file.",
        'image': "Send me a photo of the page.",
    }
    msg = update.effective_message
    text = stylize("Let's build your quiz") + f"\n\n{prompts.get(source_type, 'Send your content.')}"
    await msg.reply_text(text, parse_mode="Markdown", reply_markup=cancel_keyboard())
    return ASK_CONTENT


def make_command_entry(source_type: str):
    async def _entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if source_type == 'topic' and context.args:
            context.user_data.clear()
            context.user_data['gen_source_type'] = 'topic'
            context.user_data['gen_parts'] = [{"type": "text", "text": " ".join(context.args)}]
            return await after_content(update, context)
        return await ai_entry(update, context, source_type)
    return _entry


def make_button_entry(source_type: str):
    async def _entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.callback_query.answer()
        return await ai_entry(update, context, source_type)
    return _entry


topic_entry = make_command_entry('topic')
pdf_entry = make_command_entry('pdf')
image_entry = make_command_entry('image')

topic_button_entry = make_button_entry('topic')
pdf_button_entry = make_button_entry('pdf')
image_button_entry = make_button_entry('image')


async def receive_content(update: Update, context: ContextTypes.DEFAULT_TYPE):
    source_type = context.user_data.get('gen_source_type')
    msg = update.effective_message

    if source_type == 'topic':
        if not msg.text:
            await msg.reply_text("Please send text.")
            return ASK_CONTENT
        context.user_data['gen_parts'] = [{"type": "text", "text": msg.text}]
        return await after_content(update, context)

    if source_type == 'pdf':
        if not (msg.document and msg.document.file_name and msg.document.file_name.lower().endswith('.pdf')):
            await msg.reply_text("Please send a PDF file.")
            return ASK_CONTENT
        file = await msg.document.get_file()
        data = await file.download_as_bytearray()
        b64 = base64.b64encode(bytes(data)).decode()
        context.user_data['gen_parts'] = [{
            "type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
        }]
        return await after_content(update, context)

    if source_type == 'image':
        if not msg.photo:
            await msg.reply_text("Please send a photo.")
            return ASK_CONTENT
        file = await msg.photo[-1].get_file()
        data = await file.download_as_bytearray()
        b64 = base64.b64encode(bytes(data)).decode()
        context.user_data['gen_parts'] = [{
            "type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        }]
        return await after_content(update, context)

    return ConversationHandler.END


async def after_content(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Difficulty is fixed at 'medium' and language is auto-detected — no manual
    steps. Text sources (topic/voice) are detected directly; PDF/image content is
    detected by the AI itself at generation time (language='auto')."""
    context.user_data['gen_difficulty'] = 'medium'
    source_type = context.user_data.get('gen_source_type')
    parts = context.user_data.get('gen_parts', [])
    if source_type in ('topic', 'voice') and parts and parts[0].get('type') == 'text':
        context.user_data['gen_language'] = detect_language(parts[0]['text'])
    else:
        context.user_data['gen_language'] = 'auto'
    return await ask_count(update, context)


async def ask_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if context.user_data.get('gen_count'):
        await context.bot.send_message(chat_id, stylize("Generating your quiz") + "...")
        return await run_generation(update, context)
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("25", callback_data="cnt_25"),
        InlineKeyboardButton("50", callback_data="cnt_50"),
        InlineKeyboardButton("75", callback_data="cnt_75"),
        InlineKeyboardButton("100", callback_data="cnt_100"),
    ]])
    await context.bot.send_message(chat_id, "How many questions?", reply_markup=keyboard)
    return ASK_COUNT


async def receive_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    context.user_data['gen_count'] = int(query.data.replace('cnt_', ''))
    await context.bot.send_message(update.effective_chat.id, stylize("Generating your quiz") + "...")
    return await run_generation(update, context)


async def run_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user = get_user(user_id)

    remaining = user['question_limit'] - user['questions_used']
    if remaining <= 0:
        await context.bot.send_message(chat_id, "You've used your full quota. Contact the bot owner.")
        context.user_data.clear()
        return ConversationHandler.END

    daily_left = get_daily_ai_remaining(user_id)
    if daily_left is not None:
        if daily_left <= 0:
            await context.bot.send_message(
                chat_id,
                f"⏳ Aaj ka {DAILY_AI_QUESTION_LIMIT}-question AI limit khatam ho gaya — UTC midnight ke baad reset hoga."
            )
            context.user_data.clear()
            return ConversationHandler.END
        remaining = min(remaining, daily_left)

    count = min(context.user_data.get('gen_count', 25), remaining)
    parts = context.user_data.get('gen_parts', [])
    difficulty = context.user_data.get('gen_difficulty', 'medium')
    language = context.user_data.get('gen_language', 'auto')
    source_type = context.user_data.get('gen_source_type', 'topic')

    if not parts:
        await context.bot.send_message(chat_id, "Something went wrong — no content was captured. Please try again.")
        context.user_data.clear()
        return ConversationHandler.END

    try:
        questions = await generate_mcqs_async(user_id, parts, count, difficulty, language)
    except RuntimeError as e:
        await report_generation_failure(context.bot, chat_id, user_id, e)
        context.user_data.clear()
        return ConversationHandler.END

    # Auto shuffle (always) + auto timer, randomly picked once per quiz in 10-30s.
    questions = [shuffle_question_options(q) for q in questions]
    timer = random.randint(10, 30)

    title = "Quiz"
    for p in parts:
        if p.get('type') == 'text':
            title = p['text'][:50]
            break

    quiz_id = save_quiz(user_id, title, questions, timer_seconds=timer, quiz_type='ai_generated',
                         creator_name=user['tag'] or (update.effective_user.first_name or 'Unknown'))
    sent = await post_quiz_polls(context.bot, chat_id, questions, timer_seconds=timer, option_style=user['option_style'])
    increment_questions_used(user_id, sent)
    record_daily_ai_usage(user_id, sent)

    await context.bot.send_message(
        chat_id, f"✅ Done — {sent} question(s) posted (Quiz `{quiz_id}`). Timer: {timer}s/question.",
        parse_mode="Markdown",
    )
    context.user_data.clear()
    return ConversationHandler.END


# ==========================================
# 🎙️ VOICE → QUIZ CONVERSATION FLOW
# ==========================================
async def voice_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    text = (
        f"🎙️ **{stylize('Voice se Quiz')}**\n_{FONT_STYLE}_\n\n"
        "Ek voice message bhejo — bas bolo kis topic pe kitne questions chahiye.\n\n"
        "Jaise: _\"Science ke 25 sawal banao\"_ ya _\"Make 50 questions on Indian history\"_.\n"
        "Number na bolo to agle step me 25/50/75/100 me se chun sakte ho."
    )
    await update.effective_message.reply_text(text, parse_mode="Markdown", reply_markup=cancel_keyboard())
    return ASK_VOICE


async def voice_button_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    return await voice_entry(update, context)


async def receive_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    voice = msg.voice or msg.audio
    if not voice:
        await msg.reply_text("Please send a voice message (🎤 hold to record), or tap Cancel.", reply_markup=cancel_keyboard())
        return ASK_VOICE

    file = await voice.get_file()
    data = await file.download_as_bytearray()
    await msg.reply_text(stylize("Sun raha hoon") + "...")

    try:
        transcript = await asyncio.to_thread(transcribe_voice_sync, update.effective_user.id, bytes(data))
    except RuntimeError as e:
        await report_generation_failure(context.bot, msg.chat_id, update.effective_user.id, e)
        return ConversationHandler.END

    topic, count = parse_voice_command(transcript)
    if not topic:
        await msg.reply_text(
            f"Sun toh liya (\"{transcript}\"), lekin topic samajh nahi aaya. Fir se try karo — "
            "jaise \"History pe sawal banao\"."
        )
        return ASK_VOICE

    context.user_data['gen_source_type'] = 'voice'
    context.user_data['gen_parts'] = [{"type": "text", "text": topic}]
    if count:
        context.user_data['gen_count'] = count
    summary = f"🎧 Samjha: **{topic}**" + (f" pe **{count}** sawal." if count else ".")
    await msg.reply_text(summary, parse_mode="Markdown")
    return await after_content(update, context)


# ==========================================
# 🎛️ CALLBACK QUERY ROUTER (menu navigation + verification)
# ==========================================
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id
    user_name = query.from_user.first_name

    now = time.monotonic()
    last_tap = context.application.bot_data.setdefault('_last_callback_at', {})
    if now - last_tap.get(user_id, 0) < 0.5:
        await query.answer()
        return
    last_tap[user_id] = now

    if data == "check_verification":
        await query.answer()
        is_verified = await check_user_membership(context.bot, user_id)
        if is_verified:
            set_verified(user_id, 1)
            user_data = get_user(user_id)
            await query.edit_message_text(
                text=build_welcome_text(user_name, user_data, verified_banner=True),
                parse_mode="Markdown", reply_markup=main_menu_keyboard(),
            )
        else:
            await query.answer("❌ You haven't joined the required Channel or Group yet!", show_alert=True)
        return

    if user_id != OWNER_ID:
        user_data = get_user(user_id)
        if not user_data['is_verified']:
            if await check_user_membership(context.bot, user_id):
                set_verified(user_id, 1)
            else:
                await query.answer("🔐 Please join our update channel & group first, then tap ✅ I Am Verified.", show_alert=True)
                return

    await query.answer()

    if data == "menu_main":
        user_data = get_user(user_id)
        await query.edit_message_text(
            text=build_welcome_text(user_name, user_data),
            parse_mode="Markdown", reply_markup=main_menu_keyboard(),
        )
    elif data == "open_help":
        daily_left = get_daily_ai_remaining(user_id)
        limit_line = "♾️ Unlimited AI generation" if daily_left is None else f"🎯 {daily_left}/{DAILY_AI_QUESTION_LIMIT} AI questions left today"
        await query.edit_message_text(
            text=(f"📖 **{stylize('Help')}**\n_{FONT_STYLE}_ · {limit_line}\n\n"
                  "🧠 Topic · 📄 PDF · 🖼 Image · 🎙️ Voice — pick a source, send your content, "
                  "then choose 25/50/75/100 questions. Language, timer and shuffle are automatic."),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Main Menu", callback_data="menu_main")]]),
        )


# ==========================================
# 🚨 GLOBAL ERROR HANDLER
# ==========================================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    err_name = err.__class__.__name__

    if err_name == "Conflict":
        logger.error("⚠️ Telegram 'Conflict' — another instance of this bot is ALSO polling with the same BOT_TOKEN right now.")
        try:
            if OWNER_ID:
                await context.bot.send_message(
                    OWNER_ID,
                    "⚠️ Bot Conflict error: a SECOND instance of this bot is polling with the same token. Stop the other one.",
                )
        except Exception:
            pass
        return

    if err_name in ("NetworkError", "TimedOut", "RetryAfter"):
        logger.warning(f"Transient network issue ({err_name}): {err} — will retry automatically.")
        return

    if err_name == "Forbidden":
        logger.info(f"Forbidden (bot blocked/removed by a user or group) — ignored safely: {err}")
        return

    if err_name == "BadRequest" and "not modified" in str(err).lower():
        logger.debug(f"Harmless no-op edit ignored: {err}")
        return

    logger.error("Exception while handling an update:", exc_info=err)


# ==========================================
# 🌐 KEEP-ALIVE WEB SERVER (Render/Railway/Replit free "Web Service" tier)
# ==========================================
class _PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Quiz bot is running.")

    def log_message(self, format, *args):
        pass


def start_keepalive_server():
    port = int(os.getenv("PORT", "10000"))
    try:
        server = HTTPServer(("0.0.0.0", port), _PingHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"🌐 Keep-alive HTTP server listening on port {port}")
    except OSError as e:
        # e.g. running in Pydroid 3 with the port already taken / no permission —
        # non-fatal, the bot itself doesn't need this server to poll Telegram.
        logger.warning(f"Keep-alive server didn't start (non-fatal): {e}")


# ==========================================
# 🏁 MAIN ENTRY POINT
# ==========================================
def main():
    init_db()
    start_keepalive_server()

    async def _post_init(application):
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=64, thread_name_prefix="quizbot_worker"))
        logger.info("Thread pool executor enlarged to 64 workers.")

    request_config = HTTPXRequest(connect_timeout=60.0, read_timeout=60.0, write_timeout=60.0, pool_timeout=60.0)

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(request_config)
        .get_updates_request(request_config)
        .post_init(_post_init)
        .build()
    )

    app.add_error_handler(error_handler)

    # Verification gate: runs before every /command except /start.
    app.add_handler(MessageHandler(filters.COMMAND, verification_command_gate), group=-1)
    app.add_handler(ChatMemberHandler(handle_update_channel_group_membership, ChatMemberHandler.CHAT_MEMBER))

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))

    # --- AI quiz generation conversation (topic / pdf / image) ---
    ai_conv = ConversationHandler(
        entry_points=[
            CommandHandler("topic", topic_entry),
            CommandHandler("pdf", pdf_entry),
            CommandHandler("image", image_entry),
            CallbackQueryHandler(topic_button_entry, pattern="^create_topic$"),
            CallbackQueryHandler(pdf_button_entry, pattern="^create_pdf$"),
            CallbackQueryHandler(image_button_entry, pattern="^create_image$"),
        ],
        states={
            ASK_CONTENT: [MessageHandler((filters.TEXT | filters.Document.ALL | filters.PHOTO) & ~filters.COMMAND, receive_content)],
            ASK_COUNT: [CallbackQueryHandler(receive_count, pattern="^cnt_")],
        },
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            CommandHandler("stop", cancel_command),
            CallbackQueryHandler(cancel_callback, pattern="^gcancel$"),
        ],
        conversation_timeout=600,
    )
    app.add_handler(ai_conv)

    # --- Voice → Quiz conversation ---
    voice_conv = ConversationHandler(
        entry_points=[
            CommandHandler("voicequiz", voice_entry),
            CallbackQueryHandler(voice_button_entry, pattern="^create_voice$"),
        ],
        states={
            ASK_VOICE: [MessageHandler((filters.VOICE | filters.AUDIO) & ~filters.COMMAND, receive_voice)],
            ASK_COUNT: [CallbackQueryHandler(receive_count, pattern="^cnt_")],
        },
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            CommandHandler("stop", cancel_command),
            CallbackQueryHandler(cancel_callback, pattern="^gcancel$"),
        ],
        conversation_timeout=600,
    )
    app.add_handler(voice_conv)

    # --- Menu navigation / verification (generic router — must stay LAST among callback handlers) ---
    app.add_handler(CallbackQueryHandler(button_callback))

    # --- Unknown/invalid command catch-all ---
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    # --- Self-backup + owner usage alerts, so the bot survives restarts/host wipes ---
    if app.job_queue:
        app.job_queue.run_repeating(periodic_backup_job, interval=6 * 3600, first=120, name="db_auto_backup")
        app.job_queue.run_repeating(flush_owner_notifications_job, interval=90, first=30, name="usage_alert_flush")

    print(f"🤖 Bot Engine Started Successfully...\nStyle: {FONT_STYLE}")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


def run_forever():
    """Wraps main() in a restart loop with backoff so the bot stays up 24x7 even if
    the process itself dies for a reason other than a normal transient network
    error (which app.run_polling() already retries on its own)."""
    backoff = 5
    while True:
        try:
            main()
            break
        except Exception:
            logger.error("Bot crashed — restarting shortly. Full traceback:", exc_info=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)


if __name__ == "__main__":
    run_forever()
