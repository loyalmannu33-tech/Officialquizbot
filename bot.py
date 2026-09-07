import os
import re
import csv
import io
import json
import base64
import logging
import sqlite3
import asyncio
import time
import traceback
import random
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ChatPermissions,
    Poll,
    Bot,
)
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    PollAnswerHandler,
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
# 🔐 IMPORTANT: token ab .env / environment variable se aata hai, code me kabhi mat likhna.
# Agar tumne pehle wala hardcoded token kahin use kiya hai, use @BotFather se turant revoke karo.
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise SystemExit(
        "BOT_TOKEN environment variable is not set.\n"
        "Copy .env.example to .env, fill in a NEW token from @BotFather, then run again."
    )

OWNER_ID = int(os.getenv("OWNER_ID", "123456789"))
DB_NAME = os.getenv("DB_NAME", "quiz_bot.db")

# 🐘 Optional Postgres support — set DATABASE_URL (e.g. a free Supabase / Neon / Render
# Postgres instance) and the bot stores everything there instead of the local SQLite
# file, so data survives redeploys/restarts on free hosts with an ephemeral disk.
# Leave DATABASE_URL unset (e.g. running in Pydroid 3) and it keeps using SQLite exactly
# like before — nothing else changes.
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USE_POSTGRES = bool(DATABASE_URL) and psycopg2 is not None
if bool(DATABASE_URL) and psycopg2 is None:
    raise SystemExit(
        "DATABASE_URL is set but 'psycopg2-binary' isn't installed.\n"
        "Run: pip install psycopg2-binary"
    )
PK_TYPE = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"

# AI generation (needed for /topic, /pdf, /image, /section)
# These are the bot OWNER's fallback keys — used for anyone who hasn't added their
# own with /addapi. Set ANY of these env vars you have; every one you set becomes an
# extra fallback that EVERY user gets automatically, tried in OWNER_FALLBACK_ORDER
# below, on top of each user's own personal /addapi keys (which are always tried
# FIRST — owner keys are the safety net, not the first choice).
#
# Got MULTIPLE keys for the same provider (e.g. several Groq accounts)? Put them all
# in the same env var separated by commas — e.g. GROQ_API_KEY=key1,key2,key3 — and
# every one becomes its own fallback entry, tried in the order listed, each with its
# own independently-tracked daily usage/alerts (see /limits).
#
# Anthropic has NO ongoing free API tier (only a one-time ~$5 signup credit) — so if
# you want a genuinely free, always-on shared key, set GROQ_API_KEY / GEMINI_API_KEY /
# CEREBRAS_API_KEY / NVIDIA_API_KEY instead (or alongside). ANTHROPIC_API_KEY still
# works exactly as before if you do have a paid key — it's just no longer the ONLY
# fallback option.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "claude-sonnet-5")


def _split_keys(raw: str) -> list:
    """'key1, key2,key3' -> ['key1','key2','key3']; '' -> []. Lets any of the env
    vars below hold either one key or several comma-separated ones interchangeably."""
    return [k.strip() for k in raw.split(",") if k.strip()]


OWNER_PROVIDER_KEYS = {
    "cerebras": {"api_keys": _split_keys(os.getenv("CEREBRAS_API_KEY", ""))},
    "groq": {"api_keys": _split_keys(os.getenv("GROQ_API_KEY", ""))},
    "gemini": {"api_keys": _split_keys(os.getenv("GEMINI_API_KEY", ""))},
    "nvidia": {"api_keys": _split_keys(os.getenv("NVIDIA_API_KEY", ""))},
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
    "deepseek": {"api_keys": _split_keys(os.getenv("DEEPSEEK_API_KEY", ""))},
    "openai": {"api_keys": _split_keys(os.getenv("OPENAI_API_KEY", ""))},
    "anthropic": {"api_keys": _split_keys(ANTHROPIC_API_KEY)},
}
# Cheapest/most-generous-and-genuinely-free providers first; paid ones (openai,
# anthropic) last since owner keys are meant to be the free safety net. DeepSeek
# sits near the end of the free ones since its free tokens are a one-time signup
# credit (not a daily-reset allowance like cerebras/groq/gemini/nvidia above it).
OWNER_FALLBACK_ORDER = [
    "cerebras", "groq", "gemini", "nvidia", "cloudflare_text", "cloudflare_vision",
    "github", "openrouter_text", "openrouter_vision", "mistral", "huggingface",
    "deepseek", "openai", "anthropic",
]


# ==========================================
# 🔌 MULTI-PROVIDER AI CONFIG
# ==========================================
# Presets shown in the /addapi menu. "style" decides which HTTP call shape is used:
#   - "anthropic": Anthropic Messages API (native image/PDF support)
#   - "openai":    OpenAI-compatible /chat/completions (Groq, OpenAI, and any
#                  open-source/self-hosted server — Ollama, LM Studio, vLLM,
#                  Together, OpenRouter, etc. — text-only here)
#   - "gemini":    Google Generative Language API (text-only here)
PROVIDER_PRESETS = {
    "groq": {
        "label": "🟢 Groq",
        "style": "openai",
        "api_base": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "label": "🔵 Google Gemini",
        "style": "gemini",
        "api_base": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.5-flash",  # gemini-2.0-flash was retired by Google on 2026-06-01
    },
    "openai": {
        "label": "🟣 OpenAI",
        "style": "openai",
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "anthropic": {
        "label": "🟠 Anthropic (Claude)",
        "style": "anthropic",
        "api_base": None,
        "model": "claude-sonnet-5",
    },
    # ---- Free/extra-provider options (all OpenAI-compatible /chat/completions,
    # so they reuse _generate_via_openai_compatible as-is — including its existing
    # vision (image_url) and PDF-text-extraction handling, no new code paths needed) ----
    "github": {
        "label": "🐙 GitHub Models (gpt-4o-mini)",
        "style": "openai",
        "api_base": "https://models.inference.ai.azure.com",
        "model": "gpt-4o-mini",
        "note": "Image dekhkar sawal, PDF text aur topic se seedha poll — sabme accha. Key: GitHub → Settings → Developer settings → Personal access tokens (fine-grained, 'Models' permission).",
    },
    "openrouter_vision": {
        "label": "🌐 OpenRouter (Vision)",
        "style": "openai",
        "api_base": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "note": "Photo/image se text padhkar MCQ banane ke liye — free tier.",
    },
    "openrouter_text": {
        "label": "🌐 OpenRouter (Text/PDF)",
        "style": "openai",
        "api_base": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "note": "Bade PDF, notes ya CSV se bulk quiz banane ke liye — free tier, long context.",
    },
    "cloudflare_vision": {
        "label": "☁️ Cloudflare (Vision)",
        "style": "openai",
        "api_base": "",  # account-specific — filled in via /addapi's base-URL step
        "model": "@cf/meta/llama-3.2-11b-vision-instruct",
        "needs_base": True,
        "note": "Photo input pe tez response ke liye.",
    },
    "cloudflare_text": {
        "label": "☁️ Cloudflare (Fast Text)",
        "style": "openai",
        "api_base": "",  # account-specific — filled in via /addapi's base-URL step
        "model": "@cf/meta/llama-3.1-8b-instruct",
        "needs_base": True,
        "note": "Seedha Telegram Poll format (Question + 4 Options) jaldi generate karne ke liye.",
    },
    "mistral": {
        "label": "🌊 Mistral (Pixtral OCR)",
        "style": "openai",
        "api_base": "https://api.mistral.ai/v1",
        "model": "pixtral-12b-2409",
        "note": "Diagram, chart ya kitaab ke panno se sawal nikalne ke liye (OCR-strong).",
    },
    "cerebras": {
        "label": "⚡ Cerebras (Ultra-fast)",
        "style": "openai",
        "api_base": "https://api.cerebras.ai/v1",
        "model": "llama-3.3-70b",
        "note": "Super-fast text-only response — sirf topic/PDF-text ke liye, images nahi.",
    },
    "nvidia": {
        "label": "🟩 NVIDIA Build (NIM)",
        "style": "openai",
        "api_base": "https://integrate.api.nvidia.com/v1",
        "model": "meta/llama-3.3-70b-instruct",
        "note": "Free dev account, card nahi chahiye — build.nvidia.com se key lo (~40 req/min tak). Text-only.",
    },
    "deepseek": {
        "label": "🐋 DeepSeek",
        "style": "openai",
        "api_base": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",  # DeepSeek keeps this alias pointed at their current
        # general model as they upgrade it — if it ever 404s, check platform.deepseek.com
        # for the current model name and update this one line.
        "note": "Naye account ko free signup tokens milte hain — card nahi chahiye. Text-only.",
    },
    "huggingface": {
        "label": "🤗 Hugging Face (Vision)",
        "style": "openai",
        "api_base": "https://router.huggingface.co/v1",
        "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "note": "Image aur general question-answering ke liye.",
    },
    "custom": {
        "label": "⚙️ Custom / Open-source",
        "style": "openai",
        "api_base": "",   # user supplies their own base URL
        "model": "",      # user supplies their own model name
        "note": "Koi bhi OpenAI-compatible provider yahi se add karo — jaise AI21 "
                "(base https://api.ai21.com/studio/v1, model jamba-mini — $10 free "
                "trial credit, card nahi chahiye) ya Pollinations (base "
                "https://gen.pollinations.ai/v1, model openai — free key enter.pollinations.ai "
                "se). Base URL + apni key + model naam us provider ke docs se le lo.",
    },
}

# Update channel & group details (Include '@' or Chat IDs)
UPDATE_CHANNEL = os.getenv("UPDATE_CHANNEL", "@Telegram")  # Apna channel username yaha dalein
UPDATE_GROUP = os.getenv("UPDATE_GROUP", "@Telegram")      # Apna group username yaha dalein

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states (shared range; each ConversationHandler tracks its own state independently)
(
    ASK_CONTENT, ASK_DIFFICULTY, ASK_LANGUAGE, ASK_COUNT, ASK_SECTION_COUNT,
    ASK_CSV, ASK_EDIT_CHOICE, ASK_EDIT_TEXT,
    ASK_API_PROVIDER, ASK_API_CUSTOM_BASE, ASK_API_CUSTOM_MODEL, ASK_API_KEY,
    OQ_TITLE, OQ_TIMER, OQ_METHOD, OQ_COLLECT, OQ_BULK,
    STORE_EDIT_TITLE, STORE_ADD_QUESTION, STORE_EDIT_TIMER, STORE_EDIT_DESC,
    ASK_VOICE, IMPORT_COLLECT, IMPORT_KEYWORD,
    ASK_QUIZBLAST_TEXT,
    NEWSET_NAME, ADD_BULK_TITLE, ADD_BULK_CONTENT, SETCHANNEL_WAIT,
    ASK_AI_TIMER, ASK_AI_SHUFFLE,
) = range(31)

# ==========================================
# 🎨 STYLISH FONT HELPER
# ==========================================
# Har bot message ka HEADER/TITLE isi font me dikhega. Body text (instructions,
# quiz questions, IDs, numbers) normal rehta hai taaki readable rahe.
STYLE_MAP = {
    'a': '𝛂', 'b': 'в', 'c': 'ς', 'd': '∂', 'e': 'є', 'f': 'ƒ', 'g': 'g',
    'h': 'н', 'i': 'ι', 'j': 'ʝ', 'k': 'ĸ', 'l': 'ʟ', 'm': 'м', 'n': 'η',
    'o': 'ο', 'p': 'ρ', 'q': 'q', 'r': 'ɤ', 's': '𝛅', 't': 'τ', 'u': '𝛖',
    'v': 'ν', 'w': 'ω', 'x': 'χ', 'y': 'у', 'z': 'ᴢ',
}


def stylize(text: str) -> str:
    """Convert plain text into the bot's signature stylised font."""
    return "".join(STYLE_MAP.get(ch.lower(), ch) for ch in text)


FONT_STYLE = stylize("your stay & let's grow together")

# ==========================================
# 💾 DATABASE CONNECTION LAYER (SQLite or Postgres)
# ==========================================
class _PGCursorShim:
    """Wraps a psycopg2 cursor so all the existing sqlite-style code — '?'
    placeholders and .lastrowid — keeps working unmodified against Postgres."""
    def __init__(self, cursor):
        self._cursor = cursor
        self.lastrowid = None

    def execute(self, sql, params=()):
        pg_sql = sql.replace("?", "%s")
        self._cursor.execute(pg_sql, params)
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
    """Returns a DB connection — Postgres if DATABASE_URL is set, SQLite otherwise.

    SQLite: timeout=30 makes a connection WAIT (instead of instantly raising
    "database is locked") if another thread is mid-write, and WAL journal mode lets
    reads and writes happen concurrently — together these were a real cause of random
    crashes under load (several handlers hitting the DB from different threads via
    asyncio.to_thread at the same time)."""
    if USE_POSTGRES:
        return _PGConnShim(psycopg2.connect(DATABASE_URL, sslmode="require"))
    conn = sqlite3.connect(DB_NAME, timeout=30, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
    except sqlite3.OperationalError:
        pass  # non-fatal — worst case we're back to the old (safe, just slower) behavior
    return conn


def last_insert_id(cursor, conn, table: str, pk: str = "id"):
    """Cross-DB way to fetch the id of the row just inserted."""
    if USE_POSTGRES:
        cursor.execute(f"SELECT currval(pg_get_serial_sequence('{table}', '{pk}'))")
        return cursor.fetchone()[0]
    return cursor.lastrowid


def upsert(cursor, table: str, keys: dict, values: dict):
    """Cross-DB INSERT-or-REPLACE keyed on `keys` (sqlite INSERT OR REPLACE vs
    Postgres INSERT ... ON CONFLICT DO UPDATE)."""
    all_cols = list(keys) + list(values)
    all_vals = list(keys.values()) + list(values.values())
    placeholders = ", ".join(["?"] * len(all_cols))
    if USE_POSTGRES:
        conflict_cols = ", ".join(keys)
        update_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in values) if values else conflict_cols
        sql = (
            f"INSERT INTO {table} ({', '.join(all_cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_clause}"
        )
    else:
        sql = f"INSERT OR REPLACE INTO {table} ({', '.join(all_cols)}) VALUES ({placeholders})"
    cursor.execute(sql, all_vals)


# ==========================================
# 💾 DATABASE INITIALIZATION
# ==========================================
def init_db():
    conn = db_connect()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id BIGINT PRIMARY KEY,
            questions_used INTEGER DEFAULT 0,
            question_limit INTEGER DEFAULT 100,
            referred_by BIGINT DEFAULT 0,
            is_verified INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS filters (
            group_id BIGINT,
            keyword TEXT,
            reply TEXT,
            PRIMARY KEY (group_id, keyword)
        )
    """)

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS quizzes (
            id {PK_TYPE},
            user_id BIGINT,
            title TEXT,
            data TEXT,
            timer_seconds INTEGER DEFAULT 0,
            quiz_type TEXT DEFAULT 'quick',
            creator_name TEXT DEFAULT ''
        )
    """)

    # Groups the bot has seen — so its list of chats survives a restart even
    # though the live-quiz session itself (pause/resume state) is in-memory.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS groups (
            group_id BIGINT PRIMARY KEY,
            title TEXT,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Saved leaderboard results from finished "Official Quiz" live sessions.
    # wrong_count / total_questions / avg_time_ms let every leaderboard (live,
    # /topusers, /leaderboard) show real accuracy % and response speed instead of
    # just a raw score — and user_id is always kept alongside user_name so the
    # quiz creator can identify/contact a specific participant directly.
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS quiz_results (
            id {PK_TYPE},
            quiz_id INTEGER,
            chat_id BIGINT,
            user_id BIGINT,
            user_name TEXT,
            score INTEGER,
            correct_count INTEGER,
            wrong_count INTEGER DEFAULT 0,
            total_questions INTEGER DEFAULT 0,
            avg_time_ms INTEGER DEFAULT 0,
            finished_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Per-group welcome/goodbye auto-messages for new/leaving members.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS group_settings (
            group_id BIGINT PRIMARY KEY,
            welcome_enabled INTEGER DEFAULT 1,
            welcome_msg TEXT DEFAULT '',
            goodbye_msg TEXT DEFAULT ''
        )
    """)

    # Named folders a user can file their saved quizzes into (e.g. "Class 10 History",
    # "Weekly Current Affairs") — /newset to create, /sets to browse.
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS quiz_sets (
            id {PK_TYPE},
            owner_id BIGINT,
            name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Small global key→value store for bot-wide settings the owner can tweak without a
    # redeploy — currently just the promo message shown every 5 questions.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bot_settings (
            key TEXT PRIMARY KEY,
            value TEXT DEFAULT ''
        )
    """)

    # Channels a user has linked with /setchannel so the "📤 Send to Channel" button
    # knows where to post — the bot must already be an admin there (a Telegram
    # requirement, not something this bot can work around).
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS user_channels (
            id {PK_TYPE},
            user_id BIGINT,
            channel_id BIGINT,
            channel_title TEXT DEFAULT ''
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS redeem_codes (
            code TEXT PRIMARY KEY,
            bonus_questions INTEGER,
            max_uses INTEGER DEFAULT 0,
            used_count INTEGER DEFAULT 0,
            expires_at TEXT DEFAULT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS redeem_uses (
            code TEXT,
            user_id BIGINT,
            PRIMARY KEY (code, user_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS last_source (
            user_id BIGINT PRIMARY KEY,
            source_type TEXT,
            parts TEXT,
            difficulty TEXT,
            language TEXT,
            count INTEGER
        )
    """)

    # Per-user "bring your own API key" — everyone can save MULTIPLE keys/providers
    # here; everything persists in SQLite so it survives bot restarts. Generation
    # tries them in the order added, then falls back to the owner's shared key.
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS user_api_keys (
            id {PK_TYPE},
            user_id BIGINT,
            provider TEXT,
            api_key TEXT,
            api_base TEXT,
            model TEXT,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Daily call-count tracking per API key (or the owner's shared default key), so
    # the owner gets warned at 50/70/90/100% of that key's daily quota instead of
    # only finding out once every key is dead. `scope` = "owner:<provider>" for the
    # shared key, or "user:<user_id>:<key_id>" for a personal /addapi key. A fresh
    # row per (scope, usage_date) means the counter naturally resets every UTC day —
    # no cron/reset job needed.
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS api_key_usage (
            scope TEXT,
            usage_date TEXT,
            count INTEGER DEFAULT 0,
            last_notified_pct INTEGER DEFAULT 0,
            PRIMARY KEY (scope, usage_date)
        )
    """)

    # Small outbox so a background sync call (generate_mcqs_sync, running off the
    # main event loop via asyncio.to_thread) can queue an owner DM without needing
    # direct bot/async access — a periodic job flushes these to OWNER_ID and clears them.
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS pending_owner_notifications (
            id {PK_TYPE},
            message TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Pending /schedule jobs — lets /myschedules list + cancel them, and lets a
    # restart-orphaned row (the in-memory job itself doesn't survive a restart)
    # get quietly swept once its run_at time has passed.
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS scheduled_quizzes (
            id {PK_TYPE},
            user_id BIGINT,
            chat_id BIGINT,
            run_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 🤺 Duel Mode — one row per challenge, so a scheduled job can look up who was
    # dueling once their live quiz session finishes and post a head-to-head result.
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS duels (
            id {PK_TYPE},
            chat_id BIGINT,
            quiz_id INTEGER,
            challenger_id BIGINT,
            challenger_name TEXT,
            opponent_id BIGINT,
            opponent_name TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ☀️ Groups subscribed to the automatic daily "Quiz of the Day"
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_quiz_subs (
            chat_id BIGINT PRIMARY KEY,
            topic TEXT DEFAULT 'General Knowledge',
            hour_utc INTEGER DEFAULT 9,
            last_sent_date TEXT DEFAULT NULL,
            added_by BIGINT,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 💬 /feedback — simple inbox so users can message the owner without needing
    # the owner's personal contact info
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS feedback (
            id {PK_TYPE},
            user_id BIGINT,
            user_tag TEXT,
            message TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 📢 Owner broadcasts scheduled for a future time (run once, then removed)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS scheduled_broadcasts (
            id {PK_TYPE},
            message TEXT,
            run_at TEXT,
            created_by BIGINT,
            sent INTEGER DEFAULT 0
        )
    """)

    # 🤖 AI MODE — maps every native quiz poll AI Mode sends to its correct answer +
    # origin group, so the single shared PollAnswerHandler can score it even after a
    # bot restart (poll answers can arrive minutes/hours after the poll was sent).
    # Rows older than a week are pruned automatically (see _prune_old_ai_mode_polls).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_mode_polls (
            poll_id TEXT PRIMARY KEY,
            chat_id BIGINT,
            correct_option_id INTEGER,
            subject TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 🤖 AI MODE — running per-group, per-UTC-day score for every user who answers an
    # AI Mode poll (+5 correct / -6 wrong, see AI_MODE_CORRECT_POINTS/WRONG_POINTS).
    # Kept keyed by score_date (not reset/deleted) so the daily leaderboard job always
    # has a clean "today" bucket per group and past days remain available in the DB.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_mode_scores (
            chat_id BIGINT,
            user_id BIGINT,
            user_name TEXT,
            score_date TEXT,
            score INTEGER DEFAULT 0,
            correct_count INTEGER DEFAULT 0,
            wrong_count INTEGER DEFAULT 0,
            PRIMARY KEY (chat_id, user_id, score_date)
        )
    """)

    conn.commit()
    conn.close()


# ==========================================
# 💾 SELF-BACKUP — Telegram itself as free, permanent off-site storage
# ==========================================
# Neither hosting-platform ephemeral disks nor a free DB tier's storage cap are things
# this code can fix for you — but losing data because of them IS fixable: every table
# gets dumped to JSON and DM'd to OWNER_ID on a schedule (and on demand via
# /backupnow). Telegram never deletes messages in your own chat history, so that DM
# becomes a permanent, free backup — restorable any time with /restorebackup, even
# after switching hosts or databases entirely.
BACKUP_TABLES = [
    "users", "filters", "quizzes", "groups", "quiz_results",
    "redeem_codes", "redeem_uses", "last_source", "user_api_keys", "group_settings",
    "quiz_sets", "bot_settings", "user_channels", "scheduled_quizzes",
    "duels", "daily_quiz_subs", "feedback", "scheduled_broadcasts",
]


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


def restore_from_backup(payload: dict) -> dict:
    """INSERT OR REPLACE (via upsert on the primary key) every row from a backup
    file back into the current DB — safe to run against a brand-new empty database
    after switching providers/hosts. Returns a {table: rows_restored} summary.

    Table-by-table and row-by-row: one bad/old row (extra column from a schema
    that's since changed, a stale FK, etc.) is skipped instead of aborting the
    whole restore and leaving the DB connection mid-transaction — that half-open
    state is what used to lock the DB and take every other command down with it.
    """
    key_cols = {
        "users": ["user_id"], "filters": ["group_id", "keyword"], "quizzes": ["id"],
        "groups": ["group_id"], "quiz_results": ["id"], "redeem_codes": ["code"],
        "redeem_uses": ["code", "user_id"], "last_source": ["user_id"], "user_api_keys": ["id"],
        "group_settings": ["group_id"], "quiz_sets": ["id"], "bot_settings": ["key"],
        "user_channels": ["id"], "scheduled_quizzes": ["id"], "duels": ["id"],
        "daily_quiz_subs": ["chat_id"], "feedback": ["id"], "scheduled_broadcasts": ["id"],
    }
    conn = db_connect()
    summary = {}
    try:
        for table, rows in payload.get("tables", {}).items():
            if table not in BACKUP_TABLES or not rows:
                summary[table] = 0
                continue
            keys = key_cols.get(table, [])
            if not keys:
                summary[table] = f"skipped — unknown table, no key mapping"
                continue

            # Only write columns that actually exist in *this* DB's copy of the
            # table — a backup taken from an older/newer schema (a column added
            # or renamed since) must not blow up the whole restore.
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {table} LIMIT 0")
            valid_cols = {d[0] for d in cur.description}

            restored = 0
            skipped = 0
            for row in rows:
                row = {c: val for c, val in row.items() if c in valid_cols}
                k = {c: row[c] for c in keys if c in row}
                v = {c: val for c, val in row.items() if c not in keys}
                if not k:
                    skipped += 1
                    continue
                try:
                    row_cur = conn.cursor()
                    upsert(row_cur, table, k, v)
                    conn.commit()
                    restored += 1
                except Exception as e:
                    conn.rollback()
                    skipped += 1
                    logger.warning(f"Restore: skipped a {table} row ({k}): {e}")
            summary[table] = restored if not skipped else f"{restored} row(s), {skipped} skipped"
    finally:
        conn.close()

    # Restoring rows with their original explicit `id` values (to keep FKs like
    # quiz_id/set_id valid) leaves each SERIAL column's underlying sequence
    # pointing at wherever it was *before* the restore — the next normal INSERT
    # then reuses an id that already exists → "duplicate key value violates
    # unique constraint ..._pkey". Bump every sequence right after restoring.
    sync_serial_sequences()
    return summary
    return summary


async def send_db_backup(bot, note: str = ""):
    if not OWNER_ID:
        return
    buf = build_backup_file()
    try:
        await bot.send_document(
            chat_id=OWNER_ID, document=buf, filename=buf.name,
            caption=f"💾 Auto-backup{(' — ' + note) if note else ''}. Keep this — /restorebackup can reload it into a fresh DB any time.",
        )
    except Exception as e:
        logger.warning(f"Auto-backup DM to owner failed: {e}")


async def periodic_backup_job(context: ContextTypes.DEFAULT_TYPE):
    await send_db_backup(context.bot)


async def report_generation_failure(bot, chat_id: int, user_id: int, error) -> None:
    """User-facing side stays generic on purpose — the real error (provider names,
    key labels, quota states) is internal plumbing that a random user has no reason
    to see, and exposing it is just free recon for anyone probing the bot. The owner
    still gets the full detail by DM so it's actually actionable."""
    await bot.send_message(chat_id, "⚠️ Quiz could not be created due to an error. Please try again in a bit.")
    if OWNER_ID:
        try:
            await bot.send_message(OWNER_ID, f"🔧 Quiz generation failed for user {user_id}:\n{error}")
        except Exception as e:
            logger.warning(f"Failed to DM owner about generation failure: {e}")


async def flush_owner_notifications_job(context: ContextTypes.DEFAULT_TYPE):
    """Delivers any queued API-usage alerts (queued by record_api_key_usage, which
    runs off-thread and has no bot access of its own) to OWNER_ID, then clears them."""
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
# ☀️ /dailyquiz — auto-posts a fresh quiz to a subscribed group every day at a
# chosen UTC hour. Checked hourly; a `last_sent_date` guard means it only actually
# fires once per UTC day per group even though the check itself runs every hour.
# ==========================================
async def dailyquiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type not in ("group", "supergroup"):
        await update.message.reply_text("☀️ /dailyquiz sirf group me chalta hai.")
        return
    if not await is_chat_admin(update, context):
        await update.message.reply_text("⛔ Sirf group admins ye set kar sakte hain.")
        return
    if not context.args:
        await update.message.reply_text(
            "Usage:\n"
            "`/dailyquiz on <topic>` — daily quiz on (default 9 AM UTC)\n"
            "`/dailyquiz off` — daily quiz band karo\n"
            "`/dailyquiz hour <0-23>` — UTC hour badlo",
            parse_mode="Markdown",
        )
        return
    sub = context.args[0].lower()
    chat_id = update.effective_chat.id
    conn = db_connect()
    cur = conn.cursor()
    if sub == "off":
        cur.execute("DELETE FROM daily_quiz_subs WHERE chat_id=?", (chat_id,))
        conn.commit()
        conn.close()
        await update.message.reply_text("☀️ Daily quiz band kar diya is group ke liye.")
        return
    if sub == "hour":
        if len(context.args) < 2 or not context.args[1].isdigit() or not (0 <= int(context.args[1]) <= 23):
            conn.close()
            await update.message.reply_text("Usage: `/dailyquiz hour <0-23>` (UTC hour)", parse_mode="Markdown")
            return
        cur.execute("UPDATE daily_quiz_subs SET hour_utc=? WHERE chat_id=?", (int(context.args[1]), chat_id))
        conn.commit()
        changed = cur.rowcount
        conn.close()
        if changed:
            await update.message.reply_text(f"☀️ Daily quiz ab {context.args[1]}:00 UTC par aayega.")
        else:
            await update.message.reply_text("☀️ Pehle `/dailyquiz on <topic>` se subscribe karo.")
        return
    if sub == "on":
        topic = " ".join(context.args[1:]).strip() or "General Knowledge"
        upsert(cur, "daily_quiz_subs", {"chat_id": chat_id},
               {"topic": topic[:100], "added_by": update.effective_user.id})
        conn.commit()
        conn.close()
        await update.message.reply_text(
            f"☀️ Daily quiz ON — topic: **{topic}**, roz 9:00 UTC par (badlo `/dailyquiz hour <0-23>` se).",
            parse_mode="Markdown",
        )
        return
    conn.close()
    await update.message.reply_text("Usage: `/dailyquiz on <topic>` / `off` / `hour <0-23>`", parse_mode="Markdown")


async def daily_quiz_check_job(context: ContextTypes.DEFAULT_TYPE):
    """Runs every hour; posts to any subscribed group whose chosen hour matches
    right now (UTC) and hasn't already been sent today."""
    now = datetime.utcnow()
    today = now.strftime("%Y-%m-%d")
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT chat_id, topic, added_by FROM daily_quiz_subs WHERE hour_utc=? AND "
                "(last_sent_date IS NULL OR last_sent_date != ?)", (now.hour, today))
    due = cur.fetchall()
    conn.close()
    for chat_id, topic, added_by in due:
        try:
            questions = await generate_mcqs_async(added_by or OWNER_ID, [{"type": "text", "text": topic}],
                                                    10, "medium", "English")
            quiz_id = save_quiz(added_by or OWNER_ID, f"Quiz of the Day: {topic}"[:60], questions,
                                 timer_seconds=20, quiz_type="daily", creator_name="Daily Quiz")
            await context.bot.send_message(chat_id, f"☀️ **Quiz of the Day** — {topic}", parse_mode="Markdown")
            await initiate_live_quiz(context.bot, context.application, chat_id, "group", quiz_id,
                                      added_by or OWNER_ID, "Quiz of the Day")
        except Exception as e:
            logger.warning(f"Daily quiz failed for chat {chat_id}: {e}")
            continue
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("UPDATE daily_quiz_subs SET last_sent_date=? WHERE chat_id=?", (today, chat_id))
        conn.commit()
        conn.close()


# ==========================================
# 🤖 AI MODE — owner switch that, once ON, auto-generates AI quiz polls spanning
# every subject from school up to graduate level and blasts them as native
# (non-anonymous) quiz polls straight into EVERY tracked/synced group, then
# auto-posts a full leaderboard (every participant's id + name, no matter how many)
# to every group once a day. Scoring here is its own rule, separate from the rest
# of the bot: ✅ correct = +5, ❌ wrong = -6 (AI_MODE_CORRECT_POINTS/WRONG_POINTS).
#
# Deliberately built as its own small, DB-backed system (its own 2 tables — see
# init_db — plus a couple of lines added to the existing shared PollAnswerHandler)
# instead of reusing the lobby-based live-quiz engine above: that engine waits for
# 2 "I Am Ready" taps per group before firing a single question, which is the
# opposite of what an automatic, no-friction, every-group broadcast needs.
# ==========================================
AI_MODE_CORRECT_POINTS = 5
AI_MODE_WRONG_POINTS = -6
AI_MODE_DEFAULT_INTERVAL_HOURS = 24
AI_MODE_DEFAULT_BATCH_SIZE = 5
AI_MODE_DEFAULT_LEADERBOARD_HOUR = 20  # UTC

# One of these is picked at random every cycle — spans school-board subjects up to
# graduate/undergraduate ones, so over many cycles the full "school se graduate tak"
# range actually gets covered rather than any single subject dominating. Edit this
# list directly to add, remove, or re-weight (repeat an entry to weight it higher).
AI_MODE_SUBJECTS = [
    "Mathematics (school level)", "Physics (school level)", "Chemistry (school level)",
    "Biology (school level)", "History", "Geography", "Political Science & Civics",
    "Economics", "English Language & Literature", "Computer Science", "Environmental Science",
    "General Knowledge", "Current Affairs", "Logical Reasoning & Aptitude",
    "Indian Constitution & Polity", "World History", "Static GK",
    "Engineering Mathematics (graduate level)", "Data Structures & Algorithms (graduate level)",
    "Database Management Systems (graduate level)", "Operating Systems (graduate level)",
    "Computer Networks (graduate level)", "Organic Chemistry (graduate level)",
    "Human Physiology (graduate level)", "Microeconomics (graduate level)",
    "Macroeconomics (graduate level)", "Business Studies & Management (graduate level)",
    "Statistics (graduate level)", "Accountancy & Financial Accounting (graduate level)",
    "Psychology (graduate level)", "Sociology (graduate level)",
]


def get_ai_mode_settings() -> dict:
    return {
        "enabled": get_bot_setting("ai_mode_enabled", "0") == "1",
        "interval_hours": int(get_bot_setting("ai_mode_interval_hours", "") or AI_MODE_DEFAULT_INTERVAL_HOURS),
        "batch_size": int(get_bot_setting("ai_mode_batch_size", "") or AI_MODE_DEFAULT_BATCH_SIZE),
        "leaderboard_hour": int(get_bot_setting("ai_mode_leaderboard_hour", "") or AI_MODE_DEFAULT_LEADERBOARD_HOUR),
    }


async def aimode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return
    s = get_ai_mode_settings()
    if not context.args:
        state = "🟢 ON" if s["enabled"] else "🔴 OFF"
        conn = db_connect(); cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM groups")
        group_count = cur.fetchone()[0]
        conn.close()
        await update.message.reply_text(
            f"🤖 **{stylize('AI Mode')}** — {state}\n\n"
            f"• Har {s['interval_hours']}h me {s['batch_size']} naye AI polls (School se Graduate "
            f"tak, sabhi subjects) — {group_count} synced group me\n"
            f"• Daily leaderboard {s['leaderboard_hour']}:00 UTC par har group me (sabhi participants — "
            f"jitne bhi ho, sabka id + name)\n"
            f"• Scoring: ✅ +{AI_MODE_CORRECT_POINTS} · ❌ {AI_MODE_WRONG_POINTS}\n\n"
            "Usage:\n"
            "`/aimode on [interval_hours] [questions_per_batch]` — on karo (default 24h, 5 Q)\n"
            "`/aimode off` — band karo\n"
            "`/aimode hour <0-23>` — leaderboard UTC hour set karo",
            parse_mode="Markdown",
        )
        return
    sub = context.args[0].lower()
    if sub == "off":
        set_bot_setting("ai_mode_enabled", "0")
        await update.message.reply_text("🤖 AI Mode band kar diya — naye polls/leaderboard rukk jayenge.")
        return
    if sub == "on":
        interval = AI_MODE_DEFAULT_INTERVAL_HOURS
        batch = AI_MODE_DEFAULT_BATCH_SIZE
        if len(context.args) >= 2:
            try:
                interval = max(1, int(context.args[1]))
            except ValueError:
                await update.message.reply_text("❌ interval_hours ek number hona chahiye.")
                return
        if len(context.args) >= 3:
            try:
                batch = max(1, min(20, int(context.args[2])))
            except ValueError:
                await update.message.reply_text("❌ questions_per_batch ek number hona chahiye.")
                return
        set_bot_setting("ai_mode_enabled", "1")
        set_bot_setting("ai_mode_interval_hours", str(interval))
        set_bot_setting("ai_mode_batch_size", str(batch))
        set_bot_setting("ai_mode_last_sent_at", "")  # first batch fires on the very next check
        conn = db_connect(); cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM groups")
        group_count = cur.fetchone()[0]
        conn.close()
        await update.message.reply_text(
            f"🤖 **AI Mode ON!**\n\n"
            f"• Har {interval}h me {batch} naye AI-generated polls — {group_count} synced group me "
            f"jaayenge (pehla batch ~15 min ke andar)\n"
            f"• Roz {get_ai_mode_settings()['leaderboard_hour']}:00 UTC par har group me automatic "
            f"leaderboard — jitne bhi users ne answer diya, sabka id + name\n"
            f"• Scoring: ek sahi par +{AI_MODE_CORRECT_POINTS}, galat par {AI_MODE_WRONG_POINTS}\n\n"
            "`/aimode hour <0-23>` se leaderboard ka time badal sakte ho.",
            parse_mode="Markdown",
        )
        return
    if sub == "hour":
        if len(context.args) < 2 or not context.args[1].isdigit() or not (0 <= int(context.args[1]) <= 23):
            await update.message.reply_text("Usage: `/aimode hour <0-23>` (UTC)", parse_mode="Markdown")
            return
        set_bot_setting("ai_mode_leaderboard_hour", context.args[1])
        await update.message.reply_text(f"🤖 AI Mode leaderboard ab roz {context.args[1]}:00 UTC par post hoga.")
        return
    await update.message.reply_text(
        "Usage: `/aimode on [interval_hours] [questions_per_batch]` / `off` / `hour <0-23>`",
        parse_mode="Markdown",
    )


def _prune_old_ai_mode_polls():
    cutoff = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM ai_mode_polls WHERE created_at < ?", (cutoff,))
    conn.commit()
    conn.close()


async def ai_mode_quiz_check_job(context: ContextTypes.DEFAULT_TYPE):
    """Runs every ~15 min. When AI Mode is ON and `interval_hours` has elapsed since
    the last batch, generates a fresh AI quiz on ONE randomly-picked subject (see
    AI_MODE_SUBJECTS) and fires it as native quiz polls into every tracked group —
    no lobby, no "ready" wait, straight broadcast (like /quizblast but automatic,
    AI-written, and recurring)."""
    if get_bot_setting("ai_mode_enabled", "0") != "1":
        return
    s = get_ai_mode_settings()
    now = datetime.utcnow()
    last_sent = get_bot_setting("ai_mode_last_sent_at", "")
    if last_sent:
        try:
            if now - datetime.fromisoformat(last_sent) < timedelta(hours=s["interval_hours"]):
                return
        except ValueError:
            pass

    subject = random.choice(AI_MODE_SUBJECTS)
    try:
        questions = await generate_mcqs_async(
            OWNER_ID, [{"type": "text", "text": subject}], s["batch_size"],
            "school-to-graduate level (mix of school-board and undergraduate/graduate difficulty)",
            "English",
        )
    except Exception as e:
        logger.warning(f"AI Mode generation failed for subject '{subject}': {e}")
        return

    # Mark "sent" now, before per-group delivery — so a few unreachable/kicked-from
    # groups below can never stall every future cycle retrying the same batch.
    set_bot_setting("ai_mode_last_sent_at", now.isoformat())
    try:
        save_quiz(OWNER_ID, f"AI Mode: {subject}"[:60], questions, quiz_type="ai_mode", creator_name="AI Mode")
    except Exception as e:
        logger.warning(f"AI Mode save_quiz failed (non-fatal, polls still send): {e}")

    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT group_id FROM groups")
    group_ids = [r[0] for r in cur.fetchall()]
    conn.close()

    sent_groups = 0
    for gid in group_ids:
        try:
            await context.bot.send_message(gid, f"🤖 **AI Mode Quiz** — {subject}", parse_mode="Markdown")
        except Exception as e:
            logger.warning(f"AI Mode intro message failed for group {gid} (skipped, others unaffected): {e}")
            continue
        for q in questions:
            try:
                msg = await context.bot.send_poll(
                    chat_id=gid, question=q["question"][:300],
                    options=[o[:100] for o in q["options"]],
                    type=Poll.QUIZ, correct_option_id=q["correct_index"],
                    explanation=(q.get("explanation", "")[:200] or None),
                    is_anonymous=False,
                )
                conn = db_connect()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO ai_mode_polls (poll_id, chat_id, correct_option_id, subject) VALUES (?, ?, ?, ?)",
                    (msg.poll.id, gid, q["correct_index"], subject),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"AI Mode poll send failed for group {gid} (skipped, others unaffected): {e}")
            await asyncio.sleep(0.05)
        sent_groups += 1
    logger.info(f"AI Mode: sent {len(questions)} poll(s) on '{subject}' to {sent_groups}/{len(group_ids)} group(s).")
    _prune_old_ai_mode_polls()


def record_ai_mode_answer(chat_id: int, user_id: int, user_name: str, correct: bool):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    delta = AI_MODE_CORRECT_POINTS if correct else AI_MODE_WRONG_POINTS
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT score, correct_count, wrong_count FROM ai_mode_scores WHERE chat_id=? AND user_id=? AND score_date=?",
        (chat_id, user_id, today),
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE ai_mode_scores SET score=?, correct_count=?, wrong_count=?, user_name=? "
            "WHERE chat_id=? AND user_id=? AND score_date=?",
            (row[0] + delta, row[1] + (1 if correct else 0), row[2] + (0 if correct else 1),
             user_name[:60], chat_id, user_id, today),
        )
    else:
        cur.execute(
            "INSERT INTO ai_mode_scores (chat_id, user_id, user_name, score_date, score, correct_count, wrong_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, user_id, user_name[:60], today, delta, 1 if correct else 0, 0 if correct else 1),
        )
    conn.commit()
    conn.close()


def build_ai_mode_leaderboard_chunks(chat_id: int, score_date: str):
    """Every participant for that group+day — id + name + score, sorted best-first —
    deliberately NOT capped at a top-N like /leaderboard, since AI Mode's whole point
    is showing everyone who played, chahe kitne bhi ho. Returns None if nobody
    answered that day; otherwise a list of message chunks kept under Telegram's
    ~4096-char limit, so a very active group with hundreds of participants still
    gets a complete list — split across a few messages instead of dropping anyone."""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT user_id, user_name, score, correct_count, wrong_count FROM ai_mode_scores "
        "WHERE chat_id=? AND score_date=? ORDER BY score DESC",
        (chat_id, score_date),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return None

    medals = ["🥇", "🥈", "🥉"]
    header = (
        f"🤖 **{stylize('AI Mode — Daily Leaderboard')}**\n"
        f"📅 {score_date} UTC · ✅ +{AI_MODE_CORRECT_POINTS} · ❌ {AI_MODE_WRONG_POINTS} · "
        f"👥 {len(rows)} participant(s)\n\n"
    )
    chunks, current = [], header
    for i, (user_id, name, score, correct, wrong) in enumerate(rows):
        rank_tag = medals[i] if i < 3 else f"{i + 1}."
        line = f"{rank_tag} {name or 'Player'} — `{user_id}` — {score} pts (✅{correct} ❌{wrong})\n"
        if len(current) + len(line) > 3800:
            chunks.append(current)
            current = line
        else:
            current += line
    chunks.append(current)
    return chunks


async def ai_mode_leaderboard_job(context: ContextTypes.DEFAULT_TYPE):
    """Runs hourly; once a day, at the configured UTC hour, posts the full AI Mode
    leaderboard (every participant — see build_ai_mode_leaderboard_chunks) to every
    tracked group. Same 'checked often, guarded by a last-posted-date' pattern as
    daily_quiz_check_job above, so it fires exactly once per UTC day."""
    if get_bot_setting("ai_mode_enabled", "0") != "1":
        return
    s = get_ai_mode_settings()
    now = datetime.utcnow()
    if now.hour != s["leaderboard_hour"]:
        return
    today = now.strftime("%Y-%m-%d")
    if get_bot_setting("ai_mode_last_leaderboard_date", "") == today:
        return

    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT group_id FROM groups")
    group_ids = [r[0] for r in cur.fetchall()]
    conn.close()

    posted = 0
    for gid in group_ids:
        chunks = build_ai_mode_leaderboard_chunks(gid, today)
        if not chunks:
            continue
        try:
            for chunk in chunks:
                await context.bot.send_message(gid, chunk, parse_mode="Markdown")
                await asyncio.sleep(0.1)
            posted += 1
        except Exception as e:
            logger.warning(f"AI Mode leaderboard post failed for group {gid} (skipped, others unaffected): {e}")
        await asyncio.sleep(0.05)
    set_bot_setting("ai_mode_last_leaderboard_date", today)
    logger.info(f"AI Mode daily leaderboard posted to {posted}/{len(group_ids)} group(s) for {today}.")


# ==========================================
# 📢 /schedulebroadcast — owner sends a plain-text message to every user/group at a
# chosen future UTC time (durable across restarts — stored in DB, checked every minute).
# ==========================================
async def schedulebroadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return
    if len(context.args) < 3:
        await update.message.reply_text(
            "Usage: `/schedulebroadcast YYYY-MM-DD HH:MM <message>` (UTC time)\n"
            "Example: `/schedulebroadcast 2026-09-01 09:00 New feature is live!`",
            parse_mode="Markdown",
        )
        return
    date_str, time_str = context.args[0], context.args[1]
    message = " ".join(context.args[2:])
    try:
        run_at = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        await update.message.reply_text("❌ Date/time format samajh nahi aaya — `YYYY-MM-DD HH:MM` use karo (UTC).")
        return
    if run_at <= datetime.utcnow():
        await update.message.reply_text("❌ Ye time already beet chuka hai (UTC) — future time do.")
        return
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO scheduled_broadcasts (message, run_at, created_by) VALUES (?, ?, ?)",
                (message, run_at.isoformat(), update.effective_user.id))
    conn.commit()
    conn.close()
    await update.message.reply_text(f"📢 Scheduled for {run_at.isoformat()} UTC:\n\n{message}")


async def scheduled_broadcast_check_job(context: ContextTypes.DEFAULT_TYPE):
    now_iso = datetime.utcnow().isoformat()
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT id, message FROM scheduled_broadcasts WHERE sent=0 AND run_at<=?", (now_iso,))
    due = cur.fetchall()
    conn.close()
    for bid, message in due:
        try:
            ok, fail = await broadcast_to_all(context.bot, message)
            logger.info(f"Scheduled broadcast #{bid} sent: {ok} ok, {fail} failed.")
        except Exception as e:
            logger.warning(f"Scheduled broadcast #{bid} failed: {e}")
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("UPDATE scheduled_broadcasts SET sent=1 WHERE id=?", (bid,))
        conn.commit()
        conn.close()


# ==========================================
# 💬 /feedback — lets any user message the owner without needing their contact info
# ==========================================
async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/feedback <your message>` — seedha bot owner ko chala jayega.", parse_mode="Markdown")
        return
    message = " ".join(context.args)[:1000]
    user = update.effective_user
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback (user_id, user_tag, message) VALUES (?, ?, ?)",
                (user.id, user.username or user.first_name or "Unknown", message))
    conn.commit()
    conn.close()
    await update.message.reply_text("✅ Bhej diya — dhanyavaad! Owner tak pahunch gaya.")
    if OWNER_ID:
        try:
            await context.bot.send_message(
                OWNER_ID,
                f"💬 **Feedback from** {user.first_name} (`{user.id}`, @{user.username or 'no_username'}):\n\n{message}",
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.warning(f"Failed to forward feedback to owner: {e}")


# ==========================================
async def backupnow_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return
    await update.message.reply_text("💾 Building backup…")
    await send_db_backup(context.bot, note="manual /backupnow")


async def restorebackup_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner only. Reply to a backup .json file (the one the bot DMs you) with
    /restorebackup to reload every table from it — the fix after switching hosts/DBs."""
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return
    doc = None
    if update.message.reply_to_message and update.message.reply_to_message.document:
        doc = update.message.reply_to_message.document
    elif update.message.document:
        doc = update.message.document
    if not doc:
        await update.message.reply_text("Reply to a backup .json file with /restorebackup (or attach one with the command as the caption).")
        return
    try:
        tg_file = await doc.get_file()
        raw = await tg_file.download_as_bytearray()
        payload = json.loads(bytes(raw).decode("utf-8"))
        summary = restore_from_backup(payload)
        lines = "\n".join(f"• {t}: {n if isinstance(n, str) else f'{n} row(s)'}" for t, n in summary.items())
        await update.message.reply_text(f"✅ Restore complete:\n{lines}")
    except Exception as e:
        logger.warning(f"Restore failed: {e}")
        await update.message.reply_text(f"❌ Restore failed — is that a valid backup file? ({e})")


def run_migrations():
    """Adds columns introduced after the first release to any existing DB."""
    conn = db_connect()
    cursor = conn.cursor()

    new_columns = (
        ("users", "tag", "TEXT DEFAULT ''"),
        ("users", "description_tag", "TEXT DEFAULT ''"),
        ("users", "default_difficulty", "TEXT DEFAULT ''"),
        ("users", "default_language", "TEXT DEFAULT ''"),
        ("users", "daily_ai_used", "INTEGER DEFAULT 0"),
        ("users", "daily_ai_date", "TEXT DEFAULT ''"),
        ("users", "ai_unlimited", "INTEGER DEFAULT 0"),
        ("users", "option_style", "TEXT DEFAULT 'plain'"),
        ("quizzes", "timer_seconds", "INTEGER DEFAULT 0"),
        ("quizzes", "quiz_type", "TEXT DEFAULT 'quick'"),
        ("quizzes", "creator_name", "TEXT DEFAULT ''"),
        ("quizzes", "description", "TEXT DEFAULT ''"),
        ("quizzes", "set_id", "INTEGER DEFAULT NULL"),
        ("quizzes", "is_public", "INTEGER DEFAULT 0"),
        ("scheduled_quizzes", "quiz_id", "INTEGER"),
        ("scheduled_quizzes", "interval_minutes", "INTEGER"),
        ("scheduled_quizzes", "next_run", "TEXT"),
        ("scheduled_quizzes", "created_by", "BIGINT"),
        ("redeem_codes", "expires_at", "TEXT DEFAULT NULL"),
        ("quiz_results", "wrong_count", "INTEGER DEFAULT 0"),
        ("quiz_results", "total_questions", "INTEGER DEFAULT 0"),
        ("quiz_results", "avg_time_ms", "INTEGER DEFAULT 0"),
        ("users", "streak_count", "INTEGER DEFAULT 0"),
        ("users", "last_quiz_date", "TEXT DEFAULT NULL"),
    )
    for table, col, coltype in new_columns:
        if USE_POSTGRES:
            # Postgres supports IF NOT EXISTS directly — no error, no aborted transaction.
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {coltype}")
        else:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
            except sqlite3.OperationalError:
                pass  # column already exists

    # A DB created before this fix has these columns as 32-bit INTEGER on Postgres
    # (fine on SQLite, which never enforced a width) — widen them in place so
    # already-deployed Supabase/Render Postgres databases stop rejecting today's
    # Telegram IDs, which now regularly exceed the 32-bit ~2.1B ceiling.
    if USE_POSTGRES:
        bigint_columns = [
            ("users", "user_id"), ("users", "referred_by"), ("filters", "group_id"),
            ("quizzes", "user_id"), ("groups", "group_id"), ("quiz_results", "chat_id"),
            ("quiz_results", "user_id"), ("group_settings", "group_id"), ("quiz_sets", "owner_id"),
            ("user_channels", "user_id"), ("user_channels", "channel_id"), ("redeem_uses", "user_id"),
            ("last_source", "user_id"), ("user_api_keys", "user_id"), ("scheduled_quizzes", "user_id"),
            ("scheduled_quizzes", "chat_id"), ("scheduled_quizzes", "created_by"), ("duels", "chat_id"),
            ("duels", "challenger_id"), ("duels", "opponent_id"), ("daily_quiz_subs", "chat_id"),
            ("daily_quiz_subs", "added_by"), ("feedback", "user_id"), ("scheduled_broadcasts", "created_by"),
        ]
        for table, col in bigint_columns:
            try:
                cursor.execute(f"ALTER TABLE {table} ALTER COLUMN {col} TYPE BIGINT")
                conn.commit()
            except Exception as e:
                logger.warning(f"BIGINT migration skipped for {table}.{col}: {e}")
                conn.rollback()

    # Migrate user_api_keys from an earlier one-key-per-user schema (PRIMARY KEY
    # user_id) to the current multi-key schema — only relevant for pre-existing
    # SQLite files; a fresh Postgres DB always starts on the current schema.
    if not USE_POSTGRES:
        try:
            cursor.execute("PRAGMA table_info(user_api_keys)")
            cols = [row[1] for row in cursor.fetchall()]
            if cols and "id" not in cols:
                cursor.execute("ALTER TABLE user_api_keys RENAME TO user_api_keys_old")
                cursor.execute("""
                    CREATE TABLE user_api_keys (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        provider TEXT,
                        api_key TEXT,
                        api_base TEXT,
                        model TEXT,
                        added_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    INSERT INTO user_api_keys (user_id, provider, api_key, api_base, model)
                    SELECT user_id, provider, api_key, api_base, model FROM user_api_keys_old
                """)
                cursor.execute("DROP TABLE user_api_keys_old")
        except sqlite3.OperationalError:
            pass  # table doesn't exist yet — init_db() will create the current schema

    conn.commit()
    conn.close()


def sync_serial_sequences():
    """Bumps every SERIAL-PK sequence to max(id)+1. Needed once after any restore
    that inserted rows with explicit id values (see restore_from_backup) — safe to
    run on every startup regardless, since it's a no-op when the sequence is
    already ahead of max(id)."""
    if not USE_POSTGRES:
        return
    conn = db_connect()
    try:
        for table in ("quizzes", "quiz_results", "user_channels", "scheduled_quizzes",
                       "duels", "feedback", "scheduled_broadcasts", "user_api_keys", "quiz_sets"):
            try:
                cur = conn.cursor()
                cur.execute(
                    f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM {table}), 1))"
                )
                conn.commit()
            except Exception as e:
                logger.warning(f"Sequence sync skipped for {table}: {e}")
                conn.rollback()
    finally:
        conn.close()


# init_db()/run_migrations() used to run here at import time — outside
# run_forever()'s retry loop. Any transient DB hiccup on startup (Supabase's
# pooler briefly refusing a connection, a DNS blip, etc.) killed the whole
# process instantly instead of getting the same backoff-and-retry every other
# failure gets, which is what caused the crash-restart cycles / not staying up
# 24x7. They now run inside main() (see below), guarded by run_forever().

# ==========================================
# 🔐 DATABASE HELPERS
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
            "INSERT INTO users (user_id, questions_used, question_limit, is_verified) VALUES (?, 0, 100, 0)",
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
        "user_id": user[0],
        "questions_used": user[1],
        "question_limit": user[2],
        "is_verified": user[3],
        "tag": user[4] or '',
        "description_tag": user[5] or '',
        "default_difficulty": user[6] or '',
        "default_language": user[7] or '',
        "daily_ai_used": user[8] or 0,
        "daily_ai_date": user[9] or '',
        "ai_unlimited": bool(user[10]),
        "option_style": user[11] or 'plain',
    }


def set_verified(user_id: int, status: int = 1):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET is_verified = ? WHERE user_id = ?", (status, user_id))
    conn.commit()
    conn.close()


def set_user_field(user_id: int, field: str, value):
    allowed = {"tag", "description_tag", "default_difficulty", "default_language", "referred_by", "option_style"}
    if field not in allowed:
        raise ValueError(f"field '{field}' is not allowlisted for update")
    get_user(user_id)  # ensure row exists
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(f"UPDATE users SET {field} = ? WHERE user_id = ?", (value, user_id))
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


def increment_question_limit(user_id: int, n: int):
    get_user(user_id)
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET question_limit = question_limit + ? WHERE user_id = ?", (n, user_id))
    conn.commit()
    conn.close()


# ==========================================
# 📅 DAILY AI GENERATION QUOTA (separate from the lifetime question_limit above)
# 200 AI-generated questions/day per user, across every AI method combined
# (topic/pdf/image/section/voicequiz/importchat/regenerate) — CSV import
# is exact-from-file, not AI, so it's never counted here. Resets automatically at UTC
# midnight. The owner is always unlimited; anyone else can be marked unlimited too
# (see set_ai_unlimited / /freeuser) — for friends etc.
# ==========================================
DAILY_AI_QUESTION_LIMIT = 300


def get_daily_ai_remaining(user_id: int):
    """None = unlimited (owner, or owner-granted). Otherwise the number of AI
    questions this user can still generate today."""
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
    """Call AFTER generation with the actual number sent (not the number requested) —
    mirrors how increment_questions_used() is already used elsewhere in this file."""
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


def set_ai_unlimited(user_id: int, unlimited: bool):
    get_user(user_id)  # ensure row exists
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET ai_unlimited=? WHERE user_id=?", (1 if unlimited else 0, user_id))
    conn.commit()
    conn.close()


# ==========================================
# ⚙️ BOT-WIDE SETTINGS (owner-tweakable without a redeploy) + OPTION-LABEL STYLE
# ==========================================
def get_bot_setting(key: str, default: str = '') -> str:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT value FROM bot_settings WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] is not None else default


def set_bot_setting(key: str, value: str):
    conn = db_connect()
    cur = conn.cursor()
    upsert(cur, "bot_settings", {"key": key}, {"value": value})
    conn.commit()
    conn.close()


PROMO_EVERY_N_QUESTIONS = 5
OPTION_STYLE_LABELS = {"plain": "Plain (no prefix)", "abcd": "A) B) C) D)", "numeric": "1) 2) 3) 4)"}


def apply_option_style(options: list, style: str) -> list:
    style = style or 'plain'
    if style == 'abcd':
        return [f"{chr(65 + i)}) {o}"[:100] for i, o in enumerate(options)]
    if style == 'numeric':
        return [f"{i + 1}) {o}"[:100] for i, o in enumerate(options)]
    return [o[:100] for o in options]


async def send_promo_message(bot, chat_id):
    promo = get_bot_setting("promo_text", "").strip()
    if not promo:
        return
    try:
        await bot.send_message(chat_id, promo)
    except Exception as e:
        logger.warning(f"Promo message send failed (non-fatal, quiz continues): {e}")


# ==========================================
# 📁 QUIZ SETS — named folders for organizing saved quizzes (e.g. by subject)
# ==========================================
def create_quiz_set(owner_id: int, name: str) -> int:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO quiz_sets (owner_id, name) VALUES (?, ?)", (owner_id, name[:60]))
    conn.commit()
    set_id = last_insert_id(cur, conn, "quiz_sets")
    conn.close()
    return set_id


def get_user_sets(owner_id: int) -> list:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM quiz_sets WHERE owner_id=? ORDER BY id DESC", (owner_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1]} for r in rows]


def get_set_meta(set_id: int):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT id, owner_id, name FROM quiz_sets WHERE id=?", (set_id,))
    row = cur.fetchone()
    conn.close()
    return {"id": row[0], "owner_id": row[1], "name": row[2]} if row else None


def get_quizzes_in_set(set_id: int) -> list:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT id, title, quiz_type FROM quizzes WHERE set_id=? ORDER BY id DESC", (set_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "quiz_type": r[2]} for r in rows]


def assign_quiz_to_set(quiz_id: int, set_id):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE quizzes SET set_id=? WHERE id=?", (set_id, quiz_id))
    conn.commit()
    conn.close()


def delete_quiz_set(set_id: int, owner_id: int) -> bool:
    """Deletes the set itself; quizzes that were filed under it are kept, just
    unfiled (set_id reset to NULL) — deleting a folder was never asked to also
    delete everything inside it."""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT owner_id FROM quiz_sets WHERE id=?", (set_id,))
    row = cur.fetchone()
    if not row or row[0] != owner_id:
        conn.close()
        return False
    cur.execute("UPDATE quizzes SET set_id=NULL WHERE set_id=?", (set_id,))
    cur.execute("DELETE FROM quiz_sets WHERE id=?", (set_id,))
    conn.commit()
    conn.close()
    return True


# ==========================================
# 📤 LINKED CHANNELS — for the "Send to Channel" button (bot must already be admin there)
# ==========================================
def add_user_channel(user_id: int, channel_id: int, channel_title: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT id FROM user_channels WHERE user_id=? AND channel_id=?", (user_id, channel_id))
    if cur.fetchone():
        conn.close()
        return
    cur.execute(
        "INSERT INTO user_channels (user_id, channel_id, channel_title) VALUES (?, ?, ?)",
        (user_id, channel_id, channel_title[:100])
    )
    conn.commit()
    conn.close()


def get_user_channels(user_id: int) -> list:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT id, channel_id, channel_title FROM user_channels WHERE user_id=?", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "channel_id": r[1], "channel_title": r[2] or str(r[1])} for r in rows]


def remove_user_channel(user_id: int, row_id: int) -> bool:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM user_channels WHERE id=? AND user_id=?", (row_id, user_id))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def save_quiz(user_id: int, title: str, questions: list, timer_seconds: int = 0,
              quiz_type: str = "quick", creator_name: str = "", description: str = "", set_id=None) -> int:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO quizzes (user_id, title, data, timer_seconds, quiz_type, creator_name, description, set_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, (title or "Quiz")[:60], json.dumps(questions), timer_seconds, quiz_type,
             creator_name[:60], description[:200], set_id)
        )
        conn.commit()
        quiz_id = last_insert_id(cur, conn, "quizzes")
        return quiz_id
    finally:
        conn.close()


def get_quiz_meta(quiz_id: int):
    """Full quiz row including timer/type/creator — used by the Official Quiz flow."""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, user_id, title, data, timer_seconds, quiz_type, creator_name, description, set_id, is_public "
        "FROM quizzes WHERE id=?",
        (quiz_id,)
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0], "user_id": row[1], "title": row[2], "questions": json.loads(row[3]),
        "timer_seconds": row[4] or 0, "quiz_type": row[5] or "quick", "creator_name": row[6] or "",
        "description": row[7] or "", "set_id": row[8], "is_public": bool(row[9]),
    }


def update_quiz_description(quiz_id: int, description: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE quizzes SET description=? WHERE id=?", (description[:200], quiz_id))
    conn.commit()
    conn.close()


def update_quiz_title(quiz_id: int, title: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE quizzes SET title=? WHERE id=?", (title[:60], quiz_id))
    conn.commit()
    conn.close()


def update_quiz_questions(quiz_id: int, questions: list):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE quizzes SET data=? WHERE id=?", (json.dumps(questions), quiz_id))
    conn.commit()
    conn.close()


def delete_quiz(quiz_id: int, user_id: int) -> bool:
    """Deletes a quiz, but only if it belongs to user_id — returns whether it deleted anything."""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM quizzes WHERE id=? AND user_id=?", (quiz_id, user_id))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def toggle_quiz_public(quiz_id: int, user_id: int):
    """Flips is_public for a quiz the caller owns (or the bot owner, for support).
    Returns the new bool value, or None if the quiz doesn't exist / isn't theirs."""
    quiz = get_quiz_meta(quiz_id)
    if not quiz or (quiz["user_id"] != user_id and user_id != OWNER_ID):
        return None
    new_value = 0 if quiz["is_public"] else 1
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE quizzes SET is_public=? WHERE id=?", (new_value, quiz_id))
    conn.commit()
    conn.close()
    return bool(new_value)


def dedup_quiz_questions(quiz_id: int, user_id: int):
    """Removes duplicate questions (by question text) within a single quiz the caller
    owns. Returns the number removed, or None if not allowed / not found."""
    quiz = get_quiz_meta(quiz_id)
    if not quiz or (quiz["user_id"] != user_id and user_id != OWNER_ID):
        return None
    seen = set()
    cleaned = []
    for q in quiz["questions"]:
        key = (q.get("question") or "").strip().lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(q)
    removed = len(quiz["questions"]) - len(cleaned)
    if removed:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("UPDATE quizzes SET data=? WHERE id=?", (json.dumps(cleaned), quiz_id))
        conn.commit()
        conn.close()
    return removed


def get_public_quizzes() -> list:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, creator_name, timer_seconds, description FROM quizzes "
        "WHERE is_public=1 ORDER BY id DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "title": r[1], "creator_name": r[2] or "Unknown", "timer_seconds": r[3] or 0,
         "description": r[4] or ""}
        for r in rows
    ]


def group_is_tracked(group_id: int) -> bool:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM groups WHERE group_id=?", (group_id,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def save_quiz_result(quiz_id: int, chat_id: int, user_id: int, user_name: str, score: int, correct_count: int,
                      wrong_count: int = 0, total_questions: int = 0, avg_time_ms: int = 0):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO quiz_results (quiz_id, chat_id, user_id, user_name, score, correct_count, "
        "wrong_count, total_questions, avg_time_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (quiz_id, chat_id, user_id, user_name[:60], score, correct_count, wrong_count, total_questions, avg_time_ms)
    )
    conn.commit()
    conn.close()
    update_daily_streak(user_id)


def update_daily_streak(user_id: int) -> int:
    """Called once per finished quiz a user participates in. Bumps their streak by 1
    if today is the day right after their last counted day, resets to 1 if they broke
    the chain, and leaves it untouched if they already played today (so answering 5
    quizzes in one day doesn't inflate the count). Returns the resulting streak."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT streak_count, last_quiz_date FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    streak, last_date = (row[0] or 0, row[1]) if row else (0, None)
    if last_date == today:
        conn.close()
        return streak
    if last_date == (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d"):
        streak += 1
    else:
        streak = 1
    cur.execute("UPDATE users SET streak_count=?, last_quiz_date=? WHERE user_id=?", (streak, today, user_id))
    conn.commit()
    conn.close()
    return streak


def update_quiz_timer(quiz_id: int, timer_seconds: int):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE quizzes SET timer_seconds=? WHERE id=?", (max(0, timer_seconds), quiz_id))
    conn.commit()
    conn.close()


def get_quiz_status_rows(quiz_id: int, limit: int = 200) -> list:
    """Every finished attempt of one specific quiz — used by the 📊 Status button in
    the Official Quiz editing section: who played it (user_id, for direct contact),
    how many they got right/wrong, their accuracy %, and their average answer speed."""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT user_id, user_name, score, correct_count, wrong_count, total_questions, avg_time_ms, finished_at "
        "FROM quiz_results WHERE quiz_id=? ORDER BY score DESC, avg_time_ms ASC LIMIT ?",
        (quiz_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    results = []
    for r in rows:
        correct, wrong = r[3] or 0, r[4] or 0
        total = r[5] or (correct + wrong)
        accuracy = round((correct / total) * 100, 1) if total else 0.0
        results.append({
            "user_id": r[0], "user_name": r[1] or "Unknown", "score": r[2] or 0,
            "correct": correct, "wrong": wrong, "total": total, "accuracy": accuracy,
            "avg_time_ms": r[6] or 0, "finished_at": r[7],
        })
    return results


# ==========================================
# 👋 GROUP WELCOME / GOODBYE SETTINGS
# ==========================================
DEFAULT_WELCOME_MSG = "👋 Welcome {mention} to {group}! Type /help to see what I can do."
DEFAULT_GOODBYE_MSG = "😢 {name} left {group}. Goodbye!"


def get_group_settings(group_id: int) -> dict:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT welcome_enabled, welcome_msg, goodbye_msg FROM group_settings WHERE group_id=?",
        (group_id,)
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"enabled": True, "welcome_msg": DEFAULT_WELCOME_MSG, "goodbye_msg": DEFAULT_GOODBYE_MSG}
    return {
        "enabled": bool(row[0]) if row[0] is not None else True,
        "welcome_msg": row[1] or DEFAULT_WELCOME_MSG,
        "goodbye_msg": row[2] or DEFAULT_GOODBYE_MSG,
    }


def upsert_group_settings(group_id: int, **fields):
    """Only overwrites the keys passed in — e.g. upsert_group_settings(gid, welcome_enabled=0)
    leaves any saved welcome_msg/goodbye_msg untouched."""
    current = get_group_settings(group_id)
    merged = {
        "welcome_enabled": int(fields.get("welcome_enabled", current["enabled"])),
        "welcome_msg": fields.get("welcome_msg", current["welcome_msg"]),
        "goodbye_msg": fields.get("goodbye_msg", current["goodbye_msg"]),
    }
    conn = db_connect()
    cur = conn.cursor()
    upsert(cur, "group_settings", {"group_id": group_id}, merged)
    conn.commit()
    conn.close()


def render_welcome_template(template: str, member_name: str, member_id: int, group_title: str) -> str:
    mention = f"[{member_name}](tg://user?id={member_id})"
    return (
        (template or DEFAULT_WELCOME_MSG)
        .replace("{mention}", mention)
        .replace("{name}", member_name)
        .replace("{group}", group_title or "the group")
    )[:1000]


def track_group(group_id: int, title: str):
    conn = db_connect()
    cur = conn.cursor()
    upsert(cur, "groups", {"group_id": group_id}, {"title": (title or "")[:120]})
    conn.commit()
    conn.close()


async def track_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if chat and chat.type in ("group", "supergroup"):
        track_group(chat.id, chat.title or "")


def get_user_quizzes(user_id: int) -> list:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT id, title FROM quizzes WHERE user_id=? ORDER BY id DESC", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1]} for r in rows]


def get_quiz_by_id(quiz_id: int):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT id, user_id, title, data FROM quizzes WHERE id=?", (quiz_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "user_id": row[1], "title": row[2], "questions": json.loads(row[3])}


def update_quiz(quiz_id: int, questions: list):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE quizzes SET data=? WHERE id=?", (json.dumps(questions), quiz_id))
    conn.commit()
    conn.close()


def save_last_source(user_id: int, source_type: str, parts: list, difficulty: str, language: str, count: int):
    conn = db_connect()
    cur = conn.cursor()
    upsert(
        cur, "last_source", {"user_id": user_id},
        {"source_type": source_type, "parts": json.dumps(parts), "difficulty": difficulty,
         "language": language, "count": count}
    )
    conn.commit()
    conn.close()


def get_last_source(user_id: int):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT source_type, parts, difficulty, language, count FROM last_source WHERE user_id=?",
        (user_id,)
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"source_type": row[0], "parts": json.loads(row[1]), "difficulty": row[2], "language": row[3], "count": row[4]}


def redeem_code(user_id: int, code: str) -> str:
    get_user(user_id)
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT bonus_questions, max_uses, used_count, expires_at FROM redeem_codes WHERE code = ?", (code,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return "❌ Invalid code."
    bonus, max_uses, used_count, expires_at = row
    if expires_at and datetime.utcnow() > datetime.fromisoformat(expires_at):
        conn.close()
        return "❌ This code has expired."
    if max_uses and used_count >= max_uses:
        conn.close()
        return "❌ This code has already been fully used."
    cur.execute("SELECT 1 FROM redeem_uses WHERE code=? AND user_id=?", (code, user_id))
    if cur.fetchone():
        conn.close()
        return "❌ You've already redeemed this code."
    cur.execute("UPDATE users SET question_limit = question_limit + ? WHERE user_id = ?", (bonus, user_id))
    cur.execute("UPDATE redeem_codes SET used_count = used_count + 1 WHERE code = ?", (code,))
    cur.execute("INSERT INTO redeem_uses (code, user_id) VALUES (?, ?)", (code, user_id))
    conn.commit()
    conn.close()
    return f"✅ Redeemed! +{bonus} questions added to your quota."


def add_group_filter(group_id: int, keyword: str, reply: str):
    conn = db_connect()
    cur = conn.cursor()
    upsert(cur, "filters", {"group_id": group_id, "keyword": keyword}, {"reply": reply})
    conn.commit()
    conn.close()


def remove_group_filter(group_id: int, keyword: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM filters WHERE group_id=? AND keyword=?", (group_id, keyword))
    conn.commit()
    conn.close()


def list_group_filters(group_id: int) -> list:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT keyword, reply FROM filters WHERE group_id=?", (group_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


# ==========================================
# 🔑 PER-USER API KEY HELPERS (SQLite-backed, persists across restarts)
# Each user can save MULTIPLE keys/providers — generation tries them in the
# order added, then always falls back to the owner's shared key at the end.
# ==========================================
def add_user_api_key(user_id: int, provider: str, api_key: str, api_base: str, model: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO user_api_keys (user_id, provider, api_key, api_base, model) VALUES (?, ?, ?, ?, ?)",
        (user_id, provider, api_key, api_base, model)
    )
    conn.commit()
    conn.close()


def get_user_api_keys(user_id: int) -> list:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, provider, api_key, api_base, model FROM user_api_keys WHERE user_id=? ORDER BY id",
        (user_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "provider": r[1], "api_key": r[2], "api_base": r[3], "model": r[4]} for r in rows]


def delete_user_api_key_by_id(user_id: int, key_id: int):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM user_api_keys WHERE user_id=? AND id=?", (user_id, key_id))
    conn.commit()
    conn.close()


def delete_all_user_api_keys(user_id: int):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM user_api_keys WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def mask_key(key: str) -> str:
    if not key:
        return "(none)"
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}{'•' * 6}{key[-4:]}"


def build_provider_chain(user_id: int) -> list:
    """
    The ordered list of providers to try for this user: their own saved keys
    first (in the order they added them with /addapi), then EVERY owner-level
    key the bot owner has configured via env vars (OWNER_PROVIDER_KEYS) — including
    multiple comma-separated keys for the same provider, each treated as its own
    fallback entry with its own usage tracking — tried in OWNER_FALLBACK_ORDER,
    with Anthropic last. This keeps generation working ("meri wali to run kregi")
    no matter how many keys a user has added (or whether they've added any at
    all), and no matter which of the owner's own keys happen to be rate-limited
    at that moment.
    """
    chain = []
    for saved in get_user_api_keys(user_id):
        preset = PROVIDER_PRESETS.get(saved["provider"], {})
        chain.append({
            "provider": saved["provider"],
            "api_key": saved["api_key"],
            "api_base": saved["api_base"] or preset.get("api_base"),
            "model": saved["model"] or preset.get("model"),
            "label": preset.get("label", saved["provider"]),
            "key_id": saved["id"],
            "owner_user_id": user_id,
        })

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
                # Cloudflare-style providers need an account-specific base URL built
                # from a second env var rather than a fixed preset base. Multiple keys
                # paired with fewer account IDs reuse the last account ID given.
                if not account_ids:
                    continue  # key(s) without any account id are unusable — skip silently
                account_id = account_ids[idx] if idx < len(account_ids) else account_ids[-1]
                api_base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
            model = (AI_MODEL if provider == "anthropic" and AI_MODEL else preset.get("model"))
            label = f"Owner's shared {preset.get('label', provider)}"
            if len(api_keys) > 1:
                label += f" #{idx + 1}"
            chain.append({
                "provider": provider,
                "api_key": api_key,
                "api_base": api_base,
                "model": model,
                "label": label,
                "key_id": None,
                "owner_key_index": idx,
                "owner_user_id": OWNER_ID,
            })

    return chain


# ==========================================
# 📊 PER-KEY DAILY USAGE TRACKING + OWNER ALERTS
# Har API key (owner ki shared key ho ya kisi user ki /addapi wali) ki daily call
# count track hoti hai. 50/70/90/100% cross hote hi owner ko ek DM jaata hai — taaki
# limit "chupke se" khatam na ho, aur zaroorat pade to owner us key ko time rehte
# rok sake ya users ko warn kar sake.
#
# ⚠️ Ye defaults har provider ke PUBLISHED FREE-TIER daily request cap par based hain
# (verified ~August 2026) — lekin har provider apni free-tier limits BINA notice ke
# badal sakta hai. Har ek `<PROVIDER>_DAILY_LIMIT` env var se override ho sakta hai
# agar tumhara actual plan/limit alag hai (jaise Groq Developer tier, ya OpenRouter
# jisne $10 credit purchase kiya ho). Bot ka apna counter hamesha UTC midnight par
# reset hota hai (chaahe provider ka real reset kisi bhi cheez par ho — daily UTC,
# rolling window, ya monthly credit) — ye ek consistent internal baseline hai, provider
# ke asli reset time ka exact substitute nahi.
# ==========================================
DEFAULT_KEY_DAILY_LIMIT = int(os.getenv("KEY_DAILY_LIMIT", "1500"))
USAGE_ALERT_THRESHOLDS = [50, 70, 90, 100]

# Per-provider default daily request caps (override any of these with an env var,
# e.g. GROQ_DAILY_LIMIT=500). None = don't track (custom/self-hosted, or a provider
# whose free tier isn't a clean daily-request number).
PROVIDER_DAILY_LIMIT_DEFAULTS = {
    "groq": 1000,               # llama-3.3-70b-versatile free tier: ~1,000 RPD (resets ~UTC)
    "gemini": 1500,             # Flash models free tier: 1,500 RPD (resets ~midnight Pacific)
    "openai": None,             # not a real "free" daily cap — tied to account spend tier
    "anthropic": None,          # owner's paid key — org-specific tier, no generic public number
    "github": 150,              # gpt-4o-mini on GitHub Models: 150 RPD (per-model, ~24h rolling)
    "openrouter_vision": 50,    # :free models: 50 RPD (no credit purchased) / 1,000 RPD after $10 credit
    "openrouter_text": 50,      # same free-tier cap as above — bump via env var if you've bought credit
    "cloudflare_vision": 300,   # 10,000 Neurons/day shared pool; vision burns more Neurons/call
    "cloudflare_text": 1000,    # same 10,000-Neuron pool; small text model burns far fewer/call
    "mistral": 500,             # Mistral doesn't publish an exact free RPD — rough safety estimate
    "cerebras": 14000,          # ~14,400 RPD (buffer kept) — 1M tokens/day may bind first on long prompts
    "huggingface": 10,          # Inference Providers free credit is <$0.10/MONTH — practically a handful of calls
}


def get_provider_daily_limit(provider: str):
    """None means 'don't track/limit this provider'. Otherwise checks
    PROVIDER_DAILY_LIMIT_DEFAULTS first, then falls back to DEFAULT_KEY_DAILY_LIMIT for
    anything not listed there — always overridable via a `<PROVIDER>_DAILY_LIMIT` env var."""
    if provider == "custom":
        return None
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
    if candidate.get("key_id") is None:
        idx = candidate.get("owner_key_index", 0)
        return f"owner:{candidate['provider']}:{idx}"
    return f"user:{candidate['owner_user_id']}:{candidate['key_id']}"


def is_key_exhausted_today(candidate: dict) -> bool:
    """Cheap local check BEFORE even calling the provider — avoids wasting a real
    request once we already know (from our own counter) that today's quota for this
    key is used up."""
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
    """Call once per SUCCESSFUL provider call. Increments today's counter for that
    key and, if a new 50/70/90/100% threshold was just crossed, queues an owner DM
    (delivered by the periodic flush job — see flush_owner_notifications_job)."""
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

    upsert(cur, "api_key_usage", {"scope": scope, "usage_date": today},
           {"count": count, "last_notified_pct": last_notified})

    if crossed is not None:
        who = "Owner's shared key" if candidate.get("key_id") is None else f"User {candidate['owner_user_id']}'s key #{candidate['key_id']}"
        if crossed >= 100:
            note = (
                f"🚨 **Daily limit reached**\n{who} — {candidate['label']} ({candidate['provider']})\n"
                f"Used: {count}/{limit} calls today (100%+).\n"
                f"This key will keep failing/be skipped until it resets at UTC midnight."
            )
        else:
            note = (
                f"📊 **Usage alert — {crossed}%**\n{who} — {candidate['label']} ({candidate['provider']})\n"
                f"Used: {count}/{limit} calls today."
            )
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
        [InlineKeyboardButton("✅ I Am Verified", callback_data="check_verification")]
    ]
    return InlineKeyboardMarkup(keyboard)


def main_welcome_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("🛠️ Tools", callback_data="open_tools"),
            InlineKeyboardButton("ℹ️ Help", callback_data="open_help")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def tools_menu_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("🧠 Topic MCQ", callback_data="create_topic"),
            InlineKeyboardButton("📄 PDF MCQ", callback_data="create_pdf")
        ],
        [
            InlineKeyboardButton("📊 CSV MCQ", callback_data="create_csv"),
            InlineKeyboardButton("🖼 Image MCQ", callback_data="create_image")
        ],
        [
            InlineKeyboardButton("🔄 Poll → Quiz", callback_data="create_poll_clean"),
            InlineKeyboardButton("📋 Poll → Text", callback_data="polltxt_start")
        ],
        [
            InlineKeyboardButton("🎙️ Voice → Quiz", callback_data="create_voice"),
            InlineKeyboardButton("📥 Import Channel/Group", callback_data="create_import")
        ],
        [
            InlineKeyboardButton("⚙️ Customize", callback_data="menu_customize")
        ],
        [
            InlineKeyboardButton("♻️ Regenerate", callback_data="act_regenerate"),
            InlineKeyboardButton("🤖 Official Quiz Bot", callback_data="menu_official")
        ],
        [
            InlineKeyboardButton("🛡️ Group Management", callback_data="menu_group"),
            InlineKeyboardButton("📚 My Store", callback_data="menu_store")
        ],
        [
            InlineKeyboardButton("🌐 Public Store", callback_data="menu_publicstore")
        ],
        [
            InlineKeyboardButton("🔑 Add API Key", callback_data="menu_apikey")
        ],
        [
            InlineKeyboardButton("🔙 Back to Main Menu", callback_data="menu_main")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def back_to_tools_keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Tools", callback_data="open_tools")]])


def build_welcome_text(user_name: str, user_data: dict, verified_banner: bool = False) -> str:
    if verified_banner:
        icon, title = "✅", stylize("Verification Successful") + "!"
    else:
        icon, title = "✨", stylize("Welcome") + f", {user_name}!"
    lines = [
        f"{icon} **{title}**",
        f"_{FONT_STYLE}_",
        "",
        f"👤 **User ID:** `{user_data['user_id']}`",
        f"📊 **Questions Quota:** {user_data['questions_used']}/{user_data['question_limit']}",
        f"✅ **Status:** Verified User",
        "",
        "Click **🛠️ Tools** below to access all quiz creation & group features!",
    ]
    return "\n".join(lines)


# ==========================================
# 🤖 AI GENERATION CORE
# ==========================================
class ProviderError(RuntimeError):
    """Raised by a single provider call. `is_quota=True` means the provider itself
    refused because a rate-limit/quota/credit ceiling was hit — used to build a clear
    "API key ki limit khatam ho gayi hai" message instead of a generic failure."""
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
    return (
        "You are an expert quiz writer. Read the supplied source material and write "
        f"exactly {count} multiple-choice questions at {difficulty} difficulty.\n\n"
        f"LANGUAGE — read carefully: every single piece of text you output — the "
        f"question, all 4 options, AND the explanation, for EVERY item — must be written "
        f"ENTIRELY in {language}. This applies even if the source material below is in a "
        f"different language: translate the underlying facts/content into {language}, "
        f"don't just copy fragments from the source as-is. Never leave a question, option, "
        f"or explanation partially or fully in a different language, and never mix two "
        f"languages within the same item (e.g. Hindi question with English options is NOT "
        f"acceptable unless {language} literally is a mix like Hinglish). Proper nouns "
        f"(names, places) may stay as commonly written, but every other word must be in "
        f"{language}.\n\n"
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
    """Text-extraction fallback for providers that can't read a raw PDF file
    (used by the OpenAI-compatible path — Groq/OpenAI/custom servers)."""
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
    """Native support for text, images, and PDFs (Claude reads files directly)."""
    if not (Anthropic and api_key):
        raise ProviderError("No Anthropic key available for this step.", is_quota=False)
    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model or AI_MODEL,
            max_tokens=4096,
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
    """Builds an OpenAI-vision-style content list: text stays text, images go in
    as base64 data URIs (works for vision-capable models on OpenAI/Groq/most
    self-hosted servers), and PDFs are text-extracted locally since chat-completions
    endpoints generally can't take a raw PDF file."""
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
    """Works for Groq, OpenAI, and any self-hosted / open-source OpenAI-compatible
    server (Ollama, LM Studio, vLLM, Together, OpenRouter, etc). Images are sent
    as vision content if the chosen model supports it; PDFs are text-extracted."""
    if not api_base or not model:
        raise ProviderError("This provider isn't fully configured — use /addapi to set it up again.", is_quota=False)
    content_blocks = _build_openai_style_content(source_parts)
    # Send a plain string when it's just one text block — friendlier to strict/older servers.
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
        "temperature": 0.7,
        "max_tokens": 4096,
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
    """Gemini natively accepts inline images AND PDFs via inline_data, so both
    work directly here without any local extraction."""
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


GEN_BATCH_SIZE = 20  # generate in chunks this size — keeps each single AI response
                      # short enough to reliably finish (this is the main fix for
                      # "utne question nahi banata": asking one huge model call for
                      # 50-100 questions in one JSON blob commonly gets truncated or
                      # cut short by the provider's own output-token limit; asking
                      # for 20 at a time and stitching the batches together doesn't).


def _generate_batch(user_id: int, source_parts: list, batch_count: int, difficulty: str, language: str) -> list:
    """Runs the full provider fallback chain for ONE batch (<=GEN_BATCH_SIZE
    questions). This is exactly the original single-shot generate_mcqs_sync body —
    generate_mcqs_sync below just calls this once per batch and stitches the
    results together so a big request (50/75/100) doesn't ride on a single,
    easily-truncated AI response."""
    chain = build_provider_chain(user_id)
    if not chain:
        raise RuntimeError(
            "⚠️ Koi AI key available nahi hai. /addapi se apni key add karo, ya bot owner se "
            "ANTHROPIC_API_KEY set karne ko bolo."
        )

    errors = []       # list of (label, message, is_quota)
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
            # Non-provider failure (bad JSON from the model, unreadable PDF, empty
            # source, etc.) — never a quota/limit issue, so it must not be reported as one.
            logger.warning(f"Provider '{cand['label']}' failed for user {user_id} (non-quota): {e}")
            errors.append((cand["label"], str(e), False))
            any_non_quota = True
            continue

    if errors and not any_non_quota:
        # Every single provider in the chain failed specifically because of a
        # rate-limit/quota/credit ceiling — say so directly instead of a vague error,
        # and do NOT let it fall through to "generating questions" afterwards.
        lines = "\n".join(f"• {label}" for label, _, _ in errors)
        raise RuntimeError(
            "⚠️ **API key(s) ki limit/quota khatam ho gayi hai** — isliye poll/questions nahi ban paaye:\n"
            f"{lines}\n\n"
            "/addapi se nayi/dusri key add karo, ya thodi der (ya agle billing cycle) baad try karo."
        )

    lines = "\n".join(f"• {label}: {msg}" for label, msg, _ in errors)
    raise RuntimeError(f"⚠️ Question generate nahi ho paaye, har provider me problem aayi:\n{lines}")


def generate_mcqs_sync(user_id: int, source_parts: list, count: int, difficulty: str, language: str) -> list:
    """
    Synchronous, SINGLE-THREAD version — runs every batch one after another. Only
    used where a plain blocking call is fine (small known counts, e.g. Voice → Quiz).
    For anything driven by the /topic /pdf /image /section flow, use
    generate_mcqs_async below instead — it runs batches concurrently so one user's
    big (50-100 question) request doesn't tie up a single thread for a long time
    and starve everyone else's polls/commands while it's running.
    """
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
    """
    Preferred entry point for the AI-generation handlers. Splits `count` into
    GEN_BATCH_SIZE-sized batches and runs them CONCURRENTLY (each in its own
    worker thread via asyncio.to_thread + asyncio.gather) instead of one after
    another. This is the fix for "bot ek jagah AI chalata hai to dusri jagah
    commands/polls me time lagta hai": a single big request used to occupy one
    thread for a long, serial stretch (batch 1 done, THEN batch 2, THEN batch
    3...) — now all its batches run in parallel, so the request finishes in
    roughly the time of ONE batch instead of N, freeing up threads for other
    users' commands much sooner. See main() for the enlarged thread pool this
    relies on to actually run many of these concurrently across different users.

    A batch that fails outright raises; but if at least one batch succeeded
    before that, the partial result is still returned rather than discarding
    everything already generated (better to hand back 62/75 questions than none).
    """
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


# ==========================================
# ❓ /ask — DOUBT SOLVER
# Kisi bhi quiz question, word, ya paragraph me confusion ho to /ask se seedha
# poocho. Same provider chain use hoti hai (pehle user ki apni /addapi keys,
# aakhir me owner ki shared key) — is liye ye bina kisi key ke bhi kaam karta
# hai, jaisa quiz-generation already karta hai.
# ==========================================
def _build_doubt_prompt(doubt_context: str) -> str:
    base = (
        "You are a friendly, clear tutor helping someone who is confused about "
        "something in a quiz — it could be a specific question, an option, a "
        "word/term, or a paragraph of source text. Explain it simply in a short, "
        "direct answer (a few sentences, plain language, no markdown headers). "
        "If it's a term, define it plainly. If it's a question, explain the "
        "reasoning behind the correct answer. Reply in the same language the "
        "user asked in."
    )
    if doubt_context:
        base += f"\n\nRelevant context from the quiz/source (may or may not be needed):\n{doubt_context[:2000]}"
    return base


def _ask_via_anthropic(api_key, model, question, doubt_context) -> str:
    if not (Anthropic and api_key):
        raise ProviderError("No Anthropic key available for this step.", is_quota=False)
    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model or AI_MODEL,
            max_tokens=600,
            system=_build_doubt_prompt(doubt_context),
            messages=[{"role": "user", "content": question}],
        )
    except Exception as e:
        status_code = getattr(e, "status_code", None)
        is_quota = _looks_like_quota_error(str(e), exc=e, status_code=status_code)
        is_auth = _looks_like_auth_error(str(e), status_code=status_code)
        logger.error(f"Anthropic /ask call failed: {e}")
        raise ProviderError(_friendly_provider_message(str(e), is_quota, is_auth), is_quota=is_quota, status_code=status_code)
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()


def _ask_via_openai_compatible(api_key, api_base, model, question, doubt_context) -> str:
    if not api_base or not model:
        raise ProviderError("This provider isn't fully configured — use /addapi to set it up again.", is_quota=False)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _build_doubt_prompt(doubt_context)},
            {"role": "user", "content": question},
        ],
        "temperature": 0.5,
    }
    resp = None
    try:
        resp = requests.post(f"{api_base.rstrip('/')}/chat/completions", headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        status_code = resp.status_code if resp is not None else None
        body_text = resp.text[:300] if resp is not None else str(e)
        is_quota = _looks_like_quota_error(body_text, status_code=status_code)
        is_auth = _looks_like_auth_error(body_text, status_code=status_code)
        logger.error(f"OpenAI-compatible /ask call failed ({status_code}): {body_text}")
        raise ProviderError(_friendly_provider_message(body_text, is_quota, is_auth), is_quota=is_quota, status_code=status_code)


def _ask_via_gemini(api_key, model, question, doubt_context) -> str:
    if not api_key:
        raise ProviderError("No Gemini key available for this step.", is_quota=False)
    parts = [{"text": _build_doubt_prompt(doubt_context) + "\n\nQuestion: " + question}]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {"contents": [{"parts": parts}]}
    resp = None
    try:
        resp = requests.post(url, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        status_code = resp.status_code if resp is not None else None
        body_text = resp.text[:300] if resp is not None else str(e)
        is_quota = _looks_like_quota_error(body_text, status_code=status_code)
        is_auth = _looks_like_auth_error(body_text, status_code=status_code)
        logger.error(f"Gemini /ask call failed ({status_code}): {body_text}")
        raise ProviderError(_friendly_provider_message(body_text, is_quota, is_auth), is_quota=is_quota, status_code=status_code)


def answer_doubt_sync(user_id: int, question: str, doubt_context: str = "") -> str:
    """Synchronous helper — call via asyncio.to_thread from /ask. Uses the exact same
    fallback chain as quiz generation (user's own /addapi keys first, then the owner's
    shared key), so a user with zero keys of their own can still use /ask."""
    chain = build_provider_chain(user_id)
    if not chain:
        raise RuntimeError(
            "⚠️ Koi AI key available nahi hai abhi. /addapi se apni key add karo, ya bot "
            "owner se ANTHROPIC_API_KEY set karne ko bolo."
        )
    errors = []
    any_non_quota = False
    for cand in chain:
        if is_key_exhausted_today(cand):
            errors.append((cand["label"], "Aaj ki daily limit (tracked) khatam ho chuki hai.", True))
            continue
        try:
            if cand["provider"] == "anthropic":
                result = _ask_via_anthropic(cand["api_key"], cand["model"], question, doubt_context)
            elif cand["provider"] == "gemini":
                result = _ask_via_gemini(cand["api_key"], cand["model"], question, doubt_context)
            else:
                result = _ask_via_openai_compatible(cand["api_key"], cand["api_base"], cand["model"], question, doubt_context)
            record_api_key_usage(cand)
            return result
        except ProviderError as e:
            logger.warning(f"/ask provider '{cand['label']}' failed for user {user_id} (quota={e.is_quota}): {e}")
            errors.append((cand["label"], str(e), e.is_quota))
            if not e.is_quota:
                any_non_quota = True
            continue
        except Exception as e:
            logger.warning(f"/ask provider '{cand['label']}' failed for user {user_id}: {e}")
            errors.append((cand["label"], str(e), False))
            any_non_quota = True
            continue

    if errors and not any_non_quota:
        lines = "\n".join(f"• {label}" for label, _, _ in errors)
        raise RuntimeError(
            "⚠️ **API key(s) ki limit/quota khatam ho gayi hai** — isliye jawab nahi mil paaya:\n"
            f"{lines}\n\n/addapi se nayi key add karo, ya thodi der baad try karo."
        )
    lines = "\n".join(f"• {label}: {msg}" for label, msg, _ in errors)
    raise RuntimeError(f"⚠️ Jawab nahi mil paaya, har provider me problem aayi:\n{lines}")


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/ask <your doubt> — works standalone, or reply to a quiz poll / any message
    (question, word, paragraph) with /ask <what's confusing you> and that message's
    text is sent along as context. No personal API key required — falls back to the
    bot's shared key exactly like quiz generation does."""
    msg = update.effective_message
    question = " ".join(context.args).strip() if context.args else ""
    doubt_context = ""

    replied = msg.reply_to_message
    if replied:
        if replied.poll:
            opts = ", ".join(o.text for o in replied.poll.options)
            doubt_context = f"Poll question: {replied.poll.question}\nOptions: {opts}"
        elif replied.text:
            doubt_context = replied.text
        elif replied.caption:
            doubt_context = replied.caption

    if not question and not doubt_context:
        await msg.reply_text(
            "❓ **Kuch samajh nahi aaya?**\n\n"
            "Usage:\n"
            "• `/ask <apna sawaal>` — seedha koi bhi doubt poocho\n"
            "• Kisi quiz question, word, ya paragraph wale message par **reply** karke "
            "`/ask <kya samajh nahi aaya>` bhejo — us message ka text context ke saath jayega\n\n"
            "Koi apni API key nahi hai to bhi chalega — bot ki shared key use ho jayegi.",
            parse_mode="Markdown",
        )
        return
    if not question:
        question = "Please explain this simply."

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        answer = await asyncio.to_thread(answer_doubt_sync, update.effective_user.id, question, doubt_context)
    except RuntimeError as e:
        await msg.reply_text("⚠️ Could not get an answer due to an error. Please try again in a bit.")
        if OWNER_ID:
            try:
                await context.bot.send_message(OWNER_ID, f"🔧 /ask failed for user {update.effective_user.id}:\n{e}")
            except Exception as ex:
                logger.warning(f"Failed to DM owner about /ask failure: {ex}")
        return
    await msg.reply_text(f"💡 {answer}")


# ==========================================
# 🎙️ VOICE → QUIZ (speak the topic + how many questions, bot transcribes & builds it)
# ==========================================
def _transcribe_via_openai_whisper(api_key, api_base, model_hint, audio_bytes: bytes) -> str:
    """Groq/OpenAI-compatible /audio/transcriptions (Whisper) endpoint. Works for Groq
    (whisper-large-v3) and OpenAI (whisper-1); most self-hosted servers that mirror the
    OpenAI API also expose this route."""
    if not api_base:
        raise ProviderError("This provider isn't fully configured for audio.", is_quota=False)
    whisper_model = "whisper-1"
    if "groq" in api_base:
        whisper_model = "whisper-large-v3"
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
    """Gemini accepts inline audio and can transcribe directly — no separate ASR endpoint needed."""
    if not api_key:
        raise ProviderError("No Gemini key available for this step.", is_quota=False)
    b64 = base64.b64encode(audio_bytes).decode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [
            {"text": "Transcribe this audio exactly, in whatever language is spoken. Reply with ONLY the transcript text, nothing else."},
            {"inline_data": {"mime_type": "audio/ogg", "data": b64}},
        ]}]
    }
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
    """Tries every saved key that can actually do audio (Groq/OpenAI/custom via Whisper,
    or Gemini natively) in the user's saved order, then the owner's fallback key if it's
    Gemini-capable. Anthropic has no audio support so it's skipped, not counted as a failure."""
    chain = build_provider_chain(user_id)
    audio_capable = [c for c in chain if c["provider"] in ("groq", "openai", "custom", "gemini")]
    if not audio_capable:
        raise RuntimeError(
            "⚠️ Voice samajhne ke liye Groq, OpenAI, ya Gemini API key chahiye (Anthropic audio support nahi karta). "
            "/addapi se ek add karo."
        )

    errors = []
    any_non_quota = False
    for cand in audio_capable:
        try:
            if cand["provider"] == "gemini":
                return _transcribe_via_gemini(cand["api_key"], cand["model"], audio_bytes)
            else:
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
    """Best-effort extraction of (topic, count) from a spoken sentence like
    'science ke 10 sawal banao' or 'make 5 questions on history'. Falls back to a
    default of 10 questions if no number is heard, and returns the leftover text
    (filler words stripped) as the topic."""
    t = transcript.strip()
    count = 10
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


async def post_quiz_polls(bot, chat_id, questions: list, user_tag: str = '', explanation_tag: str = '', option_style: str = 'plain') -> int:
    sent = 0
    for q in questions:
        question_text = q["question"]
        if user_tag:
            question_text = f"{question_text}\n\n🏷 {user_tag}"
        question_text = question_text[:300]

        explanation = q.get("explanation", "")
        if explanation_tag:
            explanation = f"{explanation}\n\n💡 {explanation_tag}".strip() if explanation else f"💡 {explanation_tag}"
        explanation = explanation[:200] or None

        try:
            await bot.send_poll(
                chat_id=chat_id,
                question=question_text,
                options=apply_option_style(q["options"], option_style),
                type=Poll.QUIZ,
                correct_option_id=q["correct_index"],
                explanation=explanation,
                is_anonymous=False,
            )
            sent += 1
        except Exception as e:
            logger.warning(f"Failed to send a poll: {e}")

        # A promo message every 5 questions, mid-quiz only (never trailing after the
        # very last one) — applies no matter which tool made the quiz (AI or Official).
        if sent and sent % PROMO_EVERY_N_QUESTIONS == 0 and sent < len(questions):
            await send_promo_message(bot, chat_id)
    return sent


# ==========================================
# 🔐 VERIFICATION LOGIC
# ==========================================
async def check_user_membership(bot, user_id: int) -> bool:
    """Live-checks membership in UPDATE_CHANNEL and UPDATE_GROUP via the bot's own
    get_chat_member call. This ONLY works if the bot is an admin in both chats —
    that's a Telegram requirement, not a bug here.

    Fail-CLOSED: if the check can't be completed (bot isn't admin yet, the
    UPDATE_CHANNEL/UPDATE_GROUP value is wrong, or the user genuinely never joined —
    Telegram raises the same kind of error for "not a member" as it does for "bot
    can't see this chat"), we do NOT treat that as verified. Make the bot an admin in
    both chats and double-check the .env values if everyone is getting blocked."""
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
    """Gate used before commands run. Trusts the cached DB flag when it's already 1
    (fast path — avoids 2 Telegram API calls per message), but re-checks live and
    self-heals the flag when it's 0, so a user who rejoined doesn't have to remember
    to tap the "I Am Verified" button again. Sends the join prompt and returns False
    when the person still isn't verified."""
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
            text=(
                f"🔐 **{stylize('Verification Required')}**\n"
                f"_{FONT_STYLE}_\n\n"
                "Please join our update channel and support group, then tap "
                "**I Am Verified** below to continue."
            ),
            parse_mode="Markdown", reply_markup=verification_keyboard(),
        )
    return False


async def verification_command_gate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Runs before every /command (group=-1) so nothing works until the user is
    verified — except /start itself, which has its own verification prompt."""
    msg = update.effective_message
    if msg and msg.text and msg.text.split()[0].split('@')[0] == "/start":
        return
    if not await require_verification(update, context):
        raise ApplicationHandlerStop


async def handle_update_channel_group_membership(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Watches membership changes in UPDATE_CHANNEL / UPDATE_GROUP — requires the bot
    to be an admin in both, which is also what makes check_user_membership work at
    all. When a previously-verified user leaves or gets removed from either one, this
    flips them back to unverified in the DB and DMs them a rejoin prompt, so the bot
    stops working for them again until they rejoin and re-verify. (The DM only goes
    through if the user has started a chat with the bot before — a Telegram
    restriction, not something this code can work around.)"""
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
        return  # only care about member -> left/kicked transitions

    user = cmu.new_chat_member.user
    if user.is_bot or user.id == OWNER_ID:
        return
    set_verified(user.id, 0)
    try:
        await context.bot.send_message(
            chat_id=user.id,
            text=(
                "⚠️ **You left our update channel/group, so your verification was reset.**\n\n"
                "The bot won't work again until you rejoin and verify.\n"
                "Tap the buttons below, then hit **I Am Verified**."
            ),
            parse_mode="Markdown", reply_markup=verification_keyboard(),
        )
    except Exception as e:
        logger.warning(f"Couldn't DM rejoin notice to {user.id} (they may not have started the bot): {e}")


async def is_chat_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if update.effective_user.id == OWNER_ID:
        return True
    try:
        member = await context.bot.get_chat_member(update.effective_chat.id, update.effective_user.id)
        return member.status in ('creator', 'administrator')
    except Exception:
        return False


# ==========================================
# 🚀 BASIC COMMANDS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    user_data = get_user(user_id)

    if context.args and context.args[0].startswith("ref_") and user_data['questions_used'] == 0:
        try:
            referrer_id = int(context.args[0][4:])
            if referrer_id != user_id:
                set_user_field(user_id, "referred_by", referrer_id)
                increment_question_limit(referrer_id, 10)
        except ValueError:
            pass

    is_verified = await check_user_membership(context.bot, user_id)

    if not is_verified:
        verify_msg = (
            f"🔐 **{stylize('Verification Required')}**\n"
            f"_{FONT_STYLE}_\n\n"
            f"Hello **{user_name}**, to use this bot, please join our official update channel and support group first.\n\n"
            f"Click **I Am Verified** after joining!"
        )
        await update.message.reply_text(text=verify_msg, parse_mode="Markdown", reply_markup=verification_keyboard())
        return

    set_verified(user_id, 1)
    user_data = get_user(user_id)

    if context.args and context.args[0].startswith("quiz_") and context.args[0][5:].isdigit():
        quiz = get_quiz_meta(int(context.args[0][5:]))
        if quiz:
            timer_text = f"{quiz['timer_seconds']}s / question" if quiz['timer_seconds'] else "No limit"
            text = (
                f"📁 **{quiz['title']}**\n"
                + (f"💬 {quiz['description']}\n" if quiz.get('description') else "") +
                f"\n❓ Questions: {len(quiz['questions'])}\n"
                f"⏱ Timer: {timer_text}"
            )
            await update.message.reply_text(
                text, parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("▶️ Start Here", callback_data=f"oqstart_{quiz['id']}")]]
                ),
            )
            return

    await update.message.reply_text(
        text=build_welcome_text(user_name, user_data),
        parse_mode="Markdown",
        reply_markup=main_welcome_keyboard()
    )


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Catch-all for any /command that didn't match a real handler above it —
    registered last in its group so it only fires when nothing else claimed the
    update. Kept in plain English on purpose, and deliberately generic (no hints
    about which commands exist, no internal state) per the same
    don't-leak-internals principle as report_generation_failure."""
    await update.effective_message.reply_text(
        "❌ Invalid command. Type /help to see the list of valid commands."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    daily_left = get_daily_ai_remaining(update.effective_user.id)
    limit_line = "♾️ Unlimited AI generation" if daily_left is None else f"🎯 {daily_left}/{DAILY_AI_QUESTION_LIMIT} AI questions left today"
    help_text = (
        f"📖 **{stylize('Help & Commands List')}**\n"
        f"_{FONT_STYLE}_ · {limit_line}\n\n"
        f"📌 **Quiz Creation:**\n"
        f"• `/topic` — MCQs from a topic or text\n"
        f"• `/pdf` — MCQs from a PDF\n"
        f"• `/image` — MCQs from a photo\n"
        f"• `/csv` — MCQs from a CSV (exact, no AI)\n"
        f"• `/section` — Section-wise quiz creator\n"
        f"• `/voicequiz` — 🎙️ Bolo topic aur count, quiz ban jayegi\n"
        f"• `/importchat` — 📥 Kisi channel/group ke forwarded messages se quiz banao\n"
        f"• `/add_bulk` — 📥 Paste text OR upload a `.txt` file to bulk-create a quiz\n"
        f"• `/edit` — Edit a saved quiz\n"
        f"• `/default` — Set quiz default settings\n"
        f"• `/regenerate` — Fresh MCQs from your last source\n"
        f"• Question counts are 50/75/100. _In a **group**, `/topic`/`/pdf`/etc. also ask for a "
        f"timer + shuffle, then post live with a temporary leaderboard — same as Official Quiz. "
        f"In DM they just post all at once, no timer, like before._\n\n"
        f"🤖 **{stylize('Official Quiz')}** _(this bot's own branded quiz system)_**:**\n"
        f"• `/officialquiz` — create a titled, timed Official Quiz\n"
        f"• `/startquiz <id>` — start it live here. In a **group**, opens a waiting room "
        f"first — needs 2 people to tap **✅ I Am Ready**; auto-pauses if 2 questions in "
        f"a row get zero answers\n"
        f"• `/mystore` — manage your quizzes: ✏️ title, 💬 description, ⏱ timer, ➕/➖ "
        f"questions (all apply immediately) · 📊 **Status** (user_id + right/wrong/accuracy/"
        f"speed) · 📁 file in a Set · 📤 Send to Channel · 🌐 make a quiz Public/Private\n"
        f"• `/publicstore` — browse quizzes other users have made public\n\n"
        f"📊 **Poll & Control:**\n"
        f"• Forward any poll — auto-detects the ✅ answer when Telegram provides it "
        f"(closed/quiz polls); otherwise asks you to tap it once\n"
        f"• `/stoppoll` — Stop an active poll\n"
        f"• `/stop`, `/pause`, `/resume` — Control the current flow\n\n"
        f"🏆 **Leaderboards** _(always show user_id for direct contact)_**:**\n"
        f"• `/leaderboard` — Points / Accuracy / Today, scoped to the current group\n"
        f"• `/topusers` — Top 10 all-time by questions solved\n"
        f"• `/profile` or `/mystats` — your own stats, streak, and badges\n"
        f"• `/duel <topic>` — reply to someone in a group to challenge them to a 5-question 1v1\n\n"
        f"📁 **Quiz Sets & Library:**\n"
        f"• `/newset Name` — create a named folder for your quizzes\n"
        f"• `/sets` — browse your sets and what's filed in each\n"
        f"• `/browse [keyword]` — search your saved quizzes\n"
        f"• `/exportquiz <id>` — download a quiz as JSON\n"
        f"• `/exportresults <id>` — download a quiz's results as CSV\n\n"
        f"☀️ **Automation:**\n"
        f"• `/dailyquiz on <topic>` / `off` / `hour <0-23>` — auto-post a daily quiz to this group (admins only)\n\n"
        f"📤 **Channels:**\n"
        f"• `/setchannel` — link a channel (add me as admin there, then forward a message from it)\n"
        f"• `/mychannels` — list/remove linked channels\n\n"
        f"🏷️ **Personalize** _(applies everywhere — AI-generated and Official Quiz alike)_**:**\n"
        f"• `/settag YourTag` — shown under every question\n"
        f"• `/setdescription YourTag` — shown in the explanation\n"
        f"• `/optionstyle` — Plain / A) B) C) D) / 1) 2) 3) 4)\n"
        f"• `/redeem CODE` — add to your quota\n\n"
        f"🔑 **Your Own API Key(s):**\n"
        f"• `/addapi` — add Groq, Gemini, OpenAI, Anthropic, GitHub Models, OpenRouter, Cloudflare, "
        f"Mistral, Cerebras, Hugging Face, or any open-source/self-hosted key — add as many as you want\n"
        f"• `/myapi` — see everything saved and the order it's tried in\n"
        f"• `/limits` — see today's usage vs. daily cap for every key in your chain\n"
        f"• `/removeapi <id>` or `/removeapi all` — remove one or all, falls back to the bot's shared key\n\n"
        f"❓ **Doubt Solver:**\n"
        f"• `/ask <sawaal>` — kisi bhi quiz question, word, ya paragraph me confusion ho to seedha poocho\n"
        f"• Kisi message par reply karke `/ask <kya samajh nahi aaya>` bhejo — uska text context ke saath jayega\n"
        f"• Apni API key na ho to bhi chalega — bot ki shared key automatically use hoti hai\n\n"
        f"🛡️ **Group Management:**\n"
        f"• `/mute`, `/unmute`, `/purge`, `/info`\n"
        f"• `/promote`, `/demote`\n"
        f"• `/filter`, `/removefilter`, `/filters`\n"
        f"• `/welcome on|off` — toggle join/leave messages\n"
        f"• `/setwelcome`, `/setgoodbye` — customize them (`{{mention}}`, `{{name}}`, `{{group}}`)\n\n"
        f"💬 **Support:**\n"
        f"• `/feedback <message>` — send a message directly to the bot owner"
    )
    await update.message.reply_text(text=help_text, parse_mode="Markdown")


async def create_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"🛠️ **{stylize('Tools & Features Menu')}**\n_{FONT_STYLE}_\n\nSelect any tool below to proceed:",
        parse_mode="Markdown", reply_markup=tools_menu_keyboard()
    )


async def stoppoll_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message or not update.message.reply_to_message.poll:
        await update.message.reply_text("Reply to one of the bot's poll messages with /stoppoll.")
        return
    try:
        await context.bot.stop_poll(update.effective_chat.id, update.message.reply_to_message.message_id)
        await update.message.reply_text("🛑 Poll stopped.")
    except Exception as e:
        await update.message.reply_text(f"Couldn't stop that poll: {e}")


# --- Live "Official Quiz" control by COMMAND (no buttons) ---------------------
# /pause /resume /stopquiz /undoquiz /fastquiz /slowquiz all act on whatever
# live_quizzes[chat_id] session is running in the chat they're sent in. Only the
# person who started it, or a chat admin, may control it — same rule as before.
LIVE_QUIZ_MIN_TIMER = 5
LIVE_QUIZ_MAX_TIMER = 120
LIVE_QUIZ_SPEED_STEP = 5


async def _get_controllable_live_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Returns the running session for this chat if the sender is allowed to
    control it, else replies with why not and returns None."""
    chat_id = update.effective_chat.id
    live = context.application.bot_data.get('live_quizzes', {})
    session = live.get(chat_id)
    if not session or session.get('finished'):
        await update.message.reply_text("There's no Official Quiz currently running in this chat.")
        return None
    user_id = update.effective_user.id
    is_admin = await is_chat_admin(update, context)
    if user_id != session['creator_id'] and not is_admin:
        await update.message.reply_text("Only the quiz creator or a chat admin can control this.")
        return None
    return session


async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = await _get_controllable_live_session(update, context)
    if session is None:
        # No live quiz here — keep the old generic behaviour so /pause still
        # makes sense mid-way through any other multi-step flow.
        await update.message.reply_text("⏸ Okay — I'll wait. Send /resume (or just carry on) whenever you're ready.")
        return
    session['paused'] = True
    await update.message.reply_text("⏸ Paused — send /resume whenever you're ready.")


async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = await _get_controllable_live_session(update, context)
    if session is None:
        await update.message.reply_text("▶️ Welcome back — go ahead.")
        return
    session['paused'] = False
    await update.message.reply_text("▶️ Resumed.")


async def stopquiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = await _get_controllable_live_session(update, context)
    if session is None:
        return
    session['stopped'] = True
    await update.message.reply_text("⏹ Stopping...")
    await _finish_live_quiz(context.bot, update.effective_chat.id, context.application)


async def undoquiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = await _get_controllable_live_session(update, context)
    if session is None:
        return
    session['undo_requested'] = True
    await update.message.reply_text("↩️ Undoing last question...")


async def fastquiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = await _get_controllable_live_session(update, context)
    if session is None:
        return
    current = session['timer'] or (LIVE_QUIZ_MIN_TIMER + LIVE_QUIZ_SPEED_STEP * 3)
    session['timer'] = max(LIVE_QUIZ_MIN_TIMER, current - LIVE_QUIZ_SPEED_STEP)
    await update.message.reply_text(
        f"🐇 Sped up — next question onward gets {session['timer']}s to answer."
    )


async def slowquiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = await _get_controllable_live_session(update, context)
    if session is None:
        return
    current = session['timer'] or (LIVE_QUIZ_MIN_TIMER + LIVE_QUIZ_SPEED_STEP * 3)
    session['timer'] = min(LIVE_QUIZ_MAX_TIMER, current + LIVE_QUIZ_SPEED_STEP)
    await update.message.reply_text(
        f"🐢 Slowed down — next question onward gets {session['timer']}s to answer."
    )


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.effective_message.reply_text(
        "❌ Cancelled — nothing was lost. Your saved quizzes, API keys, groups and quota are all safe."
    )
    return ConversationHandler.END


async def cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Same as /cancel but triggered from the inline '❌ Cancel' button that's
    attached to every multi-step flow, so a mid-flow tap always exits cleanly
    without losing anything already saved to the database."""
    query = update.callback_query
    await query.answer()
    context.user_data.clear()
    try:
        await query.edit_message_text(
            "❌ Cancelled — nothing was lost. Your saved quizzes, API keys, groups and quota are all safe."
        )
    except Exception:
        pass
    return ConversationHandler.END


def cancel_keyboard(extra_rows=None):
    """Reusable '❌ Cancel' inline keyboard, optionally with extra rows above it."""
    rows = list(extra_rows or [])
    rows.append([InlineKeyboardButton("❌ Cancel", callback_data="gcancel")])
    return InlineKeyboardMarkup(rows)


# ==========================================
# 🧠 AI QUIZ GENERATION — CONVERSATION FLOW
# (shared by topic / pdf / image / section)
# ==========================================
# Telegram's OWN hard limit (not this bot's, and not adjustable in code): a normal
# hosted bot — anything not running its own self-hosted "local Bot API server" — can
# NEVER download a file bigger than 20 MB via getFile, full stop. Sending files out
# allows more (up to 50 MB via upload), but that's a different limit for a different
# direction and doesn't help here. Checked from msg.document.file_size (already in
# the update, no extra API call) so an oversized PDF gets an instant, clear message
# instead of a silent/confusing failure from Telegram partway through a download.
MAX_PDF_SIZE_MB = 20
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024


async def ai_entry(update: Update, context: ContextTypes.DEFAULT_TYPE, source_type: str):
    context.user_data['gen_source_type'] = source_type
    context.user_data['gen_parts'] = []
    prompts = {
        'topic': "Send me the topic or paste the text you want MCQs from.",
        'pdf': "Send me the PDF file.",
        'image': "Send me a photo of the page.",
        'section': "Paste the full text you want split into sections.",
    }
    msg = update.effective_message
    text = stylize("Let's build your quiz") + f"\n\n{prompts.get(source_type, 'Send your content.')}"
    keyboard = cancel_keyboard()
    await msg.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)
    return ASK_CONTENT


def make_command_entry(source_type: str):
    async def _entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if source_type == 'topic' and context.args:
            context.user_data['gen_source_type'] = 'topic'
            context.user_data['gen_parts'] = [{"type": "text", "text": " ".join(context.args)}]
            return await ask_difficulty(update, context)
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
section_entry = make_command_entry('section')

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
        return await ask_difficulty(update, context)

    if source_type == 'pdf':
        if not (msg.document and msg.document.file_name and msg.document.file_name.lower().endswith('.pdf')):
            await msg.reply_text("Please send a PDF file.")
            return ASK_CONTENT
        size_mb = (msg.document.file_size or 0) / (1024 * 1024)
        if msg.document.file_size and msg.document.file_size > MAX_PDF_SIZE_BYTES:
            await msg.reply_text(
                f"❌ Ye PDF **{size_mb:.1f} MB** ki hai — Telegram bots (bina self-hosted "
                f"local Bot API server ke) {MAX_PDF_SIZE_MB} MB se badi file download hi "
                f"nahi kar sakte. Ye Telegram ki apni hard limit hai (getFile API), is bot "
                f"ka koi bug nahi — koi bhi normal-hosted Telegram bot isse bypass nahi kar "
                f"sakta.\n\nChhoti PDF bhejo, ya usme se relevant pages nikaal ke bhejo.",
                parse_mode="Markdown",
            )
            return ASK_CONTENT
        status_msg = await msg.reply_text(f"📄 {size_mb:.1f} MB PDF mila — padh raha hoon...")
        try:
            file = await msg.document.get_file()
            data = await file.download_as_bytearray()
        except Exception as e:
            await status_msg.edit_text(f"❌ PDF download nahi ho paayi: {e}")
            return ASK_CONTENT
        b64 = base64.b64encode(bytes(data)).decode()
        context.user_data['gen_parts'] = [{
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
        }]
        try:
            await status_msg.delete()
        except Exception:
            pass  # cosmetic only — a stale/undeletable status message is harmless
        return await ask_difficulty(update, context)

    if source_type == 'image':
        if not msg.photo:
            await msg.reply_text("Please send a photo.")
            return ASK_CONTENT
        file = await msg.photo[-1].get_file()
        data = await file.download_as_bytearray()
        b64 = base64.b64encode(bytes(data)).decode()
        context.user_data['gen_parts'] = [{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        }]
        return await ask_difficulty(update, context)

    if source_type == 'section':
        if not msg.text:
            await msg.reply_text("Please send text.")
            return ASK_CONTENT
        context.user_data['gen_parts'] = [{"type": "text", "text": msg.text}]
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton(str(n), callback_data=f"sec_{n}") for n in (2, 3, 4, 5)
        ]])
        await msg.reply_text("Split into how many sections?", reply_markup=keyboard)
        return ASK_SECTION_COUNT

    return ConversationHandler.END


async def ask_difficulty(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = get_user(update.effective_user.id)
    chat_id = update.effective_chat.id
    if user['default_difficulty']:
        context.user_data['gen_difficulty'] = user['default_difficulty']
        return await ask_language(update, context)
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🟢 Easy", callback_data="diff_easy"),
        InlineKeyboardButton("🟡 Medium", callback_data="diff_medium"),
        InlineKeyboardButton("🔴 Hard", callback_data="diff_hard"),
    ]])
    await context.bot.send_message(chat_id, "Pick a difficulty:", reply_markup=keyboard)
    return ASK_DIFFICULTY


async def receive_difficulty(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data['gen_difficulty'] = query.data.replace('diff_', '')
    await query.edit_message_reply_markup(reply_markup=None)
    return await ask_language(update, context)


async def ask_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = get_user(update.effective_user.id)
    chat_id = update.effective_chat.id
    if user['default_language']:
        context.user_data['gen_language'] = user['default_language']
        return await after_language(update, context)
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("English", callback_data="lang_English"),
         InlineKeyboardButton("Hindi", callback_data="lang_Hindi")],
        [InlineKeyboardButton("✍️ Type another language", callback_data="lang_other")],
    ])
    await context.bot.send_message(chat_id, "Pick a language:", reply_markup=keyboard)
    return ASK_LANGUAGE


async def receive_language_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    if query.data == 'lang_other':
        context.user_data['awaiting_custom_language'] = True
        await context.bot.send_message(update.effective_chat.id, "Type the language name:")
        return ASK_LANGUAGE
    context.user_data['gen_language'] = query.data.replace('lang_', '')
    return await after_language(update, context)


async def receive_language_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get('awaiting_custom_language'):
        await update.message.reply_text("Please tap one of the language buttons above (or 'Other' to type your own).")
        return ASK_LANGUAGE
    context.user_data['gen_language'] = update.message.text.strip()[:30]
    context.user_data['awaiting_custom_language'] = False
    return await after_language(update, context)


async def after_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Groups get 2 extra steps (timer, then shuffle) before the count prompt, so the
    quiz can be posted live with a per-question timer and a temporary leaderboard —
    DMs skip straight to the count prompt exactly like before (no timer/leaderboard
    makes sense for a solo chat, and it keeps the DM flow fast)."""
    if update.effective_chat.type in ("group", "supergroup") and not context.user_data.get('gen_count'):
        return await ask_ai_timer(update, context)
    return await ask_count(update, context)


async def ask_ai_timer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("10s", callback_data="aitimer_10"), InlineKeyboardButton("15s", callback_data="aitimer_15"),
         InlineKeyboardButton("20s", callback_data="aitimer_20")],
        [InlineKeyboardButton("30s", callback_data="aitimer_30"), InlineKeyboardButton("60s", callback_data="aitimer_60"),
         InlineKeyboardButton("⏱ No Timer", callback_data="aitimer_0")],
    ])
    await context.bot.send_message(
        update.effective_chat.id,
        "⏱ Group me polls live post honge — har question ke liye timer chuno:",
        reply_markup=keyboard,
    )
    return ASK_AI_TIMER


async def receive_ai_timer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    context.user_data['gen_timer'] = int(query.data.replace('aitimer_', ''))
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔀 Shuffle options", callback_data="aishuffle_yes"),
        InlineKeyboardButton("↔️ Keep as-is", callback_data="aishuffle_no"),
    ]])
    await context.bot.send_message(
        update.effective_chat.id,
        "🔀 Har poll me options ka order shuffle karein?",
        reply_markup=keyboard,
    )
    return ASK_AI_SHUFFLE


async def receive_ai_shuffle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    context.user_data['gen_shuffle'] = (query.data == "aishuffle_yes")
    return await ask_count(update, context)


def shuffle_question_options(q: dict) -> dict:
    """Returns a copy of the question with its options (and correct_index) shuffled —
    the underlying MCQ content/correctness is untouched, only display order changes."""
    order = list(range(len(q["options"])))
    random.shuffle(order)
    new_options = [q["options"][i] for i in order]
    new_correct = order.index(q["correct_index"])
    shuffled = dict(q)
    shuffled["options"] = new_options
    shuffled["correct_index"] = new_correct
    return shuffled


async def ask_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if context.user_data.get('gen_count'):
        # Count was already decided upstream (e.g. spoken in a Voice → Quiz command)
        # — skip the button prompt and go straight to generation.
        await context.bot.send_message(chat_id, stylize("Generating your quiz") + "...")
        return await run_generation(update, context)
    keyboard = InlineKeyboardMarkup([[
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
        await context.bot.send_message(chat_id, "You've used your full quota. Use /redeem CODE to add more.")
        context.user_data.clear()
        return ConversationHandler.END

    daily_left = get_daily_ai_remaining(user_id)
    if daily_left is not None:
        if daily_left <= 0:
            await context.bot.send_message(
                chat_id,
                f"⏳ Aaj ka {DAILY_AI_QUESTION_LIMIT}-question AI limit khatam ho gaya — UTC midnight ke baad "
                "reset hoga. (/csv se banai gayi quiz is limit me nahi ginti.)"
            )
            context.user_data.clear()
            return ConversationHandler.END
        remaining = min(remaining, daily_left)

    count = min(context.user_data.get('gen_count', 10), remaining)
    parts = context.user_data.get('gen_parts', [])
    difficulty = context.user_data.get('gen_difficulty', 'medium')
    language = context.user_data.get('gen_language', 'English')
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

    title = "Quiz"
    for p in parts:
        if p.get('type') == 'text':
            title = p['text'][:50]
            break

    is_group_ai_flow = context.user_data.get('gen_timer') is not None
    if is_group_ai_flow:
        # Group flow: paced, timed posting + a live/temporary leaderboard, via the same
        # engine /startquiz uses — auto-saved first so it's replayable later too.
        if context.user_data.get('gen_shuffle'):
            questions = [shuffle_question_options(q) for q in questions]
        timer = context.user_data.get('gen_timer', 0)
        quiz_id = save_quiz(user_id, title, questions, timer_seconds=timer, quiz_type='ai_generated',
                             creator_name=user['tag'] or (update.effective_user.first_name or 'Unknown'))
        save_last_source(user_id, source_type, parts, difficulty, language, count)
        increment_questions_used(user_id, len(questions))
        record_daily_ai_usage(user_id, len(questions))

        timer_text = f"{timer}s / question" if timer else "no limit"
        await context.bot.send_message(
            chat_id,
            f"✅ {len(questions)} question(s) generated — Quiz ID `{quiz_id}` (saved, replay anytime with "
            f"`/startquiz {quiz_id}`). Timer: {timer_text}. Starting now...",
            parse_mode="Markdown",
        )
        await initiate_live_quiz(context.bot, context.application, chat_id, update.effective_chat.type,
                                  quiz_id, user_id, update.effective_user.first_name or "Host")
        context.user_data.clear()
        return ConversationHandler.END

    sent = await post_quiz_polls(context.bot, chat_id, questions, user_tag=user['tag'],
                                  explanation_tag=user['description_tag'], option_style=user['option_style'])
    increment_questions_used(user_id, sent)
    record_daily_ai_usage(user_id, sent)

    save_quiz(user_id, title, questions)
    save_last_source(user_id, source_type, parts, difficulty, language, count)

    await context.bot.send_message(chat_id, f"✅ Done — {sent} question(s) posted.")
    context.user_data.clear()
    return ConversationHandler.END


async def receive_section_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    n = int(query.data.replace('sec_', ''))
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user = get_user(user_id)

    full_text = context.user_data['gen_parts'][0]['text']
    chunk_size = max(1, len(full_text) // n)
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)][:n]

    daily_left = get_daily_ai_remaining(user_id)
    if daily_left is not None and daily_left <= 0:
        await context.bot.send_message(
            chat_id, f"⏳ Aaj ka {DAILY_AI_QUESTION_LIMIT}-question AI limit khatam ho gaya — UTC midnight ke baad reset hoga."
        )
        context.user_data.clear()
        return ConversationHandler.END

    await context.bot.send_message(chat_id, f"Generating {len(chunks)} section(s)...")

    total_sent = 0
    for i, chunk in enumerate(chunks, 1):
        remaining = user['question_limit'] - user['questions_used'] - total_sent
        if daily_left is not None:
            remaining = min(remaining, daily_left - total_sent)
        if remaining <= 0:
            break
        try:
            questions = await asyncio.to_thread(
                generate_mcqs_sync, user_id, [{"type": "text", "text": chunk}], min(5, remaining), 'medium', 'English'
            )
        except RuntimeError:
            continue
        tag = f"Section {i}" + (f" · {user['tag']}" if user['tag'] else '')
        sent = await post_quiz_polls(context.bot, chat_id, questions, user_tag=tag, explanation_tag=user['description_tag'])
        total_sent += sent

    increment_questions_used(user_id, total_sent)
    await context.bot.send_message(chat_id, f"✅ Done — {total_sent} question(s) across {len(chunks)} section(s).")
    context.user_data.clear()
    return ConversationHandler.END


# ==========================================
# 🎙️ VOICE → QUIZ CONVERSATION FLOW
# ==========================================
async def voice_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop('gen_count', None)
    text = (
        f"🎙️ **{stylize('Voice se Quiz')}**\n_{FONT_STYLE}_\n\n"
        "Ek voice message bhejo — bas bolo kis topic pe kitne questions chahiye.\n\n"
        "Jaise: _\"Science ke 10 sawal banao\"_ ya _\"Make 5 questions on Indian history\"_."
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
            "jaise \"History ke 5 sawal banao\"."
        )
        return ASK_VOICE

    context.user_data['gen_source_type'] = 'voice'
    context.user_data['gen_parts'] = [{"type": "text", "text": topic}]
    context.user_data['gen_count'] = count
    await msg.reply_text(
        f"🎧 Samjha: **{topic}** pe **{count}** sawal. Ab difficulty chuno.",
        parse_mode="Markdown",
    )
    return await ask_difficulty(update, context)


# ==========================================
# 📥 IMPORT FROM CHANNEL/GROUP (forward messages — no login/session needed)
# ==========================================
async def import_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['import_texts'] = []
    text = (
        f"📥 **{stylize('Import from Channel or Group')}**\n_{FONT_STYLE}_\n\n"
        "Kisi bhi channel/group ke text messages se quiz banane ke liye, unhe seedha yahan "
        "**forward** kar do — ek-ek karke, ya Telegram me multi-select karke ek saath bhi bhej sakte ho.\n\n"
        "_(Login/session string ki koi zaroorat nahi — bas jo message forward hoga, wahi content use hoga.)_\n\n"
        "Sab forward ho jaye to **✅ Done** dabao."
    )
    keyboard = cancel_keyboard([[InlineKeyboardButton("✅ Done", callback_data="import_done")]])
    await update.effective_message.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)
    return IMPORT_COLLECT


async def import_button_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    return await import_entry(update, context)


async def receive_import_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg.text:
        keyboard = cancel_keyboard([[InlineKeyboardButton("✅ Done", callback_data="import_done")]])
        await msg.reply_text("Sirf text messages forward karo (photo/PDF nahi).", reply_markup=keyboard)
        return IMPORT_COLLECT
    cleaned = _clean_poll_text(msg.text)
    if cleaned and len(cleaned) > 3:
        context.user_data.setdefault('import_texts', []).append(cleaned)
    count = len(context.user_data.get('import_texts', []))
    keyboard = cancel_keyboard([[InlineKeyboardButton("✅ Done", callback_data="import_done")]])
    await msg.reply_text(f"➕ {count} message(s) collect ho gaye. Aur forward karo, ya ✅ Done dabao.", reply_markup=keyboard)
    return IMPORT_COLLECT


async def import_done_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    texts = context.user_data.get('import_texts', [])
    if not texts:
        await query.answer("Pehle kam se kam ek message forward karo.", show_alert=True)
        return IMPORT_COLLECT
    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text(
        "🔎 Kisi specific topic/keyword pe filter karna hai? Type karke bhejo — ya *skip* likh do sab content use karne ke liye.",
        parse_mode="Markdown", reply_markup=cancel_keyboard(),
    )
    return IMPORT_KEYWORD


async def receive_import_keyword(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    texts = context.user_data.get('import_texts', [])
    keyword = (msg.text or '').strip()
    if keyword.lower() != 'skip' and keyword:
        filtered = [t for t in texts if keyword.lower() in t.lower()]
        if not filtered:
            await msg.reply_text(f"'{keyword}' se match karta hua kuch nahi mila — sab collected content use kar raha hoon.")
        else:
            texts = filtered
    combined = "\n\n".join(texts)[:15000]
    context.user_data['gen_source_type'] = 'import'
    context.user_data['gen_parts'] = [{"type": "text", "text": combined}]
    context.user_data.pop('import_texts', None)
    return await ask_difficulty(update, context)


# ==========================================
# 📊 CSV QUIZ FLOW (no AI, exact match)
# ==========================================
async def csv_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Send your CSV file.\n\n"
        "Format (first row = header):\n"
        "question,option1,option2,option3,option4,correct_option,explanation\n\n"
        "• correct_option is 1-4\n• explanation is optional",
        reply_markup=cancel_keyboard(),
    )
    return ASK_CSV


async def csv_button_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    return await csv_entry(update, context)


async def receive_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not (msg.document and msg.document.file_name and msg.document.file_name.lower().endswith('.csv')):
        await msg.reply_text("Please send a .csv file.")
        return ASK_CSV

    file = await msg.document.get_file()
    data = await file.download_as_bytearray()
    text = bytes(data).decode('utf-8', errors='replace')

    questions = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            options = [row['option1'], row['option2'], row['option3'], row['option4']]
            correct = int(row['correct_option']) - 1
            if not (0 <= correct < 4):
                correct = 0
            questions.append({
                "question": row['question'][:290],
                "options": [o[:100] for o in options],
                "correct_index": correct,
                "explanation": (row.get('explanation') or '')[:195],
            })
        except (KeyError, ValueError, TypeError):
            continue

    if not questions:
        await msg.reply_text("No valid rows found — check the CSV format and try again.")
        return ConversationHandler.END

    user_id = update.effective_user.id
    user = get_user(user_id)
    chat_id = update.effective_chat.id

    remaining = user['question_limit'] - user['questions_used']
    if remaining <= 0:
        await msg.reply_text("You've used your full quota. Use /redeem CODE to add more.")
        return ConversationHandler.END
    questions = questions[:remaining]

    sent = await post_quiz_polls(context.bot, chat_id, questions, user_tag=user['tag'], explanation_tag=user['description_tag'])
    increment_questions_used(user_id, sent)
    save_quiz(user_id, "CSV Quiz", questions)
    await msg.reply_text(f"✅ Done — {sent} question(s) posted from your CSV.")
    return ConversationHandler.END


# ==========================================
# 🔄 POLL → QUIZ CONVERSION (clean numbering / tags / links / stylish-font watermarks)
# ==========================================
# Strips leading numbering ("1.", "Q3)", "23 -"), [Tag]/#hashtag/@mention noise, any
# link (https://, t.me/..., discord.gg/...), and "stylish name" branding blocks made
# of combining marks + rare decorative Unicode scripts/symbols (the ⟵꯭᪵𓆩...🌷 style
# watermarks channels glue onto questions) — leaving only the bare question/option text.
import unicodedata

_POLL_NUMBERING_RE = re.compile(r'^\s*(?:[Qq](?:uestion|ues|uiz)?\.?\s*)?(?:\d{1,3}|[A-Da-d])\s*[\.\)\-:]\s*')
_POLL_LINK_RE = re.compile(r'(https?://\S+|www\.\S+|t\.me/\S+|discord(?:\.gg|app\.com)/\S+|@\w{4,32})', re.IGNORECASE)
_POLL_TAG_RE = re.compile(r'\[[^\]\n]{1,60}\]|\#\w+|【[^】\n]{1,60}】|〘[^〙\n]{1,60}〙')

# Unicode code-point ranges treated as "normal content" — common languages/scripts a
# real quiz question is likely written in. Everything else that's a *letter* (rare
# scripts like Meetei Mayek, Tai Viet, Kaithi, Ugaritic, IPA small-caps, etc. commonly
# reused purely to fake a "stylish font") gets dropped.
_SCRIPT_WHITELIST = (
    (0x0000, 0x024F),   # Basic Latin, Latin-1 Supplement, Latin Extended A/B
    (0x0370, 0x03FF),   # Greek
    (0x0400, 0x04FF),   # Cyrillic
    (0x0590, 0x06FF),   # Hebrew, Arabic
    (0x0900, 0x097F),   # Devanagari (Hindi)
    (0x3040, 0x30FF),   # Hiragana, Katakana
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xAC00, 0xD7A3),   # Hangul
    (0x2000, 0x206F),   # General punctuation (dashes, quotes, ellipsis...)
    (0x20A0, 0x20CF),   # Currency symbols
    (0x2200, 0x22FF),   # Mathematical operators (×÷≠≤≥ etc. — legit in quiz questions)
)


def _in_whitelist(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _SCRIPT_WHITELIST)


def _clean_poll_text(text: str) -> str:
    """Cleans a forwarded poll's question/option text: removes numbering, [Tag]/#tag
    markers, links, stylish-font "watermark" glyphs, and collapses extra whitespace."""
    if not text:
        return text
    # NFKD turns most "𝗕𝗼𝗹𝗱"/"𝘐𝘵𝘢𝘭𝘪𝘤" mathematical-alphanumeric fancy letters back
    # into plain ASCII before anything else runs.
    t = unicodedata.normalize('NFKD', text)

    kept = []
    for ch in t:
        cat = unicodedata.category(ch)
        cp = ord(ch)
        if cat == 'Cf':
            continue  # invisible joiners/format chars — always safe to drop
        if cat in ('Mn', 'Me') and not _in_whitelist(cp):
            continue  # decorative combining marks — but NOT Devanagari matras/virama etc.,
                      # which live inside the whitelisted block and must stay
        if cat == 'So' and cp > 0x2100:
            continue  # dingbats, hieroglyphs, musical/ancient/misc symbols used as decoration
        if cat == 'Sm' and not _in_whitelist(cp):
            continue  # decorative arrows/math-lookalike symbols outside real math operators
        if cat in ('Po', 'Pd', 'Ps', 'Pe', 'Pc') and cp > 0x2E00 and not _in_whitelist(cp):
            continue  # rare-script punctuation (e.g. Javanese/Meetei marks) used as decoration
        if cat.startswith('L') and not _in_whitelist(cp):
            continue  # rare/lookalike script letters used to fake a "stylish font"
        kept.append(ch)
    t = ''.join(kept)

    t = _POLL_LINK_RE.sub('', t)
    t = _POLL_TAG_RE.sub('', t)
    t = _POLL_NUMBERING_RE.sub('', t)
    t = re.sub(r'\s{2,}', ' ', t)
    t = t.strip(" -–—:.•|\u200b")
    return t or text.strip()  # never return an empty string — fall back to the raw text


async def create_quiz_from_poll_data(bot, chat_id, question, options, correct_index, user_tag: str = '', explanation_tag: str = ''):
    clean_question = _clean_poll_text(question)
    clean_options = [_clean_poll_text(o) for o in options]
    question_text = clean_question
    if user_tag:
        question_text = f"{question_text}\n\n🏷 {user_tag}"
    explanation = f"💡 {explanation_tag}" if explanation_tag else None
    await bot.send_poll(
        chat_id=chat_id,
        question=question_text[:300],
        options=[o[:100] for o in clean_options],
        type=Poll.QUIZ,
        correct_option_id=correct_index,
        explanation=explanation[:200] if explanation else None,
        is_anonymous=False,
    )
    return clean_question, clean_options


async def handle_forwarded_poll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    poll = msg.poll
    if not poll:
        return

    # If this poll was forwarded while collecting questions for an Official Quiz
    # (see officialquiz_* flow below), buffer it there instead of posting it live.
    if context.user_data.get('oq_collecting'):
        await handle_oq_forwarded_poll(update, context)
        return

    # If this poll was forwarded while the Poll → Text tool is active, buffer it
    # there instead (see polltxt_* below) — it doesn't post a quiz poll at all.
    if context.user_data.get('polltxt_collecting'):
        await handle_polltxt_forwarded_poll(update, context)
        return

    if poll.type == Poll.QUIZ and poll.is_closed and poll.correct_option_id is not None:
        user = get_user(update.effective_user.id) if update.effective_user else {"tag": "", "description_tag": ""}
        await create_quiz_from_poll_data(
            context.bot, update.effective_chat.id,
            poll.question, [o.text for o in poll.options], poll.correct_option_id,
            user_tag=user['tag'], explanation_tag=user['description_tag'],
        )
        return

    pending_id = str(msg.message_id)
    context.chat_data.setdefault('pending_polls', {})[pending_id] = {
        "question": poll.question,
        "options": [o.text for o in poll.options],
    }
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{i + 1}. {_clean_poll_text(o.text)[:30]}", callback_data=f"pollcorrect_{pending_id}_{i}")]
        for i, o in enumerate(poll.options)
    ])
    await msg.reply_text("Which option is correct?", reply_markup=keyboard)


async def handle_poll_correct_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    _, pending_id, idx = query.data.split('_', 2)
    pending = context.chat_data.get('pending_polls', {}).pop(pending_id, None)
    if not pending:
        await query.edit_message_text("This poll conversion expired — please forward the poll again.")
        return
    user = get_user(update.effective_user.id) if update.effective_user else {"tag": "", "description_tag": ""}
    await create_quiz_from_poll_data(
        context.bot, update.effective_chat.id, pending['question'], pending['options'], int(idx),
        user_tag=user['tag'], explanation_tag=user['description_tag'],
    )
    await query.edit_message_text("✅ Converted to a clean quiz poll.")


# ==========================================
# 📋 POLL → TEXT (forward any number of polls — even 500 — get one .txt back)
# ==========================================
def _polltxt_progress_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Done — Export .txt", callback_data="polltxt_done")],
        [InlineKeyboardButton("❌ Cancel", callback_data="polltxt_cancel")],
    ])


async def polltxt_start_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    context.user_data['polltxt_collecting'] = True
    context.user_data['polltxt_questions'] = []
    context.user_data.pop('polltxt_progress_msg_id', None)
    context.user_data.pop('polltxt_progress_last_edit', None)
    await query.edit_message_text(
        text=(
            f"📋 **{stylize('Poll → Text')}**\n_{FONT_STYLE}_\n\n"
            "Forward any quiz polls into this chat, one by one (or many at once — "
            "even a batch of 500). For each one I'll figure out the correct option "
            "automatically if it's a closed/answered quiz poll, or ask you to pick it.\n\n"
            "Tap **✅ Done** when you're finished and I'll send back one `.txt` file "
            "with every question, in this format:\n\n"
            "```\nQuestion text\nOption A\nOption B\nOption C ✅️\nOption D\n```"
        ),
        parse_mode="Markdown", reply_markup=_polltxt_progress_keyboard(),
    )


async def _polltxt_update_progress(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Edits a single progress message instead of sending one per forwarded poll —
    same flood-control fix used for the Official Quiz forward flow."""
    count = len(context.user_data.get('polltxt_questions', []))
    now = time.monotonic()
    last = context.user_data.get('polltxt_progress_last_edit', 0)
    msg_id = context.user_data.get('polltxt_progress_msg_id')
    text = f"➕ {count} question(s) captured so far. Forward more, or tap ✅ Done."
    if msg_id and now - last < 1.5:
        return
    try:
        if msg_id:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg_id, text=text, reply_markup=_polltxt_progress_keyboard()
            )
        else:
            sent = await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=_polltxt_progress_keyboard())
            context.user_data['polltxt_progress_msg_id'] = sent.message_id
        context.user_data['polltxt_progress_last_edit'] = now
    except Exception as e:
        logger.warning(f"Poll → Text progress update failed (non-fatal): {e}")


async def handle_polltxt_forwarded_poll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    poll = msg.poll
    if not poll:
        return
    try:
        options = [o.text for o in poll.options]
        questions = context.user_data.setdefault('polltxt_questions', [])

        if poll.type == Poll.QUIZ and poll.correct_option_id is not None:
            if not (2 <= len(options) <= 10):
                return  # skip malformed polls rather than crash the batch
            questions.append({
                "question": _clean_poll_text(poll.question),
                "options": [_clean_poll_text(o) for o in options],
                "correct_index": poll.correct_option_id,
            })
            await _polltxt_update_progress(context, msg.chat_id)
            return

        pending_id = str(msg.message_id)
        context.user_data.setdefault('polltxt_pending_polls', {})[pending_id] = {"question": poll.question, "options": options}
        pick_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(f"{i + 1}. {_clean_poll_text(o)[:30]}", callback_data=f"polltxtcorrect_{pending_id}_{i}")]
            for i, o in enumerate(options)
        ])
        await msg.reply_text("Which option is correct for this one?", reply_markup=pick_keyboard)
    except Exception as e:
        logger.warning(f"handle_polltxt_forwarded_poll error (collection preserved): {e}")


async def polltxt_poll_correct_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    try:
        _, pending_id, idx = query.data.split('_', 2)
        pending = context.user_data.get('polltxt_pending_polls', {}).pop(pending_id, None)
        if not pending:
            await query.edit_message_text("Expired — forward that poll again.")
            return
        context.user_data.setdefault('polltxt_questions', []).append({
            "question": _clean_poll_text(pending['question']),
            "options": [_clean_poll_text(o) for o in pending['options']],
            "correct_index": int(idx),
        })
        count = len(context.user_data['polltxt_questions'])
        await query.edit_message_text(f"➕ {count} question(s) captured so far. Forward more, or tap Done.")
    except Exception as e:
        logger.warning(f"polltxt_poll_correct_choice error: {e}")


async def polltxt_done_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data['polltxt_collecting'] = False
    questions = context.user_data.pop('polltxt_questions', [])
    context.user_data.pop('polltxt_pending_polls', None)
    context.user_data.pop('polltxt_progress_msg_id', None)
    if not questions:
        await query.edit_message_text(
            "You haven't forwarded any polls yet.", reply_markup=back_to_tools_keyboard()
        )
        return
    export_text = build_poll_text_export(questions)
    file_buffer = io.BytesIO(export_text.encode("utf-8"))
    file_buffer.name = "polls.txt"
    await context.bot.send_document(
        chat_id=query.message.chat_id,
        document=file_buffer,
        filename="polls.txt",
        caption=f"📋 {len(questions)} question(s) converted to text.",
    )
    await query.edit_message_text(
        f"✅ Done — sent {len(questions)} question(s) as a .txt file above.",
        reply_markup=back_to_tools_keyboard(),
    )


async def polltxt_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data['polltxt_collecting'] = False
    context.user_data.pop('polltxt_questions', None)
    context.user_data.pop('polltxt_pending_polls', None)
    context.user_data.pop('polltxt_progress_msg_id', None)
    await query.edit_message_text("❌ Cancelled — nothing was saved.", reply_markup=back_to_tools_keyboard())


# ==========================================
# 🤖 OFFICIAL QUIZ BOT (title → timer → forward/bulk → start/share → live control → leaderboard)
# ==========================================
BULK_FORMAT_HELP = (
    "Step 3/3 — send your questions in this exact format (send as many messages as "
    "you need — one big batch or several smaller ones, I'll combine them):\n\n"
    "```\n"
    "Question text here?\n"
    "Option 1\n"
    "Option 2\n"
    "Option 3 ✅️\n"
    "Option 4\n"
    "\n"
    "Next question?\n"
    "Option 1\n"
    "Option 2 ✅️\n"
    "```\n"
    "One blank line between questions, 2–4 options per question, and put the ✅️ mark "
    "right after the correct option. Tap **✅ Done** below once you're finished."
)

# Matches a ✅/✔ checkmark (with or without the emoji variation-selector) trailing an
# option line, in any of the common forms people paste: ✅️ ✅ ✔️ ✔
_CHECK_MARK_RE = re.compile(r'\s*(?:✅\uFE0F?|✔\uFE0F?)\s*$')


def _strip_check_mark(line: str) -> tuple:
    """Returns (text_without_mark, was_marked_correct)."""
    marked = bool(_CHECK_MARK_RE.search(line))
    return _CHECK_MARK_RE.sub('', line).strip(), marked


def parse_bulk_questions(text: str) -> list:
    """Question-per-block format: one blank line between questions, first line is the
    question, following lines are options (2-4), and the correct option is marked with
    a trailing ✅️. No "Q:"/"Answer:" labels needed."""
    questions = []
    blocks = re.split(r'\n\s*\n', text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if len(lines) < 3:  # need a question line + at least 2 option lines
            continue
        q_raw, _ = _strip_check_mark(lines[0])
        q_text = _clean_poll_text(q_raw)
        options = []
        correct_index = None
        for i, line in enumerate(lines[1:5]):  # cap at 4 options
            opt_raw, is_correct = _strip_check_mark(line)
            opt_raw = re.sub(r'^[A-Da-d]\s*[\.\)]\s*', '', opt_raw)  # tolerate an optional "A) " prefix
            options.append(_clean_poll_text(opt_raw))
            if is_correct:
                correct_index = i
        if len(options) < 2 or correct_index is None:
            continue
        questions.append({"question": q_text, "options": options, "correct_index": correct_index, "explanation": ""})
    return questions


def build_poll_text_export(questions: list) -> str:
    """Inverse of parse_bulk_questions — turns quiz-poll data back into the same
    checkmark text format, used by the Poll → Text tool."""
    blocks = []
    for q in questions:
        lines = [q['question']]
        for i, opt in enumerate(q['options']):
            lines.append(f"{opt} ✅️" if i == q['correct_index'] else opt)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def officialquiz_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['oq_questions'] = []
    context.user_data['oq_collecting'] = False
    await update.effective_message.reply_text(
        f"🤖 **{stylize('Official Quiz Bot')}**\n_{FONT_STYLE}_\n\nStep 1/3 — send a *title* for this quiz.",
        parse_mode="Markdown", reply_markup=cancel_keyboard(),
    )
    return OQ_TITLE


async def officialquiz_button_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    return await officialquiz_entry(update, context)


async def oq_receive_title(update: Update, context: ContextTypes.DEFAULT_TYPE):
    title = (update.message.text or "").strip()[:60]
    if not title:
        await update.message.reply_text("Please send a valid title, or tap Cancel.", reply_markup=cancel_keyboard())
        return OQ_TITLE
    context.user_data['oq_title'] = title
    keyboard = cancel_keyboard([
        [InlineKeyboardButton("10s", callback_data="oqt_10"), InlineKeyboardButton("15s", callback_data="oqt_15"),
         InlineKeyboardButton("20s", callback_data="oqt_20")],
        [InlineKeyboardButton("30s", callback_data="oqt_30"), InlineKeyboardButton("60s", callback_data="oqt_60"),
         InlineKeyboardButton("⏱ No Timer", callback_data="oqt_0")],
    ])
    await update.message.reply_text("Step 2/3 — pick a per-question timer:", reply_markup=keyboard)
    return OQ_TIMER


async def oq_receive_timer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data['oq_timer'] = int(query.data.replace("oqt_", ""))
    keyboard = cancel_keyboard([[
        InlineKeyboardButton("📥 Forward Polls", callback_data="oqm_forward"),
        InlineKeyboardButton("📝 Bulk Text Format", callback_data="oqm_bulk"),
    ]])
    await query.edit_message_text("Step 3/3 — how do you want to add questions?", reply_markup=keyboard)
    return OQ_METHOD


async def oq_receive_method(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    method = query.data.replace("oqm_", "")
    if method == "forward":
        context.user_data['oq_collecting'] = True
        keyboard = cancel_keyboard([[InlineKeyboardButton("✅ Done — Finish Quiz", callback_data="oqdone_forward")]])
        await query.edit_message_text(
            "Forward your quiz polls into this chat, one by one — numbering/tags/links are cleaned "
            "automatically. Tap **✅ Done** below once you've forwarded them all.",
            parse_mode="Markdown", reply_markup=keyboard,
        )
        return OQ_COLLECT
    await query.edit_message_text(BULK_FORMAT_HELP, parse_mode="Markdown", reply_markup=cancel_keyboard())
    return OQ_BULK


async def _oq_update_progress(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Shows collection progress by editing a single message instead of sending a new
    one per forwarded poll — sending 500 separate replies for a 500-poll batch is what
    was hitting Telegram's flood control and making the flow appear to crash."""
    count = len(context.user_data.get('oq_questions', []))
    now = time.monotonic()
    last = context.user_data.get('oq_progress_last_edit', 0)
    msg_id = context.user_data.get('oq_progress_msg_id')
    keyboard = cancel_keyboard([[InlineKeyboardButton("✅ Done — Finish Quiz", callback_data="oqdone_forward")]])
    text = f"➕ {count} question(s) added so far. Forward more, or tap ✅ Done when finished."
    if msg_id and now - last < 1.5:
        return  # throttled — the count is still accurate for the next edit or Done tap
    try:
        if msg_id:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=text, reply_markup=keyboard)
        else:
            sent = await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard)
            context.user_data['oq_progress_msg_id'] = sent.message_id
        context.user_data['oq_progress_last_edit'] = now
    except Exception as e:
        logger.warning(f"OQ progress update failed (non-fatal, collection continues): {e}")


async def handle_oq_forwarded_poll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Buffers a forwarded poll into the in-progress Official Quiz instead of
    posting it live (called from handle_forwarded_poll while oq_collecting=True)."""
    msg = update.effective_message
    poll = msg.poll
    if not poll:
        return
    try:
        options = [o.text for o in poll.options]
        questions = context.user_data.setdefault('oq_questions', [])

        if poll.type == Poll.QUIZ and poll.correct_option_id is not None:
            if not (2 <= len(options) <= 10):
                return  # silently skip malformed polls rather than crashing the batch
            questions.append({
                "question": _clean_poll_text(poll.question),
                "options": [_clean_poll_text(o) for o in options],
                "correct_index": poll.correct_option_id, "explanation": "",
            })
            await _oq_update_progress(context, msg.chat_id)
            return

        pending_id = str(msg.message_id)
        context.user_data.setdefault('oq_pending_polls', {})[pending_id] = {"question": poll.question, "options": options}
        pick_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(f"{i + 1}. {_clean_poll_text(o)[:30]}", callback_data=f"oqpollcorrect_{pending_id}_{i}")]
            for i, o in enumerate(options)
        ])
        await msg.reply_text("Which option is correct for this one?", reply_markup=pick_keyboard)
    except Exception as e:
        logger.warning(f"handle_oq_forwarded_poll error (collection preserved): {e}")


async def oq_poll_correct_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    try:
        _, pending_id, idx = query.data.split('_', 2)
        pending = context.user_data.get('oq_pending_polls', {}).pop(pending_id, None)
        if not pending:
            await query.edit_message_text("Expired — forward that poll again.")
            return OQ_COLLECT
        context.user_data.setdefault('oq_questions', []).append({
            "question": _clean_poll_text(pending['question']),
            "options": [_clean_poll_text(o) for o in pending['options']],
            "correct_index": int(idx), "explanation": "",
        })
        count = len(context.user_data['oq_questions'])
        keyboard = cancel_keyboard([[InlineKeyboardButton("✅ Done — Finish Quiz", callback_data="oqdone_forward")]])
        await query.edit_message_text(f"➕ {count} question(s) added so far. Forward more, or tap Done.", reply_markup=keyboard)
    except Exception as e:
        logger.warning(f"oq_poll_correct_choice error: {e}")
    return OQ_COLLECT


async def oq_forward_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data['oq_collecting'] = False
    if not context.user_data.get('oq_questions'):
        await query.edit_message_text("You haven't forwarded any polls yet — forward at least one, then tap Done.")
        return OQ_COLLECT
    return await oq_finalize(update, context, query=query)


async def oq_receive_bulk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Accumulates bulk text across MULTIPLE messages (a single Telegram message is
    capped at ~4096 characters, so 500 questions pasted at once would either be
    silently truncated or refused by the Telegram client) — send Q/A blocks in as many
    messages as needed, then tap ✅ Done."""
    buffer = context.user_data.get('oq_bulk_text', '')
    buffer = (buffer + "\n\n" + (update.message.text or "")).strip()
    context.user_data['oq_bulk_text'] = buffer

    try:
        questions = parse_bulk_questions(buffer)
    except Exception as e:
        logger.warning(f"parse_bulk_questions error: {e}")
        questions = []

    context.user_data['oq_questions'] = questions
    keyboard = cancel_keyboard([[InlineKeyboardButton("✅ Done — Finish Quiz", callback_data="oqdone_bulk")]])
    if not questions:
        await update.message.reply_text(
            "Couldn't find any valid questions yet — check the format above. Send more text, or Cancel.",
            reply_markup=cancel_keyboard(),
        )
    else:
        await update.message.reply_text(
            f"➕ {len(questions)} question(s) parsed so far. Send more, or tap ✅ Done when finished.",
            reply_markup=keyboard,
        )
    return OQ_BULK


async def oq_bulk_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if not context.user_data.get('oq_questions'):
        await query.edit_message_text("No valid questions parsed yet — send your Q/A text first.")
        return OQ_BULK
    return await oq_finalize(update, context, query=query)


async def oq_finalize(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    user = update.effective_user
    questions = context.user_data.get('oq_questions', [])
    title = context.user_data.get('oq_title', 'Quiz')
    timer = context.user_data.get('oq_timer', 0)
    try:
        quiz_id = save_quiz(user.id, title, questions, timer_seconds=timer, quiz_type='official',
                             creator_name=user.first_name or 'Unknown')
    except Exception as e:
        logger.error(f"oq_finalize error: {e}")
        target = query.edit_message_text if query else update.message.reply_text
        await target("⚠️ Something went wrong saving the quiz — your questions are still in this session, please tap Done again.")
        # Only oq_questions/oq_title/oq_timer are needed to retry — don't clear
        # them here, or the "tap Done again" above would be a lie and the retry
        # would always fail with "no questions parsed".
        return OQ_COLLECT if context.user_data.get('oq_collecting') is not None else OQ_BULK

    timer_text = f"{timer}s / question" if timer else "No limit"
    summary = (
        f"✅ **{stylize('Official Quiz Ready')}**\n_{FONT_STYLE}_\n\n"
        f"📝 **Title:** {title}\n"
        f"⏱ **Timer:** {timer_text}\n"
        f"🆔 **Quiz ID:** `{quiz_id}`\n"
        f"👤 **Creator:** {user.first_name} (`{user.id}`)\n"
        f"❓ **Questions:** {len(questions)}\n\n"
        f"Anyone can start it in a group or DM with `/startquiz {quiz_id}`. In a group, "
        f"it opens a waiting room until 2 people tap ✅ I Am Ready.\n"
        f"Open **🛠 Manage / Status** any time to add a description, edit it, or see who's played it."
    )
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Start Here", callback_data=f"oqstart_{quiz_id}")],
        [InlineKeyboardButton("🛠 Manage / Status", callback_data=f"stmanage_{quiz_id}"),
         InlineKeyboardButton("📤 Share Quiz ID", callback_data=f"oqshare_{quiz_id}")],
        [InlineKeyboardButton("🔙 Back to Tools", callback_data="open_tools")],
    ])
    if query:
        await query.edit_message_text(summary, parse_mode="Markdown", reply_markup=keyboard)
    else:
        await update.message.reply_text(summary, parse_mode="Markdown", reply_markup=keyboard)

    for k in ('oq_questions', 'oq_title', 'oq_timer', 'oq_collecting', 'oq_pending_polls',
              'oq_progress_msg_id', 'oq_progress_last_edit', 'oq_bulk_text'):
        context.user_data.pop(k, None)
    return ConversationHandler.END


async def oqshare_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id = query.data.replace("oqshare_", "")
    quiz = get_quiz_meta(int(quiz_id)) if quiz_id.isdigit() else None
    if not quiz:
        await query.answer("Quiz not found.", show_alert=True)
        return
    timer_text = f"{quiz['timer_seconds']}s / question" if quiz['timer_seconds'] else "No limit"
    creator_mention = f"[{quiz['creator_name'] or 'Unknown'}](tg://user?id={quiz['user_id']})"
    share_text = (
        f"🧩 **{stylize('Quiz Invite')}**\n\n📝 {quiz['title']}\n"
        + (f"💬 {quiz['description']}\n" if quiz.get('description') else "") +
        f"🆔 Quiz ID: `{quiz['id']}`\n"
        f"👤 By: {creator_mention} (`{quiz['user_id']}`)\n"
        f"⏱ {timer_text} · ❓ {len(quiz['questions'])} questions\n\n"
        f"Start it in any group or DM by sending:\n`/startquiz {quiz['id']}`"
    )
    await context.bot.send_message(query.message.chat_id, share_text, parse_mode="Markdown")


# --- Live quiz session engine (in-memory per running process; the quiz itself and
# every finished leaderboard are always saved to the database, so nothing about the
# quiz content or results is ever lost even if the bot restarts mid-session). ---
async def start_live_quiz(bot, application, chat_id, quiz_id, started_by_id, started_by_name):
    quiz = get_quiz_meta(quiz_id)
    if not quiz:
        await bot.send_message(chat_id, "❌ Quiz not found — check the Quiz ID.")
        return
    if not quiz['questions']:
        await bot.send_message(chat_id, "❌ This quiz has no questions.")
        return
    live = application.bot_data.setdefault('live_quizzes', {})
    existing = live.get(chat_id)
    if existing and not existing.get('finished', True):
        await bot.send_message(chat_id, "⚠️ A quiz is already running in this chat — stop it first.")
        return

    creator = get_user(quiz['user_id'])  # the quiz AUTHOR's tag/description/option-style — applied
    # to every question in this run, same as everywhere else the quiz is delivered.
    session = {
        "quiz_id": quiz_id, "questions": quiz['questions'], "title": quiz['title'],
        "timer": quiz['timer_seconds'], "index": 0, "paused": False, "stopped": False,
        "undo_requested": False, "finished": False,
        "creator_id": started_by_id, "creator_name": started_by_name,
        "author_id": quiz['user_id'], "author_name": quiz['creator_name'] or "Unknown",
        "author_tag": creator['tag'], "author_desc_tag": creator['description_tag'],
        "author_option_style": creator['option_style'],
        "poll_map": {}, "poll_sent_at": {}, "scores": {}, "last_poll_msg_id": None,
        "current_poll_id": None, "answers_this_question": 0, "consecutive_silent": 0,
    }
    live[chat_id] = session

    author_mention = f"[{session['author_name']}](tg://user?id={session['author_id']})"
    timer_text = f"{quiz['timer_seconds']}s / question" if quiz['timer_seconds'] else "No limit"
    await bot.send_message(
        chat_id,
        f"🚦 **{stylize('Official Quiz — Live')}**\n📝 {quiz['title']}\n"
        + (f"💬 {quiz['description']}\n" if quiz.get('description') else "") +
        f"👤 By: {author_mention} (`{session['author_id']}`)\n"
        f"⏱ {timer_text} · ❓ {len(quiz['questions'])} questions\n\n"
        f"Only {started_by_name} or a chat admin can control this — with:\n"
        f"⏸ /pause · ▶️ /resume · ⏹ /stopquiz · ↩️ /undoquiz\n"
        f"🐇 /fastquiz (speed up) · 🐢 /slowquiz (slow down)",
        parse_mode="Markdown",
    )
    application.create_task(_run_live_quiz_loop(bot, application, chat_id))


# ==========================================
# 🚪 LOBBY — starting an Official Quiz in a GROUP waits for at least 2 people to tap
# "I Am Ready" before any question is fired (DMs skip this — no lobby needed for one
# person). Mirrors how popular live-quiz bots avoid firing into an empty room.
# ==========================================
LOBBY_MIN_READY = 2
LOBBY_TIMEOUT_SECONDS = 10 * 60  # never wait forever for players who never show up


def _lobby_view(chat_id, lobby):
    ready_count = len(lobby['ready_users'])
    quiz = lobby['quiz']
    timer_text = f"{quiz['timer_seconds']}s / question" if quiz['timer_seconds'] else "No limit"
    author_mention = f"[{quiz['creator_name'] or 'Unknown'}](tg://user?id={quiz['user_id']})"
    text = (
        f"🚪 **{stylize('Waiting Room')}**\n📝 {quiz['title']}\n"
        + (f"💬 {quiz['description']}\n" if quiz.get('description') else "") +
        f"👤 By: {author_mention} (`{quiz['user_id']}`)\n"
        f"⏱ {timer_text} · ❓ {len(quiz['questions'])} questions\n\n"
        f"Waiting for at least {LOBBY_MIN_READY} players — tap below when you're ready.\n"
        f"✅ Ready: {ready_count}/{LOBBY_MIN_READY}"
    )
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"✅ I Am Ready ({ready_count})", callback_data=f"oqready_{chat_id}")],
        [InlineKeyboardButton("⏹ Cancel", callback_data=f"oqreadycancel_{chat_id}")],
    ])
    return text, keyboard


async def initiate_live_quiz(bot, application, chat_id, chat_type, quiz_id, requested_by_id, requested_by_name):
    """Entry point for both /startquiz and the ▶️ Start Here button. In a group this
    opens the ready-check lobby instead of firing questions immediately; in a DM (a
    solo quiz for one person) the lobby would just be a pointless extra tap, so it
    starts right away."""
    quiz = get_quiz_meta(quiz_id)
    if not quiz:
        await bot.send_message(chat_id, "❌ Quiz not found — check the Quiz ID.")
        return
    if not quiz['questions']:
        await bot.send_message(chat_id, "❌ This quiz has no questions.")
        return

    if chat_type not in ("group", "supergroup"):
        await start_live_quiz(bot, application, chat_id, quiz_id, requested_by_id, requested_by_name)
        return

    live = application.bot_data.setdefault('live_quizzes', {})
    existing = live.get(chat_id)
    if existing and not existing.get('finished', True):
        await bot.send_message(chat_id, "⚠️ A quiz is already running in this chat — stop it first.")
        return
    lobbies = application.bot_data.setdefault('quiz_lobbies', {})
    if chat_id in lobbies:
        await bot.send_message(chat_id, "⚠️ A waiting room is already open in this chat — tap ✅ I Am Ready above.")
        return

    lobbies[chat_id] = {
        "quiz": quiz, "quiz_id": quiz_id, "requested_by_id": requested_by_id,
        "requested_by_name": requested_by_name, "ready_users": set(), "message_id": None,
    }
    text, keyboard = _lobby_view(chat_id, lobbies[chat_id])
    msg = await bot.send_message(chat_id, text, parse_mode="Markdown", reply_markup=keyboard)
    lobbies[chat_id]["message_id"] = msg.message_id
    application.create_task(_lobby_timeout_watch(bot, application, chat_id))


async def _lobby_timeout_watch(bot, application, chat_id):
    await asyncio.sleep(LOBBY_TIMEOUT_SECONDS)
    lobbies = application.bot_data.get('quiz_lobbies', {})
    if chat_id in lobbies:
        lobbies.pop(chat_id, None)
        try:
            await bot.send_message(chat_id, "🚪 Waiting room closed — not enough players joined in time. Start it again anytime.")
        except Exception as e:
            logger.warning(f"Lobby timeout notice failed (non-fatal): {e}")


async def lobby_ready_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = int(query.data.replace("oqready_", ""))
    lobbies = context.application.bot_data.get('quiz_lobbies', {})
    lobby = lobbies.get(chat_id)
    if not lobby:
        await query.answer("This waiting room has closed.", show_alert=True)
        return
    lobby['ready_users'].add(query.from_user.id)
    await query.answer("✅ You're marked ready!")

    if len(lobby['ready_users']) >= LOBBY_MIN_READY:
        lobbies.pop(chat_id, None)
        try:
            await query.edit_message_text(f"✅ {LOBBY_MIN_READY}+ players ready — starting now!")
        except Exception as e:
            logger.warning(f"lobby_ready_callback edit failed (non-fatal): {e}")
        await start_live_quiz(context.bot, context.application, chat_id, lobby['quiz_id'],
                               lobby['requested_by_id'], lobby['requested_by_name'])
        return

    text, keyboard = _lobby_view(chat_id, lobby)
    try:
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)
    except Exception as e:
        logger.warning(f"lobby_ready_callback edit failed (non-fatal): {e}")


async def lobby_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = int(query.data.replace("oqreadycancel_", ""))
    lobbies = context.application.bot_data.get('quiz_lobbies', {})
    lobby = lobbies.get(chat_id)
    if not lobby:
        await query.answer("Already closed.", show_alert=True)
        return
    is_admin = await is_chat_admin(update, context)
    if query.from_user.id != lobby['requested_by_id'] and not is_admin:
        await query.answer("Only whoever started this, or a chat admin, can cancel it.", show_alert=True)
        return
    lobbies.pop(chat_id, None)
    await query.answer("Cancelled.")
    try:
        await query.edit_message_text("⏹ Waiting room cancelled.")
    except Exception as e:
        logger.warning(f"lobby_cancel_callback edit failed (non-fatal): {e}")


# A session left "paused" forever (host went AFK, phone died, etc.) used to keep this
# background task alive indefinitely — a real, silent stuck-loop. Auto-stop after this
# many seconds of continuous pause instead, so nothing about the bot is ever "stuck".
MAX_LIVE_QUIZ_PAUSE_SECONDS = 30 * 60
# If 2 questions in a row get zero answers from anyone, the room is probably empty —
# auto-pause instead of blindly firing the rest of the quiz into silence.
SILENT_QUESTIONS_BEFORE_AUTOPAUSE = 2


async def _run_live_quiz_loop(bot, application, chat_id):
    live = application.bot_data.get('live_quizzes', {})
    session = live.get(chat_id)
    if not session:
        return
    try:
        while session['index'] < len(session['questions']) and not session['stopped']:
            paused_elapsed = 0
            while session['paused'] and not session['stopped']:
                await asyncio.sleep(1)
                paused_elapsed += 1
                if paused_elapsed >= MAX_LIVE_QUIZ_PAUSE_SECONDS:
                    session['stopped'] = True
                    try:
                        await bot.send_message(
                            chat_id,
                            "⏹ Auto-stopped — this quiz stayed paused too long without a Resume. "
                            "Here's the leaderboard for what was played so far.",
                        )
                    except Exception as e:
                        logger.warning(f"Live quiz auto-stop notice failed: {e}")
                    break
            if session['stopped']:
                break

            q = session['questions'][session['index']]
            question_text = f"Q{session['index'] + 1}. {q['question']}"
            if session.get('author_tag'):
                question_text = f"{question_text} 🏷 {session['author_tag']}"
            explanation = q.get('explanation', '')
            if session.get('author_desc_tag'):
                explanation = f"{explanation}\n\n💡 {session['author_desc_tag']}".strip() if explanation else f"💡 {session['author_desc_tag']}"
            try:
                msg = await bot.send_poll(
                    chat_id=chat_id,
                    question=question_text[:300],
                    options=apply_option_style(q['options'], session.get('author_option_style', 'plain')),
                    type=Poll.QUIZ,
                    correct_option_id=q['correct_index'],
                    explanation=(explanation[:200] or None),
                    is_anonymous=False,
                    open_period=session['timer'] if session['timer'] else None,
                )
                session['poll_map'][msg.poll.id] = session['index']
                session['poll_sent_at'][msg.poll.id] = time.monotonic()
                session['last_poll_msg_id'] = msg.message_id
                session['current_poll_id'] = msg.poll.id
                session['answers_this_question'] = 0
            except Exception as e:
                logger.warning(f"Live quiz poll send failed (skipping to next question): {e}")
                session['current_poll_id'] = None

            session['index'] += 1
            # A promo message every 5 questions, mid-quiz only — same rule as post_quiz_polls,
            # so it behaves identically whichever way a quiz was made or is being delivered.
            if session['index'] % PROMO_EVERY_N_QUESTIONS == 0 and session['index'] < len(session['questions']):
                await send_promo_message(bot, chat_id)

            wait_seconds = (session['timer'] or 15) + 2
            elapsed = 0
            undone = False
            while elapsed < wait_seconds:
                if session['stopped']:
                    break
                if session.get('undo_requested'):
                    session['undo_requested'] = False
                    session['index'] = max(0, session['index'] - 1)  # re-ask the question just posted
                    undone = True
                    break
                if not session['paused']:
                    elapsed += 1
                await asyncio.sleep(1)
            if undone:
                try:
                    if session['last_poll_msg_id']:
                        await bot.stop_poll(chat_id, session['last_poll_msg_id'])
                except Exception:
                    pass
            elif not session['stopped']:
                # Nobody answered this question at all — probably means the room emptied
                # out. Two of these in a row auto-pauses instead of firing blindly into
                # silence; a real answer at any point resets the streak back to zero.
                if session['answers_this_question'] == 0:
                    session['consecutive_silent'] += 1
                    if session['consecutive_silent'] >= SILENT_QUESTIONS_BEFORE_AUTOPAUSE:
                        session['paused'] = True
                        session['consecutive_silent'] = 0
                        try:
                            await bot.send_message(
                                chat_id,
                                f"⏸ Auto-paused — the last {SILENT_QUESTIONS_BEFORE_AUTOPAUSE} questions got no "
                                "answers. Tap ▶️ Resume above whenever you're ready to continue.",
                            )
                        except Exception as e:
                            logger.warning(f"Auto-pause notice failed (non-fatal): {e}")
                else:
                    session['consecutive_silent'] = 0

        if not session['stopped']:
            await _finish_live_quiz(bot, chat_id, application)
    except Exception as e:
        # A bug anywhere above must never leave this chat silently stuck thinking a
        # quiz is still "live" with no loop actually driving it — always clean up and
        # still show whatever leaderboard data was collected so far.
        logger.error(f"Live quiz loop crashed for chat {chat_id} — auto-recovering: {e}", exc_info=True)
        try:
            await _finish_live_quiz(bot, chat_id, application)
        except Exception as e2:
            logger.error(f"Live quiz recovery also failed for chat {chat_id}, forcing cleanup: {e2}")
            live.pop(chat_id, None)


async def _finish_live_quiz(bot, chat_id, application):
    live = application.bot_data.get('live_quizzes', {})
    session = live.get(chat_id)
    if not session or session.get('finished'):
        return
    session['finished'] = True
    if session.get('last_poll_msg_id'):
        try:
            await bot.stop_poll(chat_id, session['last_poll_msg_id'])
        except Exception:
            pass

    total_questions = len(session['questions'])

    def _sort_key(s):
        # Rank by score, then accuracy, then average answer speed (faster = better) —
        # so an exact score tie still resolves to one unambiguous, correct position.
        avg_ms = (s['total_time_ms'] / s['answered']) if s['answered'] else float('inf')
        accuracy = (s['correct'] / s['answered']) if s['answered'] else 0
        return (-s['score'], -accuracy, avg_ms)

    ranking = sorted(session['scores'].values(), key=_sort_key)
    if ranking:
        medals = ["🥇", "🥈", "🥉"]
        lines = []
        for i, s in enumerate(ranking[:20]):
            avg_sec = round((s['total_time_ms'] / s['answered']) / 1000, 1) if s['answered'] else 0.0
            accuracy = round((s['correct'] / s['answered']) * 100, 1) if s['answered'] else 0.0
            rank_tag = medals[i] if i < 3 else f"{i + 1}."
            lines.append(
                f"{rank_tag} {s['name']} — `{s['user_id']}`\n"
                f"      {s['score']} pts · ✅ {s['correct']} ❌ {s['wrong']} · 🎯 {accuracy}% acc · ⏱ {avg_sec}s avg"
            )
        board_text = (
            f"🏆 **{stylize('Official Quiz Leaderboard')}** — {session['title']}\n"
            f"_{FONT_STYLE}_\n\n" + "\n".join(lines) +
            "\n\n👤 User ID is shown next to each name so the host can reach out directly (prizes, verification, etc.)."
        )
    else:
        board_text = f"🏆 **{stylize('Official Quiz Leaderboard')}** — {session['title']}\n\nNo answers were recorded."
    try:
        await bot.send_message(chat_id, board_text, parse_mode="Markdown")
    except Exception as e:
        logger.warning(f"Live quiz leaderboard send failed: {e}")

    for s in ranking:
        avg_ms = int(s['total_time_ms'] / s['answered']) if s['answered'] else 0
        try:
            save_quiz_result(
                session['quiz_id'], chat_id, s['user_id'], s['name'], s['score'], s['correct'],
                wrong_count=s['wrong'], total_questions=total_questions, avg_time_ms=avg_ms,
            )
        except Exception as e:
            logger.warning(f"save_quiz_result failed for user {s['user_id']}: {e}")
    live.pop(chat_id, None)


async def handle_poll_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ans = update.poll_answer
    if not ans.option_ids:
        return  # user retracted their answer
    live = context.application.bot_data.get('live_quizzes', {})
    for chat_id, session in live.items():
        if ans.poll_id in session['poll_map']:
            q_index = session['poll_map'][ans.poll_id]
            question = session['questions'][q_index]
            correct = question['correct_index'] in ans.option_ids

            # Telegram doesn't timestamp poll_answer updates, so response time is
            # measured as (now − when we sent that poll) — accurate to within normal
            # network/delivery latency, and clamped to the question's own time window
            # so a stray late update can never distort the average.
            sent_at = session.get('poll_sent_at', {}).get(ans.poll_id)
            timer_cap_ms = ((session['timer'] or 15) + 2) * 1000
            elapsed_ms = int((time.monotonic() - sent_at) * 1000) if sent_at else 0
            elapsed_ms = max(0, min(elapsed_ms, timer_cap_ms))

            entry = session['scores'].setdefault(
                ans.user.id,
                {"user_id": ans.user.id, "name": ans.user.first_name or "Player", "score": 0,
                 "correct": 0, "wrong": 0, "total_time_ms": 0, "answered": 0},
            )
            entry['name'] = ans.user.first_name or entry['name']
            entry['answered'] += 1
            entry['total_time_ms'] += elapsed_ms
            if correct:
                entry['score'] += 10
                entry['correct'] += 1
            else:
                entry['wrong'] += 1

            # Feeds the auto-pause-on-silence check in _run_live_quiz_loop — only counts
            # toward the *current* question, so a late answer to an older one can't
            # wrongly mark the current question as "answered".
            if ans.poll_id == session.get('current_poll_id'):
                session['answers_this_question'] = session.get('answers_this_question', 0) + 1
            return  # handled as part of a live-quiz session — done

    # Not part of any in-memory live-quiz session — check whether this is an AI
    # Mode broadcast poll instead (DB-backed, see ai_mode_polls in init_db, so
    # scoring still works even if the bot restarted between sending it and now).
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT chat_id, correct_option_id FROM ai_mode_polls WHERE poll_id=?", (ans.poll_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return
    ai_chat_id, correct_option_id = row
    correct = correct_option_id in ans.option_ids
    record_ai_mode_answer(ai_chat_id, ans.user.id, ans.user.first_name or "Player", correct)


async def oqstart_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id_str = query.data.replace("oqstart_", "")
    if not quiz_id_str.isdigit():
        return
    user = query.from_user
    await initiate_live_quiz(context.bot, context.application, query.message.chat_id, query.message.chat.type,
                              int(quiz_id_str), user.id, user.first_name or "Host")


async def startquiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usage: /startquiz QUIZ_ID")
        return
    user = update.effective_user
    await initiate_live_quiz(context.bot, context.application, update.effective_chat.id, update.effective_chat.type,
                              int(context.args[0]), user.id, user.first_name or "Host")


# ==========================================
# 🏷️ TAG / REGENERATE / REDEEM / STORE / DEFAULT
# ==========================================
async def settag_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /settag Your Tag Here")
        return
    tag = " ".join(context.args)[:60]
    set_user_field(update.effective_user.id, 'tag', tag)
    await update.message.reply_text(f"🏷 Tag set: {tag}", reply_markup=back_to_tools_keyboard())


async def setdescription_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setdescription Your Tag Here")
        return
    tag = " ".join(context.args)[:60]
    set_user_field(update.effective_user.id, 'description_tag', tag)
    await update.message.reply_text(f"💡 Description tag set: {tag}", reply_markup=back_to_tools_keyboard())


async def do_regenerate(bot, chat_id, user_id):
    last = get_last_source(user_id)
    if not last:
        await bot.send_message(chat_id, "Nothing to regenerate yet — create a quiz first.")
        return
    user = get_user(user_id)
    remaining = user['question_limit'] - user['questions_used']
    if remaining <= 0:
        await bot.send_message(chat_id, "You've used your full quota. Use /redeem CODE to add more.")
        return

    daily_left = get_daily_ai_remaining(user_id)
    if daily_left is not None:
        if daily_left <= 0:
            await bot.send_message(
                chat_id,
                f"⏳ Aaj ka {DAILY_AI_QUESTION_LIMIT}-question AI limit khatam ho gaya — UTC midnight ke baad reset hoga."
            )
            return
        remaining = min(remaining, daily_left)

    count = min(last['count'], remaining)
    try:
        questions = await generate_mcqs_async(user_id, last['parts'], count, last['difficulty'], last['language'])
    except RuntimeError as e:
        await report_generation_failure(bot, chat_id, user_id, e)
        return
    sent = await post_quiz_polls(bot, chat_id, questions, user_tag=user['tag'],
                                  explanation_tag=user['description_tag'], option_style=user['option_style'])
    increment_questions_used(user_id, sent)
    record_daily_ai_usage(user_id, sent)
    await bot.send_message(chat_id, f"✅ Done — {sent} fresh question(s) posted.")


async def regenerate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await do_regenerate(context.bot, update.effective_chat.id, update.effective_user.id)


async def redeem_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /redeem CODE")
        return
    result = redeem_code(update.effective_user.id, context.args[0].strip().upper())
    await update.message.reply_text(result)


# ==========================================
# 🔤 OPTION STYLE — how MCQ options are labelled (plain / A) B) C) D) / 1) 2) 3) 4))
# Every user (and the owner) sets their own — applied everywhere a quiz is posted:
# AI-generated (post_quiz_polls), Official Quiz live (_run_live_quiz_loop), and
# single-poll conversions (create_quiz_from_poll_data).
# ==========================================
def _option_style_keyboard(current: str):
    def label(key, text):
        return f"✅ {text}" if current == key else text
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(label("plain", OPTION_STYLE_LABELS["plain"]), callback_data="optstyle_plain")],
        [InlineKeyboardButton(label("abcd", OPTION_STYLE_LABELS["abcd"]), callback_data="optstyle_abcd")],
        [InlineKeyboardButton(label("numeric", OPTION_STYLE_LABELS["numeric"]), callback_data="optstyle_numeric")],
    ])


async def optionstyle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = get_user(update.effective_user.id)
    await update.message.reply_text(
        "🔤 **Option Style** — how answer options are labelled in every quiz you make:",
        parse_mode="Markdown", reply_markup=_option_style_keyboard(user['option_style']),
    )


async def optionstyle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    style = query.data.replace("optstyle_", "")
    if style not in OPTION_STYLE_LABELS:
        return
    set_user_field(update.effective_user.id, 'option_style', style)
    await query.edit_message_text(
        f"🔤 **Option Style** — set to **{OPTION_STYLE_LABELS[style]}**. Applies from your next quiz onward.",
        parse_mode="Markdown", reply_markup=_option_style_keyboard(style),
    )


# ==========================================
# 🎁 OWNER: free/unlimited AI access for specific people (friends, testers, etc.)
# ==========================================
async def freeuser_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return
    if not context.args or not context.args[0].lstrip('-').isdigit():
        await update.message.reply_text(
            "Usage:\n`/freeuser <user_id>` — make them unlimited (no 200/day AI cap)\n"
            "`/freeuser <user_id> off` — put the daily cap back",
            parse_mode="Markdown",
        )
        return
    target_id = int(context.args[0])
    turn_off = len(context.args) > 1 and context.args[1].lower() == "off"
    set_ai_unlimited(target_id, not turn_off)
    if turn_off:
        await update.message.reply_text(f"↩️ `{target_id}` is back on the normal {DAILY_AI_QUESTION_LIMIT}/day AI limit.", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"🎁 `{target_id}` now has **unlimited** AI generation — no daily cap.", parse_mode="Markdown")


# ==========================================
# 📣 OWNER: the promo message shown every 5 questions in every quiz
# ==========================================
async def setpromo_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return
    parts = update.message.text.split(maxsplit=1)
    if len(parts) < 2:
        current = get_bot_setting("promo_text", "")
        await update.message.reply_text(
            "Usage: `/setpromo Your promo text/link here`\n`/setpromo off` — turn it off\n\n"
            f"Current: {current or '(not set — no promo is shown)'}",
            parse_mode="Markdown",
        )
        return
    text = parts[1].strip()
    if text.lower() == "off":
        set_bot_setting("promo_text", "")
        await update.message.reply_text("📣 Promo message turned off.")
        return
    set_bot_setting("promo_text", text[:500])
    await update.message.reply_text(f"📣 Promo set — will appear every {PROMO_EVERY_N_QUESTIONS} questions in every quiz:\n\n{text[:500]}")


async def addcode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /addcode CODE BONUS_QUESTIONS [MAX_USES] [VALIDITY_DAYS]\n"
            "MAX_USES 0 = unlimited redemptions. VALIDITY_DAYS omitted = never expires."
        )
        return
    code = context.args[0].strip().upper()
    try:
        bonus = int(context.args[1])
        max_uses = int(context.args[2]) if len(context.args) > 2 else 0
        validity_days = int(context.args[3]) if len(context.args) > 3 else 0
    except ValueError:
        await update.message.reply_text("BONUS_QUESTIONS, MAX_USES and VALIDITY_DAYS must be numbers.")
        return
    expires_at = (datetime.utcnow() + timedelta(days=validity_days)).isoformat() if validity_days > 0 else None
    conn = db_connect()
    cur = conn.cursor()
    upsert(cur, "redeem_codes", {"code": code},
           {"bonus_questions": bonus, "max_uses": max_uses, "used_count": 0, "expires_at": expires_at})
    conn.commit()
    conn.close()
    validity_txt = f"expires in {validity_days} day(s)" if expires_at else "never expires"
    await update.message.reply_text(
        f"✅ Code {code} created: +{bonus} questions, max uses: {max_uses or '∞'}, {validity_txt}"
    )


def get_top_solvers(limit: int = 10) -> list:
    """Ranks users by total correctly-answered questions across every finished
    Official Quiz session saved in quiz_results (the only place per-question
    correctness is tracked). Also rolls up wrong-count/accuracy so every place
    this list is shown can display user_id + accuracy, not just a raw count."""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT user_id, MAX(user_name), SUM(correct_count), SUM(wrong_count) "
        "FROM quiz_results GROUP BY user_id ORDER BY SUM(correct_count) DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        correct, wrong = r[2] or 0, r[3] or 0
        total = correct + wrong
        accuracy = round((correct / total) * 100, 1) if total else 0.0
        out.append({"user_id": r[0], "user_name": r[1] or "Unknown", "total_correct": correct,
                     "total_wrong": wrong, "accuracy": accuracy})
    return out


# ==========================================
# 👤 /profile — personal stats, streak, and badges (any chat, always about the
# caller themselves — cross-group, all-time, from the same quiz_results table
# /leaderboard and /topusers already use).
# ==========================================
BADGE_THRESHOLDS = [
    (1000, "💎 Legend — 1000+ correct answers"),
    (500, "🏆 Champion — 500+ correct answers"),
    (200, "🥇 Expert — 200+ correct answers"),
    (50, "🥈 Regular — 50+ correct answers"),
    (10, "🥉 Getting Started — 10+ correct answers"),
]
STREAK_BADGE_THRESHOLDS = [
    (30, "🔥 30-Day Streak"),
    (7, "🔥 7-Day Streak"),
    (3, "🔥 3-Day Streak"),
]


def get_user_profile_stats(user_id: int) -> dict:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*), SUM(correct_count), SUM(wrong_count), SUM(total_questions) "
        "FROM quiz_results WHERE user_id=?", (user_id,)
    )
    row = cur.fetchone()
    quizzes_played = row[0] or 0
    correct = row[1] or 0
    wrong = row[2] or 0
    total_q = row[3] or 0
    cur.execute("SELECT streak_count, question_limit, questions_used FROM users WHERE user_id=?", (user_id,))
    urow = cur.fetchone()
    conn.close()
    streak = (urow[0] or 0) if urow else 0
    total = correct + wrong
    accuracy = round((correct / total) * 100, 1) if total else 0.0
    badges = [label for threshold, label in BADGE_THRESHOLDS if correct >= threshold][:1]
    badges += [label for threshold, label in STREAK_BADGE_THRESHOLDS if streak >= threshold][:1]
    return {
        "quizzes_played": quizzes_played, "correct": correct, "wrong": wrong,
        "total_questions": total_q, "accuracy": accuracy, "streak": streak, "badges": badges,
    }


async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_user = update.effective_user
    stats = get_user_profile_stats(target_user.id)
    daily_left = get_daily_ai_remaining(target_user.id)
    limit_line = "♾️ Unlimited" if daily_left is None else f"{daily_left}/{DAILY_AI_QUESTION_LIMIT}"
    badges_text = "  ".join(stats["badges"]) if stats["badges"] else "_None yet — keep playing quizzes to earn one!_"
    text = (
        f"👤 **{target_user.first_name}'s Profile**\n\n"
        f"🎯 Quizzes played: {stats['quizzes_played']}\n"
        f"✅ Correct: {stats['correct']}  ❌ Wrong: {stats['wrong']}\n"
        f"📊 Accuracy: {stats['accuracy']}%\n"
        f"🔥 Current streak: {stats['streak']} day(s)\n"
        f"🎟️ AI quota left today: {limit_line}\n\n"
        f"🏅 **Badges:** {badges_text}\n\n"
        f"_Tip: play any Official Quiz or group AI quiz to build your streak — one finished "
        f"quiz per UTC day keeps it going._"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def topusers_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    top = get_top_solvers(10)
    if not top:
        await update.message.reply_text("No quiz results yet — leaderboard fills in as people play Official Quizzes.")
        return
    medals = ["🥇", "🥈", "🥉"] + ["▫️"] * 7
    lines = [f"🏆 **{stylize('Top 10 — Most Questions Solved')}**\n_{FONT_STYLE}_\n"]
    for i, u in enumerate(top):
        lines.append(
            f"{medals[i]} {u['user_name']} — `{u['user_id']}`\n"
            f"      {u['total_correct']} correct · {u['total_wrong']} wrong · {u['accuracy']}% accuracy"
        )
    lines.append("\n_Tip: try_ `/leaderboard` _for per-group Points / Accuracy / Today views._")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ==========================================
# 🏆 /leaderboard — multi-mode leaderboard (Points · Accuracy · Today)
# Scoped to the current group when run in one; global (all chats) in DM.
# Every mode always shows user_id next to the name, per-quiz-result, so a host
# can identify/contact any participant directly.
# ==========================================
def _leaderboard_rows(chat_id, scope: str = "all"):
    """Aggregates quiz_results per user, optionally scoped to one chat_id and/or to
    'today' (UTC). Filtering 'today' in Python (not SQL) keeps this correct on both
    SQLite and Postgres without relying on either engine's date functions."""
    conn = db_connect()
    cur = conn.cursor()
    if chat_id is not None:
        cur.execute(
            "SELECT user_id, user_name, score, correct_count, wrong_count, finished_at "
            "FROM quiz_results WHERE chat_id=? ORDER BY id DESC LIMIT 5000",
            (chat_id,)
        )
    else:
        cur.execute(
            "SELECT user_id, user_name, score, correct_count, wrong_count, finished_at "
            "FROM quiz_results ORDER BY id DESC LIMIT 5000"
        )
    rows = cur.fetchall()
    conn.close()

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    agg = {}
    for user_id, name, score, correct, wrong, finished_at in rows:
        if scope == "today" and not (finished_at or "").startswith(today_str):
            continue
        e = agg.setdefault(user_id, {"user_id": user_id, "name": name or "Unknown",
                                      "points": 0, "correct": 0, "wrong": 0})
        e["name"] = name or e["name"]
        e["points"] += score or 0
        e["correct"] += correct or 0
        e["wrong"] += wrong or 0
    return list(agg.values())


def build_leaderboard_text(chat_id, mode: str = "points") -> str:
    rows = _leaderboard_rows(chat_id, scope="today" if mode == "today" else "all")
    if mode == "accuracy":
        rows = [r for r in rows if (r["correct"] + r["wrong"]) >= 3]  # needs a few attempts to be meaningful
        rows.sort(key=lambda r: (r["correct"] / (r["correct"] + r["wrong"])), reverse=True)
        title = "🎯 Accuracy Leaderboard (min. 3 answers)"
    elif mode == "today":
        rows.sort(key=lambda r: -r["points"])
        title = f"📅 Today's Leaderboard ({datetime.utcnow().strftime('%Y-%m-%d')} UTC)"
    else:
        rows.sort(key=lambda r: -r["points"])
        title = "🏆 Points Leaderboard"

    if not rows:
        return f"**{stylize(title)}**\n\nNo quiz results yet here."

    medals = ["🥇", "🥈", "🥉"]
    lines = [f"**{stylize(title)}**\n_{FONT_STYLE}_\n"]
    for i, r in enumerate(rows[:10]):
        total = r["correct"] + r["wrong"]
        accuracy = round((r["correct"] / total) * 100, 1) if total else 0.0
        rank_tag = medals[i] if i < 3 else f"{i + 1}."
        lines.append(
            f"{rank_tag} {r['name']} — `{r['user_id']}`\n"
            f"      {r['points']} pts · ✅ {r['correct']} ❌ {r['wrong']} · 🎯 {accuracy}%"
        )
    return "\n".join(lines)


def _leaderboard_mode_keyboard(chat_id):
    tag = "g" if chat_id is not None else "dm"
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🏆 Points", callback_data=f"lbmode_points_{tag}"),
        InlineKeyboardButton("🎯 Accuracy", callback_data=f"lbmode_accuracy_{tag}"),
        InlineKeyboardButton("📅 Today", callback_data=f"lbmode_today_{tag}"),
    ]])


async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    chat_id = chat.id if chat.type in ("group", "supergroup") else None
    text = build_leaderboard_text(chat_id, "points")
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=_leaderboard_mode_keyboard(chat_id))


async def leaderboard_mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    try:
        _, mode, tag = query.data.split("_", 2)
    except ValueError:
        return
    chat_id = query.message.chat_id if tag == "g" else None
    text = build_leaderboard_text(chat_id, mode)
    try:
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=_leaderboard_mode_keyboard(chat_id))
    except Exception as e:
        logger.warning(f"leaderboard_mode_callback edit failed (non-fatal): {e}")


# ==========================================
# 🤺 /duel — 1v1 challenge. Reply to your opponent's message with
# /duel <topic> to start a short 5-question head-to-head, run through the exact
# same live-quiz engine as Official Quiz (so timers/pausing/etc. all just work),
# with a head-to-head result posted once it finishes.
# ==========================================
DUEL_QUESTION_COUNT = 5
DUEL_TIMER_SECONDS = 15


async def duel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type not in ("group", "supergroup"):
        await update.message.reply_text("🤺 /duel sirf group me chalta hai — kisi group me apne opponent ke message par reply karke try karo.")
        return
    replied = update.message.reply_to_message
    if not replied or not replied.from_user:
        await update.message.reply_text(
            "🤺 **Duel kaise shuru karein:**\nApne opponent ke kisi message par **reply** karke likho:\n"
            "`/duel <topic>` — jaise `/duel World Capitals`",
            parse_mode="Markdown",
        )
        return
    challenger = update.effective_user
    opponent = replied.from_user
    if opponent.id == challenger.id:
        await update.message.reply_text("🤺 Khud se duel nahi ho sakta — kisi aur ko challenge karo!")
        return
    if opponent.is_bot:
        await update.message.reply_text("🤺 Bots ke saath duel nahi ho sakta.")
        return
    topic = " ".join(context.args).strip() if context.args else "General Knowledge"

    await update.message.reply_text(
        f"🤺 **Duel Challenge!**\n{challenger.first_name} vs {opponent.first_name}\n"
        f"Topic: {topic}\nGenerating {DUEL_QUESTION_COUNT} questions...",
        parse_mode="Markdown",
    )
    try:
        questions = await generate_mcqs_async(challenger.id, [{"type": "text", "text": topic}],
                                               DUEL_QUESTION_COUNT, "medium", "English")
    except RuntimeError as e:
        await report_generation_failure(context.bot, update.effective_chat.id, challenger.id, e)
        return

    quiz_id = save_quiz(challenger.id, f"Duel: {topic}"[:60], questions, timer_seconds=DUEL_TIMER_SECONDS,
                         quiz_type="duel", creator_name=challenger.first_name or "Unknown")

    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO duels (chat_id, quiz_id, challenger_id, challenger_name, opponent_id, opponent_name, status) "
        "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
        (update.effective_chat.id, quiz_id, challenger.id, challenger.first_name or "Unknown",
         opponent.id, opponent.first_name or "Unknown")
    )
    conn.commit()
    duel_id = last_insert_id(cur, conn, "duels")
    conn.close()

    await initiate_live_quiz(context.bot, context.application, update.effective_chat.id, update.effective_chat.type,
                              quiz_id, challenger.id, challenger.first_name or "Host")

    context.application.job_queue.run_repeating(
        _watch_duel_finish, interval=10, first=10,
        data={"duel_id": duel_id, "chat_id": update.effective_chat.id, "quiz_id": quiz_id, "tries": 0},
        name=f"duel_watch_{duel_id}",
    )


async def _watch_duel_finish(context: ContextTypes.DEFAULT_TYPE):
    """Polls the in-memory live-quiz state (population happens in start_live_quiz /
    _run_live_quiz_loop) until this duel's session is finished, then posts a
    head-to-head comparison. Gives up after ~10 minutes so a duel nobody starts
    (lobby never fills) doesn't leave a job running forever."""
    job = context.job
    data = job.data
    data["tries"] += 1
    live = context.application.bot_data.get("live_quizzes", {})
    session = live.get(data["chat_id"])
    session_matches = session and session.get("quiz_id") == data["quiz_id"]
    finished = (not session_matches) or session.get("finished", False)
    if not finished and data["tries"] < 60:
        return  # keep polling — still running or lobby hasn't started it yet

    job.schedule_removal()

    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT challenger_id, challenger_name, opponent_id, opponent_name, status FROM duels WHERE id=?",
                (data["duel_id"],))
    row = cur.fetchone()
    if not row or row[4] == "finished":
        conn.close()
        return
    challenger_id, challenger_name, opponent_id, opponent_name = row[0], row[1], row[2], row[3]

    def _score_for(uid):
        cur.execute("SELECT correct_count, wrong_count FROM quiz_results WHERE quiz_id=? AND user_id=?",
                    (data["quiz_id"], uid))
        r = cur.fetchone()
        return (r[0] or 0, r[1] or 0) if r else (0, 0)

    c_correct, c_wrong = _score_for(challenger_id)
    o_correct, o_wrong = _score_for(opponent_id)
    cur.execute("UPDATE duels SET status='finished' WHERE id=?", (data["duel_id"],))
    conn.commit()
    conn.close()

    if c_correct == 0 and c_wrong == 0 and o_correct == 0 and o_wrong == 0:
        return  # nobody answered at all (lobby likely never filled) — nothing worth posting

    if c_correct > o_correct:
        verdict = f"🏆 **{challenger_name} wins the duel!**"
    elif o_correct > c_correct:
        verdict = f"🏆 **{opponent_name} wins the duel!**"
    else:
        verdict = "🤝 **It's a tie!**"

    await context.bot.send_message(
        data["chat_id"],
        f"🤺 **Duel Result**\n\n"
        f"{challenger_name}: {c_correct}✅ / {c_wrong}❌\n"
        f"{opponent_name}: {o_correct}✅ / {o_wrong}❌\n\n{verdict}",
        parse_mode="Markdown",
    )


async def granttop10_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner-only: grants the current top-10 solvers a bonus to their quota — the
    reward for topping the leaderboard, not a permanent unlimited bypass."""
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return
    bonus = 100
    if context.args:
        try:
            bonus = max(100, int(context.args[0]))
        except ValueError:
            await update.message.reply_text("Usage: /granttop10 [BONUS_QUESTIONS] — must be a number, minimum 100.")
            return
    top = get_top_solvers(10)
    if not top:
        await update.message.reply_text("No quiz results yet — nobody to reward.")
        return
    rewarded = []
    for u in top:
        increment_question_limit(u['user_id'], bonus)
        rewarded.append(u['user_name'])
        try:
            await context.bot.send_message(
                chat_id=u['user_id'],
                text=f"🏆 You're in the Top 10 leaderboard! +{bonus} questions added to your quota. Keep it up!",
            )
        except Exception as e:
            logger.warning(f"Couldn't DM top-10 reward to {u['user_id']}: {e}")
    await update.message.reply_text(f"✅ Granted +{bonus} questions to {len(rewarded)} top user(s): " + ", ".join(rewarded))


async def broadcast_to_all(bot, text: str) -> tuple:
    """Sends text to every tracked group AND every user who has ever /start'ed the
    bot (DM). Returns (success_count, fail_count). Small delay between sends to stay
    well under Telegram's flood limits on a big broadcast."""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT group_id FROM groups")
    group_ids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT user_id FROM users")
    user_ids = [r[0] for r in cur.fetchall()]
    conn.close()

    ok, fail = 0, 0
    for chat_id in group_ids + user_ids:
        try:
            await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
            ok += 1
        except Exception as e:
            fail += 1
            logger.warning(f"Broadcast failed for {chat_id}: {e}")
        await asyncio.sleep(0.05)
    return ok, fail


async def broadcastcode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner-only: announces an existing redeem code to every group/channel the bot
    is in, and DMs every user who has started the bot."""
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /broadcastcode CODE")
        return
    code = context.args[0].strip().upper()
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT bonus_questions FROM redeem_codes WHERE code = ?", (code,))
    row = cur.fetchone()
    conn.close()
    if not row:
        await update.message.reply_text(f"❌ Code {code} doesn't exist — create it first with /addcode.")
        return
    text = f"🎁 **New Redeem Code!**\n\nUse `/redeem {code}` to claim **+{row[0]} bonus questions**!"
    await update.message.reply_text("📡 Broadcasting…")
    ok, fail = await broadcast_to_all(context.bot, text)
    await update.message.reply_text(f"✅ Broadcast done — delivered to {ok} chat(s), failed for {fail}.")


async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner-only: broadcasts any free-form message to every group/channel and every
    user who has started the bot."""
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /broadcast Your message here")
        return
    text = " ".join(context.args)
    await update.message.reply_text("📡 Broadcasting…")
    ok, fail = await broadcast_to_all(context.bot, text)
    await update.message.reply_text(f"✅ Broadcast done — delivered to {ok} chat(s), failed for {fail}.")


# ==========================================
# 📁 QUIZ SETS — /newset, /sets (named folders for organizing saved quizzes)
# ==========================================
async def browse_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/browse [keyword] — search your own saved quizzes by title. No keyword lists
    everything (most recent first)."""
    user_id = update.effective_user.id
    keyword = " ".join(context.args).strip() if context.args else ""
    conn = db_connect()
    cur = conn.cursor()
    if keyword:
        cur.execute(
            "SELECT id, title, quiz_type, timer_seconds FROM quizzes WHERE user_id=? AND title LIKE ? "
            "ORDER BY id DESC LIMIT 25",
            (user_id, f"%{keyword}%")
        )
    else:
        cur.execute(
            "SELECT id, title, quiz_type, timer_seconds FROM quizzes WHERE user_id=? ORDER BY id DESC LIMIT 25",
            (user_id,)
        )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        msg = f"No saved quizzes matching '{keyword}'." if keyword else "You haven't saved any quizzes yet."
        await update.message.reply_text(msg)
        return
    header = f"🔎 **Results for '{keyword}'**" if keyword else "📚 **Your Saved Quizzes** (most recent first)"
    lines = [header]
    for r in rows:
        timer_text = f"{r[3]}s/q" if r[3] else "no timer"
        lines.append(f"`{r[0]}` — {r[1]} _({r[2]}, {timer_text})_")
    lines.append("\n_Play any of these with_ `/startquiz <id>` _· export with_ `/exportquiz <id>` _or_ `/exportresults <id>`")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def exportquiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/exportquiz <id> — sends the quiz's questions as a downloadable JSON file
    (owner's own quiz, or bot owner for any quiz)."""
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usage: `/exportquiz <quiz id>` (see IDs with `/browse`)", parse_mode="Markdown")
        return
    quiz = get_quiz_meta(int(context.args[0]))
    if not quiz or (quiz["user_id"] != update.effective_user.id and update.effective_user.id != OWNER_ID):
        await update.message.reply_text("❌ Quiz not found, or it isn't yours.")
        return
    payload = {"title": quiz["title"], "quiz_type": quiz["quiz_type"],
               "timer_seconds": quiz["timer_seconds"], "questions": quiz["questions"]}
    buf = io.BytesIO(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    buf.name = f"quiz_{quiz['id']}.json"
    await update.message.reply_document(buf, filename=buf.name, caption=f"📤 Exported: {quiz['title']}")


async def exportresults_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/exportresults <id> — CSV of everyone who played this quiz and how they did
    (owner's own quiz, or bot owner for any quiz)."""
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usage: `/exportresults <quiz id>` (see IDs with `/browse`)", parse_mode="Markdown")
        return
    quiz_id = int(context.args[0])
    quiz = get_quiz_meta(quiz_id)
    if not quiz or (quiz["user_id"] != update.effective_user.id and update.effective_user.id != OWNER_ID):
        await update.message.reply_text("❌ Quiz not found, or it isn't yours.")
        return
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT user_id, user_name, correct_count, wrong_count, total_questions, score, avg_time_ms, finished_at "
        "FROM quiz_results WHERE quiz_id=? ORDER BY finished_at", (quiz_id,)
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        await update.message.reply_text("No results yet for this quiz — nobody has played it live.")
        return
    text_buf = io.StringIO()
    writer = csv.writer(text_buf)
    writer.writerow(["user_id", "user_name", "correct", "wrong", "total_questions", "score", "avg_time_ms", "finished_at"])
    writer.writerows(rows)
    buf = io.BytesIO(text_buf.getvalue().encode("utf-8"))
    buf.name = f"quiz_{quiz_id}_results.csv"
    await update.message.reply_document(buf, filename=buf.name, caption=f"📊 Results export: {quiz['title']}")


async def newset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /newset Your Set Name (e.g. `/newset Class 10 History`)", parse_mode="Markdown")
        return
    name = " ".join(context.args)[:60]
    set_id = create_quiz_set(update.effective_user.id, name)
    await update.message.reply_text(
        f"📁 Set created: **{name}** (`{set_id}`)\nFile a quiz into it from /mystore → open a quiz → 📁 File in Set.",
        parse_mode="Markdown",
    )


def _sets_list_view(owner_id: int):
    sets = get_user_sets(owner_id)
    if not sets:
        return "You don't have any sets yet. Create one: `/newset Your Set Name`", back_to_tools_keyboard()
    rows = [[InlineKeyboardButton(f"📁 {s['name']}", callback_data=f"setview_{s['id']}")] for s in sets[:25]]
    rows.append([InlineKeyboardButton("🔙 Back to Tools", callback_data="open_tools")])
    return f"📁 **Your Sets** ({len(sets)}):", InlineKeyboardMarkup(rows)


async def sets_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text, keyboard = _sets_list_view(update.effective_user.id)
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def set_view_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    set_id = int(query.data.replace("setview_", ""))
    set_meta = get_set_meta(set_id)
    if not set_meta or set_meta['owner_id'] != query.from_user.id:
        await query.edit_message_text("That set no longer exists.")
        return
    quizzes = get_quizzes_in_set(set_id)
    lines = [f"📁 **{set_meta['name']}** — {len(quizzes)} quiz(zes)\n"]
    rows = [[InlineKeyboardButton(f"🆔 {q['id']} · {q['title'][:30]}", callback_data=f"stmanage_{q['id']}")] for q in quizzes[:20]]
    rows.append([InlineKeyboardButton("🗑 Delete This Set (keeps the quizzes)", callback_data=f"setdel_{set_id}")])
    rows.append([InlineKeyboardButton("🔙 Back to Sets", callback_data="setlist_back")])
    await query.edit_message_text("\n".join(lines) or "Empty set.", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(rows))


async def set_list_back_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    text, keyboard = _sets_list_view(query.from_user.id)
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def set_delete_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    set_id = int(query.data.replace("setdel_", ""))
    ok = delete_quiz_set(set_id, query.from_user.id)
    text, keyboard = _sets_list_view(query.from_user.id)
    prefix = "✅ Set deleted (its quizzes are still in /mystore, just unfiled).\n\n" if ok else "❌ Couldn't delete that set.\n\n"
    await query.edit_message_text(prefix + text, parse_mode="Markdown", reply_markup=keyboard)


# ==========================================
# 📤 LINKED CHANNELS — /setchannel, /mychannels
# ==========================================
async def setchannel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📤 To link a channel: add me as **admin** there (so I'm allowed to post), then "
        "forward me any single message from that channel right now.",
        parse_mode="Markdown", reply_markup=cancel_keyboard(),
    )
    return SETCHANNEL_WAIT


async def setchannel_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    fwd_chat = getattr(msg, "forward_from_chat", None)
    if not fwd_chat or fwd_chat.type != "channel":
        await update.message.reply_text(
            "That doesn't look like a forward from a channel — forward a message from the "
            "channel itself (not a group, not a copy-paste), or Cancel.",
            reply_markup=cancel_keyboard(),
        )
        return SETCHANNEL_WAIT
    try:
        member = await context.bot.get_chat_member(fwd_chat.id, context.bot.id)
        if member.status not in ("administrator", "creator"):
            await update.message.reply_text(
                f"I'm in **{fwd_chat.title}** but not as admin yet — make me admin there first, then forward again.",
                parse_mode="Markdown", reply_markup=cancel_keyboard(),
            )
            return SETCHANNEL_WAIT
    except Exception as e:
        await update.message.reply_text(
            f"I can't see my own status in that channel yet — make sure I'm added as admin there first. ({e})",
            reply_markup=cancel_keyboard(),
        )
        return SETCHANNEL_WAIT
    add_user_channel(update.effective_user.id, fwd_chat.id, fwd_chat.title or str(fwd_chat.id))
    await update.message.reply_text(f"✅ Linked **{fwd_chat.title}** — it'll now show up under 📤 Send to Channel.",
                                     parse_mode="Markdown")
    return ConversationHandler.END


async def mychannels_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    channels = get_user_channels(update.effective_user.id)
    if not channels:
        await update.message.reply_text("No channels linked yet — use /setchannel to link one.")
        return
    rows = [[InlineKeyboardButton(f"🗑 Remove {c['channel_title']}", callback_data=f"chrm_{c['id']}")] for c in channels]
    await update.message.reply_text(f"📤 **Linked Channels** ({len(channels)}):", parse_mode="Markdown",
                                     reply_markup=InlineKeyboardMarkup(rows))


async def channel_remove_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    row_id = int(query.data.replace("chrm_", ""))
    remove_user_channel(query.from_user.id, row_id)
    channels = get_user_channels(query.from_user.id)
    if not channels:
        await query.edit_message_text("No channels linked — use /setchannel to link one.")
        return
    rows = [[InlineKeyboardButton(f"🗑 Remove {c['channel_title']}", callback_data=f"chrm_{c['id']}")] for c in channels]
    await query.edit_message_text(f"📤 **Linked Channels** ({len(channels)}):", parse_mode="Markdown",
                                   reply_markup=InlineKeyboardMarkup(rows))


async def mystore_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(*_store_list_view(update.effective_user.id, 0))


async def store_button_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    text, keyboard = _store_list_view(query.from_user.id, 0)
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def store_list_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    page = int(query.data.replace("stlist_", ""))
    text, keyboard = _store_list_view(query.from_user.id, page)
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)


STORE_PAGE_SIZE = 10


def _store_list_view(user_id: int, page: int = 0):
    """Paginated quiz list — once you've got more than one page's worth of saved
    quizzes (Official Quiz or otherwise), ⬅️/➡️ buttons page through them instead of
    either dumping everything into one giant message or silently dropping anything
    past the first 25. Tapping any quiz still opens the same Edit/Start/Share/Delete
    panel (_store_manage_view) regardless of which page it was found on."""
    quizzes = get_user_quizzes(user_id)
    if not quizzes:
        return (
            "You haven't saved any quizzes yet — create one with /topic, /officialquiz, or by forwarding a poll.",
            back_to_tools_keyboard(),
        )
    total_pages = max(1, (len(quizzes) + STORE_PAGE_SIZE - 1) // STORE_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * STORE_PAGE_SIZE
    page_quizzes = quizzes[start:start + STORE_PAGE_SIZE]

    rows = [[InlineKeyboardButton(f"🆔 {q['id']} · {q['title'][:30]}", callback_data=f"stmanage_{q['id']}")]
            for q in page_quizzes]

    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("⬅️ Back", callback_data=f"stlist_{page - 1}"))
        nav.append(InlineKeyboardButton(f"📄 {page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton("Next ➡️", callback_data=f"stlist_{page + 1}"))
        rows.append(nav)

    rows.append([InlineKeyboardButton("🔙 Back to Tools", callback_data="open_tools")])
    text = stylize("Your saved quizzes") + f":\n\n{len(quizzes)} total — tap one to manage it (▶️ Start · ✏️ Edit · 🗑 Delete)."
    return text, InlineKeyboardMarkup(rows)


def _store_manage_view(quiz_id: int):
    quiz = get_quiz_meta(quiz_id)
    if not quiz:
        return "That quiz no longer exists.", back_to_tools_keyboard()
    timer_text = f"{quiz['timer_seconds']}s / question" if quiz['timer_seconds'] else "No limit"
    is_official = quiz['quiz_type'] == 'official'
    type_label = "🤖 Official Quiz" if is_official else "⚡ Quick Quiz"
    attempts = len(get_quiz_status_rows(quiz_id, limit=1000))
    creator_name = quiz['creator_name'] or "Unknown"
    creator_mention = f"[{creator_name}](tg://user?id={quiz['user_id']})"
    set_line = ""
    if quiz.get('set_id'):
        set_meta = get_set_meta(quiz['set_id'])
        if set_meta:
            set_line = f"📁 Set: {set_meta['name']}\n"
    visibility_line = "🌐 Public — listed in /publicstore\n" if quiz["is_public"] else "🔒 Private\n"
    text = (
        f"📚 **{quiz['title']}**\n"
        + (f"_{FONT_STYLE}_\n" if is_official else "") +
        (f"💬 {quiz['description']}\n" if quiz.get('description') else "") +
        f"\n🆔 Quiz ID: `{quiz['id']}`\n"
        f"👤 Creator: {creator_mention} (`{quiz['user_id']}`)\n"
        f"⏱ Timer: {timer_text}\n"
        f"❓ Questions: {len(quiz['questions'])}\n"
        f"🏷 Type: {type_label}\n"
        f"{set_line}"
        f"{visibility_line}"
        f"👥 Attempts recorded: {attempts}\n\n"
        f"Any change below (title, description, timer, questions) saves straight to this "
        f"quiz and applies the moment it's next started — no separate publish step."
    )
    keyboard_rows = [
        [InlineKeyboardButton("▶️ Start Here", callback_data=f"oqstart_{quiz_id}"),
         InlineKeyboardButton("📤 Share ID", callback_data=f"oqshare_{quiz_id}")],
        [InlineKeyboardButton("✏️ Edit Title", callback_data=f"stedittitle_{quiz_id}"),
         InlineKeyboardButton("💬 Edit Description", callback_data=f"steditdesc_{quiz_id}")],
        [InlineKeyboardButton("⏱ Edit Timer", callback_data=f"stedittimer_{quiz_id}"),
         InlineKeyboardButton("📁 File in Set", callback_data=f"stsetpick_{quiz_id}")],
        [InlineKeyboardButton("➕ Add Question", callback_data=f"staddq_{quiz_id}"),
         InlineKeyboardButton("➖ Remove Question", callback_data=f"stremq_{quiz_id}")],
        [InlineKeyboardButton("🔒 Make Private" if quiz["is_public"] else "🌐 Make Public",
                               callback_data=f"stpub_{quiz_id}"),
         InlineKeyboardButton("🧹 Remove Duplicates", callback_data=f"stdedup_{quiz_id}")],
        [InlineKeyboardButton("📊 Status (who played, right/wrong)", callback_data=f"ststatus_{quiz_id}_0")],
        [InlineKeyboardButton("📤 Send to Channel", callback_data=f"stsendch_{quiz_id}")],
        [InlineKeyboardButton("🗑 Delete Quiz", callback_data=f"stdel_{quiz_id}")],
        [InlineKeyboardButton("🏷 My Tag", callback_data="stmytag"),
         InlineKeyboardButton("💬 My Desc Tag", callback_data="stmydesc")],
        [InlineKeyboardButton("🔤 Option Style", callback_data="stoptstyle")],
        [InlineKeyboardButton("🔙 Back to My Quizzes", callback_data="menu_store")],
    ]
    return text, InlineKeyboardMarkup(keyboard_rows)


async def store_manage_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id = int(query.data.replace("stmanage_", ""))
    text, keyboard = _store_manage_view(quiz_id)
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def store_delete_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    quiz_id = int(query.data.replace("stdel_", ""))
    quiz = get_quiz_meta(quiz_id)
    if not quiz:
        await query.answer("Already deleted.", show_alert=True)
        return
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Yes, delete it", callback_data=f"stdelok_{quiz_id}"),
         InlineKeyboardButton("❌ No, keep it", callback_data=f"stmanage_{quiz_id}")],
    ])
    await query.answer()
    await query.edit_message_text(f"🗑 Delete **{quiz['title']}** permanently? This can't be undone.",
                                   parse_mode="Markdown", reply_markup=keyboard)


async def store_delete_confirm_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    quiz_id = int(query.data.replace("stdelok_", ""))
    deleted = delete_quiz(quiz_id, query.from_user.id)
    await query.answer("Deleted." if deleted else "Couldn't delete — not your quiz.", show_alert=not deleted)
    text, keyboard = _store_list_view(query.from_user.id, 0)
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def store_toggle_public_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    quiz_id = int(query.data.replace("stpub_", ""))
    result = toggle_quiz_public(quiz_id, query.from_user.id)
    if result is None:
        await query.answer("Couldn't update — not your quiz.", show_alert=True)
    else:
        await query.answer("🌐 Now public — listed in /publicstore." if result else "🔒 Back to private.")
    text, keyboard = _store_manage_view(quiz_id)
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def store_dedup_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    quiz_id = int(query.data.replace("stdedup_", ""))
    removed = dedup_quiz_questions(quiz_id, query.from_user.id)
    if removed is None:
        await query.answer("Couldn't update — not your quiz.", show_alert=True)
    else:
        await query.answer(f"🧹 Removed {removed} duplicate question(s)." if removed else "No duplicates found.",
                            show_alert=True)
    text, keyboard = _store_manage_view(quiz_id)
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)


# ==========================================
# 🌐 PUBLIC QUIZ STORE — opt-in directory any user can browse & play,
# separate from /mystore (which only ever lists your own quizzes).
# ==========================================
PUBLIC_STORE_PAGE_SIZE = 6


def _public_store_view(page: int = 0):
    quizzes = get_public_quizzes()
    if not quizzes:
        return (
            "🌐 No public quizzes yet — open any quiz in /mystore and tap **🌐 Make Public** "
            "to be the first one listed here.",
            back_to_tools_keyboard(),
        )
    total_pages = max(1, (len(quizzes) + PUBLIC_STORE_PAGE_SIZE - 1) // PUBLIC_STORE_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * PUBLIC_STORE_PAGE_SIZE
    page_quizzes = quizzes[start:start + PUBLIC_STORE_PAGE_SIZE]

    rows = [[InlineKeyboardButton(f"📁 {q['title'][:28]} — by {q['creator_name'][:15]}",
                                   callback_data=f"pubview_{q['id']}")]
            for q in page_quizzes]

    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("⬅️ Back", callback_data=f"publist_{page - 1}"))
        nav.append(InlineKeyboardButton(f"📄 {page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton("Next ➡️", callback_data=f"publist_{page + 1}"))
        rows.append(nav)

    rows.append([InlineKeyboardButton("🔙 Back to Tools", callback_data="open_tools")])
    text = f"🌐 **{stylize('Public Quiz Store')}**\n\n{len(quizzes)} public quizzes — tap one to view and start it."
    return text, InlineKeyboardMarkup(rows)


async def publicstore_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text, keyboard = _public_store_view(0)
    await update.effective_message.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def public_store_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    page = int(query.data.replace("publist_", ""))
    text, keyboard = _public_store_view(page)
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def public_quiz_view_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id = int(query.data.replace("pubview_", ""))
    quiz = get_quiz_meta(quiz_id)
    if not quiz or not quiz["is_public"]:
        await query.edit_message_text("That quiz is no longer public.", reply_markup=back_to_tools_keyboard())
        return
    timer_text = f"{quiz['timer_seconds']}s / question" if quiz['timer_seconds'] else "No limit"
    text = (
        f"📁 **{quiz['title']}**\n"
        + (f"💬 {quiz['description']}\n" if quiz.get('description') else "") +
        f"\n👤 By: {quiz['creator_name'] or 'Unknown'}\n"
        f"❓ Questions: {len(quiz['questions'])}\n"
        f"⏱ Timer: {timer_text}"
    )
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Start Here", callback_data=f"oqstart_{quiz_id}")],
        [InlineKeyboardButton("🔙 Back to Public Store", callback_data="publist_0")],
    ])
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def store_edit_title_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id = int(query.data.replace("stedittitle_", ""))
    context.user_data['store_edit_quiz_id'] = quiz_id
    await query.edit_message_text("Send the new title for this quiz.", reply_markup=cancel_keyboard())
    return STORE_EDIT_TITLE


async def store_edit_title_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    quiz_id = context.user_data.pop('store_edit_quiz_id', None)
    if not quiz_id:
        await update.message.reply_text("Session expired — open the quiz from /mystore again.")
        return ConversationHandler.END
    new_title = (update.message.text or "").strip()[:60]
    if not new_title:
        await update.message.reply_text("Please send a valid title, or Cancel.", reply_markup=cancel_keyboard())
        context.user_data['store_edit_quiz_id'] = quiz_id
        return STORE_EDIT_TITLE
    update_quiz_title(quiz_id, new_title)
    text, keyboard = _store_manage_view(quiz_id)
    await update.message.reply_text(f"✅ Title updated — live immediately.\n\n{text}", parse_mode="Markdown", reply_markup=keyboard)
    return ConversationHandler.END


async def store_edit_timer_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id = int(query.data.replace("stedittimer_", ""))
    context.user_data['store_edit_quiz_id'] = quiz_id
    keyboard = cancel_keyboard([
        [InlineKeyboardButton("10s", callback_data="sttimerpick_10"), InlineKeyboardButton("15s", callback_data="sttimerpick_15"),
         InlineKeyboardButton("20s", callback_data="sttimerpick_20")],
        [InlineKeyboardButton("30s", callback_data="sttimerpick_30"), InlineKeyboardButton("60s", callback_data="sttimerpick_60"),
         InlineKeyboardButton("⏱ No Timer", callback_data="sttimerpick_0")],
    ])
    await query.edit_message_text(
        "Pick a new per-question timer, or type a custom number of seconds (0 = no limit).",
        reply_markup=keyboard,
    )
    return STORE_EDIT_TIMER


async def _apply_store_timer(quiz_id: int, seconds: int, edit_target, user_first_name: str = ""):
    update_quiz_timer(quiz_id, seconds)
    text, keyboard = _store_manage_view(quiz_id)
    note = f"✅ Timer updated to {seconds}s — applies immediately, from the next time this quiz is started.\n\n" if seconds else \
           "✅ Timer removed (no limit) — applies immediately, from the next time this quiz is started.\n\n"
    await edit_target(note + text, parse_mode="Markdown", reply_markup=keyboard)


async def store_edit_timer_pick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id = context.user_data.pop('store_edit_quiz_id', None)
    if not quiz_id:
        await query.edit_message_text("Session expired — open the quiz from /mystore again.")
        return ConversationHandler.END
    seconds = int(query.data.replace("sttimerpick_", ""))
    await _apply_store_timer(quiz_id, seconds, query.edit_message_text)
    return ConversationHandler.END


async def store_edit_timer_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    quiz_id = context.user_data.get('store_edit_quiz_id')
    if not quiz_id:
        await update.message.reply_text("Session expired — open the quiz from /mystore again.")
        return ConversationHandler.END
    raw = (update.message.text or "").strip()
    if not raw.isdigit() or int(raw) > 600:
        await update.message.reply_text("Send a number of seconds (0–600), or tap a button above.")
        return STORE_EDIT_TIMER
    context.user_data.pop('store_edit_quiz_id', None)
    await _apply_store_timer(quiz_id, int(raw), update.message.reply_text)
    return ConversationHandler.END


async def store_edit_desc_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id = int(query.data.replace("steditdesc_", ""))
    context.user_data['store_edit_quiz_id'] = quiz_id
    await query.edit_message_text(
        "💬 Send the new description for this quiz (shown to anyone who opens it — what it's about, "
        "who it's for, etc). Max 200 characters.",
        reply_markup=cancel_keyboard(),
    )
    return STORE_EDIT_DESC


async def store_edit_desc_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    quiz_id = context.user_data.pop('store_edit_quiz_id', None)
    if not quiz_id:
        await update.message.reply_text("Session expired — open the quiz from /mystore again.")
        return ConversationHandler.END
    description = (update.message.text or "").strip()[:200]
    update_quiz_description(quiz_id, description)
    text, keyboard = _store_manage_view(quiz_id)
    await update.message.reply_text(f"✅ Description updated — live immediately.\n\n{text}",
                                     parse_mode="Markdown", reply_markup=keyboard)
    return ConversationHandler.END


# ==========================================
# 📁 "File in Set" picker — attaches a quiz to one of the owner's named sets
# ==========================================
async def store_set_pick_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id = int(query.data.replace("stsetpick_", ""))
    sets = get_user_sets(query.from_user.id)
    rows = [[InlineKeyboardButton(f"📁 {s['name']}", callback_data=f"stsetdo_{quiz_id}_{s['id']}")] for s in sets[:20]]
    rows.append([InlineKeyboardButton("🚫 Remove from any set", callback_data=f"stsetdo_{quiz_id}_none")])
    rows.append([InlineKeyboardButton("➕ New Set (/newset)", callback_data=f"stmanage_{quiz_id}")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data=f"stmanage_{quiz_id}")])
    text = "📁 Pick a set to file this quiz under:" if sets else \
        "You don't have any sets yet — create one with `/newset Your Set Name`, then come back here."
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(rows))


async def store_set_assign_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    _, quiz_id_str, set_id_str = query.data.split('_', 2)
    quiz_id = int(quiz_id_str)
    assign_quiz_to_set(quiz_id, None if set_id_str == "none" else int(set_id_str))
    text, keyboard = _store_manage_view(quiz_id)
    await query.edit_message_text(("✅ Filed.\n\n" if set_id_str != "none" else "✅ Removed from set.\n\n") + text,
                                   parse_mode="Markdown", reply_markup=keyboard)


# ==========================================
# 📤 "Send to Channel" — posts the FULL quiz to one of the user's linked channels,
# with their tag/description/option-style/promo applied exactly like anywhere else.
# ==========================================
async def store_send_channel_pick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id = int(query.data.replace("stsendch_", ""))
    channels = get_user_channels(query.from_user.id)
    if not channels:
        await query.edit_message_text(
            "You haven't linked any channels yet.\n\n"
            "To link one: add me as **admin** in your channel, then forward any message "
            "from that channel to me with `/setchannel`.",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data=f"stmanage_{quiz_id}")]]),
        )
        return
    rows = [[InlineKeyboardButton(f"📤 {c['channel_title']}", callback_data=f"stsendchdo_{quiz_id}_{c['id']}")] for c in channels[:20]]
    rows.append([InlineKeyboardButton("🔙 Back", callback_data=f"stmanage_{quiz_id}")])
    await query.edit_message_text("Pick a channel to post this quiz to:", reply_markup=InlineKeyboardMarkup(rows))


async def store_send_channel_do(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("Sending...")
    _, quiz_id_str, row_id_str = query.data.split('_', 2)
    quiz_id, row_id = int(quiz_id_str), int(row_id_str)
    quiz = get_quiz_meta(quiz_id)
    channels = get_user_channels(query.from_user.id)
    target = next((c for c in channels if c['id'] == row_id), None)
    if not quiz or not target:
        await query.edit_message_text("That quiz or channel link no longer exists.")
        return
    user = get_user(query.from_user.id)
    try:
        sent = await post_quiz_polls(
            context.bot, target['channel_id'], quiz['questions'],
            user_tag=user['tag'], explanation_tag=user['description_tag'], option_style=user['option_style'],
        )
        await query.edit_message_text(f"✅ Posted {sent} question(s) to **{target['channel_title']}**.",
                                       parse_mode="Markdown")
    except Exception as e:
        logger.warning(f"store_send_channel_do failed: {e}")
        await query.edit_message_text(
            f"❌ Couldn't post there — make sure I'm still an admin in that channel. ({e})"
        )


async def store_option_style_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = get_user(query.from_user.id)
    await query.edit_message_text(
        "🔤 **Option Style** — how answer options are labelled in every quiz you make:",
        parse_mode="Markdown", reply_markup=_option_style_keyboard(user['option_style']),
    )


STATUS_PAGE_SIZE = 10


async def store_status_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """📊 Status — the Official Quiz editing-section view the host uses to see exactly
    who attempted their quiz: user_id (to contact them directly), right/wrong count,
    accuracy, average answer speed, and score — one row per participant."""
    query = update.callback_query
    await query.answer()
    _, quiz_id_str, page_str = query.data.split('_', 2)
    quiz_id, page = int(quiz_id_str), int(page_str)
    quiz = get_quiz_meta(quiz_id)
    if not quiz:
        await query.edit_message_text("That quiz no longer exists.", reply_markup=back_to_tools_keyboard())
        return

    rows = get_quiz_status_rows(quiz_id, limit=1000)
    total_pages = max(1, (len(rows) + STATUS_PAGE_SIZE - 1) // STATUS_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    page_rows = rows[page * STATUS_PAGE_SIZE: page * STATUS_PAGE_SIZE + STATUS_PAGE_SIZE]

    header = f"📊 **{stylize('Status')}** — {quiz['title']}\n_{FONT_STYLE}_\n\n"
    if not rows:
        body = "No attempts recorded yet — the status list fills in as people play this quiz live."
    else:
        lines = []
        for i, r in enumerate(page_rows, start=page * STATUS_PAGE_SIZE + 1):
            avg_sec = round(r['avg_time_ms'] / 1000, 1)
            lines.append(
                f"{i}. {r['user_name']} — `{r['user_id']}`\n"
                f"      ✅ {r['correct']} ❌ {r['wrong']} · 🎯 {r['accuracy']}% · ⏱ {avg_sec}s avg · {r['score']} pts"
            )
        body = "\n".join(lines) + f"\n\n👥 {len(rows)} total attempt(s). User ID is shown for direct contact."

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️ Back", callback_data=f"ststatus_{quiz_id}_{page - 1}"))
    if total_pages > 1:
        nav.append(InlineKeyboardButton(f"📄 {page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton("Next ➡️", callback_data=f"ststatus_{quiz_id}_{page + 1}"))
    rows_kb = [nav] if nav else []
    rows_kb.append([InlineKeyboardButton("🔙 Back to Quiz", callback_data=f"stmanage_{quiz_id}")])

    try:
        await query.edit_message_text(header + body, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(rows_kb))
    except Exception as e:
        logger.warning(f"store_status_callback edit failed (non-fatal): {e}")


async def store_add_question_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id = int(query.data.replace("staddq_", ""))
    context.user_data['store_edit_quiz_id'] = quiz_id
    await query.edit_message_text(
        "Send the new question in this format:\n\n"
        "```\nQ: Question text?\nA) Option 1\nB) Option 2\nC) Option 3\nAnswer: A\n```",
        parse_mode="Markdown", reply_markup=cancel_keyboard(),
    )
    return STORE_ADD_QUESTION


async def store_add_question_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    quiz_id = context.user_data.pop('store_edit_quiz_id', None)
    if not quiz_id:
        await update.message.reply_text("Session expired — open the quiz from /mystore again.")
        return ConversationHandler.END
    new_qs = parse_bulk_questions(update.message.text or "")
    if not new_qs:
        await update.message.reply_text("Couldn't parse that — check the format and resend, or Cancel.", reply_markup=cancel_keyboard())
        context.user_data['store_edit_quiz_id'] = quiz_id
        return STORE_ADD_QUESTION
    quiz = get_quiz_meta(quiz_id)
    if not quiz:
        await update.message.reply_text("That quiz no longer exists.")
        return ConversationHandler.END
    quiz['questions'].extend(new_qs)
    update_quiz_questions(quiz_id, quiz['questions'])
    text, keyboard = _store_manage_view(quiz_id)
    await update.message.reply_text(f"✅ Added {len(new_qs)} question(s).\n\n{text}", parse_mode="Markdown", reply_markup=keyboard)
    return ConversationHandler.END


async def store_remove_question_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id = int(query.data.replace("stremq_", ""))
    quiz = get_quiz_meta(quiz_id)
    if not quiz or not quiz['questions']:
        await query.edit_message_text("No questions to remove.", reply_markup=back_to_tools_keyboard())
        return
    rows = [
        [InlineKeyboardButton(f"❌ {i + 1}. {q['question'][:40]}", callback_data=f"stremqok_{quiz_id}_{i}")]
        for i, q in enumerate(quiz['questions'][:50])
    ]
    rows.append([InlineKeyboardButton("🔙 Back", callback_data=f"stmanage_{quiz_id}")])
    await query.edit_message_text("Tap a question to remove it:", reply_markup=InlineKeyboardMarkup(rows))


async def store_remove_question_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    _, quiz_id_str, idx_str = query.data.split('_', 2)
    quiz_id, idx = int(quiz_id_str), int(idx_str)
    quiz = get_quiz_meta(quiz_id)
    if not quiz or idx >= len(quiz['questions']):
        await query.answer("Already removed.", show_alert=True)
        return
    quiz['questions'].pop(idx)
    update_quiz_questions(quiz_id, quiz['questions'])
    await query.answer("Removed.")
    text, keyboard = _store_manage_view(quiz_id)
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def store_mytag_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await context.bot.send_message(
        query.message.chat_id,
        "🏷 Your tag is added to every quiz you post. Set it with:\n`/settag Your Tag Here`",
        parse_mode="Markdown",
    )


async def store_mydesc_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await context.bot.send_message(
        query.message.chat_id,
        "💬 Your description tag is added as the explanation on every quiz you post. Set it with:\n`/setdescription Your Text Here`",
        parse_mode="Markdown",
    )


async def default_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🟢 Easy", callback_data="setdef_diff_easy"),
         InlineKeyboardButton("🟡 Medium", callback_data="setdef_diff_medium"),
         InlineKeyboardButton("🔴 Hard", callback_data="setdef_diff_hard")],
        [InlineKeyboardButton("Clear default", callback_data="setdef_diff_")],
        [InlineKeyboardButton("🔙 Back to Tools", callback_data="open_tools")],
    ])
    await update.message.reply_text("Pick your default difficulty (skips the prompt next time):", reply_markup=keyboard)


async def receive_default_diff(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    diff = query.data.replace('setdef_diff_', '')
    set_user_field(update.effective_user.id, 'default_difficulty', diff)
    await query.edit_message_text(f"✅ Default difficulty: {diff or '(cleared)'}", reply_markup=back_to_tools_keyboard())


# ==========================================
# 🔑 BRING-YOUR-OWN-API-KEY CONVERSATION
# Every user can add AS MANY keys as they want — any mix of Groq / Gemini /
# OpenAI / Anthropic / open-source (OpenAI-compatible) endpoints — with
# /addapi. All saved in SQLite (user_api_keys), so they survive bot restarts.
# Generation tries them in the order added; if one fails (bad key, model can't
# read an image, provider down, etc.) it automatically moves to the next —
# and the owner's shared key is always the guaranteed final fallback.
# ==========================================
def api_provider_keyboard():
    rows, row = [], []
    for key, preset in PROVIDER_PRESETS.items():
        row.append(InlineKeyboardButton(preset["label"], callback_data=f"apiprov_{key}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⏭ Skip — meri koi API nahi hai", callback_data="apiprov_skip")])
    rows.append([InlineKeyboardButton("🗑 Remove ALL my keys", callback_data="apiprov_removeall")])
    rows.append([InlineKeyboardButton("🔙 Cancel", callback_data="apiprov_cancel")])
    return InlineKeyboardMarkup(rows)


async def addapi_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    keys = get_user_api_keys(update.effective_user.id)
    if keys:
        lines = []
        for k in keys:
            preset = PROVIDER_PRESETS.get(k['provider'], {})
            lines.append(f"• #{k['id']} {preset.get('label', k['provider'])} — `{mask_key(k['api_key'])}`")
        status = "Your saved keys (tried in this order, then the owner's key as final fallback):\n" + "\n".join(lines)
    else:
        status = "No personal keys yet — you're running on the bot's shared default key (if the owner has set one)."
    await msg.reply_text(
        f"🔑 **{stylize('Add An API Key')}**\n_{FONT_STYLE}_\n\n"
        f"{status}\n\n"
        "You can add **as many keys/providers as you like** — 1, 10, whatever. If one fails "
        "or can't handle something (e.g. a text-only model given a photo), the bot automatically "
        "tries the next one. The owner's key always works as the last resort either way.\n\n"
        "**API key nahi hai?** Koi tension nahi — neeche **⏭ Skip** dabao, quiz banana bilkul "
        "band nahi hoga, bas bot ki apni shared key use hogi.\n\n"
        "Pick a provider to add:",
        parse_mode="Markdown",
        reply_markup=api_provider_keyboard(),
    )
    return ASK_API_PROVIDER


async def addapi_button_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    return await addapi_entry(update, context)


SKIP_API_MESSAGE = (
    "👍 Koi baat nahi — API key add karna **zaroori nahi hai**.\n\n"
    "Bina kisi key ke bhi tum seedha /topic, /pdf, /image ya /ask "
    "use kar sakte ho — koi bhi paragraph, topic ya file bhejo, bot apni khud ki "
    "shared key se automatically poll bana dega. Jab chaho, /addapi se apni key "
    "add kar sakte ho — tab tak ke liye ye default hi kaafi hai."
)


async def _addapi_skip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(SKIP_API_MESSAGE, parse_mode="Markdown")
    return ConversationHandler.END


def _is_skip_text(text: str) -> bool:
    return text.strip().lower() in ("skip", "/skip")


async def receive_api_provider(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    choice = query.data.replace("apiprov_", "")

    if choice == "cancel":
        await query.edit_message_text(
            "❌ Cancelled — nothing was lost. Your saved quizzes, API keys, groups and quota are all safe."
        )
        return ConversationHandler.END

    if choice == "removeall":
        delete_all_user_api_keys(update.effective_user.id)
        await query.edit_message_text("🗑 Removed all your saved keys — back to the bot's shared default key.")
        return ConversationHandler.END

    if choice == "skip":
        await query.edit_message_text(SKIP_API_MESSAGE, parse_mode="Markdown")
        return ConversationHandler.END

    context.user_data['api_provider'] = choice
    preset = PROVIDER_PRESETS[choice]

    if choice == "custom":
        await query.edit_message_text(
            "⚙️ Send the base URL of your OpenAI-compatible endpoint\n"
            "(e.g. `https://api.together.xyz/v1`, `https://openrouter.ai/api/v1`, "
            "or a local server like `http://localhost:11434/v1` for Ollama).\n\n"
            "Or type `skip` if you don't have one, or /cancel anytime.",
            parse_mode="Markdown",
        )
        return ASK_API_CUSTOM_BASE

    if preset.get("needs_base"):
        note = f"\n\n_{preset['note']}_" if preset.get("note") else ""
        await query.edit_message_text(
            f"⚙️ **{preset['label']}** needs your account-specific base URL.\n\n"
            "Cloudflare Workers AI URL format:\n"
            "`https://api.cloudflare.com/client/v4/accounts/<YOUR_ACCOUNT_ID>/ai/v1`\n\n"
            "(Apna Account ID Cloudflare dashboard → right sidebar par milega — usko "
            "`<YOUR_ACCOUNT_ID>` ki jagah daal kar poora URL bhejo.)"
            f"{note}\n\nOr type `skip` if you don't have this, or /cancel anytime.",
            parse_mode="Markdown",
        )
        return ASK_API_CUSTOM_BASE

    await query.edit_message_text(
        f"🔐 Send your **{preset['label']}** API key now.\n\n"
        "_Tip: delete your message right after — I'll confirm once it's saved, "
        "so it doesn't sit around in the chat history._\n\n"
        "Or type `skip` if you don't have one, or /cancel anytime.",
        parse_mode="Markdown",
    )
    return ASK_API_KEY


async def receive_api_custom_base(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _is_skip_text(update.message.text):
        return await _addapi_skip(update, context)
    base = update.message.text.strip()
    if not base.startswith("http"):
        await update.message.reply_text("That doesn't look like a URL — send the full base URL (starting with http/https), or type `skip`.")
        return ASK_API_CUSTOM_BASE
    context.user_data['api_base'] = base.rstrip('/')

    preset = PROVIDER_PRESETS.get(context.user_data.get('api_provider'), {})
    if preset.get("model"):
        # A fixed-model provider that only needed the base URL (e.g. Cloudflare) —
        # skip the free-text model step entirely and go straight to the key.
        context.user_data['api_model'] = preset["model"]
        await update.message.reply_text(
            f"Got it — using model `{preset['model']}`.\n\n"
            "Now send the API key for this endpoint, type `none` if it doesn't need one, "
            "or type `skip` if you don't have one at all.",
            parse_mode="Markdown",
        )
        return ASK_API_KEY

    await update.message.reply_text(
        "Now send the model name to use (e.g. `llama3.1`, `mixtral-8x7b-32768`, `qwen2.5:7b`), or type `skip`.",
        parse_mode="Markdown",
    )
    return ASK_API_CUSTOM_MODEL


async def receive_api_custom_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _is_skip_text(update.message.text):
        return await _addapi_skip(update, context)
    context.user_data['api_model'] = update.message.text.strip()[:100]
    await update.message.reply_text(
        "Send the API key for this endpoint, type `none` if it doesn't need one "
        "(e.g. a local server on your own machine), or type `skip` if you don't have one at all.",
        parse_mode="Markdown",
    )
    return ASK_API_KEY


async def receive_api_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _is_skip_text(update.message.text):
        return await _addapi_skip(update, context)
    key_text = update.message.text.strip()
    provider = context.user_data.get('api_provider')
    preset = PROVIDER_PRESETS.get(provider, {})
    api_key = "" if key_text.lower() == "none" else key_text
    api_base = context.user_data.get('api_base') or preset.get('api_base')
    model = context.user_data.get('api_model') or preset.get('model')

    if provider != "custom" and not api_key:
        await update.message.reply_text("That key looks empty — please send a valid key, type `skip`, or /cancel.")
        return ASK_API_KEY

    add_user_api_key(update.effective_user.id, provider, api_key, api_base, model)

    try:
        await update.message.delete()
    except Exception:
        pass

    await update.effective_chat.send_message(
        f"✅ Added! **{preset.get('label', provider)}** (key: `{mask_key(api_key)}`) is now in your "
        "fallback chain — the bot tries your keys in the order you added them, then the owner's "
        "key if all of yours fail.\n\n"
        "Add more anytime with /addapi, see the full list with /myapi.",
        parse_mode="Markdown",
    )
    context.user_data.clear()
    return ConversationHandler.END


async def limits_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/limits — shows today's tracked usage vs. daily cap for every key in this
    user's fallback chain (their own /addapi keys + the owner's shared key)."""
    user_id = update.effective_user.id
    chain = build_provider_chain(user_id)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    conn = db_connect()
    cur = conn.cursor()
    lines = ["📊 **Today's API usage** (UTC date, resets at 00:00 UTC):"]
    for cand in chain:
        limit = get_provider_daily_limit(cand["provider"])
        scope = _usage_scope(cand)
        cur.execute("SELECT count FROM api_key_usage WHERE scope=? AND usage_date=?", (scope, today))
        row = cur.fetchone()
        used = row[0] if row else 0
        if limit:
            pct = int(used * 100 / limit)
            bar = "🟥" if pct >= 90 else ("🟨" if pct >= 50 else "🟩")
            lines.append(f"{bar} {cand['label']} — {used}/{limit} ({pct}%)")
        else:
            lines.append(f"⬜ {cand['label']} — {used} calls today (no fixed cap tracked)")
    conn.close()
    lines.append(
        "\n_Ye published free-tier caps par based estimates hain (provider apni limits "
        "kabhi bhi badal sakta hai) — asli, live numbers ke liye hamesha provider ke apne "
        "dashboard ko check karo. Bot ka counter hamesha UTC midnight par reset hota hai._"
    )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def myapi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keys = get_user_api_keys(update.effective_user.id)
    if not keys:
        await update.message.reply_text("No personal keys saved — you're using the bot's shared default key. Add one with /addapi.")
        return
    lines = ["🔑 Your saved keys (tried in this order, then the bot's shared key as final fallback):"]
    for k in keys:
        preset = PROVIDER_PRESETS.get(k['provider'], {})
        lines.append(f"#{k['id']} — {preset.get('label', k['provider'])} — `{mask_key(k['api_key'])}` — model: {k['model'] or '(default)'}")
    lines.append("\nRemove one: /removeapi <id>  •  Remove all: /removeapi all")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown", reply_markup=back_to_tools_keyboard())


async def removeapi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Usage: /removeapi <id> — see IDs with /myapi. Or /removeapi all to remove everything.")
        return
    arg = context.args[0].strip().lower()
    if arg == "all":
        delete_all_user_api_keys(user_id)
        await update.message.reply_text("🗑 Removed all your saved keys — back to the bot's shared default key.")
        return
    try:
        key_id = int(arg)
    except ValueError:
        await update.message.reply_text("Usage: /removeapi <id> or /removeapi all")
        return
    delete_user_api_key_by_id(user_id, key_id)
    await update.message.reply_text(f"🗑 Removed key #{key_id} (if it existed and was yours).")


# ==========================================
# ✏️ EDIT CONVERSATION
# ==========================================
async def edit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    quizzes = get_user_quizzes(update.effective_user.id)
    if not quizzes:
        await update.message.reply_text("You have no saved quizzes to edit.")
        return ConversationHandler.END
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(q['title'][:30] or f"Quiz {q['id']}", callback_data=f"editpick_{q['id']}")]
        for q in quizzes[:10]
    ])
    await update.message.reply_text("Pick a quiz to edit:", reply_markup=keyboard)
    return ASK_EDIT_CHOICE


async def edit_pick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    quiz_id = int(query.data.replace('editpick_', ''))
    quiz = get_quiz_by_id(quiz_id)
    if not quiz or quiz['user_id'] != update.effective_user.id:
        await query.edit_message_text("Quiz not found.")
        return ConversationHandler.END
    context.user_data['editing_quiz_id'] = quiz_id
    lines = [f"{i + 1}. {q['question']}" for i, q in enumerate(quiz['questions'])]
    await query.edit_message_text("Reply with:\n<number> <new question text>\n\n" + "\n".join(lines))
    return ASK_EDIT_TEXT


async def edit_apply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ''
    parts = text.split(' ', 1)
    if len(parts) < 2 or not parts[0].isdigit():
        await update.message.reply_text("Format: <number> <new question text>")
        return ASK_EDIT_TEXT
    idx = int(parts[0]) - 1
    new_text = parts[1]
    quiz_id = context.user_data.get('editing_quiz_id')
    quiz = get_quiz_by_id(quiz_id) if quiz_id else None
    if not quiz or not (0 <= idx < len(quiz['questions'])):
        await update.message.reply_text("Invalid question number.")
        return ConversationHandler.END
    quiz['questions'][idx]['question'] = new_text[:290]
    update_quiz(quiz_id, quiz['questions'])
    await update.message.reply_text("✅ Updated. Run /mystore anytime to see your quizzes.")
    context.user_data.pop('editing_quiz_id', None)
    return ConversationHandler.END


# ==========================================
# 🛡️ GROUP MANAGEMENT
# ==========================================
async def mute_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_chat_admin(update, context):
        await update.message.reply_text("⛔ Admins only.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to a user's message with /mute.")
        return
    target = update.message.reply_to_message.from_user
    await context.bot.restrict_chat_member(
        update.effective_chat.id, target.id,
        permissions=ChatPermissions(can_send_messages=False)
    )
    await update.message.reply_text(f"🔇 Muted {target.first_name}.")


async def unmute_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_chat_admin(update, context):
        await update.message.reply_text("⛔ Admins only.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to a user's message with /unmute.")
        return
    target = update.message.reply_to_message.from_user
    await context.bot.restrict_chat_member(
        update.effective_chat.id, target.id,
        permissions=ChatPermissions(
            can_send_messages=True, can_send_audios=True, can_send_documents=True,
            can_send_photos=True, can_send_videos=True, can_send_video_notes=True,
            can_send_voice_notes=True, can_send_polls=True, can_send_other_messages=True,
            can_add_web_page_previews=True,
        )
    )
    await update.message.reply_text(f"🔊 Unmuted {target.first_name}.")


async def promote_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_chat_admin(update, context):
        await update.message.reply_text("⛔ Admins only.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to a user's message with /promote.")
        return
    target = update.message.reply_to_message.from_user
    await context.bot.promote_chat_member(
        update.effective_chat.id, target.id,
        can_delete_messages=True, can_restrict_members=True,
        can_pin_messages=True, can_invite_users=True,
    )
    await update.message.reply_text(f"⬆️ Promoted {target.first_name}.")


async def demote_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_chat_admin(update, context):
        await update.message.reply_text("⛔ Admins only.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to a user's message with /demote.")
        return
    target = update.message.reply_to_message.from_user
    await context.bot.promote_chat_member(
        update.effective_chat.id, target.id,
        can_delete_messages=False, can_restrict_members=False,
        can_pin_messages=False, can_invite_users=False,
    )
    await update.message.reply_text(f"⬇️ Demoted {target.first_name}.")


async def purge_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_chat_admin(update, context):
        await update.message.reply_text("⛔ Admins only.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to the first message you want deleted, with /purge.")
        return
    start_id = update.message.reply_to_message.message_id
    end_id = update.message.message_id
    deleted = 0
    for mid in range(start_id, end_id + 1):
        try:
            await context.bot.delete_message(update.effective_chat.id, mid)
            deleted += 1
        except Exception:
            pass
    await update.effective_chat.send_message(f"🧹 Purged {deleted} messages.")


async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target = update.message.reply_to_message.from_user if update.message.reply_to_message else update.effective_user
    text = (
        f"👤 Name: {target.first_name}\n"
        f"🆔 ID: {target.id}\n"
        f"🔗 Username: @{target.username if target.username else 'N/A'}"
    )
    await update.message.reply_text(text)


async def filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_chat_admin(update, context):
        await update.message.reply_text("⛔ Admins only.")
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /filter keyword reply text")
        return
    keyword = context.args[0].lower()
    reply = " ".join(context.args[1:])
    add_group_filter(update.effective_chat.id, keyword, reply)
    await update.message.reply_text(f"✅ Filter added for '{keyword}'.")


async def removefilter_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_chat_admin(update, context):
        await update.message.reply_text("⛔ Admins only.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /removefilter keyword")
        return
    keyword = context.args[0].lower()
    remove_group_filter(update.effective_chat.id, keyword)
    await update.message.reply_text(f"🗑 Filter '{keyword}' removed.")


async def filters_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = list_group_filters(update.effective_chat.id)
    if not rows:
        await update.message.reply_text("No filters set in this chat.")
        return
    await update.message.reply_text("Filters:\n" + "\n".join(f"• {r[0]}" for r in rows))


async def check_filters(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    text = update.message.text.lower()
    for keyword, reply in list_group_filters(update.effective_chat.id):
        if keyword in text:
            await update.message.reply_text(reply)
            break


# ==========================================
# 👋 GROUP WELCOME / GOODBYE
# ==========================================
GROUP_ADD_BONUS_QUESTIONS = 100


async def handle_new_chat_members(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    msg = update.message
    if not msg or not msg.new_chat_members:
        return
    try:
        me = await context.bot.get_me()
    except Exception:
        me = None
    bot_was_added = any(m.is_bot and me and m.id == me.id for m in msg.new_chat_members)
    if bot_was_added and not group_is_tracked(chat.id) and msg.from_user:
        increment_question_limit(msg.from_user.id, GROUP_ADD_BONUS_QUESTIONS)
        try:
            await context.bot.send_message(
                msg.from_user.id,
                f"🎉 **Group Added Successfully!**\n\nThanks for adding me to **{chat.title or 'your group'}**.\n"
                f"🎁 **+{GROUP_ADD_BONUS_QUESTIONS} bonus questions** added to your quota!",
                parse_mode="Markdown",
            )
        except Exception:
            pass  # they may not have started a DM with the bot yet — bonus is still credited
    track_group(chat.id, chat.title or "")
    settings = get_group_settings(chat.id)
    if not settings["enabled"]:
        return
    try:
        me = await context.bot.get_me()
    except Exception:
        me = None
    for member in msg.new_chat_members:
        if member.is_bot and me and member.id == me.id:
            continue  # the bot itself being added isn't a "new member" to welcome
        text = render_welcome_template(settings["welcome_msg"], member.first_name or "there", member.id, chat.title or "")
        try:
            await msg.reply_text(text, parse_mode="Markdown")
        except Exception as e:
            logger.warning(f"Welcome message send failed (non-fatal, group unaffected): {e}")


async def handle_left_chat_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    msg = update.message
    member = msg.left_chat_member if msg else None
    if not member or member.is_bot:
        return
    settings = get_group_settings(chat.id)
    if not settings["enabled"]:
        return
    safe_name = member.first_name or "Someone"
    text = (settings["goodbye_msg"] or DEFAULT_GOODBYE_MSG).replace("{name}", safe_name).replace(
        "{group}", chat.title or "the group")[:1000]
    try:
        await msg.reply_text(text)
    except Exception as e:
        logger.warning(f"Goodbye message send failed (non-fatal, group unaffected): {e}")


async def welcome_toggle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type not in ("group", "supergroup"):
        await update.message.reply_text("This only works inside a group.")
        return
    if not await is_chat_admin(update, context):
        await update.message.reply_text("⛔ Admins only.")
        return
    if not context.args or context.args[0].lower() not in ("on", "off"):
        await update.message.reply_text("Usage: `/welcome on` or `/welcome off`", parse_mode="Markdown")
        return
    enabled = context.args[0].lower() == "on"
    upsert_group_settings(update.effective_chat.id, welcome_enabled=1 if enabled else 0)
    await update.message.reply_text(f"✅ Welcome/goodbye messages are now **{'ON' if enabled else 'OFF'}** in this group.",
                                     parse_mode="Markdown")


async def setwelcome_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type not in ("group", "supergroup"):
        await update.message.reply_text("This only works inside a group.")
        return
    if not await is_chat_admin(update, context):
        await update.message.reply_text("⛔ Admins only.")
        return
    parts = update.message.text.split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text(
            "Usage: `/setwelcome Your message here`\nPlaceholders: `{mention}`, `{name}`, `{group}`",
            parse_mode="Markdown",
        )
        return
    upsert_group_settings(update.effective_chat.id, welcome_msg=parts[1].strip()[:1000])
    await update.message.reply_text("✅ Welcome message updated — applies from the next member who joins.")


async def setgoodbye_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type not in ("group", "supergroup"):
        await update.message.reply_text("This only works inside a group.")
        return
    if not await is_chat_admin(update, context):
        await update.message.reply_text("⛔ Admins only.")
        return
    parts = update.message.text.split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text(
            "Usage: `/setgoodbye Your message here`\nPlaceholders: `{name}`, `{group}`",
            parse_mode="Markdown",
        )
        return
    upsert_group_settings(update.effective_chat.id, goodbye_msg=parts[1].strip()[:1000])
    await update.message.reply_text("✅ Goodbye message updated.")


# ==========================================
# 📣 OWNER QUIZ-BLAST — one question, broadcast as a poll to every tracked group
# ==========================================
async def quizblast_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return ConversationHandler.END
    await update.message.reply_text(
        "📣 **Quiz Blast** — send ONE question in this format and I'll post it as a live "
        "quiz poll to every group I'm tracking:\n\n"
        "```\nQuestion text?\nOption A\nOption B ✅\nOption C\nOption D\n```",
        parse_mode="Markdown", reply_markup=cancel_keyboard(),
    )
    return ASK_QUIZBLAST_TEXT


async def quizblast_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    questions = parse_bulk_questions(update.message.text or "")
    if not questions:
        await update.message.reply_text("Couldn't parse that — check the format above and resend, or Cancel.",
                                          reply_markup=cancel_keyboard())
        return ASK_QUIZBLAST_TEXT
    q = questions[0]
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT group_id FROM groups")
    group_ids = [r[0] for r in cur.fetchall()]
    conn.close()

    ok, fail = 0, 0
    for gid in group_ids:
        try:
            await context.bot.send_poll(
                chat_id=gid, question=q['question'][:300], options=[o[:100] for o in q['options']],
                type=Poll.QUIZ, correct_option_id=q['correct_index'], is_anonymous=False,
            )
            ok += 1
        except Exception as e:
            fail += 1
            logger.warning(f"Quiz blast failed for group {gid} (skipped, others unaffected): {e}")
        await asyncio.sleep(0.05)
    await update.message.reply_text(f"✅ Blasted to {ok} group(s), failed for {fail}.")
    return ConversationHandler.END


# ==========================================
# 📥 /add_bulk — bulk-import a whole quiz from pasted text OR an uploaded .txt file
# Same "Question / options / ✅-marked answer, blank line between questions" format as
# everywhere else in the bot — /add_bulk just also accepts it as a file attachment.
# ==========================================
async def addbulk_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📥 **Add Bulk Quiz** — first, what should this quiz be titled?",
        parse_mode="Markdown", reply_markup=cancel_keyboard(),
    )
    return ADD_BULK_TITLE


async def addbulk_receive_title(update: Update, context: ContextTypes.DEFAULT_TYPE):
    title = (update.message.text or "").strip()[:60]
    if not title:
        await update.message.reply_text("Send a title (max 60 characters), or Cancel.", reply_markup=cancel_keyboard())
        return ADD_BULK_TITLE
    context.user_data['addbulk_title'] = title
    await update.message.reply_text(
        "Now send the questions — **either** paste them as a message, **or** upload a `.txt` file "
        "in this format (blank line between questions, ✅ on the correct option):\n\n"
        "```\nQuestion text?\nOption A\nOption B ✅\nOption C\nOption D\n\nNext question?\nOption A ✅\nOption B\n```",
        parse_mode="Markdown", reply_markup=cancel_keyboard(),
    )
    return ADD_BULK_CONTENT


async def addbulk_receive_content(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    title = context.user_data.get('addbulk_title', 'Quiz')

    if msg.document:
        try:
            tg_file = await msg.document.get_file()
            raw = await tg_file.download_as_bytearray()
            text = bytes(raw).decode("utf-8", errors="ignore")
        except Exception as e:
            await update.message.reply_text(f"❌ Couldn't read that file ({e}) — try pasting the text instead, or Cancel.",
                                              reply_markup=cancel_keyboard())
            return ADD_BULK_CONTENT
    else:
        text = msg.text or ""

    questions = parse_bulk_questions(text)
    if not questions:
        await update.message.reply_text(
            "Couldn't find any valid questions in that — check the format shown above and resend "
            "(text or .txt file), or Cancel.",
            reply_markup=cancel_keyboard(),
        )
        return ADD_BULK_CONTENT

    user = update.effective_user
    quiz_id = save_quiz(user.id, title, questions, quiz_type='official', creator_name=user.first_name or 'Unknown')
    context.user_data.pop('addbulk_title', None)
    text_out, keyboard = _store_manage_view(quiz_id)
    await update.message.reply_text(f"✅ Imported {len(questions)} question(s).\n\n{text_out}",
                                     parse_mode="Markdown", reply_markup=keyboard)
    return ConversationHandler.END


# ==========================================
# ⏰ SCHEDULE
# ==========================================
# ==========================================
# ⏰ SCHEDULE — owner-only: pick a saved quiz, pick a repeat interval, and it
# broadcasts as a live quiz — same ready-check flow as any group quiz — to
# every group the bot is CURRENTLY an admin in (checked live on every run,
# not cached, since admin status can change), on that interval, until cancelled.
# ==========================================
SCHEDULE_INTERVALS = [("1 hour", 60), ("6 hours", 360), ("12 hours", 720), ("24 hours", 1440)]
SCHEDULE_PAGE_SIZE = 8


def create_recurring_schedule(quiz_id: int, interval_minutes: int, created_by: int) -> int:
    next_run = (datetime.utcnow() + timedelta(minutes=interval_minutes)).isoformat()
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO scheduled_quizzes (quiz_id, interval_minutes, next_run, created_by) "
        "VALUES (?, ?, ?, ?)",
        (quiz_id, interval_minutes, next_run, created_by),
    )
    conn.commit()
    schedule_id = last_insert_id(cur, conn, "scheduled_quizzes")
    conn.close()
    return schedule_id


def get_all_recurring_schedules() -> list:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT sq.id, sq.quiz_id, sq.interval_minutes, sq.next_run, q.title "
        "FROM scheduled_quizzes sq LEFT JOIN quizzes q ON q.id = sq.quiz_id "
        "WHERE sq.quiz_id IS NOT NULL ORDER BY sq.id"
    )
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "quiz_id": r[1], "interval_minutes": r[2], "next_run": r[3],
             "title": r[4] or "(deleted quiz)"} for r in rows]


def update_schedule_next_run(schedule_id: int, next_run_iso: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE scheduled_quizzes SET next_run=? WHERE id=?", (next_run_iso, schedule_id))
    conn.commit()
    conn.close()


def delete_recurring_schedule(schedule_id: int) -> bool:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM scheduled_quizzes WHERE id=? AND quiz_id IS NOT NULL", (schedule_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


async def broadcast_quiz_to_admin_groups(bot, application, quiz_id: int):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT group_id FROM groups")
    group_ids = [r[0] for r in cur.fetchall()]
    conn.close()

    ok, not_admin, failed = 0, 0, 0
    for gid in group_ids:
        try:
            member = await bot.get_chat_member(gid, bot.id)
            if member.status != "administrator":
                not_admin += 1
                continue
            await initiate_live_quiz(bot, application, gid, "group", quiz_id, OWNER_ID, "Scheduled Broadcast")
            ok += 1
        except Exception as e:
            failed += 1
            logger.warning(f"Scheduled broadcast skipped for group {gid} (non-fatal): {e}")
        await asyncio.sleep(0.05)
    return ok, not_admin, failed


async def run_scheduled_broadcast(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data
    schedule_id, quiz_id = data["schedule_id"], data["quiz_id"]
    if not get_quiz_meta(quiz_id):
        delete_recurring_schedule(schedule_id)
        context.job.schedule_removal()
        return
    ok, not_admin, failed = await broadcast_quiz_to_admin_groups(context.bot, context.application, quiz_id)
    next_run = (datetime.utcnow() + timedelta(minutes=data["interval_minutes"])).isoformat()
    update_schedule_next_run(schedule_id, next_run)
    logger.info(f"Scheduled broadcast #{schedule_id}: {ok} sent, {not_admin} not-admin, {failed} failed.")


def register_recurring_schedule_job(application, schedule_id: int, quiz_id: int, interval_minutes: int):
    if not application.job_queue:
        return
    application.job_queue.run_repeating(
        run_scheduled_broadcast, interval=interval_minutes * 60, first=interval_minutes * 60,
        data={"schedule_id": schedule_id, "quiz_id": quiz_id, "interval_minutes": interval_minutes},
        name=f"sched_{schedule_id}",
    )


def _schedule_quiz_picker_view(user_id: int, page: int = 0):
    quizzes = get_user_quizzes(user_id)
    if not quizzes:
        return "You don't have any saved quizzes yet — create one first.", None
    total_pages = max(1, (len(quizzes) + SCHEDULE_PAGE_SIZE - 1) // SCHEDULE_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * SCHEDULE_PAGE_SIZE
    page_quizzes = quizzes[start:start + SCHEDULE_PAGE_SIZE]

    rows = [[InlineKeyboardButton(f"📁 {q['title'][:35]}", callback_data=f"schedq_{q['id']}")]
            for q in page_quizzes]
    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("⬅️ Back", callback_data=f"schedqpage_{page - 1}"))
        nav.append(InlineKeyboardButton(f"📄 {page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton("Next ➡️", callback_data=f"schedqpage_{page + 1}"))
        rows.append(nav)
    text = "⏰ **Schedule a recurring broadcast** — pick a quiz to send on a repeating timer:"
    return text, InlineKeyboardMarkup(rows)


async def schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only — this broadcasts to every group I'm admin in.")
        return
    if not context.job_queue:
        await update.message.reply_text(
            "Scheduling isn't available — install the job-queue extra "
            "(pip install \"python-telegram-bot[job-queue]\")."
        )
        return
    text, keyboard = _schedule_quiz_picker_view(update.effective_user.id, 0)
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def schedule_quiz_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.from_user.id != OWNER_ID:
        return
    page = int(query.data.replace("schedqpage_", ""))
    text, keyboard = _schedule_quiz_picker_view(query.from_user.id, page)
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def schedule_pick_quiz_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.from_user.id != OWNER_ID:
        return
    quiz_id = int(query.data.replace("schedq_", ""))
    quiz = get_quiz_meta(quiz_id)
    if not quiz:
        await query.edit_message_text("❌ That quiz no longer exists.")
        return
    keyboard = [[InlineKeyboardButton(f"Every {label}", callback_data=f"schedi_{quiz_id}_{minutes}")]
                for label, minutes in SCHEDULE_INTERVALS]
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="schedqpage_0")])
    await query.edit_message_text(
        f"⏰ **{quiz['title']}** — how often should this go out to every group I'm admin in?",
        parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def schedule_pick_interval_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.from_user.id != OWNER_ID:
        return
    _, quiz_id_str, minutes_str = query.data.split("_")
    quiz_id, minutes = int(quiz_id_str), int(minutes_str)
    quiz = get_quiz_meta(quiz_id)
    if not quiz:
        await query.edit_message_text("❌ That quiz no longer exists.")
        return

    schedule_id = create_recurring_schedule(quiz_id, minutes, query.from_user.id)
    register_recurring_schedule_job(context.application, schedule_id, quiz_id, minutes)

    interval_label = f"{minutes // 60} hour(s)" if minutes % 60 == 0 else f"{minutes} minute(s)"
    await query.edit_message_text(
        f"✅ **{quiz['title']}** will broadcast every {interval_label} to every group I'm "
        f"currently an admin in.\nManage this anytime with /myschedules.",
        parse_mode="Markdown",
    )


async def myschedules_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.effective_message.reply_text("⛔ Owner only.")
        return
    schedules = get_all_recurring_schedules()
    if not schedules:
        await update.effective_message.reply_text("No recurring broadcasts set up.\nUse /schedule to create one.")
        return
    now = datetime.utcnow()
    lines = [f"📅 **{stylize('Scheduled Broadcasts')}**\n_{FONT_STYLE}_\n"]
    keyboard = []
    for s in schedules:
        interval_label = (f"every {s['interval_minutes'] // 60}h" if s['interval_minutes'] % 60 == 0
                           else f"every {s['interval_minutes']}m")
        try:
            mins_left = max(0, int((datetime.fromisoformat(s["next_run"]) - now).total_seconds() // 60))
            when = f"next in ~{mins_left} min"
        except (ValueError, TypeError):
            when = "next run time unknown"
        lines.append(f"• #{s['id']} — **{s['title']}** — {interval_label}, {when}")
        keyboard.append([InlineKeyboardButton(f"❌ Cancel #{s['id']}", callback_data=f"schedcancel_{s['id']}")])
    await update.effective_message.reply_text(
        "\n".join(lines), parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def schedule_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.from_user.id != OWNER_ID:
        return
    schedule_id = int(query.data.replace("schedcancel_", ""))
    deleted = delete_recurring_schedule(schedule_id)
    if context.job_queue:
        for job in context.job_queue.get_jobs_by_name(f"sched_{schedule_id}"):
            job.schedule_removal()

    await query.edit_message_text(
        f"🗑 Cancelled broadcast #{schedule_id}." if deleted else "❌ Not found — may already be cancelled."
    )


# ==========================================
# 🎛️ CALLBACK QUERY ROUTER (menu navigation + verification)
# ==========================================
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id
    user_name = query.from_user.first_name

    # Lightweight anti-flood: a user double/triple-tapping the same button (or mashing
    # buttons) can otherwise trigger duplicate work and stack up Telegram rate-limit
    # errors. This blocks nothing a real single tap would ever hit.
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
                parse_mode="Markdown", reply_markup=main_welcome_keyboard()
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
                await query.answer(
                    "🔐 Please join our update channel & group first, then tap ✅ I Am Verified.",
                    show_alert=True,
                )
                return

    await query.answer()

    if data == "menu_main":
        user_data = get_user(user_id)
        await query.edit_message_text(
            text=build_welcome_text(user_name, user_data),
            parse_mode="Markdown", reply_markup=main_welcome_keyboard()
        )

    elif data == "open_tools":
        await query.edit_message_text(
            text=f"🛠️ **{stylize('Tools & Features Menu')}**\n_{FONT_STYLE}_\n\nSelect any tool below to proceed:",
            parse_mode="Markdown",
            reply_markup=tools_menu_keyboard()
        )

    elif data == "open_help":
        help_text = (
            f"📖 **{stylize('Help & Commands List')}**\n"
            f"_{FONT_STYLE}_\n\n"
            f"📌 Use the **🛠️ Tools** button to access creation tools, or run commands like `/topic`, `/pdf`, `/mystore` directly."
        )
        await query.edit_message_text(
            text=help_text,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Main Menu", callback_data="menu_main")]])
        )

    elif data == "create_poll_clean":
        await query.edit_message_text(
            text=(
                f"🔄 **{stylize('Poll to Quiz')}**\n_{FONT_STYLE}_\n\n"
                "Just forward any MCQ poll into this chat — no command needed.\n"
                "If it's closed, I'll auto-detect the right answer; otherwise I'll ask you to pick it.\n"
                "You can forward several polls at once."
            ),
            parse_mode="Markdown", reply_markup=back_to_tools_keyboard()
        )

    elif data == "act_regenerate":
        await do_regenerate(context.bot, query.message.chat_id, user_id)

    elif data == "menu_customize":
        await query.edit_message_text(
            text=(
                f"⚙️ **{stylize('Customize')}**\n_{FONT_STYLE}_\n\n"
                "• `/settag YourTag` — shown under every question\n"
                "• `/setdescription YourTag` — shown in the explanation (💡)\n"
                "• `/default` — set a default difficulty"
            ),
            parse_mode="Markdown", reply_markup=back_to_tools_keyboard()
        )

    elif data == "menu_store":
        text, keyboard = _store_list_view(user_id, 0)
        await query.edit_message_text(text=text, parse_mode="Markdown", reply_markup=keyboard)

    elif data == "menu_publicstore":
        text, keyboard = _public_store_view(0)
        await query.edit_message_text(text=text, parse_mode="Markdown", reply_markup=keyboard)

    elif data == "menu_group":
        await query.edit_message_text(
            text=(
                f"🛡️ **{stylize('Group Management Panel')}**\n_{FONT_STYLE}_\n\n"
                "Use these directly in your group (admins only):\n"
                "`/mute`, `/unmute`, `/promote`, `/demote`, `/purge`, `/info`, `/filter`, `/removefilter`, `/filters`"
            ),
            parse_mode="Markdown", reply_markup=back_to_tools_keyboard()
        )


# 🚨 GLOBAL ERROR HANDLER
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catches anything an individual handler didn't catch itself, so ONE bad update
    never brings the whole bot down. Classifies the common "bot goes offline" causes
    (Conflict / network hiccups) so the log actually tells you what happened instead of
    a bare traceback, and — for Conflict specifically — DMs the owner, since that one
    means a SECOND copy of the bot is polling with the same token (e.g. an old Render
    deploy that never fully stopped) and code alone can't fix that; the running
    instance needs to be stopped.
    """
    err = context.error
    err_name = err.__class__.__name__

    if err_name == "Conflict":
        logger.error(
            "⚠️ Telegram 'Conflict' — another instance of this bot is ALSO polling with "
            "the same BOT_TOKEN right now (old deploy still running, or the bot running "
            "in two places at once). Stop the other instance — the bot will keep retrying "
            "in the meantime but updates will be split unpredictably between the two."
        )
        try:
            if OWNER_ID:
                await context.bot.send_message(
                    OWNER_ID,
                    "⚠️ Bot Conflict error: a SECOND instance of this bot is polling with the "
                    "same token (old deploy / duplicate process). Stop the other one — this "
                    "is very likely why the bot has been going offline/misbehaving.",
                )
        except Exception:
            pass
        return

    if err_name in ("NetworkError", "TimedOut", "RetryAfter"):
        logger.warning(f"Transient network issue ({err_name}): {err} — python-telegram-bot will retry automatically.")
        return

    if err_name == "Forbidden":
        # A user blocking the bot, or the bot getting removed/blocked from a group, is
        # normal and expected — it must NEVER crash anything or need owner attention.
        # Just log it quietly and move on; that one user/group is simply skipped next time.
        logger.info(f"Forbidden (bot blocked/removed by a user or group) — ignored safely: {err}")
        return

    if err_name == "BadRequest" and "not modified" in str(err).lower():
        # Editing a message with the exact same text/keyboard it already has — harmless,
        # not worth logging as an error.
        logger.debug(f"Harmless no-op edit ignored: {err}")
        return

    logger.error("Exception while handling an update:", exc_info=err)


# ==========================================
# 🌐 KEEP-ALIVE WEB SERVER (Render/Railway/Replit free "Web Service" tier)
# ==========================================
# Render's free plan only offers *Web Services* for free (background workers are
# paid) — but a Web Service must bind to $PORT within a startup timeout, or Render
# kills it as "Timed Out". This bot only does Telegram long-polling and never opens
# a port, so that's exactly what was happening. This tiny HTTP server runs in a
# background thread just so Render sees a live port; it also gives you a URL to
# point a free uptime pinger (UptimeRobot / cron-job.org, every 5–10 min) at, so the
# free instance doesn't spin down from inactivity after 15 min.
class _PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Quiz bot is running.")

    def log_message(self, format, *args):
        pass  # keep Render's log output clean — Telegram updates are logged separately


def start_keepalive_server():
    port = int(os.getenv("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), _PingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"🌐 Keep-alive HTTP server listening on port {port} (for Render health check + UptimeRobot)")



# ==========================================
# 🔗 MASTER COMPATIBILITY / SECOND-BOT FEATURES
# ==========================================
# These handlers are merged into the PTB engine instead of running a second
# TeleBot process. That avoids Telegram getUpdates conflicts and keeps one
# source of truth for the database/session state.

async def newquiz_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Official Quiz Bot-style builder alias. /newquiz uses the same stable
    title → timer → forward/bulk → ready flow as /officialquiz."""
    return await officialquiz_entry(update, context)

async def my_quizzes_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await mystore_command(update, context)

async def settings_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Keep the second bot's /settings name while using the richer first-bot panel.
    await update.effective_message.reply_text(
        f"⚙️ **{stylize('Personal Settings')}**\n_{FONT_STYLE}_\n\n"
        "Use the buttons below or these commands:\n"
        "• /settag — personal tag\n"
        "• /setdescription — explanation/description tag\n"
        "• /optionstyle — option formatting\n"
        "• /default — default difficulty",
        parse_mode="Markdown", reply_markup=back_to_tools_keyboard()
    )

async def todaylb_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await leaderboard_command(update, context)

async def sync_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.effective_message.reply_text("⛔ Owner only.")
        return
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM groups")
    count = cur.fetchone()[0]
    conn.close()
    await update.effective_message.reply_text(f"🔄 Database synced. {count} tracked group(s) ready.")

async def stats_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.effective_message.reply_text("⛔ Owner only.")
        return
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users"); users = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM quizzes"); quizzes = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM groups"); groups = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*), COALESCE(SUM(correct_count+wrong_count),0) FROM quiz_results")
    results, total_answers = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM user_api_keys"); user_keys = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM daily_quiz_subs"); daily_subs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM feedback"); feedback_count = cur.fetchone()[0]
    conn.close()
    await update.effective_message.reply_text(
        "📊 **BOT STATISTICS**\n\n"
        f"👤 Users: {users}\n📚 Saved quizzes: {quizzes}\n"
        f"👥 Tracked groups: {groups}\n🏆 Result records: {results}  (total answers: {total_answers})\n"
        f"🔑 Personal API keys added: {user_keys}\n☀️ Daily quiz subscriptions: {daily_subs}\n"
        f"💬 Feedback received: {feedback_count}", parse_mode="Markdown"
    )

async def restart_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.effective_message.reply_text("⛔ Owner only.")
        return
    await update.effective_message.reply_text("♻️ Restart requested. The master run_forever loop will restart the process if it exits.")
    # Do not call os._exit from a Telegram handler; that can kill the free web
    # service before Render receives a clean signal. The wrapper handles crashes.

async def backup_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await backupnow_command(update, context)

async def del_quiz_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.effective_message.reply_text("⛔ Owner only."); return
    if not context.args:
        await update.effective_message.reply_text("Usage: /del_quiz <quiz_id>"); return
    try: qid=int(context.args[0])
    except ValueError:
        await update.effective_message.reply_text("❌ Invalid quiz ID."); return
    ok=delete_quiz(qid, OWNER_ID)
    await update.effective_message.reply_text("🗑 Quiz deleted." if ok else "❌ Quiz not found or not deletable.")

async def del_set_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.effective_message.reply_text("⛔ Owner only."); return
    if not context.args:
        await update.effective_message.reply_text("Usage: /del_set <set_id>"); return
    try: sid=int(context.args[0])
    except ValueError:
        await update.effective_message.reply_text("❌ Invalid set ID."); return
    ok=delete_quiz_set(sid, OWNER_ID)
    await update.effective_message.reply_text("🗑 Set deleted." if ok else "❌ Set not found.")

async def del_duplicate_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.effective_message.reply_text("⛔ Owner only."); return
    conn=db_connect(); cur=conn.cursor()
    cur.execute("SELECT id, data FROM quizzes ORDER BY id")
    rows=cur.fetchall(); seen=set(); deleted=0
    for r in rows:
        qid=r[0]; data=r[1]
        try:
            questions=json.loads(data)
        except Exception:
            continue
        sig=json.dumps(questions, sort_keys=True, ensure_ascii=False)
        if sig in seen:
            cur.execute("DELETE FROM quizzes WHERE id=?", (qid,)); deleted+=1
        else:
            seen.add(sig)
    conn.commit(); conn.close()
    await update.effective_message.reply_text(f"🧹 Duplicate cleanup complete: {deleted} duplicate quiz set(s) removed.")

async def manage_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.effective_message.reply_text("⛔ Owner only."); return
    await mystore_command(update, context)

async def quiz_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /quiz from the second bot becomes the richer existing broadcast flow.
    return await quizblast_entry(update, context)

async def fc_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "📢 Channel forwarding is handled by /setchannel. Add the bot as channel admin, "
        "then forward a message from the channel here.\n\nUse /mychannels to manage links."
    )

async def fg_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "👥 Group routing is handled by the tracked-group system. Add the bot to the target group; "
        "then use Official Quiz /startquiz or /quizblast to post there."
    )

async def set_fc_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Use /setchannel for channel routing. The bot must be an admin in the target channel.")

async def set_fg_compat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Add the bot to the target group and run /start or /sync; tracked groups are persisted automatically.")


# ==========================================
# 🏁 MAIN ENTRY POINT
# ==========================================
def main():
    init_db()
    run_migrations()
    sync_serial_sequences()
    start_keepalive_server()

    async def _post_init(application):
        # Default executor is only min(32, cpu_count+4) workers — too small once
        # several users are generating AI quizzes (each now running several
        # concurrent batches, see generate_mcqs_async) at the same time. A bigger
        # pool means one user's AI generation is far less likely to make everyone
        # else's commands/polls sit and wait for a free thread.
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=64, thread_name_prefix="quizbot_worker"))
        logger.info("Thread pool executor enlarged to 64 workers.")

    request_config = HTTPXRequest(
        connect_timeout=60.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=60.0
    )

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(request_config)
        .get_updates_request(request_config)
        .post_init(_post_init)
        .build()
    )

    app.add_error_handler(error_handler)

    # --- Verification gate: runs before every /command except /start (group=-1,
    # so it fires before the handler that would otherwise process the command) ---
    app.add_handler(MessageHandler(filters.COMMAND, verification_command_gate), group=-1)

    # --- Leave-detection: needs the bot to be an ADMIN in UPDATE_CHANNEL and
    # UPDATE_GROUP to receive these membership updates at all ---
    app.add_handler(ChatMemberHandler(handle_update_channel_group_membership, ChatMemberHandler.CHAT_MEMBER))

    # --- Basic commands ---
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("create", create_command))
    app.add_handler(CommandHandler("stoppoll", stoppoll_command))
    app.add_handler(CommandHandler("pause", pause_command))
    app.add_handler(CommandHandler("resume", resume_command))
    app.add_handler(CommandHandler("stopquiz", stopquiz_command))
    app.add_handler(CommandHandler("undoquiz", undoquiz_command))
    app.add_handler(CommandHandler("fastquiz", fastquiz_command))
    app.add_handler(CommandHandler("slowquiz", slowquiz_command))

    # --- Compatibility commands merged from the second bot ---
    app.add_handler(CommandHandler("newquiz", newquiz_compat))
    app.add_handler(CommandHandler("my_quizzes", my_quizzes_compat))
    app.add_handler(CommandHandler("settings", settings_compat))
    app.add_handler(CommandHandler("todaylb", todaylb_compat))
    app.add_handler(CommandHandler("stats", stats_compat))
    app.add_handler(CommandHandler("sync", sync_compat))
    app.add_handler(CommandHandler("restart", restart_compat))
    app.add_handler(CommandHandler("backup", backup_compat))
    app.add_handler(CommandHandler("del_quiz", del_quiz_compat))
    app.add_handler(CommandHandler("del_set", del_set_compat))
    app.add_handler(CommandHandler("del_duplicate", del_duplicate_compat))
    app.add_handler(CommandHandler("manage", manage_compat))
    app.add_handler(CommandHandler("quiz", quiz_compat))
    app.add_handler(CommandHandler("fc", fc_compat))
    app.add_handler(CommandHandler("fg", fg_compat))
    app.add_handler(CommandHandler("set_fc", set_fc_compat))
    app.add_handler(CommandHandler("set_fg", set_fg_compat))

    # --- AI quiz generation conversation ---
    ai_conv = ConversationHandler(
        entry_points=[
            CommandHandler("topic", topic_entry),
            CommandHandler("pdf", pdf_entry),
            CommandHandler("image", image_entry),
            CommandHandler("section", section_entry),
            CallbackQueryHandler(topic_button_entry, pattern="^create_topic$"),
            CallbackQueryHandler(pdf_button_entry, pattern="^create_pdf$"),
            CallbackQueryHandler(image_button_entry, pattern="^create_image$"),
        ],
        states={
            ASK_CONTENT: [
                CommandHandler("done", receive_content),
                MessageHandler((filters.TEXT | filters.Document.ALL | filters.PHOTO) & ~filters.COMMAND, receive_content),
            ],
            ASK_DIFFICULTY: [CallbackQueryHandler(receive_difficulty, pattern="^diff_")],
            ASK_LANGUAGE: [
                CallbackQueryHandler(receive_language_button, pattern="^lang_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_language_text),
            ],
            ASK_AI_TIMER: [CallbackQueryHandler(receive_ai_timer, pattern="^aitimer_")],
            ASK_AI_SHUFFLE: [CallbackQueryHandler(receive_ai_shuffle, pattern="^aishuffle_")],
            ASK_COUNT: [CallbackQueryHandler(receive_count, pattern="^cnt_")],
            ASK_SECTION_COUNT: [CallbackQueryHandler(receive_section_count, pattern="^sec_")],
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
            ASK_DIFFICULTY: [CallbackQueryHandler(receive_difficulty, pattern="^diff_")],
            ASK_LANGUAGE: [
                CallbackQueryHandler(receive_language_button, pattern="^lang_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_language_text),
            ],
            ASK_AI_TIMER: [CallbackQueryHandler(receive_ai_timer, pattern="^aitimer_")],
            ASK_AI_SHUFFLE: [CallbackQueryHandler(receive_ai_shuffle, pattern="^aishuffle_")],
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

    # --- Import from Channel/Group conversation (forward messages, no session needed) ---
    import_conv = ConversationHandler(
        entry_points=[
            CommandHandler("importchat", import_entry),
            CallbackQueryHandler(import_button_entry, pattern="^create_import$"),
        ],
        states={
            IMPORT_COLLECT: [
                CallbackQueryHandler(import_done_button, pattern="^import_done$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_import_text),
            ],
            IMPORT_KEYWORD: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_import_keyword)],
            ASK_DIFFICULTY: [CallbackQueryHandler(receive_difficulty, pattern="^diff_")],
            ASK_LANGUAGE: [
                CallbackQueryHandler(receive_language_button, pattern="^lang_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_language_text),
            ],
            ASK_AI_TIMER: [CallbackQueryHandler(receive_ai_timer, pattern="^aitimer_")],
            ASK_AI_SHUFFLE: [CallbackQueryHandler(receive_ai_shuffle, pattern="^aishuffle_")],
            ASK_COUNT: [CallbackQueryHandler(receive_count, pattern="^cnt_")],
        },
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            CommandHandler("stop", cancel_command),
            CallbackQueryHandler(cancel_callback, pattern="^gcancel$"),
        ],
        conversation_timeout=1200,
    )
    app.add_handler(import_conv)

    # --- CSV conversation ---
    csv_conv = ConversationHandler(
        entry_points=[
            CommandHandler("csv", csv_entry),
            CallbackQueryHandler(csv_button_entry, pattern="^create_csv$"),
        ],
        states={
            ASK_CSV: [MessageHandler(filters.Document.ALL & ~filters.COMMAND, receive_csv)],
        },
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            CommandHandler("stop", cancel_command),
            CallbackQueryHandler(cancel_callback, pattern="^gcancel$"),
        ],
        conversation_timeout=600,
    )
    app.add_handler(csv_conv)

    # --- Edit conversation ---
    edit_conv = ConversationHandler(
        entry_points=[CommandHandler("edit", edit_command)],
        states={
            ASK_EDIT_CHOICE: [CallbackQueryHandler(edit_pick, pattern="^editpick_")],
            ASK_EDIT_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_apply)],
        },
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            CommandHandler("stop", cancel_command),
            CallbackQueryHandler(cancel_callback, pattern="^gcancel$"),
        ],
        conversation_timeout=600,
    )
    app.add_handler(edit_conv)

    # --- Bring-your-own-API-key conversation ---
    api_conv = ConversationHandler(
        entry_points=[
            CommandHandler("addapi", addapi_entry),
            CallbackQueryHandler(addapi_button_entry, pattern="^menu_apikey$"),
        ],
        states={
            ASK_API_PROVIDER: [CallbackQueryHandler(receive_api_provider, pattern="^apiprov_")],
            ASK_API_CUSTOM_BASE: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_api_custom_base)],
            ASK_API_CUSTOM_MODEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_api_custom_model)],
            ASK_API_KEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_api_key)],
        },
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            CommandHandler("stop", cancel_command),
            CallbackQueryHandler(cancel_callback, pattern="^gcancel$"),
        ],
        conversation_timeout=600,
    )
    app.add_handler(api_conv)
    app.add_handler(CommandHandler("myapi", myapi_command))
    app.add_handler(CommandHandler("removeapi", removeapi_command))
    app.add_handler(CommandHandler("limits", limits_command))
    app.add_handler(CommandHandler("ask", ask_command))

    # --- Official Quiz Bot conversation (title → timer → forward/bulk → summary) ---
    oq_conv = ConversationHandler(
        entry_points=[
            CommandHandler("officialquiz", officialquiz_entry),
            CallbackQueryHandler(officialquiz_button_entry, pattern="^menu_official$"),
        ],
        states={
            OQ_TITLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, oq_receive_title)],
            OQ_TIMER: [CallbackQueryHandler(oq_receive_timer, pattern="^oqt_")],
            OQ_METHOD: [CallbackQueryHandler(oq_receive_method, pattern="^oqm_")],
            OQ_COLLECT: [
                CallbackQueryHandler(oq_forward_done, pattern="^oqdone_forward$"),
                CallbackQueryHandler(oq_poll_correct_choice, pattern="^oqpollcorrect_"),
                MessageHandler(filters.POLL, handle_forwarded_poll),
            ],
            OQ_BULK: [
                CallbackQueryHandler(oq_bulk_done, pattern="^oqdone_bulk$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, oq_receive_bulk),
            ],
        },
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            CommandHandler("stop", cancel_command),
            CallbackQueryHandler(cancel_callback, pattern="^gcancel$"),
        ],
        conversation_timeout=1800,  # 30 min — enough headroom for a large (100s of polls) batch
    )
    app.add_handler(oq_conv)
    app.add_handler(CommandHandler("startquiz", startquiz_command))
    app.add_handler(CallbackQueryHandler(oqstart_callback, pattern="^oqstart_"))
    app.add_handler(CallbackQueryHandler(oqshare_callback, pattern="^oqshare_"))
    # 🚪 Lobby / ready-check — group starts wait for 2+ players before firing questions
    app.add_handler(CallbackQueryHandler(lobby_ready_callback, pattern="^oqready_"))
    app.add_handler(CallbackQueryHandler(lobby_cancel_callback, pattern="^oqreadycancel_"))
    app.add_handler(PollAnswerHandler(handle_poll_answer))

    # --- Poll -> quiz conversion (stateless — outside the Official Quiz flow) ---
    app.add_handler(MessageHandler(filters.POLL, handle_forwarded_poll))
    app.add_handler(CallbackQueryHandler(handle_poll_correct_choice, pattern="^pollcorrect_"))

    # --- Poll -> Text tool (stateless — forward polls, get one .txt back) ---
    app.add_handler(CallbackQueryHandler(polltxt_start_callback, pattern="^polltxt_start$"))
    app.add_handler(CallbackQueryHandler(polltxt_poll_correct_choice, pattern="^polltxtcorrect_"))
    app.add_handler(CallbackQueryHandler(polltxt_done_callback, pattern="^polltxt_done$"))
    app.add_handler(CallbackQueryHandler(polltxt_cancel_callback, pattern="^polltxt_cancel$"))

    # --- Settings / quota / store ---
    app.add_handler(CommandHandler("settag", settag_command))
    app.add_handler(CommandHandler("setdescription", setdescription_command))
    app.add_handler(CommandHandler("regenerate", regenerate_command))
    app.add_handler(CommandHandler("redeem", redeem_command))
    app.add_handler(CommandHandler("addcode", addcode_command))
    app.add_handler(CommandHandler("topusers", topusers_command))
    app.add_handler(CommandHandler(["profile", "mystats"], profile_command))
    app.add_handler(CommandHandler("duel", duel_command))
    app.add_handler(CommandHandler("dailyquiz", dailyquiz_command))
    app.add_handler(CommandHandler("schedulebroadcast", schedulebroadcast_command))
    app.add_handler(CommandHandler("feedback", feedback_command))
    app.add_handler(CommandHandler("granttop10", granttop10_command))
    app.add_handler(CommandHandler("broadcastcode", broadcastcode_command))
    app.add_handler(CommandHandler("broadcast", broadcast_command))
    app.add_handler(CommandHandler("backupnow", backupnow_command))
    app.add_handler(CommandHandler("restorebackup", restorebackup_command))
    app.add_handler(CommandHandler("mystore", mystore_command))
    app.add_handler(CallbackQueryHandler(store_manage_callback, pattern="^stmanage_"))
    app.add_handler(CallbackQueryHandler(store_list_page_callback, pattern="^stlist_"))
    app.add_handler(CallbackQueryHandler(lambda u, c: u.callback_query.answer(), pattern="^noop$"))
    app.add_handler(CallbackQueryHandler(store_delete_callback, pattern="^stdel_\\d"))
    app.add_handler(CallbackQueryHandler(store_delete_confirm_callback, pattern="^stdelok_"))
    app.add_handler(CallbackQueryHandler(store_toggle_public_callback, pattern="^stpub_\\d"))
    app.add_handler(CallbackQueryHandler(store_dedup_callback, pattern="^stdedup_\\d"))
    app.add_handler(CommandHandler("publicstore", publicstore_command))
    app.add_handler(CallbackQueryHandler(public_store_page_callback, pattern="^publist_\\d"))
    app.add_handler(CallbackQueryHandler(public_quiz_view_callback, pattern="^pubview_\\d"))
    app.add_handler(CallbackQueryHandler(store_remove_question_list, pattern="^stremq_\\d"))
    app.add_handler(CallbackQueryHandler(store_remove_question_confirm, pattern="^stremqok_"))
    app.add_handler(CallbackQueryHandler(store_mytag_callback, pattern="^stmytag$"))
    app.add_handler(CallbackQueryHandler(store_mydesc_callback, pattern="^stmydesc$"))
    # 📊 Status — Official Quiz editing section: user_id + right/wrong/accuracy/speed per player
    app.add_handler(CallbackQueryHandler(store_status_callback, pattern="^ststatus_"))
    # 📁 File-in-Set / 📤 Send-to-Channel / 🔤 Option Style, launched from the manage view
    app.add_handler(CallbackQueryHandler(store_set_pick_callback, pattern="^stsetpick_"))
    app.add_handler(CallbackQueryHandler(store_set_assign_callback, pattern="^stsetdo_"))
    app.add_handler(CallbackQueryHandler(store_send_channel_pick, pattern="^stsendch_"))
    app.add_handler(CallbackQueryHandler(store_send_channel_do, pattern="^stsendchdo_"))
    app.add_handler(CallbackQueryHandler(store_option_style_callback, pattern="^stoptstyle$"))

    store_conv = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(store_edit_title_start, pattern="^stedittitle_"),
            CallbackQueryHandler(store_add_question_start, pattern="^staddq_"),
            CallbackQueryHandler(store_edit_timer_start, pattern="^stedittimer_"),
            CallbackQueryHandler(store_edit_desc_start, pattern="^steditdesc_"),
        ],
        states={
            STORE_EDIT_TITLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, store_edit_title_receive)],
            STORE_ADD_QUESTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, store_add_question_receive)],
            STORE_EDIT_TIMER: [
                CallbackQueryHandler(store_edit_timer_pick, pattern="^sttimerpick_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, store_edit_timer_receive),
            ],
            STORE_EDIT_DESC: [MessageHandler(filters.TEXT & ~filters.COMMAND, store_edit_desc_receive)],
        },
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            CommandHandler("stop", cancel_command),
            CallbackQueryHandler(cancel_callback, pattern="^gcancel$"),
        ],
        conversation_timeout=600,
    )
    app.add_handler(store_conv)
    app.add_handler(CommandHandler("default", default_command))
    app.add_handler(CallbackQueryHandler(receive_default_diff, pattern="^setdef_diff_"))
    app.add_handler(CommandHandler("schedule", schedule_command))
    app.add_handler(CommandHandler("myschedules", myschedules_command))
    app.add_handler(CallbackQueryHandler(schedule_cancel_callback, pattern="^schedcancel_\\d+$"))
    app.add_handler(CallbackQueryHandler(schedule_quiz_page_callback, pattern="^schedqpage_\\d+$"))
    app.add_handler(CallbackQueryHandler(schedule_pick_quiz_callback, pattern="^schedq_\\d+$"))
    app.add_handler(CallbackQueryHandler(schedule_pick_interval_callback, pattern="^schedi_\\d+_\\d+$"))

    # --- Personalization: option style, daily-limit exemptions, promo text ---
    app.add_handler(CommandHandler("optionstyle", optionstyle_command))
    app.add_handler(CallbackQueryHandler(optionstyle_callback, pattern="^optstyle_"))
    app.add_handler(CommandHandler("freeuser", freeuser_command))
    app.add_handler(CommandHandler("setpromo", setpromo_command))

    # --- Quiz Sets (named folders) ---
    app.add_handler(CommandHandler("browse", browse_command))
    app.add_handler(CommandHandler("exportquiz", exportquiz_command))
    app.add_handler(CommandHandler("exportresults", exportresults_command))
    app.add_handler(CommandHandler("newset", newset_command))
    app.add_handler(CommandHandler("sets", sets_command))
    app.add_handler(CallbackQueryHandler(set_view_callback, pattern="^setview_"))
    app.add_handler(CallbackQueryHandler(set_list_back_callback, pattern="^setlist_back$"))
    app.add_handler(CallbackQueryHandler(set_delete_callback, pattern="^setdel_"))

    # --- Linked channels (for 📤 Send to Channel) ---
    setchannel_conv = ConversationHandler(
        entry_points=[CommandHandler("setchannel", setchannel_command)],
        states={SETCHANNEL_WAIT: [MessageHandler(filters.ALL & ~filters.COMMAND, setchannel_receive)]},
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            CommandHandler("stop", cancel_command),
            CallbackQueryHandler(cancel_callback, pattern="^gcancel$"),
        ],
        conversation_timeout=300,
    )
    app.add_handler(setchannel_conv)
    app.add_handler(CommandHandler("mychannels", mychannels_command))
    app.add_handler(CallbackQueryHandler(channel_remove_callback, pattern="^chrm_"))

    # --- /add_bulk — paste text OR upload a .txt file to bulk-create a quiz ---
    addbulk_conv = ConversationHandler(
        entry_points=[CommandHandler("add_bulk", addbulk_entry)],
        states={
            ADD_BULK_TITLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, addbulk_receive_title)],
            ADD_BULK_CONTENT: [MessageHandler((filters.TEXT | filters.Document.ALL) & ~filters.COMMAND, addbulk_receive_content)],
        },
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            CommandHandler("stop", cancel_command),
            CallbackQueryHandler(cancel_callback, pattern="^gcancel$"),
        ],
        conversation_timeout=900,
    )
    app.add_handler(addbulk_conv)

    # --- Leaderboards (always show user_id) ---
    app.add_handler(CommandHandler("leaderboard", leaderboard_command))
    app.add_handler(CallbackQueryHandler(leaderboard_mode_callback, pattern="^lbmode_"))

    # --- Owner: quiz blast (one question -> poll broadcast to every tracked group) ---
    quizblast_conv = ConversationHandler(
        entry_points=[CommandHandler("quizblast", quizblast_entry)],
        states={ASK_QUIZBLAST_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, quizblast_receive)]},
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            CommandHandler("stop", cancel_command),
            CallbackQueryHandler(cancel_callback, pattern="^gcancel$"),
        ],
        conversation_timeout=600,
    )
    app.add_handler(quizblast_conv)

    # --- Owner: AI Mode (auto AI quiz polls + daily leaderboard, every synced group) ---
    app.add_handler(CommandHandler("aimode", aimode_command))

    # --- Group management ---
    app.add_handler(CommandHandler("mute", mute_command))
    app.add_handler(CommandHandler("unmute", unmute_command))
    app.add_handler(CommandHandler("promote", promote_command))
    app.add_handler(CommandHandler("demote", demote_command))
    app.add_handler(CommandHandler("purge", purge_command))
    app.add_handler(CommandHandler("info", info_command))
    app.add_handler(CommandHandler("filter", filter_command))
    app.add_handler(CommandHandler("removefilter", removefilter_command))
    app.add_handler(CommandHandler("filters", filters_command))

    # --- Welcome / goodbye ---
    app.add_handler(CommandHandler("welcome", welcome_toggle_command))
    app.add_handler(CommandHandler("setwelcome", setwelcome_command))
    app.add_handler(CommandHandler("setgoodbye", setgoodbye_command))
    app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handle_new_chat_members))
    app.add_handler(MessageHandler(filters.StatusUpdate.LEFT_CHAT_MEMBER, handle_left_chat_member))

    # --- Menu navigation / verification (generic router — must stay LAST among callback handlers) ---
    app.add_handler(CallbackQueryHandler(button_callback))

    # --- Unknown/invalid command catch-all — must stay LAST among CommandHandlers in this
    # group, so it only fires for a /command nothing above already matched ---
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    # --- Group keyword filters (separate group so it runs alongside conversations) ---
    app.add_handler(MessageHandler(filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND, check_filters), group=1)

    # --- Remember every group the bot is in, so its group list survives a restart
    # even though a live-quiz session's pause/resume state is in-memory only ---
    app.add_handler(MessageHandler(filters.ChatType.GROUPS, track_group_message, block=False), group=2)

    # --- Self-backup: DM the owner a full DB dump every 6h (first one 2 min after
    # startup) so data survives even if the DB service/host wipes or is switched ---
    if app.job_queue:
        app.job_queue.run_repeating(periodic_backup_job, interval=6 * 3600, first=120, name="db_auto_backup")
        app.job_queue.run_repeating(flush_owner_notifications_job, interval=90, first=30, name="usage_alert_flush")
        app.job_queue.run_repeating(daily_quiz_check_job, interval=3600, first=60, name="daily_quiz_check")
        app.job_queue.run_repeating(scheduled_broadcast_check_job, interval=60, first=45, name="scheduled_broadcast_check")
        # AI Mode: checked every 15 min so any `interval_hours` (not just ones that
        # divide evenly into 24) still fires close to on time; the leaderboard only
        # needs to fire once/day so it reuses the same hourly cadence as daily_quiz_check.
        app.job_queue.run_repeating(ai_mode_quiz_check_job, interval=900, first=90, name="ai_mode_quiz_check")
        app.job_queue.run_repeating(ai_mode_leaderboard_job, interval=3600, first=75, name="ai_mode_leaderboard_check")
        # job_queue itself is in-memory — a restart (Render free-tier spin-down
        # included) drops every registered job, so recurring broadcasts must be
        # re-armed from the DB here or they'd silently stop after any restart.
        restored = get_all_recurring_schedules()
        for s in restored:
            register_recurring_schedule_job(app, s["id"], s["quiz_id"], s["interval_minutes"])
        logger.info(f"Restored {len(restored)} recurring broadcast schedule(s).")

    print(f"🤖 Bot Engine Started Successfully...\nStyle: {FONT_STYLE}")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


def _notify_owner_of_crash(exc: BaseException):
    """Best-effort DM to the owner with exactly what just crashed the process, so
    'why does it keep stopping' has a real, specific answer waiting in Telegram
    instead of living only in server logs the owner likely never opens. This is
    the ONLY place a crash gets reported like this — error_handler above only
    catches problems inside a single update/job (those never take the bot down
    at all); this covers the process-level kind that actually does. Swallows its
    own failures silently — it must never block the restart loop itself (e.g. if
    the crash WAS a total loss of network, this DM will also fail to send, and
    that's fine, restart still proceeds either way)."""
    if not OWNER_ID or not BOT_TOKEN:
        return
    try:
        tail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-600:]

        async def _send():
            bot = Bot(token=BOT_TOKEN)
            await bot.send_message(
                OWNER_ID,
                f"🔴 **Bot process crashed — restarting now.**\n\n"
                f"`{type(exc).__name__}: {str(exc)[:200]}`\n\n"
                f"```\n{tail}\n```\n\n"
                f"Agar ye BAAR BAAR aa raha hai (same error), ye wahi cheez hai jo "
                f"'24×7 chalu nahi rehti' ka असल reason hai — sirf ek baar aana normal hai "
                f"(host kabhi-kabhi restart karta hai), lekin repeat hona matlab kuch "
                f"actually crash ho raha hai jise fix karna padega.",
                parse_mode="Markdown",
            )
        asyncio.run(_send())
    except Exception:
        pass  # never let the crash-notifier itself break the restart loop


def run_forever():
    """Wraps main() in a restart loop with backoff. app.run_polling() already retries
    transient network errors internally (that's normal, not a crash), but if the
    process itself dies for any other reason — an exception during startup, the host
    briefly killing the process, etc. — this brings it straight back up instead of
    staying down until something notices and restarts it by hand. Every crash is
    logged with its full traceback AND DMed to the owner (see _notify_owner_of_crash)
    so the real cause is visible somewhere it'll actually get seen.

    If the bot is going offline repeatedly and this loop is why, the two other most
    likely causes (both external to this code, see the KEEP-ALIVE WEB SERVER comment
    near start_keepalive_server) are: (1) no uptime pinger is actually configured
    against the keep-alive URL yet, so a free host's inactivity sleep still kicks in
    between pings, or (2) a second copy of this exact bot is running somewhere else
    with the same BOT_TOKEN (error_handler's Conflict branch DMs the owner about that
    one specifically)."""
    backoff = 5
    while True:
        try:
            main()
            # run_polling() only returns on a clean shutdown (e.g. Ctrl+C) — treat
            # that as intentional and stop, rather than looping forever.
            break
        except Exception as e:
            logger.error("Bot crashed — restarting shortly. Full traceback:", exc_info=True)
            _notify_owner_of_crash(e)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)  # cap the wait at 5 minutes


if __name__ == "__main__":
    run_forever()
