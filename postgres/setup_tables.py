"""
postgres/setup_tables.py - Create authentication, chat history, and access log tables.

Run once to set up the schema.  Safe to re-run (uses IF NOT EXISTS).

Usage:
    python postgres/setup_tables.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from postgres.db import get_connection
from loguru import logger


SQL_STATEMENTS = [
    # ── Users ──────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        display_name VARCHAR(100) DEFAULT '',
        email VARCHAR(200) DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    );
    """,

    # ── Chat Sessions ──────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        title VARCHAR(500) DEFAULT 'Phiên chat mới',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,

    # ── Chat Messages ──────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id SERIAL PRIMARY KEY,
        session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
        role VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        documents JSONB DEFAULT '[]',
        processing_time_ms FLOAT DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,

    # ── Access Logs ────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS access_logs (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        action VARCHAR(50) NOT NULL,
        details JSONB DEFAULT '{}',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,

    # ── Indexes ────────────────────────────────────────────────
    "CREATE INDEX IF NOT EXISTS idx_sessions_user   ON chat_sessions(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_logs_user        ON access_logs(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_logs_created     ON access_logs(created_at);",
]


def setup():
    """Execute all DDL statements."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for stmt in SQL_STATEMENTS:
                cur.execute(stmt)
        conn.commit()
        logger.info("All auth / chat / log tables created successfully.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create tables: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    setup()
    logger.info("Done.")
