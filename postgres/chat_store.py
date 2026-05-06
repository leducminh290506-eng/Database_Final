"""
postgres/chat_store.py - Persistent chat history backed by PostgreSQL.

Provides CRUD for chat sessions, messages, and access logs.
All data lives in the same ``legal_db`` database alongside ``legal_chunks``.
"""

import json
from datetime import datetime

from postgres.db import get_connection


# ---------------------------------------------------------------------------
# Chat Sessions
# ---------------------------------------------------------------------------

def create_session(user_id: int, title: str = "Phiên chat mới") -> int:
    """Create a new chat session and return its ID."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_sessions (user_id, title)
                VALUES (%s, %s)
                RETURNING id
                """,
                (user_id, title),
            )
            session_id = cur.fetchone()[0]
            conn.commit()
        return session_id
    finally:
        conn.close()


def get_sessions(user_id: int, limit: int = 50) -> list[dict]:
    """Return the most recent chat sessions for a user (newest first)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "title": r[1],
                "created_at": str(r[2]),
                "updated_at": str(r[3]),
            }
            for r in rows
        ]
    finally:
        conn.close()


def update_session_title(session_id: int, title: str):
    """Update the title of a chat session."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET title = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (title, session_id),
            )
            conn.commit()
    finally:
        conn.close()


def delete_session(session_id: int):
    """Delete a chat session and all its messages (CASCADE)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
            conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Chat Messages
# ---------------------------------------------------------------------------

def add_message(
    session_id: int,
    role: str,
    content: str,
    documents: list | None = None,
    processing_time_ms: float = 0,
) -> int:
    """Append a message to a session and return the message ID."""
    conn = get_connection()
    try:
        docs_json = json.dumps(documents or [], ensure_ascii=False)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages
                    (session_id, role, content, documents, processing_time_ms)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (session_id, role, content, docs_json, processing_time_ms),
            )
            msg_id = cur.fetchone()[0]
            # Touch the session's updated_at
            cur.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (session_id,),
            )
            conn.commit()
        return msg_id
    finally:
        conn.close()


def get_messages(session_id: int) -> list[dict]:
    """Return all messages in a session, ordered chronologically."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, role, content, documents, processing_time_ms, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()
        results = []
        for r in rows:
            docs = r[3]
            if isinstance(docs, str):
                docs = json.loads(docs)
            results.append({
                "id": r[0],
                "role": r[1],
                "content": r[2],
                "documents": docs or [],
                "processing_time_ms": r[4],
                "created_at": str(r[5]),
            })
        return results
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Access Logs
# ---------------------------------------------------------------------------

def log_access(
    user_id: int,
    action: str,
    details: dict | None = None,
):
    """Write an entry to the access_logs table."""
    conn = get_connection()
    try:
        details_json = json.dumps(details or {}, ensure_ascii=False)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO access_logs (user_id, action, details)
                VALUES (%s, %s, %s)
                """,
                (user_id, action, details_json),
            )
            conn.commit()
    finally:
        conn.close()
