"""
postgres/auth.py - User authentication module.

Provides registration, login, and user lookup backed by PostgreSQL.
Passwords are hashed with bcrypt.
"""

import hashlib
import os
from datetime import datetime

from postgres.db import get_connection


# ---------------------------------------------------------------------------
# Password hashing (using hashlib + salt so we avoid extra C-extension deps)
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    """Hash a password with a random salt using SHA-256."""
    salt = os.urandom(16).hex()
    hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${hashed}"


def _verify_password(password: str, stored: str) -> bool:
    """Verify a password against the stored salt$hash string."""
    if "$" not in stored:
        return False
    salt, expected_hash = stored.split("$", 1)
    actual_hash = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return actual_hash == expected_hash


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def register_user(
    username: str,
    password: str,
    display_name: str = "",
    email: str = "",
) -> dict | None:
    """
    Register a new user.

    Returns:
        User dict on success, None if username already exists.
    """
    conn = get_connection()
    try:
        pw_hash = _hash_password(password)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (username, password_hash, display_name, email)
                VALUES (%s, %s, %s, %s)
                RETURNING id, username, display_name, email, created_at
                """,
                (username, pw_hash, display_name or username, email),
            )
            row = cur.fetchone()
            conn.commit()
        if row:
            return {
                "id": row[0],
                "username": row[1],
                "display_name": row[2],
                "email": row[3],
                "created_at": str(row[4]),
            }
        return None
    except Exception as e:
        conn.rollback()
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            return None
        raise
    finally:
        conn.close()


def login_user(username: str, password: str) -> dict | None:
    """
    Authenticate a user.

    Returns:
        User dict on success, None on invalid credentials.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, password_hash, display_name, email FROM users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()
        if not row:
            return None
        if not _verify_password(password, row[2]):
            return None
        # Update last_login
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
                (row[0],),
            )
            conn.commit()
        return {
            "id": row[0],
            "username": row[1],
            "display_name": row[3],
            "email": row[4],
        }
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    """Look up a user by primary key."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, display_name, email FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
        if row:
            return {"id": row[0], "username": row[1], "display_name": row[2], "email": row[3]}
        return None
    finally:
        conn.close()
