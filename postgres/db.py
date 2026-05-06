"""
postgres/db.py - Shared PostgreSQL connection helper.

Centralizes database connection using config values so all postgres/
modules use the same parameters.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

import psycopg2


def get_connection():
    """Return a new psycopg2 connection to the legal_db database."""
    return psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        dbname=config.PG_DB,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
    )
