"""
postgres/search.py - PostgreSQL search functions for Milvus vs PG comparison.

Provides ILIKE (keyword substring) and Full-Text Search (tsvector/tsquery)
queries against the ``legal_chunks`` table.  These are used by the comparison
UI tab and the ``/compare`` API endpoint.
"""

import time

from postgres.db import get_connection


def search_pg_ilike(query: str, top_k: int = 3) -> dict:
    """
    Search using PostgreSQL ILIKE (keyword substring matching).

    Splits the query into words and requires ALL words to appear in the text.
    """
    start = time.perf_counter()
    conn = get_connection()

    words = query.strip().split()
    if not words:
        return {"method": "PostgreSQL ILIKE", "time_ms": 0, "results": []}

    conditions = " AND ".join([f"text ILIKE '%' || %s || '%'" for _ in words])
    sql = f"""
        SELECT chunk_id, title, doc_number, text, text_length
        FROM legal_chunks
        WHERE {conditions}
        ORDER BY text_length ASC
        LIMIT %s
    """

    try:
        with conn.cursor() as cur:
            cur.execute(sql, words + [top_k])
            rows = cur.fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "method": "PostgreSQL ILIKE (keyword substring)",
        "time_ms": round(elapsed, 2),
        "results": [
            {
                "chunk_id": row[0],
                "title": (row[1] or "")[:120],
                "doc_number": row[2] or "",
                "score": None,
                "text_preview": (row[3] or "")[:200],
            }
            for row in rows
        ],
    }


def search_pg_fts(query: str, top_k: int = 3) -> dict:
    """
    Search using PostgreSQL Full-Text Search (tsvector + tsquery).

    Uses ``plainto_tsquery('simple', ...)`` for natural-language queries and
    ``ts_rank`` for relevance scoring.
    """
    start = time.perf_counter()
    conn = get_connection()

    sql = """
        SELECT chunk_id, title, doc_number, text,
               ts_rank(text_tsv, plainto_tsquery('simple', %s)) AS rank
        FROM legal_chunks
        WHERE text_tsv @@ plainto_tsquery('simple', %s)
        ORDER BY rank DESC
        LIMIT %s
    """

    try:
        with conn.cursor() as cur:
            cur.execute(sql, (query, query, top_k))
            rows = cur.fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "method": "PostgreSQL Full-Text Search (tsvector)",
        "time_ms": round(elapsed, 2),
        "results": [
            {
                "chunk_id": row[0],
                "title": (row[1] or "")[:120],
                "doc_number": row[2] or "",
                "score": round(float(row[4]), 6),
                "text_preview": (row[3] or "")[:200],
            }
            for row in rows
        ],
    }
