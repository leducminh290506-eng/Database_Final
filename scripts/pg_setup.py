"""
scripts/pg_setup.py - Load legal document chunks into PostgreSQL for comparison.

This script:
  1. Connects to PostgreSQL (Docker container)
  2. Creates the legal_chunks table
  3. Loads chunks from data/chunks_by_article.csv
  4. Creates GIN indexes for Full-Text Search (FTS)

Usage:
    python scripts/pg_setup.py
    python scripts/pg_setup.py --chunks-csv data/my_chunks.csv
    python scripts/pg_setup.py --limit 100
"""

import sys
import csv
import time
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
from psycopg2.extras import execute_batch
from loguru import logger

import config

# PostgreSQL connection parameters (match docker-compose.yml)
PG_HOST = "localhost"
PG_PORT = 5433
PG_DB = "legal_db"
PG_USER = "legal_user"
PG_PASSWORD = "leducminh_2006"


def get_connection():
    """Connect to PostgreSQL."""
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
    )


def create_table(conn):
    """Create the legal_chunks table with FTS support."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS legal_chunks;")
        cur.execute("""
            CREATE TABLE legal_chunks (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(200) NOT NULL,
                doc_id VARCHAR(100) NOT NULL,
                text TEXT NOT NULL,
                title VARCHAR(2000) DEFAULT '',
                doc_type VARCHAR(200) DEFAULT '',
                date_issued VARCHAR(50) DEFAULT '',
                authority VARCHAR(500) DEFAULT '',
                doc_number VARCHAR(200) DEFAULT '',
                article_title VARCHAR(200) DEFAULT '',
                text_length INTEGER DEFAULT 0,
                -- Pre-computed tsvector column for Full-Text Search
                text_tsv TSVECTOR
            );
        """)
        conn.commit()
    logger.info("Table 'legal_chunks' created")


def create_indexes(conn):
    """Create indexes for efficient keyword and full-text search."""
    with conn.cursor() as cur:
        # GIN index on tsvector for Full-Text Search
        cur.execute("""
            CREATE INDEX idx_legal_chunks_tsv
            ON legal_chunks USING GIN (text_tsv);
        """)
        # B-tree index on doc_id for exact lookups
        cur.execute("""
            CREATE INDEX idx_legal_chunks_doc_id
            ON legal_chunks (doc_id);
        """)
        # Trigram index for ILIKE searches (requires pg_trgm extension)
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute("""
            CREATE INDEX idx_legal_chunks_text_trgm
            ON legal_chunks USING GIN (text gin_trgm_ops);
        """)
        conn.commit()
    logger.info("Indexes created (GIN tsvector, B-tree doc_id, GIN trigram)")


def load_chunks_from_csv(csv_path: str, limit: int | None = None) -> list[dict]:
    """Load chunks from CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    chunks = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            text = (row.get("text") or "").strip()
            if len(text) < 20:
                continue
            chunks.append({
                "chunk_id": (row.get("chunk_id") or "").strip(),
                "doc_id": (row.get("doc_id") or "").strip(),
                "text": text,
                "title": (row.get("title") or "").strip()[:2000],
                "doc_type": (row.get("doc_type") or "").strip()[:200],
                "date_issued": (row.get("date_issued") or "").strip()[:50],
                "authority": (row.get("authority") or "").strip()[:500],
                "doc_number": (row.get("doc_number") or "").strip()[:200],
                "article_title": (row.get("article_title") or "").strip()[:200],
                "text_length": len(text),
            })
    logger.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def insert_chunks(conn, chunks: list[dict]):
    """Insert chunks into PostgreSQL and compute tsvectors."""
    insert_sql = """
        INSERT INTO legal_chunks
            (chunk_id, doc_id, text, title, doc_type, date_issued,
             authority, doc_number, article_title, text_length, text_tsv)
        VALUES
            (%(chunk_id)s, %(doc_id)s, %(text)s, %(title)s, %(doc_type)s,
             %(date_issued)s, %(authority)s, %(doc_number)s, %(article_title)s,
             %(text_length)s, to_tsvector('simple', %(text)s))
    """
    with conn.cursor() as cur:
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            execute_batch(cur, insert_sql, batch, page_size=batch_size)
            logger.info(f"  Inserted {min(i + batch_size, len(chunks))}/{len(chunks)}")
        conn.commit()
    logger.info(f"All {len(chunks)} chunks inserted into PostgreSQL")


def main():
    parser = argparse.ArgumentParser(description="Setup PostgreSQL with legal chunks")
    parser.add_argument(
        "--chunks-csv", type=str,
        default=str(config.DATA_DIR / "chunks_by_article.csv"),
        help="Path to chunk CSV file",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for testing")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    total_start = time.time()

    logger.info("=" * 60)
    logger.info("PostgreSQL Setup for Legal Chunks")
    logger.info("=" * 60)

    # Step 1: Connect
    logger.info("Connecting to PostgreSQL...")
    conn = get_connection()
    logger.info("Connected!")

    # Step 2: Create table
    create_table(conn)

    # Step 3: Load and insert chunks
    chunks = load_chunks_from_csv(args.chunks_csv, limit=args.limit)
    if not chunks:
        logger.error("No chunks to insert!")
        sys.exit(1)
    insert_chunks(conn, chunks)

    # Step 4: Create indexes
    create_indexes(conn)

    # Step 5: Verify
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM legal_chunks;")
        count = cur.fetchone()[0]

    conn.close()
    total_time = time.time() - total_start

    logger.info("=" * 60)
    logger.info("POSTGRESQL SETUP COMPLETE!")
    logger.info(f"  Rows inserted: {count}")
    logger.info(f"  Total time:    {total_time:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
