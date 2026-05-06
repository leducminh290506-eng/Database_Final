"""
scripts/compare_search.py - Compare Milvus vector search vs PostgreSQL keyword search.

Runs the SAME query through 3 search engines and compares results:
  1. Milvus ANN (IVF_FLAT + Cosine Similarity) — semantic search
  2. PostgreSQL ILIKE (trigram) — keyword substring matching
  3. PostgreSQL Full-Text Search (tsvector/tsquery) — keyword relevance

Usage:
    python scripts/compare_search.py "quy định về thuế thu nhập cá nhân"
    python scripts/compare_search.py "luật đất đai" --top-k 5
"""

import sys
import time
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
from loguru import logger

import config
from src.embedding import embed_query
from src.vector_store import MilvusVectorStore

# PostgreSQL connection parameters
PG_HOST = "localhost"
PG_PORT = 5433
PG_DB = "legal_db"
PG_USER = "legal_user"
PG_PASSWORD = "leducminh_2006"


def search_milvus(query: str, top_k: int = 3) -> dict:
    """Search using Milvus ANN (vector similarity)."""
    start = time.perf_counter()

    # Embed query
    query_embedding = embed_query(query)

    # Search Milvus
    store = MilvusVectorStore()
    store.load_collection()
    results = store.search(query_embedding, top_k=top_k)

    elapsed = (time.perf_counter() - start) * 1000  # ms

    return {
        "method": "Milvus ANN (IVF_FLAT + Cosine)",
        "time_ms": elapsed,
        "results": [
            {
                "chunk_id": r["chunk_id"],
                "title": r["metadata"].get("title", "")[:80],
                "doc_number": r["metadata"].get("doc_number", ""),
                "score": r["score"],
                "text_preview": r["text"][:150],
            }
            for r in results
        ],
    }


def search_pg_ilike(query: str, top_k: int = 3) -> dict:
    """Search using PostgreSQL ILIKE (keyword substring matching)."""
    start = time.perf_counter()

    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
    )

    # Split query into words, search for documents containing ALL words
    words = query.strip().split()
    conditions = " AND ".join([f"text ILIKE '%' || %s || '%'" for _ in words])
    sql = f"""
        SELECT chunk_id, title, doc_number, text,
               LENGTH(text) as text_len
        FROM legal_chunks
        WHERE {conditions}
        ORDER BY text_length ASC
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, words + [top_k])
        rows = cur.fetchall()

    conn.close()
    elapsed = (time.perf_counter() - start) * 1000

    return {
        "method": "PostgreSQL ILIKE (keyword substring)",
        "time_ms": elapsed,
        "results": [
            {
                "chunk_id": row[0],
                "title": (row[1] or "")[:80],
                "doc_number": row[2] or "",
                "score": None,  # No similarity score for ILIKE
                "text_preview": (row[3] or "")[:150],
            }
            for row in rows
        ],
    }


def search_pg_fts(query: str, top_k: int = 3) -> dict:
    """Search using PostgreSQL Full-Text Search (tsvector + tsquery)."""
    start = time.perf_counter()

    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
    )

    # Use plainto_tsquery for natural language queries
    # ts_rank provides relevance scoring
    sql = """
        SELECT chunk_id, title, doc_number, text,
               ts_rank(text_tsv, plainto_tsquery('simple', %s)) AS rank
        FROM legal_chunks
        WHERE text_tsv @@ plainto_tsquery('simple', %s)
        ORDER BY rank DESC
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, (query, query, top_k))
        rows = cur.fetchall()

    conn.close()
    elapsed = (time.perf_counter() - start) * 1000

    return {
        "method": "PostgreSQL Full-Text Search (tsvector)",
        "time_ms": elapsed,
        "results": [
            {
                "chunk_id": row[0],
                "title": (row[1] or "")[:80],
                "doc_number": row[2] or "",
                "score": float(row[4]),
                "text_preview": (row[3] or "")[:150],
            }
            for row in rows
        ],
    }


def print_results(search_result: dict, index: int):
    """Pretty-print search results for one method."""
    method = search_result["method"]
    time_ms = search_result["time_ms"]
    results = search_result["results"]

    print(f"\n{'='*70}")
    print(f"  [{index}] {method}")
    print(f"  ⏱️  Thời gian: {time_ms:.1f}ms | Kết quả: {len(results)}")
    print(f"{'='*70}")

    if not results:
        print("  ❌ Không tìm thấy kết quả nào.")
        return

    for i, r in enumerate(results, 1):
        score_str = f"{r['score']:.4f}" if r['score'] is not None else "N/A"
        print(f"\n  📄 Kết quả #{i}")
        print(f"     Tiêu đề:   {r['title']}")
        print(f"     Số ký hiệu: {r['doc_number']}")
        print(f"     Điểm:       {score_str}")
        print(f"     Nội dung:   {r['text_preview']}...")


def print_comparison_summary(all_results: list[dict]):
    """Print a side-by-side comparison summary."""
    print(f"\n{'='*70}")
    print("  📊 TÓM TẮT SO SÁNH (COMPARISON SUMMARY)")
    print(f"{'='*70}")

    # Speed comparison
    print("\n  ⏱️  TỐC ĐỘ TRUY VẤN:")
    for r in all_results:
        bar_len = int(r["time_ms"] / 10)
        bar = "█" * min(bar_len, 50)
        print(f"     {r['method'][:45]:<45} {r['time_ms']:>8.1f}ms {bar}")

    # Result count
    print("\n  📊 SỐ KẾT QUẢ:")
    for r in all_results:
        print(f"     {r['method'][:45]:<45} {len(r['results']):>3} kết quả")

    # Overlap analysis
    milvus_chunks = {r["chunk_id"] for r in all_results[0]["results"]} if all_results else set()
    for other in all_results[1:]:
        other_chunks = {r["chunk_id"] for r in other["results"]}
        overlap = milvus_chunks & other_chunks
        print(f"\n  🔀 Overlap Milvus ↔ {other['method'][:30]}:")
        print(f"     Chung: {len(overlap)} chunks | Chỉ Milvus: {len(milvus_chunks - other_chunks)} | "
              f"Chỉ PG: {len(other_chunks - milvus_chunks)}")

    # Key insight
    print(f"\n  💡 NHẬN XÉT CHÍNH:")
    print(f"     • Milvus (vector search) tìm theo NGỮ NGHĨA — hiểu được từ đồng nghĩa")
    print(f"     • PostgreSQL ILIKE tìm theo TỪ KHÓA CHÍNH XÁC — chỉ match chuỗi con")
    print(f"     • PostgreSQL FTS tìm theo TỪ KHÓA + xếp hạng — tốt hơn ILIKE nhưng")
    print(f"       không hiểu ngữ nghĩa (\"đất đai\" ≠ \"bất động sản\")")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Compare Milvus vs PostgreSQL search")
    parser.add_argument("query", type=str, help="Search query in Vietnamese")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results (default: 3)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    query = args.query
    top_k = args.top_k

    print(f"\n{'='*70}")
    print(f"  🔍 SO SÁNH TÌM KIẾM: Milvus vs PostgreSQL")
    print(f"  📝 Câu hỏi: \"{query}\"")
    print(f"  📊 Top-K: {top_k}")
    print(f"{'='*70}")

    all_results = []

    # 1. Milvus ANN search
    try:
        milvus_result = search_milvus(query, top_k)
        all_results.append(milvus_result)
        print_results(milvus_result, 1)
    except Exception as e:
        print(f"\n  ❌ Milvus search failed: {e}")

    # 2. PostgreSQL ILIKE
    try:
        ilike_result = search_pg_ilike(query, top_k)
        all_results.append(ilike_result)
        print_results(ilike_result, 2)
    except Exception as e:
        print(f"\n  ❌ PostgreSQL ILIKE search failed: {e}")

    # 3. PostgreSQL FTS
    try:
        fts_result = search_pg_fts(query, top_k)
        all_results.append(fts_result)
        print_results(fts_result, 3)
    except Exception as e:
        print(f"\n  ❌ PostgreSQL FTS search failed: {e}")

    # Summary
    if len(all_results) >= 2:
        print_comparison_summary(all_results)


if __name__ == "__main__":
    main()
