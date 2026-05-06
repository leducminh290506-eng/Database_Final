"""
scripts/build_index.py - Build the Milvus index from legal text chunks.

CSV-first pipeline:
    1. Load article-level chunks from CSV (data/chunks_by_article.csv)
    2. Embed all chunk texts using SentenceTransformers
    3. Build Milvus collection with IVF_FLAT index + cosine similarity

Optional refresh:
    - If CSV is missing (or --rebuild-csv is provided), regenerate it from
      HuggingFace data via article-level chunking, then load chunks from CSV.

Usage:
    # Build index from existing CSV
    python scripts/build_index.py

    # Quick test with first 100 CSV rows/chunks
    python scripts/build_index.py --limit 100

    # Use a custom CSV file
    python scripts/build_index.py --chunks-csv data/my_chunks.csv

    # Force regenerate CSV from HuggingFace before indexing
    python scripts/build_index.py --rebuild-csv
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from loguru import logger

import config
from src.data_loader import load_documents
from src.embedding import embed_texts
from chunk_by_article import (
    chunk_documents_by_article,
    export_chunks_to_csv,
    load_chunks_from_csv,
)
from src.vector_store import MilvusVectorStore


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build Milvus index for Vietnamese Legal Assistant"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Limit records for testing. When using existing CSV, limits CSV rows/chunks loaded. "
            "When regenerating CSV, limits source documents loaded from HuggingFace."
        ),
    )
    parser.add_argument(
        "--chunks-csv",
        type=str,
        default=str(config.DATA_DIR / "chunks_by_article.csv"),
        help="Path to article-level chunk CSV. Default: data/chunks_by_article.csv.",
    )
    parser.add_argument(
        "--rebuild-csv",
        action="store_true",
        help="Regenerate article chunk CSV from HuggingFace before indexing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.EMBEDDING_BATCH_SIZE,
        help=f"Embedding batch size. Default: {config.EMBEDDING_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default=config.MILVUS_URI,
        help=f"Milvus server URI. Default: {config.MILVUS_URI}.",
    )
    return parser.parse_args()


def main():
    """Run the full index-building pipeline."""
    args = parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL)
    logger.add(config.LOG_DIR / "build_index.log", rotation="10 MB", level="DEBUG")
    
    total_start = time.time()
    chunks = []
    documents_processed = 0
    
    # ============================================================
    # Step 1: Prepare Chunks
    # ============================================================
    logger.info("=" * 60)
    logger.info("STEP 1/3: Preparing chunks for indexing...")
    logger.info("=" * 60)
    
    step_start = time.time()

    csv_path = Path(args.chunks_csv)
    csv_regenerated = False

    if args.rebuild_csv or not csv_path.exists():
        if args.rebuild_csv:
            logger.info("Regenerating article CSV from HuggingFace (--rebuild-csv enabled)...")
        else:
            logger.warning(f"Chunk CSV not found at {csv_path}. Generating from HuggingFace...")

        documents = load_documents(limit=args.limit)
        logger.info(f"Loaded {len(documents)} documents for CSV generation")

        if not documents:
            logger.error("No documents loaded! Check dataset availability.")
            sys.exit(1)

        generated_chunks = chunk_documents_by_article(documents)
        if not generated_chunks:
            logger.error("No article chunks created while regenerating CSV.")
            sys.exit(1)

        export_chunks_to_csv(generated_chunks, str(csv_path))
        csv_regenerated = True
    else:
        logger.info(f"Using existing chunk CSV: {csv_path}")

    # Always load chunks from CSV before embedding/indexing.
    # If CSV was regenerated in this run, we keep full CSV content (no row-limit)
    # to preserve complete per-document chunk sets.
    csv_limit = None if csv_regenerated else args.limit
    chunks = load_chunks_from_csv(str(csv_path), limit=csv_limit)
    documents_processed = len({chunk.doc_id for chunk in chunks})
    
    step_time = time.time() - step_start
    
    logger.info(
        f"Prepared {len(chunks)} chunks from {documents_processed} documents "
        f"in {step_time:.1f}s"
    )
    
    if not chunks:
        logger.error("No chunks available for embedding.")
        sys.exit(1)
    
    # ============================================================
    # Step 2: Embed Chunks
    # ============================================================
    logger.info("=" * 60)
    logger.info("STEP 2/3: Embedding chunks with SentenceTransformers...")
    logger.info(f"  Model: {config.EMBEDDING_MODEL}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Total chunks: {len(chunks)}")
    logger.info("=" * 60)
    
    step_start = time.time()
    
    # Extract texts for embedding
    chunk_texts = [chunk.text for chunk in chunks]
    
    # Embed in batches
    embeddings = embed_texts(
        chunk_texts,
        batch_size=args.batch_size,
        show_progress=True,
        normalize=True,  # L2-normalize for better cosine similarity
    )
    
    step_time = time.time() - step_start
    throughput = len(chunks) / max(0.1, step_time)
    
    logger.info(
        f"Embedded {len(chunks)} chunks in {step_time:.1f}s "
        f"({throughput:.0f} chunks/sec)"
    )
    
    # ============================================================
    # Step 3: Build Milvus Index
    # ============================================================
    logger.info("=" * 60)
    logger.info("STEP 3/3: Building Milvus index...")
    logger.info(f"  Index type: {config.MILVUS_INDEX_TYPE} (ANN)")
    logger.info(f"  Metric: {config.MILVUS_METRIC_TYPE} (Cosine Similarity)")
    logger.info(f"  Dimension: {config.EMBEDDING_DIMENSION}")
    logger.info(f"  nlist: {config.MILVUS_NLIST} clusters")
    logger.info(f"  Milvus URI: {args.milvus_uri}")
    logger.info("=" * 60)
    
    step_start = time.time()
    
    # Create Milvus vector store and build index
    store = MilvusVectorStore(uri=args.milvus_uri)
    store.build(embeddings, chunks)
    
    step_time = time.time() - step_start
    logger.info(f"Milvus index built in {step_time:.1f}s")
    
    # ============================================================
    # Summary
    # ============================================================
    total_time = time.time() - total_start
    
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"  Documents processed: {documents_processed}")
    logger.info(f"  Chunks created:      {len(chunks)}")
    logger.info(f"  Index size:          {store.size} vectors")
    logger.info(f"  Embedding dimension: {config.EMBEDDING_DIMENSION}")
    logger.info(f"  Index type:          {config.MILVUS_INDEX_TYPE} ({config.MILVUS_METRIC_TYPE})")
    logger.info(f"  Milvus URI:          {args.milvus_uri}")
    logger.info(f"  Total time:          {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("=" * 60)
    logger.info(
        "Next steps:\n"
        "  1. Start the API:  uvicorn api:app --reload\n"
        "  2. Start the UI:   streamlit run ui.py\n"
        "  3. Open browser:   http://localhost:8501"
    )


if __name__ == "__main__":
    main()
