"""
scripts/chunk_by_article.py - Chunk Vietnamese legal documents by article ("Điều").

This script implements article-level chunking:
  1. Load documents from HuggingFace (metadata + content)
  2. Split each document into chunks where each chunk = one legal article (Điều)
  3. Content before the first Điều (preamble) is kept as a separate chunk
  4. Export all chunks to a CSV file

Vietnamese legal article patterns:
  - "Điều 1."           → Article 1
  - "Điều 1:"           → Article 1
  - "Điều 12."          → Article 12
  - "Điều 100."         → Article 100
  - "Điều 1a."          → Article 1a (sub-article variant)

Usage:
  # Full dataset
  python scripts/chunk_by_article.py

  # Limit to 100 documents (for testing)
  python scripts/chunk_by_article.py --limit 100

  # Custom output file
  python scripts/chunk_by_article.py --output data/my_chunks.csv
"""

import re
import sys
import csv
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

import config
from src.data_loader import load_documents, Document
from src.chunking import Chunk


# ============================================================
# Article-Level Splitting
# ============================================================

# Regex pattern to match Vietnamese legal article headers.
# Matches lines starting with "Điều" followed by a number (and optional letter),
# then a period, colon, or dot.
# Examples: "Điều 1.", "Điều 12:", "Điều 100.", "Điều 1a."
ARTICLE_PATTERN = re.compile(
    r"(?:^|\n)"                   # start of text or newline
    r"(Điều\s+\d+[a-zA-Z]?"      # "Điều" + space + number + optional letter
    r"\s*[.:])"                   # followed by period or colon
    , re.UNICODE
)


def extract_article_title(text: str) -> str:
    """
    Extract the article header from the beginning of a chunk text.
    
    Args:
        text: The chunk text starting with "Điều X."
    
    Returns:
        The article header (e.g., "Điều 1.") or empty string if not found.
    """
    match = re.match(r"(Điều\s+\d+[a-zA-Z]?\s*[.:])", text.strip(), re.UNICODE)
    if match:
        return match.group(1).strip()
    return ""


def split_by_article(text: str) -> list[dict]:
    """
    Split a legal document text into article-level chunks.
    
    Each chunk corresponds to one article (Điều). Content before the first
    article is treated as a preamble chunk.
    
    Args:
        text: The full cleaned document text.
    
    Returns:
        List of dicts with keys:
          - 'article_title': e.g., "Điều 1." or "preamble"
          - 'text': the full text of that article/section
    """
    if not text or len(text.strip()) < config.MIN_CHUNK_LENGTH:
        return []

    # Find all article header positions
    matches = list(ARTICLE_PATTERN.finditer(text))

    if not matches:
        # No articles found — return the entire document as a single chunk
        return [{"article_title": "full_document", "text": text.strip()}]

    chunks = []

    # Preamble: text before the first article
    first_start = matches[0].start()
    # Adjust for the newline character at the start of the match
    preamble_end = first_start
    if text[first_start] == "\n":
        preamble_end = first_start
    
    preamble = text[:preamble_end].strip()
    if len(preamble) >= config.MIN_CHUNK_LENGTH:
        chunks.append({
            "article_title": "preamble",
            "text": preamble,
        })

    # Split at each article boundary
    for i, match in enumerate(matches):
        # Start of this article's text content
        # The match includes the leading newline, so we skip it
        art_start = match.start()
        if text[art_start] == "\n":
            art_start += 1

        # End of this article = start of next article (or end of text)
        if i + 1 < len(matches):
            art_end = matches[i + 1].start()
            if text[art_end] == "\n":
                pass  # keep the newline as the boundary
        else:
            art_end = len(text)

        article_text = text[art_start:art_end].strip()

        if len(article_text) < config.MIN_CHUNK_LENGTH:
            continue

        article_title = extract_article_title(article_text)
        if not article_title:
            article_title = f"article_{i}"

        chunks.append({
            "article_title": article_title,
            "text": article_text,
        })

    return chunks


def chunk_document_by_article(document: Document) -> list[Chunk]:
    """
    Split a single Document into Chunk objects, one per legal article (Điều).
    
    Args:
        document: A Document object with text and metadata.
    
    Returns:
        List of Chunk objects, each representing one article.
    """
    article_chunks = split_by_article(document.text)

    chunks = []
    for idx, art in enumerate(article_chunks):
        chunk = Chunk(
            chunk_id=f"{document.doc_id}_art_{idx}",
            doc_id=document.doc_id,
            text=art["text"],
            chunk_index=idx,
            total_chunks=len(article_chunks),
            metadata={
                **document.metadata,
                "chunk_index": idx,
                "total_chunks": len(article_chunks),
                "article_title": art["article_title"],
            },
        )
        chunks.append(chunk)

    return chunks


def chunk_documents_by_article(documents: list[Document]) -> list[Chunk]:
    """
    Chunk all documents by article (Điều).
    
    Args:
        documents: List of Document objects.
    
    Returns:
        Flat list of all Chunk objects.
    """
    from tqdm import tqdm

    all_chunks = []
    empty_count = 0

    for doc in tqdm(documents, desc="Chunking by article (Điều)"):
        doc_chunks = chunk_document_by_article(doc)
        if not doc_chunks:
            empty_count += 1
            continue
        all_chunks.extend(doc_chunks)

    # Log statistics
    if documents:
        avg_chunks = len(all_chunks) / max(1, len(documents) - empty_count)
        avg_length = sum(len(c.text) for c in all_chunks) / max(1, len(all_chunks))
        logger.info(
            f"Article chunking complete: {len(documents)} docs → {len(all_chunks)} chunks "
            f"(avg {avg_chunks:.1f} chunks/doc, avg {avg_length:.0f} chars/chunk, "
            f"{empty_count} empty docs skipped)"
        )

    return all_chunks


# ============================================================
# CSV Export
# ============================================================

def export_chunks_to_csv(chunks: list[Chunk], output_path: str) -> None:
    """
    Export chunks to a CSV file.
    
    CSV columns:
      - chunk_id: unique chunk identifier
      - doc_id: parent document ID
      - article_title: the article header (e.g., "Điều 1.") or "preamble"
      - chunk_index: position within the document
      - total_chunks: total number of chunks for this document
      - text: the chunk text content
      - title: document title
      - doc_type: document type (e.g., Nghị định)
      - date_issued: issuance date
      - authority: issuing authority
      - doc_number: document number
      - text_length: character count of the chunk text
    
    Args:
        chunks: List of Chunk objects to export.
        output_path: Path to the output CSV file.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "chunk_id",
        "doc_id",
        "article_title",
        "chunk_index",
        "total_chunks",
        "text",
        "title",
        "doc_type",
        "date_issued",
        "authority",
        "doc_number",
        "text_length",
    ]

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for chunk in chunks:
            writer.writerow({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "article_title": chunk.metadata.get("article_title", ""),
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "text": chunk.text,
                "title": chunk.metadata.get("title", ""),
                "doc_type": chunk.metadata.get("doc_type", ""),
                "date_issued": chunk.metadata.get("date_issued", ""),
                "authority": chunk.metadata.get("authority", ""),
                "doc_number": chunk.metadata.get("doc_number", ""),
                "text_length": len(chunk.text),
            })

    logger.info(f"Exported {len(chunks)} chunks to {output}")


def load_chunks_from_csv(csv_path: str, limit: int | None = None) -> list[Chunk]:
    """
    Load article-level chunks from a CSV file.

    This lets downstream steps (embedding/indexing) consume the exported CSV
    directly instead of recomputing chunks from HuggingFace each run.

    Args:
        csv_path: Path to CSV produced by export_chunks_to_csv.
        limit: Optional maximum number of rows/chunks to read.

    Returns:
        List of Chunk objects reconstructed from CSV rows.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunk CSV not found: {path}")

    def _parse_int(value: str, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    chunks: list[Chunk] = []
    skipped = 0

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):
            if limit is not None and row_idx >= limit:
                break

            text = (row.get("text") or "").strip()
            if len(text) < config.MIN_CHUNK_LENGTH:
                skipped += 1
                continue

            doc_id = str(row.get("doc_id") or "").strip()
            chunk_index = _parse_int(row.get("chunk_index"), default=0)
            total_chunks = _parse_int(row.get("total_chunks"), default=0)
            chunk_id = str(row.get("chunk_id") or f"{doc_id}_art_{chunk_index}").strip()

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                metadata={
                    "doc_id": doc_id,
                    "title": str(row.get("title") or "").strip(),
                    "doc_type": str(row.get("doc_type") or "").strip(),
                    "date_issued": str(row.get("date_issued") or "").strip(),
                    "authority": str(row.get("authority") or "").strip(),
                    "doc_number": str(row.get("doc_number") or "").strip(),
                    "article_title": str(row.get("article_title") or "").strip(),
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                },
            )
            chunks.append(chunk)

    logger.info(
        f"Loaded {len(chunks)} chunks from CSV: {path}"
        f" ({skipped} short/empty rows skipped)"
    )
    return chunks


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk Vietnamese legal documents by article (Điều) and export to CSV"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing). Default: all.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.DATA_DIR / "chunks_by_article.csv"),
        help="Output CSV file path. Default: data/chunks_by_article.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL)
    logger.add(config.LOG_DIR / "chunk_by_article.log", rotation="10 MB", level="DEBUG")

    total_start = time.time()

    # ============================================================
    # Step 1: Load Documents
    # ============================================================
    logger.info("=" * 60)
    logger.info("STEP 1/2: Loading documents from HuggingFace...")
    logger.info("=" * 60)

    step_start = time.time()
    documents = load_documents(limit=args.limit)
    step_time = time.time() - step_start
    logger.info(f"Loaded {len(documents)} documents in {step_time:.1f}s")

    if not documents:
        logger.error("No documents loaded!")
        sys.exit(1)

    # ============================================================
    # Step 2: Chunk by Article and Export
    # ============================================================
    logger.info("=" * 60)
    logger.info("STEP 2/2: Chunking by article (Điều) and exporting to CSV...")
    logger.info("=" * 60)

    step_start = time.time()
    chunks = chunk_documents_by_article(documents)
    step_time = time.time() - step_start
    logger.info(f"Created {len(chunks)} article-level chunks in {step_time:.1f}s")

    if not chunks:
        logger.error("No chunks created!")
        sys.exit(1)

    # Export to CSV
    export_chunks_to_csv(chunks, args.output)

    # ============================================================
    # Summary
    # ============================================================
    total_time = time.time() - total_start

    # Compute statistics
    text_lengths = [len(c.text) for c in chunks]
    min_len = min(text_lengths) if text_lengths else 0
    max_len = max(text_lengths) if text_lengths else 0
    avg_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0

    logger.info("=" * 60)
    logger.info("CHUNKING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"  Documents processed: {len(documents)}")
    logger.info(f"  Chunks created:      {len(chunks)}")
    logger.info(f"  Min chunk length:    {min_len} chars")
    logger.info(f"  Max chunk length:    {max_len} chars")
    logger.info(f"  Avg chunk length:    {avg_len:.0f} chars")
    logger.info(f"  Output CSV:          {args.output}")
    logger.info(f"  Total time:          {total_time:.1f}s")
    logger.info("=" * 60)

    # Show sample chunks
    logger.info("Sample chunks:")
    for chunk in chunks[:5]:
        logger.info(
            f"  {chunk.chunk_id} | {chunk.metadata.get('article_title', '')} | "
            f"{len(chunk.text)} chars | {chunk.text[:80]}..."
        )


if __name__ == "__main__":
    main()
