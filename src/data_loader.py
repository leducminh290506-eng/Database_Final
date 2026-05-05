"""
data_loader.py - Load and preprocess Vietnamese legal documents from HuggingFace.

This module handles:
  1. Loading the 'metadata' and 'content' configs from the HuggingFace dataset
  2. Merging them on document ID
  3. Stripping HTML to plain text
  4. Cleaning and normalizing text (removing noise, control characters, etc.)
  5. Returning structured Document objects ready for chunking

Design Decision:
  We load data in streaming mode when possible to handle the ~153k document
  dataset without exhausting RAM. The HTML-to-text conversion uses BeautifulSoup
  with the fast 'lxml' parser for production-grade HTML handling.
"""

import re
import asyncio
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config


# ============================================================
# Data Models
# ============================================================

@dataclass
class Document:
    """Represents a single cleaned legal document with metadata."""
    doc_id: str
    title: str
    text: str
    doc_type: str = ""           # loai_van_ban (e.g., Nghị định, Thông tư)
    date_issued: str = ""        # ngay_ban_hanh
    authority: str = ""          # co_quan_ban_hanh
    doc_number: str = ""         # so_ky_hieu
    legal_field: str = ""        # linh_vuc
    status: str = ""             # tinh_trang_hieu_luc
    metadata: dict = field(default_factory=dict)


# ============================================================
# Text Cleaning Utilities
# ============================================================

def strip_html(html_content: str) -> str:
    """
    Convert HTML content to clean plain text.
    
    Uses BeautifulSoup with 'lxml' parser for robust HTML handling.
    Preserves paragraph structure by inserting newlines at block boundaries.
    
    Args:
        html_content: Raw HTML string from the dataset.
    
    Returns:
        Clean plain-text string with paragraph breaks preserved.
    """
    if not html_content or not isinstance(html_content, str):
        return ""
    
    try:
        soup = BeautifulSoup(html_content, "lxml")
        
        # Remove script and style elements
        for element in soup(["script", "style", "head", "meta", "link"]):
            element.decompose()
        
        # Insert newlines at block-level element boundaries
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                                   "li", "tr", "blockquote", "section", "article"]):
            tag.insert_before("\n")
            tag.insert_after("\n")
        
        text = soup.get_text()
        return text
    except Exception as e:
        logger.warning(f"HTML parsing failed: {e}")
        return html_content  # Return raw content as fallback


def clean_text(text: str) -> str:
    """
    Normalize and clean extracted text.
    
    Operations:
      - Remove NULL bytes and control characters
      - Normalize Unicode whitespace
      - Collapse multiple blank lines
      - Strip leading/trailing whitespace
    
    Args:
        text: Raw extracted text.
    
    Returns:
        Cleaned and normalized text.
    """
    if not text:
        return ""
    
    # Remove NULL bytes and control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    
    # Normalize various Unicode whitespace to regular space
    text = re.sub(r"[\u00a0\u2000-\u200b\u2028\u2029\u3000\ufeff]", " ", text)
    
    # Collapse multiple spaces on the same line
    text = re.sub(r"[^\S\n]+", " ", text)
    
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Strip each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    return text.strip()


# ============================================================
# Dataset Loading
# ============================================================

def load_metadata(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load the metadata config from HuggingFace.
    
    Args:
        limit: Optional maximum number of documents to load (for testing).
    
    Returns:
        DataFrame with metadata columns.
    """
    logger.info("Loading metadata from HuggingFace...")
    ds = load_dataset(
        config.DATASET_NAME,
        config.DATASET_METADATA_CONFIG,
        split=config.DATASET_SPLIT,
    )
    df = ds.to_pandas()
    
    if limit:
        df = df.head(limit)
    
    logger.info(f"Loaded {len(df)} metadata records")
    return df


def load_content(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load the content config (HTML full text) from HuggingFace.
    
    Note: We bypass the `datasets` library here because the content_html
    column uses Arrow's `large_string` type (total data > 2 GB). The
    `datasets` library tries to cast large_string → string and fails with
    ArrowInvalid. Reading with pd.read_parquet / pyarrow keeps the native
    type and converts to Python strings without issues.
    
    Args:
        limit: Optional maximum number of documents to load (for testing).
    
    Returns:
        DataFrame with id and content_html columns.
    """
    from huggingface_hub import hf_hub_download
    
    logger.info("Loading content from HuggingFace (direct parquet)...")
    
    # Download the parquet file (cached after first download)
    parquet_path = hf_hub_download(
        repo_id=config.DATASET_NAME,
        filename="data/content.parquet",
        repo_type="dataset",
    )
    
    # Read with pandas/pyarrow — handles large_string natively
    df = pd.read_parquet(parquet_path)
    
    if limit:
        df = df.head(limit)
    
    logger.info(f"Loaded {len(df)} content records")
    return df


def merge_and_process(
    meta_df: pd.DataFrame,
    content_df: pd.DataFrame,
    limit: Optional[int] = None,
) -> list[Document]:
    """
    Merge metadata and content DataFrames, then process into Document objects.
    
    Steps:
      1. Inner-join on 'id' to keep only docs that have both metadata and content
      2. Strip HTML from content_html → plain text
      3. Clean and normalize text
      4. Filter out empty/too-short/too-long documents
      5. Create Document objects with all metadata
    
    Args:
        meta_df: Metadata DataFrame.
        content_df: Content DataFrame.
        limit: Optional limit on output document count.
    
    Returns:
        List of Document objects.
    """
    logger.info("Merging metadata and content...")
    
    # Ensure 'id' columns have the same type (metadata=int64, content=str)
    meta_df["id"] = meta_df["id"].astype(str)
    content_df["id"] = content_df["id"].astype(str)
    
    # Inner join: only documents that have both metadata and HTML content
    merged = meta_df.merge(content_df, on="id", how="inner")
    logger.info(f"Merged dataset: {len(merged)} documents")
    
    if limit:
        merged = merged.head(limit)
    
    documents = []
    skipped = 0
    
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Processing documents"):
        # Extract and clean text from HTML
        raw_text = strip_html(row.get("content_html", ""))
        cleaned = clean_text(raw_text)
        
        # Skip empty, too-short, or excessively long documents
        if len(cleaned) < config.MIN_CHUNK_LENGTH:
            skipped += 1
            continue
        if len(cleaned) > config.MAX_DOC_LENGTH:
            cleaned = cleaned[:config.MAX_DOC_LENGTH]
            logger.debug(f"Truncated document {row['id']} to {config.MAX_DOC_LENGTH} chars")
        
        doc = Document(
            doc_id=str(row["id"]),
            title=str(row.get("title", "")).strip(),
            text=cleaned,
            doc_type=str(row.get("loai_van_ban", "")).strip(),
            date_issued=str(row.get("ngay_ban_hanh", "")).strip(),
            authority=str(row.get("co_quan_ban_hanh", "")).strip(),
            doc_number=str(row.get("so_ky_hieu", "")).strip(),
            legal_field=str(row.get("linh_vuc", "")).strip(),
            status=str(row.get("tinh_trang_hieu_luc", "")).strip(),
            metadata={
                "doc_id": str(row["id"]),
                "title": str(row.get("title", "")),
                "doc_type": str(row.get("loai_van_ban", "")),
                "date_issued": str(row.get("ngay_ban_hanh", "")),
                "authority": str(row.get("co_quan_ban_hanh", "")),
                "doc_number": str(row.get("so_ky_hieu", "")),
                "field": str(row.get("linh_vuc", "")),
                "status": str(row.get("tinh_trang_hieu_luc", "")),
            },
        )
        documents.append(doc)
    
    logger.info(f"Processed {len(documents)} documents ({skipped} skipped)")
    return documents


def load_documents(limit: Optional[int] = None) -> list[Document]:
    """
    High-level function: load, merge, and process all documents.
    
    This is the main entry point for the data pipeline.
    
    Args:
        limit: Optional limit on number of documents (for testing/development).
    
    Returns:
        List of cleaned Document objects.
    """
    meta_df = load_metadata(limit=limit)
    content_df = load_content(limit=limit)
    documents = merge_and_process(meta_df, content_df, limit=limit)
    return documents


async def load_documents_async(limit: Optional[int] = None) -> list[Document]:
    """
    Async wrapper for load_documents to support non-blocking execution
    in async contexts (e.g., FastAPI startup).
    
    Args:
        limit: Optional limit on number of documents.
    
    Returns:
        List of cleaned Document objects.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, load_documents, limit)


# ============================================================
# CLI Entry Point (for standalone testing)
# ============================================================

if __name__ == "__main__":
    import sys
    
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    docs = load_documents(limit=limit)
    
    for doc in docs[:3]:
        print(f"\n{'='*60}")
        print(f"ID:     {doc.doc_id}")
        print(f"Title:  {doc.title[:100]}")
        print(f"Type:   {doc.doc_type}")
        print(f"Date:   {doc.date_issued}")
        print(f"Auth:   {doc.authority}")
        print(f"Text:   {doc.text[:200]}...")
        print(f"Length: {len(doc.text)} chars")
