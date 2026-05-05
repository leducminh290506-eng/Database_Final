"""
chunking.py - Split legal documents into semantic chunks for embedding.

This module implements a sentence-aware chunking strategy that:
  1. Splits documents at natural boundaries (paragraph breaks, sentence endings)
  2. Maintains overlap between consecutive chunks for context continuity
  3. Preserves parent document metadata in each chunk
  4. Filters out chunks that are too short to be meaningful

Design Decisions:
  - We use character-based chunking (~1024 chars ≈ 256 tokens for Vietnamese)
    rather than token-based to avoid the overhead of tokenization during chunking.
  - Overlap of 128 chars ensures important information at chunk boundaries isn't lost.
  - Sentence-aware splitting prevents cutting mid-sentence, which would degrade
    embedding quality and retrieval relevance.

Trade-offs:
  - Smaller chunks (512 chars) → more precise retrieval but less context per chunk
  - Larger chunks (2048 chars) → more context but noisier retrieval
  - We chose 1024 chars as a balance between precision and context richness.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config
from src.data_loader import Document


# ============================================================
# Data Models
# ============================================================

@dataclass
class Chunk:
    """Represents a single text chunk derived from a legal document."""
    chunk_id: str           # Unique ID: "{doc_id}_chunk_{index}"
    doc_id: str             # Parent document ID
    text: str               # Chunk text content
    chunk_index: int        # Position within the parent document
    total_chunks: int = 0   # Total chunks for the parent document
    metadata: dict = field(default_factory=dict)  # Inherited metadata from parent


# ============================================================
# Sentence-Aware Text Splitter
# ============================================================

def find_split_point(text: str, target_pos: int) -> int:
    """
    Find the nearest natural split point (sentence/paragraph boundary)
    near the target position.
    
    We search backward from target_pos for a sentence-ending pattern.
    If none is found within a reasonable range, we fall back to
    the nearest space to avoid splitting mid-word.
    
    Args:
        text: The full text to split.
        target_pos: The ideal position to split at.
    
    Returns:
        The actual split position.
    """
    if target_pos >= len(text):
        return len(text)
    
    # Search backward up to 200 characters for a natural boundary
    search_start = max(0, target_pos - 200)
    search_region = text[search_start:target_pos]
    
    # Priority 1: Paragraph break (double newline)
    last_para = search_region.rfind("\n\n")
    if last_para != -1:
        return search_start + last_para + 2  # After the double newline
    
    # Priority 2: Sentence ending (period/question mark/exclamation + space or newline)
    sentence_end = None
    for match in re.finditer(r"[.!?;:]\s", search_region):
        sentence_end = match.end()
    if sentence_end is not None:
        return search_start + sentence_end
    
    # Priority 3: Single newline
    last_newline = search_region.rfind("\n")
    if last_newline != -1:
        return search_start + last_newline + 1
    
    # Priority 4: Nearest space (avoid splitting mid-word)
    last_space = search_region.rfind(" ")
    if last_space != -1:
        return search_start + last_space + 1
    
    # Fallback: Hard split at target position
    return target_pos


def split_text_into_chunks(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks at natural boundaries.
    
    Algorithm:
      1. Start at position 0
      2. Move forward by chunk_size characters
      3. Find the nearest natural boundary (sentence/paragraph end)
      4. Create chunk from current position to boundary
      5. Move start position forward by (chunk_size - overlap) to create overlap
      6. Repeat until end of text
    
    Args:
        text: The full document text to chunk.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
    
    Returns:
        List of text chunks.
    """
    if not text or len(text.strip()) < config.MIN_CHUNK_LENGTH:
        return []
    
    # If the text fits in a single chunk, return as-is
    if len(text) <= chunk_size:
        return [text.strip()]
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        # Calculate the end position for this chunk
        end = min(start + chunk_size, text_len)
        
        if end < text_len:
            # Find a natural split point near the end
            end = find_split_point(text, end)
        
        # Extract the chunk
        chunk_text = text[start:end].strip()
        
        if len(chunk_text) >= config.MIN_CHUNK_LENGTH:
            chunks.append(chunk_text)
        
        # Move start forward, accounting for overlap
        if end >= text_len:
            break
        
        # Advance by (chunk_size - overlap) to create overlap
        start = max(start + 1, end - chunk_overlap)
    
    return chunks


# ============================================================
# Document Chunking
# ============================================================

def chunk_document(document: Document) -> list[Chunk]:
    """
    Split a single Document into Chunk objects.
    
    Each chunk inherits the parent document's metadata and gets a unique
    chunk_id formatted as "{doc_id}_chunk_{index}".
    
    Args:
        document: A Document object with text and metadata.
    
    Returns:
        List of Chunk objects derived from the document.
    """
    text_chunks = split_text_into_chunks(document.text)
    
    chunks = []
    for idx, chunk_text in enumerate(text_chunks):
        chunk = Chunk(
            chunk_id=f"{document.doc_id}_chunk_{idx}",
            doc_id=document.doc_id,
            text=chunk_text,
            chunk_index=idx,
            total_chunks=len(text_chunks),
            metadata={
                **document.metadata,
                "chunk_index": idx,
                "total_chunks": len(text_chunks),
            },
        )
        chunks.append(chunk)
    
    return chunks


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """
    Chunk all documents in a list.
    
    This is the main entry point for the chunking pipeline.
    Includes progress tracking and summary statistics.
    
    Args:
        documents: List of Document objects to chunk.
    
    Returns:
        Flat list of all Chunk objects across all documents.
    """
    from tqdm import tqdm
    
    all_chunks = []
    empty_count = 0
    
    for doc in tqdm(documents, desc="Chunking documents"):
        doc_chunks = chunk_document(doc)
        if not doc_chunks:
            empty_count += 1
            continue
        all_chunks.extend(doc_chunks)
    
    # Log statistics
    if documents:
        avg_chunks = len(all_chunks) / max(1, len(documents) - empty_count)
        avg_length = sum(len(c.text) for c in all_chunks) / max(1, len(all_chunks))
        logger.info(
            f"Chunking complete: {len(documents)} docs → {len(all_chunks)} chunks "
            f"(avg {avg_chunks:.1f} chunks/doc, avg {avg_length:.0f} chars/chunk, "
            f"{empty_count} empty docs skipped)"
        )
    
    return all_chunks


# ============================================================
# CLI Entry Point (for standalone testing)
# ============================================================

if __name__ == "__main__":
    from src.data_loader import load_documents
    
    docs = load_documents(limit=5)
    chunks = chunk_documents(docs)
    
    for chunk in chunks[:5]:
        print(f"\n{'='*60}")
        print(f"Chunk ID:  {chunk.chunk_id}")
        print(f"Doc ID:    {chunk.doc_id}")
        print(f"Index:     {chunk.chunk_index}/{chunk.total_chunks}")
        print(f"Length:    {len(chunk.text)} chars")
        print(f"Text:      {chunk.text[:200]}...")
