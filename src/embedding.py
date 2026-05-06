"""
embedding.py - Generate sentence embeddings for legal document chunks.

This module:
  1. Loads a multilingual SentenceTransformers model optimized for Vietnamese
  2. Encodes text chunks in batches for scalability
  3. L2-normalizes embeddings so inner product == cosine similarity
  4. Provides async wrappers for non-blocking embedding in API contexts

Design Decision:
  We use 'paraphrase-multilingual-MiniLM-L12-v2' because:
    - It supports 50+ languages including Vietnamese
    - 384-dimensional output is compact yet expressive
    - 12-layer architecture balances quality and speed
    - Trained on paraphrase data → strong semantic similarity performance
  
  L2 normalization is applied so that inner-product search in Milvus
  computes cosine similarity directly. Milvus also supports native COSINE
  metric, but normalized vectors ensure compatibility with any metric.
"""

import asyncio
from typing import Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config


# ============================================================
# Singleton Embedding Model
# ============================================================

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """
    Get or initialize the SentenceTransformer model (singleton pattern).
    
    Uses a global singleton to avoid loading the model multiple times,
    which would waste GPU/CPU memory and startup time.
    
    Returns:
        Initialized SentenceTransformer model.
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        _model.max_seq_length = config.MAX_SEQ_LENGTH
        logger.info(
            f"Model loaded: dim={_model.get_embedding_dimension()}, "
            f"max_seq_length={_model.max_seq_length}"
        )
    return _model


# ============================================================
# Embedding Functions
# ============================================================

def embed_texts(
    texts: list[str],
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts into dense embedding vectors.
    
    Process:
      1. Load the SentenceTransformer model (cached singleton)
      2. Encode in batches of `batch_size` for memory efficiency
      3. L2-normalize so inner product == cosine similarity
    
    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts to encode per batch.
        show_progress: Whether to show a progress bar.
        normalize: Whether to L2-normalize embeddings.
    
    Returns:
        NumPy array of shape (n_texts, embedding_dim) with float32 values.
    """
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, config.EMBEDDING_DIMENSION)
    
    model = get_model()
    
    logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize,  # L2-norm so IP == cosine
    )
    
    # Ensure float32 for Milvus compatibility
    embeddings = embeddings.astype(np.float32)
    
    logger.info(f"Embedding complete: shape={embeddings.shape}")
    return embeddings


def embed_query(query: str, normalize: bool = True) -> np.ndarray:
    """
    Embed a single query string.
    
    This is optimized for single-query latency (no batching overhead).
    
    Args:
        query: The user query to embed.
        normalize: Whether to L2-normalize the embedding.
    
    Returns:
        NumPy array of shape (1, embedding_dim).
    """
    model = get_model()
    
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    ).astype(np.float32)
    
    return embedding


async def embed_texts_async(
    texts: list[str],
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """
    Async wrapper for embed_texts to support non-blocking execution.
    
    Runs the CPU/GPU-bound embedding in a thread executor so it doesn't
    block the async event loop (important for FastAPI).
    
    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for encoding.
        normalize: Whether to L2-normalize embeddings.
    
    Returns:
        NumPy array of embeddings.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: embed_texts(texts, batch_size=batch_size, normalize=normalize)
    )


async def embed_query_async(query: str, normalize: bool = True) -> np.ndarray:
    """
    Async wrapper for embed_query.
    
    Args:
        query: The user query to embed.
        normalize: Whether to L2-normalize.
    
    Returns:
        NumPy array of shape (1, embedding_dim).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: embed_query(query, normalize=normalize)
    )


# ============================================================
# CLI Entry Point (for standalone testing)
# ============================================================

if __name__ == "__main__":
    # Quick test with sample Vietnamese legal texts
    test_texts = [
        "Nghị định về quản lý và sử dụng đất đai trong khu công nghiệp",
        "Thông tư hướng dẫn thực hiện Luật Thuế thu nhập cá nhân",
        "Quyết định về việc ban hành quy chế hoạt động của Hội đồng nhân dân",
    ]
    
    embeddings = embed_texts(test_texts)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"L2 norms: {np.linalg.norm(embeddings, axis=1)}")  # Should be ~1.0
    
    # Test cosine similarity via inner product
    sim_matrix = embeddings @ embeddings.T
    print(f"\nSimilarity matrix:\n{sim_matrix}")
