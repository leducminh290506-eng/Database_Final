"""
retriever.py - Retrieve relevant legal document chunks for a user query.

This module orchestrates:
  1. Query validation and preprocessing
  2. Query embedding via SentenceTransformers
  3. Milvus vector search (IVF_FLAT + Cosine) for TOP-K similar chunks
  4. Relevance highlighting (most relevant sentence in each chunk)
  5. Result formatting with metadata and similarity scores

The retriever acts as the bridge between the user's natural-language query
and the Milvus vector database, converting semantic intent into concrete
document references.
"""

import asyncio
import re
from typing import Optional

import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config
from src.embedding import embed_query, embed_query_async
from src.vector_store import MilvusVectorStore


# ============================================================
# Query Validation
# ============================================================

def validate_query(query: str) -> tuple[bool, str]:
    """
    Validate a user query before processing.
    
    Checks:
      - Non-empty after stripping whitespace
      - Minimum length (at least 5 characters for meaningful queries)
      - Not just numbers or special characters
      - Not excessively long (prevent abuse)
    
    Args:
        query: Raw user query string.
    
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if not query or not query.strip():
        return False, "Câu hỏi không được để trống. (Query cannot be empty.)"
    
    query = query.strip()
    
    if len(query) < 5:
        return False, "Câu hỏi quá ngắn. Vui lòng nhập ít nhất 5 ký tự. (Query too short.)"
    
    if len(query) > 2000:
        return False, "Câu hỏi quá dài. Vui lòng rút gọn dưới 2000 ký tự. (Query too long.)"
    
    # Check if query is just numbers or special characters
    cleaned = re.sub(r"[\d\s\W]", "", query)
    if len(cleaned) < 2:
        return False, "Câu hỏi không hợp lệ. Vui lòng nhập câu hỏi bằng chữ. (Invalid query.)"
    
    return True, ""


# ============================================================
# Relevance Highlighting
# ============================================================

def highlight_relevant_sentence(chunk_text: str, query: str) -> str:
    """
    Find and mark the most relevant sentence in a chunk relative to the query.
    
    Uses simple word-overlap scoring (TF-IDF-like) to identify the sentence
    that shares the most terms with the query. The matched sentence is wrapped
    in **bold** markers for display.
    
    Algorithm:
      1. Split chunk into sentences
      2. Tokenize query into lowercase words
      3. Score each sentence by counting query-word matches
      4. Return the chunk with the top sentence highlighted
    
    Args:
        chunk_text: The chunk text to highlight within.
        query: The user query to match against.
    
    Returns:
        Chunk text with the most relevant sentence wrapped in ** markers.
    """
    if not chunk_text or not query:
        return chunk_text
    
    # Split into sentences (Vietnamese uses similar punctuation)
    sentences = re.split(r"(?<=[.!?;])\s+", chunk_text)
    if len(sentences) <= 1:
        return chunk_text
    
    # Tokenize query into words (lowercase, remove punctuation)
    query_words = set(re.findall(r"\w+", query.lower()))
    if not query_words:
        return chunk_text
    
    # Score each sentence by query-word overlap
    best_score = 0
    best_idx = 0
    
    for idx, sentence in enumerate(sentences):
        sentence_words = set(re.findall(r"\w+", sentence.lower()))
        overlap = len(query_words & sentence_words)
        # Weight by inverse sentence length to prefer concise matches
        score = overlap / max(1, len(sentence_words) ** 0.5)
        
        if score > best_score:
            best_score = score
            best_idx = idx
    
    # Highlight the best sentence
    if best_score > 0:
        sentences[best_idx] = f"**{sentences[best_idx]}**"
    
    return " ".join(sentences)


# ============================================================
# Retriever
# ============================================================

class Retriever:
    """
    Retrieves the most relevant legal document chunks for a user query.
    
    Workflow:
      1. Validate the query
      2. Embed the query using SentenceTransformers
      3. Search the Milvus index for TOP-K nearest neighbors
      4. Highlight the most relevant sentence in each result
      5. Return formatted results with scores and metadata
    """
    
    def __init__(self, vector_store: Optional[MilvusVectorStore] = None):
        """
        Initialize the retriever.
        
        Args:
            vector_store: A loaded MilvusVectorStore instance. If None,
                         loads from the default Milvus database.
        """
        if vector_store is None:
            logger.info("Loading Milvus vector store...")
            self.vector_store = MilvusVectorStore.load()
        else:
            self.vector_store = vector_store

        # Cache doc_id -> {chunk_index: chunk_text} for lightweight context expansion.
        self._doc_chunk_cache: dict[str, dict[int, str]] = {}
        
        logger.info(f"Retriever initialized with {self.vector_store.size} indexed chunks")

    @staticmethod
    def _parse_fixed_chunk_id(chunk_id: str) -> tuple[str, int] | None:
        """
        Parse chunk IDs with fixed-window format: "<doc_id>_chunk_<index>".

        Returns:
            Tuple (doc_id_prefix, chunk_index) or None if not fixed format.
        """
        match = re.match(r"^(?P<prefix>.+)_chunk_(?P<idx>\d+)$", chunk_id)
        if not match:
            return None
        return match.group("prefix"), int(match.group("idx"))

    @staticmethod
    def _merge_text_with_overlap(texts: list[str], max_overlap: int = 200) -> str:
        """
        Merge consecutive chunk texts while removing duplicated overlap tails/heads.
        """
        if not texts:
            return ""

        merged = texts[0].strip()
        for nxt in texts[1:]:
            nxt = nxt.strip()
            if not nxt:
                continue

            overlap = 0
            max_check = min(max_overlap, len(merged), len(nxt))
            for size in range(max_check, 19, -1):
                if merged[-size:] == nxt[:size]:
                    overlap = size
                    break

            if overlap > 0:
                merged += nxt[overlap:]
            else:
                if merged and not merged.endswith("\n"):
                    merged += "\n"
                merged += nxt

        return merged.strip()

    def _get_doc_fixed_chunks(self, doc_id: str) -> dict[int, str]:
        """
        Load all fixed-window chunks for one document from Milvus.

        For article-level chunks ("_art_") this returns an empty map, and we keep
        the original retrieval text unchanged.
        """
        if doc_id in self._doc_chunk_cache:
            return self._doc_chunk_cache[doc_id]

        safe_doc_id = doc_id.replace('\\', '\\\\').replace('"', '\\"')
        chunk_map: dict[int, str] = {}

        try:
            rows = self.vector_store.client.query(
                collection_name=self.vector_store.collection_name,
                filter=f'doc_id == "{safe_doc_id}"',
                output_fields=["chunk_id", "text"],
                limit=16384,
            )
        except Exception as e:
            logger.debug(f"Failed to load sibling chunks for doc_id={doc_id}: {e}")
            self._doc_chunk_cache[doc_id] = {}
            return self._doc_chunk_cache[doc_id]

        for row in rows:
            chunk_id = row.get("chunk_id", "")
            parsed = self._parse_fixed_chunk_id(chunk_id)
            if not parsed:
                continue

            _, idx = parsed
            text = row.get("text", "")
            if text:
                chunk_map[idx] = text

        self._doc_chunk_cache[doc_id] = dict(sorted(chunk_map.items()))
        return self._doc_chunk_cache[doc_id]

    def _expand_retrieved_text(
        self,
        result: dict,
        neighbor_window: int = 1,
        max_chars: int = 4000,
    ) -> str:
        """
        Expand a retrieved fixed-window chunk with adjacent chunks.

        This reduces "cut-off" snippets when the index was built using fixed
        character chunking. If the chunk is already article-level, original text
        is returned unchanged.
        """
        base_text = result.get("text", "")
        chunk_id = result.get("chunk_id", "")
        doc_id = result.get("doc_id", "")

        parsed = self._parse_fixed_chunk_id(chunk_id)
        if not parsed or not doc_id:
            return base_text

        _, center_idx = parsed
        doc_chunks = self._get_doc_fixed_chunks(doc_id)
        if not doc_chunks:
            return base_text

        selected = [
            doc_chunks[i]
            for i in range(max(0, center_idx - neighbor_window), center_idx + neighbor_window + 1)
            if i in doc_chunks
        ]

        if not selected:
            return base_text

        expanded = self._merge_text_with_overlap(selected)
        if not expanded:
            return base_text

        if len(expanded) > max_chars:
            clipped = expanded[:max_chars]
            return clipped.rsplit(" ", 1)[0].strip() or clipped.strip()

        return expanded
    
    def retrieve(
        self,
        query: str,
        top_k: int = config.TOP_K,
        min_score: float = config.MIN_SIMILARITY_SCORE,
    ) -> dict:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: The user's natural-language question.
            top_k: Number of results to return.
            min_score: Minimum cosine similarity threshold.
        
        Returns:
            Dict with keys:
              - query: The original query
              - results: List of result dicts with text, score, metadata, highlighted_text
              - is_valid: Whether the query was valid
              - error: Error message if query was invalid
        """
        # Step 1: Validate query
        is_valid, error_msg = validate_query(query)
        if not is_valid:
            return {
                "query": query,
                "results": [],
                "is_valid": False,
                "error": error_msg,
            }
        
        query = query.strip()
        
        # Step 2: Embed the query
        logger.debug(f"Embedding query: '{query[:80]}...'")
        query_embedding = embed_query(query)
        
        # Step 3: Search Milvus
        raw_results = self.vector_store.search(query_embedding, top_k=top_k, min_score=min_score)
        
        # Step 4: Highlight and format results
        results = []
        for r in raw_results:
            expanded_text = self._expand_retrieved_text(r)
            highlighted = highlight_relevant_sentence(expanded_text, query)
            results.append({
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "text": expanded_text,
                "highlighted_text": highlighted,
                "score": r["score"],
                "metadata": r["metadata"],
            })
        
        score_strs = [f"{r['score']:.3f}" for r in results]
        logger.info(
            f"Retrieved {len(results)} results for query: '{query[:50]}...' "
            f"(scores: {score_strs})"
        )
        
        return {
            "query": query,
            "results": results,
            "is_valid": True,
            "error": "",
        }
    
    async def retrieve_async(
        self,
        query: str,
        top_k: int = config.TOP_K,
        min_score: float = config.MIN_SIMILARITY_SCORE,
    ) -> dict:
        """
        Async version of retrieve for non-blocking execution in FastAPI.
        
        Args:
            query: The user's question.
            top_k: Number of results.
            min_score: Minimum similarity threshold.
        
        Returns:
            Same structure as retrieve().
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.retrieve(query, top_k=top_k, min_score=min_score)
        )


# ============================================================
# CLI Entry Point (for standalone testing)
# ============================================================

if __name__ == "__main__":
    import sys
    
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "quy định về thuế thu nhập cá nhân"
    
    retriever = Retriever()
    result = retriever.retrieve(query)
    
    if not result["is_valid"]:
        print(f"Invalid query: {result['error']}")
    else:
        print(f"\nQuery: {result['query']}")
        print(f"Found {len(result['results'])} results:\n")
        
        for i, r in enumerate(result["results"], 1):
            print(f"{'='*60}")
            print(f"Result #{i}")
            print(f"  Score:    {r['score']:.4f}")
            print(f"  Doc ID:   {r['doc_id']}")
            print(f"  Title:    {r['metadata'].get('title', 'N/A')[:80]}")
            print(f"  Type:     {r['metadata'].get('doc_type', 'N/A')}")
            print(f"  Snippet:  {r['highlighted_text'][:200]}...")
