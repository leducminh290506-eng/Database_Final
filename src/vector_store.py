"""
vector_store.py - Milvus vector database for storing and searching legal document embeddings.

This module implements:
  1. Milvus Standalone collection creation with schema definition
  2. IVF_FLAT index with cosine similarity for ANN search
  3. Batch insertion of embeddings with metadata
  4. Semantic search with similarity scores

Why Vector DB (Milvus) over a Relational DB?
  Relational databases (PostgreSQL, MySQL) are optimized for exact-match queries
  using B-tree or inverted indices. They cannot efficiently perform high-dimensional
  nearest neighbor search on dense vectors. A query like "tìm luật về đất đai" can
  match documents about "quy định sử dụng đất" even though the words are different —
  this is semantic similarity, which requires vector space operations.

  Milvus is purpose-built for vector similarity search. It supports multiple
  ANN algorithms and can scale to billions of vectors across distributed clusters.

Indexing Strategy:
  - IVF_FLAT (Inverted File with Flat quantization):
    * Partitions the vector space into `nlist` Voronoi cells (clusters)
    * At query time, searches only `nprobe` nearest clusters instead of all vectors
    * This gives sub-linear O(n/nlist * nprobe) search time
    * Trade-off: higher nprobe = better recall but slower search
  - Metric: COSINE similarity (directly supported by Milvus, no manual normalization)

Deployment:
  Milvus Standalone runs as a Docker container. Start with:
    docker-compose up -d
  The server listens on localhost:19530 (gRPC).
"""

from typing import Optional

import numpy as np
from loguru import logger
from pymilvus import MilvusClient, DataType

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config
from src.chunking import Chunk


# ============================================================
# Vector Store Class
# ============================================================

class MilvusVectorStore:
    """
    Milvus-based vector store for legal document chunk embeddings.
    
    Connects to Milvus Standalone running in Docker.
    Data is persisted inside the Docker volume.
    
    Stores:
      - Embedding vectors (384-dim float)
      - chunk_id, doc_id, text (the chunk content)
      - Metadata: title, doc_type, date_issued, authority, doc_number
    """
    
    def __init__(self, uri: Optional[str] = None):
        """
        Initialize the Milvus vector store.
        
        Connects to Milvus Standalone via gRPC URI.
        
        Args:
            uri: Milvus server URI (e.g., "http://localhost:19530").
                 Defaults to config.MILVUS_URI.
        """
        self.uri = uri or config.MILVUS_URI
        self.collection_name = config.MILVUS_COLLECTION_NAME
        self.dimension = config.EMBEDDING_DIMENSION
        
        # Connect to Milvus Standalone (Docker)
        self.client = MilvusClient(uri=self.uri)
        
        logger.info(
            f"Milvus connected: uri={self.uri}, "
            f"collection={self.collection_name}"
        )
    
    def _create_collection(self) -> None:
        """
        Create the Milvus collection with the defined schema.
        
        Schema:
          - id: INT64, auto-generated primary key
          - chunk_id: VARCHAR(200), unique identifier for the chunk
          - doc_id: VARCHAR(100), parent document ID
          - text: VARCHAR(65535), the chunk text content
          - title: VARCHAR(1000), document title
          - doc_type: VARCHAR(200), document type
          - date_issued: VARCHAR(50), issuance date
          - authority: VARCHAR(500), issuing authority
          - doc_number: VARCHAR(200), document number
          - embedding: FLOAT_VECTOR(384), the dense embedding
        """
        # Drop existing collection if it exists (fresh rebuild)
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info(f"Dropped existing collection: {self.collection_name}")
        
        # Define schema
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=2000)
        schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="date_issued", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="authority", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="doc_number", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dimension,
        )
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )
        
        logger.info(
            f"Collection '{self.collection_name}' created with schema: "
            f"dim={self.dimension}, metric=COSINE"
        )
    
    def _create_index(self) -> None:
        """
        Create the IVF_FLAT index on the embedding field.
        
        IVF_FLAT (Inverted File with Flat quantization):
          - Partitions vectors into nlist clusters using k-means
          - At query time, computes distances only within nprobe nearest clusters
          - Trade-off: nprobe=1 is fastest but low recall; nprobe=nlist is exact but slow
          - Our setting: nlist=128, nprobe=16 → good balance for ~500k vectors
        """
        index_params = self.client.prepare_index_params()
        
        index_params.add_index(
            field_name="embedding",
            index_type=config.MILVUS_INDEX_TYPE,    # "IVF_FLAT"
            metric_type=config.MILVUS_METRIC_TYPE,  # "COSINE"
            params={"nlist": config.MILVUS_NLIST},  # 128 clusters
        )
        
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )
        
        logger.info(
            f"Index created: type={config.MILVUS_INDEX_TYPE}, "
            f"metric={config.MILVUS_METRIC_TYPE}, nlist={config.MILVUS_NLIST}"
        )
    
    @property
    def size(self) -> int:
        """Return the number of vectors in the collection."""
        try:
            if not self.client.has_collection(self.collection_name):
                return 0
            stats = self.client.get_collection_stats(self.collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0
    
    def build(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """
        Build the Milvus collection from scratch: create schema, insert data, create index.
        
        This is the main entry point for index building. It drops any existing
        collection and creates a fresh one.
        
        Args:
            embeddings: NumPy array of shape (n, dimension), float32.
            chunks: List of Chunk objects (same length as embeddings).
        
        Raises:
            ValueError: If embeddings and chunks have mismatched lengths.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and chunks ({len(chunks)}) "
                f"must have the same length"
            )
        
        if len(embeddings) == 0:
            logger.warning("No embeddings to add")
            return
        
        # Step 1: Create collection with schema
        self._create_collection()
        
        # Step 2: Insert data in batches (Milvus recommends batches of ~1000)
        batch_size = 1000
        total = len(chunks)
        
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]
            
            # Prepare data for insertion
            data = []
            for chunk, emb in zip(batch_chunks, batch_embeddings):
                data.append({
                    "chunk_id": chunk.chunk_id[:200],
                    "doc_id": chunk.doc_id[:100],
                    "text": chunk.text[:65535],
                    "title": str(chunk.metadata.get("title", ""))[:2000],
                    "doc_type": str(chunk.metadata.get("doc_type", ""))[:200],
                    "date_issued": str(chunk.metadata.get("date_issued", ""))[:50],
                    "authority": str(chunk.metadata.get("authority", ""))[:500],
                    "doc_number": str(chunk.metadata.get("doc_number", ""))[:200],
                    "embedding": emb.tolist(),
                })
            
            self.client.insert(
                collection_name=self.collection_name,
                data=data,
            )
            
            logger.info(f"Inserted batch {start}-{end} / {total}")
        
        # Step 3: Create index for fast ANN search
        self._create_index()
        
        # Step 4: Load collection into memory for searching
        self.client.load_collection(self.collection_name)
        
        logger.info(f"Build complete: {total} vectors indexed in '{self.collection_name}'")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = config.TOP_K,
        min_score: float = config.MIN_SIMILARITY_SCORE,
    ) -> list[dict]:
        """
        Search for the most similar chunks to a query embedding.
        
        Uses Milvus ANN search with IVF_FLAT index and cosine similarity.
        
        Args:
            query_embedding: Query vector of shape (1, dimension) or (dimension,).
            top_k: Number of results to return.
            min_score: Minimum cosine similarity threshold.
        
        Returns:
            List of dicts with keys: text, score, metadata, chunk_id, doc_id.
            Sorted by descending similarity score.
        """
        if not self.client.has_collection(self.collection_name):
            logger.warning("Collection does not exist, no results")
            return []
        
        # Ensure correct shape
        if query_embedding.ndim == 2:
            query_vector = query_embedding[0].tolist()
        else:
            query_vector = query_embedding.tolist()
        
        # Search with Milvus
        search_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["chunk_id", "doc_id", "text", "title", "doc_type",
                           "date_issued", "authority", "doc_number"],
            search_params={
                "metric_type": config.MILVUS_METRIC_TYPE,
                "params": {"nprobe": config.MILVUS_NPROBE},
            },
        )
        
        results = []
        for hit in search_results[0]:
            score = float(hit["distance"])  # Milvus COSINE returns similarity score
            
            if score < min_score:
                continue
            
            entity = hit["entity"]
            results.append({
                "chunk_id": entity.get("chunk_id", ""),
                "doc_id": entity.get("doc_id", ""),
                "text": entity.get("text", ""),
                "score": score,
                "metadata": {
                    "title": entity.get("title", ""),
                    "doc_type": entity.get("doc_type", ""),
                    "date_issued": entity.get("date_issued", ""),
                    "authority": entity.get("authority", ""),
                    "doc_number": entity.get("doc_number", ""),
                    "chunk_id": entity.get("chunk_id", ""),
                    "doc_id": entity.get("doc_id", ""),
                },
            })
        
        return results
    
    def load_collection(self) -> None:
        """Load the collection into memory for searching (required after restart)."""
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' loaded into memory")
        else:
            raise FileNotFoundError(
                f"Collection '{self.collection_name}' not found in Milvus at {self.uri}. "
                f"Run 'python scripts/build_index.py' first."
            )
    
    @classmethod
    def load(cls, uri: Optional[str] = None) -> "MilvusVectorStore":
        """
        Load an existing Milvus vector store (connect and load collection).
        
        Args:
            uri: Milvus server URI.
        
        Returns:
            MilvusVectorStore instance with the collection loaded.
        
        Raises:
            FileNotFoundError: If the collection doesn't exist.
        """
        store = cls(uri=uri)
        store.load_collection()
        logger.info(f"Milvus store loaded: {store.size} vectors")
        return store


# Note: Previously aliased as FAISSVectorStore (removed after migration to Milvus)


# ============================================================
# CLI Entry Point (for standalone testing)
# ============================================================

if __name__ == "__main__":
    # Quick test with random vectors
    dim = 384
    n = 100
    
    store = MilvusVectorStore()
    
    # Create dummy data
    embeddings = np.random.randn(n, dim).astype(np.float32)
    
    chunks = [
        Chunk(
            chunk_id=f"doc_{i}_chunk_0",
            doc_id=f"doc_{i}",
            text=f"This is test chunk {i} for Milvus vector database testing",
            chunk_index=0,
            total_chunks=1,
            metadata={"title": f"Document {i}", "doc_type": "Test"},
        )
        for i in range(n)
    ]
    
    # Build index and search
    store.build(embeddings, chunks)
    
    query = np.random.randn(1, dim).astype(np.float32)
    
    results = store.search(query, top_k=3)
    for r in results:
        print(f"  Score: {r['score']:.4f} | {r['chunk_id']} | {r['text'][:60]}")
    
    print(f"\nStore size: {store.size}")
