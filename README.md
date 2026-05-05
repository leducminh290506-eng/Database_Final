# ⚖️ Vietnamese Legal Assistant RAG System

> **Hệ thống trợ lý pháp lý Việt Nam sử dụng Retrieval-Augmented Generation (RAG)**

A production-ready full-stack system that embeds ~153,000 Vietnamese legal documents,
indexes them in a **Milvus Standalone** vector database (Docker) using IVF_FLAT + Cosine Similarity (ANN),
retrieves the TOP-3 most relevant documents for a user's legal question, and generates
grounded answers with citations using Google Gemini.

For retrieval benchmarking, this project also includes a **PostgreSQL server** and
comparison scripts to evaluate Milvus semantic search vs PostgreSQL keyword/full-text search.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Data Flow](#-data-flow)
- [Setup Instructions](#-setup-instructions)
- [Quick Start](#-quick-start-lần-chạy-sau)
- [How to Run](#-how-to-run)
- [PostgreSQL Comparison Setup](#-postgresql-comparison-setup-milvus-vs-postgresql)
- [Benchmark Query Set](#-benchmark-query-set-for-comparison)
- [Example Query](#-example-query)
- [Design Decisions](#-design-decisions)
- [Trade-offs](#-trade-offs)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)

---

## 🎯 Project Overview

This system answers Vietnamese legal questions by:

1. **Embedding** 153k+ legal documents using SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`)
2. **Storing** embeddings in **Milvus** vector database using IVF_FLAT index with Cosine Similarity
3. **Retrieving** the TOP-3 most relevant document chunks using **ANN** (Approximate Nearest Neighbor) search
4. **Generating** answers using Google Gemini with strict anti-hallucination prompting
5. **Displaying** results in a Streamlit UI with similarity scores, highlighted snippets, and citations

### Key Features

| Feature | Implementation |
|---------|---------------|
| Semantic Search | SentenceTransformers → Milvus ANN + Cosine Similarity |
| Anti-Hallucination | Closed-book prompting with mandatory citations |
| Scalable Embedding | Batch processing (256/batch) with async wrappers |
| Text Cleaning | HTML stripping, Unicode normalization, noise removal |
| Article Chunking | Split by legal article (Điều), export to CSV |
| Production API | FastAPI with Pydantic validation, CORS, async endpoints |
| Rich UI | Streamlit dual-panel: chat + evidence with highlighted snippets |
| Login / Logout | PostgreSQL-backed user auth with persistent sessions |
| Persistent Chat | Chat history saved per user across sessions |
| Search Comparison | Side-by-side Milvus ANN vs PG ILIKE vs PG FTS in UI |
| Access Logging | User actions (login, logout, queries) logged to PostgreSQL |
| Dynamic Top-K | UI slider fixed to 1-10 (default: 3) |
| Logging | Loguru with file rotation and structured logging |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE (OFFLINE)                   │
│                                                             │
│  HuggingFace ──▶ Data Loader ──▶ Article Chunker ──▶ CSV   │
│  Dataset          (HTML→text)    (split by Điều)    Export  │
│                                        │                    │
│                                   Embedder (MiniLM)         │
│                                        │                    │
│                                  ┌─────▼─────┐             │
│                                  │  Milvus   │             │
│                                  │Standalone │             │
│                                  │(IVF_FLAT) │             │
│                                  └─────┬─────┘             │
│                                        │                    │
└────────────────────────────────────────│────────────────────┘
                                         │
┌────────────────────────────────────────│────────────────────┐
│                   SERVING PIPELINE (ONLINE)                 │
│                                        │                    │
│  User ──▶ Streamlit ──▶ FastAPI ──▶ Retriever ────┘        │
│  Query      (UI)         (API)      (embed query            │
│                            │       + ANN Cosine search)     │
│                            │                                │
│                      ┌─────▼─────┐                          │
│                      │  Gemini   │◀── Retrieved chunks      │
│                      │   LLM     │    as context             │
│                      └─────┬─────┘                          │
│                            │                                │
│                      Answer with                            │
│                      citations                              │
└─────────────────────────────────────────────────────────────┘
```

Benchmark path for retrieval comparison:

```
Query ──▶ scripts/compare_search.py
   ├─▶ Milvus ANN (semantic)
   ├─▶ PostgreSQL ILIKE (keyword substring)
   └─▶ PostgreSQL FTS (tsvector rank)
```

---

## 🔄 Data Flow

### Offline (Index Building)

```
1. HuggingFace Dataset (153k docs)
   ↓ load from "th1nhng0/vietnamese-legal-documents"
2. Merge metadata + content on document ID
   ↓ inner join → ~149k docs with both metadata and HTML content
3. HTML → Plain Text (BeautifulSoup + lxml)
   ↓ strip tags, preserve paragraph structure
4. Text Cleaning
   ↓ remove NULL bytes, normalize Unicode, collapse whitespace
5. Article-Level Chunking (by Điều)
   ↓ each legal article (Điều) = 1 chunk, preamble kept separately
   ↓ exported to data/chunks_by_article.csv
6. CSV-First Chunk Loading for Index Build
   ↓ build_index.py reads chunks directly from data/chunks_by_article.csv
   ↓ if CSV is missing (or --rebuild-csv), CSV is regenerated first
7. Batch Embedding (SentenceTransformers, 256/batch)
   ↓ 384-dim dense vectors
8. Milvus Standalone Collection (IVF_FLAT + COSINE)
   ↓ stored in Docker volume (localhost:19530)
```

### Online (Query Serving)

```
1. User enters legal question in Streamlit UI
   ↓ POST /query to FastAPI
2. Query Validation (length, content checks)
   ↓ reject invalid queries early
3. Query Embedding (same SentenceTransformers model)
   ↓ 384-dim vector
4. Milvus ANN Search (IVF_FLAT, Cosine Similarity)
   ↓ TOP-3 most similar chunks from nprobe=16 clusters
5. Relevance Highlighting (sentence-level TF-IDF overlap)
   ↓ most relevant sentence highlighted in each chunk
6. RAG Prompt Construction
   ↓ strict anti-hallucination instructions + context + question
7. Gemini Generation (temperature=0.1)
   ↓ answer with [Nguồn: ...] citations
8. Response returned to Streamlit UI
   ↓ dual-panel: chat (left) + evidence cards (right)

Fallback path in UI when API is offline:
1. Streamlit detects backend connection error
   ↓
2. Local RAGPipeline is initialized in Streamlit process
   ↓
3. Query is processed locally and returned in the same UI

Comparison path for benchmark script:
1. `scripts/compare_search.py` receives one query
2. Runs Milvus ANN search with the same query embedding
3. Runs PostgreSQL ILIKE and PostgreSQL FTS with the same raw query text
4. Prints latency, top-k results, overlap, and quick analysis
```

---

## 🛠 Setup Instructions

### Prerequisites

- **Python** 3.10+
- **pip** (latest version)
- **Docker Desktop** (for Milvus Standalone + PostgreSQL)
- **Google Cloud service account key (JSON)** with Vertex AI access

### Installation

```bash
# 1. Clone/navigate to the project directory
cd database_final

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
copy .env.example .env

# 5. Configure Google credentials for Vertex AI
copy google-creds.json\google-creds.example.json google-creds.json\google-creds.json
# Edit google-creds.json\google-creds.json with your real service account key
```

---

## ⚡ Quick Start (Lần Chạy Sau)

Nếu bạn đã có:
- `data/chunks_by_article.csv`
- Milvus index đã build
- Bảng PostgreSQL `legal_chunks` đã nạp dữ liệu

thì chỉ cần chạy các bước sau:

```bash
# 1) Start Docker services (Milvus + PostgreSQL + API + UI)
docker-compose up -d

# 2) Create auth/chat tables (first time only, local run)
python postgres/setup_tables.py

# 3) Start FastAPI backend (local run)
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 4) Start Streamlit UI (local run, new terminal)
streamlit run ui.py

# 5) (Optional) Run one comparison query (CLI)
python scripts/compare_search.py "quy định về thuế thu nhập cá nhân"
```

Mở ứng dụng tại `http://localhost:8501`.

> **Note:** `docker-compose up -d` already starts both the API and UI containers.
> Only run the local `uvicorn`/`streamlit` commands if you want to run the app outside Docker.

Nếu cần nạp lại PostgreSQL baseline:

```bash
python scripts/pg_setup.py
```

Chỉ cần chạy lại chunk/index khi có dữ liệu mới hoặc đổi model/chunking/index params:

```bash
python scripts/chunk_by_article.py
python scripts/build_index.py
```

## 🚀 How to Run

### Step 1: Start Infrastructure (Milvus + PostgreSQL + API + UI)

```bash
# Start Milvus + etcd + MinIO + PostgreSQL
docker-compose up -d

# Verify containers are healthy
docker-compose ps
```

Expected services:
- `milvus-standalone` on `localhost:19530` (vector search)
- `legal-postgres` on `localhost:5433` (keyword/full-text baseline)
- `toxagent-app` on `localhost:8000` (FastAPI backend, Docker)
- `toxagent-ui` on `localhost:8501` (Streamlit UI, Docker)

### Step 2: Chunk Documents by Article (Điều) & Export to CSV

```bash
# Quick test with 10 documents
python scripts/chunk_by_article.py --limit 10

# Full dataset (~153k docs) → exports to data/chunks_by_article.csv
python scripts/chunk_by_article.py

# Custom output path
python scripts/chunk_by_article.py --output data/my_chunks.csv
```

This splits each legal document by `Điều` (article) so that **each article = 1 chunk**.
Output CSV is saved to `data/chunks_by_article.csv`.

### Step 3: Build the Milvus Index

```bash
# Quick test with first 100 chunks from CSV (~1-5 minutes)
python scripts/build_index.py --limit 100

# Full index build from default CSV: data/chunks_by_article.csv
python scripts/build_index.py

# Custom batch size (larger = faster if you have GPU VRAM)
python scripts/build_index.py --batch-size 512

# Build from a custom chunk CSV
python scripts/build_index.py --chunks-csv data/my_chunks.csv

# Force re-download/rebuild CSV from HuggingFace, then index from CSV
python scripts/build_index.py --rebuild-csv
```

`build_index.py` uses CSV as the only indexing input path for embedding/indexing.
If CSV is missing (or `--rebuild-csv` is used), CSV is regenerated first, then loaded back for indexing.

### Step 4: Load Chunks into PostgreSQL (for Comparison)

```bash
# Quick test with first 1000 chunks
python scripts/pg_setup.py --limit 1000

# Full load from default CSV: data/chunks_by_article.csv
python scripts/pg_setup.py

# Load from custom CSV
python scripts/pg_setup.py --chunks-csv data/my_chunks.csv
```

Optional verification:

```bash
docker exec legal-postgres psql -U legal_user -d legal_db -c "SELECT COUNT(*) FROM legal_chunks;"
```

### Step 5: Start the FastAPI Backend

```bash
# Development mode with auto-reload
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Or run directly
python api.py
```

The API will be available at `http://localhost:8000`.
API docs at `http://localhost:8000/docs` (Swagger UI).

### Step 6: Start the Streamlit Frontend

```bash
# In a new terminal
streamlit run ui.py
```

The UI will open at `http://localhost:8501`.

> **Note:** If you started services with `docker-compose up -d`, the API and UI
> are already running in containers. Only run the local commands when you want
> to run the app outside Docker.

### Step 7: Run Milvus vs PostgreSQL Comparison Script

```bash
# Single query (default top-k=3)
python scripts/compare_search.py "quy định về thuế thu nhập cá nhân"

# Custom top-k
python scripts/compare_search.py "luật đất đai" --top-k 5
```

### Use the API Directly (Optional)

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quy định về thuế thu nhập cá nhân là gì?", "top_k": 3}'
```

---

## ⚖️ PostgreSQL Comparison Setup (Milvus vs PostgreSQL)

Script `scripts/compare_search.py` chạy cùng một câu truy vấn qua 3 cơ chế:

1. **Milvus ANN (IVF_FLAT + Cosine)**: semantic search theo embedding.
2. **PostgreSQL ILIKE**: match từ khóa dạng chuỗi con.
3. **PostgreSQL Full-Text Search (tsvector/tsquery)**: keyword search có điểm rank.

Kết quả script hiển thị:
- Thời gian truy vấn của từng cơ chế.
- Top-K documents cho từng cơ chế.
- Overlap giữa Milvus và PostgreSQL.
- Nhận xét nhanh về khác biệt semantic vs keyword.

PowerShell batch chạy nhiều câu hỏi:

```powershell
$queries = @(
   "quy định về thuế thu nhập cá nhân",
   "điều kiện thành lập doanh nghiệp",
   "quyền lợi khi bị cho nghỉ việc trái luật"
)

foreach ($q in $queries) {
   python scripts/compare_search.py $q --top-k 3
}
```

---

## 🧪 Benchmark Query Set for Comparison

Dùng bộ câu hỏi dưới đây để so sánh hành vi của Milvus (semantic) và PostgreSQL (keyword/FTS).

| ID | Query (Vietnamese) | Mục tiêu so sánh |
|---|---|---|
| Q01 | quy định về thuế thu nhập cá nhân | Keyword rõ ràng, cả hai hệ đều nên trả kết quả tốt |
| Q02 | điều kiện thành lập doanh nghiệp | Keyword pháp lý phổ biến |
| Q03 | hợp đồng lao động xác định thời hạn | Truy vấn chuyên ngành, kiểm tra độ chính xác thuật ngữ |
| Q04 | mức xử phạt nồng độ cồn khi lái xe | Truy vấn hành chính có từ khóa cụ thể |
| Q05 | nghĩa vụ đóng bảo hiểm xã hội của người lao động | Keyword dài, nhiều thực thể |
| Q06 | đi làm thì phải nộp thuế thế nào | Paraphrase đời thường, ưu thế semantic |
| Q07 | muốn mở công ty cần giấy tờ gì | Paraphrase của “điều kiện thành lập doanh nghiệp” |
| Q08 | bị công ty cho nghỉ trái luật thì có quyền gì | Câu hỏi tự nhiên, từ đồng nghĩa với “chấm dứt HĐLĐ trái pháp luật” |
| Q09 | nhà nước thu hồi đất thì đền bù ra sao | Semantic mapping giữa “đền bù” và “bồi thường” |
| Q10 | dùng nhạc của người khác đăng mạng có bị phạt không | Câu hỏi ngôn ngữ tự nhiên, ít keyword pháp điển |
| Q11 | doanh nghiệp chậm nộp thuế bị xử lý như thế nào | Truy vấn gồm nhiều ý trong một câu |
| Q12 | hợp đồng thuê nhà viết tay có giá trị pháp lý không | Truy vấn yes/no mang tính thực tế |
| Q13 | mua bán đất bằng giấy tay có sang tên được không | Paraphrase + ngữ cảnh dân sự |
| Q14 | điều 13 quy định gì về quyền và nghĩa vụ | Câu ngắn, mơ hồ, dễ nhiễu |
| Q15 | quy định đất đai bất động sản chuyển nhượng quyền sử dụng đất | Query nhiều từ khóa liên quan |
| Q16 | ??? thuế ??? cá nhân ??? | Query nhiễu, kiểm tra độ bền hệ thống |
| Q17 | thủ tục ly hôn đơn phương khi có con nhỏ | Chủ đề gia đình, kiểm tra coverage dữ liệu |
| Q18 | quyền và nghĩa vụ của người sử dụng lao động | Thuật ngữ pháp lý chuẩn để đối chiếu với paraphrase Q08 |

Gợi ý cách ghi nhận kết quả benchmark:

| Query ID | Milvus top-1 | PG ILIKE top-1 | PG FTS top-1 | Overlap | Nhận xét chất lượng |
|---|---|---|---|---|---|
| Q01 |  |  |  |  |  |
| Q02 |  |  |  |  |  |
| Q03 |  |  |  |  |  |


---

## 🧪 Example Query

**Input:**
```
Quy định về thuế thu nhập cá nhân là gì?
```

**Expected Output:**
```json
{
  "question": "Quy định về thuế thu nhập cá nhân là gì?",
  "answer": "Theo quy định tại Luật Thuế thu nhập cá nhân, thuế TNCN được áp dụng đối với...\n\n[Nguồn: Luật Thuế thu nhập cá nhân - 04/2007/QH12]\n\nNguồn tham khảo:\n1. Luật Thuế thu nhập cá nhân (04/2007/QH12)\n2. Nghị định 65/2013/NĐ-CP\n3. Thông tư 111/2013/TT-BTC",
  "documents": [
    {
      "title": "Luật Thuế thu nhập cá nhân",
      "doc_number": "04/2007/QH12",
      "score": 0.87,
      "highlighted_text": "**Thuế thu nhập cá nhân là loại thuế...** ..."
    }
  ],
  "processing_time_ms": 1250.5
}
```

---

## 📐 Design Decisions

### 1. Why Vector DB (Milvus) Instead of Relational DB?

| Aspect | Relational DB (SQL) | Vector DB (Milvus) |
|--------|--------------------|--------------------|
| **Query Type** | Exact match (WHERE, LIKE) | Semantic similarity |
| **Language Understanding** | None (keyword matching) | Deep (embedding space) |
| **Synonym Handling** | Manual synonyms table | Automatic via embeddings |
| **Speed at Scale** | O(n) for full-text search | O(n/nlist × nprobe) with ANN |
| **Multilingual** | No | Yes (multilingual models) |

A user searching for "luật về đất đai" (land law) should find documents about
"quy định sử dụng đất" (land use regulations) even though the words are different.
This is **semantic similarity**, which requires vector representations.

### 2. Why Milvus with IVF_FLAT Index?

- **IVF_FLAT** (Inverted File with Flat quantization):
  - Partitions all vectors into `nlist=128` Voronoi cells using k-means clustering
  - At search time, only `nprobe=16` nearest clusters are scanned (instead of all)
  - This is **ANN (Approximate Nearest Neighbor)** — gives sub-linear search time
  - Trade-off: `nprobe=nlist` = exact search (slow); `nprobe=1` = fastest (low recall)
  - Our setting (`nprobe=16/nlist=128`) scans ~12.5% of data → ~8x faster than brute force
- **COSINE** metric: directly supported by Milvus (no manual L2 normalization needed)
- **Milvus Standalone**: runs as a Docker container via `docker-compose up -d`, persistent storage in Docker volumes

### 3. Why paraphrase-multilingual-MiniLM-L12-v2?

- Supports 50+ languages including Vietnamese
- 384-dim output (compact, fast to search)
- Trained on 1B+ paraphrase pairs across languages
- 12-layer transformer (good quality/speed trade-off)
- 270MB model size (fits in memory easily)

### 4. Anti-Hallucination Strategy

The RAG prompt uses a **closed-book instruction pattern**:
- "ONLY answer using the provided context"
- "Every claim must include a citation [Nguồn: ...]"
- "If context is insufficient, say so explicitly"
- Temperature set to 0.1 (near-deterministic)

This prevents the LLM from generating plausible-sounding but incorrect legal information — critical for a legal assistant.

### 5. Article-Level Chunking (by Điều)

Instead of naive fixed-size splitting, we split by legal article boundaries:
- Each `Điều X.` (article) in a legal document becomes **one chunk**
- Content before the first `Điều` (preamble) is kept as a separate chunk
- Documents without any `Điều` are kept as a single chunk

This preserves the natural semantic unit of Vietnamese legal documents — each article
typically covers one specific topic or provision, producing higher-quality embeddings
and more precise retrieval compared to arbitrary character-based splitting.

---

## ⚖️ Trade-offs

### Milvus Deployment: Lite vs Standalone vs Distributed

| Aspect | Milvus Lite | Milvus Standalone (chosen) | Milvus Distributed |
|--------|---------------------|-------------------|--------------------|
| **Deployment** | In-process (no server) | Docker container | Kubernetes cluster |
| **Setup** | `pip install pymilvus` | `docker-compose up -d` | Helm chart |
| **Scale** | Single machine (<5M vectors) | Single machine (<100M) | Unlimited |
| **Best For** | Quick prototyping | Development & small production | Enterprise |

**Choice rationale:** Milvus Standalone via Docker provides production-grade persistence
and reliability while keeping setup simple with `docker-compose`. Data is stored in
Docker volumes, surviving container restarts.

### IVF_FLAT vs Other ANN Index Types

| Index Type | Method | Speed | Recall | Memory |
|-----------|--------|-------|--------|--------|
| FLAT | Brute force | Slow | 100% | 1x |
| **IVF_FLAT** ✓ | **Inverted file** | **Fast** | **~95%** | **1x** |
| IVF_SQ8 | Inverted + quantized | Faster | ~90% | 0.25x |
| HNSW | Graph-based | Very fast | ~98% | 1.5x |

**Choice:** IVF_FLAT gives excellent speed/recall trade-off for our ~500k vectors without
lossy compression.

### Chunking Strategy: Article-level vs Fixed-size

| Strategy | Pros | Cons |
|----------|------|------|
| Fixed 512 chars | Uniform chunk sizes | Splits articles mid-sentence |
| Fixed 1024 chars | Good context per chunk | Arbitrary boundaries |
| **By article (Điều)** ✓ | **Natural semantic units, complete legal provisions** | **Variable chunk sizes** |
| Full document | Maximum context | Too long for embedding models |

### Embedding Model Size

| Model | Dim | Size | Vietnamese Quality |
|-------|-----|------|-------------------|
| MiniLM-L6 | 384 | 80MB | Good |
| **MiniLM-L12** ✓ | **384** | **270MB** | **Very Good** |
| XLM-R-Large | 1024 | 2.2GB | Excellent |

**Choice:** L12 provides the best quality/speed ratio for production use.

---

## 🧰 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Dataset | HuggingFace Datasets | Load Vietnamese legal documents |
| Embedding | SentenceTransformers | Dense vector representations |
| Vector DB | **Milvus Standalone** (IVF_FLAT + COSINE) | ANN similarity search |
| Relational DB | **PostgreSQL 16** (ILIKE + FTS) | Keyword/full-text comparison baseline |
| LLM | Google Gemini | Answer generation with citations |
| Backend | FastAPI + Pydantic | REST API with validation |
| Frontend | Streamlit | Interactive chat + evidence UI |
| Logging | Loguru | Structured logging with rotation |
| Config | python-dotenv | Environment variable management |

---

## 📁 Project Structure

```
database_final/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Milvus + etcd + MinIO + PostgreSQL
├── .env.example                 # Environment variables template
├── google-creds.json/           # Local Google service-account credentials
│   ├── google-creds.example.json
│   └── google-creds.json        # Your real key file (do not commit)
├── config.py                    # Central configuration (incl. PG params)
│
├── src/                         # Core RAG modules
│   ├── __init__.py
│   ├── data_loader.py           # Load & preprocess HuggingFace dataset
│   ├── chunking.py              # Text chunking utilities
│   ├── embedding.py             # SentenceTransformers embedding
│   ├── vector_store.py          # Milvus vector database management
│   ├── retriever.py             # Query → retrieval pipeline
│   └── rag_pipeline.py          # RAG: retrieval + Gemini generation
│
├── postgres/                    # PostgreSQL modules (auth, chat, search)
│   ├── __init__.py
│   ├── db.py                    # Shared PG connection helper
│   ├── auth.py                  # User registration, login, password hashing
│   ├── chat_store.py            # Persistent chat sessions & messages CRUD
│   ├── search.py                # PG ILIKE & FTS search for comparison
│   └── setup_tables.py          # Create users/sessions/messages/logs tables
│
├── api.py                       # FastAPI backend (/query, /compare, /health, /stats)
├── ui.py                        # Streamlit frontend (login, chat, comparison)
│
├── scripts/
│   ├── build_index.py           # Build Milvus index from chunk CSV (CSV-first)
│   ├── chunk_by_article.py      # Chunk by Điều & export to CSV
│   ├── pg_setup.py              # Load chunk CSV into PostgreSQL + create indexes
│   ├── compare_search.py        # Compare Milvus ANN vs PostgreSQL ILIKE/FTS (CLI)
│   └── sample_data.py           # Inspect source text sample from dataset
│
├── data/                        # Generated data (not in git)
│   └── chunks_by_article.csv    # Article-level chunks (CSV export)
│
└── logs/                        # Log files (not in git)
    ├── build_index.log
    └── chunk_by_article.log
```

---

## 📜 License

This project uses the Vietnamese Legal Documents dataset released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Vietnamese legal documents are public domain under the Law on Access to Information (No. 104/2016/QH13).

---

## 🤝 Acknowledgments

- Dataset: [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents)
- Source: [vbpl.vn](https://vbpl.vn) — Official Government Legal Document Portal
- Embedding: [sentence-transformers](https://www.sbert.net/)
- Vector Search: [Milvus](https://milvus.io/)
