"""
rag_pipeline.py - Retrieval-Augmented Generation pipeline for legal Q&A.

This module:
  1. Takes retrieved document chunks as context
  2. Constructs a carefully designed prompt that forces the LLM to:
     - Answer ONLY based on the provided context
     - Cite specific documents (title, document number)
     - Refuse to answer if context is insufficient
  3. Calls Google Gemini (or OpenAI) to generate the final answer
  4. Returns a structured response with the answer and citations

Anti-Hallucination Strategy:
  The prompt uses a "closed-book" instruction pattern:
    - "ONLY answer using the provided context below"
    - "If the context does not contain sufficient information, say so explicitly"
    - "Every claim must include a citation in [Nguồn: ...] format"
  This dramatically reduces hallucination compared to open-domain generation.
"""

import asyncio
from typing import Optional

from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config


# ============================================================
# LLM Client Initialization
# ============================================================

def _get_gemini_model():
    """Initialize and return a Google Gemini generative model."""
    import google.generativeai as genai
    
    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Please set it in .env or as an environment variable."
        )
    
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    return model


# ============================================================
# Prompt Engineering
# ============================================================

SYSTEM_PROMPT = """Bạn là một trợ lý pháp lý chuyên nghiệp cho hệ thống pháp luật Việt Nam.

## QUY TẮC BẮT BUỘC (MANDATORY RULES):

1. **CHỈ trả lời dựa trên ngữ cảnh (context) được cung cấp bên dưới.** KHÔNG sử dụng kiến thức bên ngoài.
2. **Mọi khẳng định phải có trích dẫn** theo định dạng: [Nguồn: <tên văn bản> - <số ký hiệu>]
3. **Nếu ngữ cảnh không đủ thông tin để trả lời**, hãy nói rõ: "Dựa trên các tài liệu được truy xuất, tôi không tìm thấy đủ thông tin để trả lời chính xác câu hỏi này."
4. **KHÔNG bịa đặt hoặc suy đoán** thông tin không có trong ngữ cảnh.
5. Trả lời bằng **tiếng Việt**, rõ ràng, có cấu trúc.
6. Nếu có nhiều văn bản liên quan, **tổng hợp thông tin** từ tất cả các nguồn.

## ĐỊNH DẠNG TRẢ LỜI:
- Sử dụng đầu mục, danh sách nếu cần
- Trích dẫn nguồn cho mỗi điểm chính
- Kết thúc bằng phần "Nguồn tham khảo:" liệt kê các văn bản đã sử dụng
"""


def build_rag_prompt(query: str, retrieved_results: list[dict]) -> str:
    """
    Build the RAG prompt with retrieved context.
    
    The prompt structure:
      1. System instructions (anti-hallucination rules)
      2. Retrieved document contexts with metadata
      3. User question
    
    Args:
        query: The user's legal question.
        retrieved_results: List of retrieval results with text, score, and metadata.
    
    Returns:
        The complete prompt string for the LLM.
    """
    # Build context section from retrieved documents
    context_parts = []
    for i, result in enumerate(retrieved_results, 1):
        meta = result.get("metadata", {})
        title = meta.get("title", "Không có tiêu đề")
        doc_number = meta.get("doc_number", "N/A")
        doc_type = meta.get("doc_type", "N/A")
        date_issued = meta.get("date_issued", "N/A")
        authority = meta.get("authority", "N/A")
        score = result.get("score", 0.0)
        text = result.get("text", "")
        
        context_parts.append(
            f"--- TÀI LIỆU {i} (Độ liên quan: {score:.2f}) ---\n"
            f"Tiêu đề: {title}\n"
            f"Số ký hiệu: {doc_number}\n"
            f"Loại văn bản: {doc_type}\n"
            f"Ngày ban hành: {date_issued}\n"
            f"Cơ quan ban hành: {authority}\n"
            f"Nội dung:\n{text}\n"
        )
    
    context_block = "\n".join(context_parts)
    
    prompt = f"""{SYSTEM_PROMPT}

## NGỮ CẢNH (RETRIEVED CONTEXT):

{context_block}

## CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

## TRẢ LỜI:
Hãy trả lời câu hỏi trên CHỈ dựa trên ngữ cảnh đã cung cấp. Nhớ trích dẫn nguồn."""

    return prompt


# ============================================================
# RAG Pipeline
# ============================================================

class RAGPipeline:
    """
    End-to-end RAG pipeline: retrieval → prompt construction → LLM generation.
    
    This class orchestrates the full flow from a user query to a grounded,
    cited answer based on retrieved legal documents.
    """
    
    def __init__(self, retriever=None):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: A Retriever instance. If None, creates one from
                      the default Milvus index.
        """
        if retriever is None:
            from src.retriever import Retriever
            self.retriever = Retriever()
        else:
            self.retriever = retriever
        
        self.llm_model = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize the LLM client based on configuration."""
        try:
            if config.LLM_PROVIDER == "gemini":
                self.llm_model = _get_gemini_model()
                logger.info(f"LLM initialized: Gemini ({config.GEMINI_MODEL})")
            elif config.LLM_PROVIDER == "openai":
                import openai
                openai.api_key = config.OPENAI_API_KEY
                logger.info(f"LLM initialized: OpenAI ({config.OPENAI_MODEL})")
            else:
                raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}. RAG answers will be unavailable.")
            self.llm_model = None
    
    def _generate_gemini(self, prompt: str) -> str:
        """Generate a response using Google Gemini."""
        response = self.llm_model.generate_content(
            prompt,
            generation_config={
                "temperature": config.LLM_TEMPERATURE,
                "max_output_tokens": config.LLM_MAX_TOKENS,
            },
        )
        return response.text
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate a response using OpenAI."""
        import openai
        
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content
    
    def generate_answer(self, query: str, retrieved_results: list[dict]) -> str:
        """
        Generate an LLM answer grounded in retrieved context.
        
        Args:
            query: The user's question.
            retrieved_results: List of retrieval results.
        
        Returns:
            The LLM-generated answer string.
        """
        if not self.llm_model:
            return (
                "⚠️ LLM chưa được cấu hình. Vui lòng thiết lập API key trong file .env.\n"
                "(LLM not configured. Please set up the API key in .env file.)"
            )
        
        if not retrieved_results:
            return (
                "Không tìm thấy tài liệu liên quan trong cơ sở dữ liệu. "
                "Vui lòng thử lại với câu hỏi khác.\n"
                "(No relevant documents found. Please try a different question.)"
            )
        
        # Build the prompt
        prompt = build_rag_prompt(query, retrieved_results)
        
        # Generate response
        try:
            if config.LLM_PROVIDER == "gemini":
                answer = self._generate_gemini(prompt)
            elif config.LLM_PROVIDER == "openai":
                answer = self._generate_openai(prompt)
            else:
                answer = "Unsupported LLM provider."
            
            logger.info(f"Generated answer: {len(answer)} chars")
            return answer
        
        except Exception as e:
            error_str = str(e)
            logger.error(f"LLM generation failed: {error_str}")
            
            # Detect API key errors specifically
            if "API_KEY_INVALID" in error_str or "API key not valid" in error_str:
                return (
                    f"⚠️ Lỗi khi tạo câu trả lời: 400 API key not valid. "
                    f"Please pass a valid API key. [reason: \"API_KEY_INVALID\" "
                    f"domain: \"googleapis.com\" metadata {{ key: \"service\" value: "
                    f"\"generativelanguage.googleapis.com\" }}, locale: \"en-US\", "
                    f"message: \"API key not valid. Please pass a valid API key.\" ] "
                    f"Vui lòng kiểm tra API key và thử lại."
                )
            
            return (
                f"⚠️ Lỗi khi tạo câu trả lời: {error_str}\n"
                f"Vui lòng kiểm tra API key và thử lại."
            )
    
    def query(self, question: str, top_k: int = config.TOP_K) -> dict:
        """
        Full RAG pipeline: retrieve relevant docs → generate grounded answer.
        
        Args:
            question: The user's legal question.
            top_k: Number of documents to retrieve.
        
        Returns:
            Dict with keys:
              - question: Original question
              - answer: LLM-generated answer with citations
              - documents: List of retrieved document details
              - is_valid: Whether the query was valid
              - error: Error message if invalid
        """
        # Step 1: Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(question, top_k=top_k)
        
        if not retrieval_result["is_valid"]:
            return {
                "question": question,
                "answer": retrieval_result["error"],
                "documents": [],
                "is_valid": False,
                "error": retrieval_result["error"],
            }
        
        retrieved_docs = retrieval_result["results"]
        
        # Step 2: Generate answer using LLM
        answer = self.generate_answer(question, retrieved_docs)
        
        # Step 3: Format response
        documents = []
        for doc in retrieved_docs:
            documents.append({
                "chunk_id": doc["chunk_id"],
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "highlighted_text": doc["highlighted_text"],
                "score": doc["score"],
                "title": doc["metadata"].get("title", ""),
                "doc_number": doc["metadata"].get("doc_number", ""),
                "doc_type": doc["metadata"].get("doc_type", ""),
                "date_issued": doc["metadata"].get("date_issued", ""),
                "authority": doc["metadata"].get("authority", ""),
            })
        
        return {
            "question": question,
            "answer": answer,
            "documents": documents,
            "is_valid": True,
            "error": "",
        }
    
    async def query_async(self, question: str, top_k: int = config.TOP_K) -> dict:
        """
        Async version of query() for FastAPI endpoints.
        
        Args:
            question: The user's legal question.
            top_k: Number of documents to retrieve.
        
        Returns:
            Same structure as query().
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.query(question, top_k=top_k)
        )


# ============================================================
# CLI Entry Point (for standalone testing)
# ============================================================

if __name__ == "__main__":
    import sys
    
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "quy định về thuế thu nhập cá nhân"
    
    pipeline = RAGPipeline()
    result = pipeline.query(question)
    
    print(f"\n{'='*60}")
    print(f"Question: {result['question']}")
    print(f"\n{'='*60}")
    print(f"Answer:\n{result['answer']}")
    print(f"\n{'='*60}")
    print(f"Retrieved Documents ({len(result['documents'])}):")
    for i, doc in enumerate(result["documents"], 1):
        print(f"\n  [{i}] Score: {doc['score']:.4f}")
        print(f"      Title: {doc['title'][:80]}")
        print(f"      Number: {doc['doc_number']}")
        print(f"      Type: {doc['doc_type']}")
