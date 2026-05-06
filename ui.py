"""
ui.py - Streamlit frontend for the Vietnamese Legal Assistant RAG System.

Features:
  - Login / Register / Logout (PostgreSQL-backed)
  - Persistent chat history across sessions
  - Chat interface (left panel) for conversational Q&A
  - Evidence panel (right panel) with similarity scores & highlighted snippets
  - Comparison tab: Milvus ANN vs PostgreSQL ILIKE vs PostgreSQL FTS
  - Dynamic Top-K slider (max = index size)

Run with: streamlit run ui.py
"""

import re
import time

import pandas as pd
import requests
import streamlit as st

import config

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Trợ Lý Pháp Lý Việt Nam | Vietnamese Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS for Premium UI
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.8; font-size: 0.95rem; }

    .doc-card {
        background: linear-gradient(145deg, #f8f9fa, #ffffff);
        border: 1px solid #e0e6ed; border-radius: 10px;
        padding: 1.2rem; margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .doc-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }

    .score-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-weight: 600; font-size: 0.85rem; color: white;
    }
    .score-high   { background: linear-gradient(135deg, #00b09b, #96c93d); }
    .score-medium { background: linear-gradient(135deg, #f7971e, #ffd200); }
    .score-low    { background: linear-gradient(135deg, #eb3349, #f45c43); }

    .meta-tag {
        display: inline-block; background: #e8f4f8; color: #1a6b8a;
        padding: 3px 10px; border-radius: 15px; font-size: 0.78rem;
        margin: 2px 4px 2px 0; font-weight: 500;
    }
    .highlight-text {
        background: linear-gradient(120deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 2px 4px; border-radius: 3px;
    }
    .stat-card {
        background: white; border-radius: 10px; padding: 1rem;
        text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #eee;
    }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #0f3460; }
    .stat-label { font-size: 0.8rem; color: #666; margin-top: 4px; }

    .login-box {
        max-width: 420px; margin: 3rem auto; padding: 2rem;
        background: white; border-radius: 14px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    .session-btn {
        text-align: left; font-size: 0.82rem; padding: 6px 10px;
        border-radius: 8px; margin-bottom: 4px; width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Helper Functions
# ============================================================

def get_score_class(score: float) -> str:
    if score >= 0.7:
        return "score-high"
    elif score >= 0.4:
        return "score-medium"
    return "score-low"


def format_highlighted_text(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r'<span class="highlight-text">\1</span>', text)


def query_api(question: str, top_k: int = 3) -> dict | None:
    try:
        resp = requests.post(
            f"{config.API_BASE_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Không thể kết nối API backend. Sử dụng chế độ cục bộ.")
        return query_local_fallback(question, top_k=top_k)
    except requests.exceptions.Timeout:
        st.error("⚠️ Yêu cầu hết thời gian chờ.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"⚠️ API lỗi HTTP {e.response.status_code if e.response else 'N/A'}.")
        return None
    except Exception as e:
        st.error(f"⚠️ Lỗi: {e}")
        return None


@st.cache_resource(show_spinner=False)
def get_local_pipeline():
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline()


def query_local_fallback(question: str, top_k: int = 3) -> dict | None:
    try:
        pipeline = get_local_pipeline()
        start = time.perf_counter()
        result = pipeline.query(question=question, top_k=top_k)
        result["processing_time_ms"] = (time.perf_counter() - start) * 1000
        return result
    except Exception as e:
        st.error(f"⚠️ Local fallback thất bại: {e}")
        return None


def check_api_health() -> dict | None:
    try:
        resp = requests.get(f"{config.API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _get_pg_auth_status() -> tuple[str, str]:
    """
    Check PostgreSQL auth status.

    Returns:
        ("ready", "") when users/auth tables are accessible.
        ("missing_schema", message) when DB is reachable but auth tables are missing.
        ("unavailable", message) when DB cannot be reached.
    """
    conn = None
    try:
        from postgres.db import get_connection
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users LIMIT 1")
        return "ready", ""
    except Exception as e:
        err = str(e)
        low = err.lower()
        if "relation \"users\" does not exist" in low or ("users" in low and "does not exist" in low):
            return "missing_schema", err
        return "unavailable", err
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _init_pg_auth_schema() -> bool:
    """Create auth/chat/log tables on demand."""
    try:
        from postgres.setup_tables import setup
        setup()
        return True
    except Exception as e:
        st.error(f"⚠️ Không thể khởi tạo bảng đăng nhập: {e}")
        return False


# ============================================================
# Auth UI
# ============================================================

def render_login_page():
    """Show login / register form.  Returns True if user becomes logged-in."""
    st.markdown("""
    <div class="main-header" style="text-align:center;">
        <h1>⚖️ Trợ Lý Pháp Lý Việt Nam</h1>
        <p>Đăng nhập để lưu lịch sử trò chuyện</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        tab_login, tab_reg = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])

        with tab_login:
            with st.form("login_form"):
                uname = st.text_input("Tên đăng nhập", key="login_u")
                pwd = st.text_input("Mật khẩu", type="password", key="login_p")
                submitted = st.form_submit_button("Đăng nhập", use_container_width=True)
            if submitted and uname and pwd:
                from postgres.auth import login_user
                login_error = None
                try:
                    user = login_user(uname, pwd)
                except Exception as e:
                    login_error = str(e)
                    user = None
                if login_error:
                    st.error(f"Không thể đăng nhập: {login_error}")
                elif user:
                    st.session_state.user = user
                    try:
                        from postgres.chat_store import log_access
                        log_access(user["id"], "login")
                    except Exception:
                        pass
                    st.rerun()
                else:
                    st.error("Sai tên đăng nhập hoặc mật khẩu.")

        with tab_reg:
            with st.form("reg_form"):
                new_u = st.text_input("Tên đăng nhập", key="reg_u")
                new_d = st.text_input("Tên hiển thị", key="reg_d")
                new_p = st.text_input("Mật khẩu", type="password", key="reg_p")
                new_p2 = st.text_input("Xác nhận mật khẩu", type="password", key="reg_p2")
                submitted2 = st.form_submit_button("Đăng ký", use_container_width=True)
            if submitted2:
                if not new_u or not new_p:
                    st.error("Vui lòng nhập đầy đủ.")
                elif new_p != new_p2:
                    st.error("Mật khẩu không khớp.")
                elif len(new_p) < 4:
                    st.error("Mật khẩu tối thiểu 4 ký tự.")
                else:
                    from postgres.auth import register_user
                    register_error = None
                    try:
                        user = register_user(new_u, new_p, display_name=new_d)
                    except Exception as e:
                        register_error = str(e)
                        user = None
                    if register_error:
                        st.error(f"Không thể đăng ký: {register_error}")
                    elif user:
                        st.success("Đăng ký thành công! Hãy đăng nhập.")
                    else:
                        st.error("Tên đăng nhập đã tồn tại.")

        st.divider()
        if st.button("🚀 Tiếp tục không đăng nhập (Guest)", use_container_width=True):
            st.session_state.user = {"id": None, "username": "guest", "display_name": "Guest"}
            st.rerun()


# ============================================================
# Sidebar
# ============================================================

def render_sidebar() -> int:
    with st.sidebar:
        user = st.session_state.get("user", {})
        pg_status, pg_error = _get_pg_auth_status()
        st.markdown(f"### 👤 {user.get('display_name', 'Guest')}")

        if user.get("id"):
            if st.button("🚪 Đăng xuất", use_container_width=True):
                try:
                    from postgres.chat_store import log_access
                    log_access(user["id"], "logout")
                except Exception:
                    pass
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
        else:
            if pg_status == "ready":
                if st.button("🔐 Đăng nhập / Đăng ký", use_container_width=True):
                    st.session_state.pop("user", None)
                    st.rerun()
            elif pg_status == "missing_schema":
                st.warning("⚠️ PostgreSQL đã chạy nhưng chưa có bảng đăng nhập.")
                if st.button("🛠️ Tạo bảng đăng nhập", use_container_width=True):
                    with st.spinner("Đang tạo bảng users/chat/logs..."):
                        if _init_pg_auth_schema():
                            st.success("Đã tạo bảng thành công. Đang mở màn hình đăng nhập...")
                            st.session_state.pop("user", None)
                            st.rerun()
            else:
                st.info("ℹ️ PostgreSQL chưa sẵn sàng. Chạy `docker-compose up -d` để bật DB.")
                if pg_error:
                    st.caption(pg_error[:160])

        st.divider()

        # -- Session management (logged-in only) --
        if user.get("id"):
            st.markdown("#### 💬 Phiên chat")
            if st.button("➕ Phiên chat mới", use_container_width=True):
                from postgres.chat_store import create_session
                sid = create_session(user["id"])
                st.session_state.active_session = sid
                st.session_state.messages = []
                st.session_state.retrieved_docs = []
                st.rerun()

            from postgres.chat_store import get_sessions
            sessions = get_sessions(user["id"], limit=20)
            active = st.session_state.get("active_session")
            for s in sessions:
                label = f"{'▶ ' if s['id'] == active else ''}{s['title'][:35]}"
                if st.button(label, key=f"sess_{s['id']}", use_container_width=True):
                    st.session_state.active_session = s["id"]
                    from postgres.chat_store import get_messages
                    msgs = get_messages(s["id"])
                    st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in msgs]
                    st.session_state.retrieved_docs = msgs[-1].get("documents", []) if msgs else []
                    st.rerun()
            st.divider()

        # -- API health --
        health = check_api_health()
        if health:
            index_size = int(health.get("index_size", 0) or 0)
            st.success(f"✅ API Online — {index_size:,} chunks")
        else:
            st.error("❌ API Offline")

        # Keep Top-K stable for UI/UX and avoid Streamlit min=max slider errors.
        top_k = st.slider("Số tài liệu truy xuất (Top-K)", min_value=1, max_value=10, value=3)

        st.divider()
        st.markdown("### 📊 Hệ thống")
        st.markdown("- **Embedding**: MiniLM-L12-v2\n- **Vector DB**: Milvus (IVF_FLAT)\n- **LLM**: Google Gemini\n- **Metric**: Cosine Similarity")

        st.divider()
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.retrieved_docs = []
            st.rerun()

        st.divider()
        st.markdown("### 💡 Câu hỏi mẫu")
        for q in [
            "Quy định về thuế thu nhập cá nhân",
            "Điều kiện thành lập doanh nghiệp",
            "Quyền và nghĩa vụ của người lao động",
        ]:
            if st.button(f"📌 {q}", key=f"sample_{q}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()

        return top_k


# ============================================================
# Document Card
# ============================================================

def render_document_card(doc: dict, index: int):
    score = doc.get("score", 0.0)
    title = doc.get("title", "Không có tiêu đề")
    doc_number = doc.get("doc_number", "N/A")
    doc_type = doc.get("doc_type", "N/A")
    date_issued = doc.get("date_issued", "N/A")
    authority = doc.get("authority", "N/A")
    highlighted = doc.get("highlighted_text", doc.get("text", ""))
    full_text = doc.get("text", "")
    formatted_highlight = format_highlighted_text(highlighted)

    st.markdown(f"""
    <div class="doc-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <strong style="font-size:0.9rem;color:#333;">📄 Tài liệu #{index + 1}</strong>
            <span class="score-badge {get_score_class(score)}">🎯 {score:.2%}</span>
        </div>
        <div style="font-weight:600;color:#1a1a2e;margin-bottom:6px;font-size:0.88rem;">{title[:120]}</div>
        <div style="margin-bottom:8px;">
            <span class="meta-tag">📋 {doc_type}</span>
            <span class="meta-tag">📝 {doc_number}</span>
            <span class="meta-tag">📅 {date_issued}</span>
        </div>
        <div style="margin-bottom:6px;"><span class="meta-tag">🏛️ {authority[:60]}</span></div>
        <div style="font-size:0.82rem;color:#555;line-height:1.5;border-left:3px solid #0f3460;padding-left:10px;margin-top:8px;">
            {formatted_highlight[:400]}{'...' if len(formatted_highlight) > 400 else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"📖 Xem toàn bộ nội dung — {doc_number}"):
        st.text(full_text)


# ============================================================
# Comparison Tab
# ============================================================

def render_comparison_tab(top_k: int):
    st.markdown("### 📊 So sánh Milvus ANN vs PostgreSQL")
    st.info(
        "Nhập câu hỏi rồi nhấn **So sánh** để chạy cùng truy vấn qua 3 phương pháp: "
        "**Milvus ANN** (semantic), **PG ILIKE** (keyword), **PG FTS** (full-text)."
    )

    cmp_query = st.text_input("Câu truy vấn so sánh", value="quy định về thuế thu nhập cá nhân", key="cmp_q")
    cmp_k = st.number_input("Top-K", min_value=1, max_value=top_k, value=min(3, top_k), key="cmp_k")

    if st.button("🔍 So sánh", use_container_width=True):
        with st.spinner("Đang chạy 3 phương pháp tìm kiếm..."):
            try:
                resp = requests.post(
                    f"{config.API_BASE_URL}/compare",
                    json={"question": cmp_query, "top_k": cmp_k},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"Lỗi khi gọi API /compare: {e}")
                return

        methods = data.get("methods", [])
        if not methods:
            st.warning("Không có kết quả.")
            return

        # ── Latency bar chart ──
        st.markdown("#### ⏱️ Thời gian truy vấn (ms)")
        st.caption("Thời gian mỗi phương pháp cần để trả về kết quả. Bao gồm kết nối DB, embedding (Milvus), và tìm kiếm.")
        chart_data = pd.DataFrame({
            "Phương pháp": [m["method"][:35] for m in methods],
            "Thời gian (ms)": [m["time_ms"] for m in methods],
        }).set_index("Phương pháp")
        st.bar_chart(chart_data)

        # ── Result count ──
        st.markdown("#### 📊 Số kết quả trả về")
        cols = st.columns(len(methods))
        for i, m in enumerate(methods):
            with cols[i]:
                st.metric(m["method"][:25], f"{len(m['results'])} tài liệu", delta=f"{m['time_ms']:.0f}ms")

        # ── Overlap ──
        overlap = data.get("overlap", {})
        if overlap:
            st.markdown("#### 🔀 Overlap (Milvus ↔ PostgreSQL)")
            st.caption("Số chunk trùng nhau giữa Milvus và mỗi phương pháp PostgreSQL. Overlap thấp → hai hệ thống trả kết quả khác nhau (semantic vs keyword).")
            for method_name, info in overlap.items():
                c1, c2, c3 = st.columns(3)
                c1.metric("Chung", info["common"])
                c2.metric("Chỉ Milvus", info["only_milvus"])
                c3.metric(f"Chỉ PG", info["only_pg"])

        # ── Detailed results per method ──
        st.markdown("#### 📄 Kết quả chi tiết từng phương pháp")
        tabs = st.tabs([m["method"][:30] for m in methods])
        for idx, m in enumerate(methods):
            with tabs[idx]:
                st.caption(_method_explanation(m["method"]))
                if not m["results"]:
                    st.warning("Không tìm thấy kết quả.")
                    continue
                for j, r in enumerate(m["results"], 1):
                    score_str = f"{r['score']:.4f}" if r.get("score") is not None else "N/A"
                    st.markdown(f"**#{j}** | Score: `{score_str}` | {r.get('doc_number','')} — {r.get('title','')[:80]}")
                    st.text(r.get("text_preview", "")[:300])

        # ── Summary insight ──
        st.markdown("#### 💡 Nhận xét")
        st.markdown(
            "- **Milvus ANN (semantic)**: Tìm theo *ngữ nghĩa*, hiểu đồng nghĩa, paraphrase. "
            "Ví dụ: \"đất đai\" ≈ \"bất động sản\".\n"
            "- **PG ILIKE (keyword)**: Tìm *chuỗi con chính xác*. Nhanh nhưng không hiểu ngữ nghĩa.\n"
            "- **PG FTS (tsvector)**: Tìm *từ khóa* + xếp hạng theo tần suất. Tốt hơn ILIKE nhưng "
            "vẫn không hiểu đồng nghĩa."
        )


def _method_explanation(method: str) -> str:
    if "ILIKE" in method:
        return "ILIKE tìm kiếm chuỗi con (substring) trong nội dung văn bản. Yêu cầu TẤT CẢ từ khóa xuất hiện. Không có điểm tương đồng (score = N/A)."
    if "Full-Text" in method or "tsvector" in method:
        return "Full-Text Search dùng tsvector/tsquery để tìm từ khóa và xếp hạng theo ts_rank. Có điểm rank nhưng không hiểu ngữ nghĩa."
    return "Milvus ANN dùng embedding vector (384 chiều) và IVF_FLAT index với Cosine Similarity. Tìm theo ngữ nghĩa — hiểu đồng nghĩa và paraphrase."


# ============================================================
# Main Application
# ============================================================

def main():
    # -- Init session state --
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retrieved_docs" not in st.session_state:
        st.session_state.retrieved_docs = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # -- Auth gate --
    pg_status, _ = _get_pg_auth_status()
    if "user" not in st.session_state:
        if pg_status == "ready":
            render_login_page()
            return
        else:
            st.session_state.user = {"id": None, "username": "guest", "display_name": "Guest"}

    user = st.session_state.user

    # Auto-create first session for logged-in users
    if user.get("id") and "active_session" not in st.session_state:
        from postgres.chat_store import get_sessions, create_session
        sessions = get_sessions(user["id"], limit=1)
        if sessions:
            st.session_state.active_session = sessions[0]["id"]
            from postgres.chat_store import get_messages
            msgs = get_messages(sessions[0]["id"])
            st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in msgs]
            st.session_state.retrieved_docs = msgs[-1].get("documents", []) if msgs else []
        else:
            st.session_state.active_session = create_session(user["id"])

    # -- Header --
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Trợ Lý Pháp Lý Việt Nam</h1>
        <p>Vietnamese Legal Assistant — RAG (Retrieval-Augmented Generation)</p>
    </div>
    """, unsafe_allow_html=True)

    top_k = render_sidebar()

    # -- Tabs --
    tab_chat, tab_compare = st.tabs(["💬 Hỏi đáp pháp luật", "📊 So sánh Milvus vs PostgreSQL"])

    # ── Chat Tab ──────────────────────────────────────────────
    with tab_chat:
        chat_col, evidence_col = st.columns([3, 2])

        with chat_col:
            st.markdown("### 💬 Hỏi đáp pháp luật")
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "⚖️"):
                    st.markdown(msg["content"])

            pending = st.session_state.pending_query
            if pending:
                st.session_state.pending_query = None
                prompt = pending
            else:
                prompt = st.chat_input("Nhập câu hỏi pháp lý của bạn...")

            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.markdown(prompt)

                # Persist user message
                if user.get("id") and st.session_state.get("active_session"):
                    from postgres.chat_store import add_message, update_session_title, log_access
                    add_message(st.session_state.active_session, "user", prompt)
                    if len(st.session_state.messages) == 1:
                        update_session_title(st.session_state.active_session, prompt[:60])
                    log_access(user["id"], "query", {"question": prompt[:200]})

                with st.chat_message("assistant", avatar="⚖️"):
                    with st.spinner("🔍 Đang tìm kiếm và phân tích..."):
                        result = query_api(prompt, top_k=top_k)

                    if result and result.get("is_valid"):
                        answer = result["answer"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.retrieved_docs = result.get("documents", [])

                        proc_time = result.get("processing_time_ms", 0)
                        if proc_time > 0:
                            st.caption(f"⏱️ {proc_time:.0f}ms")

                        # Persist assistant message
                        if user.get("id") and st.session_state.get("active_session"):
                            from postgres.chat_store import add_message
                            add_message(
                                st.session_state.active_session, "assistant", answer,
                                documents=result.get("documents", []),
                                processing_time_ms=proc_time,
                            )
                    elif result:
                        error_msg = result.get("error", "Unknown error")
                        st.warning(f"⚠️ {error_msg}")
                        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {error_msg}"})

                st.rerun()

        with evidence_col:
            st.markdown("### 📚 Tài liệu tham khảo")
            if st.session_state.retrieved_docs:
                docs = st.session_state.retrieved_docs
                scores = [d.get("score", 0) for d in docs]
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="stat-card"><div class="stat-value">{len(docs)}</div><div class="stat-label">Tài liệu</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="stat-card"><div class="stat-value">{max(scores):.2f}</div><div class="stat-label">Điểm cao nhất</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="stat-card"><div class="stat-value">{sum(scores)/len(scores):.2f}</div><div class="stat-label">Trung bình</div></div>', unsafe_allow_html=True)
                st.markdown("")
                for i, doc in enumerate(docs):
                    render_document_card(doc, i)
            else:
                st.info("📝 Hãy đặt câu hỏi để xem tài liệu pháp luật liên quan.")

    # ── Comparison Tab ────────────────────────────────────────
    with tab_compare:
        render_comparison_tab(top_k)


if __name__ == "__main__":
    main()
