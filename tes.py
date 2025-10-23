import os
import re
import glob
import pickle
import shutil
import time
from typing import List

import numpy as np
from numpy.linalg import norm
import streamlit as st

# LangChain core
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Loaders & splitters
from langchain_community.document_loaders import PyMuPDFLoader  # pip install pymupdf
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store: Chroma
from langchain_community.vectorstores import Chroma

# Retrievers
from langchain.retrievers import BM25Retriever, EnsembleRetriever, MultiQueryRetriever

# Gemini via LangChain
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="RAG • Gemini 2.5 Flash + Chroma", page_icon="📚", layout="wide")
st.markdown("### 📚 DUPAK AI")

# --- Secrets / ENV handling ---
# Priority: sidebar input -> st.secrets -> environment
SECRETS_API_KEY = st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, "secrets") else ""
GOOGLE_API_KEY_ENV = os.getenv("GOOGLE_API_KEY", SECRETS_API_KEY)

# Sidebar
GOOGLE_API_KEY = st.sidebar.text_input(
    "Masukkan GOOGLE_API_KEY (Google AI Studio):",
    value=GOOGLE_API_KEY_ENV,
    type="password",
)
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.sidebar.header("⚙️ Pengaturan")
pdf_dir = st.sidebar.text_input("Folder PDF lokal", value="./pdfs")
chroma_dir = st.sidebar.text_input("Folder Chroma persist", value="./chroma_store")
docs_dump = os.path.join(chroma_dir, "docs.pkl")  # cache dokumen untuk BM25
chunk_size = st.sidebar.slider("Ukuran chunk (karakter)", 500, 2500, 1000, 50)
chunk_overlap = st.sidebar.slider("Overlap", 0, 600, 200, 10)
top_k = st.sidebar.slider("Top-K retrieval", 3, 15, 10, 1)

# buat folder pdf_dir kalau belum ada
os.makedirs(pdf_dir, exist_ok=True)

# Buttons
with st.sidebar:
    build_btn = st.button("🔨 Build / Refresh Index", use_container_width=True)
    clear_btn = st.button("🧹 Clear Index", use_container_width=True)
    test_btn = st.button("🔍 Test Embedding", help="Cek apakah API key & model embedding bisa diakses.", use_container_width=True)

# =========================
# HELPERS
# =========================
def _normalize_text(s: str) -> str:
    if not s:
        return s
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\b(s\.?d\.?|sd|sampai|hingga|to)\b", "-", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\b(JAM|Jam)\b", "jam", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def list_pdfs(folder: str):
    return sorted(glob.glob(os.path.join(folder, "**/*.pdf"), recursive=True))

def load_and_split(pdfs, chunk_size=1000, overlap=250):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = []
    for p in pdfs:
        try:
            loader = PyMuPDFLoader(p)
            pages = loader.load()
        except Exception as e:
            st.warning(f"Gagal membaca PDF: {os.path.basename(p)} — {e}")
            continue
        for d in pages:
            d.page_content = _normalize_text(d.page_content or "")
        parts = splitter.split_documents(pages)
        # buang dokumen kosong & terlalu panjang (pangkas panjang ekstrim untuk embed stability)
        pruned = []
        for d in parts:
            if not d.page_content or not d.page_content.strip():
                continue
            if len(d.page_content) > 12000:
                d.page_content = d.page_content[:12000]
            pruned.append(d)
        docs.extend(pruned)
    return docs

# --- semantic similarity helper ---
def cosine(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    den = (norm(a) * norm(b)) + 1e-8
    return float(np.dot(a, b) / den)

def semantic_match(question: str, context: str, min_sim: float = 0.23) -> bool:
    if not question.strip() or not context.strip():
        return False
    emb = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=os.environ.get("GOOGLE_API_KEY", ""))
    q = emb.embed_query(question[:2000])
    c = emb.embed_query(context[:8000])
    return cosine(q, c) >= min_sim

# --- range variations ---
def range_variations(q: str):
    qn = _normalize_text(q)
    m = re.search(r"(\d{1,4})-(\d{1,4})", qn)
    vars = set()
    if m:
        a, b = m.group(1), m.group(2)
        vars.update([
            f"{a}-{b}", f"{a} sampai {b}", f"{a} s.d. {b}", f">={a} dan <={b}",
            f"{a} to {b}", f"{a} – {b}", f"{a} — {b}"
        ])
    return list(vars)

# =========================
# VECTORSTORE HELPERS (Chroma)
# =========================

def _batch(items: List, n: int):
    for i in range(0, len(items), n):
        yield items[i:i+n]

def build_chroma(docs, persist_dir):
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY belum diset (Secrets/ENV).")

    last_err = None
    for model_name in ["text-embedding-004", "models/text-embedding-004", "text-embedding-003"]:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
            vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            BATCH = 32
            for chunk in _batch(docs, BATCH):
                for attempt in range(4):
                    try:
                        # pruning safety sudah dilakukan saat load_and_split
                        vs.add_documents(chunk)
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(2 ** attempt)  # 1,2,4,8 detik
                else:
                    raise last_err
            vs.persist()
            return vs
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Gagal build Chroma (embedding): {last_err}")


def load_chroma(persist_dir):
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def save_docs_pickle(docs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(docs, f)


def load_docs_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# =========================
# INDEX MANAGEMENT
# =========================
if clear_btn:
    try:
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
        st.sidebar.success("Index Chroma & cache dokumen dibersihkan.")
    except Exception as e:
        st.sidebar.error(f"Gagal clear index: {e}")

if test_btn:
    try:
        emb = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
        )
        v = emb.embed_query("ping")
        st.sidebar.success(f"Embedding OK. Dim={len(v)}")
    except Exception as e:
        st.sidebar.error(f"Gagal embed: {e}")

if build_btn:
    pdfs = list_pdfs(pdf_dir)
    if not pdfs:
        st.sidebar.error(f"Tidak ditemukan PDF di: {pdf_dir}")
    elif not os.environ.get("GOOGLE_API_KEY"):
        st.sidebar.error("Set GOOGLE_API_KEY dulu (sidebar/Secrets).")
    else:
        with st.spinner("📥 Membaca PDF & membangun index Chroma + BM25..."):
            docs = load_and_split(pdfs, chunk_size, chunk_overlap)
            if not docs:
                st.sidebar.error("Tidak ada teks yang bisa diindeks dari PDF.")
            else:
                vs_temp = build_chroma(docs, chroma_dir)
                save_docs_pickle(docs, docs_dump)
                try:
                    count = vs_temp._collection.count()
                except Exception:
                    count = "?"
                st.sidebar.success(f"Index siap ✅ ({count} chunk tersimpan)")

# =========================
# PREP LLM, VECTORSTORE, RETRIEVERS
# =========================
has_index = os.path.exists(chroma_dir) and any(glob.glob(os.path.join(chroma_dir, "*")))
has_docs = os.path.exists(docs_dump)

vs = None
docs_for_bm25 = None
if has_index and os.environ.get("GOOGLE_API_KEY"):
    try:
        vs = load_chroma(chroma_dir)
    except Exception as e:
        st.error(f"Gagal memuat index: {e}")

if has_docs:
    try:
        docs_for_bm25 = load_docs_pickle(docs_dump)
    except Exception as e:
        st.error(f"Gagal memuat cache dokumen BM25: {e}")

llm = None
if os.environ.get("GOOGLE_API_KEY"):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=os.environ.get("GOOGLE_API_KEY", ""))
    except Exception as e:
        st.error(f"Gagal inisialisasi LLM: {e}")

# =========================
# PROMPTS
# =========================
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, concise assistant. Prefer concrete numbers when the user asks 'berapa'. "
            "When context is provided, answer ONLY using facts from that text. "
            "If the text is insufficient, do not mention the text; the app will decide to ignore it. "
            "Always respond in the SAME LANGUAGE as the user's question.",
        ),
        ("human", "Additional text:\n{context}\n\nUser question:\n{question}"),
    ]
)

fallback_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, concise assistant. Always respond in the same language as the user's question."),
        ("human", "{question}"),
    ]
)

rag_chain = (rag_prompt | llm | StrOutputParser()) if llm else None
fallback_chain = (fallback_prompt | llm | StrOutputParser()) if llm else None

# Context Gate — LLM memutuskan USE atau SKIP
gate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict gatekeeper. Decide if the provided text directly contains information to answer the question. "
            "Reply with only one word: USE (if the text clearly contains the answer) or SKIP (if not). Do NOT explain.",
        ),
        ("human", "Question:\n{question}\n\nText:\n{context}\n\nReply: USE or SKIP"),
    ]
)

gate_chain = (gate_prompt | llm | StrOutputParser()) if llm else None

# MultiQuery custom prompt (kuat di rentang angka)
mq_prompt = PromptTemplate.from_template(
    "Buat 4 variasi kueri yang semakna untuk mencari jawaban dari peraturan/PO tentang: \"{question}\". "
    "Jika ada angka rentang (mis. 81–160), tuliskan juga variasi: "
    "\"81-160\", \"81 sampai 160\", \"81 s.d. 160\", \">=81 dan <=160\". "
    "Jangan beri penjelasan; hanya daftar kueri (satu per baris)."
)

# =========================
# CHAT STATE & ACTIONS
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask DUPAK AI"}]
st.session_state.setdefault("debug_context", "")

col1, col2, _ = st.columns([1, 1, 6])

def _reset_chat(greeting: str = "Ask DUPAK AI"):
    st.session_state.messages = [{"role": "assistant", "content": greeting}]
    st.session_state.debug_context = ""
    st.rerun()

with col1:
    if st.button("🆕 New chat", use_container_width=True):
        _reset_chat()

with col2:
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.debug_context = ""
        st.rerun()

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================
# CHAT INPUT + RAG (Ensemble + MultiQuery + Range + Gate + Sim)
# =========================
user_input = st.chat_input("Ketik pertanyaanmu di sini…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    answer = "Maaf, model belum siap."
    context_text = ""

    if llm is None:
        answer = "Maaf, kunci API belum diisi."  # atau gagal init LLM
    else:
        use_rag = (vs is not None) and (docs_for_bm25 is not None)
        if use_rag:
            try:
                bm25 = BM25Retriever.from_documents(docs_for_bm25)
                vec_ret = vs.as_retriever(search_kwargs={"k": top_k})
                ens = EnsembleRetriever(retrievers=[vec_ret, bm25], weights=[0.5, 0.5])

                mq = MultiQueryRetriever.from_llm(
                    retriever=ens,
                    llm=llm,
                    include_original=True,
                    prompt=mq_prompt,
                )

                retrieved_docs = []
                # original
                retrieved_docs += ens.get_relevant_documents(user_input)
                # variasi range deterministik
                for v in range_variations(user_input):
                    retrieved_docs += ens.get_relevant_documents(v)
                # multiquery dari LLM
                retrieved_docs += mq.get_relevant_documents(user_input)

                # Dedup & gabungkan
                seen = set()
                merged_docs = []
                for d in retrieved_docs:
                    key = (d.metadata.get("source", ""), d.metadata.get("page", ""), hash(d.page_content))
                    if key not in seen:
                        merged_docs.append(d)
                        seen.add(key)
                    if len(merged_docs) >= top_k:
                        break

                if merged_docs:
                    context_text = "\n\n---\n\n".join(d.page_content for d in merged_docs)
                else:
                    use_rag = False
            except Exception as e:
                st.warning(f"Retriever error, fallback ke LLM: {e}")
                use_rag = False

        st.session_state["debug_context"] = context_text

        # Gating
        use_rag_flag = False
        if use_rag and context_text.strip():
            gate_ok = False
            if gate_chain is not None:
                try:
                    decision = gate_chain.invoke({"question": user_input, "context": context_text}).strip().upper()
                    gate_ok = decision.startswith("USE")
                except Exception:
                    gate_ok = False
            sem_ok = False
            try:
                sem_ok = semantic_match(user_input, context_text, min_sim=0.23)
            except Exception:
                sem_ok = False
            use_rag_flag = gate_ok and sem_ok
        else:
            use_rag_flag = False

        with st.chat_message("assistant"):
            with st.spinner("Sedang menyusun jawaban…"):
                try:
                    if use_rag_flag and rag_chain is not None:
                        answer = rag_chain.invoke({"context": context_text, "question": user_input})
                    elif fallback_chain is not None:
                        answer = fallback_chain.invoke({"question": user_input})
                    else:
                        answer = "Model LLM tidak tersedia. Pastikan GOOGLE_API_KEY sudah diisi."
                except Exception as e:
                    answer = f"Terjadi kesalahan saat menyusun jawaban: {e}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# # Debug context (opsional)
# with st.expander("🔎 Debug: potongan konteks yang dipakai"):
#     ctx = st.session_state.get("debug_context", "")
#     if ctx:
#         st.code(ctx[:3500])
#     else:
#         st.caption("Tidak ada konteks (fallback langsung ke Gemini atau belum ada pertanyaan).")
