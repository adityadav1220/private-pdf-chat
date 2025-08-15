# app.py — FastAPI backend for Privacy-First PDF QA (versioned-index dirs, chat, guardrails)

import os
import re
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional, Literal, TypedDict



from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
#from langchain.schema import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END


# ---------- Paths ----------
DOCS_DIR = Path("data/docs")
INDEX_ROOT = Path("data/index")                 # all versions live under here
ACTIVE_META = Path("data/active_index.json")    # pointer to the active index dir

DOCS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- PII patterns (regex) ----------
PII_PATTERNS = {
    "email": (
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "[REDACTED_EMAIL]",
    ),
    "phone": (
        re.compile(r"\+?\d[\d\-\s\(\)]{7,}\d"),
        "[REDACTED_PHONE]",
    ),
    "creditcard": (
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "[REDACTED_CC]",
    ),
    "ssn": (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
}

# ---------- Helpers ----------
def apply_redactions(text: str, enabled_types) -> tuple[str, dict]:
    counts = {}
    for t in enabled_types:
        pat, repl = PII_PATTERNS[t]
        text, n = pat.subn(repl, text)
        counts[t] = counts.get(t, 0) + n
    return text, counts

def load_docs():
    docs = []

    # PDFs
    for pdf_path in DOCS_DIR.glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()
        for p in pages:
            p.metadata["source_file"] = pdf_path.name
        docs.extend(pages)

    # .txt
    for txt_path in DOCS_DIR.glob("*.txt"):
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = txt_path.read_text(errors="ignore")
        docs.append(Document(page_content=text, metadata={"source_file": txt_path.name, "page": 0}))

    # .md
    for md_path in DOCS_DIR.glob("*.md"):
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = md_path.read_text(errors="ignore")
        docs.append(Document(page_content=text, metadata={"source_file": md_path.name, "page": 0}))

    return docs

def set_active_index_dir(dir_path: str):
    ACTIVE_META.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_META, "w") as f:
        json.dump({"persist_directory": dir_path}, f)

def get_active_index_dir() -> Optional[str]:
    if ACTIVE_META.exists():
        try:
            obj = json.load(open(ACTIVE_META))
            return obj.get("persist_directory")
        except Exception:
            return None
    # backward compat: if there is anything under INDEX_ROOT, default to root
    try:
        if any(INDEX_ROOT.iterdir()):
            return str(INDEX_ROOT.resolve())
    except FileNotFoundError:
        pass
    return None

def history_to_text(messages: list, max_turns: int = 6) -> str:
    """Flatten last few chat turns to plain text."""
    msgs = messages[-max_turns:]
    lines = []
    for m in msgs:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{'Assistant' if role=='assistant' else 'User'}: {content}")
    return "\n".join(lines)

def rewrite_query(q: str) -> str:
    q0 = (q or "").strip()
    ql = q0.lower()
    if len(ql.split()) <= 2:  # very short → expand
        if "duration" in ql:
            return "flight duration travel time total hours minutes"
        if "departure" in ql or "depart" in ql:
            return "departure time depart time scheduled departure"
        if "arrival" in ql or "arrive" in ql:
            return "arrival time arrive time scheduled arrival"
        if "flight" in ql and "number" in ql:
            return "flight number code aa flight 292 293"
        if "airport" in ql:
            return "airport origin destination city New York JFK Delhi DEL"
        if "seat" in ql:
            return "seat number seat assignment"
        if "class" in ql or "cabin" in ql:
            return "class cabin booking class economy"
        if "price" in ql or "total" in ql or "fare" in ql:
            return "total paid fare amount subtotal taxes fees INR"
    return q0  # otherwise keep as-is


# ---------- Build Index (no deletion; always new versioned dir) ----------
def build_index(
    redact_on: bool = True,
    types: Optional[List[str]] = None,
    reset: bool = False,         # accepted but ignored: we never delete
    chunk_size: int = 1200,
    chunk_overlap: int = 120,
):
    docs = load_docs()
    if not docs:
        return {"ok": False, "message": "No PDFs found in data/docs."}

    enabled = [t for t in (types or []) if t in PII_PATTERNS]
    totals = {t: 0 for t in enabled}

    # Apply PII redaction (in-place)
    if redact_on and enabled:
        for d in docs:
            redacted_text, counts = apply_redactions(d.page_content, enabled)
            d.page_content = redacted_text
            for t, n in counts.items():
                totals[t] += n

    # Keep a page-level corpus (after redaction) for lexical BM25
    corpus_rows = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]

    # Page-aware splitting for vector index
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for d in docs:
        if len(d.page_content) <= 1500:
            chunks.append(d)
        else:
            chunks.extend(splitter.split_documents([d]))

    # Brand-new directory for this build
    build_id = f"docqa_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    new_dir = str((INDEX_ROOT / build_id).resolve())
    os.makedirs(new_dir, exist_ok=True)

    # Vector index
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=new_dir,
    )

    # Save BM25 corpus
    with open(os.path.join(new_dir, "corpus.jsonl"), "w") as f:
        for row in corpus_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Flip pointer
    set_active_index_dir(new_dir)

    return {
        "ok": True,
        "chunks": len(chunks),
        "redactions": totals,
        "redact_on": redact_on,
        "persist_directory": new_dir,
    }

# ---------- LangGraph pipeline (numbered refs) ----------
class QAState(TypedDict):
    question: str
    history: str
    docs: List
    context_text: str
    raw_answer: str
    parsed: Optional[dict]
    status: Literal["init","answered","refused","bad_json","no_docs","weak_evidence"]

def load_retriever():
    active_dir = get_active_index_dir()
    if not active_dir:
        raise RuntimeError("No active index. Build the index first.")
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    vs = Chroma(embedding_function=embedder, persist_directory=active_dir)
    # Diversify and grab more candidates
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 24, "lambda_mult": 0.5},
    )

def format_context_with_refs(docs) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page", 0)
        src  = d.metadata.get("source_file", "document")
        snippet = d.page_content.strip().replace("\n", " ")
        snippet = (snippet[:350] + "…") if len(snippet) > 350 else snippet
        lines.append(f"[{i}] {src} p.{page+1}: {snippet}")
    return "\n\n".join(lines)

SYSTEM_PROMPT = """You are a careful assistant.
ONLY answer using the provided context. If the context does not contain the answer,
return an empty string for "answer" and an empty list for "evidence".
If there are multiple valid answers (e.g., outbound and return), list them all clearly.
Output JSON ONLY, no prose.

Always cite evidence using the bracketed reference NUMBERS shown before each context line.
You may include a short exact quote (<=160 chars), but the "ref" number is REQUIRED.

JSON schema:
{{
  "answer": "string",  // 1-3 sentences or "" if unknown
  "evidence": [        // 1-5 items
    {{ "ref": number, "quote": "string" }}
  ]
}}
"""

USER_PROMPT = """Conversation so far:
{history}

Question: {question}

Context with numbered references:
{context}

Return ONLY valid JSON that matches the schema exactly.
- Use the "ref" numbers from the context (e.g., 1, 2, 3).
- If the answer is not supported by the context, return empty answer and empty evidence.
"""

def build_graph():
    def retrieve_node(state: QAState) -> QAState:
        retriever = load_retriever()
        expanded = rewrite_query(state["question"])
        vdocs = retriever.invoke(expanded)
        bm_docs = []
        active_dir = get_active_index_dir()
        corpus_path = Path(active_dir) / "corpus.jsonl" if active_dir else None
        if corpus_path and corpus_path.exists():
            with open(corpus_path, "r") as f:
                for line in f:
                    row = json.loads(line)
                    bm_docs.append(Document(page_content=row["page_content"], metadata=row.get("metadata", {})))
            bm25 = BM25Retriever.from_documents(bm_docs)
            ldocs = bm25.invoke(expanded)[:10]  # top lexical hits
        else:
            ldocs = []

        # 3) merge (dedupe by (file,page,first80chars))
        seen = set()
        merged = []
        for d in (vdocs + ldocs):
            key = (d.metadata.get("source_file"), d.metadata.get("page"), d.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)

        docs = merged[:12]
        if not docs:
            return {**state, "docs": [], "status": "no_docs"}
        return {**state, "docs": docs, "context_text": format_context_with_refs(docs)}

    def generate_node(state: QAState) -> QAState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ]).format_messages(
            context=state["context_text"],
            question=state["question"],
            history=state["history"],
        )
        llm = ChatOllama(model="llama3.1:8b", temperature=0.1)
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.DOTALL)
        try:
            parsed = json.loads(raw)
        except Exception:
            return {**state, "raw_answer": raw, "status": "bad_json", "parsed": None}
        return {**state, "raw_answer": raw, "parsed": parsed, "status": "answered"}

    def verify_node(state: QAState) -> QAState:
        if state.get("status") in ("no_docs", "bad_json"):
            return state
        parsed = state.get("parsed") or {}
        answer = (parsed.get("answer") or "").strip()
        evidence = parsed.get("evidence") or []
        if not answer or not evidence:
            return {**state, "status": "weak_evidence"}
        k = len(state.get("docs", []))
        valid_refs = sum(1 for ev in evidence[:5] if isinstance(ev.get("ref"), int) and 1 <= ev["ref"] <= k)
        if valid_refs == 0:
            return {**state, "status": "weak_evidence"}
        return state

    def router(state: QAState) -> str:
        status = state.get("status")
        if status in ("no_docs", "bad_json", "weak_evidence"):
            return "refuse"
        return "finalize"

    def refuse_node(state: QAState) -> QAState:
        return {**state, "parsed": {"answer": "", "evidence": []}, "status": "refused"}

    def finalize_node(state: QAState) -> QAState:
        return state

    graph = StateGraph(QAState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify",   verify_node)
    graph.add_node("refuse",   refuse_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", router, {"refuse": "refuse", "finalize": "finalize"})
    graph.add_edge("refuse", END)
    graph.add_edge("finalize", END)

    return graph.compile()

GRAPH_APP = build_graph()

# ---------- FastAPI app ----------
app = FastAPI(title="Privacy-First PDF QA", version="0.1.0")

@app.get("/health")
def health():
    return {"ok": True, "message": "ready"}

@app.get("/status")
def status_endpoint():
    return {"ok": True, "active_persist_directory": get_active_index_dir()}

@app.post("/index")
async def index_endpoint(
    files: Optional[List[UploadFile]] = File(default=None),
    redact: bool = Form(default=True),
    types: str = Form(default="email,phone,creditcard,ssn"),
    reset: bool = Form(default=False),     # accepted but ignored
):
    try:
        saved = []
        if files:
            for f in files:
                if not f.filename.lower().endswith(".pdf"):
                    return JSONResponse(status_code=400, content={"ok": False, "message": f"Not a PDF: {f.filename}"})
                dest = DOCS_DIR / f.filename
                content = await f.read()
                with open(dest, "wb") as out:
                    out.write(content)
                saved.append(f.filename)

        enabled_types = [t.strip() for t in types.split(",") if t.strip() in PII_PATTERNS]
        result = build_index(redact_on=redact, types=enabled_types, reset=reset)

        return {"ok": result.get("ok", False), "saved": saved, **result}

    except Exception as e:
        print("INDEX ERROR:", repr(e))
        return JSONResponse(status_code=500, content={"ok": False, "message": f"index failed: {e.__class__.__name__}: {e}"})

@app.post("/ask")
def ask_endpoint(payload: dict = Body(...)):
    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse(status_code=400, content={"ok": False, "message": "question is required"})
    state: QAState = {
        "question": question,
        "history": "",
        "docs": [],
        "context_text": "",
        "raw_answer": "",
        "parsed": None,
        "status": "init"
    }
    try:
        result = GRAPH_APP.invoke(state)
    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"ok": False, "message": str(e)})
    status = result.get("status")
    parsed = result.get("parsed") or {"answer": "", "evidence": []}
    refused = status in ("refused", "no_docs", "weak_evidence")
    return {"ok": True, "refused": refused, "status": status, "answer": parsed["answer"], "evidence": parsed["evidence"]}

@app.post("/chat")
def chat_endpoint(payload: dict = Body(...)):
    messages = payload.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return JSONResponse(status_code=400, content={"ok": False, "message": "messages list is required"})
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if not last_user or not (last_user.get("content") or "").strip():
        return JSONResponse(status_code=400, content={"ok": False, "message": "last user message required"})
    question = last_user["content"].strip()
    history_text = history_to_text(messages[:-1])
    state: QAState = {
        "question": question,
        "history": history_text,
        "docs": [],
        "context_text": "",
        "raw_answer": "",
        "parsed": None,
        "status": "init"
    }
    try:
        result = GRAPH_APP.invoke(state)
    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"ok": False, "message": str(e)})
    status = result.get("status")
    parsed = result.get("parsed") or {"answer": "", "evidence": []}
    refused = status in ("refused", "no_docs", "weak_evidence")
    return {"ok": True, "refused": refused, "status": status, "answer": parsed["answer"], "evidence": parsed["evidence"]}
