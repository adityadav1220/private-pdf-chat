# graph_ask.py — LangGraph pipeline with numbered refs:
# retrieve -> generate(JSON with refs) -> verify(refs) -> refuse/finalize

import json
import re
from typing import List, TypedDict, Literal, Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

INDEX_DIR = "data/index"

# --------- State for the graph ----------
class QAState(TypedDict):
    question: str
    docs: List  # retrieved docs
    context_text: str
    raw_answer: str
    parsed: Optional[dict]
    status: Literal["init", "answered", "refused", "bad_json", "no_docs", "weak_evidence"]

# --------- Utilities ----------
def load_retriever():
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    vs = Chroma(embedding_function=embedder, persist_directory=INDEX_DIR)
    # pull a few chunks to be safe
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": 8})

def format_context_with_refs(docs) -> str:
    """
    Produce numbered context entries like:
    [1] sample.pdf p.1: <snippet>
    [2] sample.pdf p.1: <snippet>
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page", 0)  # 0-based from PyMuPDFLoader
        src  = d.metadata.get("source_file", "document")
        snippet = d.page_content.strip().replace("\n", " ")
        snippet = (snippet[:350] + "…") if len(snippet) > 350 else snippet
        lines.append(f"[{i}] {src} p.{page+1}: {snippet}")
    return "\n\n".join(lines)

# JSON-only output with numbered refs
SYSTEM_PROMPT = """You are a careful assistant.
ONLY answer using the provided context. If the context does not contain the answer,
return an empty string for "answer" and an empty list for "evidence".
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

USER_PROMPT = """Question: {question}

Context with numbered references:
{context}

Return ONLY valid JSON that matches the schema exactly.
- Use the "ref" numbers from the context (e.g., 1, 2, 3).
- If the answer is not supported by the context, return empty answer and empty evidence.
"""

# --------- Graph nodes ----------
def retrieve_node(state: QAState) -> QAState:
    retriever = load_retriever()
    docs = retriever.invoke(state["question"])
    if not docs:
        return {**state, "docs": [], "status": "no_docs"}
    return {**state, "docs": docs, "context_text": format_context_with_refs(docs)}

def generate_node(state: QAState) -> QAState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]).format_messages(
        context=state["context_text"],
        question=state["question"]
    )
    llm = ChatOllama(model="llama3.1:8b", temperature=0.1)
    resp = llm.invoke(prompt)
    raw = resp.content.strip()
    # Strip code fences if present
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.DOTALL)
    try:
        parsed = json.loads(raw)
    except Exception:
        return {**state, "raw_answer": raw, "status": "bad_json", "parsed": None}
    return {**state, "raw_answer": raw, "parsed": parsed, "status": "answered"}

def verify_node(state: QAState) -> QAState:
    """
    Gate the answer using numbered refs.
    - If no docs or bad JSON -> keep status.
    - Require at least 1 valid ref within [1..len(docs)].
    - (Optional) If a quote is provided, lightly sanity-check it.
    """
    if state.get("status") in ("no_docs", "bad_json"):
        return state

    parsed = state.get("parsed") or {}
    answer = (parsed.get("answer") or "").strip()
    evidence = parsed.get("evidence") or []
    if not answer or not evidence:
        return {**state, "status": "weak_evidence"}

    k = len(state.get("docs", []))
    valid_refs = 0
    for ev in evidence[:5]:
        ref = ev.get("ref")
        if isinstance(ref, int) and 1 <= ref <= k:
            valid_refs += 1

    if valid_refs == 0:
        return {**state, "status": "weak_evidence"}

    # Optional light quote sanity-check (non-blocking)
    # If you want to enforce quotes, you can add a normalized substring check here.

    return state

# --------- Conditional edges (router) ----------
def router(state: QAState) -> str:
    status = state.get("status")
    if status in ("no_docs", "bad_json", "weak_evidence"):
        return "refuse"
    return "finalize"

def refuse_node(state: QAState) -> QAState:
    return {
        **state,
        "parsed": {"answer": "", "evidence": []},
        "status": "refused",
    }

def finalize_node(state: QAState) -> QAState:
    return state

# --------- Build the graph ----------
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

app = graph.compile()

# --------- CLI ----------
if __name__ == "__main__":
    question = input("Ask a question about your PDF: ")
    state: QAState = {
        "question": question,
        "docs": [],
        "context_text": "",
        "raw_answer": "",
        "parsed": None,
        "status": "init"
    }
    result = app.invoke(state)

    status = result["status"]
    parsed = result.get("parsed") or {"answer": "", "evidence": []}

    if status in ("refused", "no_docs", "weak_evidence"):
        print("\nI don't know based on the provided document (insufficient evidence).\n")
    elif status == "bad_json":
        print("\nSorry, the model returned malformed JSON. Try rephrasing.\n")
    else:
        print("\nAnswer:", parsed["answer"])
        print("Evidence:")
        for ev in parsed.get("evidence", []):
            ref = ev.get("ref")
            quote = ev.get("quote","")
            print(f"* [ref {ref}] {quote[:120]}{'…' if len(quote)>120 else ''}")
        print()
