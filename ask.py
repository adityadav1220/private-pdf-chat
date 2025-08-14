# ask.py — asks a question, retrieves top chunks, answers with evidence
from typing import List
from langchain_ollama import OllamaEmbeddings, ChatOllama    # NEW imports
from langchain_chroma import Chroma                           # NEW import
from langchain.prompts import ChatPromptTemplate

INDEX_DIR = "data/index"

SYSTEM_PROMPT = """You are a careful assistant.
ONLY answer using the provided context. If the context does not contain the answer,
say: "I don't know based on the provided document."
Always include a brief evidence section with page numbers and quoted snippets.
Keep answers concise and factual.
"""

USER_PROMPT = """Question: {question}

Context (excerpts from the document):
{context}

Answer with 2 parts:
1) The answer in 1-3 sentences.
2) Evidence: bullet list of quotes with page numbers like [p. 1]: "quoted text".
"""

def load_retriever():
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    vs = Chroma(embedding_function=embedder, persist_directory=INDEX_DIR)
    # Use plain similarity (no threshold) to avoid score-scale issues
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever

def format_context(docs) -> str:
    lines = []
    for d in docs:
        page = d.metadata.get("page", 0)  # PyMuPDFLoader 0-based
        src  = d.metadata.get("source_file", "document")
        snippet = d.page_content.strip().replace("\n", " ")
        snippet = (snippet[:350] + "…") if len(snippet) > 350 else snippet
        lines.append(f"[{src} p.{page+1}] {snippet}")
    return "\n\n".join(lines)

def answer(question: str):
    retriever = load_retriever()
    # NEW: .invoke() instead of deprecated get_relevant_documents
    docs: List = retriever.invoke(question)

    if not docs:
        print("I don't know based on the provided document. (No matches)")
        return

    context_text = format_context(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]).format_messages(context=context_text, question=question)

    llm = ChatOllama(model="llama3.1:8b", temperature=0.1)
    resp = llm.invoke(prompt)
    print("\n" + resp.content + "\n")

if __name__ == "__main__":
    q = input("Ask a question about your PDF: ")
    answer(q)
