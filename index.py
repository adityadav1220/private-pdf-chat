# index.py — build the index with optional PII redaction + reset
import re
import argparse
import shutil
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOCS_DIR = Path("data/docs")
INDEX_DIR = "data/index"

# ---- PII patterns (regex-based; fast and offline) ----
PII_PATTERNS = {
    "email": (
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "[REDACTED_EMAIL]",
    ),
    "phone": (
        re.compile(r"\+?\d[\d\-\s\(\)]{7,}\d"),
        "[REDACTED_PHONE]",
    ),
    # very rough "credit-card-ish" numbers (13-16 digits with spaces/dashes)
    "creditcard": (
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "[REDACTED_CC]",
    ),
    # US-style SSN
    "ssn": (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
}

def apply_redactions(text: str, enabled_types) -> tuple[str, dict]:
    counts = {}
    for t in enabled_types:
        pat, repl = PII_PATTERNS[t]
        text, n = pat.subn(repl, text)
        counts[t] = counts.get(t, 0) + n
    return text, counts

def load_pdfs():
    docs = []
    for pdf_path in DOCS_DIR.glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()
        for p in pages:
            p.metadata["source_file"] = pdf_path.name
        docs.extend(pages)
    return docs

def main():
    parser = argparse.ArgumentParser(description="Build local index with optional redaction.")
    parser.add_argument("--reset", action="store_true", help="Delete existing index before building.")
    parser.add_argument("--redact", type=str, default="true", choices=["true", "false"],
                        help="Redact PII before indexing (default: true).")
    parser.add_argument("--types", type=str, default="email,phone,creditcard,ssn",
                        help="Comma-separated PII types to redact.")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=120)
    args = parser.parse_args()

    if args.reset and Path(INDEX_DIR).exists():
        shutil.rmtree(INDEX_DIR)

    docs = load_pdfs()
    if not docs:
        print("No PDFs found in data/docs. Add a PDF and run again.")
        return

    redact_on = (args.redact.lower() == "true")
    enabled_types = [t.strip() for t in args.types.split(",") if t.strip() in PII_PATTERNS]

    # redact before indexing (if enabled)
    redaction_totals = {t: 0 for t in enabled_types}
    if redact_on and enabled_types:
        for d in docs:
            redacted_text, counts = apply_redactions(d.page_content, enabled_types)
            d.page_content = redacted_text
            for t, n in counts.items():
                redaction_totals[t] += n

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embedder = OllamaEmbeddings(model="nomic-embed-text")
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=INDEX_DIR,
    )

    print(f"Indexed {len(chunks)} chunks into {INDEX_DIR}")
    if redact_on and enabled_types:
        print("Redaction report:")
        for t in enabled_types:
            print(f"  {t}: {redaction_totals.get(t, 0)} replaced")
    else:
        print("Redaction is OFF")

if __name__ == "__main__":
    main()
