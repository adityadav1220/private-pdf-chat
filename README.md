# Privacy-First Doc QA (Offline, On-Device)

Ask questions about your PDFs **locally** with **evidence** and **PII redaction**.  
No cloud. No data leaves your laptop. Built with **Ollama**, **LangChain**, **LangGraph**, and **Chroma**.

## ✨ Features
- Fully **offline** (Ollama local LLM + embeddings)
- **Evidence-backed** answers with citations
- **Guardrails** via LangGraph: retrieve → generate(JSON) → verify(refs) → refuse
- **PII redaction** (email, phone, CC-like, SSN-like) before indexing
- Simple CLI now; tiny UI (FastAPI + Streamlit) coming next

## 🧱 Architecture (short)
PDFs → split → embeddings → Chroma (index)
↓
Question → retrieve top-k → LLM (JSON: answer + [ref]) → verify refs → answer/refuse


## 🛠 Setup
```bash
# 1) Install Ollama and pull local models (first time only)
brew install ollama        # macOS (or see ollama.com for other OS)
ollama serve               # keep this running in a Terminal tab
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 2) Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3) Add PDFs
Put your PDFs in data/docs/.
This folder is .gitignored so your private files are never uploaded.
If the folder is empty, create a placeholder:
mkdir -p data/docs data/index
touch data/docs/.gitkeep

# 4) 4) Build the index (with redaction ON)
python index.py --reset --redact true

# 5) Ask guarded questions (LangGraph pipeline)
python graph_ask.py
# example questions:
#   List all items under the SKILLS section.
#   What is the total amount due?
#   What is the deadline date?

🔐 Redaction controls
# turn OFF redaction (not recommended for private docs)
python index.py --reset --redact false

# choose types (comma-separated)
python index.py --reset --redact true --types email,phone
# available: email, phone, creditcard, ssn


🧩 Project structure
privacy-first-pdf-qa/
├─ index.py         # build local index with optional PII redaction
├─ ask.py           # simple ask (no graph) with citations
├─ graph_ask.py     # LangGraph pipeline with ref-based verification
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ .gitignore
├─ data/
│  ├─ docs/         # put PDFs here (ignored by git)
│  │  └─ .gitkeep
│  └─ index/        # Chroma index (ignored by git)
└─ .venv/           # local virtualenv (ignored by git)
