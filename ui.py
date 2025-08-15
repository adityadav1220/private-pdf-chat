# ui.py — Streamlit chat UI for Privacy-First PDF QA + Markdown exports

import streamlit as st
import requests
from datetime import datetime

API_URL = "http://127.0.0.1:8001"  # FastAPI backend

st.set_page_config(page_title="Privacy-First PDF QA", page_icon="🔒", layout="wide")
st.title("🔒 Privacy-First PDF QA (Offline)")

# ---------- Session state ----------
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role":"user"/"assistant","content": str}
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None  # {"question","answer","evidence","ts"}

# ---------- Sidebar settings ----------
with st.sidebar:
    st.markdown("## Settings")
    redact = st.toggle("Redaction ON", value=True)
    types = st.multiselect(
        "Redact types",
        ["email", "phone", "creditcard", "ssn"],
        default=["email", "phone", "creditcard", "ssn"]
    )
    reset = st.toggle("Reset index before build", value=False)
    st.divider()
    st.caption(f"API: {API_URL}")

col_left, col_right = st.columns([1.2, 1], vertical_alignment="top")

# ---------- Utility: Markdown builders ----------
def build_answer_md(item: dict) -> str:
    ts = item.get("ts", "")
    q  = item.get("question","")
    a  = item.get("answer","")
    ev = item.get("evidence",[]) or []
    lines = [
        f"# Answer Export",
        f"_Generated_: {ts}",
        "",
        f"**Question:** {q}",
        "",
        f"**Answer:** {a}" if a else "**Answer:** (no answer)",
        "",
        "**Evidence**:",
    ]
    if ev:
        for e in ev:
            lines.append(f"- [ref {e.get('ref')}] {e.get('quote','')}")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)

def build_chat_md(msgs: list) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# Chat Export", f"_Generated_: {ts}", ""]
    for m in msgs:
        role = m.get("role","user").capitalize()
        content = m.get("content","")
        lines.append(f"**{role}:** {content}")
        lines.append("")
    return "\n".join(lines)

# ---------- Left: Build Index ----------
with col_left:
    st.subheader("1) Upload documents and build the index")

    uploaded = st.file_uploader(
        "Upload one or more files",
        type=["pdf","txt","md"],           # ← allow txt and md now
        accept_multiple_files=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        build_disabled = st.session_state.index_ready and not reset
        build_btn = st.button("🚀 Build Index", use_container_width=True, disabled=build_disabled)
    with c2:
        rebuild_btn = st.button("♻️ Rebuild (reset)", use_container_width=True)
    with c3:
        health_btn = st.button("🩺 Check API health", use_container_width=True)

    if health_btn:
        try:
            r = requests.get(f"{API_URL}/health", timeout=10)
            st.success(f"API OK: {r.json()}")
        except Exception as e:
            st.error(f"API not reachable: {e}")

    if build_btn or rebuild_btn:
        try:
            # Send any uploaded files (pdf/txt/md); backend ignores type at this point
            files = []
            for f in (uploaded or []):
                mime = "application/pdf" if f.name.lower().endswith(".pdf") else "text/plain"
                files.append(("files", (f.name, f.getvalue(), mime)))

            data = {
                "redact": str(redact).lower(),
                "types": ",".join(types),
                "reset": str(rebuild_btn or reset).lower(),  # rebuild forces reset=true (safe with versioned dirs)
            }
            with st.spinner("Building index..."):
                r = requests.post(f"{API_URL}/index", files=files if files else None, data=data, timeout=600)
            try:
                payload = r.json()
            except ValueError:
                st.error(f"API returned non-JSON (status {r.status_code}): {r.text[:300]}")
                payload = {"ok": False}
            if not payload.get("ok"):
                st.error(payload.get("message", f"Index build failed (status {r.status_code})"))
            else:
                st.session_state.index_ready = True
                chunks = payload.get("chunks", 0)
                redactions = payload.get("redactions", {})
                st.success(f"Index built ✅  (chunks: {chunks})")
                if redactions:
                    with st.expander("Redaction report"):
                        for k, v in redactions.items():
                            st.write(f"- {k}: {v} replaced")
        except Exception as e:
            st.error(f"Index build failed: {e}")

    st.caption("Tip: After the first build, use **Rebuild (reset)** if you changed files or settings.")

# ---------- Right: Chat ----------
with col_right:
    st.subheader("2) Chat with your documents")

    if not st.session_state.index_ready:
        st.info("Build the index first, then start chatting.")
    else:
        # show chat history
        for msg in st.session_state.messages:
            with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask anything about your PDFs, .txt, or .md…")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        r = requests.post(f"{API_URL}/chat", json={"messages": st.session_state.messages}, timeout=180)
                        data = r.json()
                        if not data.get("ok"):
                            st.error(data.get("message", "Unknown error"))
                        else:
                            if data.get("refused"):
                                answer = "I don't know based on the provided document (insufficient evidence)."
                                st.warning(answer)
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                                st.session_state.last_answer = {
                                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "question": user_input, "answer": "", "evidence": []
                                }
                            else:
                                answer = data.get("answer","")
                                st.success(answer if answer else "(no answer)")
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                                st.session_state.last_answer = {
                                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "question": user_input, "answer": answer,
                                    "evidence": data.get("evidence", [])
                                }
                                evs = data.get("evidence", [])
                                if evs:
                                    st.markdown("**Evidence:**")
                                    for ev in evs:
                                        st.write(f"- [ref {ev.get('ref')}] {ev.get('quote','')}")
            except Exception as e:
                st.error(f"/chat failed: {e}")

        cols = st.columns(4)
        if cols[0].button("🧹 Clear chat"):
            st.session_state.messages = []
            st.session_state.last_answer = None
            st.rerun()

        # Export last answer
        if cols[1].button("⬇️ Export last answer (.md)"):
            if st.session_state.last_answer:
                md = build_answer_md(st.session_state.last_answer)
                st.download_button(
                    "Download answer.md",
                    data=md.encode("utf-8"),
                    file_name=f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                )
            else:
                st.warning("No answer to export yet.")

        # Export chat
        if cols[2].button("⬇️ Export chat (.md)"):
            if st.session_state.messages:
                md = build_chat_md(st.session_state.messages)
                st.download_button(
                    "Download chat.md",
                    data=md.encode("utf-8"),
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                )
            else:
                st.warning("Chat is empty.")

        if cols[3].button("🔄 New session (keep index)"):
            st.session_state.messages = []
            st.session_state.last_answer = None
            st.success("Started a new chat session. Index is unchanged.")
