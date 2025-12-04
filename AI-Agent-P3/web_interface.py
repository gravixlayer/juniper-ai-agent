#!/usr/bin/env python3
import os
import streamlit as st

# Import new web-enabled agent
from juniper_web_enabled_agent import (
    chat_with_web_agent,
    USER_ID
)

# ORIGINAL LOGO (sidebar header)
LOGO_PATH = "/home/kjamwal/juniper_agent/logo.png"

# NEW: Assistant Chat Bubble Icon
ASSISTANT_AVATAR = "/home/kjamwal/juniper_agent/Fav.png"

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Juniper Specialist Agent",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "💡",
    layout="wide",
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=240)

st.sidebar.title("Juniper Specialist Agent")
st.sidebar.markdown("""
- RAG from Juniper PDFs  
- TAC-grade Junos intelligence  
- Memory & Reflections  
- Live Google Web Search (latest Juniper updates)  
""")
st.sidebar.divider()

if "last_pdf_refs" not in st.session_state:
    st.session_state.last_pdf_refs = []

# ------------------------------------------------------------
# SIDEBAR PDF REFS
# ------------------------------------------------------------
st.sidebar.subheader("📘 PDF References")

if st.session_state.last_pdf_refs:
    for i, ref in enumerate(st.session_state.last_pdf_refs, 1):
        with st.sidebar.expander(
            f"{i}. {ref['file']} (page {ref['page']}, chunk {ref['chunk']})"
        ):
            st.write(ref["text"])
else:
    st.sidebar.info("Ask a question to load PDF sources.")

st.sidebar.divider()

# ------------------------------------------------------------
# SESSION INIT
# ------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "reflections" not in st.session_state:
    st.session_state.reflections = []
if "memory_cache" not in st.session_state:
    st.session_state.memory_cache = []

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("Juniper Specialist Intelligence Dashboard")

# ------------------------------------------------------------
# REFLECTIONS + MEMORY PANELS
# ------------------------------------------------------------
col1, col2 = st.columns(2)

# Reflections Panel
with col1:
    st.subheader("🧩 Learned Reflections")

    from juniper_web_enabled_agent import chat_with_web_agent
    from Juniper_Specialist import retrieve_memories

    if st.button("🔄 Refresh Reflections"):
        raw = retrieve_memories(USER_ID, "")

        reflections = []
        current_block = []

        for line in raw.split("\n"):
            if "[Reflection]" in line:
                if current_block:
                    reflections.append("\n".join(current_block).strip())
                    current_block = []
                current_block.append(line)
            else:
                if current_block:
                    current_block.append(line)

        if current_block:
            reflections.append("\n".join(current_block).strip())

        st.session_state.reflections = reflections

    if st.session_state.reflections:
        for idx, item in enumerate(st.session_state.reflections, 1):
            with st.expander(f"Reflection #{idx}"):
                st.markdown(item)
    else:
        st.info("No reflections yet.")

# Memory Panel
from Juniper_Specialist import retrieve_memories

with col2:
    st.subheader("📚 Memory Store")
    if st.button("🔄 Refresh Memory"):
        mem = retrieve_memories(USER_ID, "")
        st.session_state.memory_cache = mem.split("\n") if mem else []

    if st.session_state.memory_cache:
        for idx, item in enumerate(st.session_state.memory_cache, 1):
            with st.expander(f"Memory #{idx}"):
                st.write(item)
    else:
        st.info("Memory empty.")

st.divider()

# ------------------------------------------------------------
# CHAT INTERFACE
# ------------------------------------------------------------
chat_window = st.container()
user_input = st.chat_input("Ask about Junos, EVPN, BGP, SRX, RIB, FPC, latest releases etc...")

# Render previous messages
with chat_window:
    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            avatar = (
                ASSISTANT_AVATAR
                if os.path.exists(ASSISTANT_AVATAR)
                else "💡"
            )
            st.chat_message("assistant", avatar=avatar).markdown(msg["content"])

        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])

# Handle new input
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # NOTE: PDF references remain from RAG (handled by underlying agent)
    from Juniper_Specialist import retrieve_context
    st.session_state.last_pdf_refs = retrieve_context(user_input)

    with st.spinner("Fetching Juniper TAC + Google Web Intelligence…"):
        answer = chat_with_web_agent(user_input)

    # Save assistant answer
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.chat_message(
        "assistant",
        avatar=ASSISTANT_AVATAR if os.path.exists(ASSISTANT_AVATAR) else "💡"
    ).markdown(answer)

    st.success("Response stored and Google-verified!")

st.sidebar.caption("© 2025 Gravix Layer | Juniper AI Assistant")
