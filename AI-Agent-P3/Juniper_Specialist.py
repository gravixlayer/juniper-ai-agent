#!/usr/bin/env python3
"""
Juniper Specialist Agent (RAG + Memory + Reflection)
──────────────────────────────────────────────────────────────
- RAG from Juniper PDFs
- TAC-grade Junos reasoning
- Long-term memory storage
- Automatic self-reflection (integrated)
"""

import os
from datetime import datetime
from gravixlayer import GravixLayer

# ============================================================
# ENV VARS
# ============================================================
API_KEY = os.getenv("GRAVIXLAYER_API_KEY")
INDEX_ID = os.getenv("GRAVIX_VECTOR_INDEX_ID")

if not API_KEY:
    raise SystemExit("GRAVIXLAYER_API_KEY not set.")
if not INDEX_ID:
    raise SystemExit("GRAVIX_VECTOR_INDEX_ID not set.")

# Models
EMBED_MODEL = "baai/bge-large-en-v1.5"
CHAT_MODEL  = "juniper-specialist"

# Memory config
MEMORY_INDEX_NAME = "juniper-memories"
USER_ID = "juniper_cli_user"

# Gravix client
client = GravixLayer(api_key=API_KEY)

# Vector DB (RAG)
kb_vectors = client.vectors.index(INDEX_ID)

# Memory interface
memory = client.memory(
    embedding_model=EMBED_MODEL,
    inference_model=CHAT_MODEL,
    index_name=MEMORY_INDEX_NAME,
    cloud_provider="Gravix",
    region="eu-west-1",
)

# ============================================================
# ENSURE MEMORY INDEX EXISTS (Fix for "index not found")
# ============================================================
def ensure_memory_index():
    try:
        # Attempt a dummy search to check if index exists
        memory.search(query="healthcheck", user_id=USER_ID, limit=1)
        print("[MEMORY] Index exists.")
    except Exception as e:
        if "index not found" in str(e).lower():
            print("[MEMORY] Index missing. Creating...")
            try:
                client.memory.create_index(index_name=MEMORY_INDEX_NAME)
                print("[MEMORY] Index created successfully.")
            except Exception as ce:
                print(f"[MEMORY] Failed to create index: {ce}")
        else:
            print("[MEMORY] Unexpected memory error:", e)

# Run index creation check on startup
ensure_memory_index()

# ============================================================
# SAFE RAG EXTRACTION (fix for no .text attribute)
# ============================================================
def retrieve_context(query, top_k=7):
    try:
        results = kb_vectors.search_text(
            query=query,
            model=EMBED_MODEL,
            top_k=top_k,
        )
    except Exception:
        return []

    hits = getattr(results, "hits", []) or []
    output = []

    for hit in hits:
        meta = getattr(hit, "metadata", {}) or {}
        doc  = getattr(hit, "document", {}) or {}

        # Try multiple possible text fields (Gravix index formats differ)
        text = (
            doc.get("text") or
            doc.get("chunk_text") or
            meta.get("text") or
            ""
        )

        output.append({
            "file": meta.get("file", "unknown"),
            "page": meta.get("page", "?"),
            "chunk": meta.get("chunk", "?"),
            "text": text.strip(),
        })

    return output

# ============================================================
# MEMORY RETRIEVAL
# ============================================================
def retrieve_memories(user_id, query, limit=5):
    try:
        res = memory.search(query=query, user_id=user_id, limit=limit)
        return "\n".join([m["memory"] for m in res.get("results", [])])
    except Exception:
        return ""

# ============================================================
# MEMORY WRITE
# ============================================================
def maybe_store_memory(user_id, query, answer):
    if len(query) < 60 and len(answer) < 120:
        return

    payload = (
        f"User query:\n{query}\n\n"
        f"Assistant answer:\n{answer}\n\n"
        "Extract only stable, technical, long-term facts."
    )

    try:
        memory.add(messages=payload, user_id=user_id, infer=True)
        print("[MEMORY] Stored memory.")
    except Exception as e:
        print(f"[MEMORY] Failed to store: {e}")

# ============================================================
# REFLECTION ENGINE (Integrated)
# ============================================================

reflection_cache = set()

def generate_reflection_text(query: str, answer: str) -> str:
    """Generate concise technical insights."""
    prompt = f"""
You are a reflection engine for a Juniper TAC-grade assistant.
Extract 2–5 concise reusable insights.

Q: {query}
A: {answer}

Return bullet points only.
"""

    try:
        comp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You write precise, technical insights."},
                {"role": "user", "content": prompt},
            ],
        )
        return comp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[REFLECTION] Generation failed: {e}")
        return ""

def store_reflection(insight: str):
    if not insight:
        return

    if insight in reflection_cache:
        return

    reflection_cache.add(insight)

    payload = (
        f"[Reflection]\n"
        f"Timestamp: {datetime.utcnow().isoformat()}Z\n"
        f"{insight}"
    )

    try:
        memory.add(messages=payload, user_id=USER_ID)
        print("[REFLECTION] Stored reflection.")
    except Exception as e:
        print(f"[REFLECTION] Failed to store: {e}")

def reflect_from_interaction(query: str, answer: str):
    """Create + store reflection after each response."""
    if len(query) < 20 or len(answer) < 40:
        return

    insight = generate_reflection_text(query, answer)
    if insight:
        store_reflection(insight)

    return insight

# ============================================================
# CHAT LOGIC
# ============================================================
def chat_with_agent(query, user_id=USER_ID):

    kb_context = retrieve_context(query)
    mem_context = retrieve_memories(user_id, query)

    system_prompt = """
You are Juniper_Specialist, a world-class Juniper TAC engineer.
Use ONLY the provided RAG context + relevant memories.
"""

    ctx_blocks = ["=== Knowledge Base ==="]
    for c in kb_context:
        ctx_blocks.append(
            f"[{c['file']} page {c['page']} chunk {c['chunk']}]\n{c['text']}"
        )

    if mem_context:
        ctx_blocks.append("\n=== Past Memories ===")
        ctx_blocks.append(mem_context)

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "\n\n".join(ctx_blocks)},
        {"role": "user", "content": query},
    ]

    try:
        comp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=msgs,
        )
        answer = comp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

    # Store long-term memory
    maybe_store_memory(user_id, query, answer)

    # Store reflections
    try:
        reflect_from_interaction(query, answer)
    except Exception as e:
        print("[REFLECTION ERROR]", e)

    return answer

# ============================================================
# CLI TEST
# ============================================================
if __name__ == "__main__":
    print("Juniper Specialist Agent (Reflection Enabled)\n")
    while True:
        q = input("You: ")
        if q.lower() in ("quit", "exit"):
            break
        print("\n", chat_with_agent(q), "\n")
