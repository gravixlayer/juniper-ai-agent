#!/usr/bin/env python3
"""
Juniper Specialist Agent

"""

import os
import time
from gravixlayer import GravixLayer

# ============================================================
# ENV VARS
# ============================================================
API_KEY = os.getenv("GRAVIXLAYER_API_KEY")
INDEX_ID = os.getenv("GRAVIX_VECTOR_INDEX_ID")  # UUID of your index

if not API_KEY:
    raise SystemExit("GRAVIXLAYER_API_KEY not set.")
if not INDEX_ID:
    raise SystemExit("GRAVIX_VECTOR_INDEX_ID not set.")

client = GravixLayer(api_key=API_KEY)

# Embedding + Chat models
EMBED_MODEL = "baai/bge-large-en-v1.5"
CHAT_MODEL  = "juniper-specialist"

# Memory config
MEMORY_INDEX_NAME = "juniper-memories"
USER_ID = "juniper_cli_user"

# Vector handle for your KB
kb_vectors = client.vectors.index(INDEX_ID)

# Memory handle
memory = client.memory(
    embedding_model=EMBED_MODEL,
    inference_model=CHAT_MODEL,
    index_name=MEMORY_INDEX_NAME,
    cloud_provider="AWS",
    region="us-east-1",
)


# ============================================================
# RAG SEARCH
# ============================================================
def retrieve_context(query, top_k=5):
    """Use Gravix vector index to find relevant KB chunks."""
    try:
        results = kb_vectors.search_text(
            query=query,
            model=EMBED_MODEL,
            top_k=top_k,
        )
    except Exception as e:
        print(f"RAG search failed: {e}")
        return ""

    hits = getattr(results, "hits", []) or []
    if not hits:
        return ""

    blocks = []
    for hit in hits:
        text = getattr(hit, "text", "") or ""
        meta = getattr(hit, "metadata", {}) or {}

        file = meta.get("file", "unknown")
        page = meta.get("page", "?")
        chunk = meta.get("chunk", "?")

        blocks.append(f"[{file}, page {page}, chunk {chunk}]\n{text}")

    return "\n\n".join(blocks)


# ============================================================
# MEMORY RETRIEVAL
# ============================================================
def retrieve_memories(user_id, query, limit=5):
    """Retrieve only relevant memories."""
    try:
        res = memory.search(query=query, user_id=user_id, limit=limit)
    except Exception as e:
        print(f"Memory search failed: {e}")
        return ""

    mems = res.get("results", []) or []
    return "\n".join([f"- {m['memory']}" for m in mems if 'memory' in m])


# ============================================================
# MEMORY WRITE
# ============================================================
def maybe_store_memory(user_id, query, answer):
    """Mode B: store only meaningful facts."""
    try:
        if len(query) < 60 and len(answer) < 120:
            return

        payload = (
            f"User query:\n{query}\n\n"
            f"Assistant answer:\n{answer}\n\n"
            "Extract only stable, technical, long-term facts."
        )

        res = memory.add(messages=payload, user_id=user_id, infer=True)
        count = len(res.get("results", []))
        if count:
            print(f"Stored {count} memory items.")
    except Exception as e:
        print(f"Memory write failed: {e}")


# ============================================================
# MAIN CHAT LOGIC
# ============================================================
def chat_with_agent(query, user_id=USER_ID):

    # 1. Retrieve KB context
    kb_context = retrieve_context(query, top_k=7)

    # 2. Retrieve long-term memories
    mem_context = retrieve_memories(user_id, query)

    system_prompt = """
You are Juniper_Specialist, a world-class Juniper TAC engineer.

Rules:
- Use ONLY the provided RAG context + relevant memories.
- If info is missing, say so and request data.
- Use real Junos configuration syntax.
- Provide TAC-quality troubleshooting steps.
"""

    ctx = []
    ctx.append("=== Knowledge Base (PDFs) ===\n" + (kb_context or "No KB context."))
    if mem_context:
        ctx.append("\n=== Past Memories ===\n" + mem_context)

    context_message = "\n\n".join(ctx)

    # Messages
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": context_message},
        {"role": "user", "content": query},
    ]

    # 3. Chat completion (FIXED ACCESS)
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=msgs,
        )

        # FIXED — object-style access, NOT dict
        answer = completion.choices[0].message.content

    except Exception as e:
        print(f"Chat completion failed: {e}")
        return "Sorry, an error occurred generating the answer."

    # 4. Save memory (selective)
    maybe_store_memory(user_id, query, answer)

    return answer


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    print("Juniper Specialist Agent Ready (RAG + Memory).\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        print("\nAnswer:\n")
        print(chat_with_agent(q))
        print("\n------------------------------------------------------------\n")
