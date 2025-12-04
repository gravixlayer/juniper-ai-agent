#!/usr/bin/env python3
"""
Juniper Specialist Agent + Google Web Search
─────────────────────────────────────────────
Adds live Google search for Juniper-related queries:
- latest Junos versions
- new hardware announcements
- CVEs / security advisories
- roadmap / release notes
"""

import os
import requests
from Juniper_Specialist import (
    chat_with_agent,
    USER_ID
)
from gravixlayer import GravixLayer


# -------------------------------------------------------------------
# GOOGLE SEARCH CONFIG
# -------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

def google_search_enabled():
    return GOOGLE_API_KEY and GOOGLE_CSE_ID


# -------------------------------------------------------------------
# DETECT IF QUERY NEEDS WEB SEARCH
# -------------------------------------------------------------------

def query_requires_web_search(query: str) -> bool:
    q = query.lower()
    trigger_words = [
        "latest", "new", "current", "release", "version",
        "cve", "vulnerability", "security", "advisory",
        "eol", "roadmap", "today", "this month",
        "junos", "juniper", "mx", "qfx", "srx", "acx", "apstra"
    ]
    return any(word in q for word in trigger_words)


# -------------------------------------------------------------------
# GOOGLE SEARCH FUNCTION
# -------------------------------------------------------------------

def run_google_search(query: str, num_results: int = 5):
    if not google_search_enabled():
        print("[GOOGLE SEARCH DISABLED] Missing API key or CSE ID.")
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": f"Juniper {query}",
        "num": num_results,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("items", []):
            results.append({
                "name": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet", "")
            })
        return results

    except Exception as e:
        print("[GOOGLE SEARCH ERROR]", e)
        return []


# -------------------------------------------------------------------
# FUSION: RAG + GOOGLE SEARCH + JUNIPER MODEL
# -------------------------------------------------------------------

def chat_with_web_agent(query: str, user_id=USER_ID):
    """
    - Runs RAG (your existing Juniper specialist agent)
    - If query is about latest updates, performs Google search
    - Uses Juniper model to combine both sources cleanly
    """

    # 1. Run normal RAG specialist logic
    rag_answer = chat_with_agent(query)

    # 2. Decide if Google Search is needed
    google_results = []
    if query_requires_web_search(query):
        print("[WEB SEARCH] Google search triggered.")
        google_results = run_google_search(query)

    # 3. If no Google search occurred, return normal answer
    if not google_results:
        return rag_answer

    # 4. Prepare Google result context
    google_context = "=== Google Web Results ===\n"
    for r in google_results:
        google_context += f"- {r['name']}\n{r['snippet']}\n{r['url']}\n\n"

    # 5. Fuse RAG answer + Google results using the Juniper model
    fusion_prompt = f"""
You are a Juniper TAC AI Assistant with real-time internet knowledge.

User query:
{query}

Internal Juniper specialist answer:
{rag_answer}

Live Google Search Results:
{google_context}

Combine all information into a single authoritative response.
Correct outdated data, and prioritize the latest Juniper information.
"""

    client = GravixLayer(api_key=os.getenv("GRAVIXLAYER_API_KEY"))
    comp = client.chat.completions.create(
        model="juniper-specialist",
        messages=[
            {"role": "system", "content": "You are a Juniper TAC expert with internet augmentation."},
            {"role": "user", "content": fusion_prompt},
        ],
    )

    return comp.choices[0].message.content


# -------------------------------------------------------------------
# CLI TEST
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("Juniper Specialist Agent + Google Search Enabled\n")
    while True:
        user_q = input("You: ")
        if user_q.lower() in ("exit", "quit"):
            break
        print("\n", chat_with_web_agent(user_q), "\n")
