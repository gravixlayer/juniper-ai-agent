#!/usr/bin/env python3
"""
Chunk & Upload PDF's
"""

import os
import re
import uuid
import json
from datetime import datetime, timezone
from PyPDF2 import PdfReader
from gravixlayer import GravixLayer
from tqdm import tqdm

# ============================================================
# ENVIRONMENT
# ============================================================
API_KEY = os.getenv("GRAVIXLAYER_API_KEY")
INDEX_ID = os.getenv("GRAVIX_VECTOR_INDEX_ID")

if not API_KEY:
    raise SystemExit("ERROR: GRAVIXLAYER_API_KEY is missing.")
if not INDEX_ID:
    raise SystemExit("ERROR: GRAVIX_VECTOR_INDEX_ID is missing (must be UUID).")

client = GravixLayer()

# Correct vector index object
vectors = client.vectors.index(INDEX_ID)

# Embedding model
EMBED_MODEL = "baai/bge-large-en-v1.5"

# Chunk settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MANIFEST_FILE = "ingestion_manifest.json"


# ============================================================
# SANITIZE VECTOR ID (critical)
# ============================================================
def sanitize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.replace("\x00", "").strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


# ============================================================
# CHUNKING
# ============================================================
def make_chunks(text):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# MANIFEST
# ============================================================
def update_manifest(file_name, chunk_count):
    manifest = {}
    if os.path.exists(MANIFEST_FILE):
        manifest = json.load(open(MANIFEST_FILE))

    entry = {
        "chunks": chunk_count,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    manifest.setdefault(file_name, []).append(entry)
    json.dump(manifest, open(MANIFEST_FILE, "w"), indent=2)


# ============================================================
# PROCESS A SINGLE PDF
# ============================================================
def process_pdf(path):
    raw_name = os.path.basename(path)
    safe_file_name = sanitize_id(raw_name)

    print(f"\nProcessing {raw_name}")

    pages = extract_text(path)
    if not pages:
        print("No text found in PDF.")
        return

    all_chunks = []
    meta_info = []

    for page in tqdm(pages, desc=f"Chunking {raw_name}", unit="page"):
        chunks = make_chunks(page["text"])

        for ci, c in enumerate(chunks):
            all_chunks.append(c)
            meta_info.append({
                "file": safe_file_name,
                "page": page["page"],
                "chunk": ci
            })

    print(f"Total chunks: {len(all_chunks)}")
    print("Uploading (auto-embedding)...")

    for chunk_text, meta in zip(all_chunks, meta_info):

        vector_id = sanitize_id(
            f"{meta['file']}_p{meta['page']}_c{meta['chunk']}_{uuid.uuid4().hex}"
        )

        vectors.upsert_text(
            text=chunk_text,
            model=EMBED_MODEL,
            id=vector_id,
            metadata={
                **meta,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    update_manifest(raw_name, len(all_chunks))

    print(f"Completed {raw_name}: {len(all_chunks)} chunks uploaded.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    folder = "/home/kjamwal/juniper_agent/pdfs"

    pdfs = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ]

    if not pdfs:
        print("No PDFs found.")
        exit()

    print(f"Starting ingestion for {len(pdfs)} PDFs...")

    for pdf in pdfs:
        process_pdf(pdf)

    print("\nALL PDFs PROCESSED SUCCESSFULLY!")
