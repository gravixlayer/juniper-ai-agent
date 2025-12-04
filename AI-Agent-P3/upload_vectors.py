#!/usr/bin/env python3
"""
Chunk + Upload PDFs to Gravix Vector Index (Production Ready)
--------------------------------------------------------------
✅ Real-time progress tracking with detailed metrics
✅ Parallel batch uploads for high performance
✅ Comprehensive error handling and retry logic
✅ Production-ready with proper logging
"""

from __future__ import annotations
import os, re, json, uuid, logging, time, sys
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from PyPDF2 import PdfReader
import requests

# -------------------- CONFIG --------------------
API_KEY   = os.getenv("GRAVIXLAYER_API_KEY")
INDEX_ID  = os.getenv("GRAVIX_VECTOR_INDEX_ID")
BASE_URL  = os.getenv("GRAVIXLAYER_BASE_URL") or "https://api.gravixlayer.com/v1"

EMBED_MODEL = "baai/bge-large-en-v1.5"

CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 150
BATCH_SIZE    = 100
TIMEOUT       = 300
MAX_RETRIES   = 3
PARALLEL_UPLOADS = 5  # Number of parallel batch uploads
VECTOR_MANIFEST = "vector_manifest.json"

# Configure logging (file only, console uses progress display)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('pdf_upload.log')]
)
logger = logging.getLogger(__name__)

# Setup session with connection pooling
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=PARALLEL_UPLOADS * 2,
    pool_maxsize=PARALLEL_UPLOADS * 4,
    max_retries=0,  # We handle retries manually
    pool_block=False
)
session.mount('http://', adapter)
session.mount('https://', adapter)
session.headers.update({
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
})


# -------------------- PROGRESS TRACKING --------------------
@dataclass
class UploadStats:
    """Thread-safe upload statistics tracker"""
    total_vectors: int = 0
    total_batches: int = 0
    batch_size: int = BATCH_SIZE
    
    vectors_uploaded: int = 0
    batches_completed: int = 0
    batches_failed: int = 0
    
    start_time: float = field(default_factory=time.time)
    batch_times: List[float] = field(default_factory=list)
    
    _lock: Lock = field(default_factory=Lock)
    
    def record_batch_complete(self, vectors_count: int, duration: float):
        with self._lock:
            self.vectors_uploaded += vectors_count
            self.batches_completed += 1
            self.batch_times.append(duration)
    
    def record_batch_failed(self):
        with self._lock:
            self.batches_failed += 1
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def avg_batch_time(self) -> float:
        with self._lock:
            return sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
    
    @property
    def vectors_per_second(self) -> float:
        elapsed = self.elapsed_time
        return self.vectors_uploaded / elapsed if elapsed > 0 else 0
    
    @property
    def eta_seconds(self) -> float:
        remaining = self.total_vectors - self.vectors_uploaded
        rate = self.vectors_per_second
        return remaining / rate if rate > 0 else 0
    
    @property
    def progress_percent(self) -> float:
        return (self.vectors_uploaded / self.total_vectors * 100) if self.total_vectors > 0 else 0


def format_time(seconds: float) -> str:
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins, secs = divmod(seconds, 60)
        return f"{int(mins)}m {int(secs)}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        mins, secs = divmod(remainder, 60)
        return f"{int(hours)}h {int(mins)}m {int(secs)}s"


def create_progress_bar(percent: float, width: int = 40) -> str:
    """Create a text-based progress bar"""
    filled = int(width * percent / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def display_progress(stats: UploadStats, current_batch: int = 0, batch_duration: float = 0):
    """Display real-time progress status"""
    # Clear line and move cursor
    sys.stdout.write("\033[2K\r")
    
    progress_bar = create_progress_bar(stats.progress_percent)
    
    status_lines = [
        "",
        f"{'─' * 60}",
        f"  📊 UPLOAD PROGRESS",
        f"{'─' * 60}",
        f"  {progress_bar} {stats.progress_percent:5.1f}%",
        "",
        f"  Vectors:  {stats.vectors_uploaded:,} / {stats.total_vectors:,}",
        f"  Batches:  {stats.batches_completed} / {stats.total_batches} ({stats.batches_failed} failed)",
        "",
        f"  ⏱  Elapsed:     {format_time(stats.elapsed_time)}",
        f"  ⏱  ETA:         {format_time(stats.eta_seconds)}",
        f"  ⚡ Speed:       {stats.vectors_per_second:.1f} vectors/sec",
        f"  📦 Avg Batch:   {format_time(stats.avg_batch_time)}",
    ]
    
    if batch_duration > 0:
        status_lines.append(f"  📦 Last Batch:  {format_time(batch_duration)}")
    
    status_lines.extend([
        f"{'─' * 60}",
        ""
    ])
    
    # Move cursor up and rewrite
    num_lines = len(status_lines)
    sys.stdout.write(f"\033[{num_lines}A")
    for line in status_lines:
        sys.stdout.write(f"\033[2K{line}\n")
    sys.stdout.flush()


def init_progress_display(stats: UploadStats):
    """Initialize the progress display area"""
    print("\n" * 15)  # Create space for progress display
    display_progress(stats)


# ---------------- EXTRACT TEXT -------------------
def extract_pdf_text(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: List[Dict[str, Any]] = []

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", "").strip()
        if text:
            pages.append({"page": i, "text": text})

    return pages


# ----------------- CHUNK TEXT --------------------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap

    return [c for c in chunks if c]


# -------- DIRECT API UPLOADER --------
def upload_batch(
    index_id: str, 
    vectors: List[Dict[str, Any]], 
    batch_num: int,
    stats: UploadStats
) -> Dict[str, Any]:
    """
    Upload a single batch with retry logic
    Returns: {'success': bool, 'vectors_uploaded': int, 'duration': float, 'error': str|None}
    """
    url = f"{BASE_URL}/vectors/{index_id}/text/upsert"
    payload = {"vectors": vectors}
    
    start_time = time.time()
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(url, json=payload, timeout=TIMEOUT)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                count = result.get('count') or len(result.get('ids', []))
                
                stats.record_batch_complete(count, duration)
                logger.info(f"Batch {batch_num} uploaded: {count} vectors in {duration:.2f}s")
                
                return {
                    'success': True,
                    'batch_num': batch_num,
                    'vectors_uploaded': count,
                    'duration': duration,
                    'error': None
                }
                
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
                logger.warning(f"Batch {batch_num}: Rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
                continue
                
            elif response.status_code == 413:
                # Payload too large - split and retry recursively
                logger.warning(f"Batch {batch_num}: Payload too large, splitting")
                mid = len(vectors) // 2
                r1 = upload_batch(index_id, vectors[:mid], batch_num, stats)
                r2 = upload_batch(index_id, vectors[mid:], batch_num, stats)
                
                return {
                    'success': r1['success'] and r2['success'],
                    'batch_num': batch_num,
                    'vectors_uploaded': r1['vectors_uploaded'] + r2['vectors_uploaded'],
                    'duration': time.time() - start_time,
                    'error': r1.get('error') or r2.get('error')
                }
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.error(f"Batch {batch_num}: {last_error}")
                
        except requests.exceptions.Timeout:
            last_error = "Request timeout"
            logger.warning(f"Batch {batch_num}: Timeout (attempt {attempt + 1})")
            
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            logger.error(f"Batch {batch_num}: {last_error}")
        
        # Exponential backoff
        if attempt < MAX_RETRIES - 1:
            time.sleep(min(2 ** attempt, 30))
    
    # All retries exhausted
    stats.record_batch_failed()
    duration = time.time() - start_time
    
    return {
        'success': False,
        'batch_num': batch_num,
        'vectors_uploaded': 0,
        'duration': duration,
        'error': last_error
    }


def upsert_chunks(index_id: str, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Upload vectors with parallel batch processing and real-time progress
    """
    # Add model to each vector
    for vector in vectors:
        vector["model"] = EMBED_MODEL
    
    # Create batches
    batches = [vectors[i:i + BATCH_SIZE] for i in range(0, len(vectors), BATCH_SIZE)]
    
    # Initialize stats
    stats = UploadStats(
        total_vectors=len(vectors),
        total_batches=len(batches),
        batch_size=BATCH_SIZE,
        start_time=time.time()
    )
    
    # Display initial info
    print(f"\n{'═' * 60}")
    print(f"  📤 STARTING UPLOAD")
    print(f"{'═' * 60}")
    print(f"  Total Vectors:    {stats.total_vectors:,}")
    print(f"  Total Batches:    {stats.total_batches}")
    print(f"  Batch Size:       {stats.batch_size}")
    print(f"  Parallel Workers: {PARALLEL_UPLOADS}")
    print(f"{'═' * 60}\n")
    
    logger.info(f"Starting upload: {len(vectors)} vectors in {len(batches)} batches")
    
    # Initialize progress display
    init_progress_display(stats)
    
    results = []
    
    # Parallel upload with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=PARALLEL_UPLOADS) as executor:
        futures = {
            executor.submit(upload_batch, index_id, batch, i + 1, stats): i
            for i, batch in enumerate(batches)
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            display_progress(stats, result['batch_num'], result['duration'])
    
    # Final summary
    total_duration = time.time() - stats.start_time
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    
    print(f"\n{'═' * 60}")
    print(f"  ✅ UPLOAD COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Vectors Uploaded: {stats.vectors_uploaded:,} / {stats.total_vectors:,}")
    print(f"  Batches:          {successful} succeeded, {failed} failed")
    print(f"  Total Time:       {format_time(total_duration)}")
    print(f"  Avg Speed:        {stats.vectors_uploaded / total_duration:.1f} vectors/sec")
    print(f"{'═' * 60}\n")
    
    logger.info(f"Upload complete: {stats.vectors_uploaded}/{stats.total_vectors} vectors in {total_duration:.2f}s")
    
    # Report errors
    errors = [r for r in results if not r['success']]
    if errors:
        print(f"  ⚠️  Failed batches:")
        for err in errors[:5]:  # Show first 5 errors
            print(f"      Batch {err['batch_num']}: {err['error']}")
        if len(errors) > 5:
            print(f"      ... and {len(errors) - 5} more")
    
    return {
        'success': failed == 0,
        'vectors_uploaded': stats.vectors_uploaded,
        'total_vectors': stats.total_vectors,
        'duration': total_duration,
        'failed_batches': failed
    }


# ---------------- MANIFEST ------------------------
def update_manifest(file_name: str, chunks: int) -> None:
    manifest: Dict[str, Any] = {}

    if os.path.exists(VECTOR_MANIFEST):
        with open(VECTOR_MANIFEST, "r") as f:
            manifest = json.load(f)

    entry = {
        "chunks": chunks,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

    manifest.setdefault(file_name, []).append(entry)

    with open(VECTOR_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------- PDF PROCESSOR -------------------
def process_and_upload_pdfs(uploaded_files, index_id: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    
    print(f"\n{'═' * 60}")
    print(f"  📄 PDF PROCESSING")
    print(f"{'═' * 60}")
    print(f"  Files to process: {len(uploaded_files)}")
    print(f"{'═' * 60}\n")
    
    logger.info(f"Starting processing for {len(uploaded_files)} PDFs")

    all_vectors: List[Dict[str, Any]] = []
    file_vector_counts: Dict[str, int] = {}

    # Phase 1: Extract and chunk all PDFs
    for uploaded in uploaded_files:
        print(f"  📖 Extracting: {uploaded.name}")
        logger.info(f"Processing file: {uploaded.name}")

        try:
            pdf_bytes = uploaded.read()
            pages = extract_pdf_text(pdf_bytes)

            if not pages:
                print(f"  ⚠️  No readable text in {uploaded.name}")
                logger.warning(f"No readable text in {uploaded.name}")
                results.append({"filename": uploaded.name, "status": "skipped", "reason": "no text"})
                continue

            safe_file = re.sub(r"[^a-zA-Z0-9_-]", "_", uploaded.name)
            file_vectors = []

            for p in pages:
                for idx, chunk in enumerate(chunk_text(p["text"])):
                    file_vectors.append({
                        "id": f"{safe_file}_{uuid.uuid4().hex}_p{p['page']}_c{idx}",
                        "text": chunk,
                        "metadata": {
                            "source": "pdf_upload",
                            "file": uploaded.name,
                            "page": p["page"],
                            "chunk": idx,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    })

            if file_vectors:
                all_vectors.extend(file_vectors)
                file_vector_counts[uploaded.name] = len(file_vectors)
                print(f"  ✓  Extracted {len(file_vectors):,} chunks from {uploaded.name}")
                logger.info(f"Extracted {len(file_vectors)} vectors from {uploaded.name}")
            else:
                print(f"  ⚠️  No chunks extracted from {uploaded.name}")
                results.append({"filename": uploaded.name, "status": "skipped", "reason": "no chunks"})

        except Exception as e:
            error_msg = f"Error processing {uploaded.name}: {e}"
            print(f"  ❌ {error_msg}")
            logger.error(error_msg)
            results.append({"filename": uploaded.name, "status": "error", "error": str(e)})

    if not all_vectors:
        print("\n  ❌ No vectors to upload")
        return results

    print(f"\n  📊 Total chunks extracted: {len(all_vectors):,}")

    # Phase 2: Upload all vectors
    upload_result = upsert_chunks(index_id, all_vectors)

    # Update manifest and results for successful files
    for filename, count in file_vector_counts.items():
        if upload_result['success'] or upload_result['vectors_uploaded'] > 0:
            update_manifest(filename, count)
            results.append({"filename": filename, "status": "success", "chunks": count})
        else:
            results.append({"filename": filename, "status": "partial", "chunks": count})

    # Final summary
    print(f"\n{'═' * 60}")
    print(f"  📋 FINAL SUMMARY")
    print(f"{'═' * 60}")
    for r in results:
        status_icon = "✅" if r.get("status") == "success" else "⚠️" if r.get("status") == "partial" else "❌"
        chunks_info = f" ({r.get('chunks', 0):,} chunks)" if 'chunks' in r else ""
        print(f"  {status_icon} {r.get('filename', 'unknown')}: {r.get('status', 'unknown')}{chunks_info}")
    print(f"{'═' * 60}\n")

    return results


# ---------------- CLI ENTRY -----------------------
if __name__ == "__main__":
    import glob

    if len(sys.argv) < 2:
        print("Usage: python vectors-api_local.py 'pdfs/*.pdf'")
        raise SystemExit(1)

    if not API_KEY:
        print("❌ Error: API_KEY not configured")
        raise SystemExit(1)
        
    if not INDEX_ID:
        print("❌ Error: INDEX_ID not configured")
        raise SystemExit(1)

    all_paths = []
    for arg in sys.argv[1:]:
        all_paths.extend(glob.glob(arg))

    pdf_paths = [p for p in all_paths if p.lower().endswith('.pdf')]
    
    if not pdf_paths:
        print("❌ No PDF files found")
        raise SystemExit(1)
    
    logger.info(f"Found {len(pdf_paths)} PDF files: {pdf_paths}")
    
    print(f"\n{'═' * 60}")
    print(f"  🚀 GRAVIX VECTOR UPLOAD TOOL")
    print(f"{'═' * 60}")
    print(f"  Index ID:  {INDEX_ID}")
    print(f"  API URL:   {BASE_URL}")
    print(f"  Model:     {EMBED_MODEL}")
    print(f"  PDF Files: {len(pdf_paths)}")
    print(f"{'═' * 60}")

    class _F:
        def __init__(self, p):
            self._p = p
            self.name = os.path.basename(p)
        def read(self):
            with open(self._p, 'rb') as f:
                return f.read()

    fake_files = [_F(p) for p in pdf_paths]

    try:
        process_and_upload_pdfs(fake_files, INDEX_ID)
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Upload interrupted by user")
        logger.warning("Upload interrupted by user")
        raise SystemExit(1)
    except Exception as e:
        print(f"\n\n  ❌ Fatal error: {e}")
        logger.exception("Fatal error during upload")
        raise SystemExit(1)

