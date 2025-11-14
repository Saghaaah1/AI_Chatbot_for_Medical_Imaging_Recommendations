# src/ingestion/create_index.py
"""
Create a searchable vector index (Chroma) from ADERIM-style JSON files.

What this script does:
1) Loads all JSON files from ./data (each file is a list of records).
2) Builds a clear, canonical text string from each record (e.g., Pathologie, Symptômes, Modalité...).
3) Converts those texts to embeddings using the multilingual-e5 model.
4) Saves everything into a persistent Chroma vector store (./vectorstore).
5) (Optional) Splits very long texts into overlapping chunks for better retrieval.

"""
import hashlib           # used to create a fingerprint of text to detect duplicates
import unicodedata       # used to normalize text (fix accents/spacing so they are consistent)
from pathlib import Path # nicer way to handle file/folder paths than plain strings
import json
import os
import glob
import shutil
from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "Imagerie-RAG")

# ----------------------------
# Hardware detection (GPU/CPU)
# ----------------------------
# We'll try to use Apple GPU (M1/M2) if available, else CPU.

try:
    import torch
    # torch.backends.mps.is_available() is True on Apple Silicon (M1/M2) with Metal (Apple's GPU API)
    if torch.backends.mps.is_available():
        DEVICE = "mps"      #  Use Apple GPU (fast + saves memory)
    
    # Otherwise, fall back to CPU (still works, just slower)
    else:
        DEVICE = "cpu"

# If for some reason torch fails to import, default to CPU so the script still runs
except Exception:
    DEVICE = "cpu"

# ----------------------------
# Configuration 
# ----------------------------
DATA_DIR            = "data"            # Folder where *.json files live
VECTOR_DIR          = "vectorstore"     # Folder to store the persistent Chroma index
COLLECTION_NAME     = "aderim"          # Name of the Chroma collection
EMBEDDING_MODEL     = "intfloat/multilingual-e5-small"  # Good multilingual embedding model

NUKE_BEFORE_BUILD   = True   # True = delete existing VECTOR_DIR before building (clean rebuild)
ENABLE_CHUNKING     = False   # True = split long texts to improve recall
CHUNK_SIZE_CHARS    = 800     # Max characters per chunk (only used if ENABLE_CHUNKING=True)
CHUNK_OVERLAP       = 120     # Overlap between chunks to avoid cutting important sentences

# ---------- Small helper functions ----------

def u_normalize(text: str) -> str:
    """
    Make all texts look consistent before they get turned into embeddings.
    - NFC normalization: standardizes Unicode (accents, special chars)
    - ' '.join(t.split()): collapses all weird spaces/tabs/newlines into single spaces
    Why? This avoids tiny formatting differences producing different vectors.
    """
    t = unicodedata.normalize("NFC", text or "")
    return " ".join(t.split())

def text_hash(text: str) -> str:
    """
    Create a stable fingerprint ("hash") of some text using SHA-256.
    If two texts have the exact same content, they get the same hash.
    We use this to SKIP duplicates when building the index.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# -----------------------------------------------------------------------------------
# Helper: Chroma only supports simple metadata types. Convert lists/dicts to strings.
# -----------------------------------------------------------------------------------
def sanitize_metadata(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure metadata values are only str/int/float/bool/None.
    Lists/dicts are JSON-encoded to strings so Chroma can store them.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            out[k] = json.dumps(v, ensure_ascii=False)
        elif v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out

# -------------------------------------------------------------------
# Helper: Build a canonical, human-readable text from a JSON record.
# This is the text we embed.
# -------------------------------------------------------------------
def build_canonical_text(rec: Dict[str, Any]) -> str:
    get = rec.get

    systeme   = get("systeme", "")
    patho     = get("pathologie", "")
    modalite  = get("modalite", "")
    resume    = (get("resume", "") or "").strip()

    # Support either `populations: [...]` or `population: "..."`.
    pops_list = get("populations", []) or ([get("population")] if get("population") else [])
    pops      = ", ".join([p for p in pops_list if p])

    sympt     = "; ".join(get("symptomes", []) or [])
    posi      = "; ".join(get("indications_positives", []) or [])
    nega      = "; ".join(get("indications_negatives", []) or [])
    syns      = ", ".join(get("synonymes", []) or [])

    parts = [
        f"[Système] {systeme}" if systeme else "",
        f"[Pathologie] {patho}" if patho else "",
        f"[Population] {pops}" if pops else "",
        f"[Symptômes] {sympt}" if sympt else "",
        f"[Modalité] {modalite}" if modalite else "",
        f"[Indiqué] {posi}" if posi else "",
        f"[Non indiqué] {nega}" if nega else "",
        f"[Résumé] {resume}" if resume else "",
        f"[Synonymes] {syns}" if syns else "",
    ]

    canonical = " | ".join([p for p in parts if p])
    # Make text consistent (accents/whitespace) so embeddings are stable.   
    canonical = u_normalize(canonical)

    # Multilingual-E5 expects prefixes: "passage:" for documents, "query:" for queries.
    return f"passage: {canonical}"


# ------------------------------------------------------------
# Helper: Load all JSON records from DATA_DIR/*.json
# Each JSON file can be a single object or a list of objects.
# ------------------------------------------------------------
def load_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    paths = glob.glob(os.path.join(DATA_DIR, "*.json"))

    if not paths:
        print(f" No JSON files found in '{DATA_DIR}/'. Add files and retry.")
        return records

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f" Skipping {path}: {e}")
            continue

        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            records.append(data)
        else:
            print(f" Ignoring {path}: JSON must be object or array.")

    print(f" Loaded {len(records)} records from '{DATA_DIR}/'")
    return records

# ---------------------------------------------------------------------------------
# Helper: Simple fixed-size chunking with overlap (optional).
# Used if some texts are very long. Overlap prevents cutting key info.
# ---------------------------------------------------------------------------------
def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

# --------------------------------------------------------------------------------
# Convert one JSON record to one or more Documents (if chunking is enabled).
# We keep a parent_id and chunk indices so we can group results later.
# --------------------------------------------------------------------------------

def record_to_documents(rec: Dict[str, Any]) -> List[Document]:
    text = build_canonical_text(rec)

    # Rich metadata to filter later (e.g., by systeme, population, etc.).
    meta_raw = {
        "id": rec.get("id", ""),
        "systeme": rec.get("systeme", ""),
        "pathologie": rec.get("pathologie", ""),
        "modalite": rec.get("modalite", ""),
        "urgence": rec.get("urgence_enum", rec.get("urgence", "")),
        "populations": rec.get("populations") or ([rec.get("population")] if rec.get("population") else []),
        "symptomes": rec.get("symptomes", []),
        "indications_positives": rec.get("indications_positives", []),
        "indications_negatives": rec.get("indications_negatives", []),
        "delai_recommande": rec.get("delai_recommande", ""),
        "ionisant": rec.get("ionisant", None),
        "requires_contrast": rec.get("requires_contrast", None),
        "reference_section": rec.get("reference_section", ""),
        "source": rec.get("source", ""),
        "year": rec.get("year", ""),
        "link": rec.get("link", ""),
        "synonymes": rec.get("synonymes", []),
    }
    # Derive a single 'population' string for filtering (e.g., first or most specific)
    pops_list = rec.get("populations") or ([rec.get("population")] if rec.get("population") else [])
    pops_list = [p for p in pops_list if p]
    meta_raw["population"] = (pops_list[0] if pops_list else "")
    meta = sanitize_metadata(meta_raw)

    parent_id = rec.get("id", "")
    base_meta = {**meta, "parent_id": parent_id}

    # If chunking is off, return a single Document.
    if not ENABLE_CHUNKING:
        return [Document(page_content=text, metadata={**base_meta, "chunk_idx": 0, "chunk_total": 1})]

    # If chunking is on, split and create one Document per chunk.
    pieces = chunk_text(text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP)
    total = len(pieces)
    docs = []
    for j, ch in enumerate(pieces):
        docs.append(Document(page_content=ch, metadata={**base_meta, "chunk_idx": j, "chunk_total": total}))
    return docs


# ------------
# Main script
# ------------
@traceable(name="build_index", tags=["ingestion"], metadata={"collection": COLLECTION_NAME})
def main():
    # Optional: start from a clean slate by deleting any previous index.
    if NUKE_BEFORE_BUILD and os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)

    # 1) Load all records from DATA_DIR.
    records = load_records()
    if not records:
        print(" No records found. Add JSON files to ./data and rerun.")
        return

    # 2) Convert records to Documents.
    docs: List[Document] = []
    ids:  List[str]      = []

    # Keep fingerprints (hashes) of texts we've already indexed
    seen_hashes = set()
    for i, rec in enumerate(records):
        # Use the record's "id" if present; otherwise make a stable fallback.
        base_id = rec.get("id", f"auto-{i}")

        rec_docs = record_to_documents(rec)

        # If chunking is enabled, each chunk gets a unique suffix (-0, -1, ...)
        for j, d in enumerate(rec_docs):
            # Compute a fingerprint of the text (after the normalization step)
            h = text_hash(d.page_content)
            if h in seen_hashes:
                # already indexed this exact text -> skip
                continue
            seen_hashes.add(h)
            suffix = f"-{j}" if ENABLE_CHUNKING else ""
            ids.append(f"{base_id}{suffix}")
            docs.append(d)
    print(f" Prepared {len(docs)} unique documents (after dedup).")

    # 3) Create an embedding model (multilingual, normalized vectors recommended).
    # If we’re on Apple GPU (mps), use float16 to save memory.
    model_kwargs = {"device": DEVICE}
    try:
        import torch
        if DEVICE == "mps":
            model_kwargs = {"device": DEVICE}
    except Exception:
        pass

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4) Build and persist the Chroma vector store.
    # Collection_metadata sets cosine distance, which works well with e5 models.
    print(f" Building vector index in '{VECTOR_DIR}/' (collection='{COLLECTION_NAME}', device={DEVICE})...")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DIR,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
        ids=ids,  # stable IDs prevent duplicates on re-runs
    )
    print(" Chroma ready (langchain_chroma). Fast ANN via hnswlib is enabled if installed.")
    print(f" Done! Indexed {len(docs)} documents into '{VECTOR_DIR}/' (collection='{COLLECTION_NAME}').")


    # Quick test (optional) – checks that retrieval returns something
    TRY_QUICK_TEST = True   # set to False if you don’t want to run this

    if TRY_QUICK_TEST:
        try:
            test_query = "query: Dyspnée aiguë chez l’adulte"
            hits = db.similarity_search(test_query, k=3)
            if not hits:
                print(" Quick test: no results (check your data and model).")
            else:
                # show a tiny peek (pathologie + first 80 chars)
                print(" Quick test: got", len(hits), "hits. First:")
                h0 = hits[0]
                print("  •", h0.metadata.get("pathologie", "(?)"),
                      "—", h0.page_content[:80].replace("\n", " "), "…")
        except Exception as e:
            print(" Quick test failed:", e)
    
    
    # Tip to query later:
    # Use: similarity_search("query: <your question>", k=5)
    # because e5 models expect 'query:' prefix for queries and 'passage:' for docs.
    return {
        "docs_indexed": len(docs),
        "device": DEVICE,
        "collection": COLLECTION_NAME,
    }

if __name__ == "__main__":
    main()
