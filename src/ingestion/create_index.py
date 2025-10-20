# src/ingestion/create_index.py
import json, os, glob
from typing import Dict, Any, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

VECTOR_DIR = "vectorstore"
DATA_DIR = "data"
COLLECTION = "aderim"

def sanitize_metadata(d: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma only accepts str/int/float/bool/None. JSON-encode lists/dicts."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            out[k] = json.dumps(v, ensure_ascii=False)
        elif v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out

def build_canonical_text(rec: Dict[str, Any]) -> str:
    get = rec.get
    systeme   = get("systeme", "")
    patho     = get("pathologie", "")
    modalite  = get("modalite", "")
    urgence   = get("urgence_enum", get("urgence", "")) or ""
    resume    = get("resume", "").strip()
    pops      = ", ".join(get("populations", []) or ([get("population")] if get("population") else []))
    sympt     = "; ".join(get("symptomes", []))
    posi      = "; ".join(get("indications_positives", []))
    nega      = "; ".join(get("indications_negatives", []))
    contre    = "; ".join(get("contre_indications", []))
    delai     = get("delai_recommande", "")
    source    = get("source", "")
    year      = get("year", "")
    refsec    = get("reference_section", "")

    parts = [
        f"[Système] {systeme}",
        f"[Pathologie] {patho}",
        f"[Population] {pops}" if pops else "",
        f"[Symptômes] {sympt}" if sympt else "",
        f"[Modalité] {modalite}",
        f"[Urgence] {urgence}" if urgence else "",
        f"[Délai] {delai}" if delai else "",
        f"[Indiqué] {posi}" if posi else "",
        f"[Non indiqué] {nega}" if nega else "",
        f"[Contre-indications] {contre}" if contre else "",
        f"[Résumé] {resume}" if resume else "",
        f"[Référence] {source} {year} — {refsec}".strip() if (source or refsec or year) else "",
    ]
    canonical = " | ".join([p for p in parts if p])
    return f"passage: {canonical}"

def load_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            records.append(data)
        else:
            print(f"⚠️  Ignoring {path}: not JSON object/array.")
    return records

def to_document(rec: Dict[str, Any]) -> Document:
    text = build_canonical_text(rec)
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
        "contre_indications": rec.get("contre_indications", []),
        "delai_recommande": rec.get("delai_recommande", ""),
        "ionisant": rec.get("ionisant", None),
        "requires_contrast": rec.get("requires_contrast", None),  # True/False/"depends"
        "reference_section": rec.get("reference_section", ""),
        "source": rec.get("source", ""),
        "year": rec.get("year", ""),
        "link": rec.get("link", ""),
        "synonymes": rec.get("synonymes", []),
    }
    meta = sanitize_metadata(meta_raw)
    return Document(page_content=text, metadata=meta)

def main():
    records = load_records()
    if not records:
        print(" No JSON records found in /data. Add your files first.")
        return

    docs = [to_document(r) for r in records]

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DIR,
        collection_name=COLLECTION,
    )
    print(f" Indexed {len(docs)} records into '{VECTOR_DIR}/' (collection='{COLLECTION}', device={DEVICE})")

if __name__ == "__main__":
    main()
