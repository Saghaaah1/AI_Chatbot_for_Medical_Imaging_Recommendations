# src/retrieval.py
"""
Retrieval:
- Opens the local Chroma DB with the right embedding model
- Adds E5 'query:' prefix automatically
- Auto-detects population from natural language (enfant/femme/homme/personne âgée/femme enceinte)
- Optional CLI for quick tests:
    uv run python -m src.rag.retrieval "dyspnée aiguë adulte"
    uv run python -m src.rag.retrieval --systeme thorax "dyspnée aiguë"
    uv run python -m src.rag.retrieval --population enfant "douleur abdominale"
"""

import os
import sys
from typing import Dict, List, Optional
import os as _os

# Vector DB + embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langsmith import traceable

_os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
_os.environ.setdefault("LANGCHAIN_PROJECT", "Medical-Imaging-RAG")


# ---- Config (matches your indexer) ----
DB_DIR = "vectorstore"
COLLECTION = "aderim"
EMBEDDING_MODEL = os.getenv("EMBED_MODEL_PATH", "intfloat/multilingual-e5-small")

# Device (M1 GPU if available)
try:
    import torch
    DEVICE = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

DEFAULT_K = 6
QUERY_PREFIX = "query: "

# ----------------- Building blocks -----------------
def get_embeddings():
    """Use the SAME model & settings as indexing (normalize=True)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_store():
    """Open the persisted Chroma collection and attach the embedding function (for query encoding)."""
    emb = get_embeddings()
    db = Chroma(
        persist_directory=DB_DIR,
        collection_name=COLLECTION,
        embedding_function=emb,
    )
    return db

def _infer_population_filter(user_query: str) -> Optional[Dict[str, str]]:
    """
    Detects population from natural language.
    Adjust right-hand values to match how your JSON stores 'population' in metadata.
    """
    q = (user_query or "").lower()

    # Pregnancy first (most specific)
    if any(k in q for k in ["enceinte", "grossesse", "gestante", "pregnant"]):
        return {"population": "femme_enceinte"}

    # Sex
    if any(k in q for k in ["femme", "fémin", "femin", "woman", "female"]):
        return {"population": "femme"}
    if any(k in q for k in ["homme", "masculin", "man", "male"]):
        return {"population": "homme"}

    # Age groups
    if any(k in q for k in ["personne âgée", "personne agee", "gériat", "geriat", "elderly", "senior", "agé", "agee"]):
        return {"population": "personne_agee"}
    if any(k in q for k in ["enfant", "pédiat", "pediat", "child", "paediatric", "pediatric"]):
        return {"population": "enfant"}
    if "adulte" in q or "adult" in q:
        return {"population": "adulte"}

    return None

# ---------------------------
# RETRIEVAL 
# ---------------------------
@traceable(name="retrieve", run_type="retriever")
def retrieve(query_text: str, k: int = DEFAULT_K, filters: Optional[Dict[str, str]] = None) -> List[Document]:
    """Main retrieval function we'll reuse everywhere."""
    db = get_store()

    q = query_text.strip()
    if not q.lower().startswith(QUERY_PREFIX):
        q = QUERY_PREFIX + q

    # Auto population filter from NL; explicit filters win if they provide 'population'
    auto = _infer_population_filter(query_text)
    final_filter = dict(filters or {})
    if auto and "population" not in final_filter:
        final_filter.update(auto)
        
    # Attach run attributes to help debug in LangSmith
    try:
        from langsmith.run_helpers import trace
        trace.set_attributes({
            "k": k,
            "filters": final_filter or {},
            "auto_population": (auto or {}).get("population", None),
            "device": DEVICE,
            "model": EMBEDDING_MODEL,
            "collection": COLLECTION,
        })
    except Exception:
        pass


    if final_filter:
        return db.similarity_search(q, k=k, filter=final_filter)
    return db.similarity_search(q, k=k)




def pretty(doc: Document) -> str:
    """Short one-line display for CLI/tests."""
    m = doc.metadata or {}
    return f"{m.get('id')} | {m.get('systeme')} | {m.get('modalite')} | {m.get('pathologie')} | pop={m.get('population')}"

# ----------------- Optional CLI (dev helper) -----------------
def main():
    if len(sys.argv) == 1:
        print(
            "Usage:\n"
            "  uv run python -m src.rag.retrieval \"dyspnée aiguë adulte\"\n"
            "  uv run python -m src.rag.retrieval --systeme thorax \"dyspnée aiguë\"\n"
            "  uv run python -m src.rag.retrieval --population enfant \"douleur abdominale\"\n"
        )
        return

    args = sys.argv[1:]
    filters: Dict[str, str] = {}

    # Tiny flag parser: --systeme VALUE, --population VALUE
    i = 0
    parts = []
    while i < len(args):
        if args[i] == "--systeme" and i + 1 < len(args):
            filters["systeme"] = args[i + 1]
            i += 2
        elif args[i] == "--population" and i + 1 < len(args):
            filters["population"] = args[i + 1].lower()
            i += 2
        else:
            parts.append(args[i])
            i += 1

    query = " ".join(parts).strip()
    print("Query:", query)
    if filters:
        print("Filters:", filters)

    hits = retrieve(query, k=DEFAULT_K, filters=filters)
    print("\nTop-k:")
    for idx, d in enumerate(hits, 1):
        print(f"#{idx}", pretty(d))

if __name__ == "__main__":
    main()
