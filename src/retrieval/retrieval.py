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
from sentence_transformers import CrossEncoder
from functools import lru_cache

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

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """
    Load the cross-encoder reranking model only once.

    Why we need this:
    -----------------
    - Our vector store (Chroma + embeddings) is good to find "candidate" ADERIM
      documents, but it can sometimes be fuzzy.
    - The cross-encoder looks at (query, full document text) together and gives
      a more precise relevance score.
    - We will use this score later to reorder (rerank) the candidate documents.

    Why @lru_cache(maxsize=1):
    --------------------------
    - Loading the model from disk (or internet cache) is slow.
    - With this decorator, the first time we call get_reranker(), it loads
      the model and stores it in memory.
    - Next calls reuse the same model instead of reloading it.

    Model choice:
    -------------
    - We use "BAAI/bge-reranker-base".
    - It is a general-purpose reranker, good at semantic matching, and light
      enough to run on a laptop.
    """
    model_name = "BAAI/bge-reranker-base"
    reranker = CrossEncoder(model_name)
    return reranker

def rerank_documents(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """
    Rerank a list of candidate documents using the cross-encoder reranker.

    Parameters
    ----------
    query : str
        The clinical question / scenario asked by the user.
    docs : List[Document]
        Documents returned by the vector store (Chroma).
        Each Document has:
          - page_content : the text we built from ADERIM fields
          - metadata     : a dict with fields like "id", "pathologie", "modalite", etc.
    top_k : int
        How many documents we want to keep after reranking.
        Example: if we pass 5, we keep the 5 best candidates.

    Returns
    -------
    List[Document]
        The same documents, but:
          - sorted by relevance according to the reranker
          - truncated to the top_k most relevant ones.
    """
    # If there are no documents, just return an empty list
    if not docs:
        return []

    # 1) Get the reranker model (loaded once thanks to get_reranker())
    reranker = get_reranker()

    # 2) Build a list of (query, text) pairs.
    #    For each Document, we use its page_content as the text to score.
    pairs = []
    for d in docs:
        text = d.page_content
        pairs.append((query, text))

    # 3) Ask the reranker to score each (query, text) pair.
    #    It returns a list of scores, one per document.
    scores = reranker.predict(pairs)
    # Now: scores[i] is the relevance score for docs[i].
    # Higher score = document is more relevant to the query.

    # 4) Attach each score to its document so we can sort them together.
    docs_with_scores = list(zip(docs, scores))

    # 5) Sort documents by score in descending order (best first).
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)

    # 6) Keep only the top_k documents (or fewer if we had less docs).
    top_docs = [doc for doc, score in docs_with_scores[:top_k]]

    # 7) Return the reranked list
    return top_docs


@traceable(name="retrieve_with_reranker", run_type="retriever")
def retrieve_documents_with_reranker(
    query: str,
    k: int = 5,
) -> List[Document]:
    """
    Two-stage retrieval: vector store + reranker.

    Step 1: Use the vector store (Chroma + embeddings) to get 'initial_k'
            candidate ADERIM documents.
            - This is fast but approximate.
    Step 2: Use the cross-encoder reranker to score each candidate more precisely
            with the full text, and keep only the top 'k'.

    Parameters
    ----------
    query : str
        Clinical question / scenario (what the user types).
    k : int
        Final number of documents you want after reranking (e.g. 3 or 5).
    initial_k : int
        Number of documents to request from the vector store before reranking.
        - Example: initial_k = 20, k = 5
        - Rule: initial_k should be >= k.

    Returns
    -------
    List[Document]
        The reranked top-k ADERIM documents (LangChain Document objects).
    """
    # 1) First-stage: get candidate documents from the vector store
    store = get_store()
    initial_docs = store.similarity_search(query, k=k)

    # If nothing was found, return an empty list
    if not initial_docs:
        return []

    # 2) Second-stage: rerank these candidates with the cross-encoder
    final_docs = rerank_documents(query, initial_docs, top_k=k)

    # 3) Return the reranked top-k
    return final_docs

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
