# src/generation/pipeline.py
from typing import Dict, Any, List, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.generation.synthesizer import synthesize_from_doc
from src.generation.schema import Recommendation

ENABLE_HINTS = True
DEBUG = True

def load_vectorstore(device: str = "cpu") -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings,
        collection_name="aderim",
    )

def build_query(user_text: str, patient: Dict[str, Any] | None = None) -> str:
    base = (user_text or "").strip().lower()
    tags: List[str] = []

    if ENABLE_HINTS:
        if any(w in base for w in ["nourrisson", "3 mois", "fontanelle", "macrocrân", "périmètre crânien"]):
            tags.append("pédiatrie crâne non trauma macrocrânie transfontanellaire échographie")
        if "grossesse" in base or "enceinte" in base:
            tags.append("non ionisant IRM échographie éviter scanner")

    extras: List[str] = []
    if patient:
        extras.append(" ".join(f"{k}:{v}" for k, v in patient.items() if v))

    q = " | ".join([s for s in [base, *tags, *extras] if s])
    return q

def run_pipeline(user_text: str, patient: Dict[str, Any] | None = None, device: str = "cpu") -> Recommendation:
    if not user_text or not user_text.strip():
        raise ValueError("Empty user_text.")

    db = load_vectorstore(device=device)
    q = build_query(user_text, patient)

    results: List[Tuple[Any, float]] = db.similarity_search_with_relevance_scores(q, k=6)
    if not results:
        raise RuntimeError("No retrieval results.")

    # Optional tiny bias for macrocrânie (safe nudge, not required)
    def boost(doc) -> float:
        m = doc.metadata or {}
        mod = (m.get("modalite") or "").lower()
        patho = (m.get("pathologie") or "").lower()
        t = user_text.lower()
        score = 0.0
        if any(w in t for w in ["macrocrân", "périmètre crânien", "fontanelle", "nourrisson", "3 mois"]):
            if "échographie transfontanellaire" in mod:
                score += 0.15
            if "traumatisme" in patho:
                score -= 0.10
        return score

    results = sorted(results, key=lambda rs: boost(rs[0]), reverse=True)

    top_k_docs = [r[0] for r in results[:5]]
    top_doc = top_k_docs[0]

    rec = synthesize_from_doc(top_doc, neighbor_docs=top_k_docs, user_text=user_text)

    if DEBUG:
        for i, (doc, score) in enumerate(results[:min(5, len(results))], start=1):
            print(f"[DEBUG] cand#{i} id={doc.metadata.get('id')} score={score:.3f} modalite={doc.metadata.get('modalite')}")
        print(f"[DEBUG] chosen={top_doc.metadata.get('id')}")

    return rec