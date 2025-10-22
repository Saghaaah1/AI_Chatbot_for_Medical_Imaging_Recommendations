# src/generation/pipeline.py
from typing import Dict, Any, List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.generation.synthesizer import synthesize_from_doc
from src.generation.schema import Recommendation

# safety_notes is optional; guard the import
try:
    from src.generation.guardrails import safety_notes  # def safety_notes(user_text, modality, requires_contrast) -> List[str]
except Exception:  # pragma: no cover
    def safety_notes(*args, **kwargs) -> List[str]:
        return []

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

def build_query(user_text: str, patient: Optional[Dict[str, Any]] = None) -> str:
    base = user_text.strip().lower()
    tags = []
    if any(w in base for w in ["nourrisson", "3 mois", "fontanelle", "macrocrân", "périmètre crânien"]):
        tags.append("pédiatrie crâne non trauma macrocrânie transfontanellaire échographie")
    if "grossesse" in base or "enceinte" in base:
        tags.append("non ionisant IRM échographie éviter scanner")
    extra = " | ".join(tags)
    if patient:
        extras = " ".join(f"{k}:{v}" for k, v in patient.items() if v)
        return f"{base} | {extra} | {extras}".strip(" |")
    return f"{base} | {extra}".strip(" |")

def run_pipeline(
    user_text: str,
    patient: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    k: int = 6,
) -> Recommendation:
    if not user_text or not user_text.strip():
        raise ValueError("Empty user_text.")

    db = load_vectorstore(device=device)
    q = build_query(user_text, patient)

    # Retrieve top-k (with scores so you can log/inspect)
    results = db.similarity_search_with_relevance_scores(q, k=k)
    if not results:
        raise RuntimeError("No retrieval results.")

    # Unpack documents and keep neighbors for alternative suggestions
    docs: List = [doc for (doc, _score) in results]
    top_doc = docs[0]

    # Synthesize a structured recommendation (passes user_text through)
    rec = synthesize_from_doc(
        doc=top_doc,
        neighbor_docs=docs,
        user_text=user_text,
    )

    # Optional: append short safety notes based on modality/contrast context
    # requires_contrast = top_doc.metadata.get("requires_contrast")
    # notes = safety_notes(user_text, rec.modalite_recommandee, requires_contrast)
    # if notes:
    #    rec.justification = (rec.justification.rstrip().rstrip(".") + ". " + " | ".join(notes)).strip()

    # Debug: log top candidates
    for i, (doc, score) in enumerate(results[:min(5, len(results))], start=1):
        mid = doc.metadata.get("id")
        mod = doc.metadata.get("modalite")
        print(f"[DEBUG] cand#{i} id={mid} score={score:.3f} modalite={mod}")
    print(f"[DEBUG] chosen={top_doc.metadata.get('id')}")

    return rec



