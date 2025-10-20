# src/generation/pipeline.py (lean)
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.generation.synthesizer import synthesize_from_doc
from src.generation.schema import Recommendation
from src.generation.guardrails import safety_notes  # optional, but recommended

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
    return f"query: {user_text.strip()}"

def run_pipeline(user_text: str, patient: Dict[str, Any] | None = None, device: str = "cpu") -> Recommendation:
    if not user_text or not user_text.strip():
        raise ValueError("Empty user_text.")
    db = load_vectorstore(device=device)
    q = build_query(user_text, patient)

    # No filter needed if you deleted all “non indiqué” modality entries.
    results = db.similarity_search_with_relevance_scores(q, k=6)
    if not results:
        raise RuntimeError("No retrieval results.")
    top_doc, _ = results[0]

    rec = synthesize_from_doc(top_doc, [d for d, _ in results[:4]], user_text=user_text)

    notes = safety_notes(user_text, rec.modalite_recommandee, top_doc.metadata.get("requires_contrast"))
    if notes:
        rec.justification = (rec.justification.rstrip() + " " + " | ".join(notes)).strip()
    return rec

