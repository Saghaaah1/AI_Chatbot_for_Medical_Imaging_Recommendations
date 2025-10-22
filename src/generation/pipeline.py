# src/generation/pipeline.py
from typing import Dict, Any, List, Tuple, Optional
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

# ---------- helpers: soft filter + small rerank + disambiguation ----------

def soft_context_filter(meta: Dict[str, Any], user_text: str) -> bool:
    """
    Soft preference: if pregnancy/child context, prefer non-ionising.
    We'll apply this first; if it removes everything, we'll fall back.
    """
    t = (user_text or "").lower()
    wants_non_ionising = any(k in t for k in ["enceinte", "grossesse", "enfant", "pédiat"])
    if not wants_non_ionising:
        return True
    # keep non-ionising; allow ionising to be filtered out in the first pass
    return (meta.get("ionisant") is False) or (meta.get("ionisant") is None)

def boost_score_for_context(doc, user_text: str) -> float:
    """
    Tiny contextual nudges that don't override the base relevance score.
    """
    m = doc.metadata or {}
    mod = (m.get("modalite") or "").lower()
    patho = (m.get("pathologie") or "").lower()
    t = (user_text or "").lower()
    score = 0.0

    # Macrocranie/nourrisson -> nudge to transfontanellaire
    if any(w in t for w in ["macrocrân", "périmètre crânien", "fontanelle", "nourrisson", "3 mois"]):
        if "échographie transfontanellaire" in mod:
            score += 0.15
        if "traumatisme" in patho:
            score -= 0.10

    # Pregnancy/child -> nudge non-ionising up
    if any(k in t for k in ["enceinte", "grossesse", "enfant", "pédiat"]):
        if m.get("ionisant") is False:
            score += 0.10
        if m.get("ionisant") is True:
            score -= 0.05

    return score

def disambiguation_note(user_text: str, docs: List[Any]) -> Optional[str]:
    """
    If the top 2 candidates diverge meaningfully (e.g., CXR vs angio-CT),
    add a short note to guide clinicians. We don't change schema/IO here.
    """
    if len(docs) < 2:
        return None

    m1 = docs[0].metadata or {}
    m2 = docs[1].metadata or {}
    mod1 = (m1.get("modalite") or "").lower()
    mod2 = (m2.get("modalite") or "").lower()

    # Only trigger if modalities are clearly different families
    families = lambda s: ("angio" if "angio" in s else
                          "scanner" if "scanner" in s or "ct" in s else
                          "radio" if "radio" in s or "cliché" in s else
                          "irm" if "irm" in s else
                          "us" if "échographie" in s or "ultras" in s else
                          "other")
    f1, f2 = families(mod1), families(mod2)
    if f1 == f2:
        return None

    t = (user_text or "").lower()

    # Some handy domain-specific nudges
    if ("ep" in t or "embolie" in t) and {"radio", "angio"} == {f1, f2}:
        return "Si suspicion d’EP élevée (score clinique), privilégier l’angioscanner pulmonaire ; la radiographie ne doit pas retarder l’examen de référence."
    if any(k in t for k in ["enceinte", "grossesse"]) and {"scanner", "irm"} == {f1, f2}:
        return "Grossesse : privilégier les techniques non ionisantes (IRM/échographie) ; le scanner n’est envisagé qu’en situation vitale."
    if any(k in t for k in ["enfant", "pédiat"]) and {"scanner", "us"} == {f1, f2}:
        return "Pédiatrie : privilégier l’échographie/IRM ; éviter l’irradiation si un examen non ionisant répond à la question."
    if {"radio", "scanner"} == {f1, f2} and any(k in t for k in ["douleur thoracique", "dissection", "instabilité"]):
        return "Douleur thoracique à haut risque : ne pas retarder un scanner/angioscanner par une radiographie si instabilité ou dissection suspectée."
    return None

# --------------------------------------------------------------------------
# detects which "systeme" are relevant from user_text
def detect_allowed_systems(user_text: str) -> list[str]:
    t = (user_text or "").lower()
    allow: list[str] = []

    # Thorax / cardio (chest pain, dyspnea, PE…)
    if any(w in t for w in ["douleur thorac", "thorax", "dyspn", "hémopty", "sibil", "crépit", "ep ", "embolie", "oap"]):
        allow.extend(["thorax", "cardio"])

    # Neuro (headache, neuro trauma…)
    if any(w in t for w in ["céphal", "migraine", "raideur de nuque", "hsa", "coup de tonnerre", "traumatisme crân", "macrocrân", "périmètre crânien", "fontanelle"]):
        allow.append("neuro")

    # ORL (facial pain, stridor, cervical mass…)
    if any(w in t for w in ["douleur de la face", "névralgie", "stridor", "masse cervicale", "orl"]):
        allow.append("orl")

    # Digestif (abdominal pain, appendicitis…)
    if any(w in t for w in ["douleur abdom", "appendic", "colique hépat", "diverticul", "foie", "hépat", "pancréas"]):
        allow.append("digestif")

    # Rachis
    if any(w in t for w in ["cervicalgie", "radiculalgie", "rachis", "traumatisme cervical"]):
        allow.append("rachis")

    # De-duplicate
    return list(dict.fromkeys(allow))

def run_pipeline(user_text: str, patient: Dict[str, Any] | None = None, device: str = "cpu") -> Recommendation:
    if not user_text or not user_text.strip():
        raise ValueError("Empty user_text.")

    db = load_vectorstore(device=device)
    q = build_query(user_text, patient)

    allowed = detect_allowed_systems(user_text)

    if allowed:
        raw_results: List[Tuple[Any, float]] = db.similarity_search_with_relevance_scores(
        q, k=12, filter={"systeme": {"$in": allowed}})
    
    else:
        raw_results: List[Tuple[Any, float]] = db.similarity_search_with_relevance_scores(q, k=12)

    if not raw_results:
        raise RuntimeError("No retrieval results.")


    # 1) try soft context filter
    filtered = [(d, s) for (d, s) in raw_results if soft_context_filter(d.metadata or {}, user_text)]

    # fall back to unfiltered if we filtered everything out
    results = filtered if filtered else raw_results

    # 2) apply tiny contextual boost (stable sort by base score + small adjustment)
    results = sorted(
        results,
        key=lambda rs: (rs[1], boost_score_for_context(rs[0], user_text)),
        reverse=True,
    )

    top_k_docs = [r[0] for r in results[:5]]
    top_doc = top_k_docs[0]

    # 3) synthesize recommendation
    rec = synthesize_from_doc(top_doc, neighbor_docs=top_k_docs, user_text=user_text)

    # 4) optional disambiguation hint
    note = disambiguation_note(user_text, top_k_docs[:2])
    if note:
        rec.justification = (rec.justification.rstrip() + " " + note).strip()

    if DEBUG:
        for i, (doc, score) in enumerate(results[:min(5, len(results))], start=1):
            print(f"[DEBUG] cand#{i} id={doc.metadata.get('id')} score={score:.3f} modalite={doc.metadata.get('modalite')}")
        print(f"[DEBUG] chosen={top_doc.metadata.get('id')}")

    return rec
