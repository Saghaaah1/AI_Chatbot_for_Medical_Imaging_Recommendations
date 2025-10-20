# src/generation/synthesizer.py
from typing import Dict, Any, List, Optional
from .schema import Recommendation

def make_reference(meta: Dict[str, Any]) -> str:
    source = meta.get("source") or "ADERIM"
    year = meta.get("year") or "2025"
    refsec = meta.get("reference_section") or ""
    return f"{source} {year} — {refsec}".strip()

def pick_alternative(candidates: List[Dict[str, Any]], current_id: str, user_text: str) -> Optional[str]:
    """
    Prefer a different modality within the same pathologie.
    If user_text hints pregnancy/child, prefer non-ionising alternatives (IRM/US).
    """
    t = (user_text or "").lower()
    wants_non_ionising = ("enceinte" in t) or ("grossesse" in t) or ("enfant" in t) or ("pédiat" in t)

    base = next((m for m in candidates if m.get("id") == current_id), None)
    if not base:
        return None

    same_pathos = [
        m for m in candidates
        if m.get("pathologie") == base.get("pathologie") and m.get("id") != current_id
    ]

    # Prefer non-ionising if requested by context
    if wants_non_ionising:
        for m in same_pathos:
            mod = m.get("modalite")
            if mod and mod != base.get("modalite") and m.get("ionisant") is False:
                return mod

    # Otherwise, any different modality within same pathology
    for m in same_pathos:
        mod = m.get("modalite")
        if mod and mod != base.get("modalite"):
            return mod

    return None

def synthesize_from_doc(doc, neighbor_docs: List, user_text: str) -> Recommendation:
    """
    Build a structured, cited recommendation from a retrieved doc.
    Now accepts user_text so we can choose a smarter alternative.
    """
    meta = doc.metadata

    modalite = meta.get("modalite", "Imagerie (à préciser)")
    urgence  = meta.get("urgence", "standard")
    delai    = meta.get("delai_recommande")

    # Pull symptoms from canonical text: "passage: ... | [Symptômes] a; b; c | ..."
    text = doc.page_content or ""
    sympt: List[str] = []
    if "[Symptômes]" in text:
        try:
            seg = text.split("[Symptômes]")[1].split("|")[0].strip()
            sympt = [s.strip() for s in seg.split(";") if s.strip()]
        except Exception:
            sympt = []

    # Justification: prefer [Résumé], else derive from [Indiqué]
    just = ""
    if "[Résumé]" in text:
        try:
            just = text.split("[Résumé]")[1].split("|")[0].strip()[:280]
        except Exception:
            just = ""
    if not just and "[Indiqué]" in text:
        try:
            j2 = text.split("[Indiqué]")[1].split("|")[0].strip()
            just = ("Indiqué lorsque " + j2)[:260]
        except Exception:
            pass

    ref = make_reference(meta)
    alt = pick_alternative([d.metadata for d in neighbor_docs], meta.get("id",""), user_text)

    rec = Recommendation(
        modalite_recommandee = modalite,
        symptomes_cles = sympt[:3] if sympt else [],
        hypothese_clinique = meta.get("pathologie"),
        urgence = urgence,
        delai_recommande = delai,
        justification = just or "Recommandation conforme à la section citée.",
        reference = ref,
        alternative = alt
    )
    return rec
