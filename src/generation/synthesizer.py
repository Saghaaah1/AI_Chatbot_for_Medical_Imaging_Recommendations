# src/generation/synthesizer.py
from typing import Dict, Any, List, Optional
from .schema import Recommendation

def _has_word(text: str, *words: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(w.lower() in t for w in words)

def _poplist(meta: Dict[str, Any]) -> List[str]:
    pops = meta.get("populations") or meta.get("population")
    if isinstance(pops, list):
        return [str(x).lower() for x in pops]
    if isinstance(pops, str):
        return [pops.lower()]
    return []

def _symplist(meta: Dict[str, Any]) -> List[str]:
    s = meta.get("symptomes") or meta.get("symptômes")
    if isinstance(s, list):
        return [str(x).lower() for x in s]
    if isinstance(s, str):
        return [s.lower()]
    return []

def _apply_clinical_rules(
    meta: Dict[str, Any],
    user_text: str,
) -> List[str]:
    """
    Return a list of short, context-aware safety/alternative notes to append to the justification.
    """
    notes: List[str] = []
    modality = str(meta.get("modalite", "") or meta.get("modalité", ""))
    populations = _poplist(meta)
    symptomes = _symplist(meta)
    requires_contrast = meta.get("requires_contrast", None)

    # 1) Alternatives for IRM-first
    if _has_word(modality, "irm") and not _has_word(modality, "angio"):
        notes.append("Alternative : scanner si IRM contre-indiquée/indisponible")

    # 2) Angioscanner safety (iodinated contrast)
    if _has_word(modality, "angioscanner") or _has_word(modality, "angio-ct", "angio ct", "cta"):
        notes.append("Angioscanner : nécessite PDC iodé ; créatinine si >60 ans ou ATCD rénaux ; protocole en cas d’allergie iodée")

    # 3) CT (scanner) with contrast — general safety
    if (requires_contrast in (True, "true", "depends")) and _has_word(modality, "scanner"):
        notes.append("CT injecté : vérifier fonction rénale et antécédents d’allergie (prémédication si besoin)")

    # 4) IRM safety basics
    if _has_word(modality, "irm"):
        notes.append("IRM : vérifier compatibilité des dispositifs implantables ; claustrophobie à anticiper")
        if "enceinte" in populations or "grossesse" in symptomes or _has_word(user_text, "grossesse", "enceinte"):
            notes.append("Grossesse : IRM à éviter si <3 mois si non urgent ; CT à éviter ; privilégier US/IRM selon contexte")

    # 5) Thunderclap headache: ensure the combo is explicit
    if _has_word(meta.get("pathologie",""), "coup de tonnerre", "ictale") or _has_word(user_text, "coup de tonnerre", "céphalée brutale"):
        if not _has_word(modality, "angio"):
            notes.append("Complément : ajouter angioscanner en plus du scanner sans injection si suspicion d’HSA")

    # 6) Pediatrics macrocrania → IRM if signs of raised ICP in user text
    if _has_word(meta.get("pathologie",""), "périmètre crânien") and _has_word(user_text, "vomissements", "bombement", "fontanelle bombante"):
        notes.append("Pédiatrie : IRM si signes d’HTIC ou augmentation brutale du périmètre crânien")

    # 7) Oncology context headache → allow IRM alternative
    if _has_word(meta.get("pathologie",""), "oncologique") or _has_word(user_text, "oncolog"):
        notes.append("Alternative : IRM possible selon préférence clinique/indication")

    # Deduplicate while preserving order
    dedup = []
    seen = set()
    for n in notes:
        if n not in seen:
            dedup.append(n)
            seen.add(n)
    return dedup


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
    Now accepts user_text so we can choose a smarter alternative and add safety notes.
    """
    meta = doc.metadata

    modalite = meta.get("modalite", "Imagerie (à préciser)")
    urgence  = meta.get("urgence", meta.get("urgence_enum", "standard"))
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
    if not just:
        just = "Recommandation conforme à la section citée."

    # Build reference
    ref = make_reference(meta)

    # Alternative (prefer within same pathology, non-ionising if pregnancy/child)
    alt = pick_alternative([d.metadata for d in neighbor_docs], meta.get("id",""), user_text)

    # Apply clinical notes
    extra_notes = _apply_clinical_rules(meta, user_text)
    if extra_notes:
        just = just.rstrip(".")
        just = just + ". " + " | ".join(extra_notes)

    rec = Recommendation(
        modalite_recommandee = modalite,
        symptomes_cles = sympt[:3] if sympt else [],
        hypothese_clinique = meta.get("pathologie"),
        urgence = urgence,
        delai_recommande = delai,
        justification = just,
        reference = ref,
        alternative = alt
    )
    return rec
