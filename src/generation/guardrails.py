# src/generation/guardrails.py
from typing import List

def safety_notes(user_text: str, modalite: str, requires_contrast) -> List[str]:
    t = (user_text or "").lower()
    m = (modalite or "").lower()

    pregnant = ("enceinte" in t) or ("grossesse" in t)

    # Modality heuristics
    is_ct   = ("scanner" in m) or ("angioscanner" in m) or (" ct" in m) or m.startswith("ct")
    is_xray = ("radiographie" in m) or ("rx" in m) or ("asp" in m)
    is_mri  = ("irm" in m)

    # Contrast heuristics
    contrast_flag = (requires_contrast is True) or ("angioscanner" in m) or ("inject" in m)

    notes: List[str] = []

    # Pregnancy + ionising radiation
    if pregnant and (is_ct or is_xray):
        notes.append("Grossesse: limiter l’irradiation si possible (privilégier US/IRM selon le contexte).")

    # Iodinated contrast (CT/angio-CT)
    if is_ct and contrast_flag:
        notes.append("PDC iodé: vérifier fonction rénale et antécédents d’allergie (prémédication si besoin).")

    # MRI specifics
    if is_mri:
        if pregnant:
            notes.append("Grossesse <3 mois: éviter l’IRM si non urgent; gadolinium à éviter sauf nécessité.")
        notes.append("IRM: vérifier compatibilité des dispositifs implantables; claustrophobie à anticiper.")
        if contrast_flag and not pregnant:
            notes.append("IRM injectée (gadolinium): vérifier antécédents d’allergie.")

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for n in notes:
        if n not in seen:
            deduped.append(n)
            seen.add(n)
    return deduped

