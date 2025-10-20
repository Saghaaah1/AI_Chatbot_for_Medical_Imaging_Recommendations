# src/generation/guardrails.py
def safety_notes(user_text: str, modalite: str, requires_contrast) -> list[str]:
    t = (user_text or "").lower()
    notes = []
    pregnant = ("enceinte" in t) or ("grossesse" in t)

    m = (modalite or "").lower()
    is_ct = ("scanner" in m) or ("ct" in m)
    is_mri = ("irm" in m)
    is_xray = ("radiographie" in m) or ("asp" in m)

    if pregnant and (is_ct or is_xray):
        notes.append("Grossesse: éviter l'irradiation si possible (préférer US/IRM).")
    if is_ct and requires_contrast is True:
        notes.append("CT injecté: vérifier créatinine si >60 ans ou ATCD rénaux; protocole si allergie iodée.")
    if is_mri:
        if pregnant:
            notes.append("Grossesse <3 mois: IRM à éviter si non urgent.")
        notes.append("IRM: vérifier compatibilité des dispositifs implantables; claustrophobie à anticiper.")
        if requires_contrast is True:
            notes.append("IRM injectée: vérifier antécédents d'allergie au gadolinium.")
    return notes
