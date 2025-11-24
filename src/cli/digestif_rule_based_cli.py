from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import argparse
import json
import logging

from src.generation.generation import generate_answer

# ---------- dataclass ----------
@dataclass
class DigestifRuleState:
    chief_complaint: str
    age: Optional[int] = None
    sex: Optional[str] = None  # "m" ou "f"

    # Signes généraux / gravité
    weight_loss: Optional[bool] = None
    anemia: Optional[bool] = None
    persistent_vomiting: Optional[bool] = None
    hemodynamic_instability: Optional[bool] = None
    fever: Optional[bool] = None

    # Médicaments / terrain
    nsaids: Optional[bool] = None
    anticoagulants: Optional[bool] = None
    steroids: Optional[bool] = None
    ipp: Optional[bool] = None

    # Haut appareil digestif
    upper_symptoms: Optional[bool] = None
    epigastric_pain: Optional[bool] = None
    heartburn: Optional[bool] = None
    regurgitation: Optional[bool] = None
    dysphagia: Optional[bool] = None
    odynophagia: Optional[bool] = None
    upper_gi_bleeding: Optional[bool] = None  # hématémèse / méléna

    # Bas appareil digestif
    bowel_change: Optional[bool] = None
    diarrhea: Optional[bool] = None
    constipation: Optional[bool] = None
    rectal_bleeding: Optional[bool] = None
    abdominal_pain: Optional[bool] = None

    # Foie / biliaire
    jaundice: Optional[bool] = None
    dark_urine: Optional[bool] = None
    pale_stools: Optional[bool] = None
    ruq_pain: Optional[bool] = None  # douleur hypochondre droit

    # Grossesse
    pregnant: Optional[bool] = None
    pregnancy_weeks: Optional[int] = None


# ---------- helpers ----------
def ask_yes_no(prompt: str) -> Optional[bool]:
    """Pose une question oui/non en boucle jusqu'à réponse claire."""
    while True:
        rep = input(prompt + " (o/n, Enter = inconnu) : ").strip().lower()
        if rep == "":
            return None
        if rep.startswith("o"):
            return True
        if rep.startswith("n"):
            return False
        print("Réponse non comprise, merci de répondre par 'o' ou 'n'.")


def state_to_json(state: DigestifRuleState) -> Dict[str, Any]:
    """Return a dict with only non-None fields (useful for logging / tests)."""
    return {k: v for k, v in asdict(state).items() if v is not None}


# ---------- intake ----------
def run_digestif_intake() -> DigestifRuleState:
    """
    Chatbot d'anamnèse pour symptômes digestifs.
    Remplit un état structuré à partir de questions simples.
    """
    print("=== Chatbot digestif (intake simple) ===")

    chief = input("Motif principal (ex: 'douleurs abdominales', 'diarrhée', 'reflux') : ").strip()
    if chief == "":
        chief = "symptômes digestifs"
    state = DigestifRuleState(chief_complaint=chief)

    # Âge
    while True:
        age_raw = input("Âge du patient (en années, Enter si inconnu) : ").strip()
        if age_raw == "":
            state.age = None
            break
        try:
            age = int(age_raw)
            if age < 0 or age > 120:
                print("Âge non plausible, réessayer.")
                continue
            state.age = age
            break
        except ValueError:
            print("Merci d'entrer un nombre entier.")

    # Sexe
    while True:
        sex_raw = input("Sexe du patient (m/f, Enter si inconnu) : ").strip().lower()
        if sex_raw == "":
            state.sex = None
            break
        if sex_raw in ("m", "f"):
            state.sex = sex_raw
            break
        print("Merci de répondre 'm' ou 'f'.")

    # Signes généraux / red flags
    state.weight_loss = ask_yes_no("Amaigrissement involontaire récent ?")
    state.anemia = ask_yes_no("Anémie documentée biologiquement ?")
    state.persistent_vomiting = ask_yes_no("Vomissements persistants ou répétés ?")
    state.hemodynamic_instability = ask_yes_no(
        "Signes d'instabilité hémodynamique (hypotension, tachycardie, malaise) ?"
    )
    state.fever = ask_yes_no("Fièvre associée aux symptômes digestifs ?")

    # Médicaments / terrain
    state.nsaids = ask_yes_no("Prise récente ou actuelle d'AINS / aspirine ?")
    state.anticoagulants = ask_yes_no("Traitement par anticoagulants / anti-agrégants ?")
    state.steroids = ask_yes_no("Traitement prolongé par corticoïdes ?")
    state.ipp = ask_yes_no("Traitement actuel ou récent par IPP (inhibiteurs de la pompe à protons) ?")

    # Haut appareil digestif
    state.upper_symptoms = ask_yes_no("Les symptômes sont-ils plutôt hauts digestifs (épigastre, reflux, brûlures) ?")
    if state.upper_symptoms:
        state.epigastric_pain = ask_yes_no("Douleur ou gêne épigastrique ?")
        state.heartburn = ask_yes_no("Brûlures rétro-sternales / pyrosis ?")
        state.regurgitation = ask_yes_no("Régurgitations acides ?")
        state.dysphagia = ask_yes_no("Dysphagie (sensation de blocage des aliments) ?")
        state.odynophagia = ask_yes_no("Odynophagie (douleur à la déglutition) ?")
        state.upper_gi_bleeding = ask_yes_no("Signes de saignement digestif haut (hématémèse ou méléna) ?")

    # Bas appareil digestif
    state.bowel_change = ask_yes_no("Y a-t-il une modification récente du transit intestinal ?")
    if state.bowel_change:
        state.diarrhea = ask_yes_no("Diarrhée prédominante ?")
        state.constipation = ask_yes_no("Constipation prédominante ?")
    state.rectal_bleeding = ask_yes_no("Présence de sang rouge dans les selles (rectorragies) ?")
    state.abdominal_pain = ask_yes_no("Douleurs abdominales associées ?")

    # Foie / biliaire
    state.jaundice = ask_yes_no("Ictère clinique (jaunisse) observé ?")
    if state.jaundice:
        state.dark_urine = ask_yes_no("Urines foncées ('coca-cola') ?")
        state.pale_stools = ask_yes_no("Selles décolorées ou très pâles ?")
    state.ruq_pain = ask_yes_no("Douleur de l'hypochondre droit (zone foie/vésicule) ?")

    # Grossesse (si femme en âge de procréer)
    if state.sex == "f" and (state.age is None or 15 <= state.age <= 50):
        state.pregnant = ask_yes_no("Patiente enceinte ?")
        if state.pregnant:
            while True:
                sa_raw = input("Nombre de semaines d'aménorrhée (SA, Enter si inconnu) : ").strip()
                if sa_raw == "":
                    state.pregnancy_weeks = None
                    break
                try:
                    sa = int(sa_raw)
                    if sa < 0 or sa > 45:
                        print("Valeur de SA non plausible, réessayer.")
                        continue
                    state.pregnancy_weeks = sa
                    break
                except ValueError:
                    print("Merci d'entrer un nombre entier pour les SA.")
    else:
        state.pregnant = None
        state.pregnancy_weeks = None

    print("\n[DEBUG] État clinique structuré (digestif) :", state_to_json(state))
    return state


# ---------- build query ----------
def build_rag_query_from_state(state: DigestifRuleState) -> str:
    """
    Construit une requête textuelle claire pour le RAG à partir de l'état digestif.
    """
    parts = []

    # Motif principal
    parts.append(state.chief_complaint or "symptômes digestifs")

    # Sexe / âge
    if state.sex == "m":
        sexe_txt = "homme"
    elif state.sex == "f":
        sexe_txt = "femme"
    else:
        sexe_txt = "patient"

    if state.age is not None:
        parts.append(f"{sexe_txt} de {state.age} ans")
    else:
        parts.append(sexe_txt)

    # Red flags
    if state.weight_loss:
        parts.append("amaigrissement involontaire")
    if state.anemia:
        parts.append("anémie documentée")
    if state.persistent_vomiting:
        parts.append("vomissements persistants")
    if state.hemodynamic_instability:
        parts.append("signes d'instabilité hémodynamique")
    if state.fever:
        parts.append("fièvre associée")

    # Médicaments
    if state.nsaids:
        parts.append("prise d'AINS")
    if state.anticoagulants:
        parts.append("traitement anticoagulant/anti-agrégant")
    if state.steroids:
        parts.append("corticothérapie prolongée")
    if state.ipp:
        parts.append("traitement par IPP")

    # Haut digestif
    if state.upper_symptoms:
        parts.append("symptômes hauts digestifs")
    if state.dysphagia:
        parts.append("dysphagie")
    if state.odynophagia:
        parts.append("odynophagie")
    if state.upper_gi_bleeding:
        parts.append("hémorragie digestive haute (hématémèse/méléna)")

    # Bas digestif
    if state.bowel_change:
        parts.append("modification du transit")
    if state.diarrhea:
        parts.append("diarrhée")
    if state.constipation:
        parts.append("constipation")
    if state.rectal_bleeding:
        parts.append("rectorragies")
    if state.abdominal_pain:
        parts.append("douleurs abdominales")

    # Hépatobiliaire
    if state.jaundice:
        parts.append("ictère")
    if state.ruq_pain:
        parts.append("douleur hypochondre droit")

    # Grossesse
    if state.sex == "f" and state.pregnant:
        if state.pregnancy_weeks is not None:
            parts.append(f"grossesse de {state.pregnancy_weeks} SA")
        else:
            parts.append("grossesse en cours")

    return ", ".join(parts)


# ---------- main / CLI ----------
def main(interactive: bool = True, domain: str = "digestif"):
    """
    interactive: if False, use a small example state (useful for tests)
    domain: passed to generate_answer so your RAG pipeline can pick an index
    """
    logger = logging.getLogger(__name__)

    if interactive:
        state = run_digestif_intake()
    else:
        # lightweight example state for non-interactive runs (tests / CI)
        state = DigestifRuleState(
            chief_complaint="douleurs abdominales",
            age=50,
            sex="f",
            abdominal_pain=True,
            diarrhea=False,
            fever=False,
        )
        logger.info("[DEBUG] Using non-interactive example state: %s", state_to_json(state))

    rag_query = build_rag_query_from_state(state)
    logger.info("Requête envoyée au RAG (digestif): %s", rag_query)

    # NOTE: generate_answer should accept domain (index/namespace) and return a dict.
    result: Dict[str, Any] = generate_answer(rag_query)

    print("\n=== RECOMMANDATION RAG (digestif) ===")
    try:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except TypeError:
        # fallback if result is not serializable
        print(result)

    if isinstance(result, dict) and "recommandation" in result:
        print("\nExamen recommandé :", result.get("recommandation"))
        if "indications_positives" in result and result["indications_positives"]:
            print("Indications positives :")
            for ind in result["indications_positives"]:
                print("  -", ind)
        if "precautions" in result and result["precautions"]:
            print("Précautions :")
            for p in result["precautions"]:
                print("  -", p)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Chatbot digestif (intake simple)")
    parser.add_argument("--noninteractive", action="store_true", help="Run with an example state (no inputs)")
    parser.add_argument("--domain", default="digestif", help="RAG domain/index to query (default: digestif)")
    args = parser.parse_args()
    main(interactive=not args.noninteractive, domain=args.domain)

