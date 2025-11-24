# src/cli/cephalees_rule_based_cli.py

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from src.generation.generation import generate_answer


@dataclass
class CephaleesRuleState:
    chief_complaint: str
    age: Optional[int] = None
    sex: Optional[str] = None      # "m" ou "f"
    onset_brutal: Optional[bool] = None
    fever: Optional[bool] = None
    neuro_deficit: Optional[bool] = None
    vertigo: Optional[bool] = None
    oncologic_context: Optional[bool] = None
    recent_surgery: Optional[bool] = None
    pacemaker: Optional[bool] = None
    claustrophobia: Optional[bool] = None
    pregnant: Optional[bool] = None
    pregnancy_weeks: Optional[int] = None


def ask_yes_no(prompt: str) -> Optional[bool]:
    """
    Pose une question oui/non en boucle jusqu'à réponse claire.
    Retourne True/False, ou None si utilisateur laisse vide.
    """
    while True:
        rep = input(prompt + " (o/n, Enter = inconnu) : ").strip().lower()
        if rep == "":
            return None
        if rep.startswith("o"):
            return True
        if rep.startswith("n"):
            return False
        print("Réponse non comprise, merci de répondre par 'o' ou 'n'.")


def run_cephalees_intake() -> CephaleesRuleState:
    """
    Chatbot d'anamnèse céphalées 100% hard-codé.
    Remplit un état structuré à partir de questions simples.
    """
    print("=== Chatbot céphalées (intake simple) ===")

    chief = input("Motif principal (ex: 'céphalées depuis 2 jours') : ").strip()
    state = CephaleesRuleState(chief_complaint=chief)

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

    # Questions cliniques clés
    state.onset_brutal = ask_yes_no("Le début de la céphalée a-t-il été brutal (coup de tonnerre) ?")
    state.fever = ask_yes_no("Fièvre associée ?")
    state.neuro_deficit = ask_yes_no("Déficit neurologique (faiblesse, trouble de la parole, etc.) ?")
    state.vertigo = ask_yes_no("Vertiges associés ?")
    state.oncologic_context = ask_yes_no("Contexte oncologique (cancer connu) ?")
    state.recent_surgery = ask_yes_no("Chirurgie récente avec matériel (< 6 semaines) ?")
    state.pacemaker = ask_yes_no("Porteur de pacemaker / stimulateur cardiaque ?")
    state.claustrophobia = ask_yes_no("Claustrophobie importante ?")

    # Grossesse : seulement si femme en âge probable de grossesse
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

    print("\n[DEBUG] État clinique structuré :", asdict(state))
    return state


def build_rag_query_from_state(state: CephaleesRuleState) -> str:
    """
    Construit une requête textuelle claire pour le RAG à partir de l'état structuré.
    Exemple : "Homme 50 ans, céphalée non brutale, sans fièvre, sans déficit neurologique..."
    """
    parts = []

    # Base : plainte principale
    parts.append(state.chief_complaint or "céphalée")

    # Âge + sexe
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

    # Brutalité
    if state.onset_brutal is True:
        parts.append("céphalée brutale")
    elif state.onset_brutal is False:
        parts.append("céphalée non brutale")

    # Fièvre
    if state.fever is True:
        parts.append("avec fièvre")
    elif state.fever is False:
        parts.append("sans fièvre")

    # Déficit neuro
    if state.neuro_deficit is True:
        parts.append("avec déficit neurologique")
    elif state.neuro_deficit is False:
        parts.append("sans déficit neurologique")

    # Vertiges
    if state.vertigo is True:
        parts.append("avec vertiges")
    elif state.vertigo is False:
        parts.append("sans vertiges")

    # Contexte oncologique
    if state.oncologic_context is True:
        parts.append("contexte de cancer connu")
    elif state.oncologic_context is False:
        parts.append("sans contexte oncologique connu")

    # Grossesse
    if state.sex == "f" and state.pregnant is True:
        if state.pregnancy_weeks is not None:
            parts.append(f"grossesse de {state.pregnancy_weeks} SA")
        else:
            parts.append("grossesse en cours")

    # On assemble tout
    query = ", ".join(parts)
    return query


def main():
    # 1) Anamnèse simple
    state = run_cephalees_intake()

    # 2) Construire la query pour le RAG
    rag_query = build_rag_query_from_state(state)
    print("\n[DEBUG] Requête envoyée au RAG :", rag_query)

    # 3) Appel au RAG (ta fonction existante)
    result: Dict[str, Any] = generate_answer(rag_query)

    print("\n=== RECOMMANDATION RAG ===")
    print("Examen recommandé :", result.get("recommandation"))
    if "indications_positives" in result:
        print("Indications positives :")
        for ind in result["indications_positives"]:
            print("  -", ind)
    if "precautions" in result:
        print("Précautions :")
        for p in result["precautions"]:
            print("  -", p)


if __name__ == "__main__":
    main()
