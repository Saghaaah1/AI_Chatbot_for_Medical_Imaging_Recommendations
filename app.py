# app.py
import re
import gradio as gr
from typing import Dict, Any, List, Tuple

# import your RAG recommender
from src.generation.pipeline import run_pipeline

# ---------------------------
# Simple chat state & helpers
# ---------------------------
def init_state() -> Dict[str, Any]:
    # slots: what we want to disambiguate with quick replies
    return {
        "slots": {
            "pregnant": None,      # True / False / None
            "age_group": None,     # "enfant" / "adulte" / None
        }
    }

def _normalize(s: str) -> str:
    return (s or "").strip().lower()

def _update_slots_from_text(text: str, slots: Dict[str, Any]) -> None:
    t = _normalize(text)
    # pregnancy
    if "enceinte" in t or "grossesse" in t:
        slots["pregnant"] = True
    if "pas enceinte" in t or "non enceinte" in t:
        slots["pregnant"] = False
    # age group
    if "enfant" in t or "pédiat" in t:
        slots["age_group"] = "enfant"
    if "adulte" in t or "adult" in t:
        slots["age_group"] = "adulte"

def _need_more_info(user_text: str, slots: Dict[str, Any]) -> List[str]:
    """Return a list of quick-reply choices we still want to ask."""
    asks: List[str] = []
    # If user mentioned thoracic stuff or pregnancy-ish, these help a lot
    t = _normalize(user_text)
    thoraxish = any(w in t for w in ["douleur thoracique", "dyspnée", "hémoptysie", "toux", "ep", "embolie"])
    neuro_peds = any(w in t for w in ["macrocrân", "fontanelle", "périmètre crânien", "céphalée", "céphalées"])

    if slots.get("pregnant") is None and ("enceinte" in t or thoraxish):
        asks.append("Patiente enceinte")
        asks.append("Patiente non enceinte")
    if slots.get("age_group") is None and (thoraxish or neuro_peds):
        asks.append("Enfant")
        asks.append("Adulte")

    # remove duplicates while preserving order
    seen = set()
    dedup = []
    for a in asks:
        if a not in seen:
            dedup.append(a); seen.add(a)
    return dedup

def _slots_to_patient(slots: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "age_group": slots.get("age_group"),
        "pregnant": slots.get("pregnant"),
    }

def _format_rec(rec) -> str:
    lines = [
        "— Recommandation —",
        f"• Modalité recommandée : {rec.modalite_recommandee}",
    ]
    if rec.symptomes_cles:
        lines.append(f"• Symptômes clés : {', '.join(rec.symptomes_cles)}")
    if rec.hypothese_clinique:
        lines.append(f"• Hypothèse clinique : {rec.hypothese_clinique}")
    if rec.urgence:
        lines.append(f"• Urgence : {rec.urgence}")
    if getattr(rec, 'delai_recommande', None):
        lines.append(f"• Délai recommandé : {rec.delai_recommande}")
    if getattr(rec, 'alternative', None):
        lines.append(f"• Alternative : {rec.alternative}")
    if rec.justification:
        lines.append(f"• Justification : {rec.justification}")
    if rec.reference:
        lines.append(f"• Référence : {rec.reference}")
    return "\n".join(lines)

# ---------------------------
# Chat runtime
# ---------------------------
def chat_fn(message: str, chat_history: List[Tuple[str, str]], state_obj: Dict[str, Any]):
    user_msg = (message or "").strip()
    if not user_msg:
        return chat_history, state_obj, []

    # Update slots based on the user message / quick reply
    slots = state_obj.get("slots", {})
    _update_slots_from_text(user_msg, slots)
    state_obj["slots"] = slots

    # Append user to chat
    chat_history = chat_history + [(user_msg, None)]

    # Decide if we need to ask a quick clarifying question
    pending = _need_more_info(user_msg, slots)

    if pending:
        # Ask a short follow-up and provide quick replies
        question_lines = []
        if "Patiente enceinte" in pending or "Patiente non enceinte" in pending:
            question_lines.append("La patiente est-elle enceinte ?")
        if "Enfant" in pending or "Adulte" in pending:
            question_lines.append("S'agit-il d'un enfant ou d'un adulte ?")
        bot_text = "\n".join(question_lines) if question_lines else "Précisez SVP."
        chat_history[-1] = (chat_history[-1][0], bot_text)
        return chat_history, state_obj, pending

    # If we have enough info (or didn’t need any), call the RAG pipeline
    try:
        patient = _slots_to_patient(slots)
        rec = run_pipeline(user_msg, patient=patient, device="cpu")
        bot_text = _format_rec(rec)
    except Exception as e:
        bot_text = f"⚠️ Erreur: {e}"

    chat_history[-1] = (chat_history[-1][0], bot_text)
    # After an answer, offer a single quick-reply to start a new case
    return chat_history, state_obj, ["Nouvelle requête"]

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks(title="Imaging Recommender (chat)") as demo:
    gr.Markdown("## Clinical imaging recommender (chat)")

    chat = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(placeholder="Ex: femme enceinte, douleur thoracique, dyspnée…")

    # Start with no choices; we’ll fill them dynamically
    options = gr.Radio(label="Réponses rapides", choices=[], interactive=True, visible=True)

    state = gr.State(init_state())

    def on_submit(user_msg, chat_history, state_obj):
        new_chat, new_state, choices = chat_fn(user_msg, chat_history, state_obj)
        return new_chat, new_state, gr.update(choices=choices or [], value=None)

    def on_option(select, chat_history, state_obj):
        if not select:
            return chat_history, state_obj, gr.update()
        # “Nouvelle requête” clears the textbox and offers no options
        if select.lower().startswith("nouvelle"):
            chat_history.append(("—", "D'accord. Décrivez le nouveau cas clinique."))  # small prompt
            return chat_history, state_obj, gr.update(choices=[], value=None)

        new_chat, new_state, choices = chat_fn(select, chat_history, state_obj)
        return new_chat, new_state, gr.update(choices=choices or [], value=None)

    txt.submit(on_submit, [txt, chat, state], [chat, state, options], queue=False)
    txt.submit(lambda: "", None, txt, queue=False)

    try:
        options.select(on_option, [options, chat, state], [chat, state, options], queue=False)
    except Exception:
        options.change(on_option, [options, chat, state], [chat, state, options], queue=False)

if __name__ == "__main__":
    demo.launch()

