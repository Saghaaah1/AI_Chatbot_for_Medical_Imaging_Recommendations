# app_chat.py
from __future__ import annotations
import traceback
import gradio as gr

from src.generation.pipeline import run_pipeline, load_vectorstore
from src.generation.schema import Recommendation

# --- warm up: load vectorstore once to avoid cold-start on first query ---
# (run_pipeline will also load if needed, but preloading helps UX)
try:
    _ = load_vectorstore()
except Exception:
    # vectorstore might not exist yet; user will be told to index
    pass

APP_TITLE = "Clinical Imaging Recommender (ADERIM-based)"
DISCLAIMER = (
    "⚠️ Outil d'aide à la décision **pédagogique**. "
    "Ne remplace pas l'avis d'un radiologue/clinicien. Respectez vos protocoles locaux."
)

def format_rec(rec: Recommendation) -> str:
    lines = []
    lines.append("— **Recommandation** —")
    lines.append(f"• **Modalité recommandée** : {rec.modalite_recommandee}")
    if rec.symptomes_cles:
        lines.append(f"• **Symptômes clés** : {', '.join(rec.symptomes_cles)}")
    if rec.hypothese_clinique:
        lines.append(f"• **Hypothèse clinique** : {rec.hypothese_clinique}")
    if rec.urgence:
        lines.append(f"• **Urgence** : {rec.urgence}")
    if rec.delai_recommande:
        lines.append(f"• **Délai recommandé** : {rec.delai_recommande}")
    if rec.alternative:
        lines.append(f"• **Alternative** : {rec.alternative}")
    if rec.justification:
        lines.append(f"• **Justification** : {rec.justification}")
    if rec.reference:
        lines.append(f"• **Référence** : {rec.reference}")
    return "\n".join(lines)

def make_patient_meta(age: str, sex: str, pregnant: bool) -> dict:
    meta = {}
    if age and age.strip():
        meta["age"] = age.strip()
    if sex in ("femme", "homme"):
        meta["sexe"] = sex
    if pregnant and sex == "femme":
        meta["grossesse"] = "oui"
    return meta

def chat_fn(history, user_text, age, sex, pregnant):
    """
    history: list[(user, assistant)] from Gradio
    user_text: new clinical case text
    """
    if not user_text or not user_text.strip():
        return history, gr.update(value="")

    # Build small patient meta
    patient = make_patient_meta(age, sex, pregnant)

    try:
        rec = run_pipeline(user_text=user_text, patient=patient, device="cpu")
        answer = format_rec(rec)
    except Exception as e:
        # Friendly error with hint
        hint = ""
        if "No retrieval results" in str(e):
            hint = " (Indice: avez-vous lancé l’indexation ? `uv run python src/ingestion/create_index.py`)"
        elif "Empty user_text" in str(e):
            hint = " (Veuillez saisir un cas clinique.)"
        answer = f"⚠️ Erreur: {e}{hint}\n\n```\n{traceback.format_exc()}\n```"

    history = history + [[user_text, answer]]
    return history, gr.update(value="")

with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"## {APP_TITLE}\n{DISCLAIMER}")

    with gr.Row():
        with gr.Column(scale=2):
            chat = gr.Chatbot(
                label="Dialogue",
                height=420,
                render_markdown=True,
                avatar_images=(None, None),
            )
            user_box = gr.Textbox(
                label="Cas clinique (ex: 'femme enceinte, douleur FID, suspicion appendicite')",
                placeholder="Saisissez le cas puis Entrée…",
            )
            with gr.Row():
                age = gr.Textbox(label="Âge (optionnel)", placeholder="ex: 67, ou 'nourrisson 3 mois'")
                sex = gr.Radio(choices=["", "femme", "homme"], value="", label="Sexe")
                pregnant = gr.Checkbox(label="Grossesse", value=False)

            submit = gr.Button("Analyser")
            clear = gr.Button("Effacer l’historique")

        with gr.Column(scale=1):
            gr.Markdown(
                "### Exemples\n"
                "- *'femme enceinte, douleur FID, suspicion appendicite'*\n"
                "- *'dyspnée aiguë, douleur thoracique, EP probable (Wells élevé)'*\n"
                "- *'enfant 3 mois, macrocrânie, vomissements'*\n"
                "- *'céphalée coup de tonnerre, raideur de nuque'*\n"
            )
            gr.Markdown(
                "### Astuces\n"
                "- Plus vous donnez de **symptômes clés**, mieux c’est.\n"
                "- Mentionnez **grossesse** / **enfant** si pertinent.\n"
                "- Si l’outil se trompe de domaine, reformulez en précisant l’organe."
            )

    def _on_submit(user_text, history, age, sex, pregnant):
        return chat_fn(history, user_text, age, sex, pregnant)

    user_box.submit(_on_submit, [user_box, chat, age, sex, pregnant], [chat, user_box])
    submit.click(_on_submit, [user_box, chat, age, sex, pregnant], [chat, user_box])
    clear.click(lambda: ([], ""), None, [chat, user_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
