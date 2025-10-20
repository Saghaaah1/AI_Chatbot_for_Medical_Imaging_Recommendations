# main.py
from src.generation.pipeline import run_pipeline

def cli():
    print("Clinical imaging recommender (MVP). Type 'quit' to exit.\n")
    while True:
        q = input("Cas clinique: ").strip()
        if not q or q.lower()=="quit":
            break
        try:
            rec = run_pipeline(q, patient=None)
            print("\n— Recommandation —")
            print(f"• Modalité recommandée : {rec.modalite_recommandee}")
            if rec.symptomes_cles:
                print(f"• Symptômes clés : {', '.join(rec.symptomes_cles)}")
            if rec.hypothese_clinique:
                print(f"• Hypothèse clinique : {rec.hypothese_clinique}")
            print(f"• Urgence : {rec.urgence}")
            if rec.alternative:
                print(f"• Alternative : {rec.alternative}")
            if rec.delai_recommande:
                print(f"• Délai recommandé : {rec.delai_recommande}")
            print(f"• Justification : {rec.justification}")
            print(f"• Référence : {rec.reference}")
            print("—" * 60 + "\n")
        except Exception as e:
            print("⚠️  Impossible de générer une recommandation:", e)

if __name__ == "__main__":
    cli()

