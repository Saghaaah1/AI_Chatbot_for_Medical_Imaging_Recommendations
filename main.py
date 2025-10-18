# main.py
"""
AI Chatbot for Medical Imaging Recommendations (RAG Project)
-----------------------------------------------------------
Author: Sara El Bari
Description: Clinical decision-support chatbot that recommends imaging
based on French medical guidelines (ADERIM, SFR, HAS).
"""

from src.ingestion.create_index import create_index
from src.retrieval.query_engine import query_engine

if __name__ == "__main__":
    print("===RAG Medical Chatbot ===")
    print("1. Building or loading the vector database...")
    create_index()

    print("2. Asking a test clinical question...")
    response = query_engine("Patiente de 45 ans, dyspnée aiguë, douleur thoracique pleuritique.")
    print("\nResponse:\n", response)

