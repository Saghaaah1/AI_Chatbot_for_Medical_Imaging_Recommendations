from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def test_retrieval():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings,
        collection_name="aderim",
    )
    print("Vectorstore loaded. (type 'quit' to exit)\n")

    while True:
        user_q = input("Question médicale: ").strip()
        if user_q.lower() == "quit":
            break

        q = f"query: {user_q}"  # E5 needs the 'query:' prefix
        results = db.similarity_search_with_relevance_scores(q, k=5)

        print("\n🔎 Top résultats:\n")
        seen = set()
        rank = 0
        for doc, score in results:
            doc_id = doc.metadata.get("id", "N/A")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            print(f"{rank}. {doc_id}  |  score={score:.3f}")
            print(f"   Pathologie : {doc.metadata.get('pathologie')}")
            print(f"   Modalité   : {doc.metadata.get('modalite')}")
            print(f"   Urgence    : {doc.metadata.get('urgence')}")
            print(f"   Populations: {doc.metadata.get('populations')}")
            print(f"   Extrait    : {doc.page_content[:160]}...\n")
        print("—" * 80)

if __name__ == "__main__":
    test_retrieval()

