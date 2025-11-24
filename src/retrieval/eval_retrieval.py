"""
Evaluation script for ADERIM RAG retriever.

GOAL
----
We want to compare:
    1) Baseline retriever  ->  existing `retrieve(...)` function
    2) Reranker retriever  -> `retrieve_documents_with_reranker(...)`

We only check:
    "Did we retrieve the correct ADERIM record (by its 'id') in top-k?"

If the correct 'id' is retrieved, then the prescribed exam (modalite),
contra-indications, etc. come directly from ADERIM JSON.

METRIC
------
hit_rate@k = (# of queries where expected id is in top-k) / (total queries)
"""

from dataclasses import dataclass
from typing import List, Callable

from langchain_core.documents import Document
from langsmith import traceable
from langsmith.run_helpers import trace



from src.retrieval.retrieval import (
    retrieve as retrieve_baseline,              #  existing main retriever
    retrieve_documents_with_reranker,          # new two-stage retriever
)


# -------------------------------------------------------------------
# 1) Define a small test set based on your real ADERIM JSON records
# -------------------------------------------------------------------

@dataclass
class TestCase:
    """
    One test case = a clinical query + the ADERIM 'id' we expect.

    IMPORTANT:
    ----------
    - expected_id MUST correspond to a real ADERIM record (from your JSON).
    - We NEVER hard-code exams here; we only reference the 'id'.
      The actual exam (modalite) is always read from ADERIM.
    """
    query: str
    expected_id: str


TEST_CASES: List[TestCase] = [
    TestCase(
        query="Douleur thoracique aiguë avec dyspnée chez un enfant",
        expected_id="thorax_douleur_enfant_rx_v1",
    ),
    TestCase(
        query="Bruits respiratoires anormaux chez un adulte, radio non contributive",
        expected_id="thorax_bruits_anormaux_ct_sans_injection_v1",
    ),
    TestCase(
        query="Douleur FID chez une femme enceinte, suspicion d'appendicite",
        expected_id="abdomen_appendicite_grossesse_v1",
    ),
    TestCase(
        query="Douleur FID chez un enfant, suspicion appendicite",
        expected_id="abdomen_appendicite_enfant_v2",
    ),
    TestCase(
        query="Cancer du côlon, besoin d'un bilan d'extension",
        expected_id="abdomen_cancer_colon_v1",
    ),
    TestCase(
        query="Suspicion de cancer du pancréas avec douleur abdominale",
        expected_id="abdomen_cancer_pancreas_v1",
    ),
    TestCase(
        query="Céphalée brutale type coup de tonnerre chez un adulte",
        expected_id="neuro_cephalees_adulte_coup_de_tonnerre_ct_angio_v1",
    ),
    TestCase(
        query="Céphalées aiguës fébriles avec syndrome méningé et signes de localisation chez un enfant",
        expected_id="neuro_cephalees_enfant_febriles_irm_v1",
    ),
    TestCase(
        query="Traumatisme cervical sévère chez un enfant polytraumatisé",
        expected_id="rachis_trauma_cervical_enfant_ct_v1",
    ),
    TestCase(
        query="Traumatisme cervical chez un adulte de plus de 65 ans avec signes neurologiques",
        expected_id="rachis_trauma_cervical_adulte_ct_v1",
    ),
]


# -------------------------------------------------------------------
# 2) Generic evaluation function (works for ANY retriever)
# -------------------------------------------------------------------

def evaluate_hit_rate(
    retrieve_fn: Callable[[str, int], List[Document]],
    k: int = 5,
    name: str = "method",
) -> None:
    """
    Evaluate a retrieval function on the TEST_CASES.

    Metrics:
    --------
    - hit_rate@k : % of queries where the correct id appears in the top-k.
    - accuracy@1 : % of queries where the correct id is rank 1.
    """
    total = len(TEST_CASES)
    hits_k = 0          # for hit_rate@k
    hits_at_1 = 0       # for accuracy@1 (top-1)

    for case in TEST_CASES:
        # 1) Call the retriever to get top-k candidate documents
        docs = retrieve_fn(case.query, k=k)

        print(f"\n=== Query: {case.query}")
        print(f"Expected ADERIM id: {case.expected_id}")
        found_in_top_k = False

        # 2) Inspect each retrieved document
        for i, d in enumerate(docs, start=1):
            doc_id = d.metadata.get("id", "???")
            pathologie = d.metadata.get("pathologie", "???")
            modalite = d.metadata.get("modalite", "???")

            print(f"  [{i}] id        : {doc_id}")
            print(f"       pathologie: {pathologie}")
            print(f"       modalite  : {modalite}")

            # Check if this doc is the expected one
            if doc_id == case.expected_id:
                found_in_top_k = True
                # If it's rank 1, count for accuracy@1
                if i == 1:
                    hits_at_1 += 1

        # 3) Mark HIT or MISS for top-k
        if found_in_top_k:
            hits_k += 1
            print("  -> HIT  (expected ADERIM guideline found in top-k)")
        else:
            print("  -> MISS  (expected guideline NOT in top-k)")

    if total == 0:
        print(f"\n{name} - no test cases defined.")
        return

    # 4) Compute metrics
    hit_rate_k = hits_k / total
    accuracy_1 = hits_at_1 / total

    print(f"\n{name} - hit_rate@{k}: {hits_k}/{total} = {hit_rate_k:.2f}")
    print(f"{name} - accuracy@1 : {hits_at_1}/{total} = {accuracy_1:.2f}")
    
    try:
        trace.log_metric(f"{name}_hit_rate@{k}", hit_rate_k)
        trace.log_metric(f"{name}_accuracy@1", accuracy_1)
    except Exception:
        pass

# -------------------------------------------------------------------
# Main block: compare baseline vs reranker
# -------------------------------------------------------------------

@traceable(name="eval_retrieval", run_type="chain")
def run_all_evals():
    print("### Baseline (no reranker) ###")
    evaluate_hit_rate(retrieve_baseline, k=5, name="baseline")

    print("\n\n### With cross-encoder reranker ###")
    evaluate_hit_rate(retrieve_documents_with_reranker, k=5, name="reranker")

if __name__ == "__main__":
    run_all_evals()
