import os
import pytest
from src.generation.pipeline import run_pipeline

@pytest.mark.skipif(not os.path.isdir("vectorstore"), reason="index missing; run create_index first")
@pytest.mark.parametrize("query, must_contain", [
    ("céphalée coup de tonnerre, raideur nuque", "scanner"),                 # CT + angio
    ("grossesse, douleur FID, suspicion appendicite", "irm"),               # IRM AP
    ("nourrisson 3 mois, macrocrânie", "échographie"),                      # US transfontanellaire
])
def test_core_recommendations(query, must_contain):
    rec = run_pipeline(query)
    out = (rec.modalite_recommandee or "").lower()
    assert must_contain in out
