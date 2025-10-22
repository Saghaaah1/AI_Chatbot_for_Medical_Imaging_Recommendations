# src/retrieval/selector.py
import json

def _as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except Exception:
        return [str(v)]

def rerank_with_profile(results_with_scores, user_text: str):
    t = (user_text or "").lower()
    is_preg = ("enceinte" in t) or ("grossesse" in t)
    is_child = ("enfant" in t) or ("pédiat" in t)

    adjusted = []
    for doc, base in results_with_scores:
        meta = doc.metadata
        pops = _as_list(meta.get("populations"))
        ionisant = meta.get("ionisant")
        patho = (meta.get("pathologie") or "").lower()

        bonus = 0.0
        if is_preg:
            if "enceinte" in pops: bonus += 0.20
            if ionisant is False:  bonus += 0.10
            if ionisant is True:   bonus -= 0.25
            if "enfant" in pops:   bonus -= 0.20
        if is_child:
            if "enfant" in pops:   bonus += 0.20

        for kw in ("appendicite","diverticulite","hépatite","fécalome","cancer"):
            if kw in t and kw in patho:
                bonus += 0.05

        adjusted.append((doc, base + bonus))

    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted
