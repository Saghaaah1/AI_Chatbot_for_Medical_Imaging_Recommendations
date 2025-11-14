"""
RAG Generation Engine (local, Ollama, JSON-first)

- Uses your retrieval() to get top-k docs
- Builds compact, numbered context from ADERIM docs (strips 'passage:' prefix)
- Asks the model to answer STRICTLY from context
- Returns a validated JSON dict + integer citations (1..k)
- Graceful error if Ollama is not running, with a single retry

Set model via env:
  OLLAMA_MODEL (default: "qwen2.5:3b-instruct")
  OLLAMA_URL   (default: "http://localhost:11434")
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, List, Iterable, Optional
import requests
import os as _os2

# Retrieval entrypoint (your single-file retriever)
from src.retrieval.retrieval import retrieve, DEFAULT_K
from langsmith import traceable


_os2.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
_os2.environ.setdefault("LANGCHAIN_PROJECT", "Medical-Imaging-RAG")


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")

# --------------------------- Prompt scaffolding ---------------------------

SYSTEM_MSG = (
    "Tu es un assistant clinique d’imagerie. Réponds en français, "
    "avec prudence et concision.\n"
    "RÈGLES STRICTES:\n"
    "1) Utilise UNIQUEMENT le contexte fourni.\n"
    "2) Si l’info manque ou est ambiguë: écris 'information insuffisante'.\n"
    "3) Retourne UNIQUEMENT un JSON valide, sans texte autour, sans commentaires."
)

USER_TEMPLATE = """Question: {question}

Contexte (extraits numérotés):
{context}

Exigence de sortie — retourne UNIQUEMENT un JSON avec les clés:
- "recommandation": texte court (modalité + délai si présent)
- "indications_positives": liste de puces (max 5, ≤12 mots chacune). Pas de nombres seuls.
- "indications_negatives": liste de puces (max 5, ≤12 mots chacune). Pas de nombres seuls.
- "precautions": liste de puces (max 5, ≤12 mots chacune). Pas de nombres seuls.
- "citations": liste d'entiers appartenant à 1..{k} (ex: [1,3])

Ne mets pas de texte hors JSON. Si l'information manque, mets "recommandation": "information insuffisante" et explique brièvement dans "precautions".
"""

# --------------------------- Helpers ---------------------------

def _strip_passage_prefix(text: str) -> str:
    """Hide the E5 'passage:' prefix before giving text to the LLM."""
    t = (text or "").lstrip()
    if t.lower().startswith("passage:"):
        t = t[len("passage:"):].lstrip()
    return t

def _format_ctx(docs, max_chars_snippet: int = 800) -> str:
    """
    Make short, numbered blocks (good for 8 GB).
    Each block includes: [#] systeme • pathologie • modalite, snippet, source.
    """
    blocks: List[str] = []
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        head = f"[{i}] {m.get('systeme','?')} • {m.get('pathologie','?')} • {m.get('modalite','?')}"
        body = _strip_passage_prefix((d.page_content or "").replace("\n", " ").strip())[:max_chars_snippet]
        src  = m.get("link") or m.get("source") or m.get("id") or "N/A"
        blocks.append(f"{head}\n{body}\n(Source: {src})")
    return "\n\n".join(blocks)

def _normalize_citations(cites, k: int) -> List[int]:
    """Make citations a clean, sorted, unique list of ints in 1..k."""
    out: List[int] = []
    if isinstance(cites, str):
        tmp = cites.replace("[", " ").replace("]", " ").replace(",", " ").split()
        for t in tmp:
            if t.isdigit():
                out.append(int(t))
    elif isinstance(cites, (list, tuple)):
        for c in cites:
            if isinstance(c, int):
                out.append(c)
            elif isinstance(c, str):
                c = c.strip().strip("[]")
                if c.isdigit():
                    out.append(int(c))
    # keep only valid indices
    return sorted({x for x in out if 1 <= x <= max(1, k)})

def _ensure_schema(d: dict, k: int) -> dict:
    """
    Guarantee required keys/types; cap lists; normalize citations.
    Also coerce non-list fields into 1-item lists if needed.
    """
    d = dict(d or {})
    d.setdefault("recommandation", "")
    for key in ("indications_positives", "indications_negatives", "precautions"):
        v = d.get(key, [])
        if not isinstance(v, list):
            v = [str(v)]
        # drop empties, keep short, stringified
        v = [str(x).strip() for x in v if str(x).strip()]
        d[key] = v[:5]
    d["citations"] = _normalize_citations(d.get("citations", []), k)
    return d

def _strip_fences(text: str) -> str:
    """Remove ```json fences if the model adds them."""
    t = (text or "").strip()
    if t.startswith("```"):
        # Remove leading ``` or ```json
        t = t.lstrip("`").lstrip()
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
        # Remove trailing ```
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()

# --------------------------- Ollama calls ---------------------------

@traceable(name="ollama_chat", run_type="llm")
def _ollama_chat(system: str, user: str, *, temperature: float = 0.2, num_ctx: int = 3072, stream: bool = False):
    """
    Call Ollama /api/chat (non-stream or stream).
    Includes a small retry if the first request fails (e.g., model not warm).
    """
    # Attach LLM call details to the run
    try:
        from langsmith.run_helpers import trace
        trace.set_attributes({
            "provider": "ollama",
            "model": MODEL,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "stream": stream,
            "base_url": OLLAMA_URL,
        })
    except Exception:
        pass

    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": temperature, "num_ctx": num_ctx},
        "stream": stream,
    }

    # First attempt
    try:
        r = requests.post(url, json=payload, timeout=300, stream=stream)
        r.raise_for_status()
        return r
    except requests.exceptions.RequestException as e:
        # Quick retry after a short pause (helps when the model was cold)
        time.sleep(1.5)
        try:
            r = requests.post(url, json=payload, timeout=300, stream=stream)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException:
            # Raise a clean error message for the caller/UI
            raise RuntimeError(
                f"Ollama chat failed. Is Ollama running at {OLLAMA_URL}? "
                f"Model='{MODEL}'. Original error: {e}"
            )

@traceable(name="generate_answer", run_type="chain", tags=["rag"])
def generate_answer(
    question: str,
    k: int = DEFAULT_K,
    *,
    max_chars_snippet: int = 800,
    temperature: float = 0.0,
    num_ctx: int = 3072,
) -> Dict:
    """
    Non-streaming generation: returns a JSON dict with normalized schema.
    Safe defaults for M1 8 GB (short context, low temperature).
    """
    # Record inputs
    try:
        from langsmith.run_helpers import trace
        trace.set_attributes({
            "k": k,
            "question": question,
            "max_chars_snippet": max_chars_snippet,
            "temperature": temperature,
            "num_ctx": num_ctx,
        })
    except Exception:
        pass

    docs = retrieve(question, k=k)
    try:
        from langsmith.run_helpers import trace
        trace.set_attributes({
            "retrieved_count": len(docs),
            "retrieved_ids": [d.metadata.get("id") for d in docs] if docs else [],
        })
    except Exception:
        pass

    if not docs:
        # No context: return a predictable, safe empty result
        return {
            "recommandation": "information insuffisante",
            "indications_positives": [],
            "indications_negatives": [],
            "precautions": ["Aucune source trouvée pour cette question dans la base locale."],
            "citations": [],
        }

    ctx = _format_ctx(docs, max_chars_snippet=max_chars_snippet)
    user_prompt = USER_TEMPLATE.format(question=question, context=ctx, k=len(docs))

    resp = _ollama_chat(SYSTEM_MSG, user_prompt, temperature=temperature, num_ctx=num_ctx, stream=False)
    data = resp.json()  # {"message": {"content": "..."} ...}
    raw = data.get("message", {}).get("content", "")
    cleaned = _strip_fences(raw)

    # Parse → normalize → return
    try:
        out = json.loads(cleaned)
    except Exception:
        out = {
            "recommandation": "information insuffisante",
            "indications_positives": [],
            "indications_negatives": [],
            "precautions": ["Sortie du modèle non-JSON. Réduis k ou la taille des extraits, ou baisse temperature."],
            "citations": [],
            "_raw": raw,
        }

    # compute → trace → return
    result = _ensure_schema(out, len(docs))
    try:
        from langsmith.run_helpers import trace
        trace.set_attributes({"output_keys": list(result.keys())})
    except Exception:
        pass
    return result




@traceable(name="generate_stream", run_type="chain", tags=["rag", "stream"])
def generate_stream(
    question: str,
    k: int = DEFAULT_K,
    *,
    max_chars_snippet: int = 800,
    temperature: float = 0.1,
    num_ctx: int = 3072,
) -> Iterable[str]:
    """
    Streaming generator (yields raw text chunks) — handy for Streamlit.
    You can buffer and parse at the end, or display the stream to the user.
    """
    try:
        from langsmith.run_helpers import trace
        trace.set_attributes({
            "k": k,
            "question": question,
            "max_chars_snippet": max_chars_snippet,
            "temperature": temperature,
            "num_ctx": num_ctx,
        })
    except Exception:
        pass

    docs = retrieve(question, k=k)
    try:
        from langsmith.run_helpers import trace
        trace.set_attributes({
            "retrieved_count": len(docs),
            "retrieved_ids": [d.metadata.get("id") for d in docs] if docs else [],
        })
    except Exception:
        pass

    if not docs:
        yield '{"recommandation":"information insuffisante","indications_positives":[],"indications_negatives":[],"precautions":["Aucune source trouvée"],"citations":[]}'
        return

    ctx = _format_ctx(docs, max_chars_snippet=max_chars_snippet)
    user_prompt = USER_TEMPLATE.format(question=question, context=ctx, k=len(docs))

    r = _ollama_chat(SYSTEM_MSG, user_prompt, temperature=temperature, num_ctx=num_ctx, stream=True)
    # Ollama streams JSON lines; each has a "message":{"content": "..."} delta
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
            delta = obj.get("message", {}).get("content", "")
            if delta:
                yield delta
        except Exception:
            # If a chunk is not JSON (e.g., keepalive), skip it
            continue
    # (Caller can join chunks & then parse JSON using _strip_fences + json.loads)
