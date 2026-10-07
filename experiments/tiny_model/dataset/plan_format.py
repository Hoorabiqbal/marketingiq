"""
The tiny model's input and output contract, shared by dataset building, training and evaluation.

Output: the EXISTING MarketingIQ analytics plan (backend/query_planner.py SCHEMA), written in its
"sparse" form: the same keys and values, with every field that equals its default left out, e.g.

    {"intent": "breakdown", "metrics": ["revenue"], "dimension": "platform"}

to_canonical() fills the defaults back in, giving exactly the JSON the Groq planner returns, and
production's validate_plan() is the only gate either way. Sparse output is ~3x fewer tokens to
decode on CPU; it is not a second schema (unknown keys are rejected, never dropped).

Input: a short fixed system prompt plus the question, and for follow-ups the previous plan in the
same wording query_planner.answer() uses for Groq.

Nothing here runs a model, calls Groq or changes the backend.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BACKEND = REPO / "backend"
CSV_PATH = REPO / "data" / "tech_advertising_campaigns_dataset.csv"

# Values a sparse plan may omit. "order" defaults to desc: validate_plan() reads "none" as desc too.
DEFAULTS = {"dimension": "none", "entities": [], "filters": [], "months": [], "years": [], "grain": "none",
            "conditions": [], "order": "desc", "limit": 0, "visualization": "none", "reason": "none",
            "dimension2": "none", "exclude": []}
ALWAYS = ("intent", "metrics")

SYSTEM_PROMPT = ("You are the MarketingIQ planner. Translate the question into a MarketingIQ analytics plan "
                 "as compact JSON. Never answer the question and never write data values.")

_qp = None


def backend():
    """query_planner with the campaign data loaded (validate_plan needs the dataset's real values)."""
    global _qp
    if _qp is None:
        sys.path.insert(0, str(BACKEND))
        import data_tools as dt
        dt.load_data(str(CSV_PATH))
        import query_planner
        _qp = query_planner
    return _qp


def field_order() -> list:
    return list(backend().SCHEMA["properties"])


def to_sparse(raw: dict) -> dict:
    """Canonical (or already sparse) plan -> sparse plan, keys in SCHEMA order."""
    out = {}
    for key in field_order():
        if key not in raw:
            continue
        if key in ALWAYS or raw[key] != DEFAULTS[key]:
            out[key] = raw[key]
    return out


def to_canonical(sparse) -> str:
    """Sparse plan (dict or JSON text) -> the full canonical JSON text validate_plan() expects.
    Malformed JSON and unknown keys are passed through for validate_plan() to reject."""
    raw = json.loads(sparse) if isinstance(sparse, str) else sparse
    if not isinstance(raw, dict):
        return json.dumps(raw)
    full = {**DEFAULTS, **raw}
    if raw.get("intent") == "not_answerable":
        full.setdefault("metrics", [])
    return json.dumps(full)


def sparse_schema() -> dict:
    """JSON schema for constrained decoding (llama.cpp / Ollama): production's SCHEMA with only
    intent and metrics required."""
    schema = json.loads(json.dumps(backend().SCHEMA))
    schema["required"] = list(ALWAYS)
    return schema


def user_prompt(question: str, previous: dict = None) -> str:
    if previous is None:
        return question
    return (f"Previous analysis in this conversation, as a plan (resolve words like this, that, same, "
            f"instead or it against it): {json.dumps(to_sparse(previous))}\nQuestion: {question}")


def messages(question: str, previous: dict = None, plan: dict = None) -> list:
    """Chat messages for one example; with `plan`, the assistant turn is the training target."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(question, previous)}]
    if plan is not None:
        msgs.append({"role": "assistant", "content": json.dumps(to_sparse(plan), separators=(",", ":"))})
    return msgs


def postprocess(plan, question: str):
    """What query_planner.answer() does to a validated plan before running it."""
    qp = backend()
    if not qp.cb.VIZ_RE.search(question):
        plan.visualization = None
    return qp.normalize_cross_year(plan, question)
