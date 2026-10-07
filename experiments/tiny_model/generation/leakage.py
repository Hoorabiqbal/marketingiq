"""
Leakage checks between generated questions and the frozen evaluation set (and between splits).

A question is compared three ways:
- normalized text: lower-case words only;
- word-set Jaccard similarity (near-duplicates);
- its *skeleton*: metric aliases, dimension aliases, dataset values, months, years and numbers
  replaced by placeholders (<m>, <d>, <v>, <mon>, <y>, <n>). Two questions with the same skeleton
  are the same template with slots changed ("top 5 campaigns by profit" ~ "top 3 campaigns by
  revenue"), so a training question whose skeleton is (nearly) an evaluation skeleton is dropped.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset"))
import plan_format as pf  # noqa: E402

NEAR_DUP = 0.8          # raw word-set Jaccard
SKELETON_NEAR = 0.75    # skeleton token-set Jaccard

MONTH_WORDS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
               "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
               "sept", "oct", "nov", "dec"]
_value_re = None


def normalize(q: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", q.lower()))


def words(q: str) -> frozenset:
    return frozenset(normalize(q).split())


def _values_re():
    global _value_re
    if _value_re is None:
        pf.backend()
        import schema_catalog as catalog
        vals = sorted({v for d in catalog.dimensions().values() for v in d.values if v not in ("All", "True", "False")
                       and not v.isdigit()}, key=len, reverse=True)
        _value_re = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(v) for v in vals) + r")(?![\w-])", re.IGNORECASE)
    return _value_re


def skeleton(q: str) -> tuple:
    pf.backend()
    import schema_catalog as catalog
    text = q.lower()
    for _, pattern in catalog.METRIC_ALIAS_RES:
        text = pattern.sub(" zm ", text)
    for _, pattern in catalog.DIMENSION_ALIAS_RES:
        text = pattern.sub(" zd ", text)
    text = _values_re().sub(" zv ", text)
    text = re.sub(r"\b(?:" + "|".join(MONTH_WORDS) + r")\b", " zmon ", text)
    text = re.sub(r"\bq[1-4]\b", " zmon ", text)
    text = re.sub(r"\b20\d\d\b", " zy ", text)
    text = re.sub(r"\$?\d[\d,.]*k?%?", " zn ", text)
    text = re.sub(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b", " zn ", text)
    return tuple(re.findall(r"[a-z0-9]+", text))


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class Index:
    """Questions a candidate must not resemble (the frozen evaluation set, or another split)."""

    def __init__(self, questions):
        self.norm = {}
        self.items = []
        for q in questions:
            self.norm.setdefault(normalize(q), q)
            self.items.append((q, words(q), skeleton(q), frozenset(skeleton(q))))
        self.skeletons = {s: q for q, _, s, _ in self.items}

    def conflict(self, question: str):
        """None, or (kind, the question it collides with)."""
        n = normalize(question)
        if n in self.norm:
            return "normalized_duplicate", self.norm[n]
        w, s = words(question), skeleton(question)
        if s in self.skeletons:
            return "same_skeleton", self.skeletons[s]
        sset = frozenset(s)
        for q, qw, _, qs in self.items:
            if jaccard(w, qw) >= NEAR_DUP:
                return "near_duplicate", q
            if len(sset) >= 2 and jaccard(sset, qs) >= SKELETON_NEAR:
                return "near_skeleton", q
        return None
