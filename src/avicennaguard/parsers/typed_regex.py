"""
Typed regex extraction for research evaluation (Stage 1).

Uses dataset qtype hints for reproducible IEEE eval runs.
Production API uses LLMParser instead.
"""

import re
from typing import Optional, Tuple


_TAX_SYNONYMS = {
    "living things": "living_thing",
    "living thing": "living_thing",
    "living": "living_thing",
    "warm blooded": "warm_blooded",
    "cold blooded": "cold_blooded",
}


def _normalize_word(word: str) -> str:
    w = word.lower().strip()
    irregulars = {
        "mice": "mouse", "geese": "goose", "feet": "foot", "teeth": "tooth",
        "fish": "fish", "sheep": "sheep", "deer": "deer", "children": "child",
        "people": "person", "rhombuses": "rhombus", "buses": "bus", "foxes": "fox",
        "boxes": "box", "taxes": "tax", "axes": "axis", "vertices": "vertex",
        "indices": "index", "matrices": "matrix", "appendices": "appendix",
        "radii": "radius",
    }
    if w in irregulars:
        return irregulars[w]
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("ves") and len(w) > 3:
        return w[:-3] + "f"
    if w.endswith("ses") and len(w) > 3:
        return w[:-2]
    if w.endswith("xes") and len(w) > 3:
        return w[:-2]
    if w.endswith("ches") and len(w) > 4:
        return w[:-2]
    if w.endswith("shes") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _normalize_predicate(pred: str) -> str:
    p = pred.strip().lower()
    return _TAX_SYNONYMS.get(p, _normalize_word(p))


def extract_taxonomic(q: str) -> Tuple[Optional[str], Optional[str], str]:
    t = q.lower().strip().rstrip("?")
    m = re.match(r"are all (\w+)\s+(?:a\s+|an\s+)?([\w ]+)", t)
    if m:
        return m.group(1), _normalize_predicate(m.group(2).strip()), "success"
    m = re.match(r"is\s+(\w+)\s+a[n]?\s+([\w ]+)", t)
    if m:
        return m.group(1), _normalize_predicate(m.group(2).strip()), "success"
    m = re.match(r"do all (\w+)\s+belong to\s+([\w ]+)", t)
    if m:
        return m.group(1), _normalize_predicate(m.group(2).strip()), "regex_fallback"
    return None, None, "parse_failure"


def extract_categorical(q: str) -> Tuple[Optional[str], Optional[str], str]:
    t = q.lower().strip().rstrip("?")
    m = re.match(r"do all (\w+)\s+(?:have|need)\s+(?:a\s+|an\s+|the\s+)?(.+)", t)
    if m:
        return m.group(1), m.group(2).strip(), "success"
    m = re.match(r"do all (\w+)\s+live\s+in\s+(\w+)", t)
    if m:
        return m.group(1), f"live_in_{m.group(2)}", "regex_fallback"
    m = re.match(
        r"do all (\w+)\s+(lay|give|grow|breathe|produce)\s+(?:a\s+|an\s+|the\s+)?(\w.*)", t
    )
    if m:
        prop = f"{m.group(2)}_{m.group(3).strip().replace(' ', '_')}"
        return m.group(1), prop, "regex_fallback"
    m = re.match(r"do all (\w[\w ]*?)\s+(lay|give|grow|breathe|produce)\s*$", t)
    if m:
        entity = m.group(1).strip().replace(" ", "_")
        return entity, m.group(2), "regex_fallback"
    m = re.match(r"does\s+(?:a|an)?\s*(\w+)\s+have\s+(.+)", t)
    if m:
        return m.group(1), m.group(2).strip(), "regex_fallback"
    m = re.match(r"are\s+(\w[\w ]*?)\s+effective\s+against\s+(.+)", t)
    if m:
        return m.group(1).strip().replace(" ", "_"), m.group(2).strip().replace(" ", "_"), "regex_fallback"
    return None, None, "parse_failure"


def extract_hypothetical(q: str) -> Tuple[Optional[str], Optional[str], bool, str]:
    t = q.lower().strip().rstrip("?")
    if "if" not in t:
        return None, None, False, "parse_failure"
    after_if = t.split("if", 1)[1].strip()
    if "then" in after_if:
        parts = after_if.split("then", 1)
        parse_status = "success"
    elif "," in after_if:
        parts = after_if.split(",", 1)
        parse_status = "regex_fallback"
    else:
        return None, None, False, "parse_failure"

    cond = parts[0].strip()
    cons = parts[1].strip()
    is_neg = False
    for prefix in ["is there", "is the", "does it", "is it", "is", "does", "do we need"]:
        if cons.startswith(prefix + " "):
            rem = cons[len(prefix):].strip()
            if rem.startswith("no ") or rem.startswith("not ") or " no " in rem or " not " in rem:
                is_neg = True
                rem = (
                    rem.replace("no ", "", 1)
                    .replace("not ", "", 1)
                    .replace(" no ", " ", 1)
                    .replace(" not ", " ", 1)
                    .strip()
                )
            cons = rem
            break
    if not is_neg and (" no " in cons or " not " in cons):
        is_neg = True
        cons = cons.replace(" no ", " ", 1).replace(" not ", " ", 1).strip()
    return cond, cons, is_neg, parse_status
