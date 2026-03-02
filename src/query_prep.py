# src/query_prep.py
import re
from typing import Tuple, List, Dict

_WS = re.compile(r"\s+")

def normalize_segment(raw: str | None) -> str:
    # corporate is intentionally NOT a request segment for now
    seg = (raw or "personal").strip().lower()
    return seg if seg in {"personal", "business"} else "personal"


def build_segment_filter(req_segment: str) -> Dict:
    # only allow the requested segment + generic
    # corporate is quarantined by not being included here
    return {"$or": [{"segment": req_segment}, {"segment": "generic"}]}

def formalise_query(query: str, req_segment: str) -> Tuple[str, List[str]]:
    """
    Returns (retrieval_query, notes)
    - retrieval_query: used ONLY for vector search
    - notes: list of applied transformations for debugging / eval output
    """
    notes: List[str] = []
    q_raw = query.strip()
    q = _WS.sub(" ", q_raw.lower())

    # -------------------------
    # 1) Business product prefixing (do REPLACEMENT early)
    # -------------------------
    if req_segment == "business":
        changed = False

        if "low rate" in q and "business low rate" not in q:
            q = q.replace("low rate", "business low rate")
            changed = True

        if "platinum awards" in q and "business platinum awards" not in q:
            q = q.replace("platinum awards", "business platinum awards")
            changed = True

        # Do this AFTER platinum awards, so we don't double-prefix it
        if "awards" in q and "business awards" not in q:
            q = q.replace("awards", "business awards")
            changed = True

        if "interest free days" in q and "business interest free days" not in q:
            q = q.replace("interest free days", "business interest free days")
            changed = True

        if changed:
            notes.append("add business product prefix")

    # -------------------------
    # 2) Canonical wording (map many -> one)
    # -------------------------
    if "minimum payment" in q and "minimum repayment" not in q:
        q = q.replace("minimum payment", "minimum repayment")
        notes.append("minimum payment -> minimum repayment")

    if "withdraw cash" in q:
        q = q.replace("withdraw cash", "cash advance")
        notes.append("withdraw cash -> cash advance")

    if "cash withdrawal" in q:
        q = q.replace("cash withdrawal", "cash advance")
        notes.append("cash withdrawal -> cash advance")
    
    # -------------------------
    # 3) Build limited variants (additive expansions)
    # -------------------------
    expansions = [q]

    # annual fee vs monthly fee
    if "annual fee" in q and "monthly fee" not in q:
        expansions.append(q.replace("annual fee", "monthly fee"))
        notes.append("annual fee + monthly fee expansion")

    # interest free vs interest-free
    if "interest free" in q and "interest-free" not in q:
        expansions.append(q.replace("interest free", "interest-free"))
        notes.append("interest free spelling expansion")

    # interest rate clarification (only when ambiguous)
    if "interest rate" in q and "purchase" not in q and "cash" not in q:
        expansions.append(q.replace("interest rate", "purchase interest rate"))
        notes.append("interest rate clarification")

    # -------------------------
    # 4) Deduplicate + join
    # -------------------------
    seen = set()
    uniq = []
    for e in expansions:
        e = e.strip()
        if e and e not in seen:
            seen.add(e)
            uniq.append(e)

    retrieval_query = " | ".join(uniq)
    return retrieval_query, notes

def prep_retrieval(query: str, segment: str | None) -> Tuple[str, str, Dict, List[str]]:
    """
    Convenience wrapper used by main.py and run_eval.py.
    Returns: (req_segment, retrieval_query, where_filter, notes)
    """
    req_segment = normalize_segment(segment)
    retrieval_query, notes = formalise_query(query, req_segment)
    where = build_segment_filter(req_segment)
    return req_segment, retrieval_query, where, notes