import json
from typing import Any, Dict, Optional
from pathlib import Path
from collections import defaultdict

from src.query_prep import prep_retrieval
from langchain_rag.rag_pipeline import (
    get_vectorstore,
    llm_chain,
    REFUSAL_TEXT,
)

# ----------------------------
# Config
# ----------------------------
RELEVANCE_THRESHOLD = 0.40

SMOKE_TEST = False  # set True for quick runs
SMOKE_N = 3

# Default retrieval depth for eval runs
EVAL_TOP_K = 8


# ----------------------------
# Core QA logic
# ----------------------------
def run_query_with_diagnostics(query: str, req_segment: str = "personal", k: int = 10) -> Dict[str, Any]:
    vs = get_vectorstore()

    # Centralized segment normalization + query formalisation + filter building
    req_segment, retrieval_query, where, q_notes = prep_retrieval(query, req_segment)

    docs_and_scores = vs.similarity_search_with_relevance_scores(
        retrieval_query,
        k=k,
        filter=where,
    )

    retrieved_docs = len(docs_and_scores)
    best_score = max((score for _, score in docs_and_scores), default=None)

    # Retrieval preview for human review
    retrieval_preview = []
    for doc, score in docs_and_scores:
        retrieval_preview.append(
            {
                "score": float(score),
                "title": doc.metadata.get("title"),
                "url": doc.metadata.get("url"),
                "doc_segment": doc.metadata.get("segment"),
                "snippet": doc.page_content[:300],
            }
        )

    # Defaults
    answer = REFUSAL_TEXT
    refusal_reason: Optional[str] = None
    llm_called = False

    # Pre-LLM Refusal 1 - empty retrieval
    if not docs_and_scores:
        refusal_reason = "empty_retrieval"
        return {
            "request_segment": req_segment,
            "retrieval_query": retrieval_query,
            "query_notes": q_notes,
            "answer": answer,
            "refused": True,
            "refusal_reason": refusal_reason,
            "llm_called": llm_called,
            "retrieved_docs": retrieved_docs,
            "best_score": best_score,
            "retrieval_preview": retrieval_preview,
        }

    # Pre-LLM Refusal 2 - low relevance
    if best_score is not None and best_score < RELEVANCE_THRESHOLD:
        refusal_reason = "low_relevance"
        return {
            "request_segment": req_segment,
            "retrieval_query": retrieval_query,
            "query_notes": q_notes,
            "answer": answer,
            "refused": True,
            "refusal_reason": refusal_reason,
            "llm_called": llm_called,
            "retrieved_docs": retrieved_docs,
            "best_score": best_score,
            "retrieval_preview": retrieval_preview,
        }

    # Build context & call LLM
    docs = [d for d, _ in docs_and_scores]
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    resp = llm_chain.invoke({"context": context, "question": query})
    answer = resp.content
    llm_called = True

    refused = (answer == REFUSAL_TEXT)
    if refused:
        refusal_reason = "llm_refusal"

    return {
        "request_segment": req_segment,
        "retrieval_query": retrieval_query,
        "query_notes": q_notes,
        "answer": answer,
        "refused": refused,
        "refusal_reason": refusal_reason,
        "llm_called": llm_called,
        "retrieved_docs": retrieved_docs,
        "best_score": best_score,
        "retrieval_preview": retrieval_preview,
    }


# ----------------------------
# Evaluation runner
# ----------------------------
def evaluate(eval_file: str | Path):
    with open(eval_file, "r") as f:
        questions = json.load(f)

    if SMOKE_TEST:
        questions = questions[:SMOKE_N]

    results = []
    metrics = defaultdict(
        lambda: {
            "total": 0,
            "guardrail_pass": 0,
            "segment_match_all": 0,
            "segment_check_applicable": 0,
        }
    )

    for item in questions:
        qid = item["id"]
        bucket = (item["bucket"] or "").strip().lower()
        question = item["question"]

        print(f"Running {qid} ({bucket})...")

        diag = run_query_with_diagnostics(
            question,
            req_segment=item.get("req_segment"),  # may be None; prep_retrieval defaults it
            k=EVAL_TOP_K,
        )

        req_segment = diag["request_segment"]  # canonical / normalized segment actually used
        retrieval_q = diag["retrieval_query"]
        q_notes = diag["query_notes"]
        answer = diag["answer"]
        topk_docs = diag["retrieval_preview"]
        refused = diag["refused"]
        refusal_reason = diag.get("refusal_reason")

        # --- segment integrity validation (all retrieved docs) ---
        allowed = {req_segment, "generic"}  # corporate should not appear here now
        mismatch_count = 0
        retrieved_segments = set()

        for d in topk_docs:
            doc_seg = d.get("doc_segment")
            if doc_seg is not None:
                retrieved_segments.add(doc_seg)

            match = (doc_seg in allowed)
            d["segment_match"] = match

            if not match:
                mismatch_count += 1

        segment_match_all = (mismatch_count == 0) if topk_docs else None

        if segment_match_all is not None:
            metrics[bucket]["segment_check_applicable"] += 1
            if segment_match_all:
                metrics[bucket]["segment_match_all"] += 1

        # --- guardrail behaviour validation ---
        guardrail_pass = False
        if bucket == "answerable":
            guardrail_pass = not refused
        elif bucket in {"unanswerable", "oos", "seg_conflict"}:
            guardrail_pass = refused
        else:
            # Unknown bucket: don't count as pass
            guardrail_pass = False

        metrics[bucket]["total"] += 1
        if guardrail_pass:
            metrics[bucket]["guardrail_pass"] += 1

        # --- reserved fields for manual validation ---
        answer_validation = None
        review_notes = None

        results.append(
            {
                "id": qid,
                "bucket": bucket,
                "request_segment": req_segment,
                "question": question,
                "retrieval_query": retrieval_q,
                "query_notes": q_notes,
                # guardrail behaviour metrics
                "expected": "answer" if bucket == "answerable" else "refuse",
                "answer": answer,
                "refused": refused,
                "refusal_reason": refusal_reason,
                "guardrail_pass": guardrail_pass,
                # segment integrity metrics
                "segment_match": segment_match_all,
                "segment_mismatch_count": mismatch_count,
                "retrieved_segments": sorted(retrieved_segments),
                # retrieved docs review
                "retrieved_docs": diag["retrieved_docs"],
                "best_score": diag["best_score"],
                "llm_called": diag["llm_called"],
                "topk": topk_docs,
                # manual review placeholders
                "answer_validation": answer_validation,
                "review_notes": review_notes,
            }
        )

    return results, metrics


# ----------------------------
# Summary printer
# ----------------------------
def print_summary(metrics):
    print("\n=== Evaluation Summary ===")

    for bucket, stats in metrics.items():
        total = stats["total"]

        gp = stats["guardrail_pass"]
        gp_rate = gp / total if total else 0

        seg_app = stats.get("segment_check_applicable", 0)
        seg_match = stats.get("segment_match_all", 0)
        seg_rate = seg_match / seg_app if seg_app else 0

        print(
            f"{bucket}: "
            f"guardrail_pass {gp}/{total} = {gp_rate:.2%} | "
            f"segment_match_all {seg_match}/{seg_app} = {seg_rate:.2%}"
        )


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    eval_path = Path("eval/eval_set2.json")

    results, metrics = evaluate(eval_path)

    print_summary(metrics)

    with open("eval/eval_set2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Failures for quick review
    guardrail_failures = [r for r in results if not r["guardrail_pass"]]
    with open("eval/eval_set2_guardrail_fail.json", "w") as f:
        json.dump(guardrail_failures, f, indent=2)

    segment_mismatch = [r for r in results if not r["segment_match"]]
    with open("eval/eval_set2_segment_mismatch.json", "w") as f:
        json.dump(segment_mismatch, f, indent=2)