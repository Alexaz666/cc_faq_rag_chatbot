import time
import logging
from typing import List, Optional, Literal
from fastapi import FastAPI
from pydantic import BaseModel
from src.query_prep import prep_retrieval


app = FastAPI()

# -------------------------
# Config
# -------------------------
RELEVANCE_THRESHOLD = 0.25

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Request / Response Models
# -------------------------
class QueryRequest(BaseModel):
    query: str
    segment: Optional[Literal["personal", "business"]] = "personal"

class Source(BaseModel):
    title: Optional[str]
    url: Optional[str]

class Snippet(Source):
    snippet: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]
    snippets: List[Snippet]
    refused: bool = False
    refusal_reason: Optional[str] = None

# -------------------------
# Utils
# -------------------------
#def build_segment_filter(segment: str) -> dict:
    # segment is "personal" or "business"
#    return {
#        "$or": [
#            {"segment": segment},
#            {"segment": "generic"},
#        ]
#    }

def make_response(answer: str, sources=None, snippets=None, refusal_reason: str | None = None):
    return {
        "answer": answer,
        "sources": sources or [],
        "snippets": snippets or [],
        "refused": refusal_reason is not None,
        **({"refusal_reason": refusal_reason} if refusal_reason else {}),
    }
# -------------------------
# Global state (initialised on startup)
# -------------------------
vectorstore = None
llm_chain = None
REFUSAL_TEXT = None


# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
def startup():
    global vectorstore, llm_chain, REFUSAL_TEXT

    from langchain_rag.rag_pipeline import (
        get_vectorstore,
        llm_chain as chain,
        REFUSAL_TEXT as refusal_text,
    )

    vectorstore = get_vectorstore()
    llm_chain = chain
    REFUSAL_TEXT = refusal_text

    logger.info("RAG pipeline initialised")

# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Main QA endpoint
# -------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QueryRequest):
    query = request.query
    req_segment, retrieval_query, where, q_notes = prep_retrieval(query, request.segment)

    t0 = time.time()
    logger.info("Received query")

    # 1) Retrieve with relevance scores
    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(
        retrieval_query,
        k=5,
        filter=where
    )

    logger.info(
        "Retrieved %d docs in %.2fs",
        len(docs_and_scores),
        time.time() - t0,
    )

    if not docs_and_scores:
        return make_response(REFUSAL_TEXT, refusal_reason="empty_retrieval")

    # 2) Relevance gate
    best_score = max(score for _, score in docs_and_scores)
    if best_score < RELEVANCE_THRESHOLD:
        return make_response(REFUSAL_TEXT, refusal_reason="low_relevance")

    # 3) Build context
    docs = [d for d, _ in docs_and_scores]
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # 4) Generate answer
    resp = llm_chain.invoke({"context": context, "question": query})
    answer = resp.content

    logger.info("LLM completed in %.2fs", time.time() - t0)

    # 5) Post-answer guardrail
    if answer == REFUSAL_TEXT:
        return make_response(REFUSAL_TEXT, refusal_reason="llm_refusal")

    # 6) Attach sources and snippets
    sources = []
    snippets = []
    seen_urls = set()

    for d in docs:
        url = d.metadata.get("url")
        title = d.metadata.get("title")

        if url and url not in seen_urls:
            sources.append({"title": title, "url": url})
            seen_urls.add(url)

        snippets.append(
            {
                "title": title,
                "url": url,
                "snippet": d.page_content[:300],
            }
        )

    return make_response(answer, sources[:5], snippets[:5])