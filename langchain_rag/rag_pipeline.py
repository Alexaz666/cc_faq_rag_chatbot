# langchain_rag/rag_pipeline.py
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSIST_DIR = str(PROJECT_ROOT / "chroma_db" / "cc_faq_openai")
COLLECTION_NAME = "cc_faq"

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant that helps customers with credit card inquiries.

Use ONLY the provided context to answer. Do NOT use outside knowledge.

Answering rules:
- If the context contains the answer but uses different wording, answer using the context's wording and clarify.
  Examples:
  - If asked for "annual fee" but the context states a "monthly fee", answer with the monthly fee and explicitly say it is charged monthly.
  - Treat common synonyms as equivalent when the context clearly supports them (e.g., "minimum payment" vs "minimum repayment", "withdraw cash" vs "cash advance").
  - If the context includes conditions or exceptions (e.g., personal vs business cards, thresholds that differ by card type), include the relevant condition(s) in the answer. Do not omit business-specific thresholds if the context states them.
  - Keep the answer concise but complete: include the key figure(s) and any important condition(s).
  - If the answer is not found in the context, respond exactly with:
  "I couldn't find this in the provided credit card pages."

Context:
{context}

Question:
{question}

Answer:
"""
)

embedding_fn = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    openai_api_key=OPENAI_API_KEY
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_fn,
)

def get_vectorstore():
    return vectorstore

REFUSAL_TEXT = "I couldn't find this in the provided credit card pages."

llm = ChatOpenAI(model="gpt-4-turbo", api_key=OPENAI_API_KEY)
llm_chain = custom_prompt | llm