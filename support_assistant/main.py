from fastapi import FastAPI
from pydantic import BaseModel, Field

from support_assistant.graph_app import (
    MOCK_LLM,
    SupportResponse,
    run_query,
)


# =========================================================
# MODULE 3 - TASK 5
# FASTAPI WRAPPER
# =========================================================


# =========================================================
# REQUEST MODEL
# =========================================================

class AskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Customer question"
    )


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "Offline-first Zepto policy support assistant "
        "using LangGraph, ChromaDB, and local embeddings."
    ),
    version="1.0.0",
)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def root():
    return {
        "service": "Zepto Support Assistant",
        "status": "running",
        "mock_llm": MOCK_LLM,
    }


# =========================================================
# POST /ask
# =========================================================

@app.post(
    "/ask",
    response_model=SupportResponse,
)
def ask(
    request: AskRequest
) -> SupportResponse:

    response = run_query(
        request.query
    )

    return response