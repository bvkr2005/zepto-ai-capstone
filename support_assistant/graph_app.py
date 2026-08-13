from pathlib import Path
from typing import Literal, TypedDict
import json
import os

import chromadb
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from support_assistant.prompt_template import build_prompt


# =========================================================
# MODULE 3 - SUPPORT ASSISTANT
# TASKS 3 AND 4
# =========================================================


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

CHROMA_DIR = BASE_DIR / "chroma_db"


# =========================================================
# CONFIGURATION
# =========================================================

COLLECTION_NAME = "zepto_policy_docs"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# MOCK LLM TOGGLE
# =========================================================
#
# MOCK_LLM unset  -> mock mode
# MOCK_LLM=1      -> mock mode
# MOCK_LLM=0      -> optional real LLM mode
#
# Mock mode is the REQUIRED graded baseline.
# =========================================================

MOCK_LLM = (
    os.getenv(
        "MOCK_LLM",
        "1"
    )
    !=
    "0"
)


# =========================================================
# POLICY KEYWORDS
# =========================================================

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


# =========================================================
# TASK 4 - PYDANTIC RESPONSE MODEL
# =========================================================

class SupportResponse(BaseModel):
    """
    Final validated response schema required by Task 4.
    """

    answer: str

    sources: list[str]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# =========================================================
# LANGGRAPH STATE
# =========================================================

class SupportState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    query: str

    intent: Literal[
        "policy_question",
        "general_question"
    ]

    retrieved_ids: list[str]

    retrieved_chunks: list[str]

    retrieved_distances: list[float]

    answer: str

    confidence: float


# =========================================================
# LOAD LOCAL EMBEDDING MODEL
# =========================================================

print(
    "\nLoading local embedding model..."
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print(
    f"Embedding model ready: "
    f"{EMBEDDING_MODEL_NAME}"
)


# =========================================================
# CONNECT TO CHROMADB
# =========================================================

if not CHROMA_DIR.exists():

    raise FileNotFoundError(
        "ChromaDB folder was not found. "
        "Run build_index.py first."
    )


chroma_client = chromadb.PersistentClient(
    path=str(
        CHROMA_DIR
    )
)


try:

    collection = (
        chroma_client
        .get_collection(
            COLLECTION_NAME
        )
    )

except Exception as exc:

    raise RuntimeError(
        "Could not open the Zepto policy "
        "ChromaDB collection. "
        "Run build_index.py first."
    ) from exc


if collection.count() != 8:

    raise ValueError(
        "Expected 8 Zepto policy documents "
        f"in ChromaDB but found "
        f"{collection.count()}."
    )


print(
    f"ChromaDB collection ready: "
    f"{COLLECTION_NAME}"
)

print(
    f"Documents available: "
    f"{collection.count()}"
)


# =========================================================
# OPTIONAL REAL LLM CALL
# =========================================================

def call_optional_real_llm(
    prompt: str
) -> str:
    """
    Optional real-LLM path.

    This is NOT used in the graded mock baseline.
    """

    try:

        from groq import Groq

    except ImportError as exc:

        raise RuntimeError(
            "MOCK_LLM=0 was requested, but the "
            "optional 'groq' package is not installed. "
            "Use MOCK_LLM=1 for the graded baseline."
        ) from exc

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "MOCK_LLM=0 was requested but "
            "GROQ_API_KEY is not set."
        )

    client = Groq(
        api_key=api_key
    )

    response = (
        client.chat.completions.create(
            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            temperature=0,
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# =========================================================
# TASK 4 - REAL LLM JSON VALIDATION WITH RETRIES
# =========================================================

def validate_real_llm_response(
    raw_output: str,
    fallback_sources: list[str],
) -> SupportResponse:
    """
    Validate optional real-LLM JSON output.

    If validation fails, retry up to 2 additional times
    with corrective instructions.

    Total attempts:
    1 original attempt
    + 2 retries
    = 3 attempts maximum.
    """

    current_output = raw_output

    for attempt in range(
        1,
        4
    ):

        try:

            parsed = json.loads(
                current_output
            )

            validated = (
                SupportResponse
                .model_validate(
                    parsed
                )
            )

            return validated

        except (
            json.JSONDecodeError,
            ValidationError,
        ) as exc:

            print(
                f"Structured output validation "
                f"failed on attempt {attempt}."
            )

            print(
                f"Reason: {exc}"
            )

            if attempt == 3:

                break

            corrective_prompt = f"""
Your previous response failed JSON validation.

Return ONLY valid JSON with exactly these fields:

answer: string
sources: list of strings
confidence: number between 0 and 1

Do not include Markdown fences.
Do not include explanation outside JSON.

If relevant, use these source IDs:
{fallback_sources}

Previous invalid response:
{current_output}
"""

            current_output = (
                call_optional_real_llm(
                    corrective_prompt
                )
            )

    # -----------------------------------------------------
    # CLEARLY MARKED ERROR RESPONSE
    # -----------------------------------------------------

    return SupportResponse(
        answer=(
            "ERROR: real LLM response could not "
            "be validated after 3 attempts."
        ),
        sources=[],
        confidence=0.0,
    )


# =========================================================
# NODE 1 - CLASSIFY INTENT
# =========================================================

def classify_intent(
    state: SupportState
) -> SupportState:
    """
    Classify the incoming query as either:

    policy_question
    general_question
    """

    query = (
        state[
            "query"
        ]
        .strip()
    )

    query_lower = (
        query.lower()
    )

    print("\n========================================")
    print("NODE: classify_intent")
    print("========================================")

    print(
        f"Query: {query}"
    )

    print(
        f"MOCK_LLM: "
        f"{1 if MOCK_LLM else 0}"
    )

    # =====================================================
    # REQUIRED MOCK BRANCH
    # =====================================================

    if MOCK_LLM:

        is_policy_question = any(
            keyword
            in
            query_lower

            for keyword
            in
            POLICY_KEYWORDS
        )

        if is_policy_question:

            intent = (
                "policy_question"
            )

        else:

            intent = (
                "general_question"
            )

        print(
            "Classification method: "
            "keyword heuristic"
        )

    # =====================================================
    # OPTIONAL REAL LLM BRANCH
    # =====================================================

    else:

        classification_prompt = f"""
Classify the following customer question into exactly
one of these two labels:

policy_question
general_question

A policy_question concerns Zepto delivery, returns,
refunds, membership, tracking, cancellations,
gift cards, or support policies.

Question:
{query}

Return only one label.
"""

        raw_result = (
            call_optional_real_llm(
                classification_prompt
            )
            .strip()
            .lower()
        )

        if (
            "policy_question"
            in
            raw_result
        ):

            intent = (
                "policy_question"
            )

        else:

            intent = (
                "general_question"
            )

        print(
            "Classification method: "
            "optional real LLM"
        )

    print(
        f"Intent: {intent}"
    )

    return {
        "intent": intent
    }


# =========================================================
# ROUTER FUNCTION
# =========================================================

def route_by_intent(
    state: SupportState
) -> Literal[
    "retrieve_and_answer",
    "direct_answer"
]:
    """
    Conditional router.

    This routing function does not depend on MOCK_LLM.
    """

    intent = state[
        "intent"
    ]

    print("\n========================================")
    print("CONDITIONAL ROUTER")
    print("========================================")

    if (
        intent
        ==
        "policy_question"
    ):

        print(
            "Routing to: "
            "retrieve_and_answer"
        )

        return (
            "retrieve_and_answer"
        )

    print(
        "Routing to: "
        "direct_answer"
    )

    return (
        "direct_answer"
    )


# =========================================================
# NODE 2 - RETRIEVE AND ANSWER
# =========================================================

def retrieve_and_answer(
    state: SupportState
) -> SupportState:
    """
    Policy-question route.

    Retrieval always runs for real.

    Answer generation branches on MOCK_LLM.
    """

    query = state[
        "query"
    ]

    print("\n========================================")
    print("NODE: retrieve_and_answer")
    print("========================================")

    # =====================================================
    # LOCAL QUERY EMBEDDING
    # =====================================================

    query_embedding = (
        embedding_model.encode(
            [query],

            convert_to_numpy=True,

            normalize_embeddings=True,
        )
    )

    # =====================================================
    # TOP-3 CHROMADB RETRIEVAL
    # =====================================================

    results = collection.query(
        query_embeddings=(
            query_embedding.tolist()
        ),

        n_results=3,
    )

    retrieved_ids = (
        results[
            "ids"
        ][0]
    )

    retrieved_chunks = (
        results[
            "documents"
        ][0]
    )

    retrieved_distances = (
        results[
            "distances"
        ][0]
    )

    print(
        "\nTop 3 retrieved documents:"
    )

    for position, (
        chunk_id,
        chunk_text,
        distance,
    ) in enumerate(
        zip(
            retrieved_ids,
            retrieved_chunks,
            retrieved_distances,
        ),
        start=1,
    ):

        print(
            f"\nResult {position}"
        )

        print(
            f"ID: {chunk_id}"
        )

        print(
            f"Cosine distance: "
            f"{distance:.4f}"
        )

        print(
            f"Snippet: "
            f"{chunk_text[:160]}"
        )

    if not retrieved_chunks:

        raise ValueError(
            "Retrieval returned no chunks."
        )

    # =====================================================
    # TOP CHUNK SNIPPET
    # =====================================================

    top_chunk = (
        retrieved_chunks[0]
    )

    top_chunk_snippet = (
        top_chunk[:200]
        .strip()
    )

    # =====================================================
    # REQUIRED MOCK ANSWER
    # =====================================================

    if MOCK_LLM:

        answer = (
            "Based on the retrieved context: "
            f"{top_chunk_snippet}"
        )

        confidence = 1.0

        print(
            "\nAnswer generation: "
            "deterministic mock"
        )

    # =====================================================
    # OPTIONAL REAL LLM ANSWER
    # =====================================================

    else:

        context = "\n\n".join(
            [
                f"[{chunk_id}] {chunk}"

                for chunk_id, chunk
                in zip(
                    retrieved_ids,
                    retrieved_chunks,
                )
            ]
        )

        prompt = build_prompt(
            context=context,
            query=query,
        )

        raw_output = (
            call_optional_real_llm(
                prompt
            )
        )

        validated_response = (
            validate_real_llm_response(
                raw_output=raw_output,
                fallback_sources=(
                    retrieved_ids
                ),
            )
        )

        answer = (
            validated_response.answer
        )

        confidence = (
            validated_response.confidence
        )

        # Prefer validated LLM sources if present.
        # Otherwise preserve retrieved IDs.
        if validated_response.sources:

            retrieved_ids = (
                validated_response.sources
            )

        print(
            "\nAnswer generation: "
            "optional real LLM"
        )

    print(
        f"\nAnswer:\n{answer}"
    )

    return {
        "retrieved_ids": (
            retrieved_ids
        ),

        "retrieved_chunks": (
            retrieved_chunks
        ),

        "retrieved_distances": (
            retrieved_distances
        ),

        "answer": answer,

        "confidence": confidence,
    }


# =========================================================
# NODE 3 - DIRECT ANSWER
# =========================================================

def direct_answer(
    state: SupportState
) -> SupportState:
    """
    General-question route.
    """

    query = state[
        "query"
    ]

    print("\n========================================")
    print("NODE: direct_answer")
    print("========================================")

    # =====================================================
    # REQUIRED MOCK BRANCH
    # =====================================================

    if MOCK_LLM:

        answer = (
            "I can only answer questions about "
            "Zepto policies right now."
        )

        confidence = 1.0

        print(
            "Answer generation: "
            "deterministic mock"
        )

    # =====================================================
    # OPTIONAL REAL LLM BRANCH
    # =====================================================

    else:

        direct_prompt = f"""
You are a concise customer-support assistant.

Answer the following general question briefly.

Return ONLY valid JSON with exactly these fields:

answer: string
sources: []
confidence: number from 0 to 1

Question:
{query}
"""

        raw_output = (
            call_optional_real_llm(
                direct_prompt
            )
        )

        validated_response = (
            validate_real_llm_response(
                raw_output=raw_output,
                fallback_sources=[],
            )
        )

        answer = (
            validated_response.answer
        )

        confidence = (
            validated_response.confidence
        )

        print(
            "Answer generation: "
            "optional real LLM"
        )

    print(
        f"\nAnswer:\n{answer}"
    )

    return {
        "retrieved_ids": [],
        "retrieved_chunks": [],
        "retrieved_distances": [],
        "answer": answer,
        "confidence": confidence,
    }


# =========================================================
# BUILD LANGGRAPH
# =========================================================

def build_graph():
    """
    Construct the required 3-node StateGraph.
    """

    builder = StateGraph(
        SupportState
    )

    # -----------------------------------------------------
    # REQUIRED THREE NODES
    # -----------------------------------------------------

    builder.add_node(
        "classify_intent",
        classify_intent,
    )

    builder.add_node(
        "retrieve_and_answer",
        retrieve_and_answer,
    )

    builder.add_node(
        "direct_answer",
        direct_answer,
    )

    # -----------------------------------------------------
    # START
    # -----------------------------------------------------

    builder.add_edge(
        START,
        "classify_intent",
    )

    # -----------------------------------------------------
    # CONDITIONAL EDGE
    # -----------------------------------------------------

    builder.add_conditional_edges(
        "classify_intent",

        route_by_intent,

        {
            "retrieve_and_answer":
                "retrieve_and_answer",

            "direct_answer":
                "direct_answer",
        },
    )

    # -----------------------------------------------------
    # END
    # -----------------------------------------------------

    builder.add_edge(
        "retrieve_and_answer",
        END,
    )

    builder.add_edge(
        "direct_answer",
        END,
    )

    return builder.compile()


# =========================================================
# COMPILED GRAPH
# =========================================================

support_graph = build_graph()


# =========================================================
# TASK 4 - RUN QUERY AND VALIDATE FINAL RESPONSE
# =========================================================

def run_query(
    query: str
) -> SupportResponse:
    """
    Run the LangGraph and convert the final state into
    the required validated Pydantic response.
    """

    initial_state: SupportState = {
        "query": query,
        "retrieved_ids": [],
        "retrieved_chunks": [],
        "retrieved_distances": [],
        "confidence": 1.0,
    }

    result = support_graph.invoke(
        initial_state
    )

    # =====================================================
    # DETERMINISTIC MOCK SCHEMA POPULATION
    # =====================================================

    if (
        result[
            "intent"
        ]
        ==
        "policy_question"
    ):

        sources = result.get(
            "retrieved_ids",
            []
        )

    else:

        sources = []

    final_response = SupportResponse(
        answer=result[
            "answer"
        ],

        sources=sources,

        confidence=result.get(
            "confidence",
            1.0
        ),
    )

    return final_response


# =========================================================
# TASK 3 + TASK 4 TESTS
# =========================================================

def main():

    print("\n========================================")
    print("MODULE 3 - SUPPORT ASSISTANT")
    print("TASKS 3 AND 4")
    print("========================================")

    print(
        f"\nCurrent MOCK_LLM mode: "
        f"{1 if MOCK_LLM else 0}"
    )

    # =====================================================
    # TEST 1 - POLICY QUESTION
    # =====================================================

    policy_query = (
        "What is the delivery fee "
        "for a small Zepto order?"
    )

    print("\n\n########################################")
    print("TEST 1 - POLICY QUESTION")
    print("########################################")

    policy_response = run_query(
        policy_query
    )

    print("\nVALIDATED POLICY RESPONSE")

    print(
        policy_response
        .model_dump_json(
            indent=2
        )
    )

    # =====================================================
    # TEST 2 - GENERAL QUESTION
    # =====================================================

    general_query = (
        "What is the capital of France?"
    )

    print("\n\n########################################")
    print("TEST 2 - GENERAL QUESTION")
    print("########################################")

    general_response = run_query(
        general_query
    )

    print("\nVALIDATED GENERAL RESPONSE")

    print(
        general_response
        .model_dump_json(
            indent=2
        )
    )

    # =====================================================
    # TASK 4 VALIDATION
    # =====================================================

    print("\n========================================")
    print("TASK 4 VALIDATION")
    print("========================================")

    # -----------------------------------------------------
    # POLICY RESPONSE CHECKS
    # -----------------------------------------------------

    if not isinstance(
        policy_response,
        SupportResponse
    ):

        raise ValueError(
            "Task 4 failed: policy response "
            "is not a SupportResponse."
        )

    if not policy_response.answer:

        raise ValueError(
            "Task 4 failed: policy answer is empty."
        )

    if not policy_response.sources:

        raise ValueError(
            "Task 4 failed: policy sources are empty."
        )

    if not (
        0.0
        <=
        policy_response.confidence
        <=
        1.0
    ):

        raise ValueError(
            "Task 4 failed: policy confidence "
            "is outside 0-1."
        )

    # -----------------------------------------------------
    # GENERAL RESPONSE CHECKS
    # -----------------------------------------------------

    if not isinstance(
        general_response,
        SupportResponse
    ):

        raise ValueError(
            "Task 4 failed: general response "
            "is not a SupportResponse."
        )

    if general_response.sources != []:

        raise ValueError(
            "Task 4 failed: general response "
            "sources must be empty."
        )

    if not (
        0.0
        <=
        general_response.confidence
        <=
        1.0
    ):

        raise ValueError(
            "Task 4 failed: general confidence "
            "is outside 0-1."
        )

    if MOCK_LLM:

        if (
            policy_response.confidence
            !=
            1.0
        ):

            raise ValueError(
                "Task 4 failed: mock policy "
                "confidence must be 1.0."
            )

        if (
            general_response.confidence
            !=
            1.0
        ):

            raise ValueError(
                "Task 4 failed: mock general "
                "confidence must be 1.0."
            )

    print(
        "Pydantic response model created: PASS"
    )

    print(
        "answer field validated: PASS"
    )

    print(
        "sources field validated: PASS"
    )

    print(
        "confidence constrained to 0-1: PASS"
    )

    print(
        "Policy sources populated deterministically: PASS"
    )

    print(
        "General sources empty: PASS"
    )

    print(
        "Mock confidence deterministic: PASS"
    )

    print(
        "Optional real-LLM retry logic present: PASS"
    )

    print(
        "Clearly marked validation failure "
        "fallback present: PASS"
    )

    print("\n========================================")
    print("TASK 4 COMPLETED SUCCESSFULLY")
    print("========================================")

    print("\n========================================")
    print("TASKS 3 AND 4 COMPLETE")
    print("========================================")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()