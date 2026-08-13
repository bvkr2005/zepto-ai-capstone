# Module 3 — Zepto Support Assistant

## Overview

This module implements a small RAG-based Zepto customer-support assistant using:

- local document embeddings
- Sentence Transformers
- ChromaDB
- LangGraph
- Pydantic
- FastAPI
- Docker

The graded baseline runs entirely in mock mode using:

`MOCK_LLM=1`

or with `MOCK_LLM` unset.

No external LLM API key is required for the graded solution.

The optional real-LLM path is activated only when:

`MOCK_LLM=0`


# Document Corpus

The module uses 8 Zepto policy documents:

1. `doc_01.txt` — Delivery Policy
2. `doc_02.txt` — Returns & Refunds
3. `doc_03.txt` — Membership Tiers
4. `doc_04.txt` — Order Tracking
5. `doc_05.txt` — Order Cancellation Policy
6. `doc_06.txt` — Damaged or Missing Items
7. `doc_07.txt` — Gift Cards
8. `doc_08.txt` — Customer Support Hours

These files are stored in:

`support_assistant/docs/`


# Task 1 — Ingestion, Embedding, and ChromaDB

The script:

`build_index.py`

loads all 8 policy documents.

Because the documents are short, each document is treated as one chunk.

Embeddings are generated locally using:

`all-MiniLM-L6-v2`

from the `sentence-transformers` library.

The embeddings are stored in a persistent ChromaDB collection named:

`zepto_policy_docs`

The vector store is saved under:

`support_assistant/chroma_db/`

The script validates that exactly 8 documents are stored and demonstrates top-3 similarity retrieval.


# Task 2 — Structured Prompt

The prompt template is defined in:

`prompt_template.py`

It follows the required structure:

- ROLE
- CONTEXT
- TASK
- FORMAT
- LENGTH

The prompt also contains:

- an explicit negative constraint
- a few-shot example

The negative constraint instructs the assistant not to use information outside the supplied context and not to invent policies, fees, timelines, benefits, contact methods, or exceptions.

The few-shot example demonstrates the expected structured answer format.


# Task 3 — LangGraph StateGraph

The LangGraph implementation is defined in:

`graph_app.py`

The shared graph state is implemented using a `TypedDict`.

The graph contains the required three nodes:

1. `classify_intent`
2. `retrieve_and_answer`
3. `direct_answer`

The graph begins at:

`classify_intent`

and then uses a conditional edge to route the query.


## classify_intent

In mock mode, the query is lowercased and checked for any of these keywords:

- delivery
- return
- refund
- membership
- tracking
- cancel
- gift card
- support hours

If a keyword is present, the query is classified as:

`policy_question`

Otherwise it is classified as:

`general_question`

No LLM call is made in mock mode.


## retrieve_and_answer

For a `policy_question`, the query is embedded locally using `all-MiniLM-L6-v2`.

The graph then retrieves the top 3 most similar policy documents from ChromaDB using cosine similarity.

Retrieval runs for real in both mock and optional real-LLM modes.

In mock mode, the response follows the required template:

`Based on the retrieved context: <top chunk snippet>`

No external LLM call is made.


## direct_answer

For a `general_question`, no retrieval is performed.

In mock mode, the fixed response is:

`I can only answer questions about Zepto policies right now.`

No external LLM call is made.


# Task 4 — Structured JSON Output

The final response is validated using a Pydantic model.

The response schema contains:

- `answer` — string
- `sources` — list of document/chunk IDs
- `confidence` — float between 0 and 1

For policy questions in mock mode:

- `sources` contains the top retrieved document IDs
- `confidence = 1.0`

For general questions:

- `sources = []`
- `confidence = 1.0`

The optional real-LLM path also includes validation-retry logic.

If the real LLM returns invalid JSON, validation is retried up to 2 additional times.

If validation still fails, the application returns a clearly marked error response.


# Task 5 — FastAPI

The FastAPI application is defined in:

`main.py`

The main endpoint is:

`POST /ask`

The request schema is:

```json
{
  "query": "string"
}

Example API Call 1 — Policy Question

Request:

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What is the delivery fee for a small Zepto order?"}' |
ConvertTo-Json -Depth 5

Example response:

{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_03"
  ],
  "confidence": 1.0
}

This query contains the keyword delivery, so it is classified as policy_question.

The query is then routed to retrieve_and_answer.

The most relevant source was doc_01, the Delivery Policy document.

Example API Call 2 — General Question

Request:

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What is the capital of France?"}' |
ConvertTo-Json -Depth 5

Response:

{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}

This query contains none of the policy keywords, so it is classified as general_question.

The graph routes it directly to direct_answer without retrieval.

Task 6 — Docker

The required Dockerfile is located at:

support_assistant/Dockerfile

The Docker image can be built from the project root using:

docker build -f support_assistant/Dockerfile -t zepto-support-assistant .

The container can be run locally with:

docker run --rm -p 7860:7860 zepto-support-assistant

The API is then available at:

http://127.0.0.1:7860/ask

Example Docker-based policy request:

Invoke-RestMethod `
  -Uri "http://127.0.0.1:7860/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What is the delivery fee for a small Zepto order?"}' |
ConvertTo-Json -Depth 5

The Docker container runs with:

MOCK_LLM=1

by default.

Task 7 — RAG Architecture

The RAG pipeline follows this flow:

Zepto Policy Documents
        |
        v
     Ingestion
        |
        v
SentenceTransformer
all-MiniLM-L6-v2
        |
        v
    Embeddings
        |
        v
    ChromaDB
zepto_policy_docs
        |
        v
Incoming Customer Query
        |
        v
LangGraph classify_intent
        |
        +-------------------------+
        |                         |
        v                         v
policy_question             general_question
        |                         |
        v                         v
retrieve_and_answer          direct_answer
        |
        v
Query Embedding
        |
        v
Top-3 ChromaDB Retrieval
        |
        v
Grounded Answer Generation
        |
        v
Pydantic Validation
        |
        v
FastAPI JSON Response
Stage 1 — Ingestion

The ingestion stage is handled by:

build_index.py

The script reads the 8 files from:

support_assistant/docs/

Each document is treated as one policy chunk.

Stage 2 — Embedding

The embedding stage is also handled by:

build_index.py

Each policy chunk is converted into a vector using:

SentenceTransformer("all-MiniLM-L6-v2")

The vectors are stored in the ChromaDB collection:

zepto_policy_docs

Stage 3 — Retrieval

Retrieval is handled by the LangGraph node:

retrieve_and_answer

The customer's query is embedded using the same local Sentence Transformer model.

ChromaDB returns the top 3 most similar policy chunks using cosine similarity.

This retrieval step always runs locally and does not depend on the MOCK_LLM setting.

Stage 4 — Generation

Answer generation is handled by:

retrieve_and_answer for policy questions
direct_answer for general questions

The structured prompt used by the optional real-LLM path is defined in:

prompt_template.py

MOCK_LLM Behavior

The application checks the environment variable:

MOCK_LLM

Default / Graded Mode

When:

MOCK_LLM is unset

or:

MOCK_LLM=1

the application runs fully offline.

In this mode:

intent classification uses the required keyword heuristic
retrieval still uses real local embeddings and ChromaDB
policy answers use the deterministic retrieved-context template
general answers use a fixed canned response
no LLM API network calls are made
confidence is deterministically set to 1.0
Optional Real-LLM Mode

When:

MOCK_LLM=0

the generation stages may call a real LLM.

In this optional mode:

classify_intent may use the real LLM for classification
retrieve_and_answer still performs real ChromaDB retrieval first
retrieved context is inserted into the structured prompt
direct_answer may call the LLM without retrieval
real-LLM JSON output is validated by Pydantic
invalid structured output is retried up to 2 additional times

The optional real-LLM path is not required for grading.

Installation

Install the required packages:

pip install sentence-transformers chromadb langgraph fastapi uvicorn pydantic
Running the Module

First build the vector database:

python support_assistant/build_index.py

Test the structured prompt:

python support_assistant/prompt_template.py

Test the LangGraph and Pydantic flow:

$env:MOCK_LLM="1"
python support_assistant/graph_app.py

Start FastAPI:

$env:MOCK_LLM="1"
uvicorn support_assistant.main:app --reload

Or run the Dockerized application:

docker build -f support_assistant/Dockerfile -t zepto-support-assistant .

docker run --rm -p 7860:7860 zepto-support-assistant
Conclusion

This module implements a complete offline-first RAG support assistant.

The pipeline performs document ingestion, local embedding generation, persistent ChromaDB storage, intent classification, conditional LangGraph routing, similarity-based retrieval, deterministic mock answer generation, Pydantic validation, FastAPI serving, and Docker containerization.

The default graded implementation works without an LLM account, API key, or external LLM network request.


Save it with **Ctrl + S**.

At that point, all Module 3 requirements are implemented: the 8 corpus documents and vector index, structured prompt, 3-node conditional LangGraph, validated schema, FastAPI examples, Dockerfile, and architecture documentation. :contentReference[oaicite:0]{index=0}

The next step is the **final Module 3 file/Git check**. Run:

```powershell
git status