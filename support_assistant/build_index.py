from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

DOCS_DIR = BASE_DIR / "docs"

CHROMA_DIR = BASE_DIR / "chroma_db"


# =========================================================
# CONFIGURATION
# =========================================================

COLLECTION_NAME = "zepto_policy_docs"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# LOAD DOCUMENTS
# =========================================================

def load_documents():
    """
    Load all .txt files from the docs folder.

    Each document is treated as one chunk because
    the supplied Zepto policy documents are short.
    """

    document_files = sorted(
        DOCS_DIR.glob("doc_*.txt")
    )

    if len(document_files) != 8:
        raise ValueError(
            f"Expected 8 policy documents, "
            f"but found {len(document_files)}."
        )

    documents = []

    for file_path in document_files:
        text = file_path.read_text(
        encoding="utf-8-sig"
        ).strip()

        if not text:
            raise ValueError(
                f"{file_path.name} is empty."
            )

        documents.append(
            {
                "id": file_path.stem,
                "text": text,
                "source": file_path.name,
            }
        )

    return documents


# =========================================================
# BUILD CHROMADB INDEX
# =========================================================

def build_index():
    """
    Load all 8 policy documents, create local embeddings
    with all-MiniLM-L6-v2, and persist them in ChromaDB.
    """

    print("\n========================================")
    print("MODULE 3 - SUPPORT ASSISTANT")
    print("TASK 1 - BUILD VECTOR INDEX")
    print("========================================")

    documents = load_documents()

    print(
        f"\nLoaded {len(documents)} documents."
    )

    # -----------------------------------------------------
    # LOAD LOCAL EMBEDDING MODEL
    # -----------------------------------------------------

    print(
        "\nLoading sentence-transformers model..."
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print(
        f"Embedding model loaded: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    # -----------------------------------------------------
    # CREATE EMBEDDINGS
    # -----------------------------------------------------

    texts = [
        document["text"]
        for document in documents
    ]

    print(
        "\nGenerating embeddings..."
    )

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    # -----------------------------------------------------
    # CREATE PERSISTENT CHROMADB CLIENT
    # -----------------------------------------------------

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Recreate collection so rerunning the script
    # produces a clean deterministic index.
    try:
        client.delete_collection(
            COLLECTION_NAME
        )
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    # -----------------------------------------------------
    # INSERT DOCUMENTS
    # -----------------------------------------------------

    ids = [
        document["id"]
        for document in documents
    ]

    metadatas = [
        {
            "source": document["source"]
        }
        for document in documents
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    # -----------------------------------------------------
    # VALIDATE COLLECTION
    # -----------------------------------------------------

    count = collection.count()

    print(
        f"\nDocuments stored in ChromaDB: "
        f"{count}"
    )

    if count != 8:
        raise ValueError(
            "Task 1 failed: ChromaDB does not "
            "contain exactly 8 documents."
        )

    # -----------------------------------------------------
    # TEST RETRIEVAL
    # -----------------------------------------------------

    test_query = (
        "How much does delivery cost?"
    )

    print("\n========================================")
    print("TEST RETRIEVAL")
    print("========================================")

    print(
        f"Query: {test_query}"
    )

    query_embedding = embedding_model.encode(
        [test_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,
    )

    result_ids = results[
        "ids"
    ][0]

    result_documents = results[
        "documents"
    ][0]

    result_distances = results[
        "distances"
    ][0]

    for index, (
        result_id,
        result_document,
        result_distance,
    ) in enumerate(
        zip(
            result_ids,
            result_documents,
            result_distances,
        ),
        start=1,
    ):

        print(
            f"\nResult {index}"
        )

        print(
            f"ID: {result_id}"
        )

        print(
            f"Cosine distance: "
            f"{result_distance:.4f}"
        )

        print(
            f"Text: "
            f"{result_document[:200]}"
        )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if not result_ids:
        raise ValueError(
            "Task 1 failed: retrieval returned "
            "no results."
        )

    print("\n========================================")
    print("TASK 1 VALIDATION")
    print("========================================")

    print(
        "8 policy documents loaded: PASS"
    )

    print(
        "Local embedding model loaded: PASS"
    )

    print(
        "Embeddings generated: PASS"
    )

    print(
        "ChromaDB collection created: PASS"
    )

    print(
        "8 documents stored in ChromaDB: PASS"
    )

    print(
        "Top-3 retrieval works: PASS"
    )

    print("\n========================================")
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("========================================")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    build_index()