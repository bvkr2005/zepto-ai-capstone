"""Tests for Module 3 (support assistant API).

These need the assistant's dependencies AND the all-MiniLM-L6-v2 embedding model
(downloaded on first use, so internet access is needed once). If anything is
missing the whole module is skipped rather than failing.
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOCK_LLM", "1")

try:
    from fastapi.testclient import TestClient
    from support_assistant.main import app
    client = TestClient(app)
    SKIP_REASON = None
except Exception as exc:  # missing deps, or embedding model unavailable
    client = None
    SKIP_REASON = f"support assistant unavailable: {type(exc).__name__}: {exc}"


@unittest.skipIf(client is None, SKIP_REASON or "")
class AskEndpointTests(unittest.TestCase):
    def test_health_endpoint(self):
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "running")

    def test_policy_question_returns_sources(self):
        r = client.post("/ask", json={"query": "What is the delivery fee for a small order?"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(set(body), {"answer", "sources", "confidence"})
        self.assertGreater(len(body["sources"]), 0)
        self.assertTrue(all(s.startswith("doc_") for s in body["sources"]))
        self.assertTrue(0.0 <= body["confidence"] <= 1.0)

    def test_general_question_is_not_answered(self):
        r = client.post("/ask", json={"query": "What is the capital of France?"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["sources"], [])
        self.assertIn("only answer questions about Zepto policies", body["answer"])

    def test_empty_query_is_rejected(self):
        self.assertEqual(client.post("/ask", json={"query": ""}).status_code, 422)

    def test_missing_query_is_rejected(self):
        self.assertEqual(client.post("/ask", json={}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
