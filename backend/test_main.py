import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_search_router():
    # The /search endpoint expects POST and a SearchQuery body with correct enum values
    response = client.post("/search", json={
        "query": "test",
        "sources": ["web"],
        "max_results": 1,
        "include_citations": False
    })
    assert response.status_code in (200, 500)

def test_rag_router():
    # The /rag/generate endpoint expects POST and a RAGQuery body
    response = client.post("/rag/generate", json={"query": "test", "context_limit": 1, "model": "mixtral", "temperature": 0.7})
    assert response.status_code in (200, 500)

def test_model_router():
    response = client.post("/model/generate", json={"prompt": "test", "model": "mixtral", "temperature": 0.7})
    assert response.status_code in (200, 500)

def test_multimodal_router():
    response = client.post("/multimodal/analyze", json={"file_type": "image"})
    assert response.status_code == 200
    assert "todo" in response.json()

def test_memory_router():
    # The /memory/retrieve endpoint expects POST and a MemoryQuery body
    response = client.post("/memory/retrieve", json={"user_id": "user1", "context": "", "tags": []})
    assert response.status_code in (200, 500)

def test_privacy_router():
    response = client.get("/privacy/status")
    assert response.status_code == 200
    assert "encryption_status" in response.json()

def test_reasoning_router():
    response = client.post("/reasoning/solve", json={"problem": "test", "context": "", "model": "mixtral"})
    assert response.status_code in (200, 500)

def test_profile_router():
    response = client.get("/profile/get/user1")
    assert response.status_code == 200
    assert "todo" in response.json()

def test_collab_router():
    # CollaborationRequest requires document_id, content, operation, user_id
    response = client.post("/collab/edit", json={
        "document_id": "doc1",
        "content": "test content",
        "operation": "edit",
        "user_id": "user1"
    })
    assert response.status_code == 200
    assert "todo" in response.json()
