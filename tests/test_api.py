"""
Unit Tests for FastAPI Endpoints
================================
Validates health checks, status queries, and mock upload requests.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check():
    """Verify health diagnostics endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Nexus" in data["service"]
    assert data["custom_attention"] == "NumPy (CLO01/CLO02)"


def test_list_videos():
    """Verify listing uploaded videos returns array."""
    response = client.get("/api/videos/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_invalid_job_status():
    """Verify non-existent job ID returns 404."""
    response = client.get("/api/processing/non_existent_job_12345")
    assert response.status_code == 404
