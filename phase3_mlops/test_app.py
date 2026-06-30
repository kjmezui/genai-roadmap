import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

app_module = import_module("03_fastapi_app")
app = app_module.app


@pytest.fixture
def client():
    """Crée un client de test qui simule des requêtes HTTP sans serveur réel."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Le endpoint /health doit toujours répondre 200 avec le RAG chargé."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["rag_charge"] is True


def test_ask_question_valide(client):
    """Une question valide doit retourner une réponse structurée."""
    response = client.post(
        "/ask",
        json={"question": "Quel est le délai de remboursement en cas de fraude ?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "reponse" in data
    assert len(data["reponse"]) > 0
    assert "latence_ms" in data


def test_ask_question_trop_courte(client):
    """Une question trop courte doit être rejetée par la validation."""
    response = client.post("/ask", json={"question": "Oui"})
    assert response.status_code == 422


def test_ask_question_hors_contexte(client):
    """Une question hors corpus ne doit jamais halluciner une réponse."""
    response = client.post(
        "/ask",
        json={"question": "Quelle est la couleur du logo de la banque ?"}
    )
    assert response.status_code == 200
    data = response.json()
    reponse_minuscule = data["reponse"].lower()
    # Vérifie que SENTINEL admet ne pas savoir plutôt que d'halluciner
    assert ("ne trouve pas" in reponse_minuscule
            or "non applicable" in reponse_minuscule
            or "pas mentionné" in reponse_minuscule
            or "pas disponible" in reponse_minuscule)


def test_metrics_endpoint(client):
    """Le endpoint /metrics doit retourner des compteurs cohérents."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["nb_requetes"] >= 0
    assert data["latence_moyenne_ms"] >= 0