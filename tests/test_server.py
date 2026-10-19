import json
import httpx
import pytest
from fastapi.testclient import TestClient

from sys2.engine import System2Engine
from sys2.server import create_app


@pytest.fixture(scope="module")
def client():
    # Use CPU for deterministic and fast test execution
    engine = System2Engine(device="cpu")
    app = create_app(engine=engine)
    return TestClient(app)


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["non_autoregressive"] is True
    assert "resident_domain" in data
    assert "available_domains" in data
    assert "science" in data["available_domains"]


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "system2" in model_ids
    assert "system2-guard" in model_ids
    assert "system2:science" in model_ids
    assert "system2:code" in model_ids
    assert "system2:med" in model_ids


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
    assert data["model"] == "system2"
    assert "state_of_matter" in data["answers"]

    ans = data["answers"]["state_of_matter"]
    assert ans["type"] == "choice"
    assert ans["choice"] == "opt_liquid"
    assert ans["confidence"] > 0.0
    assert "probabilities" in ans
    assert ans["probabilities"]["opt_liquid"] > ans["probabilities"]["opt_solid"]
    assert sum(ans["probabilities"].values()) == pytest.approx(1.0, abs=0.01)

    assert data["usage"]["output_tokens"] == 0
    assert data["usage"]["input_tokens"] > 0


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

    noul_ans = data["answers"]["is_liquid"]
    assert noul_ans["type"] == "noul"
    assert 0.0 <= noul_ans["noul"] <= 1.0
    assert noul_ans["noul"] > 0.5  # Liquid is correct

    bool_ans = data["answers"]["is_liquid_bool"]
    assert bool_ans["type"] == "bool"
    assert bool_ans["bool"] is True
    assert 0.0 <= bool_ans["probability"] <= 1.0


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

    score_ans = data["answers"]["temperature_level"]
    assert score_ans["type"] == "score"
    assert 0.0 <= score_ans["score"] <= 2.0
    assert len(score_ans["probabilities"]) == 3
    assert sum(score_ans["probabilities"]) == pytest.approx(1.0, abs=0.01)
    assert score_ans["legend"]["0"] == "Solid frozen ice"
    assert score_ans["legend"]["1"] == "Liquid water"


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
            {
                "type": "score",
                "name": "energy_scale",
                "instructions": "Relative photon absorption intensity.",
                "levels": [
                    {"label": "minimal absorption"},
                    {"label": "moderate absorption"},
                    {"label": "maximal absorption"},
                ],
            },
        ],
    }
    resp = client.post("/v1/decisions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "system2:science"
    assert len(data["answers"]) == 3

    # 1. Predicate
    p_ans = data["answers"][0]
    assert p_ans["name"] == "reflects_green"
    assert p_ans["type"] == "predicate"
    assert 0.0 <= p_ans["probability"] <= 1.0
    assert p_ans["probability"] > 0.5

    # 2. Choice
    c_ans = data["answers"][1]
    assert c_ans["name"] == "leaf_color"
    assert c_ans["type"] == "choice"
    assert c_ans["choice"] == "green"
    assert c_ans["probability"] > 0.5

    # 3. Score
    s_ans = data["answers"][2]
    assert s_ans["name"] == "energy_scale"
    assert s_ans["type"] == "score"
    assert 0.0 <= s_ans["score"] <= 2.0

    # Usage
    assert data["usage"]["completion_tokens"] == 0
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"]


def test_systemone_invalid_question_type(client):
    payload = {
        "model": "system2",
        "state": "Some context",
        "questions": {
            "bad_q": {
                "type": "unsupported_magic_type",
                "instructions": "Do something",
            }
        },
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 400
    assert "Unsupported question type" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_reverse_proxy_chat_completions(monkeypatch):
    engine = System2Engine(device="cpu")
    app = create_app(engine=engine, upstream_llm="http://upstream-llm:1234")

    async def mock_send(self, request, **kwargs):
        content = json.dumps({
            "id": "chatcmpl-mock",
            "choices": [{"message": {"role": "assistant", "content": "Generative answer from LM Studio."}}],
        }).encode("utf-8")
        return httpx.Response(200, content=content, headers={"content-type": "application/json"})

    monkeypatch.setattr(httpx.AsyncClient, "send", mock_send)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Generative answer from LM Studio."

