"""Tests for System2 server fixes, /v1/route, /v1/verify, and Docker packaging."""

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from routing.domain_router import DomainRouter
from sys2.engine import System2Engine
from sys2.guard import GuardResult, GuardVerdict
from sys2.server import create_app


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def engine():
    return System2Engine(device="cpu")


@pytest.fixture(scope="module")
def client(engine):
    app = create_app(engine=engine, device="cpu")
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. Server Bug Fixes: Thresholds, Validation, Reverse Proxy Cleanup
# -----------------------------------------------------------------------------


def test_guard_preserves_explicit_zero_thresholds(monkeypatch):
    """Explicit 0.0 thresholds must not be coerced to 0.70 / 0.40 via falsy `or`."""
    captured = {}

    class SpyGatekeeper:
        def __init__(self, engine=None, device=None, allow_threshold=0.70, block_threshold=0.70, abstain_threshold=0.40):
            captured["allow_threshold"] = allow_threshold
            captured["block_threshold"] = block_threshold
            captured["abstain_threshold"] = abstain_threshold

        def evaluate(self, command: str) -> GuardResult:
            return GuardResult(
                command=command,
                verdict=GuardVerdict.ALLOWED,
                confidence=0.50,
                consensus="3/5",
                category="safe",
                explanation="ok",
                per_head_choices=[2, 2, 2, 2, 2],
                latency_ms=1.0,
            )

    monkeypatch.setattr("sys2.server.AgentGatekeeper", SpyGatekeeper)
    app = create_app(engine=MagicMock(), router=MagicMock(), device="cpu")
    tc = TestClient(app)

    resp = tc.post(
        "/guard",
        json={
            "command": "echo ok",
            "allow_threshold": 0.0,
            "block_threshold": 0.0,
            "abstain_threshold": 0.0,
        },
    )
    assert resp.status_code == 200
    assert captured == {
        "allow_threshold": 0.0,
        "block_threshold": 0.0,
        "abstain_threshold": 0.0,
    }


def test_deliberate_preserves_explicit_zero_max_len():
    """Explicit max_len=0 must not be replaced by 384 via falsy `or`."""
    captured = {}

    class StubResult:
        selected_choice = "A"
        selected_index = 0
        confidence = 0.9
        head_consensus = "5/5"
        choice_probabilities = [0.9, 0.1]
        domain = "science"
        per_head_choices = [0, 0, 0, 0, 0]
        deliberation_latency_ms = 1.0

    mock_engine = MagicMock()

    def fake_adjudicate(question, choices, domain=None, max_len=1024):
        captured["max_len"] = max_len
        return StubResult()

    mock_engine.adjudicate.side_effect = fake_adjudicate
    app = create_app(engine=mock_engine, router=MagicMock(), device="cpu")
    tc = TestClient(app)

    resp = tc.post(
        "/deliberate",
        json={"question": "Q?", "choices": ["A", "B"], "max_len": 0},
    )
    assert resp.status_code == 200
    assert captured["max_len"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {
            "model": "system2",
            "input": "test",
            "questions": [{"type": "choice", "name": "q1", "instructions": "pick", "choices": []}],
        },
        {
            "model": "system2",
            "input": "test",
            "questions": [{"type": "choice", "name": "q1", "instructions": "pick"}],
        },
        {
            "model": "system2",
            "input": "test",
            "questions": [{"type": "score", "name": "q1", "instructions": "rate", "levels": []}],
        },
        {
            "model": "system2",
            "input": "test",
            "questions": [{"type": "score", "name": "q1", "instructions": "rate"}],
        },
        {
            "model": "system2",
            "input": "test",
            "questions": [{"type": "unknown_kind", "name": "q1", "instructions": "what"}],
        },
    ],
)
def test_decisions_invalid_inputs_return_400(payload):
    app = create_app(engine=MagicMock(), router=MagicMock(), device="cpu")
    tc = TestClient(app)
    resp = tc.post("/v1/decisions", json=payload)
    assert resp.status_code == 400


@pytest.mark.parametrize(
    "payload",
    [
        {
            "model": "system2",
            "state": "ctx",
            "questions": {"q1": {"type": "choice", "instructions": "pick", "criteria": {}}},
        },
        {
            "model": "system2",
            "state": "ctx",
            "questions": {"q1": {"type": "score", "instructions": "rate", "criteria": []}},
        },
        {
            "model": "system2",
            "state": "ctx",
            "questions": {"q1": {"type": "bool", "instructions": "flag", "criteria": {}}},
        },
        {
            "model": "system2",
            "state": "ctx",
            "questions": {"q1": {"type": "noul", "instructions": "prob", "criteria": []}},
        },
        {
            "model": "system2",
            "state": "ctx",
            "questions": {"q1": {"type": "bogus", "instructions": "what"}},
        },
    ],
)
def test_systemone_invalid_inputs_return_400(payload):
    app = create_app(engine=MagicMock(), router=MagicMock(), device="cpu")
    tc = TestClient(app)
    resp = tc.post("/v1/systemone", json=payload)
    assert resp.status_code == 400


def test_reverse_proxy_background_aclose_and_lazy_client(monkeypatch):
    """Verify reverse proxy works without lifespan and awaits resp.aclose via BackgroundTask."""
    events = {"send_calls": 0, "aclose_awaited": False}

    class ByteAsyncStream(httpx.AsyncByteStream):
        def __init__(self, data: bytes):
            self._data = data

        async def __aiter__(self):
            yield self._data

    class TrackingResponse(httpx.Response):
        async def aclose(self) -> None:
            events["aclose_awaited"] = True
            await super().aclose()

    async def mock_send(self, request, **kwargs):
        events["send_calls"] += 1
        assert kwargs.get("stream") is True
        return TrackingResponse(
            200,
            stream=ByteAsyncStream(
                b'{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"ok"}}]}'
            ),
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", mock_send)

    app = create_app(
        engine=MagicMock(),
        router=MagicMock(),
        device="cpu",
        upstream_llm="http://upstream-llm:1234",
    )

    # TestClient without `with` skips FastAPI lifespan, exercising lazy http_client init.
    tc = TestClient(app)
    resp = tc.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert events["send_calls"] == 1
    assert events["aclose_awaited"] is True


def test_reverse_proxy_upstream_error_returns_502(monkeypatch):
    async def mock_send_fail(self, request, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "send", mock_send_fail)
    app = create_app(
        engine=MagicMock(),
        router=MagicMock(),
        device="cpu",
        upstream_llm="http://upstream-llm:1234",
    )
    with TestClient(app) as tc:
        resp = tc.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 502


# -----------------------------------------------------------------------------
# 2. /v1/route + "general" Fallback Track
# -----------------------------------------------------------------------------


def test_domain_router_tracks_and_general_fallback(engine):
    assert len(DomainRouter.TRACKS) == 13
    assert "general" not in DomainRouter.TRACKS
    assert DomainRouter.ALL_TRACKS == DomainRouter.TRACKS + ["general"]
    assert "general" in DomainRouter.TRACK_NAMES

    # Verify engine adapter loading cleanly maps "general" -> science checkpoint.
    _, sys2_general = engine._load_adapter("general")
    _, sys2_science = engine._load_adapter("science")
    assert sys2_general is not None and sys2_science is not None

    # Missing centroids: default fallback="science" vs explicit fallback="general"
    no_centroid_router = DomainRouter(centroids_path="/nonexistent/domain_centroids.pt", device="cpu")
    dec_default = no_centroid_router.route("Tell me something interesting.")
    assert dec_default.selected_track == "science"
    assert dec_default.method == "fallback"

    dec_general = no_centroid_router.route("Tell me something interesting.", fallback_track="general")
    assert dec_general.selected_track == "general"
    assert dec_general.method == "fallback"


def test_v1_route_contract_and_general_fallback(client, engine, monkeypatch):
    # Stage 1 lexical hit still returns specialized track
    resp_code = client.post("/v1/route", json={"prompt": "```python\ndef add(a, b):\n    return a + b\n```"})
    assert resp_code.status_code == 200
    data_code = resp_code.json()
    assert data_code["track"] == "code"
    assert data_code["selected_track"] == "code"
    assert data_code["method"] == "lexical"
    assert data_code["confidence"] >= 0.88

    # Stage 2 OOD / ambiguous embedding -> /v1/route returns "general", /route keeps "science"
    centroids = engine.router.centroids
    assert centroids is not None
    # Construct unit vector in the nullspace of the 13 centroids so all cosine sims == 0
    # and softmax is uniform (1/13 < 1.5/13 ambiguity threshold).
    _, _, vh = torch.linalg.svd(centroids.float(), full_matrices=True)
    orthogonal_emb = vh[-1].unsqueeze(0)

    monkeypatch.setattr(engine.router, "encode_text", lambda _text: orthogonal_emb)

    ood_prompt = "Zqxj kqpw blorp flimflam neutral utterance."
    resp_v1 = client.post("/v1/route", json={"prompt": ood_prompt})
    assert resp_v1.status_code == 200
    v1_data = resp_v1.json()
    assert v1_data["track"] == "general"
    assert v1_data["selected_track"] == "general"
    assert v1_data["track_name"] == DomainRouter.TRACK_NAMES["general"]
    assert v1_data["method"] == "fallback"

    resp_legacy = client.post("/route", json={"prompt": ood_prompt})
    assert resp_legacy.status_code == 200
    assert resp_legacy.json()["selected_track"] == "science"
    assert resp_legacy.json()["method"] == "fallback"


# -----------------------------------------------------------------------------
# 3. /v1/verify Across All Four Domains
# -----------------------------------------------------------------------------


def test_v1_verify_units(client):
    ok_resp = client.post("/v1/verify", json={"code_or_expression": "10 N * 5 m == 50 J", "domain": "units"})
    assert ok_resp.status_code == 200
    assert ok_resp.json()["valid"] is True

    bad_resp = client.post("/v1/verify", json={"code_or_expression": "10 N / 5 m == 50 J", "domain": "units"})
    assert bad_resp.status_code == 200
    bad_data = bad_resp.json()
    assert bad_data["valid"] is False
    assert bad_data["code"] == "UNIT_DIMENSION_MISMATCH"

    ctx_bad = client.post(
        "/v1/verify",
        json={"code_or_expression": "50 joules", "domain": "units", "context": "What is the velocity?"},
    )
    assert ctx_bad.status_code == 200
    assert ctx_bad.json()["valid"] is False
    assert ctx_bad.json()["code"] == "UNIT_DIMENSION_MISMATCH"


def test_v1_verify_algebra(client):
    ok_resp = client.post(
        "/v1/verify",
        json={"code_or_expression": "(x+1)^2 = x^2 + 2*x + 1", "domain": "algebra"},
    )
    assert ok_resp.status_code == 200
    assert ok_resp.json()["valid"] is True

    bad_resp = client.post(
        "/v1/verify",
        json={"code_or_expression": "x + 1 = x + 2", "domain": "algebra"},
    )
    assert bad_resp.status_code == 200
    assert bad_resp.json()["valid"] is False
    assert bad_resp.json()["code"] == "ALGEBRA_MISMATCH"

    parse_resp = client.post(
        "/v1/verify",
        json={"code_or_expression": "((x + = x", "domain": "algebra"},
    )
    assert parse_resp.status_code == 200
    assert parse_resp.json()["valid"] is False
    assert parse_resp.json()["code"] == "ALGEBRA_PARSE_ERROR"


def test_v1_verify_logic(client):
    valid_entail = client.post(
        "/v1/verify",
        json={"code_or_expression": "P, P -> Q |= Q", "domain": "logic"},
    )
    assert valid_entail.status_code == 200
    assert valid_entail.json()["valid"] is True

    invalid_fallacy = client.post(
        "/v1/verify",
        json={"code_or_expression": "P -> Q, Q |= P", "domain": "logic"},
    )
    assert invalid_fallacy.status_code == 200
    assert invalid_fallacy.json()["valid"] is False
    assert invalid_fallacy.json()["code"] == "LOGIC_VIOLATION"

    contra = client.post(
        "/v1/verify",
        json={"code_or_expression": "P & ~P", "domain": "logic"},
    )
    assert contra.status_code == 200
    assert contra.json()["valid"] is False
    assert contra.json()["code"] == "LOGIC_VIOLATION"

    neg_ctx = client.post(
        "/v1/verify",
        json={
            "code_or_expression": "The material will conduct electricity.",
            "context": "The material does not conduct electricity.",
            "domain": "logic",
        },
    )
    assert neg_ctx.status_code == 200
    assert neg_ctx.json()["valid"] is False
    assert neg_ctx.json()["code"] == "LOGIC_VIOLATION"


def test_v1_verify_safety(client):
    safe_resp = client.post("/v1/verify", json={"code_or_expression": "ls -la", "domain": "safety"})
    assert safe_resp.status_code == 200
    assert safe_resp.json()["valid"] is True

    block_resp = client.post("/v1/verify", json={"code_or_expression": "rm -rf /", "domain": "safety"})
    assert block_resp.status_code == 200
    block_data = block_resp.json()
    assert block_data["valid"] is False
    assert block_data["code"] == "SAFETY_BLOCKED"
    assert block_data["reason"]

    abstain_resp = client.post("/v1/verify", json={"code_or_expression": "reboot", "domain": "safety"})
    assert abstain_resp.status_code == 200
    abstain_data = abstain_resp.json()
    assert abstain_data["valid"] is False
    assert abstain_data["code"] == "SAFETY_ABSTAIN"
    assert abstain_data["reason"]


def test_v1_verify_unexpected_verifier_error_fails_closed(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic verifier crash")

    monkeypatch.setattr("sys2.server.UnitVerifier.verify_statement", boom)
    app = create_app(engine=MagicMock(), router=MagicMock(), device="cpu")
    tc = TestClient(app)
    resp = tc.post("/v1/verify", json={"code_or_expression": "10 m == 10 m", "domain": "units"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert data["code"] == "VERIFIER_ERROR"
    assert "synthetic verifier crash" in (data["error"] or "")


# -----------------------------------------------------------------------------
# 4. Dockerfile & .dockerignore Packaging Validation
# -----------------------------------------------------------------------------


def test_dockerfile_packages_and_copy_targets_exist():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    for pkg in ("pint", "sympy", "z3-solver"):
        assert pkg in dockerfile, f"Missing {pkg} in Dockerfile"
    assert 'VOLUME ["/app/models"]' in dockerfile
    assert "mkdir -p /app/models" in dockerfile

    copy_lines = [line.strip() for line in dockerfile.splitlines() if line.strip().startswith("COPY ")]
    assert len(copy_lines) > 0
    for line in copy_lines:
        parts = line.split()
        # COPY <src> <dst>
        assert len(parts) >= 3
        for src in parts[1:-1]:
            assert (REPO_ROOT / src).exists(), f"Dockerfile COPY target missing: {src}"

    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in (".git", "scratch/", "data/", "models/test_smoke*.pt", "runs/", "notes_local/", "__pycache__/", ".pytest_cache/"):
        assert pattern in dockerignore
    # Never ignore routing/domain_centroids.pt via global *.pt
    assert "*.pt" not in [ln.strip() for ln in dockerignore.splitlines()]
