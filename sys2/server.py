"""server.py - System2 Non-Autoregressive Deliberation & Decision Server.

Provides high-performance asynchronous REST endpoints:
  1. POST /deliberate   : Full 5-head parallel feedforward deliberation (< 30ms on MPS, < 200ms on CPU)
  2. POST /route        : Ultra-fast two-stage domain routing (< 0.1ms lexical, < 1ms centroid)
  3. POST /v1/route     : Pi-compatible domain routing with "general" fallback track
  4. POST /v1/verify    : Symbolic & safety verification (units, algebra, logic, safety)
  5. POST /guard        : Non-autoregressive agent command safety gatekeeper
  6. GET  /health       : Server liveness, memory status, and track inventory
  7. GET  /tracks       : Metadata inventory across all 13 specialized tracks
  8. POST /v1/systemone : Jev / TypeSafe AI decision endpoint compatibility
  9. POST /v1/decisions : OpenAI-compatible structured decision endpoint compatibility
 10. Reverse-proxy      : Transparently proxies generative LLM routes (/v1/chat/completions) if upstream-llm configured

Usage:
  # Standalone server:
  python3 -m sys2.server --port 8000

  # With upstream LLM proxy (e.g. LM Studio on port 1235):
  python3 -m sys2.server --port 1234 --upstream-llm http://localhost:1235
"""

import argparse
import asyncio
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
import torch
import uvicorn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from routing.domain_router import DomainRouter  # noqa: E402
from sys2.engine import CHECKPOINT_MAP, System2Engine  # noqa: E402
from sys2.guard import AgentGatekeeper, GuardVerdict  # noqa: E402
from sys2.symbolic import AlgebraVerifier, LogicVerifier, UnitVerifier  # noqa: E402


# -----------------------------------------------------------------------------
# Data Models: Native System2 REST Endpoints
# -----------------------------------------------------------------------------

class DeliberateRequest(BaseModel):
    question: str = Field(..., description="The query, vignette, or question to adjudicate")
    choices: List[str] = Field(..., min_length=2, max_length=4, description="List of 2 to 4 candidate choices")
    domain: Optional[str] = Field(default=None, description="Optional manual domain override (e.g. 'code', 'med', 'safety')")
    max_len: Optional[int] = Field(default=384, description="Max sequence length ceiling")


class DeliberateResponse(BaseModel):
    winning_choice: str
    winning_index: int
    letter: str
    confidence: float
    consensus: str
    probabilities: List[float]
    domain: str
    per_head_choices: List[int]
    latency_ms: float


class RouteRequest(BaseModel):
    prompt: str = Field(..., description="Prompt or query text to route")


class RouteResponse(BaseModel):
    selected_track: str
    track_name: str
    confidence: float
    probabilities: Dict[str, float]
    method: str
    latency_ms: float


class V1RouteResponse(BaseModel):
    track: str
    selected_track: str
    track_name: str
    confidence: float
    probabilities: Dict[str, float]
    method: str
    latency_ms: float


class VerifyRequest(BaseModel):
    code_or_expression: str = Field(..., description="Command, code snippet, or symbolic expression to verify")
    domain: Literal["units", "algebra", "logic", "safety"] = Field(..., description="Verification domain")
    context: Optional[str] = Field(default=None, description="Optional context or target specification")


class VerifyResponse(BaseModel):
    valid: bool
    code: Optional[str] = None
    reason: Optional[str] = None
    error: Optional[str] = None
    details: Optional[str] = None


class GuardRequest(BaseModel):
    command: str = Field(..., description="Shell or agent terminal command to verify")
    allow_threshold: Optional[float] = Field(default=0.70, ge=0.0, le=1.0)
    block_threshold: Optional[float] = Field(default=0.70, ge=0.0, le=1.0)
    abstain_threshold: Optional[float] = Field(default=0.40, ge=0.0, le=1.0)


class GuardResponse(BaseModel):
    command: str
    verdict: str  # "ALLOWED" | "BLOCKED" | "ABSTAIN"
    confidence: float
    head_consensus: str
    category: str
    explanation: str
    per_head_choices: List[int]
    latency_ms: float


class TrackMetadata(BaseModel):
    id: str
    name: str
    checkpoint: str
    checkpoint_exists: bool


class TracksResponse(BaseModel):
    total_tracks: int
    tracks: List[TrackMetadata]


class HealthResponse(BaseModel):
    status: str
    non_autoregressive: bool
    resident_domain: Optional[str]
    available_domains: List[str]
    device: str
    memory_allocated_mb: float


# -----------------------------------------------------------------------------
# Data Models: /v1/systemone (Jev / TypeSafe AI Specification)
# -----------------------------------------------------------------------------

class SystemOneQuestion(BaseModel):
    type: str = Field(..., description="'choice', 'noul', 'bool', or 'score'")
    instructions: Optional[str] = Field(default="", description="Instruction or question text")
    criteria: Optional[Union[Dict[str, Any], List[Any], Any]] = Field(
        default=None, description="Criteria mapping"
    )


class SystemOneRequest(BaseModel):
    model: Optional[str] = Field(default="system2", description="Decision model identifier")
    state: Union[str, Dict[str, Any], List[Any]] = Field(..., description="Application context or state")
    questions: Dict[str, SystemOneQuestion] = Field(..., description="Map of question IDs to definitions")


class SystemOneResponse(BaseModel):
    model: str
    answers: Dict[str, Any]
    usage: Dict[str, int]


# -----------------------------------------------------------------------------
# Data Models: /v1/decisions (OpenAI Structured Decision Specification)
# -----------------------------------------------------------------------------

class DecisionQuestion(BaseModel):
    type: str = Field(..., description="'choice', 'score', or 'predicate'")
    name: str = Field(..., description="Unique question identifier")
    instructions: str = Field(..., description="Question prompt")
    choices: Optional[List[Dict[str, Any]]] = None
    levels: Optional[List[Dict[str, Any]]] = None


class DecisionRequest(BaseModel):
    model: Optional[str] = Field(default="system2", description="Model identifier")
    input: Union[str, Dict[str, Any], List[Any]] = Field(..., description="Context input")
    questions: List[DecisionQuestion] = Field(..., description="List of decision questions")


class DecisionResponse(BaseModel):
    model: str
    answers: List[Dict[str, Any]]
    usage: Dict[str, int]


# -----------------------------------------------------------------------------
# Application Factory
# -----------------------------------------------------------------------------

def create_app(
    engine: Optional[System2Engine] = None,
    router: Optional[DomainRouter] = None,
    gatekeeper: Optional[AgentGatekeeper] = None,
    device: Optional[str] = None,
    upstream_llm: Optional[str] = None,
) -> FastAPI:
    """Create and configure the System2 FastAPI application."""
    target_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    # State containers for lazy loading
    state: Dict[str, Any] = {
        "engine": engine,
        "router": router,
        "gatekeeper": gatekeeper,
        "device": target_device,
        "lock": asyncio.Lock(),
        "http_client": None,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if upstream_llm and state["http_client"] is None:
            state["http_client"] = httpx.AsyncClient(base_url=upstream_llm, timeout=120.0)
        yield
        if state["http_client"] is not None:
            await state["http_client"].aclose()
            state["http_client"] = None

    app = FastAPI(
        title="System2 Parallel Deliberation Server",
        description="High-speed non-autoregressive deliberation, domain routing, and agent gatekeeper",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Stateless symbolic verifiers shared across /v1/verify requests
    unit_verifier = UnitVerifier()
    algebra_verifier = AlgebraVerifier()
    logic_verifier = LogicVerifier()

    # -------------------------------------------------------------------------
    # Lazy Loader Helpers
    # -------------------------------------------------------------------------
    def get_http_client() -> httpx.AsyncClient:
        # Lazily create AsyncClient when callers use TestClient or ASGITransport
        # without entering FastAPI lifespan, avoiding NoneType errors on proxy calls.
        if state["http_client"] is None:
            state["http_client"] = httpx.AsyncClient(base_url=upstream_llm, timeout=120.0)
        return state["http_client"]

    def get_engine() -> System2Engine:
        if state["engine"] is None:
            state["engine"] = System2Engine(device=state["device"])
        return state["engine"]

    def get_router() -> DomainRouter:
        if state["router"] is None:
            if state["engine"] is not None:
                # Reuse the engine's router so /route shares its wired backbone
                # and avoids loading a duplicate centroid copy.
                state["router"] = state["engine"].router
            else:
                state["router"] = DomainRouter(device="cpu")
        return state["router"]

    def get_gatekeeper(
        allow_th: float = 0.70,
        block_th: float = 0.70,
        abstain_th: float = 0.40,
    ) -> AgentGatekeeper:
        injected = state["gatekeeper"]
        if (
            injected is not None
            and injected.allow_threshold == allow_th
            and injected.block_threshold == block_th
            and injected.abstain_threshold == abstain_th
        ):
            return injected
        eng = get_engine()
        return AgentGatekeeper(
            engine=eng,
            device=state["device"],
            allow_threshold=allow_th,
            block_threshold=block_th,
            abstain_threshold=abstain_th,
        )

    # -------------------------------------------------------------------------
    # System & Discovery Endpoints
    # -------------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    def health_check():
        mem_mb = 0.0
        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        elif hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)

        eng = state["engine"]
        resident = eng.current_domain if eng is not None else None

        return HealthResponse(
            status="healthy",
            non_autoregressive=True,
            resident_domain=resident,
            available_domains=DomainRouter.TRACKS,
            device=state["device"],
            memory_allocated_mb=round(mem_mb, 2),
        )

    @app.get("/tracks", response_model=TracksResponse)
    def list_tracks():
        tracks = []
        for track_id in DomainRouter.TRACKS:
            name = DomainRouter.TRACK_NAMES.get(track_id, track_id.upper())
            ckpt_path = CHECKPOINT_MAP.get(track_id, "")
            exists = os.path.exists(ckpt_path) if ckpt_path else False
            tracks.append(TrackMetadata(
                id=track_id,
                name=name,
                checkpoint=ckpt_path,
                checkpoint_exists=exists,
            ))
        return TracksResponse(total_tracks=len(tracks), tracks=tracks)

    @app.get("/v1/models")
    def list_models():
        model_list = [
            {"id": "system2", "object": "model", "owned_by": "system2", "type": "deliberation"},
            {"id": "system2-guard", "object": "model", "owned_by": "system2", "type": "safety_gatekeeper"},
        ]
        for t in DomainRouter.TRACKS:
            model_list.append({
                "id": f"system2:{t}",
                "object": "model",
                "owned_by": "system2",
                "type": "domain_adapter",
                "name": DomainRouter.TRACK_NAMES.get(t, t),
            })
        return {"object": "list", "data": model_list}

    # -------------------------------------------------------------------------
    # Core Deliberation & Routing Endpoints
    # -------------------------------------------------------------------------
    @app.post("/deliberate", response_model=DeliberateResponse)
    async def deliberate(req: DeliberateRequest):
        choices = list(req.choices)
        if len(choices) < 2:
            raise HTTPException(status_code=400, detail="At least 2 choices required for deliberation.")

        # Hold state["lock"] across lazy engine init as well as inference so a
        # cold .to(device)/apply_lora cannot race an in-flight /route call.
        async with state["lock"]:
            eng = await asyncio.to_thread(get_engine)
            res = await asyncio.to_thread(
                eng.adjudicate,
                req.question,
                choices,
                domain=req.domain,
                max_len=384 if req.max_len is None else req.max_len,
            )

        letters = ["A", "B", "C", "D"]
        letter = letters[res.selected_index] if res.selected_index < len(letters) else "?"
        choice_probs = res.choice_probabilities or [round(1.0 / len(choices), 4) for _ in choices]

        return DeliberateResponse(
            winning_choice=res.selected_choice,
            winning_index=res.selected_index,
            letter=letter,
            confidence=round(res.confidence, 4),
            consensus=res.head_consensus,
            probabilities=choice_probs,
            domain=res.domain,
            per_head_choices=res.per_head_choices,
            latency_ms=round(res.deliberation_latency_ms, 2),
        )

    @app.post("/route", response_model=RouteResponse)
    async def route_prompt(req: RouteRequest):
        # Serialize with deliberation and lazy engine init: /route shares the
        # cached ModernBERT backbone with System2Engine.
        async with state["lock"]:
            rtr = await asyncio.to_thread(get_router)
            decision = await asyncio.to_thread(rtr.route, req.prompt)
        track_name = DomainRouter.TRACK_NAMES.get(decision.selected_track, decision.selected_track.upper())

        return RouteResponse(
            selected_track=decision.selected_track,
            track_name=track_name,
            confidence=round(decision.confidence, 4),
            probabilities=decision.probabilities,
            method=decision.method,
            latency_ms=round(decision.latency_ms, 3),
        )

    @app.post("/v1/route", response_model=V1RouteResponse)
    async def v1_route_prompt(req: RouteRequest):
        # Share the same lock discipline as /route because DomainRouter uses the
        # resident ModernBERT backbone when falling through to Stage 2 centroids.
        async with state["lock"]:
            rtr = await asyncio.to_thread(get_router)
            # Pi clients expect unspecialized or OOD prompts to route to "general"
            # rather than silently pinning them to the "science" LoRA checkpoint.
            decision = await asyncio.to_thread(rtr.route, req.prompt, fallback_track="general")
        track_name = DomainRouter.TRACK_NAMES.get(decision.selected_track, decision.selected_track.upper())

        return V1RouteResponse(
            track=decision.selected_track,
            selected_track=decision.selected_track,
            track_name=track_name,
            confidence=round(decision.confidence, 4),
            probabilities=decision.probabilities,
            method=decision.method,
            latency_ms=round(decision.latency_ms, 3),
        )

    @app.post("/guard", response_model=GuardResponse)
    async def guard_command(req: GuardRequest):
        async with state["lock"]:
            # Use explicit None checks so legitimate 0.0 thresholds are not treated as falsy.
            gk = await asyncio.to_thread(
                get_gatekeeper,
                allow_th=0.70 if req.allow_threshold is None else req.allow_threshold,
                block_th=0.70 if req.block_threshold is None else req.block_threshold,
                abstain_th=0.40 if req.abstain_threshold is None else req.abstain_threshold,
            )
            res = await asyncio.to_thread(gk.evaluate, req.command)

        return GuardResponse(
            command=res.command,
            verdict=res.verdict.value,
            confidence=round(res.confidence, 4),
            head_consensus=res.consensus,
            category=res.category,
            explanation=res.explanation,
            per_head_choices=res.per_head_choices,
            latency_ms=round(res.latency_ms, 2),
        )

    @app.post("/v1/verify", response_model=VerifyResponse)
    async def verify_expression(req: VerifyRequest):
        expr = req.code_or_expression
        if not expr.strip():
            return VerifyResponse(
                valid=False,
                code="EMPTY_INPUT",
                reason="code_or_expression must not be empty.",
            )

        try:
            if req.domain == "units":
                has_eq = ("=" in expr) or ("==" in expr)
                stmt_res = None
                if has_eq:
                    stmt_res = unit_verifier.verify_statement(expr)
                    if not stmt_res.is_valid or not stmt_res.is_dimensionally_homogeneous:
                        return VerifyResponse(
                            valid=False,
                            code="UNIT_DIMENSION_MISMATCH",
                            reason=stmt_res.details,
                            details=stmt_res.details,
                        )
                if req.context:
                    ctx_res = unit_verifier.verify_natural_language_claim(req.context, expr)
                    if not ctx_res.is_valid or not ctx_res.is_dimensionally_homogeneous:
                        return VerifyResponse(
                            valid=False,
                            code="UNIT_DIMENSION_MISMATCH",
                            reason=ctx_res.details,
                            details=ctx_res.details,
                        )
                    return VerifyResponse(valid=True, details=ctx_res.details)
                if stmt_res is not None:
                    return VerifyResponse(valid=True, details=stmt_res.details)
                return VerifyResponse(
                    valid=True,
                    details="No explicit unit equality or context target to mismatch.",
                )

            if req.domain == "algebra":
                # Split only on standalone '=' / '==' so relational operators like '>='
                # or '<=' are never torn into invalid fragments.
                parts = re.split(r"(?<![!<>:=])==?(?![=>])", expr)
                if len(parts) >= 2:
                    last_details = ""
                    for i in range(len(parts) - 1):
                        alg_res = algebra_verifier.verify_equivalence(parts[i], parts[i + 1])
                        if alg_res.error == "ParseError":
                            return VerifyResponse(
                                valid=False,
                                code="ALGEBRA_PARSE_ERROR",
                                reason=alg_res.details,
                                error=alg_res.error,
                                details=alg_res.details,
                            )
                        if alg_res.error is not None:
                            return VerifyResponse(
                                valid=False,
                                code="VERIFIER_ERROR",
                                reason=alg_res.details,
                                error=alg_res.error,
                                details=alg_res.details,
                            )
                        if not alg_res.is_valid:
                            return VerifyResponse(
                                valid=False,
                                code="ALGEBRA_MISMATCH",
                                reason=alg_res.details,
                                details=alg_res.details,
                            )
                        last_details = alg_res.details
                    return VerifyResponse(valid=True, details=last_details)

                parsed = algebra_verifier.parse(expr)
                if parsed is None:
                    return VerifyResponse(
                        valid=False,
                        code="ALGEBRA_PARSE_ERROR",
                        reason=f"Failed to parse algebraic expression: '{expr}'",
                        error="ParseError",
                    )
                if req.context and req.context.strip():
                    if algebra_verifier.parse(req.context) is None:
                        return VerifyResponse(
                            valid=False,
                            code="ALGEBRA_PARSE_ERROR",
                            reason=f"Failed to parse algebraic context: '{req.context}'",
                            error="ParseError",
                        )
                    alg_res = algebra_verifier.verify_equivalence(req.context, expr)
                    if not alg_res.is_valid:
                        return VerifyResponse(
                            valid=False,
                            code="ALGEBRA_MISMATCH",
                            reason=alg_res.details,
                            details=alg_res.details,
                        )
                    return VerifyResponse(valid=True, details=alg_res.details)
                return VerifyResponse(valid=True, details=f"Parsed algebraic expression: {parsed}")

            if req.domain == "logic":
                logic_res = logic_verifier.verify_expression(expr, context=req.context)
                if not logic_res.is_valid:
                    return VerifyResponse(
                        valid=False,
                        code="LOGIC_VIOLATION",
                        reason=logic_res.details,
                        error=logic_res.error,
                        details=logic_res.details,
                    )
                return VerifyResponse(valid=True, details=logic_res.details)

            if req.domain == "safety":
                async with state["lock"]:
                    gk = await asyncio.to_thread(get_gatekeeper)
                    guard_res = await asyncio.to_thread(gk.evaluate, expr)
                if guard_res.verdict == GuardVerdict.ALLOWED:
                    return VerifyResponse(
                        valid=True,
                        details=f"{guard_res.verdict.value} ({guard_res.confidence:.2%}, {guard_res.consensus}): {guard_res.explanation}",
                    )
                if guard_res.verdict == GuardVerdict.BLOCKED:
                    return VerifyResponse(
                        valid=False,
                        code="SAFETY_BLOCKED",
                        reason=guard_res.explanation,
                        details=f"category={guard_res.category}, confidence={guard_res.confidence:.4f}, consensus={guard_res.consensus}",
                    )
                # Fail closed on ABSTAIN so ambiguous/uncertain commands require explicit review.
                return VerifyResponse(
                    valid=False,
                    code="SAFETY_ABSTAIN",
                    reason=guard_res.explanation,
                    details=f"category={guard_res.category}, confidence={guard_res.confidence:.4f}, consensus={guard_res.consensus}",
                )

            return VerifyResponse(
                valid=False,
                code="VERIFIER_ERROR",
                reason=f"Unsupported verification domain: {req.domain}",
                error="UnsupportedDomain",
            )
        except Exception as exc:
            # Fail closed with a structured response instead of an unhandled 500.
            return VerifyResponse(
                valid=False,
                code="VERIFIER_ERROR",
                reason=f"Verification failed due to internal error: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )

    # -------------------------------------------------------------------------
    # Backward Compatibility: /v1/systemone (Jev Specification)
    # -------------------------------------------------------------------------
    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def handle_systemone(req: SystemOneRequest):
        # Validate all questions upfront so empty criteria or unknown types fail
        # fast with 400 before loading/running the deliberation engine.
        for q_id, q_def in req.questions.items():
            q_type = q_def.type.lower()
            if q_type == "choice":
                if not isinstance(q_def.criteria, dict) or not q_def.criteria:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Criteria for '{q_id}' must be a non-empty dict mapping key -> description",
                    )
            elif q_type in ("noul", "bool"):
                if q_def.criteria is not None and (not isinstance(q_def.criteria, dict) or not q_def.criteria):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Criteria for '{q_id}' must be a non-empty dict when provided",
                    )
            elif q_type == "score":
                if not isinstance(q_def.criteria, list) or not q_def.criteria:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Criteria for '{q_id}' must be a non-empty list of level descriptions",
                    )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported question type: '{q_type}'")

        async with state["lock"]:
            eng = await asyncio.to_thread(get_engine)
        state_str = json.dumps(req.state) if not isinstance(req.state, str) else req.state
        answers: Dict[str, Any] = {}

        for q_id, q_def in req.questions.items():
            q_type = q_def.type.lower()
            prompt = f"Context: {state_str}\n\nTask: {q_def.instructions}"

            if q_type == "choice":
                choice_keys = list(q_def.criteria.keys())
                choice_descs = [f"{k}: {q_def.criteria[k]}" for k in choice_keys]

                async with state["lock"]:
                    res = await asyncio.to_thread(eng.adjudicate, prompt, choice_descs)

                win_key = choice_keys[res.selected_index]
                c_probs = res.choice_probabilities or [1.0 / len(choice_keys)] * len(choice_keys)
                probs_dict = {choice_keys[idx]: round(c_probs[idx], 4) for idx in range(len(choice_keys))}
                answers[q_id] = {
                    "type": "choice",
                    "choice": win_key,
                    "confidence": round(res.confidence, 4),
                    "probabilities": probs_dict,
                }

            elif q_type in ("noul", "bool"):
                criteria = (
                    q_def.criteria
                    if q_def.criteria is not None
                    else {"true": "Condition is met / affirmative", "false": "Condition is not met / negative"}
                )
                cand_descs = [f"True: {criteria.get('true', 'True')}", f"False: {criteria.get('false', 'False')}"]

                async with state["lock"]:
                    res = await asyncio.to_thread(eng.adjudicate, prompt, cand_descs)

                c_probs = res.choice_probabilities or [0.5, 0.5]
                p_true = round(c_probs[0], 4)
                if q_type == "noul":
                    answers[q_id] = {"type": "noul", "noul": p_true}
                else:
                    answers[q_id] = {"type": "bool", "bool": bool(p_true >= 0.5), "probability": p_true}

            elif q_type == "score":
                levels = q_def.criteria
                cand_descs = [f"Level {idx}: {desc}" for idx, desc in enumerate(levels)]

                async with state["lock"]:
                    res = await asyncio.to_thread(eng.adjudicate, prompt, cand_descs)

                c_probs = res.choice_probabilities or [1.0 / len(levels)] * len(levels)
                expected_score = sum(idx * c_probs[idx] for idx in range(len(levels)))
                answers[q_id] = {
                    "type": "score",
                    "score": round(expected_score, 4),
                    "probabilities": [round(p, 4) for p in c_probs],
                    "legend": {str(i): levels[i] for i in range(len(levels))},
                }

        input_tokens = len(state_str.split()) * 2
        return SystemOneResponse(
            model=req.model or "system2",
            answers=answers,
            usage={"input_tokens": input_tokens, "output_tokens": 0, "total_tokens": input_tokens},
        )

    # -------------------------------------------------------------------------
    # Backward Compatibility: /v1/decisions (OpenAI Specification)
    # -------------------------------------------------------------------------
    @app.post("/v1/decisions", response_model=DecisionResponse)
    async def handle_decisions(req: DecisionRequest):
        # Validate all questions upfront so empty choices/levels or unknown types
        # return 400 cleanly instead of triggering ValueError/ZeroDivisionError.
        for q in req.questions:
            q_type = q.type.lower()
            if q_type == "predicate":
                continue
            if q_type == "choice":
                if not q.choices:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Question '{q.name}' of type 'choice' requires a non-empty choices list",
                    )
            elif q_type == "score":
                if not q.levels:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Question '{q.name}' of type 'score' requires a non-empty levels list",
                    )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported question type: '{q_type}'")

        async with state["lock"]:
            eng = await asyncio.to_thread(get_engine)
        inp_str = json.dumps(req.input) if not isinstance(req.input, str) else req.input
        answers: List[Dict[str, Any]] = []

        domain_override = None
        if req.model and ":" in req.model:
            domain_override = req.model.split(":", 1)[1]

        for q in req.questions:
            q_type = q.type.lower()
            prompt = f"Context: {inp_str}\n\nDecision Query: {q.instructions}"

            if q_type == "predicate":
                cand_descs = ["Condition is satisfied / True", "Condition is violated / False"]
                async with state["lock"]:
                    res = await asyncio.to_thread(eng.adjudicate, prompt, cand_descs, domain=domain_override)
                c_probs = res.choice_probabilities or [0.5, 0.5]
                p_true = round(c_probs[0], 4)
                answers.append({
                    "name": q.name,
                    "type": "predicate",
                    "predicate": bool(p_true >= 0.5),
                    "probability": p_true,
                })

            elif q_type == "choice":
                choices_list = q.choices or []
                values = [c.get("value", f"opt_{i}") for i, c in enumerate(choices_list)]
                descs = [f"{c.get('value')}: {c.get('description', '')}" for c in choices_list]
                async with state["lock"]:
                    res = await asyncio.to_thread(eng.adjudicate, prompt, descs, domain=domain_override)
                win_val = values[res.selected_index]
                answers.append({
                    "name": q.name,
                    "type": "choice",
                    "choice": win_val,
                    "probability": round(res.confidence, 4),
                })

            elif q_type == "score":
                levels = q.levels or []
                descs = [lvl.get("label", str(i)) for i, lvl in enumerate(levels)]
                async with state["lock"]:
                    res = await asyncio.to_thread(eng.adjudicate, prompt, descs, domain=domain_override)
                c_probs = res.choice_probabilities or [1.0 / len(levels)] * len(levels)
                expected = sum(i * c_probs[i] for i in range(len(levels)))
                answers.append({
                    "name": q.name,
                    "type": "score",
                    "score": round(expected, 4),
                })

        prompt_tokens = len(inp_str.split()) * 2
        return DecisionResponse(
            model=req.model or "system2",
            answers=answers,
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens},
        )

    # -------------------------------------------------------------------------
    # Upstream Reverse Proxy Fallback
    # -------------------------------------------------------------------------
    if upstream_llm:
        @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
        async def reverse_proxy(request: Request, path: str):
            http_client = get_http_client()
            url = f"{upstream_llm.rstrip('/')}/{path}"
            headers = dict(request.headers)
            headers.pop("host", None)
            body = await request.body()

            req = http_client.build_request(
                method=request.method,
                url=url,
                headers=headers,
                params=request.query_params,
                content=body,
            )

            try:
                resp = await http_client.send(req, stream=True)
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to connect to upstream LLM at {upstream_llm}: {exc}",
                )

            return StreamingResponse(
                resp.aiter_raw(),
                status_code=resp.status_code,
                headers=dict(resp.headers),
                # Starlette awaits BackgroundTask(resp.aclose) after streaming finishes.
                background=BackgroundTask(resp.aclose),
            )

    return app


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="System2 High-Speed Decision Server & Dynamic Domain Gateway",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('mps', 'cuda', 'cpu')")
    parser.add_argument(
        "--upstream-llm",
        type=str,
        default=None,
        help="Optional upstream generative LLM URL to proxy chat/completions to",
    )
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    parser.add_argument("--log-level", type=str, default="info", help="Uvicorn log level")

    args = parser.parse_args()

    app = create_app(
        device=args.device,
        upstream_llm=args.upstream_llm,
    )

    print("=" * 72)
    print(" SYSTEM 2 PARALLEL DELIBERATION & AGENT GATEKEEPER SERVER")
    print(f" Listening on:      http://{args.host}:{args.port}")
    print(f" Deliberation API:  POST http://{args.host}:{args.port}/deliberate")
    print(f" Domain Routing:    POST http://{args.host}:{args.port}/route")
    print(f" Safety Gatekeeper: POST http://{args.host}:{args.port}/guard")
    print(f" Health & Status:   GET  http://{args.host}:{args.port}/health")
    print(f" Track Inventory:   GET  http://{args.host}:{args.port}/tracks")
    if args.upstream_llm:
        print(f" Reverse Proxy:     Forwarding unhandled routes to {args.upstream_llm}")
    print("=" * 72)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
