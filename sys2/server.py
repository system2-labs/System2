"""server.py - System2 Non-Autoregressive Deliberation & Decision Server.

Provides high-performance asynchronous REST endpoints:
  1. POST /deliberate : Full 5-head parallel feedforward deliberation (< 30ms on MPS, < 200ms on CPU)
  2. POST /route      : Ultra-fast two-stage domain routing (< 0.1ms lexical, < 1ms centroid)
  3. POST /guard      : Non-autoregressive agent command safety gatekeeper
  4. GET  /health     : Server liveness, memory status, and track inventory
  5. GET  /tracks     : Metadata inventory across all 11 specialized tracks
  6. POST /v1/systemone : Jev / TypeSafe AI decision endpoint compatibility
  7. POST /v1/decisions : OpenAI-compatible structured decision endpoint compatibility
  8. Reverse-proxy    : Transparently proxies generative LLM routes (/v1/chat/completions) if upstream-llm configured

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
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import torch
import uvicorn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from routing.domain_router import DomainRouter
from sys2.engine import System2Engine, CHECKPOINT_MAP
from sys2.guard import AgentGatekeeper, GuardVerdict


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
        if upstream_llm:
            state["http_client"] = httpx.AsyncClient(base_url=upstream_llm, timeout=120.0)
        yield
        if state["http_client"] is not None:
            await state["http_client"].aclose()

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

    # -------------------------------------------------------------------------
    # Lazy Loader Helpers
    # -------------------------------------------------------------------------
    def get_engine() -> System2Engine:
        if state["engine"] is None:
            state["engine"] = System2Engine(device=state["device"])
        return state["engine"]

    def get_router() -> DomainRouter:
        if state["router"] is None:
            state["router"] = DomainRouter(device="cpu")
        return state["router"]

    def get_gatekeeper(
        allow_th: float = 0.70,
        block_th: float = 0.70,
        abstain_th: float = 0.40,
    ) -> AgentGatekeeper:
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

        eng = get_engine()
        async with state["lock"]:
            res = await asyncio.to_thread(
                eng.adjudicate,
                req.question,
                choices,
                domain=req.domain,
                max_len=req.max_len or 384,
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
        rtr = get_router()
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

    @app.post("/guard", response_model=GuardResponse)
    async def guard_command(req: GuardRequest):
        gk = get_gatekeeper(
            allow_th=req.allow_threshold or 0.70,
            block_th=req.block_threshold or 0.70,
            abstain_th=req.abstain_threshold or 0.40,
        )

        async with state["lock"]:
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

    # -------------------------------------------------------------------------
    # Backward Compatibility: /v1/systemone (Jev Specification)
    # -------------------------------------------------------------------------
    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def handle_systemone(req: SystemOneRequest):
        eng = get_engine()
        state_str = json.dumps(req.state) if not isinstance(req.state, str) else req.state
        answers: Dict[str, Any] = {}

        for q_id, q_def in req.questions.items():
            q_type = q_def.type.lower()
            prompt = f"Context: {state_str}\n\nTask: {q_def.instructions}"

            if q_type == "choice":
                if not isinstance(q_def.criteria, dict):
                    raise HTTPException(status_code=400, detail=f"Criteria for '{q_id}' must be dict mapping key -> description")
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
                criteria = q_def.criteria or {"true": "Condition is met / affirmative", "false": "Condition is not met / negative"}
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
                if not isinstance(q_def.criteria, list):
                    raise HTTPException(status_code=400, detail=f"Criteria for '{q_id}' must be a list of level descriptions")
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
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported question type: '{q_type}'")

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
        eng = get_engine()
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
                descs = [l.get("label", str(i)) for i, l in enumerate(levels)]
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
            http_client: httpx.AsyncClient = state["http_client"]
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
                background=resp.aclose,
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
