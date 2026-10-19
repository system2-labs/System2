"""server.py - System2 Decision Endpoint Server & LM Studio / Jev Adapter.

Provides drop-in compatibility with:
  1. POST /v1/systemone : Jev (TypeSafe AI) / LM Studio System One Decision Endpoint
  2. POST /v1/decisions : OpenAI-Compatible Structured Decision Endpoint
  3. GET  /v1/models    : Model inventory discovery
  4. GET  /health       : Liveness and status probe
  5. Reverse-proxy mode : Proxies generative LLM calls (/v1/chat/completions, etc.) to upstream LM Studio

Usage:
  # Standalone decision server on LM Studio default port:
  python3 -m sys2.server --port 1234

  # Reverse-proxy alongside LM Studio running on port 1235:
  python3 -m sys2.server --port 1234 --upstream-llm http://localhost:1235
"""

import argparse
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
import uvicorn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sys2.engine import System2Engine, CHECKPOINT_MAP
from sys2.guard import AgentGatekeeper, GuardVerdict


# -----------------------------------------------------------------------------
# Data Models: /v1/systemone (Jev / TypeSafe AI Specification)
# -----------------------------------------------------------------------------

class SystemOneQuestion(BaseModel):
    type: str = Field(..., description="'choice', 'noul', 'bool', or 'score'")
    instructions: Optional[str] = Field(default="", description="Instruction or question text")
    criteria: Optional[Union[Dict[str, Any], List[Any], Any]] = Field(
        default=None, description="Criteria mapping (dict for choice/noul, list for score)"
    )


class SystemOneRequest(BaseModel):
    model: Optional[str] = Field(default="system2", description="Decision model identifier")
    state: Union[str, Dict[str, Any], List[Any]] = Field(
        ..., description="Application context or state evaluated against questions"
    )
    questions: Dict[str, SystemOneQuestion] = Field(
        ..., description="Map of question IDs to question definitions"
    )


class SystemOneUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: Dict[str, Any]
    usage: SystemOneUsage


# -----------------------------------------------------------------------------
# Data Models: /v1/decisions (OpenAI Decision Specification)
# -----------------------------------------------------------------------------

class DecisionQuestion(BaseModel):
    type: Literal["predicate", "choice", "score"]
    name: Optional[str] = Field(default=None, description="Identifier for question in response")
    instructions: str = Field(..., description="Prompt rule or question to evaluate")
    choices: Optional[List[Union[Dict[str, Any], str]]] = Field(
        default=None, description="Choices for choice questions"
    )
    levels: Optional[List[Union[Dict[str, Any], str]]] = Field(
        default=None, description="Ordered levels for score questions"
    )


class DecisionRequest(BaseModel):
    model: Optional[str] = Field(default="system2", description="Decision model identifier")
    input: Union[str, List[Any], Dict[str, Any]] = Field(
        ..., description="Evidence or context to evaluate"
    )
    questions: List[DecisionQuestion] = Field(
        ..., description="List of questions to evaluate"
    )


class DecisionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class DecisionResponse(BaseModel):
    model: str
    answers: List[Dict[str, Any]]
    usage: DecisionUsage


# -----------------------------------------------------------------------------
# Server Application Factory
# -----------------------------------------------------------------------------

def create_app(
    engine: Optional[System2Engine] = None,
    upstream_llm: Optional[str] = None,
    device: Optional[str] = None,
) -> FastAPI:
    """Build and configure the System2 FastAPI application."""

    # Shared singletons resident in memory
    app_engine: System2Engine = engine or System2Engine(device=device)
    app_guard = AgentGatekeeper(engine=app_engine, device=app_engine.device)
    http_client: Optional[httpx.AsyncClient] = httpx.AsyncClient(timeout=120.0) if upstream_llm else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if http_client:
            await http_client.aclose()

    app = FastAPI(
        title="System2 Decision Server",
        description="High-Speed Non-Autoregressive Deliberation Engine & LM Studio / Jev Adapter",
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
    # Helper: Normalize input context & domain extraction
    # -------------------------------------------------------------------------
    def stringify_context(raw: Union[str, Dict[str, Any], List[Any]]) -> str:
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, list):
            parts = []
            for item in raw:
                if isinstance(item, dict) and "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        return json.dumps(raw, ensure_ascii=False)

    def resolve_domain(model_str: Optional[str]) -> Optional[str]:
        if not model_str:
            return None
        m = model_str.lower().strip()
        if m in ("system2-guard", "sys2-guard", "guard", "safety"):
            return "safety"
        if ":" in m:
            _, sub = m.split(":", 1)
            if sub in CHECKPOINT_MAP:
                return sub
        if m in CHECKPOINT_MAP:
            return m
        return None

    def estimate_input_tokens(text: str) -> int:
        try:
            return len(app_engine.tokenizer.encode(text, truncation=True, max_length=1024))
        except Exception:
            return max(1, len(text.split()))

    # -------------------------------------------------------------------------
    # Endpoint: Liveness & Health
    # -------------------------------------------------------------------------
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "system": "System2 Parallel Hypothesis Deliberation Engine",
            "version": "0.2.0",
            "device": app_engine.device,
            "resident_domain": app_engine.current_domain,
            "available_domains": list(CHECKPOINT_MAP.keys()),
            "non_autoregressive": True,
            "upstream_llm": upstream_llm,
        }

    # -------------------------------------------------------------------------
    # Endpoint: Models Inventory
    # -------------------------------------------------------------------------
    @app.get("/v1/models")
    async def list_models():
        native_models = [
            {
                "id": "system2",
                "object": "model",
                "type": "decision",
                "owned_by": "system2",
                "permission": [],
                "description": "Unified 5-Head Parallel Deliberation Architecture with Dynamic Domain Routing",
            },
            {
                "id": "system2-guard",
                "object": "model",
                "type": "decision",
                "owned_by": "system2",
                "permission": [],
                "description": "Command and Operational Security Gatekeeper with Calibrated Abstention",
            },
        ]
        for dom in sorted(CHECKPOINT_MAP.keys()):
            native_models.append({
                "id": f"system2:{dom}",
                "object": "model",
                "type": "decision",
                "owned_by": "system2",
                "permission": [],
                "description": f"System2 Deliberation Adapter specialized for [{dom.upper()}] domain",
            })

        # If proxy mode is active, fetch and append upstream LLM models
        if http_client and upstream_llm:
            try:
                resp = await http_client.get(f"{upstream_llm.rstrip('/')}/v1/models")
                if resp.status_code == 200:
                    upstream_data = resp.json().get("data", [])
                    native_models.extend(upstream_data)
            except Exception:
                pass

        return {"object": "list", "data": native_models}

    # -------------------------------------------------------------------------
    # Endpoint: /v1/systemone (Jev / TypeSafe AI Specification)
    # -------------------------------------------------------------------------
    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def handle_systemone(payload: SystemOneRequest):
        context_str = stringify_context(payload.state)
        domain = resolve_domain(payload.model)
        total_input_tokens = estimate_input_tokens(context_str)
        answers: Dict[str, Any] = {}

        for q_id, q in payload.questions.items():
            q_type = q.type.lower().strip()
            instructions = (q.instructions or "").strip()
            prompt = f"{context_str}\n\nQuestion: {instructions}" if instructions else context_str

            if q_type == "choice":
                # Choices from criteria
                crit = q.criteria or {}
                if isinstance(crit, dict):
                    keys = list(crit.keys())
                    choice_texts = [f"{k}: {v}" if v else str(k) for k, v in crit.items()]
                elif isinstance(crit, list):
                    keys = [str(i) for i in range(len(crit))]
                    choice_texts = [str(x) for x in crit]
                else:
                    keys = ["A", "B"]
                    choice_texts = ["Yes", "No"]

                res = app_engine.adjudicate(prompt, choice_texts, domain=domain)
                win_key = keys[res.selected_index] if 0 <= res.selected_index < len(keys) else keys[0]
                probs = res.choice_probabilities or [0.5, 0.5]

                prob_dict = {
                    k: (probs[i] if i < len(probs) else 0.0)
                    for i, k in enumerate(keys)
                }

                answers[q_id] = {
                    "type": "choice",
                    "choice": win_key,
                    "confidence": res.confidence,
                    "probabilities": prob_dict,
                }

            elif q_type in ("noul", "bool"):
                # Binary decision
                crit = q.criteria if isinstance(q.criteria, dict) else {}
                true_label = str(crit.get("true", "Yes, " + (instructions or "true condition")))
                false_label = str(crit.get("false", "No, " + (instructions or "false condition")))
                choices = [true_label, false_label]

                res = app_engine.adjudicate(prompt, choices, domain=domain)
                probs = res.choice_probabilities or [0.5, 0.5]
                prob_true = probs[0] if len(probs) > 0 else 0.5

                if q_type == "bool":
                    answers[q_id] = {
                        "type": "bool",
                        "bool": bool(prob_true >= 0.5),
                        "probability": round(prob_true, 4),
                        "confidence": res.confidence,
                    }
                else:
                    answers[q_id] = {
                        "type": "noul",
                        "noul": round(prob_true, 4),
                        "confidence": res.confidence,
                    }

            elif q_type == "score":
                # Ordinal rating across criteria levels
                levels = q.criteria if isinstance(q.criteria, list) else ["Low", "Medium", "High"]
                choice_texts = [str(lvl) for lvl in levels]
                res = app_engine.adjudicate(prompt, choice_texts, domain=domain)
                probs = res.choice_probabilities or [1.0 / len(levels)] * len(levels)

                expected_score = sum(idx * p for idx, p in enumerate(probs))
                answers[q_id] = {
                    "type": "score",
                    "score": round(expected_score, 4),
                    "confidence": res.confidence,
                    "probabilities": [round(p, 4) for p in probs],
                    "legend": {str(idx): str(lvl) for idx, lvl in enumerate(levels)},
                }

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported question type '{q.type}'. Supported types: 'choice', 'noul', 'bool', 'score'",
                )

        return SystemOneResponse(
            model=payload.model or "system2",
            answers=answers,
            usage=SystemOneUsage(input_tokens=total_input_tokens, output_tokens=0),
        )

    # -------------------------------------------------------------------------
    # Endpoint: /v1/decisions (OpenAI Decisions Specification)
    # -------------------------------------------------------------------------
    @app.post("/v1/decisions", response_model=DecisionResponse)
    async def handle_decisions(payload: DecisionRequest):
        context_str = stringify_context(payload.input)
        domain = resolve_domain(payload.model)
        total_input_tokens = estimate_input_tokens(context_str)
        answers: List[Dict[str, Any]] = []

        for q in payload.questions:
            q_type = q.type.lower().strip()
            instructions = q.instructions.strip()
            prompt = f"{context_str}\n\nQuestion: {instructions}" if instructions else context_str
            ans_record: Dict[str, Any] = {"name": q.name, "type": q_type}

            if q_type == "predicate":
                choices = [
                    f"TRUE: {instructions}",
                    f"FALSE: Not {instructions}",
                ]
                res = app_engine.adjudicate(prompt, choices, domain=domain)
                probs = res.choice_probabilities or [0.5, 0.5]
                prob_true = probs[0] if len(probs) > 0 else 0.5
                ans_record["probability"] = round(prob_true, 4)

            elif q_type == "choice":
                raw_choices = q.choices or ["Yes", "No"]
                keys = []
                choice_texts = []
                for idx, c in enumerate(raw_choices):
                    if isinstance(c, dict):
                        val = str(c.get("value", idx))
                        desc = str(c.get("description", ""))
                        keys.append(val)
                        choice_texts.append(f"{val}: {desc}" if desc else val)
                    else:
                        keys.append(str(c))
                        choice_texts.append(str(c))

                res = app_engine.adjudicate(prompt, choice_texts, domain=domain)
                win_key = keys[res.selected_index] if 0 <= res.selected_index < len(keys) else keys[0]
                probs = res.choice_probabilities or [0.5, 0.5]
                win_prob = probs[res.selected_index] if res.selected_index < len(probs) else 0.5

                ans_record["choice"] = win_key
                ans_record["probability"] = round(win_prob, 4)

            elif q_type == "score":
                raw_levels = q.levels or ["Low", "Medium", "High"]
                choice_texts = []
                for lvl in raw_levels:
                    if isinstance(lvl, dict):
                        label = str(lvl.get("label", ""))
                        desc = str(lvl.get("description", ""))
                        choice_texts.append(f"{label}: {desc}" if desc else label)
                    else:
                        choice_texts.append(str(lvl))

                res = app_engine.adjudicate(prompt, choice_texts, domain=domain)
                probs = res.choice_probabilities or [1.0 / len(choice_texts)] * len(choice_texts)
                score = sum(idx * p for idx, p in enumerate(probs))
                ans_record["score"] = round(score, 4)

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported question type '{q.type}'. Supported: 'predicate', 'choice', 'score'",
                )

            answers.append(ans_record)

        return DecisionResponse(
            model=payload.model or "system2",
            answers=answers,
            usage=DecisionUsage(
                prompt_tokens=total_input_tokens,
                completion_tokens=0,
                total_tokens=total_input_tokens,
            ),
        )

    # -------------------------------------------------------------------------
    # Upstream Reverse Proxy (Transparent forwarder for chat/completions)
    # -------------------------------------------------------------------------
    if upstream_llm and http_client:
        target_base = upstream_llm.rstrip("/")

        @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
        async def reverse_proxy(request: Request, full_path: str):
            url = f"{target_base}/{full_path}"
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
        description="System2 High-Speed Decision Server (LM Studio / Jev Compatible)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=1234, help="Port to listen on (1234 matches LM Studio)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('mps', 'cuda', 'cpu')")
    parser.add_argument(
        "--upstream-llm",
        type=str,
        default=None,
        help="Optional upstream generative LLM URL to proxy chat/completions to (e.g. http://localhost:1235)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    parser.add_argument("--log-level", type=str, default="info", help="Uvicorn log level")

    args = parser.parse_args()

    app = create_app(
        device=args.device,
        upstream_llm=args.upstream_llm,
    )

    print("=" * 72)
    print(" SYSTEM 2 STRUCTURED DECISION SERVER")
    print(f" Listening on:   http://{args.host}:{args.port}")
    print(f" Jev Endpoint:   POST http://{args.host}:{args.port}/v1/systemone")
    print(f" Decisions API:  POST http://{args.host}:{args.port}/v1/decisions")
    print(f" Models API:     GET  http://{args.host}:{args.port}/v1/models")
    print(f" Health Check:   GET  http://{args.host}:{args.port}/health")
    if args.upstream_llm:
        print(f" Reverse Proxy:  Transparently forwarding unhandled routes to {args.upstream_llm}")
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
