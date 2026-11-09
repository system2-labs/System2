"""Unit tests for System 1 Fast Perceptual Screening & Routing."""

import pytest
from sys1 import (
    System1Engine,
    PerceptionResult,
    DomainRouter,
    AgentGatekeeper,
    GuardVerdict,
    GuardResult,
)


def test_sys1_exports():
    assert System1Engine is not None
    assert PerceptionResult is not None
    assert DomainRouter is not None
    assert AgentGatekeeper is not None
    assert GuardVerdict is not None
    assert GuardResult is not None


def test_domain_router_sys1():
    router = DomainRouter(device="cpu")
    res = router.route("A contract requires offer, acceptance, and consideration.")
    assert res.selected_track == "legal"
    assert res.latency_ms > 0
