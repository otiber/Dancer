"""Adaptive BPMN+ baseline — closed-world assumption system.

AUDIT F3/F9 REWRITE: the original implementation returned a hardcoded
utility (0.870) with a hardcoded 5% random failure — no proposal was ever
evaluated against the episode's agent constraints, so its 'constant utility'
in the paper was a literal, not a measurement.

This version models what an imperative engine actually does: it owns ONE
pre-modeled exception handler per disruption class it was designed for
(parametric only — a boundary-event reroute fixed at design time), commits
that handler's pre-configured proposal, and the outcome is judged under the
CANONICAL criterion shared by every system (core/evaluation.py):
mean utility over the episode's agent set >= tau_min.

Structural failures and sovereignty conflicts have no exception sub-process →
the engine stalls (this is the closed-world limitation under test).
"""
from __future__ import annotations

import time
from typing import List

from core.models import Proposal, ShippingMethod, RunResult
from core.evaluation import canonical_outcome

def _prov(proposal, outcome):
    """Provenance kwargs for RunResult (audit F10)."""
    return dict(
        proposal_method=proposal.method.value if proposal else None,
        proposal_cost=proposal.cost_normalized if proposal else None,
        proposal_time=proposal.time_days if proposal else None,
        proposal_new_agent=proposal.new_agent_id if proposal else None,
        n_accepting=outcome.n_individually_accepting if outcome else None,
    )

# The single pre-modeled boundary-event handler (fixed at design time).
# It cannot adapt its parameters to the episode — that is the point.
_PARAMETRIC_HANDLER = Proposal(
    id="bpmn_boundary_1",
    method=ShippingMethod.EXPEDITED_GROUND,
    cost_normalized=0.45,
    time_days=2.0,
    description="BPMN+ pre-modeled boundary-event reroute (design-time fixed)",
    requires_new_agent=False,
    new_agent_id=None,
)

_KNOWN_HANDLERS = {"parametric_fluctuation"}


def run_bpmn_plus(
    scenario: str,
    agent_dicts: List[dict],
    active_agent_ids: List[str],
    registry: List[dict],
    disruption_type: str,
    iteration: int = 0,
    tau_min: float = 0.45,
) -> RunResult:
    start = time.time()

    if disruption_type not in _KNOWN_HANDLERS:
        # No exception sub-process exists → process engine STALLS
        return RunResult(
            scenario=scenario, system="BPMN+", iteration=iteration,
            success=False, rounds=0, messages_sent=0, total_tokens=0,
            final_utility=0.0, latency_seconds=time.time() - start,
        )

    outcome = canonical_outcome(_PARAMETRIC_HANDLER, agent_dicts, registry,
                                tau_min, active_agent_ids=active_agent_ids,
                                disruption_type=disruption_type)
    return RunResult(
        scenario=scenario, system="BPMN+", iteration=iteration,
        success=outcome.success, rounds=1, messages_sent=2, total_tokens=0,
        final_utility=outcome.aggregate_utility,
        latency_seconds=time.time() - start,
        **_prov(_PARAMETRIC_HANDLER, outcome),
    )
