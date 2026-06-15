"""
Extended scenario definitions for IEEE TSC large-scale evaluation.

Scenarios A–C replicate the original paper (imported from scenarios_base.py).
New contributions:

  E. Cascading Disruptions  — Sequential structural + sovereignty failures
     within a single process instance. Tests protocol resilience under
     compounding open-world failures.

  F. Adversarial Agents (Byzantine)  — One agent in the network deliberately
     misreports utility scores. Tests robustness under strategic manipulation.
     Two Byzantine modes: SABOTEUR (always reject) | FREERIDER (always accept).

  G. Multi-Constraint Conflict  — Simultaneous tight constraints across all
     three utility dimensions (f_res, f_time, f_pol), creating a near-infeasible
     negotiation space. Tests convergence under extreme constraint density.

Each builder returns: (agent_dicts, active_ids, registry, disruption_type, tau_min)
"""
from __future__ import annotations

import random
from typing import Tuple, List

from core.agents import IPAgent

# Re-export base scenarios for unified import
from scenarios.scenarios_base import (
    scenario_a_parametric,
    scenario_b_structural,
    scenario_c_sovereignty,
    scenario_d_scale,
)


# ── Byzantine Agent ───────────────────────────────────────────────────────────

class ByzantineAgent(IPAgent):
    """An agent that deliberately misreports its utility score.

    byzantine_mode:
      "saboteur"  — always returns utility=0.0, forcing rejection of all proposals.
      "freerider" — always returns utility=1.0, appearing to accept everything
                    (inflates consensus; may award infeasible proposals).
    """

    def __init__(self, *args, byzantine_mode: str = "saboteur", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.byzantine_mode = byzantine_mode
        self._real_evaluate = super().evaluate_proposal

    def evaluate_proposal(self, proposal):  # type: ignore[override]
        from core.models import Bid, UtilityVector
        real_bid = self._real_evaluate(proposal)
        if self.byzantine_mode == "saboteur":
            return Bid(
                agent_id=self.agent_id,
                proposal_id=real_bid.proposal_id,
                utility_vector=UtilityVector(f_res=0.0, f_time=0.0, f_pol=0.0),
                utility_score=0.0,
                accepted=False,
                rejection_reason=f"[BYZANTINE SABOTEUR] Fabricated rejection.",
            )
        elif self.byzantine_mode == "freerider":
            return Bid(
                agent_id=self.agent_id,
                proposal_id=real_bid.proposal_id,
                utility_vector=UtilityVector(f_res=1.0, f_time=1.0, f_pol=1.0),
                utility_score=1.0,
                accepted=True,
                rejection_reason=None,
            )
        return real_bid

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["byzantine_mode"] = self.byzantine_mode
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ByzantineAgent":
        mode = d.pop("byzantine_mode", "saboteur")
        agent = super().from_dict(d)
        agent.byzantine_mode = mode
        agent.__class__ = cls
        agent._real_evaluate = IPAgent.evaluate_proposal.__get__(agent, type(agent))
        return agent


# ── Scenario E: Cascading Disruptions ────────────────────────────────────────

def scenario_e_cascading() -> Tuple[List[dict], List[str], List[dict], str, float]:
    """Scenario E — Sequential structural failure followed by sovereignty conflict.

    Two disruptions occur within one process instance:
      Phase 1: Primary carrier removed (structural_failure).
      Phase 2: After structural recovery, Manufacturer tightens budget (sovereignty).

    The compound disruption_type "cascading" triggers both failure modes.
    Systems that can only handle one disruption type will stall at phase 2.
    DANCER must negotiate two sequential re-choreographies.
    """
    # Tight Manufacturer: very sensitive to cost (max_cost=0.40)
    mfr = IPAgent(
        agent_id="Manufacturer",
        role="manufacturer",
        max_cost=0.40,          # sovereignty constraint active from start
        min_utility=0.70,
        max_delay_days=8.0,
        availability=1.0,
        weights={"w_res": 0.15, "w_time": 0.25, "w_pol": 0.60},
    )
    # Warehouse: fast turnaround required
    wh = IPAgent(
        agent_id="Warehouse",
        role="warehouse",
        max_cost=0.80,
        min_utility=0.55,
        max_delay_days=1.5,     # tight time constraint
        availability=1.0,
        weights={"w_res": 0.25, "w_time": 0.55, "w_pol": 0.20},
    )
    # No carrier (structural failure)
    agents = [mfr, wh]
    active_ids = ["Manufacturer", "Warehouse"]
    registry = [IPAgent(
        agent_id="3PL_Agent",
        role="third_party_logistics",
        max_cost=0.80,
        min_utility=0.45,
        max_delay_days=5.0,
        availability=0.88,
        weights={"w_res": 0.40, "w_time": 0.40, "w_pol": 0.20},
    ).to_dict()]

    return (
        [a.to_dict() for a in agents],
        active_ids,
        registry,
        "cascading",        # combined disruption type
        0.45,               # τ_min — tight; both constraints must be met
    )


# ── Scenario F: Byzantine (Adversarial) Agents ───────────────────────────────

def scenario_f_byzantine(
    byzantine_mode: str = "saboteur",
    byzantine_fraction: float = 0.33,
) -> Tuple[List[dict], List[str], List[dict], str, float]:
    """Scenario F — One agent deliberately misreports utility scores.

    byzantine_fraction: fraction of peer agents that are Byzantine (≈ 1/3).
    byzantine_mode: "saboteur" | "freerider"

    Under saboteur mode:
      - The Byzantine agent always rejects, suppressing consensus.
      - Tests DANCER's temporal decay (can the remaining agents clear τ_min?).
      - Mirrors paper Section 8.1 threat model.

    Under freerider mode:
      - The Byzantine agent always accepts, inflating aggregate utility.
      - May admit infeasible proposals if the freerider's vote tips the average.
      - Tests whether DANCER's formal utility gate protects against false acceptance.
    """
    mfr = IPAgent(
        agent_id="Manufacturer",
        role="manufacturer",
        max_cost=0.50,
        min_utility=0.65,
        max_delay_days=10.0,
        availability=1.0,
        weights={"w_res": 0.20, "w_time": 0.20, "w_pol": 0.60},
    )
    wh = IPAgent(
        agent_id="Warehouse",
        role="warehouse",
        max_cost=0.80,
        min_utility=0.55,
        max_delay_days=2.0,
        availability=1.0,
        weights={"w_res": 0.30, "w_time": 0.50, "w_pol": 0.20},
    )

    # Byzantine carrier — misreports utility
    carrier_base = dict(
        agent_id="Carrier",
        role="carrier",
        max_cost=0.80,
        min_utility=0.50,
        max_delay_days=7.0,
        availability=0.80,
        weights={"w_res": 0.50, "w_time": 0.30, "w_pol": 0.20},
        is_active=True,
        committed_f_res=0.90,
    )
    carrier_dict = {**carrier_base, "byzantine_mode": byzantine_mode}

    agents_dicts = [mfr.to_dict(), wh.to_dict(), carrier_dict]
    active_ids = ["Manufacturer", "Warehouse", "Carrier"]
    registry = [IPAgent(
        agent_id="3PL_Agent",
        role="third_party_logistics",
        max_cost=0.80,
        min_utility=0.45,
        max_delay_days=5.0,
        availability=0.92,
        weights={"w_res": 0.40, "w_time": 0.40, "w_pol": 0.20},
    ).to_dict()]

    return (
        agents_dicts,
        active_ids,
        registry,
        "structural_failure",
        0.40,
    )


# ── Scenario G: Multi-Constraint Conflict ────────────────────────────────────

def scenario_g_multi_constraint() -> Tuple[List[dict], List[str], List[dict], str, float]:
    """Scenario G — All three utility dimensions simultaneously constrained.

    Each agent specializes in one dimension, making all three constraints binding:
      Manufacturer: policy constraint (max_cost=0.35 — very tight)
      Warehouse:    time constraint   (max_delay=1.0 day — very tight)
      Carrier:      resource constraint (availability=0.55 — resource-uncertain)

    The proposal space is severely restricted:
      A feasible proposal must be cheap (cost ≤ 0.35), fast (time ≤ 1.0 day),
      AND have an available carrier.
    Only Express_Air_Freight can potentially satisfy both cost (if discounted)
    and time, but cost 0.35 for air freight is at the boundary.

    This scenario tests whether DANCER's Contrastive Refinement can navigate
    a near-infeasible constraint space within R_max rounds.
    τ_min is set lower (0.35) to allow suboptimal but feasible solutions.
    """
    mfr = IPAgent(
        agent_id="Manufacturer",
        role="manufacturer",
        max_cost=0.35,          # extreme cost pressure
        min_utility=0.65,
        max_delay_days=12.0,    # relaxed on time
        availability=1.0,
        weights={"w_res": 0.10, "w_time": 0.10, "w_pol": 0.80},
    )
    wh = IPAgent(
        agent_id="Warehouse",
        role="warehouse",
        max_cost=0.90,          # relaxed on cost
        min_utility=0.55,
        max_delay_days=1.0,     # extreme time pressure
        availability=1.0,
        weights={"w_res": 0.20, "w_time": 0.70, "w_pol": 0.10},
    )
    carrier = IPAgent(
        agent_id="Carrier",
        role="carrier",
        max_cost=0.90,
        min_utility=0.50,
        max_delay_days=10.0,
        availability=0.55,      # resource-constrained
        weights={"w_res": 0.80, "w_time": 0.10, "w_pol": 0.10},
        committed_f_res=random.uniform(0.50, 0.85),
    )
    agents = [mfr, wh, carrier]
    active_ids = ["Manufacturer", "Warehouse", "Carrier"]
    registry = [IPAgent(
        agent_id="3PL_Agent",
        role="third_party_logistics",
        max_cost=0.85,
        min_utility=0.40,
        max_delay_days=1.2,
        availability=0.90,
        weights={"w_res": 0.40, "w_time": 0.40, "w_pol": 0.20},
    ).to_dict()]

    return (
        [a.to_dict() for a in agents],
        active_ids,
        registry,
        "multi_constraint",
        0.35,           # τ_min — permissive to allow near-feasible solutions
    )


# ── Scenario registry ─────────────────────────────────────────────────────────

SCENARIO_REGISTRY = {
    "A": ("Parametric Fluctuation",   scenario_a_parametric,    "parametric_fluctuation"),
    "B": ("Structural Failure",        scenario_b_structural,    "structural_failure"),
    "C": ("Sovereignty Conflict",      scenario_c_sovereignty,   "sovereignty_conflict"),
    "E": ("Cascading Disruptions",     scenario_e_cascading,     "cascading"),
    "F": ("Byzantine Agents",          scenario_f_byzantine,     "structural_failure"),
    "G": ("Multi-Constraint Conflict", scenario_g_multi_constraint, "multi_constraint"),
}

SCALABILITY_SCENARIO = scenario_d_scale
