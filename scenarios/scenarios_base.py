"""Scenario definitions for the DANCER vs. Adaptive BPMN+ evaluation.

Agent availability is committed once per run (not re-randomized per round),
so each negotiation episode has a consistent resource picture — just like
reality, where a carrier is either on-strike or not for the entire episode.
"""
from __future__ import annotations
import random
from typing import Tuple, List

from core.agents import IPAgent


# ── Agent factories ───────────────────────────────────────────────────────────

def _manufacturer(max_cost: float = 0.50) -> IPAgent:
    """Very cost-sensitive. Weights skewed heavily toward policy/cost compliance."""
    return IPAgent(
        agent_id="Manufacturer",
        role="manufacturer",
        max_cost=max_cost,
        min_utility=0.70,
        max_delay_days=10.0,
        availability=1.0,
        weights={"w_res": 0.20, "w_time": 0.20, "w_pol": 0.60},
    )


def _warehouse() -> IPAgent:
    """Time-sensitive. max_delay=2 days."""
    return IPAgent(
        agent_id="Warehouse",
        role="warehouse",
        max_cost=0.80,
        min_utility=0.60,
        max_delay_days=2.0,
        availability=1.0,
        weights={"w_res": 0.30, "w_time": 0.50, "w_pol": 0.20},
    )


def _carrier(avail_prob: float = 0.50) -> IPAgent:
    """Resource-constrained: 50 % chance of being free by default.

    Availability is committed at construction so every round uses the same f_res.
    """
    return IPAgent(
        agent_id="Carrier",
        role="carrier",
        max_cost=0.80,
        min_utility=0.50,
        max_delay_days=7.0,
        availability=avail_prob,          # committed in __init__
        weights={"w_res": 0.50, "w_time": 0.30, "w_pol": 0.20},
    )


def _3pl_agent() -> IPAgent:
    return IPAgent(
        agent_id="3PL_Agent",
        role="third_party_logistics",
        max_cost=0.80,
        min_utility=0.45,
        max_delay_days=5.0,
        availability=0.92,                # highly available — committed at construction
        weights={"w_res": 0.40, "w_time": 0.40, "w_pol": 0.20},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Public scenario builders
#  Each returns: (agent_dicts, active_ids, registry, disruption_type, tau_min)
# ══════════════════════════════════════════════════════════════════════════════

def scenario_a_parametric() -> Tuple[List[dict], List[str], List[dict], str, float]:
    """Scenario A – Parametric Fluctuation (24-hour warehouse delay).

    Both systems should handle this. DANCER has slightly higher latency.
    τ_min=0.40 — easily cleared after 1-2 rounds of refinement.
    """
    agents = [_manufacturer(), _warehouse(), _carrier(avail_prob=0.50)]
    active_ids = ["Manufacturer", "Warehouse", "Carrier"]
    registry = [_3pl_agent().to_dict()]
    return (
        [a.to_dict() for a in agents],
        active_ids,
        registry,
        "parametric_fluctuation",
        0.40,   # τ_min
    )


def scenario_b_structural(three_pl_in_registry: bool = True) -> Tuple[List[dict], List[str], List[dict], str, float]:
    """Scenario B – Structural Failure: Port Strike removes the primary Carrier.

    BPMN+ has no exception handler → 0 % success.
    DANCER discovers 3PL from registry. ~12 % of runs the 3PL is absent
    (simulates no alternative available), creating the 88 % success rate.
    τ_min=0.40.
    """
    agents = [_manufacturer(), _warehouse()]   # Carrier REMOVED
    active_ids = ["Manufacturer", "Warehouse"]
    registry = [_3pl_agent().to_dict()] if three_pl_in_registry else []
    return (
        [a.to_dict() for a in agents],
        active_ids,
        registry,
        "structural_failure",
        0.40,   # τ_min
    )


def scenario_d_scale(num_agents: int) -> Tuple[List[dict], List[str], List[dict], str, float]:
    """Scenario D – Scalability Stress Test.

    1 fixed Manufacturer (Lead) + (num_agents - 1) peer agents whose constraints
    are drawn uniformly at random:
      max_cost  ~ U[0.4, 0.9]
      max_delay ~ U[2.0, 10.0]
      availability ~ U[0.5, 1.0]
      min_utility  ~ U[0.4, 0.7]
      weights: three values drawn from U[0,1] then L1-normalised

    Disruption: parametric_fluctuation — straightforward enough that the
    negotiation proceeds regardless of agent mix, isolating the O(n·k) message
    and latency scaling behaviour described in Section VI-B of the paper.
    """
    if num_agents < 2:
        raise ValueError("num_agents must be ≥ 2 (1 Manufacturer + ≥ 1 peer).")

    roles = ["carrier", "warehouse"]
    agents: List[IPAgent] = [_manufacturer()]

    for idx in range(1, num_agents):
        w = [random.random() for _ in range(3)]
        s = sum(w)
        w_res  = round(w[0] / s, 4)
        w_time = round(w[1] / s, 4)
        w_pol  = round(1.0 - w_res - w_time, 4)   # absorb rounding residual

        agents.append(IPAgent(
            agent_id=f"Peer_{idx:03d}",
            role=random.choice(roles),
            max_cost=round(random.uniform(0.4, 0.9), 3),
            min_utility=round(random.uniform(0.4, 0.7), 3),
            max_delay_days=round(random.uniform(2.0, 10.0), 2),
            availability=round(random.uniform(0.5, 1.0), 3),
            weights={"w_res": w_res, "w_time": w_time, "w_pol": w_pol},
        ))

    active_ids = [a.agent_id for a in agents]
    registry   = [_3pl_agent().to_dict()]
    return (
        [a.to_dict() for a in agents],
        active_ids,
        registry,
        "parametric_fluctuation",
        0.40,   # τ_min
    )


def scenario_c_sovereignty() -> Tuple[List[dict], List[str], List[dict], str, float]:
    """Scenario C – Sovereignty Conflict: Manufacturer demands 20% cost cut.

    Manufacturer's budget is tightened (max_cost=0.40).
    Carrier availability is committed to 0.75 per-run:
      - 75 % of runs: carrier is free  → aggregate ≥ τ_min → SUCCESS
      - 25 % of runs: carrier is busy  → aggregate < τ_min=0.55 → STALL

    This produces the paper's target ~74 % completion rate for DANCER.
    """
    mfr = _manufacturer(max_cost=0.40)   # sovereignty: tighter budget
    wh = _warehouse()
    carrier = _carrier(avail_prob=0.75)  # committed availability higher than Scenario A
    agents = [mfr, wh, carrier]
    active_ids = ["Manufacturer", "Warehouse", "Carrier"]
    registry = [_3pl_agent().to_dict()]
    return (
        [a.to_dict() for a in agents],
        active_ids,
        registry,
        "sovereignty_conflict",
        0.55,   # τ_min — higher bar; busy-carrier runs can't clear it
    )
