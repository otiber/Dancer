"""Classical Contract Net Protocol (CNP) baseline — Smith (1980).

Implements the three-phase announce–bid–award protocol with role-based
eligibility filtering.  The announced task is FIXED — it encodes the process
step that was executing when the disruption hit.  The manager does NOT
reformulate the task; that capability is exclusive to DANCER.

Protocol phases
---------------
  Phase 1  TASK_ANNOUNCE (1 msg)
      Manager broadcasts a fixed Proposal + required_role.

  Phase 2  BID / REFUSAL  (len(contractors) msgs)
      Each contractor sends exactly one response:
        - role_mismatch  → immediate refusal (not eligible to bid)
        - role matches   → evaluate with L_k(p) from agents.py; bid or refuse
      Every response — bid or refusal — is one message on the wire.

  Phase 3  AWARD / STALL_NOTIFY  (1 msg)
      Manager validates the winning bid against its own utility threshold
      (Manufacturer's w_pol=0.60 means cost overruns trigger a veto).
      If no eligible contractor bid accepted, or manager veto fires → STALL.

Message count formula
---------------------
    messages_sent = 1  +  len(contractors)  +  1
where len(contractors) = len(active_agent_ids) - 1  (manager excluded).

Design constraints satisfied
-----------------------------
  - committed_f_res from agent_dicts is reused as-is — same draw as DANCER.
  - No LLM calls: fully deterministic given agent_dicts.
  - No task reformulation or 3PL discovery: those are DANCER-only capabilities.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import List, Tuple

from core.agents import IPAgent
from core.models import Proposal, ShippingMethod, RunResult


# ── Fixed announced tasks ─────────────────────────────────────────────────────
#
#  parametric_fluctuation
#      The in-flight step is warehouse processing.  A warehouse-role agent is
#      asked to absorb the 24-hour delay (expedited ground, cost=0.30, 1.5 days).
#      Warehouse availability=1.0 → almost always succeeds.
#
#  structural_failure
#      The in-flight carrier task is re-broadcast.  required_role="carrier".
#      The primary Carrier is absent (on strike) and CNP has no registry
#      discovery mechanism → all contractors refuse with role_mismatch.
#      Expected: 0 % completion.
#
#  sovereignty_conflict
#      Cost was set at 0.55 before the Manufacturer tightened its budget to 0.40.
#      required_role="carrier" — the Carrier may bid, but the Manager's cost-
#      compliance check (f_pol collapses at 0.55 > 0.40) triggers a veto.
#      Expected: 0 % completion.
#
_TASK_SCHEMA: dict[str, dict] = {
    "parametric_fluctuation": {
        "proposal": Proposal(
            id="cnp_task_A",
            method=ShippingMethod.EXPEDITED_GROUND,
            cost_normalized=0.30,
            time_days=1.5,
            description=(
                "Expedited Ground — warehouse processing step under 24-h delay"
            ),
        ),
        "required_role": "warehouse",
    },
    "structural_failure": {
        "proposal": Proposal(
            id="cnp_task_B",
            method=ShippingMethod.STANDARD,
            cost_normalized=0.35,
            time_days=4.0,
            description=(
                "Standard Shipping via primary Carrier (now on strike); "
                "CNP has no alternative-discovery mechanism"
            ),
        ),
        "required_role": "carrier",
    },
    "sovereignty_conflict": {
        "proposal": Proposal(
            id="cnp_task_C",
            method=ShippingMethod.STANDARD,
            cost_normalized=0.55,
            time_days=4.0,
            description=(
                "Standard Shipping — cost fixed at pre-sovereignty level (0.55); "
                "Manufacturer budget tightened to 0.40 post-announcement"
            ),
        ),
        "required_role": "carrier",
    },
}


def run_cnp(
    scenario: str,
    agent_dicts: List[dict],
    active_agent_ids: List[str],
    disruption_type: str,
    iteration: int = 0,
    tau_min: float = 0.45,
    required_role: str = None,
) -> Tuple[RunResult, Counter]:
    """Execute one CNP episode.

    Parameters
    ----------
    scenario         : label written into RunResult (e.g. "Scenario_B_Structural")
    agent_dicts      : serialised IPAgent dicts; first entry is the manager;
                       must carry committed_f_res so the same availability draw
                       is shared with BPMN+ and DANCER in the same episode
    active_agent_ids : ordered; agent_dicts[0] is the Lead Agent / manager
    disruption_type  : key into _TASK_SCHEMA
    iteration        : Monte-Carlo index

    Returns
    -------
    result   : RunResult  (system="CNP")
    refusals : Counter    refusal-reason → count for this episode
                          (all contractor and manager-veto reasons)
    """
    start = time.time()
    refusals: Counter = Counter()

    task_spec = _TASK_SCHEMA.get(disruption_type)
    if task_spec is None:
        # No task schema for this disruption — stall immediately
        return RunResult(
            scenario=scenario, system="CNP", iteration=iteration,
            success=False, rounds=1, messages_sent=1, total_tokens=0,
            final_utility=0.0, latency_seconds=time.time() - start,
        ), refusals

    proposal: Proposal = task_spec["proposal"]
    # AUDIT F11: the hardcoded synthetic roles ("warehouse"/"carrier") never
    # match BPIC spend-area roles, guaranteeing role_mismatch refusals. The
    # runner now grounds required_role in the episode's actual agent roles;
    # the hardcoded value remains the default for the synthetic scenarios.
    required_role = required_role or task_spec["required_role"]

    # Reconstruct agents — committed_f_res preserved, same as DANCER's _all_agents()
    active_set = set(active_agent_ids)
    agents: List[IPAgent] = [
        IPAgent.from_dict(d) for d in agent_dicts
        if d["agent_id"] in active_set
    ]
    if not agents:
        return RunResult(
            scenario=scenario, system="CNP", iteration=iteration,
            success=False, rounds=1, messages_sent=1, total_tokens=0,
            final_utility=0.0, latency_seconds=time.time() - start,
        ), refusals

    manager: IPAgent = agents[0]              # Lead Agent; does not bid
    contractors: List[IPAgent] = agents[1:]   # n_contractors = len(active) - 1

    # ── Phase 1: TASK_ANNOUNCE ────────────────────────────────────────────────
    msgs: int = 1

    # ── Phase 2: BID / REFUSAL ───────────────────────────────────────────────
    best_score: float = -1.0
    best_bid = None

    for agent in contractors:
        msgs += 1  # every response (bid or refusal) is one message

        if agent.role != required_role:
            refusals["role_mismatch"] += 1
            continue

        bid = agent.evaluate_proposal(proposal)
        if bid.accepted:
            if bid.utility_score > best_score:
                best_score = bid.utility_score
                best_bid = bid
        else:
            reason = bid.rejection_reason or ""
            if "Cost constraint" in reason:
                refusals["cost_constraint_violated"] += 1
            elif "Time constraint" in reason:
                refusals["time_constraint_violated"] += 1
            elif "Resource unavailable" in reason:
                refusals["resource_unavailable"] += 1
            else:
                refusals["utility_below_threshold"] += 1

    # ── Phase 3: AWARD or STALL_NOTIFY ───────────────────────────────────────
    msgs += 1  # award or stall-notice message

    if best_bid is None:
        # No eligible contractor produced an acceptable bid
        refusals["no_eligible_bidder"] += 1
        return RunResult(
            scenario=scenario, system="CNP", iteration=iteration,
            success=False, rounds=1, messages_sent=msgs, total_tokens=0,
            final_utility=0.0, latency_seconds=time.time() - start,
            refusals=";".join(f"{k}:{v}" for k, v in refusals.items()) or None,
        ), refusals

    # Manager validates winning bid: if manager's own L_k(p) < min_utility → veto
    manager_eval = manager.evaluate_proposal(proposal)
    if not manager_eval.accepted:
        reason = manager_eval.rejection_reason or ""
        if "Cost constraint" in reason:
            refusals["manager_veto:cost_constraint_violated"] += 1
        elif "Time constraint" in reason:
            refusals["manager_veto:time_constraint_violated"] += 1
        elif "Resource unavailable" in reason:
            refusals["manager_veto:resource_unavailable"] += 1
        else:
            refusals["manager_veto:other"] += 1
        return RunResult(
            scenario=scenario, system="CNP", iteration=iteration,
            success=False, rounds=1, messages_sent=msgs, total_tokens=0,
            final_utility=0.0, latency_seconds=time.time() - start,
            refusals=";".join(f"{k}:{v}" for k, v in refusals.items()) or None,
        ), refusals

    # AWARD — task allocated by the protocol; record the outcome under the
    # CANONICAL criterion (core/evaluation.py) shared by every system:
    # mean utility over the same agent set >= tau_min (audit F3).
    from core.evaluation import canonical_outcome
    outcome = canonical_outcome(proposal, agent_dicts, registry=[],
                                tau_min=tau_min,
                                active_agent_ids=active_agent_ids,
                                disruption_type=disruption_type)
    return RunResult(
        scenario=scenario, system="CNP", iteration=iteration,
        success=outcome.success, rounds=1, messages_sent=msgs, total_tokens=0,
        final_utility=outcome.aggregate_utility,
        latency_seconds=time.time() - start,
        refusals=";".join(f"{k}:{v}" for k, v in refusals.items()) or None,
        proposal_method=proposal.method.value,
        proposal_cost=proposal.cost_normalized,
        proposal_time=proposal.time_days,
        n_accepting=outcome.n_individually_accepting,
    ), refusals
