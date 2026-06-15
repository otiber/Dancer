"""
ζ-CNP: Extended Contract Net Protocol with Task Decomposition.

Reference: Ye, F., Shen, W., & Hao, Q. (2017). Extended contract net protocol
           for dynamic task allocation in multi-agent systems. IEEE CSCWD, 123-128.

Extension over classical CNP:
  - Phase 2b: if no eligible bidder found, the manager DECOMPOSES the task
    into subtasks and re-announces each independently (one decomposition only).
  - Decomposition rules are pre-registered per disruption type.
  - Still NO semantic LLM reasoning and NO task REFORMULATION.
  - Decomposition allows some structural failures to be resolved if a suitable
    3PL agent exists in the contractor pool.

Design choices for fair comparison:
  - Same utility function L_k(p) as DANCER and CNP.
  - Same committed_f_res (shared availability draw per episode).
  - Decomposition rules include a warehouse + 3PL split for structural failure.
  - One additional decomposition round → messages_sent = 2*(n+1) in worst case.
  - Role matching applied after decomposition (same as CNP Phase 2).

Compared to CNP:
  + One-level task decomposition recovers some structural failures.
  - Schema is still fixed within each subtask.
  - Cannot discover agents outside the contractor list (no registry).

Compared to DANCER:
  - No semantic reasoning; decomposition rules are hard-coded.
  - No contrastive refinement; failure in subtask = stall.
  - No sovereignty preservation (manager sees all bids).
  - No temporal decay mechanism.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import List, Tuple, Dict

from core.models import Proposal, ShippingMethod, RunResult
from core.agents import IPAgent


# ── Task schemas (same as CNP for initial announcement) ──────────────────────

_TASK_SCHEMA: Dict[str, Dict] = {
    "parametric_fluctuation": {
        "proposal": Proposal(
            id="zcnp_task_A",
            method=ShippingMethod.EXPEDITED_GROUND,
            cost_normalized=0.30,
            time_days=1.5,
            description="ζ-CNP: expedited ground — warehouse delay recovery",
        ),
        "required_role": "warehouse",
    },
    "structural_failure": {
        "proposal": Proposal(
            id="zcnp_task_B",
            method=ShippingMethod.STANDARD,
            cost_normalized=0.35,
            time_days=4.0,
            description="ζ-CNP: standard shipping — primary carrier on strike",
        ),
        "required_role": "carrier",
    },
    "sovereignty_conflict": {
        "proposal": Proposal(
            id="zcnp_task_C",
            method=ShippingMethod.STANDARD,
            cost_normalized=0.55,
            time_days=4.0,
            description="ζ-CNP: standard shipping — pre-sovereignty cost level",
        ),
        "required_role": "carrier",
    },
}

# ── Decomposition rules ───────────────────────────────────────────────────────
# Each rule maps a disruption to 1-2 subtasks that REPLACE the failed task.
# weight: fraction of total cost assigned to this subtask.

_DECOMPOSITION_RULES: Dict[str, List[Dict]] = {
    "structural_failure": [
        {
            "proposal": Proposal(
                id="zcnp_sub_B1",
                method=ShippingMethod.EXPEDITED_GROUND,
                cost_normalized=0.30,
                time_days=1.5,
                description="ζ-CNP subtask: warehouse dispatch leg",
            ),
            "required_role": "warehouse",
        },
        {
            "proposal": Proposal(
                id="zcnp_sub_B2",
                method=ShippingMethod.THIRD_PARTY_3PL,
                cost_normalized=0.40,
                time_days=3.0,
                description="ζ-CNP subtask: 3PL delivery leg",
            ),
            "required_role": "third_party_logistics",
        },
    ],
    "sovereignty_conflict": [
        {
            "proposal": Proposal(
                id="zcnp_sub_C1",
                method=ShippingMethod.CONSOLIDATED_RAIL,
                cost_normalized=0.38,
                time_days=3.5,
                description="ζ-CNP subtask: consolidated rail — cost reduction",
            ),
            "required_role": "carrier",
        },
    ],
    "parametric_fluctuation": [],   # no decomposition needed
}


# ── Bid / Award helpers (same logic as cnp.py) ────────────────────────────────

def _run_bid_phase(
    proposal: Proposal,
    required_role: str,
    contractors: List[IPAgent],
    refusals: Counter,
) -> Tuple[float, object]:  # (best_score, best_bid | None)
    best_score = -1.0
    best_bid = None
    for agent in contractors:
        if agent.role != required_role:
            refusals["role_mismatch"] += 1
            continue
        bid = agent.evaluate_proposal(proposal)
        if bid.accepted and bid.utility_score > best_score:
            best_score = bid.utility_score
            best_bid = bid
        elif not bid.accepted:
            reason = bid.rejection_reason or ""
            if "Cost" in reason:
                refusals["cost_constraint_violated"] += 1
            elif "Time" in reason:
                refusals["time_constraint_violated"] += 1
            else:
                refusals["utility_below_threshold"] += 1
    return best_score, best_bid


# ── Public runner ─────────────────────────────────────────────────────────────

def run_zeta_cnp(
    scenario: str,
    agent_dicts: List[dict],
    active_agent_ids: List[str],
    registry: List[dict],
    disruption_type: str,
    iteration: int = 0,
    tau_min: float = 0.45,
    required_role: str = None,
) -> Tuple[RunResult, Counter]:
    """Execute one ζ-CNP episode with optional task decomposition.

    Returns
    -------
    result   : RunResult
    refusals : Counter of refusal reasons (for analysis)
    """
    start = time.time()
    refusals: Counter = Counter()
    msgs = 0

    task_spec = _TASK_SCHEMA.get(disruption_type)
    if task_spec is None:
        return RunResult(
            scenario=scenario, system="ζ-CNP", iteration=iteration,
            success=False, rounds=1, messages_sent=1, total_tokens=0,
            final_utility=0.0, latency_seconds=time.time() - start,
        ), refusals

    # Build agent pool (include registry agents for structural failures)
    active_set = set(active_agent_ids)
    registry_map = {r["agent_id"]: r for r in registry}
    all_dicts = list(agent_dicts)
    if disruption_type in ("structural_failure",):
        for aid, rd in registry_map.items():
            if not any(d["agent_id"] == aid for d in all_dicts):
                all_dicts.append(rd)

    agents: List[IPAgent] = [
        IPAgent.from_dict(d) for d in all_dicts
        if d["agent_id"] in active_set or d["agent_id"] in registry_map
    ]
    manager = agents[0]
    contractors = agents[1:]

    # ── Round 1: Original Task Announcement ──────────────────────────────────
    proposal = task_spec["proposal"]
    required_role = required_role or task_spec["required_role"]   # audit F11
    msgs += 1   # TASK_ANNOUNCE

    best_score, best_bid = _run_bid_phase(proposal, required_role, contractors, refusals)
    msgs += len(contractors)    # one response per contractor

    # ── Manager Award Check ──────────────────────────────────────────────────
    if best_bid is not None:
        manager_eval = manager.evaluate_proposal(proposal)
        msgs += 1   # AWARD or STALL_NOTIFY
        if manager_eval.accepted:
            # Canonical outcome (audit F3): shared criterion across systems.
            from core.evaluation import canonical_outcome
            outcome = canonical_outcome(proposal, agent_dicts, registry,
                                        tau_min,
                                        active_agent_ids=active_agent_ids,
                                        disruption_type=disruption_type)
            return RunResult(
                scenario=scenario, system="ζ-CNP", iteration=iteration,
                success=outcome.success, rounds=1, messages_sent=msgs,
                total_tokens=0,
                final_utility=outcome.aggregate_utility,
                latency_seconds=time.time() - start,
                refusals=";".join(f"{k}:{v}" for k, v in refusals.items()) or None,
                proposal_method=proposal.method.value,
                proposal_cost=proposal.cost_normalized,
                proposal_time=proposal.time_days,
                n_accepting=outcome.n_individually_accepting,
            ), refusals
        else:
            reason = manager_eval.rejection_reason or ""
            if "Cost" in reason:
                refusals["manager_veto:cost_constraint_violated"] += 1
            else:
                refusals["manager_veto:other"] += 1

    # ── Round 2: Task Decomposition ──────────────────────────────────────────
    decomp_rules = _DECOMPOSITION_RULES.get(disruption_type, [])
    if not decomp_rules:
        # No decomposition defined → stall (same as CNP)
        msgs += 1
        return RunResult(
            scenario=scenario, system="ζ-CNP", iteration=iteration,
            success=False, rounds=2, messages_sent=msgs, total_tokens=0,
            final_utility=0.0, latency_seconds=time.time() - start,
        ), refusals

    # Announce each subtask independently
    subtask_scores: List[float] = []
    subtask_proposals: List[Proposal] = []
    all_subtasks_awarded = True

    for subtask_spec in decomp_rules:
        sub_proposal = subtask_spec["proposal"]
        sub_role = subtask_spec["required_role"]
        msgs += 1   # subtask TASK_ANNOUNCE

        sub_score, sub_bid = _run_bid_phase(sub_proposal, sub_role, contractors, refusals)
        msgs += len(contractors)

        if sub_bid is None:
            all_subtasks_awarded = False
            refusals["subtask_no_eligible_bidder"] += 1
            msgs += 1   # STALL_NOTIFY for this subtask
            break

        # Manager validates subtask
        manager_sub_eval = manager.evaluate_proposal(sub_proposal)
        msgs += 1
        if not manager_sub_eval.accepted:
            all_subtasks_awarded = False
            refusals["manager_veto:subtask_cost_constraint"] += 1
            break

        sub_all_scores = [a.evaluate_proposal(sub_proposal).utility_score for a in agents]
        subtask_scores.append(sum(sub_all_scores) / len(sub_all_scores) if sub_all_scores else sub_score)
        subtask_proposals.append(sub_proposal)

    if all_subtasks_awarded and subtask_scores:
        # Canonical outcome (audit F3): each awarded subtask is judged under
        # the shared criterion; episode succeeds iff the MEAN canonical
        # aggregate over subtasks clears tau_min.
        from core.evaluation import canonical_outcome
        sub_outcomes = [
            canonical_outcome(sp, agent_dicts, registry, tau_min,
                              active_agent_ids=active_agent_ids,
                              disruption_type=disruption_type)
            for sp in subtask_proposals
        ]
        mean_agg = sum(o.aggregate_utility for o in sub_outcomes) / len(sub_outcomes)
        return RunResult(
            scenario=scenario, system="ζ-CNP", iteration=iteration,
            success=mean_agg >= tau_min, rounds=2, messages_sent=msgs,
            total_tokens=0,
            final_utility=round(mean_agg, 6),
            latency_seconds=time.time() - start,
            refusals=";".join(f"{k}:{v}" for k, v in refusals.items()) or None,
            proposal_method="DECOMPOSED:" + "+".join(
                sp.method.value for sp in subtask_proposals),
            proposal_cost=round(sum(sp.cost_normalized for sp in subtask_proposals)
                                / len(subtask_proposals), 4),
            proposal_time=round(sum(sp.time_days for sp in subtask_proposals)
                                / len(subtask_proposals), 2),
        ), refusals

    return RunResult(
        scenario=scenario, system="ζ-CNP", iteration=iteration,
        success=False, rounds=2, messages_sent=msgs, total_tokens=0,
        final_utility=0.0, latency_seconds=time.time() - start,
    ), refusals
