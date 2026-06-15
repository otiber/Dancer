"""
core/evaluation.py — Canonical outcome evaluation shared by ALL systems.

WHY THIS MODULE EXISTS (audit finding F3, 2026-06):
  The original harness recorded success under HETEROGENEOUS criteria:
    - DANCER:        mean utility over received bids >= tau_r (decaying to 0.40)
    - LLM-Orchestra: ALL agents individually accept (L_k >= min_utility_k)
    - ReAct-MAS:     majority LLM vote AND unanimous individual acceptance
    - MAPE-K:        mean utility >= fixed 0.55
    - CNP / z-CNP:   contractor individual acceptance + manager veto
  Under structural failure the resource agent stays active with f_res = 0 and
  the highest w_res, so any UNANIMITY criterion that includes it can never be
  satisfied — the 0% rows for LLM baselines were guaranteed by the harness,
  not by the systems. This module removes that artifact.

CANONICAL CRITERION (identical for every system):
  Given the committed proposal p, the outcome agent set is
      A* = active base agents  ∪  {registry agent}  if p.requires_new_agent
  (the unavailable carrier REMAINS in A* — same dilution for everyone), and
      S_p = (1/|A*|) Σ_{a_k ∈ A*} L_k(p)
      success  ⇔  S_p >= tau_min(scenario)
  This is exactly DANCER's Definition 3 aggregate evaluated at the scenario
  floor tau_min — the weakest bar DANCER itself can pass at — applied
  uniformly. Each system's INTERNAL mechanics (CNP role matching, manager
  veto, ReAct voting, MAPE-K planning, DANCER negotiation) still decide
  WHETHER a proposal gets committed at all; this module only judges the
  committed proposal.

Returned per-agent scores allow secondary analyses (e.g., how many agents
individually accept under each system) without changing the primary metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.agents import IPAgent
from core.models import Proposal


@dataclass
class CanonicalOutcome:
    success: bool
    aggregate_utility: float            # S_p over the canonical agent set
    per_agent: Dict[str, float] = field(default_factory=dict)
    n_individually_accepting: int = 0   # secondary diagnostic (L_k >= min_utility_k)
    agent_set: List[str] = field(default_factory=list)
    tau_min: float = 0.0


def feasibility_oracle(
    agent_dicts: List[dict],
    registry: List[dict],
    tau_min: float,
    disruption_type: Optional[str] = None,
    active_agent_ids: Optional[List[str]] = None,
) -> bool:
    """True iff SOME envelope-feasible proposal clears the canonical bar.

    Both f_pol and f_time are monotone non-increasing in cost and time, so
    each method's optimum sits at its envelope corner (cost_lo, time_lo);
    checking the 7 corners (with mandatory registry engagement under
    structural failure) decides episode feasibility exactly. Episodes where
    this returns False are unsolvable BY CONSTRUCTION — completion-rate
    comparisons should be reported on the feasible subset (CR_feasible).
    """
    from core.models import Proposal, ShippingMethod
    from core.proposal_envelopes import ENVELOPES
    structural = disruption_type == "structural_failure"
    reg_id = registry[0]["agent_id"] if registry else None
    if structural and reg_id is None:
        return False
    for m, ((c_lo, _), (t_lo, _)) in ENVELOPES.items():
        p = Proposal(id="oracle", method=ShippingMethod(m),
                     cost_normalized=c_lo, time_days=t_lo,
                     description="oracle corner",
                     requires_new_agent=structural,
                     new_agent_id=reg_id if structural else None)
        o = canonical_outcome(p, agent_dicts, registry, tau_min,
                              active_agent_ids=active_agent_ids,
                              disruption_type=disruption_type)
        if o.success:
            return True
    return False


def build_outcome_agents(
    agent_dicts: List[dict],
    registry: List[dict],
    proposal: Optional[Proposal],
    active_agent_ids: Optional[List[str]] = None,
) -> List[IPAgent]:
    """Reconstruct the canonical agent set for outcome evaluation.

    - All ACTIVE base agents are included (the disrupted carrier with
      committed_f_res = 0 included — identical treatment for every system).
    - The registry agent is added iff the committed proposal requires it
      (requires_new_agent + new_agent_id present in registry).
    - committed_f_res in agent_dicts is preserved, so every system in the
      same episode evaluates against the SAME availability draw.
    """
    active = set(active_agent_ids) if active_agent_ids else {
        d["agent_id"] for d in agent_dicts
    }
    agents = [IPAgent.from_dict(d) for d in agent_dicts if d["agent_id"] in active]

    if proposal is not None and proposal.requires_new_agent and proposal.new_agent_id:
        registry_map = {r["agent_id"]: r for r in registry}
        rid = proposal.new_agent_id
        if rid in registry_map and not any(a.agent_id == rid for a in agents):
            agents.append(IPAgent.from_dict(registry_map[rid]))
    return agents


def canonical_outcome(
    proposal: Optional[Proposal],
    agent_dicts: List[dict],
    registry: List[dict],
    tau_min: float,
    active_agent_ids: Optional[List[str]] = None,
    disruption_type: Optional[str] = None,
) -> CanonicalOutcome:
    """Evaluate the committed proposal under the canonical criterion.

    proposal=None (nothing committed) → failure with utility 0.0.

    AUDIT F12 — two environment-level feasibility rules, identical for every
    system:
      (a) ENVELOPE: a proposal whose (cost, time) lie outside its method's
          market envelope is economically infeasible → failure.
      (b) CAPABILITY RESTORATION: under structural failure the disrupted
          capability is vacant; a proposal that does not engage a registry
          agent cannot execute → failure. (Without this rule, systems scored
          100% 'structural recovery' while leaving the dead carrier dead in
          89–100% of episodes.)
    The aggregate utility is still computed and reported for diagnostics.
    """
    if proposal is None:
        return CanonicalOutcome(success=False, aggregate_utility=0.0,
                                tau_min=tau_min)

    from core.proposal_envelopes import is_feasible
    infeasible = not is_feasible(proposal)
    if (disruption_type == "structural_failure"
            and not proposal.requires_new_agent):
        infeasible = True

    agents = build_outcome_agents(agent_dicts, registry, proposal,
                                  active_agent_ids)
    if not agents:
        return CanonicalOutcome(success=False, aggregate_utility=0.0,
                                tau_min=tau_min)

    per_agent: Dict[str, float] = {}
    n_accept = 0
    for a in agents:
        bid = a.evaluate_proposal(proposal)
        per_agent[a.agent_id] = bid.utility_score
        if bid.accepted:
            n_accept += 1

    s = sum(per_agent.values()) / len(per_agent)
    return CanonicalOutcome(
        success=(s >= tau_min) and not infeasible,
        aggregate_utility=round(s, 6),
        per_agent=per_agent,
        n_individually_accepting=n_accept,
        agent_set=[a.agent_id for a in agents],
        tau_min=tau_min,
    )
