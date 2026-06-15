"""DANCER negotiation protocol implemented as a LangGraph state machine.

Implements Algorithm 1 from the paper:
  INIT → generate_proposals → validate → evaluate → aggregate
       → check_consensus → (SUCCESS | STALL | refine → validate → ...)
"""
from __future__ import annotations
import math
import random
import time
from typing import TypedDict, List, Dict, Optional

from langgraph.graph import StateGraph, END

from core.agents import IPAgent
from llm.llm_factory import get_llm, get_llm_by_name
from core.models import Proposal, Bid


# ══════════════════════════════════════════════════════════════════════════════
#  State schema
# ══════════════════════════════════════════════════════════════════════════════

class DANCERState(TypedDict):
    # Protocol hyper-parameters
    r_max: int
    tau_initial: float
    tau_min: float
    gamma: float
    k_max: int                   # max |P| per round (Proposition 2 premise)
    # Runtime state
    round: int
    tau: float
    # Data payloads (serialized as plain dicts for LangGraph compatibility)
    proposals: List[dict]
    valid_proposals: List[dict]
    last_raw_proposals: List[dict]
    bids: List[dict]
    best_proposal: Optional[dict]
    best_score: float
    # Agent info
    agent_dicts: List[dict]      # base agents (serialised IPAgent.to_dict())
    active_agent_ids: List[str]
    registry: List[dict]         # 3PL / discoverable agents
    # Scenario context
    disruption_type: str
    goal: str
    # Tracking
    status: str   # RUNNING | SUCCESS | STALL | ESCALATED
    rejection_reasons: List[dict]
    messages_sent: int
    total_tokens: int
    final_utility: float
    start_time: float
    # LLM backend selector: "mock" | "heuristic" | "real"
    llm_backend: str
    # Domain vocabulary (DomainContext from llm.domain_context) or None →
    # synthetic SupplyChainContext fallback inside ProductionLLM.
    domain_context: object
    # Fault injection: probability [0,1] that any single PEER_BID is lost in transit
    communication_drop_rate: float


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _all_agents(state: DANCERState) -> List[IPAgent]:
    """Reconstruct IPAgent objects from serialised state, including registry agents."""
    agents = [IPAgent.from_dict(d) for d in state["agent_dicts"]]
    registry_map = {r["agent_id"]: r for r in state["registry"]}
    for aid in state["active_agent_ids"]:
        if aid in registry_map and not any(a.agent_id == aid for a in agents):
            agents.append(IPAgent.from_dict(registry_map[aid]))
    return agents


def _p(d: dict) -> Proposal:
    return Proposal(**d)


# ══════════════════════════════════════════════════════════════════════════════
#  Nodes
# ══════════════════════════════════════════════════════════════════════════════

def generate_proposals_node(state: DANCERState) -> dict:
    """Cognitive Layer (π_θ): stochastic proposal generation.

    PRIVACY (audit F2): the Lead's LLM receives ONLY
      (i)   the Lead's OWN constraint vector θ_lead (agent_dicts[0]),
      (ii)  public registry capability adverts (discoverable by design),
      (iii) — at refinement time — the structured rejection vectors.
    Peer θ_j are NEVER placed in any prompt; this is the Dec-POSG
    information asymmetry the paper claims (O_k excludes peer θ_j).
    """
    llm = get_llm_by_name(state["llm_backend"]) if state["llm_backend"] != "env" else get_llm()
    lead_only = state["agent_dicts"][:1]   # θ_lead only — never the peer table
    if hasattr(llm, "set_context"):
        llm.set_context(
            agent_dicts=lead_only,
            tau_current=state["tau"],
            tau_min=state["tau_min"],
            domain_context=state.get("domain_context"),
            registry=state["registry"],
            n_peers=max(0, len(state["active_agent_ids"]) - 1),
        )
    proposals, tokens = llm.generate_proposals(
        disruption_type=state["disruption_type"],
        goal=state["goal"],
        available_agents=state["active_agent_ids"],
    )
    return {
        "proposals": [p.model_dump() for p in proposals],
        "total_tokens": state["total_tokens"] + tokens,
        "messages_sent": state["messages_sent"] + 1,   # NEGOT_INIT broadcast
    }


def validate_proposals_node(state: DANCERState) -> dict:
    """Deterministic Safeguard Layer (audit F2 + F4).

    Paper semantics (§4.1): proposals failing schema validation, containing
    out-of-bound values, or referencing unknown registry agents are DISCARDED
    — never silently repaired, and never validated against PEER budgets
    (the safeguard has no visibility of peer θ_j).

    Changes vs. original implementation:
      - REMOVED: cost clamp computed from min(max_cost) across ALL agents.
        It (a) read every peer's private budget and (b) actively pulled every
        proposal into the feasible cost region, inflating completion rates.
      - ADDED:   |P| <= K_max enforcement (Proposition 2 premise). When the
        accumulated proposal set exceeds K_max, the NEWEST proposals are kept
        (refinements supersede stale candidates); the current best proposal,
        if any, is always retained.
    """
    from core.proposal_envelopes import is_feasible
    from core.proposal_envelopes import infeasibility_reason
    registry_ids = {r["agent_id"] for r in state["registry"]}
    active = list(state["active_agent_ids"])
    valid = []
    discard_msgs = []   # F14: safeguard feedback for regeneration

    for p_dict in state["proposals"]:
        p = _p(p_dict)
        if not (0.0 <= p.cost_normalized <= 1.0 and p.time_days > 0.0):
            discard_msgs.append(f"{p.id}: schema violation "
                                f"(cost={p.cost_normalized}, time={p.time_days})")
            continue  # schema violation → discard
        env_reason = infeasibility_reason(p)
        if env_reason is not None:
            discard_msgs.append(f"{p.id}: {env_reason}")
            continue  # outside the method's market envelope (F12) → discard
        if (state["disruption_type"] == "structural_failure"
                and not p.requires_new_agent):
            # F13a: the disrupted capability is vacant; a proposal that does
            # not engage a registry agent cannot execute.
            discard_msgs.append(
                f"{p.id}: requires_new_agent missing — the disrupted "
                f"capability is VACANT; you MUST set requires_new_agent=true "
                f"with a registry agent_id")
            continue

        if p.requires_new_agent:
            if p.new_agent_id in registry_ids:
                valid.append(p_dict)
                if p.new_agent_id not in active:
                    active.append(p.new_agent_id)
            # else: not in registry → discard (hallucinated agent)
        else:
            valid.append(p_dict)

    # ── K_max cap (Proposition 2): keep newest first, preserve current best ──
    k_max = state.get("k_max", 5)
    if len(valid) > k_max:
        best = state.get("best_proposal")
        kept = list(valid[-k_max:])          # newest K_max
        if best is not None and best.get("id") not in {q["id"] for q in kept}:
            kept[0] = best                    # retain incumbent best
        valid = kept

    update = {"valid_proposals": valid, "active_agent_ids": active,
              "last_raw_proposals": list(state["proposals"])}

    if not valid:
        # F14: ALL proposals discarded. The original implementation escalated
        # immediately at round 0, so DANCER forfeited every such episode
        # without using a single negotiation round (v4: 59/60 failures were
        # round-0 escalations with exactly one generation call). Per
        # Algorithm 1 the round loop continues: the safeguard's discard
        # reasons become the rejection feedback for the next refinement,
        # one round is consumed, and τ decays — bounded termination
        # (Proposition 1) is preserved because r still strictly increases.
        r = state["round"]
        new_tau = max(state["tau_min"],
                      state["tau"] * math.exp(-state["gamma"] * r))
        update.update({
            "rejection_reasons": [{
                "agent_id": "SAFEGUARD",
                "reason": ("ALL proposals were discarded before broadcast: "
                           + "; ".join(discard_msgs[:6])),
                "utility": None,
                "utility_vector": {},
            }],
            "round": r + 1,
            "tau": new_tau,
        })
    return update


def evaluate_proposals_node(state: DANCERState) -> dict:
    """Each active agent evaluates every valid proposal → PEER_BID messages.

    Fault injection: each bid is independently dropped with probability
    `communication_drop_rate`, simulating unreliable network delivery.
    A dropped bid is counted as a sent message (the agent computed and
    transmitted it) but never reaches the aggregation node.
    """
    agents = _all_agents(state)
    active = set(state["active_agent_ids"])
    drop_rate = state["communication_drop_rate"]
    bids: List[dict] = []
    msgs = 0

    for p_dict in state["valid_proposals"]:
        p = _p(p_dict)
        for agent in agents:
            if agent.agent_id not in active:
                continue
            bid: Bid = agent.evaluate_proposal(p)
            msgs += 1                              # message was sent
            if random.random() < drop_rate:
                continue                           # lost in transit — do not append
            bids.append(bid.model_dump())

    return {"bids": bids, "messages_sent": state["messages_sent"] + msgs}


def aggregate_scores_node(state: DANCERState) -> dict:
    """Compute weighted aggregate S_p = (1/|received|) Σ L_k(p) for each proposal.

    Under packet loss, fewer bids than active agents may arrive for a given
    proposal. The aggregate is computed over *received* bids only, which has
    two effects:
      1. Fewer votes → higher variance in S_p, potentially pushing borderline
         proposals above τ (optimistic bias) or causing stable ones to fall
         below τ (pessimistic bias) depending on whose bid was dropped.
      2. If ALL bids for a proposal are lost, that proposal is absent from
         `scores` and is effectively invisible to the consensus check for
         this round — it may resurface via Contrastive Refinement.
    """
    scores: Dict[str, List[float]] = {}
    for b in state["bids"]:
        pid = b["proposal_id"]
        scores.setdefault(pid, []).append(b["utility_score"])

    if not scores:
        return {"best_score": 0.0, "best_proposal": None}

    # Average over received bids only — correct behaviour under partial delivery
    aggregated = {pid: sum(vs) / len(vs) for pid, vs in scores.items()}
    best_id = max(aggregated, key=aggregated.__getitem__)
    best_score = aggregated[best_id]
    best_proposal = next(
        (p for p in state["valid_proposals"] if p["id"] == best_id), None
    )
    return {"best_score": best_score, "best_proposal": best_proposal}


def check_consensus_node(state: DANCERState) -> dict:
    """Check S_{p*} >= τ_r; apply temporal decay and collect rejection reasons."""
    if state["best_score"] >= state["tau"] and state["best_proposal"] is not None:
        return {"status": "SUCCESS"}

    if state["round"] >= state["r_max"]:
        return {"status": "STALL"}

    # Collect rejection reasons WITH full utility vectors for Contrastive Refinement.
    # Passing f_res, f_time, f_pol lets the LLM identify the BINDING constraint
    # dimension precisely, rather than inferring it from free-text reasons alone.
    best_id = state["best_proposal"]["id"] if state["best_proposal"] else None
    rejections = [
        {
            "agent_id":       b["agent_id"],
            "proposal_id":    b["proposal_id"],
            "reason":         b["rejection_reason"],
            "utility":        b["utility_score"],
            "utility_vector": b.get("utility_vector", {}),  # f_res, f_time, f_pol
        }
        for b in state["bids"]
        if not b["accepted"] and b["rejection_reason"] and b["proposal_id"] == best_id
    ]

    # Temporal decay: τ_{r+1} = max(τ_min, τ_r * e^{-γr})
    new_tau = max(
        state["tau_min"],
        state["tau"] * math.exp(-state["gamma"] * state["round"]),
    )

    return {
        "status": "RUNNING",
        "tau": new_tau,
        "rejection_reasons": rejections,
        "round": state["round"] + 1,
    }


def refine_proposals_node(state: DANCERState) -> dict:
    """Contrastive Refinement (Algorithm 1, line 18): LLM mutates proposals.

    Same privacy contract as generation: θ_lead + registry + the structured
    rejection vectors V (the protocol's designed inter-round channel). The
    rejection vectors carry per-dimension scores f_res/f_time/f_pol — the
    'normalized margin' of Definition 4 — NOT the peers' raw θ_j parameters.
    """
    llm = get_llm_by_name(state["llm_backend"]) if state["llm_backend"] != "env" else get_llm()
    lead_only = state["agent_dicts"][:1]
    if hasattr(llm, "set_context"):
        llm.set_context(
            agent_dicts=lead_only,
            tau_current=state["tau"],
            tau_min=state["tau_min"],
            domain_context=state.get("domain_context"),
            registry=state["registry"],
            n_peers=max(0, len(state["active_agent_ids"]) - 1),
        )
    prev_dicts = state["valid_proposals"] or state.get("last_raw_proposals", [])
    prev = [_p(d) for d in prev_dicts]
    refined, tokens = llm.refine_proposals(
        previous_proposals=prev,
        rejection_reasons=state["rejection_reasons"],
        round_num=state["round"],
        disruption_type=state["disruption_type"],
    )
    # Merge new proposals; avoid duplicating existing IDs
    existing_ids = {p["id"] for p in state["valid_proposals"]}
    new_dicts = [p.model_dump() for p in refined if p.id not in existing_ids]

    return {
        "proposals": state["valid_proposals"] + new_dicts,
        "total_tokens": state["total_tokens"] + tokens,
        "messages_sent": state["messages_sent"] + 1,
    }


def commit_node(state: DANCERState) -> dict:
    """CONSENSUS + TOKEN_XFER: process resumes."""
    return {"status": "SUCCESS", "final_utility": state["best_score"]}


def escalate_node(state: DANCERState) -> dict:
    """Negotiation failed: escalate to human supervisor."""
    return {"status": "ESCALATED", "final_utility": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
#  Routing
# ══════════════════════════════════════════════════════════════════════════════

def _route_after_validate(state: DANCERState) -> str:
    if state["valid_proposals"]:
        return "evaluate"
    # F14: regenerate from safeguard feedback while the round budget lasts
    if state["round"] <= state["r_max"]:
        return "refine"
    return "escalate"


def _route_after_consensus(state: DANCERState) -> str:
    if state["status"] == "SUCCESS":
        return "commit"
    elif state["status"] == "STALL":
        return "escalate"
    return "refine"


# ══════════════════════════════════════════════════════════════════════════════
#  Graph construction
# ══════════════════════════════════════════════════════════════════════════════

def build_dancer_graph():
    g = StateGraph(DANCERState)

    g.add_node("generate", generate_proposals_node)
    g.add_node("validate", validate_proposals_node)
    g.add_node("evaluate", evaluate_proposals_node)
    g.add_node("aggregate", aggregate_scores_node)
    g.add_node("check_consensus", check_consensus_node)
    g.add_node("refine", refine_proposals_node)
    g.add_node("commit", commit_node)
    g.add_node("escalate", escalate_node)

    g.set_entry_point("generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        # F14: an emptied proposal set routes back through refinement with
        # the safeguard's discard feedback while the round budget lasts.
        {"evaluate": "evaluate", "escalate": "escalate", "refine": "refine"},
    )
    g.add_edge("evaluate", "aggregate")
    g.add_edge("aggregate", "check_consensus")
    g.add_conditional_edges(
        "check_consensus",
        _route_after_consensus,
        {"commit": "commit", "escalate": "escalate", "refine": "refine"},
    )
    g.add_edge("refine", "validate")
    g.add_edge("commit", END)
    g.add_edge("escalate", END)

    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
#  Public runner
# ══════════════════════════════════════════════════════════════════════════════

def run_dancer(
    base_agents: List[dict],
    active_agent_ids: List[str],
    registry: List[dict],
    disruption_type: str,
    goal: str = "Complete process choreography despite disruption",
    r_max: int = 5,
    tau_initial: float = 0.75,
    tau_min: float = 0.40,
    gamma: float = 0.30,
    k_max: int = 5,                       # Proposition 2 cap on |P| per round
    llm_backend: str = "env",             # "env" | "mock" | "heuristic" | "real"
    communication_drop_rate: float = 0.0, # fraction of PEER_BIDs lost in transit
    domain_context: object = None,        # DomainContext from BPICAdapter (real-log runs)
) -> dict:
    """Execute one DANCER negotiation episode. Returns final state dict.

    llm_backend:
      "env"       — respects USE_REAL_LLM env var (default, backward-compatible)
      "mock"      — forces MockLLM (Contrastive Refinement)
      "heuristic" — same GENERATION backend, blind ±20% refinement (ablation)
      "real"      — forces ProductionLLM (DANCER_BASE_URL / OpenRouter / OpenAI)
    """
    graph = build_dancer_graph()
    initial: DANCERState = {
        "r_max": r_max,
        "tau_initial": tau_initial,
        "tau_min": tau_min,
        "gamma": gamma,
        "k_max": k_max,
        "round": 0,
        "tau": tau_initial,
        "proposals": [],
        "valid_proposals": [],
        "last_raw_proposals": [],
        "bids": [],
        "best_proposal": None,
        "best_score": 0.0,
        "agent_dicts": base_agents,
        "active_agent_ids": list(active_agent_ids),
        "registry": registry,
        "disruption_type": disruption_type,
        "goal": goal,
        "status": "RUNNING",
        "rejection_reasons": [],
        "messages_sent": 0,
        "total_tokens": 0,
        "final_utility": 0.0,
        "start_time": time.time(),
        "llm_backend": llm_backend,
        "domain_context": domain_context,
        "communication_drop_rate": communication_drop_rate,
    }
    final = graph.invoke(initial)
    final["latency_seconds"] = time.time() - final["start_time"]
    return final
