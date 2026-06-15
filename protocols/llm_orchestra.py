"""
LLM-Orchestra: Centralized LLM Orchestrator Baseline.

Motivation:
  A natural question for decentralized LLM-MAS is: why not simply query a single
  powerful LLM with full visibility of all agents and their constraints?
  LLM-Orchestra represents this "LLM monolith" design.

Architecture:
  - A single Orchestrator LLM receives the COMPLETE state of all agents,
    including their private constraint matrices (max_cost, max_delay, weights).
  - It generates a single binding decision in one shot (no negotiation rounds).
  - The decision is validated against the formal utility function L_k(p) post-hoc.

Compared to DANCER:
  ✗ Violates organizational sovereignty: all private constraints are exposed.
  ✗ Single point of failure: one LLM call; no iterative refinement.
  ✗ No structured rejection feedback loop.
  ✗ No registry discovery mechanism (agents must be listed in prompt).
  ✓ Lowest latency (single LLM call per episode).
  ✓ Highest information access (centralized visibility).
  ~ Same LLM model as DANCER's Cognitive Layer.

This baseline tests whether the communication overhead of DANCER's decentralized
protocol is justified by the sovereignty and resilience properties it provides.

Cost note: 1 LLM call per episode (vs DANCER's O(R_max) calls).
  At similar token counts, LLM-Orchestra has lower inference cost but higher
  sovereignty cost — a trade-off the paper makes explicit.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional, Tuple

from openai import OpenAI

from core.models import Proposal, ShippingMethod, RunResult
from core.agents import IPAgent

_VALID_METHODS = {m.value for m in ShippingMethod}


# ── LLM client ───────────────────────────────────────────────────────────────

def _get_client() -> Tuple[OpenAI, str]:
    if os.environ.get("DANCER_BASE_URL"):
        client = OpenAI(
            api_key=os.environ.get("DANCER_API_KEY", "dummy"),
            base_url=os.environ["DANCER_BASE_URL"],
        )
    elif os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    elif os.environ.get("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        raise EnvironmentError("Set DANCER_BASE_URL, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    model = os.environ.get("DANCER_MODEL", "openai/gpt-4o-mini")
    return client, model


_SYSTEM_PROMPT = """\
You are a centralized supply-chain orchestrator with FULL visibility of all agents.
Given a disruption and complete agent constraints, output a SINGLE optimal logistics proposal.

Output ONLY a JSON object (no markdown, no prose):
{
  "method": "<one of the allowed values>",
  "cost_normalized": <float 0.0-1.0>,
  "time_days": <positive float>,
  "description": "<one sentence>",
  "requires_new_agent": <true|false>,
  "new_agent_id": "<string or null>"
}

Allowed method values:
  Standard_Shipping | Express_Air_Freight | Consolidated_Rail |
  Land_Transport | Third_Party_Logistics | Alternative_Port_B | Expedited_Ground

Rules:
  - HARD constraint: cost_normalized and time_days MUST lie inside the
    method's market envelope appended below — out-of-envelope proposals are
    rejected as economically infeasible. NEVER price below an envelope floor,
    even when an agent's budget is lower than the floor.
  - SOFT tolerances: agents' max_cost / max_delay_days are utility tolerances,
    not hard limits — each agent's score degrades gradually as cost or time
    exceeds its tolerance (it remains positive up to ~1.5x budget and ~2x
    delay). When a budget lies below every envelope floor, choose the
    cheapest envelope-feasible option; do not chase the budget below a floor.
  - If requires_new_agent=true, new_agent_id MUST be "3PL_Agent".
  - Choose the method that achieves the highest average utility across all agents."""


def _build_agent_context(agents: List[IPAgent], registry: List[dict]) -> str:
    lines = ["Agent Profiles (PRIVATE — centralized view only):"]
    for a in agents:
        lines.append(
            f"  {a.agent_id} [role={a.role}]: "
            f"max_cost={a.max_cost:.2f}, max_delay={a.max_delay_days:.1f}d, "
            f"availability={a.committed_f_res:.2f}, "
            f"weights=(res={a.weights['w_res']:.2f}, time={a.weights['w_time']:.2f}, "
            f"pol={a.weights['w_pol']:.2f})"
        )
    if registry:
        lines.append(f"\nRegistry agents available: {[r['agent_id'] for r in registry]}")
    return "\n".join(lines)


def _parse_proposal(text: str, pid: str = "orch_1") -> Optional[Proposal]:
    """Extract and validate a Proposal from LLM output."""
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct JSON parse
    candidates = [text]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group())

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            method = obj.get("method", "Standard_Shipping")
            if method not in _VALID_METHODS:
                method = "Standard_Shipping"
            cost = max(0.0, min(1.0, float(obj.get("cost_normalized", 0.5))))
            time_d = max(0.1, float(obj.get("time_days", 3.0)))
            req_new = bool(obj.get("requires_new_agent", False))
            new_id = obj.get("new_agent_id", None)
            if req_new and not new_id:
                new_id = "3PL_Agent"
            return Proposal(
                id=pid,
                method=ShippingMethod(method),
                cost_normalized=cost,
                time_days=time_d,
                description=str(obj.get("description", "Orchestrator decision")),
                requires_new_agent=req_new,
                new_agent_id=new_id,
            )
        except Exception:
            continue
    return None


# ── Public runner ─────────────────────────────────────────────────────────────

def run_llm_orchestra(
    scenario: str,
    agent_dicts: List[dict],
    active_agent_ids: List[str],
    registry: List[dict],
    disruption_type: str,
    iteration: int = 0,
    tau_min: float = 0.45,
) -> RunResult:
    """Execute one LLM-Orchestra episode (single centralized LLM call).

    Privacy violation BY DESIGN: all agent constraint matrices are exposed to
    the LLM — this is the centralized 'oracle' baseline contrasting with
    DANCER's privacy-preserving prompts.

    AUDIT F3: success was previously `all(agents accept individually)`, an
    unanimity criterion that the unavailable carrier (f_res = 0, highest
    w_res) could never satisfy on structural failures — guaranteeing 0%
    regardless of the LLM's proposal. The recorded outcome now uses the
    canonical criterion shared by every system: mean utility over the same
    agent set >= tau_min.
    """
    start = time.time()
    client, model = _get_client()

    # Reconstruct all agents (including registry for structural failures)
    active_set = set(active_agent_ids)
    registry_map = {r["agent_id"]: r for r in registry}
    all_dicts = list(agent_dicts)
    if disruption_type == "structural_failure":
        for aid, rd in registry_map.items():
            if not any(d["agent_id"] == aid for d in all_dicts):
                all_dicts.append(rd)

    agents: List[IPAgent] = [IPAgent.from_dict(d) for d in all_dicts
                              if d["agent_id"] in active_set
                              or d["agent_id"] in registry_map]

    if not agents:
        return RunResult(scenario=scenario, system="LLM-Orchestra", iteration=iteration,
                         success=False, rounds=1, messages_sent=1, total_tokens=0,
                         final_utility=0.0, latency_seconds=time.time() - start)

    from core.proposal_envelopes import envelope_prompt_block
    agent_context = _build_agent_context(agents, registry) + "\n\n" + \
        envelope_prompt_block()

    disruption_hints = {
        "parametric_fluctuation": (
            "A 24-hour warehouse delay occurred. Find a faster shipping alternative "
            "that fits within all agents' cost and time constraints."
        ),
        "structural_failure": (
            "The primary Carrier is unavailable (port strike). "
            "Use the 3PL_Agent from the registry as the logistics provider. "
            "Set requires_new_agent=true, new_agent_id='3PL_Agent'."
        ),
        "sovereignty_conflict": (
            "The lead organization has tightened its cost budget mid-process "
            "(see its max_cost in the agent profiles). Select the most "
            "cost-effective option that remains INSIDE the market feasibility "
            "envelopes below — never price below an envelope floor."
        ),
    }
    hint = disruption_hints.get(disruption_type, "Resolve the supply-chain disruption.")

    user_prompt = (
        f"Disruption Type: {disruption_type}\n"
        f"Scenario Guidance: {hint}\n\n"
        f"{agent_context}\n\n"
        "Output a single proposal that satisfies all agent constraints above."
    )

    # Single LLM call — centralized decision
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.3,   # lower temp for deterministic orchestration
        "max_tokens": 400,
    }
    try:
        kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        kwargs.pop("response_format", None)
        resp = client.chat.completions.create(**kwargs)

    text = resp.choices[0].message.content or ""
    total_tokens = resp.usage.total_tokens if resp.usage else len(text) // 4

    proposal = _parse_proposal(text, pid=f"orch_{iteration}")

    if proposal is None:
        return RunResult(scenario=scenario, system="LLM-Orchestra", iteration=iteration,
                         success=False, rounds=1, messages_sent=1,
                         total_tokens=total_tokens, final_utility=0.0,
                         latency_seconds=time.time() - start)

    # ── Canonical outcome (shared criterion — core/evaluation.py) ────────────
    from core.evaluation import canonical_outcome
    outcome = canonical_outcome(proposal, agent_dicts, registry, tau_min,
                                active_agent_ids=active_agent_ids,
                                disruption_type=disruption_type)

    return RunResult(
        scenario=scenario, system="LLM-Orchestra", iteration=iteration,
        success=outcome.success, rounds=1, messages_sent=1,
        total_tokens=total_tokens, final_utility=outcome.aggregate_utility,
        latency_seconds=time.time() - start,
        proposal_method=proposal.method.value,
        proposal_cost=proposal.cost_normalized,
        proposal_time=proposal.time_days,
        proposal_new_agent=proposal.new_agent_id,
        n_accepting=outcome.n_individually_accepting,
    )
