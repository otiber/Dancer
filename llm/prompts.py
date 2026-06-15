"""
Domain-agnostic LLM prompts for DANCER — PRIVACY-PRESERVING revision.

AUDIT FIX F2 (2026-06): the original prompt builders placed EVERY agent's
private constraint vector θ_k (budget_limit, max_duration, availability,
weights) into the Lead's LLM context, contradicting the paper's Dec-POSG
information-asymmetry claim (O_k excludes peer θ_j) and erasing the
informational distinction from the LLM-Orchestra baseline.

The Cognitive Layer now receives ONLY:
  (i)   the Lead agent's OWN constraint vector (it owns it);
  (ii)  public registry capability adverts (discoverable by design, §4.1 G);
  (iii) at refinement time, the structured rejection vectors V — per-rejecting-
        agent dimension scores f_res/f_time/f_pol (Definition 4's normalized
        margins). These are the protocol's DESIGNED inter-round channel and
        never contain raw peer parameters.

Zero hardcoded domain vocabulary — everything derived from DomainContext.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                     InternalServerError, OpenAI)
from core.models import Proposal, ShippingMethod

_VALID_METHODS = {m.value for m in ShippingMethod}


def _get_client() -> Tuple[OpenAI, str]:
    if os.environ.get("DANCER_BASE_URL"):
        client = OpenAI(api_key=os.environ.get("DANCER_API_KEY", "dummy"),
                        base_url=os.environ["DANCER_BASE_URL"])
    elif os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                        base_url="https://openrouter.ai/api/v1")
    elif os.environ.get("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        raise EnvironmentError("Set DANCER_BASE_URL, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    return client, os.environ.get("DANCER_MODEL", "openai/gpt-4o-mini")


# ── Synthetic fallback context (supply-chain scenarios A–G) ───────────────────
class SupplyChainContext:
    domain_name       = "Supply-Chain Choreography (Synthetic)"
    process_token     = "shipment"
    agent_descriptor  = "logistics partner"
    lead_agent_role   = "manufacturer (cost gatekeeper)"
    time_agent_role   = "warehouse (throughput node)"
    resource_agent_role = "carrier (resource-constrained)"
    cost_unit         = "normalised [0–1]"
    avg_throughput_days = 4.0

    disruption_descriptions = {
        "parametric_fluctuation":
            "A 24-hour warehouse processing delay has stalled the shipment. "
            "Find a faster routing.",
        "structural_failure":
            "The primary carrier is unavailable — no carrier agent is active. "
            "Propose alternatives; a third-party provider may be discoverable "
            "in the registry (requires_new_agent=true).",
        "sovereignty_conflict":
            "The lead organization has cut its cost budget mid-process. "
            "The previously planned routing violates the new budget. "
            "Find a cheaper route.",
        "cascading":
            "The primary carrier is unavailable AND the lead's budget was "
            "tightened simultaneously. The proposal must be cheap AND use a "
            "registry provider (requires_new_agent=true).",
        "multi_constraint":
            "All three utility dimensions are near their limits simultaneously. "
            "Near-infeasible constraint space — favour balanced proposals.",
    }

    # NOTE (audit F10): action descriptions must contain NO concrete numbers —
    # numeric hints act as anchors that collapse LLM outputs onto constant
    # (cost, time) values across episodes, destroying proposal diversity.
    recovery_actions = [
        "Standard_Shipping      — lowest cost, longest duration",
        "Expedited_Ground       — fast ground transport, moderate cost",
        "Express_Air_Freight    — fastest option, highest cost",
        "Consolidated_Rail      — balanced cost and duration",
        "Land_Transport         — flexible road routing, moderate cost",
        "Third_Party_Logistics  — outsource to a registry agent (requires_new_agent=true)",
        "Alternative_Port_B     — bypass the disrupted primary port",
    ]

    structural_recovery = (
        "Engage a registry agent: set requires_new_agent=true and use its agent_id."
    )


# ── Prompt building blocks ─────────────────────────────────────────────────────

def _lead_constraints_block(lead: Optional[dict]) -> str:
    """The Lead agent's OWN constraints — the only private θ the LLM may see."""
    if not lead:
        return "  (no lead constraints available)"
    w    = lead.get("weights", {})
    fres = lead.get("committed_f_res", lead.get("availability", 1.0))
    mc   = lead.get("max_cost", 1.0)
    return (
        f"  YOU are {lead['agent_id']} (role: {lead.get('role', 'lead')}).\n"
        f"  Your OWN constraints (private — peers cannot see these, and you "
        f"cannot see theirs):\n"
        f"    budget_limit={mc:.2f}   "
        f"max_duration={lead.get('max_delay_days', 10.0):.1f} days   "
        f"availability={fres:.2f}\n"
        f"    utility_weights: w_res={w.get('w_res', 0.4):.2f}  "
        f"w_time={w.get('w_time', 0.4):.2f}  w_pol={w.get('w_pol', 0.2):.2f}\n"
        f"    Your own f_pol reaches 0 if cost_normalized > {mc * 1.5:.3f}."
    )


def _registry_block(registry: List[dict], ctx) -> str:
    """Public capability adverts of discoverable registry agents (§4.1, G)."""
    if not registry:
        return "  (registry empty — no external providers discoverable)"
    lines = []
    for r in registry:
        lines.append(
            f"  • {r['agent_id']}  (role: {r.get('role', ctx.agent_descriptor)}) — "
            f"advertised capability: handles tasks up to "
            f"{r.get('max_delay_days', 5.0):.1f} days; "
            f"availability={r.get('availability', 0.9):.2f}. "
            f"Engage via requires_new_agent=true, new_agent_id='{r['agent_id']}'."
        )
    return "\n".join(lines)


def _peer_visibility_note(n_peers: int) -> str:
    return (
        f"There are {n_peers} peer agents with PRIVATE constraints you cannot "
        f"observe. You will learn about their feasibility ONLY through the "
        f"structured rejection vectors returned after each round. Therefore "
        f"generate proposals SPANNING the cost/time trade-off space, not a "
        f"single point."
    )


def _round_strategy(round_num: int, tau: float, tau_min: float) -> str:
    if round_num == 1:
        return "ROUND 1 — make targeted adjustments to the binding constraint dimension."
    if round_num == 2:
        return ("ROUND 2 — be more aggressive: move the binding dimension ≥20%, "
                "or switch to a completely different recovery action.")
    return (f"ROUND {round_num} — NEAR ESCALATION (τ={tau:.3f}, min={tau_min:.2f}). "
            "Generate the most broadly feasible proposal possible.")


def build_generation_prompt(
    agent_dicts: List[dict],          # [lead] ONLY — enforced by the protocol
    disruption_type: str,
    goal: str,
    available_agents: List[str],
    ctx,
    registry: Optional[List[dict]] = None,
    n_peers: int = 0,
) -> Tuple[str, str]:
    lead = agent_dicts[0] if agent_dicts else None
    registry = registry or []

    from core.proposal_envelopes import envelope_prompt_block
    envelope_block = envelope_prompt_block()
    disruption_desc = ctx.disruption_descriptions.get(
        disruption_type,
        "A process disruption has occurred. The case token cannot proceed normally."
    )
    recovery_block = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(ctx.recovery_actions))

    structural_note = ""
    if disruption_type in ("structural_failure", "cascading"):
        structural_note = (
            f"\nSTRUCTURAL RECOVERY — MANDATORY: {ctx.structural_recovery} "
            "The disrupted capability is VACANT: every proposal MUST set "
            "requires_new_agent=true with a registry agent_id, or it will be "
            "discarded by the safeguard.\n"
        )

    system = f"""\
You are the Cognitive Layer (π_θ) of the DANCER decentralised negotiation protocol.
Domain: {ctx.domain_name}
Process token: {ctx.process_token}

Each agent k evaluates proposals with a PRIVATE utility
  L_k(p) = w_res·f_res + w_time·f_time + w_pol·f_pol
where f_time degrades with time_days relative to that agent's max_duration and
f_pol degrades with cost_normalized relative to that agent's budget_limit.
You only know YOUR OWN parameters.

━━━ YOUR OWN CONSTRAINTS (Lead Agent) ━━━
{_lead_constraints_block(lead)}

━━━ PEERS ━━━
{_peer_visibility_note(n_peers)}

━━━ DISCOVERABLE REGISTRY (public adverts) ━━━
{_registry_block(registry, ctx)}

━━━ DISRUPTION ━━━
{disruption_desc}
{structural_note}
━━━ AVAILABLE RECOVERY ACTIONS ━━━
{recovery_block}

━━━ {envelope_block} ━━━

━━━ OUTPUT — raw JSON array only, no prose ━━━
[{{
  "id":                 "<unique string>",
  "action":             "<one of the recovery actions listed above>",
  "cost_normalized":    <float in [0.0, 1.0]>,
  "time_days":          <positive float>,
  "description":        "<one sentence stating the cost/time trade-off this proposal targets>",
  "requires_new_agent": <true | false>,
  "new_agent_id":       "<registry agent ID if true, else null>"
}}]

Rules:
  - cost_normalized MUST be in [0.0, 1.0]; time_days MUST be > 0
  - Generate 2–3 proposals spanning DIFFERENT cost/time trade-offs (peers'
    budgets are unknown — diversity is your only hedge)
  - Stay within your OWN budget_limit and max_duration where possible
"""
    user = (f"Active agents (IDs only — constraints private): "
            f"{', '.join(available_agents)}\n"
            f"Goal: {goal}\nDisruption: {disruption_type}\n\nGenerate proposals now.")
    return system, user


def build_refinement_prompt(
    agent_dicts: List[dict],          # [lead] ONLY
    disruption_type: str,
    previous_proposals: List[Proposal],
    rejection_reasons: List[Dict],
    round_num: int,
    tau_current: float,
    tau_min: float,
    ctx,
    registry: Optional[List[dict]] = None,
    n_peers: int = 0,
) -> Tuple[str, str]:
    from core.proposal_envelopes import envelope_prompt_block
    envelope_block = envelope_prompt_block()
    lead = agent_dicts[0] if agent_dicts else None
    registry = registry or []
    strategy = _round_strategy(round_num, tau_current, tau_min)

    system = f"""\
You are the Cognitive Layer (π_θ) performing CONTRASTIVE REFINEMENT in DANCER.
Domain: {ctx.domain_name}  |  Process token: {ctx.process_token}

You CANNOT see peers' constraint parameters. You CAN see their structured
rejection vectors below: per-dimension scores f_res/f_time/f_pol in [0,1].
Identify the BINDING dimension (lowest f_*) per rejecting agent and generate
proposals that move it:

  f_res low  → resource/availability problem → switch action type or engage a registry agent
  f_time low → duration too long for that agent → reduce time_days
  f_pol low  → cost too high for that agent's (unknown) budget → reduce cost_normalized
               substantially; the score tells you HOW FAR you are, not the budget itself

━━━ YOUR OWN CONSTRAINTS (Lead Agent) ━━━
{_lead_constraints_block(lead)}

━━━ DISCOVERABLE REGISTRY (public adverts) ━━━
{_registry_block(registry, ctx)}

━━━ {envelope_block} ━━━

━━━ STRATEGY ━━━
{strategy}
{('STRUCTURAL RECOVERY — MANDATORY: every proposal MUST set requires_new_agent=true with a registry agent_id, or it will be discarded.' if disruption_type in ('structural_failure', 'cascading') else '')}

━━━ OUTPUT — raw JSON array of 1–2 NEW proposals ━━━
Same schema as generation. The description must state which dimension this
proposal targets and why, referencing the rejection vectors above.

Rules:
  - no repeated IDs; cost_normalized in [0,1]; time_days > 0
  - derive cost_normalized and time_days from THIS episode's rejection
    vectors and your own constraints — never reuse numbers from these
    instructions or from typical/round values; vary magnitudes between
    proposals
"""

    # ── Structured rejection vectors V (Definition 4) — the designed channel ──
    lines = []
    for r in rejection_reasons:
        agent  = r.get("agent_id", "?")
        reason = r.get("reason", "")
        uv     = r.get("utility_vector", {})
        line   = f"  • {agent}: {reason}"
        if uv and all(isinstance(uv.get(k), float) for k in ["f_res", "f_time", "f_pol"]):
            f_res, f_time, f_pol = uv["f_res"], uv["f_time"], uv["f_pol"]
            binding = min(
                [("resource(f_res)", f_res),
                 ("time(f_time)",    f_time),
                 ("cost/policy(f_pol)", f_pol)],
                key=lambda x: x[1])
            line += (f"\n      f_res={f_res:.3f}  f_time={f_time:.3f}  f_pol={f_pol:.3f}"
                     f"\n      ▶ BINDING: {binding[0]} = {binding[1]:.3f}")
        elif r.get("utility") is not None:
            line += f"  [score={r['utility']:.3f}]"
        lines.append(line)
    rejections_block = "\n".join(lines) or "  No structured rejections — aggregate below τ."

    prev_block = json.dumps(
        [{"id": p.id,
          "action": getattr(p, "action", p.method.value),
          "cost_normalized": p.cost_normalized,
          "time_days": p.time_days}
         for p in previous_proposals], indent=2)

    user = (f"=== CONTRASTIVE REFINEMENT — Round {round_num} ===\n"
            f"Domain: {ctx.domain_name}\nDisruption: {disruption_type}\n"
            f"Peers with private constraints: {n_peers}\n\n"
            f"FAILED proposals:\n{prev_block}\n\n"
            f"STRUCTURED REJECTION VECTORS:\n{rejections_block}\n\n"
            "Generate refined proposals targeting the binding dimensions above.")
    return system, user


# ── Proposal parser ────────────────────────────────────────────────────────────

def _map_to_shipping_method(action: str) -> str:
    s = action.lower()
    if any(k in s for k in ["air", "express", "expedit"]): return "Express_Air_Freight"
    if any(k in s for k in ["rail", "train", "consolidat"]): return "Consolidated_Rail"
    if any(k in s for k in ["3pl", "third", "outsourc", "partner", "agency", "registry",
                            "redirect", "reroute", "deputy", "reassign", "overflow",
                            "transfer", "alternative", "telemed", "cohort", "instalment",
                            "collection"]):
        return "Third_Party_Logistics"
    if any(k in s for k in ["land", "road", "ground", "truck"]): return "Land_Transport"
    if any(k in s for k in ["port", "harbour", "sea"]): return "Alternative_Port_B"
    if any(k in s for k in ["fast", "urgent", "priority"]): return "Expedited_Ground"
    return "Standard_Shipping"


def _parse_proposals(text: str, id_prefix: str = "p") -> List[Proposal]:
    text = re.sub(r"```(?:json)?", "", text.strip()).strip()
    raw_list = []
    for candidate in [text, re.search(r"\[.*\]", text, re.DOTALL)]:
        if candidate is None:
            continue
        s = candidate if isinstance(candidate, str) else candidate.group()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                raw_list = parsed
            elif isinstance(parsed, dict):
                # response_format=json_object forces a top-level OBJECT, so
                # compliant models wrap the requested array, e.g.
                # {"proposals": [...]}. Unwrap the first list-valued field;
                # fall back to treating the dict as a single proposal.
                nested = next((v for v in parsed.values() if isinstance(v, list)), None)
                raw_list = nested if nested is not None else [parsed]
            else:
                raw_list = []
            break
        except json.JSONDecodeError:
            continue

    proposals = []
    for i, item in enumerate(raw_list):
        try:
            action_str = item.get("action") or item.get("method") or "Standard_Shipping"
            method     = _map_to_shipping_method(action_str)
            cost       = max(0.0, min(1.0, float(item.get("cost_normalized", 0.5))))
            timed      = max(0.1, float(item.get("time_days", 3.0)))
            req        = bool(item.get("requires_new_agent", False))
            nid        = item.get("new_agent_id")
            if req and not nid:
                nid = "3PL_Agent"
            desc       = str(item.get("description", f"Proposal {id_prefix}_{i+1}"))
            p = Proposal(
                id=f"{id_prefix}_{i+1}",
                method=ShippingMethod(method),
                cost_normalized=cost, time_days=timed,
                description=desc, requires_new_agent=req, new_agent_id=nid,
            )
            p.__dict__["action"] = action_str   # preserve original for traceability
            proposals.append(p)
        except Exception:
            continue
    return proposals


# ── ProductionLLM ─────────────────────────────────────────────────────────────

class ProductionLLM:
    """Domain-agnostic LLM backend with the privacy-preserving context contract."""

    def __init__(self) -> None:
        self._client: Optional[OpenAI] = None
        self._model:  Optional[str]    = None
        self.total_tokens: int         = 0
        self._use_json_mode            = True
        self._agent_dicts: List[dict]  = []     # [lead] only, by contract
        self._registry: List[dict]     = []
        self._n_peers: int             = 0
        self._tau_current: float       = 0.75
        self._tau_min:     float       = 0.40
        self._ctx                      = SupplyChainContext()

    def set_context(self, agent_dicts, tau_current=0.75, tau_min=0.40,
                    domain_context=None, registry=None, n_peers=0):
        # Defensive truncation: even if a caller passes the full table,
        # only the first (Lead) entry is ever used in prompts.
        self._agent_dicts = list(agent_dicts)[:1] if agent_dicts else []
        self._tau_current = tau_current
        self._tau_min     = tau_min
        self._registry    = list(registry) if registry else []
        self._n_peers     = n_peers
        self._ctx         = domain_context if domain_context is not None else SupplyChainContext()

    def _ensure_client(self):
        if self._client is None:
            self._client, self._model = _get_client()
        return self._client, self._model

    # Transient server/network failures (connection resets, timeouts, 5xx) —
    # retried with exponential backoff so a brief endpoint hiccup mid-campaign
    # does not abort the whole run (audit: --scaling crashed on a single
    # httpx.ReadError / Connection reset by peer from the vLLM endpoint).
    _TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError)
    _MAX_RETRIES = 4

    def _call(self, system: str, user: str, temperature: float = 0.7) -> Tuple[str, int]:
        client, model = self._ensure_client()
        kwargs = dict(model=model,
                      messages=[{"role": "system", "content": system},
                                {"role": "user",   "content": user}],
                      temperature=temperature, max_tokens=1024)
        if self._use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                if self._use_json_mode and "response_format" in str(exc).lower():
                    self._use_json_mode = False
                    kwargs.pop("response_format", None)
                    continue
                if isinstance(exc, self._TRANSIENT_ERRORS) and attempt < self._MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 16))
                    continue
                if (isinstance(exc, APIStatusError) and exc.status_code >= 500
                        and attempt < self._MAX_RETRIES):
                    time.sleep(min(2 ** attempt, 16))
                    continue
                raise
        text   = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else len(text) // 4
        self.total_tokens += tokens
        return text, tokens

    def generate_proposals(self, disruption_type, goal, available_agents):
        system, user = build_generation_prompt(
            self._agent_dicts, disruption_type, goal, available_agents,
            self._ctx, registry=self._registry, n_peers=self._n_peers)
        text, tokens = self._call(system, user, temperature=0.7)
        return _parse_proposals(text, "p"), tokens

    def refine_proposals(self, previous_proposals, rejection_reasons,
                         round_num, disruption_type):
        system, user = build_refinement_prompt(
            self._agent_dicts, disruption_type, previous_proposals,
            rejection_reasons, round_num, self._tau_current, self._tau_min,
            self._ctx, registry=self._registry, n_peers=self._n_peers)
        text, tokens = self._call(system, user, temperature=0.5)
        return _parse_proposals(text, f"r{round_num}"), tokens
