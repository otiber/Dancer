"""Real LLM integration for DANCER using OpenRouter (OpenAI-compatible API).

Also supports native OpenAI and Anthropic keys via env vars.

Environment variables (set ONE group):
  OPENROUTER_API_KEY    — OpenRouter key  (default provider)
  OPENAI_API_KEY        — direct OpenAI
  ANTHROPIC_API_KEY     — direct Anthropic (via openai-compat shim on OR)

  DANCER_MODEL          — model slug, e.g. "openai/gpt-4o-mini" (default)
                          Other good choices:
                            "anthropic/claude-3-haiku"
                            "meta-llama/llama-3.1-8b-instruct:free"
                            "google/gemini-flash-1.5"
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from core.models import Proposal, ShippingMethod

_VALID_METHODS = {m.value for m in ShippingMethod}

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an Intelligent Process Agent (IPA) participating in the DANCER \
decentralized negotiation protocol for supply-chain choreography.

Your ONLY job is to output a JSON array of logistics proposals — no prose, \
no markdown fences, no extra keys. Just the raw JSON array.

Each proposal must match this exact schema:
{
  "id":                 "<string, unique ID>",
  "method":             "<one of the allowed values below>",
  "cost_normalized":    <float 0.0–1.0  (0=free, 1=max budget consumed)>,
  "time_days":          <positive float>,
  "description":        "<one short sentence>",
  "requires_new_agent": <true | false>,
  "new_agent_id":       "<string | null>  (use "3PL_Agent" when true)"
}

Allowed method values:
  Standard_Shipping | Express_Air_Freight | Consolidated_Rail |
  Land_Transport | Third_Party_Logistics | Alternative_Port_B | Expedited_Ground

Hard rules:
  - cost_normalized MUST be in [0.0, 1.0]
  - time_days MUST be > 0
  - If requires_new_agent is true → new_agent_id must be "3PL_Agent"
  - Output 2–3 proposals for initial generation, 1–2 for refinement rounds
"""


# ── JSON extraction helpers ───────────────────────────────────────────────────

def _extract_json_array(text: str) -> List[dict]:
    """Robustly extract a JSON array from raw LLM output."""
    text = text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # 1. Try the whole string as an array
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # 2. Try as an object with a "proposals" key  {"proposals": [...]}
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            for key in ("proposals", "data", "results"):
                if isinstance(obj.get(key), list):
                    return obj[key]
        except json.JSONDecodeError:
            pass

    # 3. Find first [...] substring
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return []


def _parse_proposals(text: str, id_prefix: str = "p") -> List[Proposal]:
    """Convert raw LLM text into validated Proposal objects."""
    raw_list = _extract_json_array(text)
    proposals: List[Proposal] = []
    for i, item in enumerate(raw_list):
        try:
            item["id"] = f"{id_prefix}_{i + 1}"
            if item.get("method") not in _VALID_METHODS:
                item["method"] = "Standard_Shipping"
            item["cost_normalized"] = max(0.0, min(1.0, float(item.get("cost_normalized", 0.5))))
            item["time_days"] = max(0.1, float(item.get("time_days", 3.0)))
            item.setdefault("requires_new_agent", False)
            item.setdefault("new_agent_id", None)
            if item["requires_new_agent"] and not item["new_agent_id"]:
                item["new_agent_id"] = "3PL_Agent"
            proposals.append(Proposal(**item))
        except Exception:
            continue
    return proposals


# ── RealLLM class ─────────────────────────────────────────────────────────────

class RealLLM:
    """Calls a real LLM via OpenRouter (or direct OpenAI/Anthropic)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        # Resolve provider
        self.model = model or os.environ.get("DANCER_MODEL", "openai/gpt-4o-mini")

        if base_url:
            resolved_base = base_url
            resolved_key = api_key or ""
        elif os.environ.get("DANCER_BASE_URL"):
            resolved_base = os.environ["DANCER_BASE_URL"]
            resolved_key = api_key or os.environ.get("DANCER_API_KEY", "dummy")
        elif os.environ.get("OPENROUTER_API_KEY"):
            resolved_base = "https://openrouter.ai/api/v1"
            resolved_key = api_key or os.environ["OPENROUTER_API_KEY"]
        elif os.environ.get("OPENAI_API_KEY"):
            resolved_base = "https://api.openai.com/v1"
            resolved_key = api_key or os.environ["OPENAI_API_KEY"]
            # Strip provider prefix for direct OpenAI
            if "/" in self.model:
                self.model = self.model.split("/", 1)[1]
        else:
            raise EnvironmentError(
                "No API key found. Set OPENROUTER_API_KEY, OPENAI_API_KEY, "
                "or pass api_key= explicitly."
            )

        self.client = OpenAI(api_key=resolved_key, base_url=resolved_base)
        self.total_tokens: int = 0
        self._use_json_mode = True   # disabled automatically on first failure

    # ── Internal call ─────────────────────────────────────────────────────────

    def _call(self, user_prompt: str) -> Tuple[str, int]:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 900,
        }
        if self._use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # json_object mode not supported by this model — retry without
            if self._use_json_mode and ("response_format" in str(exc).lower()
                                        or "json" in str(exc).lower()):
                self._use_json_mode = False
                kwargs.pop("response_format", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise

        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else len(text) // 4
        self.total_tokens += tokens
        return text, tokens

    # ── Public interface (mirrors MockLLM) ───────────────────────────────────

    def generate_proposals(
        self,
        disruption_type: str,
        goal: str,
        available_agents: List[str],
    ) -> Tuple[List[Proposal], int]:
        guidance = {
            "structural_failure": (
                "The primary carrier has been removed (port strike). Propose at least one "
                "option using Third_Party_Logistics with requires_new_agent=true and "
                "new_agent_id='3PL_Agent'."
            ),
            "parametric_fluctuation": (
                "A 24-hour warehouse delay occurred. Propose faster shipping to recover "
                "the lost time."
            ),
            "sovereignty_conflict": (
                "The manufacturer has tightened its cost budget mid-process. "
                "Propose options spanning the cost/time trade-off space."
            ),
        }.get(disruption_type, "Resolve the supply-chain disruption.")

        prompt = f"""Disruption: {disruption_type}
Available agents: {', '.join(available_agents)}
Goal: {goal}

Scenario guidance: {guidance}

Output ONLY a JSON array of 2–3 proposal objects."""

        text, tokens = self._call(prompt)
        proposals = _parse_proposals(text, id_prefix="p")
        return proposals, tokens

    def refine_proposals(
        self,
        previous_proposals: List[Proposal],
        rejection_reasons: List[Dict],
        round_num: int,
        disruption_type: str,
    ) -> Tuple[List[Proposal], int]:
        prev_json = json.dumps(
            [p.model_dump(mode="json") for p in previous_proposals], indent=2
        )
        reasons_json = json.dumps(rejection_reasons, indent=2)

        prompt = f"""CONTRASTIVE REFINEMENT — Round {round_num}

Disruption: {disruption_type}

Proposals that FAILED consensus:
{prev_json}

Rejection reasons (from agents who rejected the best proposal):
{reasons_json}

Instructions:
  - "cost" rejections   → reduce cost_normalized (target < 0.50)
  - "time" rejections   → reduce time_days
  - "availability" rejections → switch shipping method or use Third_Party_Logistics
  - For structural_failure: always set requires_new_agent=true, new_agent_id="3PL_Agent"
  - Generate 1–2 NEW proposals that bypass the specific constraints above.

Output ONLY a JSON array of new proposal objects."""

        text, tokens = self._call(prompt)
        proposals = _parse_proposals(text, id_prefix=f"r{round_num}")
        return proposals, tokens


class HeuristicPolicyGenerator:
    """Ablation baseline isolating CONTRASTIVE REFINEMENT (audit F7).

    GENERATION is identical to the full system: if a production endpoint is
    configured (DANCER_BASE_URL / OPENROUTER_API_KEY / OPENAI_API_KEY), the
    same ProductionLLM with the same privacy-preserving prompts produces the
    round-0 proposals; otherwise MockLLM is used (dev/CI). This guarantees
    that any performance difference vs. the full system is attributable to
    the refinement operator, not to different starting candidates.

    REFINEMENT ignores rejection vectors entirely and applies a blind ±20%
    random perturbation to cost_normalized and time_days of every previous
    proposal (no LLM call, zero tokens). If DANCER-LLM converges in fewer
    rounds and at higher utility than DANCER-Heuristic on PAIRED episodes,
    the difference is attributable solely to the structured use of rejection
    feedback.
    """

    _MUTATION = 0.20

    def __init__(self) -> None:
        self.total_tokens: int = 0
        self._inner = None          # lazily resolved generation backend
        self._ctx_kwargs: dict = {}

    # Context passthrough so the protocol's privacy contract reaches the
    # inner generation backend unchanged.
    def set_context(self, **kwargs) -> None:
        self._ctx_kwargs = kwargs
        if self._inner is not None and hasattr(self._inner, "set_context"):
            self._inner.set_context(**kwargs)

    def _generation_backend(self):
        if self._inner is None:
            has_endpoint = any(os.environ.get(k) for k in
                               ("DANCER_BASE_URL", "OPENROUTER_API_KEY",
                                "OPENAI_API_KEY"))
            if has_endpoint:
                from llm.prompts import ProductionLLM
                self._inner = ProductionLLM()
            else:
                from llm.llm_mock import MockLLM
                self._inner = MockLLM()
            if self._ctx_kwargs and hasattr(self._inner, "set_context"):
                self._inner.set_context(**self._ctx_kwargs)
        return self._inner

    def generate_proposals(
        self,
        disruption_type: str,
        goal: str,
        available_agents: List[str],
    ) -> Tuple[List[Proposal], int]:
        backend = self._generation_backend()
        proposals, tokens = backend.generate_proposals(
            disruption_type, goal, available_agents)
        self.total_tokens += tokens
        return proposals, tokens

    def refine_proposals(
        self,
        previous_proposals: List[Proposal],
        rejection_reasons: List[Dict],  # accepted for interface parity; never read
        round_num: int,
        disruption_type: str,
    ) -> Tuple[List[Proposal], int]:
        tokens = 0   # no LLM call — refinement is blind

        mutated: List[Proposal] = []
        for i, p in enumerate(previous_proposals):
            factor_cost = random.uniform(1.0 - self._MUTATION, 1.0 + self._MUTATION)
            factor_time = random.uniform(1.0 - self._MUTATION, 1.0 + self._MUTATION)
            mutated.append(
                Proposal(
                    id=f"h_r{round_num}_{i + 1}",
                    method=p.method,
                    cost_normalized=round(max(0.01, min(1.0, p.cost_normalized * factor_cost)), 4),
                    time_days=round(max(0.1, p.time_days * factor_time), 2),
                    description=f"Blind ±20% mutation of {p.id}",
                    requires_new_agent=p.requires_new_agent,
                    new_agent_id=p.new_agent_id,
                )
            )
        return mutated, tokens
