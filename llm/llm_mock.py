"""Mock LLM policy generator (π_θ) with Contrastive Refinement.

Replaces a real LLM call with deterministic-but-stochastic heuristics so the
simulation is self-contained. Swap generate_proposals / refine_proposals for
real OpenAI/Anthropic calls to run with actual models.
"""
from __future__ import annotations
import random
from typing import List, Dict, Tuple

from core.models import Proposal, ShippingMethod

# Approximate token cost per operation (simulated)
_TOKENS_GENERATE = 350
_TOKENS_REFINE = 280


class MockLLM:
    """Simulates LLM proposal generation and Contrastive Refinement."""

    def __init__(self) -> None:
        self.total_tokens: int = 0

    # ------------------------------------------------------------------ #
    #  Initial proposal generation                                         #
    # ------------------------------------------------------------------ #
    def generate_proposals(
        self,
        disruption_type: str,
        goal: str,
        available_agents: List[str],
    ) -> Tuple[List[Proposal], int]:
        tokens = _TOKENS_GENERATE + random.randint(-40, 40)
        self.total_tokens += tokens

        if disruption_type == "structural_failure":
            proposals = [
                Proposal(
                    id="p1",
                    method=ShippingMethod.EXPRESS_AIR,
                    cost_normalized=0.88,
                    time_days=1.0,
                    description="Express air freight via alternative hub",
                ),
                Proposal(
                    id="p2",
                    method=ShippingMethod.ALTERNATIVE_PORT,
                    cost_normalized=0.65,
                    time_days=3.0,   # inside Alternative_Port_B envelope
                    description="Reroute via Alternative Port B with rail leg",
                    requires_new_agent=True,
                    new_agent_id="3PL_Agent",
                ),
            ]

        elif disruption_type == "parametric_fluctuation":
            proposals = [
                Proposal(
                    id="p1",
                    method=ShippingMethod.EXPEDITED_GROUND,
                    cost_normalized=0.55,
                    time_days=2.0,
                    description="Expedited ground to recover warehouse delay",
                ),
                Proposal(
                    id="p2",
                    method=ShippingMethod.STANDARD,
                    cost_normalized=0.30,
                    time_days=5.0,
                    description="Standard shipping with adjusted delivery window",
                ),
            ]

        elif disruption_type == "sovereignty_conflict":
            # Manufacturer demands 20% cost cut → first attempt is Express Air (rejected on cost)
            proposals = [
                Proposal(
                    id="p1",
                    method=ShippingMethod.EXPRESS_AIR,
                    cost_normalized=0.88,
                    time_days=1.0,
                    description="Express delivery to meet renegotiated terms",
                ),
            ]

        else:
            proposals = []

        return proposals, tokens

    # ------------------------------------------------------------------ #
    #  Contrastive Refinement                                              #
    # ------------------------------------------------------------------ #
    def refine_proposals(
        self,
        previous_proposals: List[Proposal],
        rejection_reasons: List[Dict],
        round_num: int,
        disruption_type: str,
    ) -> Tuple[List[Proposal], int]:
        """Reason-Aware Revision: inspect rejection JSON and generate targeted alternatives."""
        tokens = _TOKENS_REFINE + random.randint(-30, 30)
        self.total_tokens += tokens

        reasons_text = " ".join(r.get("reason", "") for r in rejection_reasons).lower()
        cost_issue = "cost" in reasons_text
        time_issue = "time" in reasons_text

        refined: List[Proposal] = []

        # ── Structural failure (Port Strike) ──────────────────────────
        if disruption_type == "structural_failure":
            if cost_issue and not time_issue:
                refined += [
                    Proposal(
                        id=f"p_r{round_num}_1",
                        method=ShippingMethod.CONSOLIDATED_RAIL,
                        cost_normalized=0.45,
                        time_days=3.0,
                        description="Consolidated Rail via 3PL – cost optimised",
                        requires_new_agent=True,
                        new_agent_id="3PL_Agent",
                    ),
                    Proposal(
                        id=f"p_r{round_num}_2",
                        method=ShippingMethod.LAND_TRANSPORT,
                        cost_normalized=0.50,
                        time_days=4.0,
                        description="Land transport via 3PL – balanced cost/time",
                        requires_new_agent=True,
                        new_agent_id="3PL_Agent",
                    ),
                ]
            elif time_issue and not cost_issue:
                refined.append(
                    Proposal(
                        id=f"p_r{round_num}_1",
                        method=ShippingMethod.EXPEDITED_GROUND,
                        cost_normalized=0.65,
                        time_days=2.0,
                        description="Expedited ground via 3PL hub",
                        requires_new_agent=True,
                        new_agent_id="3PL_Agent",
                    )
                )
            else:
                # Both cost and time issues or general → progressive cost reduction
                cost_f = max(0.35, 0.70 - round_num * 0.07)
                time_f = max(2.0, 4.5 - round_num * 0.4)
                refined.append(
                    Proposal(
                        id=f"p_r{round_num}_1",
                        method=ShippingMethod.THIRD_PARTY_3PL,
                        cost_normalized=round(cost_f, 3),
                        time_days=round(time_f, 1),
                        description=f"3PL negotiated package (round-{round_num} refinement)",
                        requires_new_agent=True,
                        new_agent_id="3PL_Agent",
                    )
                )

        # ── Sovereignty conflict (cost cut demand) ────────────────────
        elif disruption_type == "sovereignty_conflict":
            if round_num == 1:
                refined.append(
                    Proposal(
                        id=f"p_r{round_num}_1",
                        method=ShippingMethod.CONSOLIDATED_RAIL,
                        cost_normalized=0.42,
                        time_days=3.0,
                        description="Consolidated Rail – 20% cost reduction via bulk booking",
                    )
                )
            else:
                refined.append(
                    Proposal(
                        id=f"p_r{round_num}_1",
                        method=ShippingMethod.LAND_TRANSPORT,
                        cost_normalized=0.36,
                        time_days=3.5,
                        description="Optimised land transport – maximum cost reduction path",
                    )
                )

        # ── Parametric fluctuation ────────────────────────────────────
        elif disruption_type == "parametric_fluctuation":
            refined.append(
                Proposal(
                    id=f"p_r{round_num}_1",
                    method=ShippingMethod.EXPEDITED_GROUND,
                    cost_normalized=0.50,
                    time_days=2.0,
                    description="Expedited ground – time-recovery after warehouse delay",
                )
            )

        return refined, tokens
