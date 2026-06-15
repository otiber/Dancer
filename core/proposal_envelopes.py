"""
core/proposal_envelopes.py — Method-grounded feasibility envelopes (audit F12).

WHY: proposal cost_normalized and time_days were self-declared free variables
with no coupling to the recovery method. Once LLM sampling was fixed (v2),
all LLM systems discovered the degenerate optimum — near-zero declared cost
("Express Air Freight at cost 0.05") — yielding instant round-0 consensus and
meaningless utility comparisons (observed committed cost means: 0.05–0.08).

These envelopes implement the Deterministic Safeguard Layer's ontology bounds
(paper §4.1: proposals "containing out-of-bound numeric values ... are
discarded"). They encode the basic economics of each recovery action — fast
is expensive, cheap is slow — and are PUBLIC market knowledge: the same
envelope table is shown to every LLM system (DANCER, LLM-Orchestra,
ReAct-MAS) and enforced identically on all of them, so no system gains an
informational edge.

All fixed baseline proposals (BPMN+ handler, MAPE-K KB, CNP/ζ-CNP task
schemas, MockLLM) lie inside these envelopes by construction — verified in
tests; any future constant must respect them too.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from core.models import Proposal

# method -> ((cost_lo, cost_hi), (time_lo_days, time_hi_days))
ENVELOPES: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "Express_Air_Freight":   ((0.55, 0.95), (0.5, 2.0)),
    "Expedited_Ground":      ((0.30, 0.65), (1.0, 3.0)),
    "Consolidated_Rail":     ((0.25, 0.60), (2.0, 6.0)),
    "Land_Transport":        ((0.30, 0.60), (2.0, 5.0)),
    "Third_Party_Logistics": ((0.30, 0.70), (2.0, 6.0)),
    "Alternative_Port_B":    ((0.30, 0.70), (3.0, 8.0)),
    "Standard_Shipping":     ((0.10, 0.60), (3.5, 10.0)),
}


def is_feasible(proposal: Proposal) -> bool:
    """True iff (cost, time) lie inside the method's envelope."""
    env = ENVELOPES.get(proposal.method.value)
    if env is None:
        return False
    (c_lo, c_hi), (t_lo, t_hi) = env
    return (c_lo <= proposal.cost_normalized <= c_hi
            and t_lo <= proposal.time_days <= t_hi)


def infeasibility_reason(proposal: Proposal) -> Optional[str]:
    """None if feasible; otherwise a short diagnostic string."""
    env = ENVELOPES.get(proposal.method.value)
    if env is None:
        return f"unknown method {proposal.method.value!r}"
    (c_lo, c_hi), (t_lo, t_hi) = env
    if not (c_lo <= proposal.cost_normalized <= c_hi):
        return (f"cost {proposal.cost_normalized:.2f} outside "
                f"[{c_lo:.2f}, {c_hi:.2f}] for {proposal.method.value}")
    if not (t_lo <= proposal.time_days <= t_hi):
        return (f"time {proposal.time_days:.1f}d outside "
                f"[{t_lo:.1f}, {t_hi:.1f}]d for {proposal.method.value}")
    return None


def envelope_prompt_block() -> str:
    """Public 'market price book' block, identical for every LLM system."""
    lines = ["MARKET FEASIBILITY ENVELOPES (public; proposals outside these "
             "bounds are REJECTED as economically infeasible):"]
    for m, ((cl, ch), (tl, th)) in ENVELOPES.items():
        lines.append(f"  {m:<23} cost in [{cl:.2f}, {ch:.2f}]   "
                     f"time in [{tl:.1f}, {th:.1f}] days")
    lines.append("Pick values INSIDE these ranges; note the trade-off — "
                 "faster methods cost more, cheaper methods take longer.")
    return "\n".join(lines)
