"""
MAPE-K Autonomic Computing Baseline for IEEE TSC Evaluation.

Reference: Kephart, J.O. & Chess, D.M. (2003). The vision of autonomic computing.
           Computer, 36(1), 41-50. IEEE.
           Huebscher, M.C. & McCann, J.A. (2008). A survey of autonomic computing.
           ACM Computing Surveys, 40(3).

The MAPE-K loop: Monitor → Analyze → Plan → Execute over a shared Knowledge Base.

Design choices for fair comparison:
  - Knowledge Base (KB) pre-populated with structural/sovereignty recovery rules,
    giving MAPE-K the maximum fair advantage on open-world scenarios.
  - KB registry mirrors the DANCER agent registry (same 3PL entry).
  - Plan phase applies heuristic cost optimization (no LLM).
  - Execute phase validates against agent utility functions (same L_k as DANCER).
  - Post-execution KB update models the learning loop (runtime KB enrichment).

Compared to BPMN+:
  + Runtime KB learning; can grow the handler set across episodes.
  + Structural recovery via registry lookup.
  - Planning remains rule-based; cannot synthesize novel paths.

Compared to DANCER:
  - No peer-to-peer negotiation; centralized planning violates sovereignty.
  - No semantic LLM reasoning; KB must enumerate disruption patterns.
  - No bounded contrastive refinement; fixed rule application.
"""
from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from core.models import Proposal, ShippingMethod, RunResult
from core.agents import IPAgent


# ── Knowledge Base ────────────────────────────────────────────────────────────

@dataclass
class KBEntry:
    """A single recovery rule in the Knowledge Base."""
    disruption_type: str
    handler_type: str                 # "parametric" | "structural" | "sovereignty"
    proposal: Optional[dict]          # pre-built Proposal dict (or None if rule-based)
    confidence: float = 1.0
    use_count: int = 0
    success_count: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.use_count)


class KnowledgeBase:
    """Shared KB pre-populated with supply-chain recovery rules.

    Structural entries reference 3PL_Agent explicitly — same registry agent
    as DANCER — so MAPE-K gets the maximum fair advantage on Scenario B.
    """

    def __init__(self, registry: List[dict]) -> None:
        self.registry = {r["agent_id"]: r for r in registry}
        self._entries: List[KBEntry] = [
            # ── Parametric: pre-modeled boundary event ────────────────
            KBEntry(
                disruption_type="parametric_fluctuation",
                handler_type="parametric",
                proposal=dict(
                    id="mapek_param_1",
                    method="Expedited_Ground",
                    cost_normalized=0.45,
                    time_days=2.0,
                    description="MAPE-K: expedited ground to absorb warehouse delay",
                    requires_new_agent=False,
                    new_agent_id=None,
                ),
                confidence=0.95,
            ),
            KBEntry(
                disruption_type="parametric_fluctuation",
                handler_type="parametric",
                proposal=dict(
                    id="mapek_param_2",
                    method="Standard_Shipping",
                    cost_normalized=0.30,
                    time_days=4.0,
                    description="MAPE-K: standard shipping with adjusted SLA",
                    requires_new_agent=False,
                    new_agent_id=None,
                ),
                confidence=0.85,
            ),
            # ── Structural: 3PL registry substitution ────────────────
            KBEntry(
                disruption_type="structural_failure",
                handler_type="structural",
                proposal=dict(
                    id="mapek_struct_1",
                    method="Third_Party_Logistics",
                    cost_normalized=0.55,
                    time_days=3.5,
                    description="MAPE-K: route via KB-registered 3PL fallback",
                    requires_new_agent=True,
                    new_agent_id="3PL_Agent",
                ),
                confidence=0.80,
            ),
            KBEntry(
                disruption_type="structural_failure",
                handler_type="structural",
                proposal=dict(
                    id="mapek_struct_2",
                    method="Consolidated_Rail",
                    cost_normalized=0.48,
                    time_days=4.0,
                    description="MAPE-K: consolidated rail via 3PL hub",
                    requires_new_agent=True,
                    new_agent_id="3PL_Agent",
                ),
                confidence=0.72,
            ),
            # ── Sovereignty: cost-reduction rules ────────────────────
            KBEntry(
                disruption_type="sovereignty_conflict",
                handler_type="sovereignty",
                proposal=dict(
                    id="mapek_sov_1",
                    method="Consolidated_Rail",
                    cost_normalized=0.42,
                    time_days=3.0,
                    description="MAPE-K: consolidated rail for 20% cost reduction",
                    requires_new_agent=False,
                    new_agent_id=None,
                ),
                confidence=0.78,
            ),
            KBEntry(
                disruption_type="sovereignty_conflict",
                handler_type="sovereignty",
                proposal=dict(
                    id="mapek_sov_2",
                    method="Land_Transport",
                    cost_normalized=0.36,
                    time_days=3.5,
                    description="MAPE-K: land transport — maximum cost reduction path",
                    requires_new_agent=False,
                    new_agent_id=None,
                ),
                confidence=0.70,
            ),
        ]

    def query(self, disruption_type: str) -> List[KBEntry]:
        """Return KB entries for this disruption type, sorted by confidence."""
        entries = [e for e in self._entries if e.disruption_type == disruption_type]
        return sorted(entries, key=lambda e: e.confidence, reverse=True)

    def update(self, entry: KBEntry, success: bool) -> None:
        """Post-execution KB update (learning loop)."""
        entry.use_count += 1
        if success:
            entry.success_count += 1
            entry.confidence = min(0.99, entry.confidence * 1.05)
        else:
            entry.confidence = max(0.10, entry.confidence * 0.90)

    def add_entry(self, entry: KBEntry) -> None:
        """Runtime KB enrichment — new patterns added after successful recoveries."""
        self._entries.append(entry)


# ── MAPE-K Loop ───────────────────────────────────────────────────────────────

class MAPEKController:
    """Autonomic controller implementing the full MAPE-K loop."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.analysis_log: List[dict] = []

    # 1. Monitor ──────────────────────────────────────────────────────────────
    def monitor(self, disruption_type: str, active_agents: List[str]) -> dict:
        return {
            "disruption_type": disruption_type,
            "severity": self._assess_severity(disruption_type),
            "active_agents": active_agents,
            "timestamp": time.time(),
        }

    def _assess_severity(self, disruption_type: str) -> str:
        mapping = {
            "parametric_fluctuation": "LOW",
            "structural_failure": "HIGH",
            "sovereignty_conflict": "MEDIUM",
            "cascading": "CRITICAL",
        }
        return mapping.get(disruption_type, "MEDIUM")

    # 2. Analyze ──────────────────────────────────────────────────────────────
    def analyze(self, monitor_report: dict, agents: List[IPAgent]) -> dict:
        dtype = monitor_report["disruption_type"]
        kb_entries = self.kb.query(dtype)
        feasible = len(kb_entries) > 0

        # Check registry for structural failures
        registry_agents = []
        if dtype == "structural_failure":
            registry_agents = list(self.kb.registry.keys())

        analysis = {
            "disruption_type": dtype,
            "kb_entries_found": len(kb_entries),
            "feasible": feasible,
            "registry_agents": registry_agents,
            "num_agents": len(agents),
            "severity": monitor_report["severity"],
        }
        self.analysis_log.append(analysis)
        return analysis

    # 3. Plan ─────────────────────────────────────────────────────────────────
    def plan(self, analysis: dict) -> List[tuple[Proposal, KBEntry]]:
        """Query KB and build ranked plan. Returns (Proposal, KBEntry) pairs."""
        dtype = analysis["disruption_type"]
        entries = self.kb.query(dtype)
        if not entries:
            return []

        plans: List[tuple[Proposal, KBEntry]] = []
        for entry in entries:
            if entry.proposal is None:
                continue
            # Structural: confirm agent is in registry
            if entry.handler_type == "structural":
                aid = entry.proposal.get("new_agent_id")
                if aid and aid not in self.kb.registry:
                    continue
            try:
                p = Proposal(**entry.proposal)
                plans.append((p, entry))
            except Exception:
                continue
        return plans

    # 4. Execute ──────────────────────────────────────────────────────────────
    def execute(self, plans: List[tuple[Proposal, KBEntry]],
                agent_dicts: List[dict], registry: List[dict],
                active_agent_ids: List[str], tau_min: float,
                disruption_type: str = None) -> dict:
        """Try each plan in priority order; commit the first that clears the
        CANONICAL criterion (core/evaluation.py): mean utility over the shared
        agent set >= tau_min — identical to every other system (audit F3;
        previously a fixed 0.55 threshold inconsistent with DANCER's 0.40/0.45).
        """
        from core.evaluation import canonical_outcome
        for proposal, kb_entry in plans:
            outcome = canonical_outcome(proposal, agent_dicts, registry,
                                        tau_min,
                                        active_agent_ids=active_agent_ids,
                                        disruption_type=disruption_type)
            if outcome.success:
                self.kb.update(kb_entry, success=True)
                return {
                    "success": True,
                    "proposal": proposal,
                    "avg_utility": outcome.aggregate_utility,
                    "kb_entry": kb_entry,
                }
            self.kb.update(kb_entry, success=False)

        return {"success": False, "proposal": None, "avg_utility": 0.0, "kb_entry": None}


# ── Public runner ─────────────────────────────────────────────────────────────

def run_mape_k(
    scenario: str,
    agent_dicts: List[dict],
    active_agent_ids: List[str],
    registry: List[dict],
    disruption_type: str,
    iteration: int = 0,
    tau_min: float = 0.45,
) -> RunResult:
    """Execute one MAPE-K autonomic loop episode (canonical outcome, audit F3)."""
    start = time.time()
    msgs = 0

    # Reconstruct agents
    active_set = set(active_agent_ids)
    agents = [IPAgent.from_dict(d) for d in agent_dicts if d["agent_id"] in active_set]

    # Add registry agents for structural failures
    registry_map = {r["agent_id"]: r for r in registry}
    if disruption_type == "structural_failure":
        for aid, rd in registry_map.items():
            if not any(a.agent_id == aid for a in agents):
                agents.append(IPAgent.from_dict(rd))

    kb = KnowledgeBase(registry)
    controller = MAPEKController(kb)

    # MAPE-K loop (1 iteration = 4 messages: M, A, P, E broadcast)
    monitor_report = controller.monitor(disruption_type, active_agent_ids)
    msgs += 1   # Monitor signal

    analysis = controller.analyze(monitor_report, agents)
    msgs += 1   # Analyze broadcast

    if not analysis["feasible"]:
        return RunResult(
            scenario=scenario, system="MAPE-K", iteration=iteration,
            success=False, rounds=1, messages_sent=msgs, total_tokens=0,
            final_utility=0.0, latency_seconds=time.time() - start,
        )

    plans = controller.plan(analysis)
    msgs += 1   # Plan message

    if not plans:
        return RunResult(
            scenario=scenario, system="MAPE-K", iteration=iteration,
            success=False, rounds=1, messages_sent=msgs, total_tokens=0,
            final_utility=0.0, latency_seconds=time.time() - start,
        )

    result = controller.execute(plans, agent_dicts, registry,
                                active_agent_ids, tau_min,
                                disruption_type=disruption_type)
    msgs += len(agents)   # Execute: notification to each agent

    latency = time.time() - start
    p = result.get("proposal")
    return RunResult(
        scenario=scenario, system="MAPE-K", iteration=iteration,
        success=result["success"], rounds=1, messages_sent=msgs, total_tokens=0,
        final_utility=result["avg_utility"] if result["success"] else 0.0,
        latency_seconds=latency,
        proposal_method=p.method.value if p else None,
        proposal_cost=p.cost_normalized if p else None,
        proposal_time=p.time_days if p else None,
        proposal_new_agent=p.new_agent_id if p else None,
    )
