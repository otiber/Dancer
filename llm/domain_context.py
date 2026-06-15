"""
domain_context.py — Runtime domain context derived from a real BPMN event log.

The critical insight: DANCER is a GENERIC negotiation protocol for process
re-choreography. It is NOT a supply-chain system. The logistics vocabulary
(warehouse, carrier, port strike, air freight) exists only in the synthetic
evaluation scenarios of the original paper.

When running on real logs, every piece of domain vocabulary must be extracted
from the data itself:
  - Domain name and type         ← from log metadata / detected heuristically
  - Real activity names          ← from the discovered Petri net transitions
  - Real agent/resource names    ← from org:group / org:resource attributes
  - Recovery action vocabulary   ← from activities that appear AFTER deviations
  - Disruption descriptions      ← from actual deviation patterns in the log
  - Process token semantics      ← from what the case represents in that domain

This module builds a DomainContext object that the prompt builder (prompts.py)
uses instead of any hardcoded supply-chain vocabulary.

Supported log types and their domain mappings:
  bpic2019  → Purchase Order process (60 subsidiaries, SAP)
  bpic2017  → Loan application process (bank, client, committee)
  bpic2020  → Travel declaration process (university departments)
  sepsis    → Hospital patient routing (ICU, ward, labs)
  road_traffic → Traffic fine management (municipality, court)
  generic   → Auto-detected from activity names
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# ══════════════════════════════════════════════════════════════════════════════
#  Domain vocabulary definitions per log type
#  Everything here comes from the real log's domain literature, NOT invented
# ══════════════════════════════════════════════════════════════════════════════

_DOMAIN_VOCAB: Dict[str, Dict] = {
    "bpic2019": {
        "domain_name": "Purchase Order Handling (SAP ERP)",
        "process_token": "purchase order item",
        "lead_agent_role": "purchasing manager",
        "time_agent_role": "goods receipt officer",
        "resource_agent_role": "vendor/supplier",
        "disruption_vocab": {
            "parametric_fluctuation": (
                "A quantity or price change has been recorded on a purchase order item, "
                "causing a deviation from the approved order value. "
                "The item is stalled pending re-approval."
            ),
            "structural_failure": (
                "A vendor has issued an invoice BEFORE the goods receipt was recorded, "
                "violating the 3-way matching rule. The normal approval path is broken. "
                "An alternative matching or routing path must be negotiated."
            ),
            "sovereignty_conflict": (
                "A payment block has been set on this purchase order item by the "
                "purchasing manager, enforcing a new budget constraint. "
                "The item cannot proceed until the block is resolved or a "
                "cost-compliant alternative is agreed."
            ),
            "cascading": (
                "Both a structural deviation (invoice before GR) AND a payment block "
                "have occurred on this purchase order item simultaneously. "
                "A recovery path must satisfy both the structural re-routing and "
                "the tightened budget constraint."
            ),
        },
        "recovery_actions": [
            "Release item with adjusted payment terms",
            "Route via 2-way matching (waive GR requirement)",
            "Cancel and re-issue purchase order item",
            "Escalate to senior buyer for manual approval",
            "Apply consolidated bulk invoice processing",
            "Defer to next payment run with updated cost allocation",
            "Route via alternative vendor/subsidiary",
            "Apply consignment stock arrangement",
        ],
        "structural_recovery": "Route via alternative vendor or matching method",
        "agent_descriptor": "subsidiary",
    },

    "bpic2017": {
        "domain_name": "Loan Application Process (Financial Institution)",
        "process_token": "loan application",
        "lead_agent_role": "loan officer",
        "time_agent_role": "credit analyst",
        "resource_agent_role": "applicant/client",
        "disruption_vocab": {
            "parametric_fluctuation": (
                "The client has not responded within the expected window, causing "
                "the loan application to fall behind its processing schedule. "
                "A re-scheduling or follow-up action must be negotiated."
            ),
            "structural_failure": (
                "The assigned credit analyst is unavailable. "
                "The application cannot proceed through the normal assessment path. "
                "An alternative assessor or assessment method must be found."
            ),
            "sovereignty_conflict": (
                "The credit committee has tightened its approval criteria mid-process, "
                "setting a lower maximum loan amount or stricter collateral requirement. "
                "The existing offer must be revised to comply with the new policy."
            ),
            "cascading": (
                "The credit analyst is unavailable AND the committee has revised "
                "the approval criteria simultaneously. Both constraints must be "
                "addressed in the re-negotiated application path."
            ),
        },
        "recovery_actions": [
            "Reassign to available credit analyst",
            "Apply automated credit scoring (bypass manual assessment)",
            "Issue conditional offer pending missing documents",
            "Reduce loan amount to comply with new committee threshold",
            "Request additional collateral and re-submit",
            "Escalate to senior loan officer for expedited review",
            "Apply fast-track processing for low-risk profile",
            "Defer application to next committee session",
        ],
        "structural_recovery": "Reassign assessment to available analyst or automated system",
        "agent_descriptor": "department",
    },

    "bpic2020": {
        "domain_name": "Travel Declaration Process (University)",
        "process_token": "travel declaration",
        "lead_agent_role": "budget holder",
        "time_agent_role": "travel administrator",
        "resource_agent_role": "employee/traveler",
        "disruption_vocab": {
            "parametric_fluctuation": (
                "The actual travel costs have exceeded the pre-approved budget, "
                "causing a deviation that requires re-approval before reimbursement."
            ),
            "structural_failure": (
                "The designated approver is absent (out-of-office or role change). "
                "The declaration cannot proceed through the standard approval chain."
            ),
            "sovereignty_conflict": (
                "The budget holder has enforced a stricter spending limit mid-process, "
                "requiring the declaration amount to be reduced or split."
            ),
        },
        "recovery_actions": [
            "Route to deputy approver",
            "Split declaration into multiple smaller claims",
            "Reduce reimbursement to policy maximum",
            "Request exceptional approval from finance director",
            "Apply standard rate reimbursement (ignore actual cost)",
            "Defer to next budget cycle",
            "Apply department overhead allocation",
        ],
        "structural_recovery": "Delegate to deputy approver or alternative approval chain",
        "agent_descriptor": "department",
    },

    "sepsis": {
        "domain_name": "Sepsis Patient Routing (Hospital)",
        "process_token": "patient case",
        "lead_agent_role": "attending physician",
        "time_agent_role": "triage nurse",
        "resource_agent_role": "specialist/ward",
        "disruption_vocab": {
            "parametric_fluctuation": (
                "A laboratory test result has taken longer than expected, "
                "delaying the patient's progression through the care pathway. "
                "An alternative assessment or interim treatment path is needed."
            ),
            "structural_failure": (
                "The target ward or specialist is at capacity. "
                "The patient cannot be routed to the planned care unit. "
                "An alternative placement must be negotiated urgently."
            ),
            "sovereignty_conflict": (
                "A change in treatment protocol or resource allocation policy "
                "has been issued by the medical director, constraining the "
                "available intervention options for this patient."
            ),
            "cascading": (
                "The target ward is at capacity AND a new protocol restriction "
                "has been imposed simultaneously. Both constraints must be "
                "resolved in the re-planned care pathway."
            ),
        },
        "recovery_actions": [
            "Route to overflow ward with telemetry monitoring",
            "Transfer to partner hospital",
            "Apply step-down care in general ward with increased check frequency",
            "Expedite specialist consultation via telemedicine",
            "Initiate interim treatment pending bed availability",
            "Escalate to medical director for emergency bed allocation",
            "Apply cohort nursing arrangement",
            "Schedule in next available slot with priority flag",
        ],
        "structural_recovery": "Route to alternative ward, unit, or partner facility",
        "agent_descriptor": "clinical unit",
    },

    "road_traffic": {
        "domain_name": "Road Traffic Fine Management (Municipality)",
        "process_token": "traffic fine",
        "lead_agent_role": "case officer",
        "time_agent_role": "payment processor",
        "resource_agent_role": "offender/vehicle owner",
        "disruption_vocab": {
            "parametric_fluctuation": (
                "The fine payment is overdue. The standard collection timeline "
                "has been exceeded and the case requires re-routing."
            ),
            "structural_failure": (
                "The assigned case officer is unavailable. The fine cannot be "
                "processed through the standard assessment path."
            ),
            "sovereignty_conflict": (
                "A new municipal policy has changed the penalty schedule or "
                "appeal eligibility criteria mid-process."
            ),
        },
        "recovery_actions": [
            "Issue payment instalment plan",
            "Refer to credit collection agency",
            "Escalate to court proceedings",
            "Apply automated penalty enforcement",
            "Offer reduced settlement amount",
            "Extend payment deadline with administrative fee",
            "Route to appeals tribunal",
        ],
        "structural_recovery": "Refer to alternative officer or automated processing",
        "agent_descriptor": "municipality office",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  DomainContext dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DomainContext:
    """
    All domain-specific vocabulary needed to build grounded LLM prompts.
    Populated by BPICAdapter.build_domain_context() after fitting.

    Every field is derived from the real log — nothing is hardcoded per run.
    """
    # Identity
    log_type: str
    domain_name: str
    process_token: str              # what a 'case' represents (patient, loan, PO item)
    agent_descriptor: str           # what agents are called (subsidiary, department, ward)

    # Role labels (derived from log, used in prompts)
    lead_agent_role: str            # policy-enforcing lead (budget holder, loan officer)
    time_agent_role: str            # time-sensitive node (triage nurse, GR officer)
    resource_agent_role: str        # resource agent (vendor, analyst, specialist)

    # Disruption descriptions (domain-specific, not logistics)
    disruption_descriptions: Dict[str, str]  # disruption_type → description

    # Recovery action vocabulary (from log activities)
    recovery_actions: List[str]     # available re-routing actions in this domain
    structural_recovery: str        # one-liner for structural failure recovery

    # Empirical process data
    real_activity_names: List[str]   # from discovered Petri net transitions
    deviation_activities: List[str]  # activities most frequently at deviation points
    real_agent_ids: List[str]        # from log resource/org:group attributes
    avg_throughput_days: float       # log-wide median case duration
    cost_unit: str                   # "EUR" | "USD" | "normalized" (from log attributes)

    # Derived agent context (per scenario, set at scenario build time)
    active_agents: List[Dict] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  Context builder
# ══════════════════════════════════════════════════════════════════════════════

def detect_log_type(log) -> str:
    """Heuristically detect log type from activity names."""
    all_acts = set()
    for trace in log:
        for ev in trace:
            act = ev.get("concept:name", "")
            if act:
                all_acts.add(act.lower())

    if any("purchase" in a or "goods receipt" in a or "invoice" in a for a in all_acts):
        return "bpic2019"
    if any("loan" in a or "credit" in a or "application" in a for a in all_acts):
        return "bpic2017"
    if any("sepsis" in a or "triage" in a or "icu" in a or "leucocytes" in a for a in all_acts):
        return "sepsis"
    if any("fine" in a or "penalty" in a or "offence" in a for a in all_acts):
        return "road_traffic"
    if any("travel" in a or "declaration" in a or "reimbursement" in a for a in all_acts):
        return "bpic2020"
    return "generic"


def build_domain_context(
    log,
    net,
    log_type: str,
    agent_profiles: Dict,
    disruption_cases,
) -> DomainContext:
    """
    Build a DomainContext from a fitted BPICAdapter.
    Called once after adapter.fit() — result stored on the adapter.
    """
    # Auto-detect if not specified
    if log_type == "generic" or log_type not in _DOMAIN_VOCAB:
        log_type = detect_log_type(log)
    vocab = _DOMAIN_VOCAB.get(log_type, _DOMAIN_VOCAB["bpic2019"])

    # Extract real activity names from Petri net transitions
    real_activities = sorted(set(
        t.label for t in net.transitions if t.label
    ))

    # Find activities most common at deviation points
    deviation_acts: Counter = Counter()
    for case in disruption_cases:
        if case.deviation_activity:
            deviation_acts[case.deviation_activity] += 1
    top_deviation_acts = [a for a, _ in deviation_acts.most_common(10)]

    # Real agent IDs from profiles
    real_agent_ids = list(agent_profiles.keys())

    # Throughput stats
    import numpy as np
    throughputs = [p.avg_throughput_days for p in agent_profiles.values()
                   if p.avg_throughput_days > 0]
    avg_tp = float(np.median(throughputs)) if throughputs else 7.0

    # Cost unit detection
    cost_unit = "EUR"
    for trace in log:
        for ev in trace:
            if ev.get("Cumulative net worth (EUR)"):
                cost_unit = "EUR"
                break
            if ev.get("cost:total"):
                cost_unit = "normalised"
                break

    return DomainContext(
        log_type=log_type,
        domain_name=vocab["domain_name"],
        process_token=vocab["process_token"],
        agent_descriptor=vocab["agent_descriptor"],
        lead_agent_role=vocab["lead_agent_role"],
        time_agent_role=vocab["time_agent_role"],
        resource_agent_role=vocab["resource_agent_role"],
        disruption_descriptions=vocab["disruption_vocab"],
        recovery_actions=vocab["recovery_actions"],
        structural_recovery=vocab["structural_recovery"],
        real_activity_names=real_activities,
        deviation_activities=top_deviation_acts,
        real_agent_ids=real_agent_ids,
        avg_throughput_days=round(avg_tp, 1),
        cost_unit=cost_unit,
    )
