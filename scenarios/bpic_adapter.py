"""
BPIC Event Log Adapter for DANCER Large-Scale Experimentation.

Converts real BPMN event logs (XES format) into grounded DANCER scenarios.

Pipeline:
  1. LOAD     — Read XES log via pm4py
  2. DISCOVER — Mine Petri net (Inductive Miner) → process graph Φ
  3. CONFORM  — Token-based replay → identify real deviation cases
  4. PROFILE  — Extract per-resource/subsidiary agent constraint profiles
                from empirical throughput times, costs, activity frequencies
  5. INJECT   — Classify each deviation as parametric / structural / sovereignty
  6. EMIT     — Yield (agent_dicts, active_ids, registry, disruption_type, tau_min)
                compatible with run_ieee_experiment.py

Supported logs (drop any of these in the data/ directory):
  BPIC 2019  — Purchase Order Handling (recommended; 60 subsidiaries)
               doi:10.4121/uuid:d06aff4b  |  ~694 MB XES
  BPIC 2017  — Loan Application           doi:10.4121/uuid:5f3067df
  BPIC 2020  — Travel Declarations        doi:10.4121/uuid:52fb97d4
  SEPSIS     — Hospital Cases             doi:10.4121/uuid:915d2bfb
  Road Traffic Fines                      doi:10.4121/uuid:270fd440

If the real log is not present, the adapter generates a structurally identical
synthetic log (same number of subsidiaries, activities, deviation rate, and
throughput statistics) for development and CI purposes.

Usage:
    from scenarios.bpic_adapter import BPICAdapter

    adapter = BPICAdapter("data/BPIChallenge2019.xes")
    adapter.fit()                                    # discover + conform + profile

    # Iterate over grounded disruption scenarios
    for scenario in adapter.iter_scenarios(n=300, disruption_type="structural_failure"):
        agents, active, registry, dtype, tau_min = scenario
        # Pass directly to run_system()

    # Or get a batch as a list
    scenarios = adapter.sample_scenarios(n=300)
    print(adapter.summary())
"""
from __future__ import annotations

import os
import random
import datetime
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# ── pm4py imports ─────────────────────────────────────────────────────────────
try:
    import pm4py
    from pm4py.objects.log.obj import EventLog
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    from pm4py.statistics.traces.generic.log import case_statistics
    from pm4py.statistics.attributes.log import get as attr_get
    _PM4PY_OK = True
except ImportError:
    _PM4PY_OK = False
    warnings.warn("pm4py not installed. Install with: pip install pm4py")

# ── Project imports ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.agents import IPAgent
try:
    from llm.domain_context import build_domain_context
    _DC_OK = True
except ImportError:
    _DC_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentProfile:
    """Empirically derived agent constraint profile from a real log resource/group."""
    agent_id: str
    role: str
    max_cost: float              # 95th-percentile normalised cost this resource handles
    max_delay_days: float        # 95th-percentile throughput time for this agent's steps
    availability: float          # fraction of cases this resource was active and on-time
    min_utility: float           # = 1 - observed deviation rate for this resource
    weights: Dict[str, float]    # w_res, w_time, w_pol derived from activity profile
    case_count: int = 0
    deviation_count: int = 0
    avg_throughput_days: float = 0.0

    def to_ipa(self, committed_f_res: Optional[float] = None) -> IPAgent:
        return IPAgent(
            agent_id=self.agent_id,
            role=self.role,
            max_cost=self.max_cost,
            min_utility=self.min_utility,
            max_delay_days=self.max_delay_days,
            availability=self.availability,
            weights=self.weights,
            committed_f_res=committed_f_res,
        )


@dataclass
class DisruptionCase:
    """A single real log case identified as a disruption episode."""
    case_id: str
    disruption_type: str          # parametric_fluctuation | structural_failure | sovereignty_conflict
    missing_tokens: int
    remaining_tokens: int
    throughput_days: float
    deviation_activity: Optional[str]  # activity where deviation was detected
    org_group: Optional[str]
    item_type: Optional[str]
    cost_total: float
    raw_attributes: Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
#  Log classification rules
# ══════════════════════════════════════════════════════════════════════════════

# BPIC 2019 specific activity → disruption type mapping
# Based on domain knowledge from the dataset description
_BPIC2019_DISRUPTION_RULES = {
    "Set Payment Block":             "sovereignty_conflict",
    "Change Approval for Good Receipt": "sovereignty_conflict",
    "SRM: Awaiting Approval":        "sovereignty_conflict",
    "Cancel Purchase Order Item":    "structural_failure",
    "Vendor creates invoice":        "structural_failure",   # invoice before GR = process deviation
    "Change Quantity":               "parametric_fluctuation",
    "Change Price":                  "parametric_fluctuation",
    "Change Payment Term":           "parametric_fluctuation",
    "Change Delivery Indicator":     "parametric_fluctuation",
    "Release Purchase Order Item":   "parametric_fluctuation",
}

# Role assignment: map org:group / activity patterns to IPA roles
_ROLE_HEURISTICS = [
    (["carrier", "transport", "logistics", "shipping", "delivery", "3pl"], "carrier"),
    (["warehouse", "store", "goods", "receipt", "gr"], "warehouse"),
    (["manufacturer", "vendor", "supplier", "production"], "manufacturer"),
    (["finance", "payment", "invoice", "clear", "accounts"], "manufacturer"),  # policy-holder
    (["procurement", "purchase", "buyer", "srm"], "warehouse"),
]


def _infer_role(agent_id: str, dominant_activities: List[str]) -> str:
    """Infer IPA role from agent ID and dominant activities in log."""
    combined = (agent_id + " " + " ".join(dominant_activities)).lower()
    for keywords, role in _ROLE_HEURISTICS:
        if any(kw in combined for kw in keywords):
            return role
    return "warehouse"   # default


def _infer_disruption_type(
    trace,
    replayed_result: dict,
    rules: Dict[str, str],
) -> Optional[str]:
    """Classify a deviating trace as a disruption type.

    Priority:
      1. Activity-level rule match (most specific)
      2. Token-based replay signature:
         - missing_tokens > 0 + remaining > 0  → structural_failure (agent absent)
         - missing_tokens > 0 + remaining == 0  → sovereignty_conflict (policy block)
         - remaining > 0 only                   → parametric_fluctuation (delay)
    """
    # Check activity-level rules first
    for event in trace:
        act = event.get("concept:name", "")
        if act in rules:
            return rules[act]

    # Check attribute-level hint (set during synthetic generation)
    attr_hint = trace.attributes.get("disruption_type")
    if attr_hint:
        return attr_hint

    # Fallback: token replay signature
    mt = replayed_result.get("missing_tokens", 0)
    rt = replayed_result.get("remaining_tokens", 0)

    if mt > 0 and rt > 0:
        return "structural_failure"
    elif mt > 0 and rt == 0:
        return "sovereignty_conflict"
    elif rt > 0:
        return "parametric_fluctuation"

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Core Adapter
# ══════════════════════════════════════════════════════════════════════════════

class BPICAdapter:
    """
    Converts a real BPMN event log into grounded DANCER scenarios.

    Parameters
    ----------
    xes_path : str
        Path to the .xes log file. If the file does not exist, a synthetic
        log with matching statistical properties is generated and saved there.
    log_type : str
        One of "bpic2019" | "bpic2017" | "bpic2020" | "sepsis" | "road_traffic"
        Determines which activity→disruption rules and role heuristics to apply.
    noise_threshold : float
        Inductive Miner noise threshold [0.0, 1.0]. Higher = simpler model
        but more deviating traces classified as conforming. Default 0.2.
    max_cases : int
        Subsample log to this many cases (useful for very large logs like BPIC 2018).
        None = use all cases.
    seed : int
    """

    def __init__(
        self,
        xes_path: str,
        log_type: str = "bpic2019",
        noise_threshold: float = 0.2,
        max_cases: Optional[int] = None,
        seed: int = 42,
        allow_synthetic: bool = False,
    ) -> None:
        if not _PM4PY_OK:
            raise ImportError("pm4py is required. Install: pip install pm4py")

        self.xes_path = xes_path
        self.log_type = log_type
        self.noise_threshold = noise_threshold
        self.max_cases = max_cases
        self.seed = seed
        self.allow_synthetic = allow_synthetic
        random.seed(seed)
        np.random.seed(seed)

        # Populated by fit()
        self.log: Optional[EventLog] = None
        self.net = None
        self.im = None
        self.fm = None
        self.replayed: List[dict] = []
        self.disruption_cases: List[DisruptionCase] = []
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self._fitted = False

        # Disruption rules for this log type
        self._rules = _BPIC2019_DISRUPTION_RULES   # extend per log_type if needed

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self) -> "BPICAdapter":
        """Full pipeline: load → discover → conform → profile."""
        self._load()
        self._discover()
        self._conform()
        self._profile_agents()
        if _DC_OK:
            self.domain_context = build_domain_context(
                self.log, self.net, self.log_type,
                self.agent_profiles, self.disruption_cases,
            )
        else:
            self.domain_context = None
        self._fitted = True
        return self

    def iter_scenarios(
        self,
        n: int = 300,
        disruption_type: Optional[str] = None,
        with_registry: bool = True,
        with_meta: bool = False,
    ) -> Generator:
        """Yield grounded DANCER scenario tuples derived from real log cases.

        AUDIT F8.1: the original implementation selected the SAME three agent
        profiles (global argmax of w_pol / w_time / w_res) for every episode,
        collapsing the '74 diverse spend-area profiles' claim to one fixed
        triple and producing identical baseline ceilings across episodes.
        Each episode now SAMPLES its triple from the top-q pools per dominant
        dimension, using a per-episode seeded RNG so that:
          (a) episodes differ in agent composition,
          (b) the entire campaign is reproducible from --seed,
          (c) every system in the same episode receives the identical
              constraint matrix (committed_f_res frozen in the dicts).

        with_meta=True → yields (scenario_tuple, case_id) so duplicate draws
        from small pools (e.g. the 29-case sovereignty pool) are traceable.
        """
        assert self._fitted, "Call fit() first."

        pool = [c for c in self.disruption_cases
                if disruption_type is None or c.disruption_type == disruption_type]

        if not pool:
            raise ValueError(f"No disruption cases found for type={disruption_type!r}. "
                             f"Available: {set(c.disruption_type for c in self.disruption_cases)}")

        for idx in range(n):
            rng = random.Random(self.seed * 1_000_003 + idx)
            case = rng.choice(pool)   # with replacement (pool may be small)
            scenario = self._case_to_scenario(case, with_registry=with_registry,
                                              rng=rng)
            yield (scenario, case.case_id) if with_meta else scenario

    def sample_scenarios(
        self,
        n: int = 300,
        disruption_type: Optional[str] = None,
        with_meta: bool = False,
    ) -> List[Tuple]:
        return list(self.iter_scenarios(n=n, disruption_type=disruption_type,
                                        with_meta=with_meta))

    def summary(self) -> str:
        assert self._fitted, "Call fit() first."
        lines = [
            f"\n{'='*70}",
            f"  BPICAdapter Summary — {os.path.basename(self.xes_path)}",
            f"{'='*70}",
            f"  Cases loaded        : {len(self.log):,}",
            f"  Process model       : {len(self.net.places)} places, "
            f"{len(self.net.transitions)} transitions",
            f"  Deviating cases     : {len(self.disruption_cases):,} "
            f"({len(self.disruption_cases)/len(self.log)*100:.1f}%)",
            f"  Agent profiles      : {len(self.agent_profiles)}",
            "",
        ]

        # Breakdown by type
        by_type: Dict[str, int] = defaultdict(int)
        for c in self.disruption_cases:
            by_type[c.disruption_type] += 1
        lines.append("  Disruption breakdown:")
        for dtype, cnt in sorted(by_type.items()):
            lines.append(f"    {dtype:<35} {cnt:>6} cases ({cnt/len(self.disruption_cases)*100:.1f}%)")

        lines += [
            "",
            "  Top agent profiles (by case count):",
        ]
        for pid, prof in sorted(self.agent_profiles.items(),
                                key=lambda x: -x[1].case_count)[:8]:
            lines.append(
                f"    {pid:<20} role={prof.role:<20} "
                f"max_cost={prof.max_cost:.2f}  "
                f"max_delay={prof.max_delay_days:.1f}d  "
                f"avail={prof.availability:.2f}  "
                f"cases={prof.case_count}"
            )
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)

    # ── Internal pipeline steps ───────────────────────────────────────────────

    def _load(self) -> None:
        """Load XES log. Missing file is an ERROR unless allow_synthetic=True.

        AUDIT F8.5: the original code silently generated a synthetic log when
        the real XES was absent, so a 'real-log' campaign could run entirely
        on fabricated data without any signal in the results. Synthetic
        generation now requires explicit opt-in and is loudly labelled.
        """
        if not os.path.exists(self.xes_path):
            if not self.allow_synthetic:
                raise FileNotFoundError(
                    f"Real event log not found at {self.xes_path!r}. "
                    f"Download it (see module docstring) or pass "
                    f"allow_synthetic=True ONLY for development/CI runs — "
                    f"synthetic results must never be reported as real-log results."
                )
            print(f"[adapter] Log not found at {self.xes_path!r}.")
            print("[adapter] *** SYNTHETIC MODE (allow_synthetic=True) — "
                  "results are NOT real-log grounded. ***")
            self._generate_synthetic_log()
            self.is_synthetic = True
        else:
            self.is_synthetic = False

        print(f"[adapter] Loading {self.xes_path} …")
        self.log = xes_importer.apply(
            self.xes_path,
            parameters={"show_progress_bar": False},
        )

        # Subsample if requested
        if self.max_cases and len(self.log) > self.max_cases:
            indices = random.sample(range(len(self.log)), self.max_cases)
            from pm4py.objects.log.obj import EventLog
            sub = EventLog()
            for i in indices:
                sub.append(self.log[i])
            self.log = sub
            print(f"[adapter] Subsampled to {len(self.log):,} cases.")
        else:
            print(f"[adapter] Loaded {len(self.log):,} cases, "
                  f"{sum(len(t) for t in self.log):,} events.")

    def _discover(self) -> None:
        """Mine Petri net using Inductive Miner with noise filtering."""
        print(f"[adapter] Mining process model (noise_threshold={self.noise_threshold}) …")
        self.net, self.im, self.fm = pm4py.discover_petri_net_inductive(
            self.log, noise_threshold=self.noise_threshold,
        )
        print(f"[adapter] Model: {len(self.net.places)} places, "
              f"{len(self.net.transitions)} transitions, "
              f"{len(self.net.arcs)} arcs.")

    def _conform(self) -> None:
        """Token-based replay → collect deviating cases."""
        print("[adapter] Running token-based replay …")
        self.replayed = pm4py.conformance_diagnostics_token_based_replay(
            self.log, self.net, self.im, self.fm,
        )

        # Compute throughput times
        stats = case_statistics.get_cases_description(self.log)

        # Pre-compute median throughput once — avoids O(n²) recomputation per trace
        all_durations = [v["caseDuration"] / 86400 for v in stats.values()
                         if v["caseDuration"] > 0]
        median_tp = float(np.median(all_durations)) if all_durations else 0.0

        self.disruption_cases = []
        for trace, replay in zip(self.log, self.replayed):
            cid = trace.attributes.get("concept:name", "unknown")
            mt = replay.get("missing_tokens", 0)
            rt = replay.get("remaining_tokens", 0)

            is_deviating = (mt > 0 or rt > 0)
            has_rule_match = any(
                ev.get("concept:name", "") in self._rules for ev in trace
            )
            has_attr_hint = bool(trace.attributes.get("disruption_type"))

            if not (is_deviating or has_rule_match or has_attr_hint):
                # Conforming trace — include as parametric if unusually slow
                throughput = stats.get(cid, {}).get("caseDuration", 0) / 86400
                if throughput > median_tp * 1.5:
                    is_deviating = True   # slow conforming case = parametric

            if not (is_deviating or has_rule_match or has_attr_hint):
                continue

            dtype = _infer_disruption_type(trace, replay, self._rules)
            if dtype is None:
                continue

            throughput_days = stats.get(cid, {}).get("caseDuration", 0) / 86400
            cost = sum(
                float(ev.get("cost:total", 0) or ev.get("Cumulative net worth (EUR)", 0) or 0)
                for ev in trace
            )

            # Find deviation activity
            dev_act = None
            for ev in trace:
                if ev.get("concept:name", "") in self._rules:
                    dev_act = ev["concept:name"]
                    break

            self.disruption_cases.append(DisruptionCase(
                case_id=cid,
                disruption_type=dtype,
                missing_tokens=mt,
                remaining_tokens=rt,
                throughput_days=throughput_days,
                deviation_activity=dev_act,
                org_group=trace.attributes.get("org:group"),
                item_type=trace.attributes.get("Item Type"),
                cost_total=cost,
                raw_attributes=dict(trace.attributes),
            ))

        print(f"[adapter] Disruption cases: {len(self.disruption_cases):,} "
              f"({len(self.disruption_cases)/len(self.log)*100:.1f}% of log)")

        by_type: Dict[str, int] = defaultdict(int)
        for c in self.disruption_cases:
            by_type[c.disruption_type] += 1
        for t, cnt in sorted(by_type.items()):
            print(f"  {t}: {cnt}")

    def _profile_agents(self) -> None:
        """Extract empirical agent constraint profiles from the log.

        Strategy:
          - Group events by org:group (subsidiary) or org:resource
          - Compute per-group: median throughput, cost percentile, deviation rate
          - Normalize costs to [0, 1] relative to log-wide max
          - Derive weights from activity role proportions
        """
        print("[adapter] Profiling agents from log …")

        # Collect per-group stats
        group_costs: Dict[str, List[float]] = defaultdict(list)
        group_throughputs: Dict[str, List[float]] = defaultdict(list)
        group_deviations: Dict[str, int] = defaultdict(int)
        group_cases: Dict[str, int] = defaultdict(int)
        group_activities: Dict[str, List[str]] = defaultdict(list)

        stats = case_statistics.get_cases_description(self.log)

        for trace, replay in zip(self.log, self.replayed):
            group = (trace.attributes.get("org:group")
                     or trace.attributes.get("org:resource")
                     or trace.attributes.get("Sub spend area text")  # BPIC 2019
                     or trace.attributes.get("Spend area text")       # BPIC 2019 fallback
                     or trace.attributes.get("Company")               # generic fallback
                     or "UNKNOWN")
            cid = trace.attributes.get("concept:name", "")
            throughput = stats.get(cid, {}).get("caseDuration", 0) / 86400

            group_cases[group] += 1
            group_throughputs[group].append(throughput)

            if replay.get("missing_tokens", 0) > 0:
                group_deviations[group] += 1

            for ev in trace:
                cost = float(ev.get("cost:total", 0) or 0)
                if cost > 0:
                    group_costs[group].append(cost)
                act = ev.get("concept:name", "")
                if act:
                    group_activities[group].append(act)

        # Normalize costs log-wide
        all_costs = [c for costs in group_costs.values() for c in costs]
        global_max_cost = np.percentile(all_costs, 99) if all_costs else 1.0
        global_max_tp = np.percentile(
            [t for tps in group_throughputs.values() for t in tps], 99
        )

        # Build profiles for groups with enough data
        for group, n_cases in group_cases.items():
            if n_cases < 3:
                continue

            costs = group_costs.get(group, [0])
            tps = group_throughputs.get(group, [1.0])
            acts = group_activities.get(group, [])
            dev_rate = group_deviations.get(group, 0) / max(n_cases, 1)

            # Normalized max_cost: 95th-percentile of this group's costs
            raw_cost_95 = np.percentile(costs, 95) if costs else global_max_cost
            max_cost = min(0.95, max(0.10, raw_cost_95 / (global_max_cost + 1e-9)))

            # max_delay_days: 95th-percentile throughput
            raw_tp_95 = np.percentile(tps, 95) if tps else 7.0
            max_delay = min(30.0, max(0.5, raw_tp_95))

            # Availability: fraction of cases with no deviation
            availability = max(0.10, min(0.99, 1.0 - dev_rate))

            # Min utility: generous (0.4-0.7) — agents accept if reasonable
            min_utility = max(0.40, min(0.75, 0.5 + dev_rate * 0.2))

            # Infer role from dominant activities
            role = _infer_role(group, acts[:20])

            # Weights derived from activity types:
            # Policy-heavy (lots of approvals) → high w_pol
            # Time-critical (lots of GR/receipts) → high w_time
            policy_acts = sum(1 for a in acts if any(k in a.lower()
                              for k in ["payment", "approval", "block", "srm"]))
            time_acts = sum(1 for a in acts if any(k in a.lower()
                            for k in ["receipt", "goods", "clear", "invoice"]))
            total_acts = max(1, len(acts))
            w_pol = min(0.70, max(0.15, policy_acts / total_acts * 2))
            w_time = min(0.60, max(0.20, time_acts / total_acts * 1.5))
            w_res = max(0.10, 1.0 - w_pol - w_time)
            # Normalize
            s = w_pol + w_time + w_res
            weights = {
                "w_res":  round(w_res / s, 3),
                "w_time": round(w_time / s, 3),
                "w_pol":  round(w_pol / s, 3),
            }

            self.agent_profiles[group] = AgentProfile(
                agent_id=group,
                role=role,
                max_cost=round(max_cost, 3),
                max_delay_days=round(max_delay, 2),
                availability=round(availability, 3),
                min_utility=round(min_utility, 3),
                weights=weights,
                case_count=n_cases,
                deviation_count=group_deviations.get(group, 0),
                avg_throughput_days=round(float(np.mean(tps)), 2),
            )

        print(f"[adapter] Profiled {len(self.agent_profiles)} agent groups.")

    # ── Scenario builder ──────────────────────────────────────────────────────

    def _case_to_scenario(
        self,
        case: DisruptionCase,
        with_registry: bool = True,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[dict], List[str], List[dict], str, float]:
        """Convert one real disruption case to a DANCER scenario tuple.

        Agent selection (audit F8.1 — per-episode SAMPLING, not global argmax):
          - Lead Agent  (policy archetype):  sampled from the top-q profiles by w_pol
          - Time Agent  (time archetype):    sampled from the top-q profiles by w_time
          - Resource Agent (res archetype):  sampled from the top-q profiles by w_res,
            marked unavailable (committed_f_res = 0) for structural failures
          with q = min(8, |profiles|) and the three groups forced distinct.

        Availability draws (committed_f_res) are made HERE with the episode
        RNG and frozen into the dicts, so every system in the episode shares
        the same draw and the campaign is reproducible from the seed.
        """
        rng = rng or random.Random(self.seed)
        profiles = list(self.agent_profiles.values())

        def _draw_committed(prof: AgentProfile) -> float:
            # Mirrors IPAgent's construction-time logic, but deterministic
            # under the per-episode RNG.
            if rng.random() < prof.availability:
                return round(rng.uniform(0.85, 1.0), 4)
            return 0.0

        if len(profiles) >= 3:
            q = min(8, len(profiles))
            by_pol  = sorted(profiles, key=lambda p: -p.weights["w_pol"])[:q]
            by_time = sorted(profiles, key=lambda p: -p.weights["w_time"])[:q]
            by_res  = sorted(profiles, key=lambda p: -p.weights["w_res"])[:q]

            lead = rng.choice(by_pol)
            time_pool = [p for p in by_time if p.agent_id != lead.agent_id] or by_time
            time_ = rng.choice(time_pool)
            res_pool = [p for p in by_res
                        if p.agent_id not in (lead.agent_id, time_.agent_id)] or by_res
            res = rng.choice(res_pool)

            # Adjust max_cost for sovereignty conflict: tighten lead agent budget
            if case.disruption_type == "sovereignty_conflict":
                tighter_cost = round(lead.max_cost * 0.75, 3)
                lead = AgentProfile(
                    **{**lead.__dict__, "max_cost": tighter_cost, "agent_id": lead.agent_id}
                )

            # Structural failure: resource agent committed unavailable
            if case.disruption_type == "structural_failure":
                res_f = 0.0
            else:
                res_f = _draw_committed(res)

            agents = [
                lead.to_ipa(committed_f_res=_draw_committed(lead)).to_dict(),
                time_.to_ipa(committed_f_res=_draw_committed(time_)).to_dict(),
                res.to_ipa(committed_f_res=res_f).to_dict(),
            ]
            active_ids = [lead.agent_id, time_.agent_id, res.agent_id]
        else:
            # Fallback to default synthetic agents
            agents, active_ids = self._default_agents(case.disruption_type)

        # Registry: add a synthetic 3PL agent with empirically reasonable constraints
        registry = []
        if with_registry:
            # 3PL constraints: slightly more expensive but very available
            avg_cost = np.mean([p.max_cost for p in profiles]) if profiles else 0.60
            registry = [IPAgent(
                agent_id="3PL_Agent",
                role="third_party_logistics",
                max_cost=min(0.90, avg_cost * 1.20),
                min_utility=0.40,
                max_delay_days=max(3.0, case.throughput_days * 0.8),
                availability=0.92,
                weights={"w_res": 0.40, "w_time": 0.40, "w_pol": 0.20},
            ).to_dict()]

        # tau_min: fixed at 0.40 for structural failure (carrier unavailable pulls
        # aggregate utility down), 0.45 otherwise.  The DSL cost-clamping in
        # validate_proposals_node ensures proposals stay within agent budgets,
        # so the achievable utility with a clamped proposal is ~0.60-0.80 —
        # well above these thresholds.
        if case.disruption_type == "structural_failure":
            tau_min = 0.40
        elif case.disruption_type == "sovereignty_conflict":
            tau_min = 0.45
        else:
            tau_min = 0.45

        return agents, active_ids, registry, case.disruption_type, tau_min

    def _default_agents(
        self, disruption_type: str
    ) -> Tuple[List[dict], List[str]]:
        """Minimal synthetic fallback when log has too few distinct groups."""
        mfr = IPAgent("Manufacturer", "manufacturer", max_cost=0.50, min_utility=0.70,
                      max_delay_days=10.0, availability=1.0,
                      weights={"w_res": 0.20, "w_time": 0.20, "w_pol": 0.60}).to_dict()
        wh  = IPAgent("Warehouse", "warehouse", max_cost=0.80, min_utility=0.60,
                      max_delay_days=2.0, availability=1.0,
                      weights={"w_res": 0.30, "w_time": 0.50, "w_pol": 0.20}).to_dict()
        carrier_f_res = 0.0 if disruption_type == "structural_failure" else None
        car = IPAgent("Carrier", "carrier", max_cost=0.80, min_utility=0.50,
                      max_delay_days=7.0, availability=0.50,
                      weights={"w_res": 0.50, "w_time": 0.30, "w_pol": 0.20},
                      committed_f_res=carrier_f_res).to_dict()
        return [mfr, wh, car], ["Manufacturer", "Warehouse", "Carrier"]

    # ── Synthetic log generator ───────────────────────────────────────────────

    def _generate_synthetic_log(self) -> None:
        """Generate a BPIC 2019-equivalent synthetic XES log and save to xes_path."""
        from pm4py.objects.log.obj import EventLog, Trace, Event

        print("[adapter] Generating synthetic BPIC 2019-equivalent log …")
        ACTIVITIES = [
            "Create Purchase Order Item", "Change Quantity", "Record Goods Receipt",
            "Record Invoice Receipt", "Clear Invoice", "Set Payment Block",
            "Remove Payment Block", "Change Price", "Vendor creates invoice",
            "Change Approval for Good Receipt", "Release Purchase Order Item",
            "Cancel Purchase Order Item", "Change Payment Term",
            "Change Delivery Indicator", "SRM: Awaiting Approval", "SRM: Complete",
        ]
        SUBSIDIARIES = [f"SUB_{i:03d}" for i in range(1, 61)]
        ITEM_TYPES = ["3-way match, invoice before GR", "2-way match",
                      "3-way match, invoice after GR", "consignment"]

        log = EventLog()
        N = 5000

        for idx in range(N):
            trace = Trace()
            trace.attributes["concept:name"] = f"PO_{idx:06d}"
            trace.attributes["org:group"] = random.choice(SUBSIDIARIES)
            trace.attributes["Item Type"] = random.choice(ITEM_TYPES)
            trace.attributes["Company"] = random.choice(SUBSIDIARIES[:20])

            base = datetime.datetime(2018, 1, 2) + datetime.timedelta(
                days=random.randint(0, 364))
            seq: List[Tuple[str, datetime.datetime, float]] = []

            t = base
            seq.append(("Create Purchase Order Item", t, random.uniform(500, 50000)))

            disruption: Optional[str] = None
            if random.random() < 0.08:
                t += datetime.timedelta(hours=random.randint(1, 48))
                seq.append(("Vendor creates invoice", t, random.uniform(100, 5000)))
                disruption = "structural_failure"
            elif random.random() < 0.06:
                t += datetime.timedelta(hours=random.randint(2, 72))
                seq.append(("Set Payment Block", t, 0))
                disruption = "sovereignty_conflict"
            elif random.random() < 0.20:
                disruption = "parametric_fluctuation"
                t += datetime.timedelta(hours=random.randint(1, 24))
                seq.append(("Change Quantity", t, random.uniform(50, 2000)))

            t += datetime.timedelta(days=random.randint(1, 30))
            seq.append(("Record Goods Receipt", t, random.uniform(200, 30000)))
            t += datetime.timedelta(days=random.randint(1, 14))
            seq.append(("Record Invoice Receipt", t, random.uniform(200, 30000)))

            if disruption == "sovereignty_conflict" and random.random() < 0.5:
                t += datetime.timedelta(days=random.randint(1, 7))
                seq.append(("Remove Payment Block", t, 0))

            t += datetime.timedelta(days=random.randint(1, 45))
            seq.append(("Clear Invoice", t, random.uniform(100, 5000)))

            if disruption:
                trace.attributes["disruption_type"] = disruption

            for act, ts, cost in seq:
                ev = Event()
                ev["concept:name"] = act
                ev["time:timestamp"] = ts
                ev["org:resource"] = f"User_{random.randint(1, 500)}"
                ev["org:group"] = trace.attributes["org:group"]
                ev["lifecycle:transition"] = "complete"
                ev["cost:total"] = round(cost, 2)
                trace.append(ev)

            log.append(trace)

        os.makedirs(os.path.dirname(self.xes_path) or ".", exist_ok=True)
        xes_exporter.apply(log, self.xes_path, parameters={"show_progress_bar": False})
        total_d = sum(1 for t in log if t.attributes.get("disruption_type"))
        print(f"[adapter] Saved {N} cases ({total_d} with disruptions) → {self.xes_path}")
