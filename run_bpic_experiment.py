"""
run_bpic_experiment.py — DANCER evaluation grounded in real BPMN event logs.
REVISED (audit 2026-06). Changes vs. the original runner:

  F2  DANCER receives the adapter's DomainContext (real-log vocabulary) and
      its LLM context is privacy-preserving (lead-only; see llm/prompts.py).
  F3  Every system is judged under the CANONICAL criterion (core/evaluation):
      tau_min is now passed to ALL runners; DANCER's recorded success is
      protocol-commit AND canonical.
  F7  --ablation runs the REAL paired DANCER-LLM vs DANCER-Heuristic
      comparison on identical scenarios (replaces the synthetic Fig. 8).
      --dropsweep runs the REAL message-drop robustness sweep (replaces the
      synthetic Fig. 7). --scaling runs the REAL O(n) message experiment
      (replaces the synthetic Fig. 6).
  F8  Synthetic-log fallback requires explicit --allow-synthetic; every
      episode row records the source case_id (sovereignty-pool provenance).
  F9  Scenario matrices are constructed up-front in the main thread from the
      per-episode seeded RNG; thread-pool parallelism affects only LLM calls.

Usage:
    python run_bpic_experiment.py --log data/BPIChallenge2019.xes --iters 100
    python run_bpic_experiment.py --log data/BPIChallenge2019.xes --ablation --iters 100
    python run_bpic_experiment.py --log data/BPIChallenge2019.xes --dropsweep
    python run_bpic_experiment.py --log data/BPIChallenge2019.xes --scaling
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

from core.models import RunResult, Proposal
from core.evaluation import canonical_outcome, feasibility_oracle
from protocols.bpmn_plus       import run_bpmn_plus
from protocols.cnp             import run_cnp
from protocols.mape_k          import run_mape_k
from protocols.zeta_cnp        import run_zeta_cnp
from protocols.react_mas       import run_react_mas
from protocols.llm_orchestra   import run_llm_orchestra
from protocols.dancer_protocol import run_dancer
from scenarios.bpic_adapter    import BPICAdapter
from analysis.statistics       import (run_full_analysis, print_summary_table,
                                       to_latex_table, ablation_wilcoxon)

SC_LABEL = {
    "parametric_fluctuation": "A_real",
    "structural_failure":     "B_real",
    "sovereignty_conflict":   "C_real",
}
DISRUPTION_TYPES = list(SC_LABEL.keys())


# ── DANCER wrapper: protocol commit + canonical outcome ───────────────────────

def _dancer_row(sc, agents, active, registry, dtype, tau_min, i,
                domain_context=None, llm_backend="real",
                drop_rate: float = 0.0) -> dict:
    final = run_dancer(agents, active, registry, dtype,
                       tau_min=tau_min, llm_backend=llm_backend,
                       communication_drop_rate=drop_rate,
                       domain_context=domain_context)
    committed = final["status"] == "SUCCESS" and final.get("best_proposal")
    if committed:
        outcome = canonical_outcome(
            Proposal(**final["best_proposal"]), agents, registry, tau_min,
            active_agent_ids=final.get("active_agent_ids", active),
            disruption_type=dtype)
    else:
        outcome = canonical_outcome(None, agents, registry, tau_min)
    bp_obj = Proposal(**final["best_proposal"]) if committed else None
    return {
        "scenario": sc, "system": "DANCER", "iteration": i,
        # Recorded success = protocol commit AND canonical criterion (F3)
        "success": bool(committed and outcome.success),
        "protocol_success": final["status"] == "SUCCESS",
        "rounds": final["round"],
        "messages_sent": final["messages_sent"],
        "total_tokens": final["total_tokens"],
        "final_utility": outcome.aggregate_utility,
        "protocol_utility": final["final_utility"],
        "n_individually_accepting": outcome.n_individually_accepting,
        "latency_seconds": final["latency_seconds"],
        # Committed-proposal provenance (audit F10): a near-zero std of
        # proposal_cost/proposal_time within a scenario means the LLM is
        # emitting constant proposals (greedy decoding or prompt anchoring).
        "proposal_method": bp_obj.method.value if bp_obj else None,
        "proposal_cost": bp_obj.cost_normalized if bp_obj else None,
        "proposal_time": bp_obj.time_days if bp_obj else None,
        "proposal_new_agent": bp_obj.new_agent_id if bp_obj else None,
    }


def _baseline_row(system, runner_result: RunResult) -> dict:
    row = runner_result.model_dump()
    row["protocol_success"] = row["success"]
    row["protocol_utility"] = row["final_utility"]
    row["n_individually_accepting"] = None
    return row


def _grounded_role(agent_dicts, dtype):
    """required_role for CNP/ζ-CNP, grounded in the episode's agents (F11).

    parametric → the time agent's actual role; structural/sovereignty → the
    resource agent's actual role. The original hardcoded synthetic labels
    ("warehouse"/"carrier") never matched BPIC spend-area roles, so every
    contractor refused with role_mismatch and CNP scored 0% by construction.
    """
    try:
        idx = 1 if dtype == "parametric_fluctuation" else 2
        return agent_dicts[idx].get("role")
    except Exception:
        return None


SYSTEM_RUNNERS = {
    "BPMN+": lambda sc, ag, ac, rg, dt, tau, i, ctx:
        _baseline_row("BPMN+", run_bpmn_plus(sc, ag, ac, rg, dt, i, tau)),
    "CNP": lambda sc, ag, ac, rg, dt, tau, i, ctx:
        _baseline_row("CNP", run_cnp(sc, ag, ac, dt, i, tau,
                                     required_role=_grounded_role(ag, dt))[0]),
    "ζ-CNP": lambda sc, ag, ac, rg, dt, tau, i, ctx:
        _baseline_row("ζ-CNP", run_zeta_cnp(sc, ag, ac, rg, dt, i, tau,
                                            required_role=_grounded_role(ag, dt))[0]),
    "MAPE-K": lambda sc, ag, ac, rg, dt, tau, i, ctx:
        _baseline_row("MAPE-K", run_mape_k(sc, ag, ac, rg, dt, i, tau)),
    "ReAct-MAS": lambda sc, ag, ac, rg, dt, tau, i, ctx:
        _baseline_row("ReAct-MAS", run_react_mas(sc, ag, ac, rg, dt, i, tau)),
    "LLM-Orchestra": lambda sc, ag, ac, rg, dt, tau, i, ctx:
        _baseline_row("LLM-Orchestra",
                      run_llm_orchestra(sc, ag, ac, rg, dt, i, tau)),
    "DANCER": lambda sc, ag, ac, rg, dt, tau, i, ctx:
        _dancer_row(sc, ag, ac, rg, dt, tau, i, domain_context=ctx),
}

LLM_SYSTEMS = {"DANCER", "ReAct-MAS", "LLM-Orchestra"}


def _fit_adapter(args) -> BPICAdapter:
    log_path = args.log or os.path.join(_ROOT, "data", "bpic2019_synthetic.xes")
    adapter = BPICAdapter(
        xes_path=log_path,
        log_type=args.log_type,
        noise_threshold=args.noise_threshold,
        max_cases=args.max_cases,
        seed=args.seed,
        allow_synthetic=args.allow_synthetic,
    )
    adapter.fit()
    print(adapter.summary())
    return adapter


def _run_cell(system, runner, scenarios_meta, sc_label, dtype, ctx, workers):
    """Run one (system, disruption_type) cell over pre-built scenarios."""
    cell_rows = []

    def _run_one(arg):
        i, (scenario_tuple, case_id) = arg
        agents, active, registry, sc_dtype, tau_min = scenario_tuple
        row = runner(sc_label, agents, active, registry, sc_dtype,
                     tau_min, i, ctx)
        if row:
            row["scenario"] = sc_label
            row["system"] = system
            row["iteration"] = i
            row["log_disruption_type"] = dtype
            row["case_id"] = case_id          # provenance (F8)
            row["tau_min"] = tau_min
            # F13d: flag episodes that are unsolvable by construction
            row["oracle_feasible"] = feasibility_oracle(
                agents, registry, tau_min, disruption_type=sc_dtype,
                active_agent_ids=active)
        return row

    n_workers = workers if system in LLM_SYSTEMS else 1
    items = list(enumerate(scenarios_meta))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, it): it[0] for it in items}
        with tqdm(total=len(futures), desc=f"  {system:<14}",
                  unit="run", dynamic_ncols=True, leave=True) as pbar:
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                    if row:
                        cell_rows.append(row)
                        ok = sum(r["success"] for r in cell_rows)
                        pbar.set_postfix(CR=f"{ok/len(cell_rows)*100:.1f}%")
                except Exception as e:
                    print(f"\n    [!] {system}/iter={futures[fut]}: {e}")
                pbar.update(1)
    return cell_rows


# ── Main comparison campaign ──────────────────────────────────────────────────

def campaign_main(args, adapter: BPICAdapter) -> pd.DataFrame:
    systems = [s.strip() for s in args.systems.split(",")]
    ctx = getattr(adapter, "domain_context", None)
    all_rows = []

    for dtype in DISRUPTION_TYPES:
        try:
            scenarios_meta = adapter.sample_scenarios(
                n=args.iters, disruption_type=dtype, with_meta=True)
        except ValueError as e:
            print(f"  [!] Skipping {dtype}: {e}")
            continue

        sc_label = SC_LABEL[dtype]
        uniq = len({cid for _, cid in scenarios_meta})
        print(f"\n[{sc_label}] {dtype} — {len(scenarios_meta)} episodes "
              f"from {uniq} unique log cases")

        for system in systems:
            if system not in SYSTEM_RUNNERS:
                print(f"  [!] Unknown system {system!r}, skipping.")
                continue
            t0 = time.time()
            rows = _run_cell(system, SYSTEM_RUNNERS[system], scenarios_meta,
                             sc_label, dtype, ctx, args.workers)
            all_rows.extend(rows)
            cr = sum(r["success"] for r in rows) / max(1, len(rows)) * 100
            print(f"  {system:<16} CR={cr:5.1f}%  "
                  f"({len(rows)} runs, {time.time()-t0:.1f}s)")

    return pd.DataFrame(all_rows)


# ── Ablation campaign (F7): paired DANCER-LLM vs DANCER-Heuristic ────────────

def campaign_ablation(args, adapter: BPICAdapter) -> pd.DataFrame:
    """REAL paired ablation replacing the synthetic Fig. 8 generator.

    The SAME scenario list (same seed → same agent matrices, same case
    draws) is run twice: llm_backend='real' (full contrastive refinement)
    and llm_backend='heuristic' (identical generation backend, blind ±20%
    refinement, zero refinement tokens).
    """
    ctx = getattr(adapter, "domain_context", None)
    rows = []
    for dtype in (args.ablation_types or ["structural_failure"]):
        scenarios_meta = adapter.sample_scenarios(
            n=args.iters, disruption_type=dtype, with_meta=True)
        sc_label = SC_LABEL[dtype]
        for backend, label in (("real", "DANCER-LLM"),
                               ("heuristic", "DANCER-Heuristic")):
            print(f"\n[ablation/{sc_label}] backend={backend}")
            def runner(sc, ag, ac, rg, dt, tau, i, c, _b=backend, _l=label):
                r = _dancer_row(sc, ag, ac, rg, dt, tau, i,
                                domain_context=c, llm_backend=_b)
                r["system"] = _l
                return r
            cell = _run_cell(label, runner, scenarios_meta, sc_label, dtype,
                             ctx, args.workers)
            rows.extend(cell)
            cr = sum(r["success"] for r in cell) / max(1, len(cell)) * 100
            print(f"  {label:<18} CR={cr:5.1f}%")

    df = pd.DataFrame(rows)
    # Paired Wilcoxon on per-episode utility (same iteration index = same scenario)
    for sc in df["scenario"].unique():
        sub = df[df["scenario"] == sc]
        piv = sub.pivot_table(index="iteration", columns="system",
                              values="final_utility", aggfunc="first")
        if {"DANCER-LLM", "DANCER-Heuristic"} <= set(piv.columns):
            paired = piv.dropna()
            res = ablation_wilcoxon(paired["DANCER-LLM"].tolist(),
                                    paired["DANCER-Heuristic"].tolist())
            print(f"\n[ablation/{sc}] Wilcoxon (paired, one-sided): {res}")
    return df


# ── Drop-rate robustness sweep (F7): real Fig. 7 ─────────────────────────────

def campaign_dropsweep(args, adapter: BPICAdapter) -> pd.DataFrame:
    ctx = getattr(adapter, "domain_context", None)
    drop_rates = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    n_per_rate = args.drop_iters
    scenarios_meta = adapter.sample_scenarios(
        n=n_per_rate, disruption_type="structural_failure", with_meta=True)
    rows = []
    for dr in drop_rates:
        print(f"\n[dropsweep] p_drop={dr:.2f}")
        def runner(sc, ag, ac, rg, dt, tau, i, c, _dr=dr):
            r = _dancer_row(sc, ag, ac, rg, dt, tau, i,
                            domain_context=c, drop_rate=_dr)
            r["drop_rate"] = _dr
            return r
        cell = _run_cell("DANCER", runner, scenarios_meta, "B_real",
                         "structural_failure", ctx, args.workers)
        rows.extend(cell)
        cr = sum(r["success"] for r in cell) / max(1, len(cell)) * 100
        print(f"  p_drop={dr:.2f}  CR={cr:5.1f}%")
    return pd.DataFrame(rows)


# ── Scaling experiment (F7): real Fig. 6 ──────────────────────────────────────

def campaign_scaling(args, adapter: BPICAdapter) -> pd.DataFrame:
    """Messages-per-episode vs. number of peer agents, with REAL episodes.

    Builds n-agent scenarios from sampled empirical profiles (lead + n-1
    peers), runs DANCER, CNP, ζ-CNP, and records messages_sent.
    """
    ctx = getattr(adapter, "domain_context", None)
    profiles = list(adapter.agent_profiles.values())
    if len(profiles) < 3:
        raise RuntimeError("Too few empirical profiles for scaling experiment.")
    ns = [int(x) for x in args.scaling_ns.split(",")]
    rows = []
    for n in ns:
        for rep in range(args.scaling_reps):
            rng = random.Random(args.seed * 7_919 + n * 101 + rep)
            lead = max(profiles, key=lambda p: p.weights["w_pol"])
            peers_pool = [p for p in profiles if p.agent_id != lead.agent_id]
            k = min(n - 1, len(peers_pool))
            peers = rng.sample(peers_pool, k)
            sel = [lead] + peers
            agent_dicts, active = [], []
            for prof in sel:
                committed = (round(rng.uniform(0.85, 1.0), 4)
                             if rng.random() < prof.availability else 0.0)
                agent_dicts.append(prof.to_ipa(committed_f_res=committed).to_dict())
                active.append(prof.agent_id)
            registry = []
            dtype, tau_min = "parametric_fluctuation", 0.45

            for system in ("DANCER", "CNP", "ζ-CNP"):
                if system == "DANCER":
                    r = _dancer_row("scaling", agent_dicts, active, registry,
                                    dtype, tau_min, rep, domain_context=ctx)
                elif system == "CNP":
                    r = _baseline_row("CNP", run_cnp("scaling", agent_dicts,
                                                     active, dtype, rep,
                                                     tau_min)[0])
                else:
                    r = _baseline_row("ζ-CNP", run_zeta_cnp(
                        "scaling", agent_dicts, active, registry, dtype, rep,
                        tau_min)[0])
                r["system"] = system
                r["num_agents"] = len(active)
                rows.append(r)
        print(f"[scaling] n={n} done")
    return pd.DataFrame(rows)


# ── LLM decoding probe (audit F10) ───────────────────────────────────────────

def probe_llm(args, adapter) -> None:
    """Diagnose whether the endpoint honours sampling, and whether outputs
    vary across episodes.

    Verdicts:
      A) 5 identical completions for ONE prompt at temperature 0.7
         → the serving endpoint is decoding GREEDILY (temperature ignored).
           Fix the server config (e.g. enable do_sample / per-request
           temperature) before any campaign — results are otherwise
           pseudo-deterministic and proposal diversity is zero.
      B) Completions vary per prompt but committed (cost,time) are constant
         ACROSS episodes → prompt anchoring / episode-insensitivity; the
         de-anchored prompts in this revision address it; report remaining
         constants to me.
    """
    from llm.prompts import ProductionLLM
    ctx = getattr(adapter, "domain_context", None)
    scenarios = adapter.sample_scenarios(n=3,
                                         disruption_type="structural_failure")
    print("\n=== PROBE 1: same prompt × 5 (temperature honoured?) ===")
    agents, active, registry, dtype, tau = scenarios[0]
    outs = []
    for k in range(5):
        llm = ProductionLLM()
        llm.set_context(agent_dicts=agents[:1], tau_current=0.75, tau_min=tau,
                        domain_context=ctx, registry=registry,
                        n_peers=len(active) - 1)
        props, _ = llm.generate_proposals(dtype, "probe", active)
        sig = [(p.cost_normalized, p.time_days, p.method.value) for p in props]
        outs.append(tuple(sig))
        print(f"  call {k+1}: {sig}")
    if len(set(outs)) == 1:
        print("  >>> VERDICT A: IDENTICAL outputs — endpoint ignores "
              "temperature (greedy decoding). FIX THE SERVER before rerunning.")
    else:
        print(f"  >>> sampling OK: {len(set(outs))}/5 distinct outputs.")

    print("\n=== PROBE 2: 3 different episodes × 1 (episode sensitivity) ===")
    sigs = []
    for j, (agents, active, registry, dtype, tau) in enumerate(scenarios):
        llm = ProductionLLM()
        llm.set_context(agent_dicts=agents[:1], tau_current=0.75, tau_min=tau,
                        domain_context=ctx, registry=registry,
                        n_peers=len(active) - 1)
        props, _ = llm.generate_proposals(dtype, "probe", active)
        sig = [(p.cost_normalized, p.time_days) for p in props]
        sigs.append(tuple(sig))
        print(f"  episode {j+1} (lead={agents[0]['agent_id']}, "
              f"budget={agents[0]['max_cost']:.2f}): {sig}")
    if len(set(sigs)) == 1:
        print("  >>> outputs do NOT respond to episode context — prompt "
              "anchoring persists; send me these transcripts.")
    else:
        print("  >>> episode sensitivity OK.")


# ── Orchestration ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    has_key = any(os.environ.get(k) for k in
                  ("DANCER_BASE_URL", "OPENROUTER_API_KEY", "OPENAI_API_KEY"))
    if not has_key:
        print("\n[ERROR] Set DANCER_BASE_URL, OPENROUTER_API_KEY, or OPENAI_API_KEY.\n")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    print(f"\n{'='*70}\n  DANCER — Real BPMN Log Experiment (revised harness)")
    print(f"  Log: {args.log or '(synthetic — dev only)'}  |  seed={args.seed}")
    print(f"  Model: {os.environ.get('DANCER_MODEL', 'openai/gpt-4o-mini')}  "
          f"|  temperature: 0.7 gen / 0.5 refine")
    print(f"  Canonical criterion: S_p >= tau_min over the shared agent set "
          f"(core/evaluation.py)\n{'='*70}\n")

    adapter = _fit_adapter(args)
    t0 = time.time()

    if args.probe_llm:
        probe_llm(args, adapter)
        return

    if args.dropsweep:
        df = campaign_dropsweep(args, adapter)
        df.to_csv(os.path.join(args.out_dir, "robustness_raw.csv"), index=False)
        agg = (df.groupby("drop_rate")["success"].mean() * 100).round(2)
        agg.to_csv(os.path.join(args.out_dir, "robustness_summary.csv"))
        print(f"\n[+] robustness_raw.csv / robustness_summary.csv\n{agg}")

    elif args.ablation:
        df = campaign_ablation(args, adapter)
        df.to_csv(os.path.join(args.out_dir, "ablation_raw.csv"), index=False)
        print(f"[+] ablation_raw.csv ({len(df)} rows)")

    elif args.scaling:
        df = campaign_scaling(args, adapter)
        df.to_csv(os.path.join(args.out_dir, "scaling_raw.csv"), index=False)
        agg = df.groupby(["system", "num_agents"])["messages_sent"].agg(
            ["mean", "std"]).round(2)
        agg.to_csv(os.path.join(args.out_dir, "scaling_summary.csv"))
        print(f"[+] scaling_raw.csv / scaling_summary.csv\n{agg}")

    else:
        df = campaign_main(args, adapter)
        df.to_csv(os.path.join(args.out_dir, "bpic_results_raw.csv"), index=False)

        summary_rows = []
        for (sc, sy), grp in df.groupby(["scenario", "system"]):
            summary_rows.append({
                "scenario": sc, "system": sy, "n": len(grp),
                "completion_pct": round(grp["success"].mean() * 100, 2),
                "protocol_completion_pct":
                    round(grp["protocol_success"].mean() * 100, 2),
                "avg_rounds":  round(grp["rounds"].mean(), 3),
                "avg_utility": round(grp["final_utility"].mean(), 4),
                "avg_utility_succ":
                    round(grp.loc[grp["success"], "final_utility"].mean(), 4)
                    if grp["success"].any() else float("nan"),
                "avg_tokens":  round(grp["total_tokens"].mean(), 1),
                "avg_latency_s": round(grp["latency_seconds"].mean(), 4),
                "unique_cases": grp["case_id"].nunique(),
                "feasible_pct": round(grp["oracle_feasible"].mean() * 100, 2),
                "CR_feasible":
                    round(grp.loc[grp["oracle_feasible"], "success"].mean()
                          * 100, 2)
                    if grp["oracle_feasible"].any() else float("nan"),
            })
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(args.out_dir, "bpic_results_summary.csv"), index=False)

        analysis = run_full_analysis(df)
        print_summary_table(analysis)
        for sc, data in analysis.items():
            if data.get("pairwise_fisher") is not None:
                print(f"\n[{sc}] Fisher exact (CR vs DANCER):")
                print(data["pairwise_fisher"].to_string(index=False))

        latex = to_latex_table(
            analysis,
            caption=(f"DANCER Real-Log Evaluation on "
                     f"{os.path.basename(args.log or 'synthetic')} "
                     f"($N={args.iters}$ per disruption type; canonical "
                     f"success criterion; seed {args.seed})."),
            label="tab:bpic_results")
        with open(os.path.join(args.out_dir, "bpic_results_table.tex"), "w") as f:
            f.write(latex)
        print(f"[+] LaTeX table → {args.out_dir}/bpic_results_table.tex")

    tok = int(df["total_tokens"].sum()) if "total_tokens" in df else 0
    print(f"\n{'='*70}\n  Done in {(time.time()-t0)/60:.1f} min  |  "
          f"tokens: {tok:,}\n  Outputs in: {args.out_dir}/\n{'='*70}\n")


def parse_args():
    p = argparse.ArgumentParser(description="DANCER Real BPMN Log Experiment (revised)")
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--log-type", type=str, default="bpic2019",
                   choices=["bpic2019", "bpic2017", "bpic2020", "sepsis",
                            "road_traffic"])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--noise-threshold", type=float, default=0.2)
    p.add_argument("--systems", type=str,
                   default="BPMN+,CNP,ζ-CNP,MAPE-K,ReAct-MAS,LLM-Orchestra,DANCER")
    p.add_argument("--out-dir", type=str,
                   default=os.path.join(_ROOT, "outputs", "bpic"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--allow-synthetic", action="store_true",
                   help="Permit synthetic log generation (DEV ONLY — never "
                        "report as real-log results)")
    # Sub-campaigns (F7 — replace the synthetic figure generators)
    p.add_argument("--ablation", action="store_true",
                   help="Paired DANCER-LLM vs DANCER-Heuristic (real Fig. 8)")
    p.add_argument("--ablation-types", nargs="*", default=None,
                   help="Disruption types for the ablation "
                        "(default: structural_failure)")
    p.add_argument("--dropsweep", action="store_true",
                   help="Message-drop robustness sweep (real Fig. 7)")
    p.add_argument("--drop-iters", type=int, default=50,
                   help="Episodes per drop rate [default: 50]")
    p.add_argument("--scaling", action="store_true",
                   help="O(n) message scaling experiment (real Fig. 6)")
    p.add_argument("--scaling-ns", type=str, default="3,5,8,12,16,20")
    p.add_argument("--scaling-reps", type=int, default=10)
    p.add_argument("--probe-llm", action="store_true",
                   help="Diagnose endpoint decoding (run FIRST, before any "
                        "campaign): 5x same prompt + 3 episode prompts")
    return p.parse_args()


if __name__ == "__main__":
    main()
