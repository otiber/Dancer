"""
make_figures.py — Generate all paper figures and the LaTeX table from REAL
result CSVs. 

Usage (from the project root, after the campaigns):
    python make_figures.py --results outputs/bpic/bpic_results_raw.csv \
        --ablation outputs/bpic/ablation_raw.csv \
        --robustness outputs/bpic/robustness_raw.csv \
        --scaling outputs/bpic/scaling_raw.csv \
        --out-dir outputs/figures
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from analysis.statistics import run_full_analysis, print_summary_table, to_latex_table
from analysis.visualization import render_all_figures


def _load(path: str, label: str):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[+] {label}: {path} ({len(df)} rows)")
        return df
    print(f"[!] {label}: MISSING ({path!r}) — its figure will be SKIPPED, "
          f"never synthesized.")
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="outputs/bpic/bpic_results_raw.csv")
    p.add_argument("--ablation", default="outputs/bpic/ablation_raw.csv")
    p.add_argument("--robustness", default="outputs/bpic/robustness_raw.csv")
    p.add_argument("--scaling", default="outputs/bpic/scaling_raw.csv")
    p.add_argument("--out-dir", default="outputs/figures")
    p.add_argument("--caption", default="DANCER Real-Log Evaluation on BPIC 2019")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = _load(args.results, "main results")
    if results is None:
        sys.exit("[ERROR] main results CSV is required.")
    ablation   = _load(args.ablation, "ablation")
    robustness = _load(args.robustness, "robustness")
    scaling    = _load(args.scaling, "scaling")

    # Optional v4 column: report feasibility-adjusted CR alongside raw CR
    if "oracle_feasible" in results.columns:
        feas = results.groupby(["scenario", "system"]).apply(
            lambda g: pd.Series({
                "feasible_pct": g["oracle_feasible"].mean() * 100,
                "CR_all": g["success"].mean() * 100,
                "CR_feasible": (g.loc[g["oracle_feasible"], "success"].mean()
                                * 100 if g["oracle_feasible"].any()
                                else float("nan")),
            })).round(2)
        feas.to_csv(os.path.join(args.out_dir, "cr_feasible_breakdown.csv"))
        print("[+] CR_feasible breakdown → cr_feasible_breakdown.csv")
        print(feas.to_string())

    analysis = run_full_analysis(results)
    print_summary_table(analysis)

    latex = to_latex_table(analysis, caption=args.caption,
                           label="tab:bpic_results")
    tex_path = os.path.join(args.out_dir, "bpic_results_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"[+] LaTeX table → {tex_path}")

    figs = render_all_figures(
        results_df=results,
        analysis=analysis,
        scaling_df=scaling,
        robustness_df=robustness,
        ablation_df=ablation,
        out_dir=args.out_dir,
    )
    print(f"[+] Figures rendered: {sorted(figs.keys()) if figs else figs}")
    print(f"[+] All outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()
