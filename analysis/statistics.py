"""
Statistical analysis for DANCER IEEE TSC evaluation.

Tests applied:
  - Kruskal-Wallis H-test: non-parametric k-sample test for each scenario.
  - Dunn's post-hoc with Bonferroni correction: pairwise comparisons.
  - Cliff's delta: non-parametric effect size (no normality assumption).
  - Bootstrap 95% CI: for completion rate and mean utility.
  - Wilcoxon signed-rank: DANCER-LLM vs DANCER-Heuristic (ablation).

Reporting follows IEEE TSC conventions:
  - Report H-statistic and p-value for omnibus test.
  - Flag pairwise p < 0.05 (*), p < 0.01 (**), p < 0.001 (***).
  - Effect size: negligible (<0.15) | small (0.15-0.33) | medium (0.33-0.47) | large (>0.47)
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (mannwhitneyu, wilcoxon, kruskal, bootstrap,
                         fisher_exact, beta as beta_dist)

try:
    import scikit_posthocs as sp
    _POSTHOCS_AVAILABLE = True
except ImportError:
    _POSTHOCS_AVAILABLE = False
    warnings.warn("scikit-posthocs not available; Dunn's test will be skipped.")


# ── Effect size ───────────────────────────────────────────────────────────────

def cliffs_delta(x: List[float], y: List[float]) -> Tuple[float, str]:
    """Cliff's delta: non-parametric effect size in [-1, 1].

    Returns (delta, magnitude_label).
    Thresholds from Romano et al. (2006) as used in software engineering.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x) * len(y)
    if n == 0:
        return 0.0, "negligible"
    dom = sum(int(xi > yi) - int(xi < yi) for xi in x for yi in y)
    delta = dom / n

    abs_d = abs(delta)
    if abs_d < 0.147:
        label = "negligible"
    elif abs_d < 0.330:
        label = "small"
    elif abs_d < 0.474:
        label = "medium"
    else:
        label = "large"
    return round(delta, 4), label


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(values: List[float], stat: str = "mean",
                 confidence: float = 0.95, n_resamples: int = 9999,
                 rng_seed: int = 42) -> Tuple[float, float]:
    """Bootstrap confidence interval for mean or proportion."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)

    if stat == "mean":
        fn = np.mean
    else:  # proportion
        fn = np.mean

    rng = np.random.default_rng(rng_seed)
    bs = bootstrap(
        (arr,), statistic=fn, n_resamples=n_resamples,
        confidence_level=confidence, method="percentile", random_state=rng,
    )
    return (round(bs.confidence_interval.low, 4),
            round(bs.confidence_interval.high, 4))


def clopper_pearson_ci(successes: int, n: int,
                       confidence: float = 0.95) -> Tuple[float, float]:
    """Exact (Clopper–Pearson) binomial CI for a completion rate.

    AUDIT F9: bootstrapping a degenerate all-success sample collapses to
    [1.00, 1.00], which is not a valid 95% interval and contradicts the
    reported replication variance. The exact interval for 100/100 is
    [0.9638, 1.0000]; for 0/100 it is [0.0000, 0.0362].
    """
    if n == 0:
        return (0.0, 1.0)
    alpha = 1.0 - confidence
    lo = 0.0 if successes == 0 else float(
        beta_dist.ppf(alpha / 2, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(
        beta_dist.ppf(1 - alpha / 2, successes + 1, n - successes))
    return (round(lo, 4), round(hi, 4))


def pairwise_fisher(groups_success: Dict[str, List[float]],
                    reference: str = "DANCER") -> pd.DataFrame:
    """Fisher exact tests on completion COUNTS, reference vs each baseline.

    The conventional test for a binary outcome (audit F9: Kruskal–Wallis /
    Mann–Whitney on a 0/1 indicator is unusual; KW is now run on per-episode
    utility, and CR differences are tested here).
    """
    ref = groups_success.get(reference, [])
    ref_s, ref_f = int(sum(ref)), len(ref) - int(sum(ref))
    rows = []
    for name, vals in groups_success.items():
        if name == reference or not vals:
            continue
        s, f = int(sum(vals)), len(vals) - int(sum(vals))
        try:
            odds, p = fisher_exact([[ref_s, ref_f], [s, f]],
                                   alternative="two-sided")
        except Exception:
            odds, p = float("nan"), 1.0
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else
                                       ("*" if p < 0.05 else ""))
        rows.append({"system": name,
                     "ref_CR": round(ref_s / max(1, len(ref)), 4),
                     "sys_CR": round(s / max(1, len(vals)), 4),
                     "odds_ratio": (round(odds, 4)
                                    if odds == odds else float("nan")),
                     "p_value": round(p, 8),
                     "significance": sig})
    return pd.DataFrame(rows)


# ── Omnibus test ──────────────────────────────────────────────────────────────

def kruskal_wallis(groups: Dict[str, List[float]]) -> Tuple[float, float]:
    """Kruskal-Wallis H-test across k groups.

    Returns (H-statistic, p-value).
    """
    samples = [v for v in groups.values() if len(v) >= 2]
    if len(samples) < 2:
        return 0.0, 1.0
    H, p = kruskal(*samples)
    return round(H, 4), round(p, 6)


# ── Post-hoc ─────────────────────────────────────────────────────────────────

def dunns_test(df: pd.DataFrame, group_col: str,
               value_col: str) -> Optional[pd.DataFrame]:
    """Dunn's post-hoc pairwise comparison with Bonferroni correction.

    Returns a p-value matrix (systems × systems) or None if scikit_posthocs
    is unavailable.
    """
    if not _POSTHOCS_AVAILABLE:
        return None
    result = sp.posthoc_dunn(df, val_col=value_col, group_col=group_col, p_adjust="bonferroni")
    return result.round(4)


def pairwise_mann_whitney(groups: Dict[str, List[float]],
                          reference: str = "DANCER") -> pd.DataFrame:
    """Pairwise Mann-Whitney U tests: reference system vs all others.

    Returns DataFrame with columns: system, U, p_value, significance,
    cliffs_delta, effect_size.
    """
    ref = groups.get(reference, [])
    rows = []
    for name, vals in groups.items():
        if name == reference or len(vals) < 2:
            continue
        try:
            U, p = mannwhitneyu(ref, vals, alternative="two-sided")
        except Exception:
            U, p = 0.0, 1.0
        d, mag = cliffs_delta(ref, vals)
        sig = ""
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        rows.append({
            "system": name,
            "U_statistic": round(U, 2),
            "p_value": round(p, 6),
            "significance": sig,
            "cliffs_delta": d,
            "effect_magnitude": mag,
        })
    return pd.DataFrame(rows)


# ── Full analysis pipeline ────────────────────────────────────────────────────

def run_full_analysis(results_df: pd.DataFrame) -> Dict:
    """Run the complete statistical analysis pipeline.

    Parameters
    ----------
    results_df : DataFrame with columns
        scenario, system, success (bool), rounds, messages_sent,
        final_utility, latency_seconds, total_tokens

    Returns
    -------
    analysis : dict with keys per scenario:
        {
          "completion_rate": {system: rate},
          "kruskal_wallis": {"H": float, "p": float},
          "pairwise_mw": DataFrame,
          "ci_95": {system: (low, high)},
          "utility_stats": {system: {mean, std, median}},
          "rounds_stats": {system: {mean, std, max}},
          "effect_sizes": {system: (delta, magnitude)},
        }
    """
    analysis: Dict = {}

    for scenario in results_df["scenario"].unique():
        sc_df = results_df[results_df["scenario"] == scenario].copy()

        # ── Completion rates ──────────────────────────────────────────
        cr: Dict[str, float] = {}
        ci_95: Dict[str, tuple] = {}
        groups_success: Dict[str, List[float]] = {}
        groups_utility: Dict[str, List[float]] = {}
        rounds_stats: Dict[str, dict] = {}
        utility_stats: Dict[str, dict] = {}

        for system in sc_df["system"].unique():
            sys_df = sc_df[sc_df["system"] == system]
            successes = sys_df["success"].astype(float).tolist()
            utilities = sys_df["final_utility"].tolist()
            rounds_list = sys_df["rounds"].tolist()

            cr[system] = round(float(np.mean(successes)) * 100, 2)
            # Exact binomial CI on the completion rate (audit F9)
            ci_95[system] = clopper_pearson_ci(int(sum(successes)),
                                               len(successes))
            groups_success[system] = successes
            groups_utility[system] = utilities

            utility_stats[system] = {
                "mean":   round(float(np.mean(utilities)), 4),
                "std":    round(float(np.std(utilities)), 4),
                "median": round(float(np.median(utilities)), 4),
                "q25":    round(float(np.percentile(utilities, 25)), 4),
                "q75":    round(float(np.percentile(utilities, 75)), 4),
            }
            rounds_stats[system] = {
                "mean": round(float(np.mean(rounds_list)), 3),
                "std":  round(float(np.std(rounds_list)), 3),
                "max":  int(np.max(rounds_list)),
            }

        # ── Kruskal-Wallis across all systems — on PER-EPISODE GLOBAL UTILITY
        #    (failures contribute 0), not on the binary success indicator
        #    (audit F9). CR differences are tested with Fisher exact below.
        H, p = kruskal_wallis(groups_utility)

        # ── Pairwise Mann-Whitney on utility (DANCER as reference) ────
        pw_mw = pairwise_mann_whitney(groups_utility, reference="DANCER")

        # ── Pairwise Fisher exact on completion counts ────────────────
        pw_fisher = pairwise_fisher(groups_success, reference="DANCER")

        # ── Dunn's post-hoc (on utility) ──────────────────────────────
        if _POSTHOCS_AVAILABLE:
            long_df = sc_df[["system", "final_utility"]].copy()
            dunns = dunns_test(long_df, "system", "final_utility")
        else:
            dunns = None

        # ── Effect sizes vs DANCER ────────────────────────────────────
        dancer_util = groups_utility.get("DANCER", [])
        effect_sizes = {}
        for sys_name, util_vals in groups_utility.items():
            if sys_name == "DANCER":
                continue
            d, mag = cliffs_delta(dancer_util, util_vals)
            effect_sizes[sys_name] = (d, mag)

        analysis[scenario] = {
            "completion_rate":  cr,
            "kruskal_wallis":   {"H": H, "p": p},
            "pairwise_mw":      pw_mw,
            "pairwise_fisher":  pw_fisher,
            "dunns_posthoc":    dunns,
            "ci_95":            ci_95,
            "utility_stats":    utility_stats,
            "rounds_stats":     rounds_stats,
            "effect_sizes":     effect_sizes,
        }

    return analysis


# ── Ablation Wilcoxon ─────────────────────────────────────────────────────────

def ablation_wilcoxon(llm_utilities: List[float],
                      heuristic_utilities: List[float]) -> Dict:
    """Wilcoxon signed-rank test for paired ablation (same episodes)."""
    if len(llm_utilities) != len(heuristic_utilities):
        return {"error": "Lengths differ — samples must be paired."}
    try:
        stat, p = wilcoxon(llm_utilities, heuristic_utilities, alternative="greater")
    except Exception as e:
        return {"error": str(e)}
    d, mag = cliffs_delta(llm_utilities, heuristic_utilities)
    return {
        "W_statistic": round(float(stat), 4),
        "p_value":     round(float(p), 6),
        "significance": "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns")),
        "cliffs_delta": d,
        "effect_magnitude": mag,
    }


# ── LaTeX table generator ─────────────────────────────────────────────────────

def to_latex_table(analysis: Dict, caption: str = "", label: str = "tab:results") -> str:
    """Generate IEEE-style LaTeX table from analysis dict.

    Columns: Scenario | System | CR% | 95% CI | Avg.Rounds | Avg.Utility | p-value
    """
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\centering",
        r"\begin{tabular}{llrrrrc}",
        r"\toprule",
        r"Scenario & System & CR\% & 95\% CI & Avg.~Rounds & Avg.~Utility & $p$-value \\",
        r"\midrule",
    ]

    for scenario, data in sorted(analysis.items()):
        cr = data["completion_rate"]
        ci = data["ci_95"]
        rounds = data["rounds_stats"]
        utility = data["utility_stats"]
        H = data["kruskal_wallis"]["H"]
        p = data["kruskal_wallis"]["p"]
        p_str = f"{p:.3e}" if p < 0.001 else f"{p:.4f}"

        systems = list(cr.keys())
        for i, system in enumerate(systems):
            lo, hi = ci.get(system, (0, 0))
            r_mean = rounds.get(system, {}).get("mean", 0)
            u_mean = utility.get(system, {}).get("mean", 0)
            r_str = f"{r_mean:.2f}" if r_mean > 0 else "—"
            # Bold DANCER
            sys_str = r"\textbf{" + system + "}" if system == "DANCER" else system
            cr_val = cr.get(system, 0)
            cr_str = r"\textbf{" + f"{cr_val:.1f}" + "}" if system == "DANCER" else f"{cr_val:.1f}"

            scenario_str = scenario if i == 0 else ""
            p_cell = p_str if i == 0 else ""

            lines.append(
                f"{scenario_str} & {sys_str} & {cr_str} & "
                f"[{lo:.2f}, {hi:.2f}] & {r_str} & "
                f"{u_mean:.3f} & {p_cell} \\\\"
            )
        lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{l}{\footnotesize CR: completion rate. "
        r"CI: 95\% Clopper--Pearson exact (binomial). "
        r"$p$-value: Kruskal-Wallis $H$-test on per-episode global utility "
        r"(failures $=0$) across all systems.} \\",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def print_summary_table(analysis: Dict) -> None:
    """Print a readable console summary."""
    print("\n" + "=" * 100)
    print(f"  {'Scenario':<25} {'System':<16} {'CR%':>6} {'95% CI':>14} "
          f"{'Rounds':>8} {'Utility':>8} {'p-val':>10}")
    print("=" * 100)
    for scenario, data in sorted(analysis.items()):
        cr = data["completion_rate"]
        ci = data["ci_95"]
        rounds = data["rounds_stats"]
        utility = data["utility_stats"]
        p = data["kruskal_wallis"]["p"]
        p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"

        for i, system in enumerate(cr):
            lo, hi = ci.get(system, (0, 0))
            r_mean = rounds.get(system, {}).get("mean", 0)
            u_mean = utility.get(system, {}).get("mean", 0)
            r_str = f"{r_mean:.2f}" if r_mean > 0 else "—"
            sc_str = scenario[:24] if i == 0 else ""
            p_display = p_str if i == 0 else ""
            tag = " ◀" if system == "DANCER" else ""
            print(f"  {sc_str:<25} {system:<16} {cr[system]:>5.1f}% "
                  f"[{lo:.2f},{hi:.2f}] {r_str:>8} {u_mean:>8.3f} {p_display:>10}{tag}")
        print("-" * 100)
    print()
