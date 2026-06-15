# DANCER — Decentralised Negotiation for Process Re-Choreography

This repository contains the experimental harness used to evaluate **DANCER**,
a decentralised, LLM-assisted negotiation protocol for recovering Business
Process Model and Notation (BPMN) process instances from runtime
disruptions, against six baseline coordination systems on **real-world event
logs**.

The harness grounds every experiment in an actual process-mining pipeline:
an event log is mined into a process model, conformance-checked to find
genuine deviations, and used to derive empirically realistic agent
populations and disruption scenarios. No synthetic agent parameters are used
in the reported experiments.

---

## 1. Conceptual background

### 1.1 Problem setting

A business process is modelled as a set of autonomous **Intelligent Process
Agents (IPAs)**, each responsible for one organisational role (e.g. a
purchasing department, a warehouse, a carrier). Each agent holds **private**
constraints — a cost budget, a maximum acceptable delay, an availability
level, and a set of utility weights — that are never fully shared with other
agents. This corresponds to a **Decentralised Partially Observable Semi-
Markov Decision Process (Dec-POSG)**: every agent observes only its own
state and a limited public channel.

When a runtime disruption occurs, the affected agents must agree on a
recovery proposal without a central coordinator that can see everyone's
private constraints.

### 1.2 Disruption taxonomy

Three disruption classes are mined directly from the event log's
non-conforming traces:

| Type | Description |
|---|---|
| `parametric_fluctuation` | A timing or quantity parameter has drifted (e.g. a processing delay), but the process structure is intact. |
| `structural_failure` | A required activity/agent in the normal path is missing or unavailable; the planned routing is broken. |
| `sovereignty_conflict` | An organisational constraint (e.g. a budget) has tightened mid-process, invalidating the previously planned route. |

### 1.3 Agent utility model

Each agent `k` evaluates a recovery proposal `p` with a private, additive
utility function:

```
L_k(p) = w_res · f_res(p) + w_time · f_time(p) + w_pol · f_pol(p)
```

where `f_res` reflects the agent's own resource availability, `f_time`
reflects how well `p`'s duration fits the agent's deadline, and `f_pol`
reflects how well `p`'s cost fits the agent's budget. The weights
`(w_res, w_time, w_pol)` and the constraint values are derived empirically
per agent from the event log (see §3).

### 1.4 Canonical success criterion

To allow a fair comparison across structurally different coordination
systems, every system's outcome is judged by the **same** criterion
(`core/evaluation.py`). Given the proposal `p` that a system ultimately
commits to, and the canonical agent set `A*` (all active agents, plus a
registry/third-party agent if `p` engages one):

```
S_p = (1 / |A*|) · Σ_{k ∈ A*} L_k(p)
success  ⇔  S_p ≥ τ_min   AND   p lies within its method's market envelope
            AND (for structural failures) p restores the missing capability
```

`τ_min` is the scenario-specific feasibility floor. Each system's *internal*
mechanics (negotiation, voting, planning, role matching, …) decide *which*
proposal gets committed; this module only judges the result.

---

## 2. Systems under evaluation

| System | Family | Key reference |
|---|---|---|
| **DANCER** | Decentralised LLM negotiation with a deterministic safeguard layer and bounded contrastive refinement | this work |
| **BPMN+** | Closed-world adaptive BPMN engine with design-time exception handlers | — |
| **CNP** | Classical Contract Net Protocol (announce / bid / award) | Smith (1980) |
| **ζ-CNP** | Extended CNP with one round of task decomposition | Ye, Shen & Hao (2017) |
| **MAPE-K** | Autonomic computing loop (Monitor–Analyze–Plan–Execute over a Knowledge Base) | Kephart & Chess (2003) |
| **ReAct-MAS** | Unstructured multi-agent LLM system using Thought→Action→Observation loops | Yao et al. (2023) |
| **LLM-Orchestra** | Single centralised LLM with full visibility of all agents' private constraints | — |

All LLM-driven systems (`DANCER`, `ReAct-MAS`, `LLM-Orchestra`) use the same
underlying language model, configured once via environment variables
(§5).

---

## 3. Repository layout

```
New Version/
├── run_bpic_experiment.py     # Main entry point — all campaigns
├── core/
│   ├── models.py               # Pydantic schemas: Proposal, Bid, UtilityVector, RunResult
│   ├── agents.py                # IPAgent: private constraints + utility function L_k(p)
│   ├── evaluation.py            # Canonical outcome criterion (shared across systems)
│   └── proposal_envelopes.py    # Public market feasibility bounds per recovery method
├── llm/
│   ├── prompts.py                # Prompt construction + ProductionLLM (real endpoint client)
│   ├── llm_real.py               # Heuristic policy generator (ablation backend)
│   ├── llm_mock.py                # Deterministic mock LLM (offline/dev backend)
│   ├── llm_factory.py            # Backend selection (mock / heuristic / real)
│   └── domain_context.py          # Derives domain vocabulary from the event log
├── protocols/
│   ├── dancer_protocol.py        # DANCER: LangGraph state machine (Algorithm 1)
│   ├── bpmn_plus.py
│   ├── cnp.py
│   ├── zeta_cnp.py
│   ├── mape_k.py
│   ├── react_mas.py
│   └── llm_orchestra.py
├── scenarios/
│   ├── bpic_adapter.py            # XES log → mined model → agent profiles → scenarios
│   ├── scenarios_base.py          # Synthetic scenario builders (dev/CI fallback)
│   └── scenarios_extended.py      # Additional synthetic scenarios (cascading, Byzantine, …)
├── analysis/
│   ├── statistics.py              # Significance tests, effect sizes, confidence intervals
│   └── visualization.py           # Figure generation for the paper
├── data/                           # Place input event logs here (not version-controlled)
├── outputs/                        # Generated CSV/LaTeX artefacts (not version-controlled)
├── requirements.txt
└── .env                            # Local LLM/API configuration (not version-controlled)
```

---

## 4. The BPIC adapter pipeline

`scenarios/bpic_adapter.py` converts a raw XES event log into grounded
negotiation episodes through six stages:

1. **Load** — parse the XES log with `pm4py`.
2. **Discover** — mine a Petri net process model with the Inductive Miner.
3. **Conform** — run token-based replay to identify cases that deviate from
   the mined model.
4. **Profile** — derive, per organisational resource/group, an empirical
   agent profile: a cost ceiling, a maximum delay, an availability rate, and
   utility weights, all computed from observed throughput times, costs, and
   activity frequencies.
5. **Inject** — classify each deviating case into one of the three
   disruption types (§1.2) based on the structure of its conformance
   deviation.
6. **Emit** — for each sampled episode, produce a tuple
   `(agent_dicts, active_agent_ids, registry, disruption_type, tau_min)`
   that every protocol runner consumes through an identical interface.

This means every system is evaluated on **the same population of agents and
the same disruption episodes**, sampled deterministically from a seeded RNG.

---

## 5. Installation

**Requirements:** Python ≥ 3.10.

```bash
git clone <this-repository-url>
cd "New Version"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `pm4py` (process mining), `langgraph` (DANCER state
machine), `pydantic` (data schemas), `openai` (LLM client), `pandas`,
`scipy`, `scikit-posthocs`, `matplotlib`, `seaborn`.

---

## 6. LLM configuration

All LLM-driven systems read their configuration from environment variables
(optionally via a local `.env` file loaded with `python-dotenv`). Create a
`.env` file in `New Version/` based on the following template:

```ini
# Use a real LLM (false = deterministic mock backend, no network calls)
USE_REAL_LLM=true

# Option A — self-hosted OpenAI-compatible endpoint (e.g. vLLM), highest priority
DANCER_BASE_URL=http://<host>:<port>/v1
DANCER_API_KEY=<token>
DANCER_MODEL=<served-model-name>

# Option B — OpenRouter (used if DANCER_BASE_URL is not set)
# OPENROUTER_API_KEY=<key>

# Option C — OpenAI directly (used if neither of the above is set)
# OPENAI_API_KEY=<key>
```

Resolution order is `DANCER_BASE_URL` → `OPENROUTER_API_KEY` →
`OPENAI_API_KEY`. At least one must be set; otherwise the harness exits
immediately.

**Before running any campaign**, validate that the configured endpoint
honours sampling temperature and reacts to episode context:

```bash
python run_bpic_experiment.py --log data/BPIChallenge2019.xes --probe-llm
```

This issues five identical generation requests (checking for output
diversity under `temperature=0.7`) and three episode-conditioned requests
(checking that proposals adapt to each lead agent's constraints).

---

## 7. Data acquisition

Place the event log(s) you intend to use inside `data/`. The harness is
pre-configured for the **BPI Challenge 2019** purchase-order log (~700 MB,
~250k cases). This and the other supported logs (BPIC 2017, BPIC 2020,
SEPSIS, Road Traffic Fines) are publicly distributed by the 4TU Centre for
Research Data — see the header of `scenarios/bpic_adapter.py` for the exact
dataset identifiers.

> Event log files are large and are **not** committed to this repository.
> Download the log separately and place it under `data/`.

For development without a real log, pass `--allow-synthetic`: the adapter
will generate a structurally equivalent synthetic log (same number of
agents, activity set, and deviation rate). Synthetic runs must never be
reported as real-log results.

---

## 8. Running the experiments

All commands are issued from `New Version/`. `--seed` controls the RNG used
for episode sampling, agent availability draws, and message-drop simulation
— use the same seed to reproduce an identical episode set.

### 8.1 Main comparison campaign

Runs every system in `--systems` over `--iters` episodes for each of the
three disruption types.

```bash
python run_bpic_experiment.py \
    --log data/BPIChallenge2019.xes \
    --iters 100 \
    --seed 42
```

Useful flags:
- `--systems "DANCER,CNP,ζ-CNP"` — restrict to a subset of systems
  (default: all seven).
- `--workers 8` — thread-pool size for concurrent LLM calls (non-LLM systems
  always run single-threaded for determinism).
- `--max-cases N` — cap the number of log cases the adapter loads.
- `--noise-threshold 0.2` — Inductive Miner noise filtering threshold.
- `--out-dir outputs/bpic` — output directory (created if missing).

### 8.2 Paired ablation: DANCER-LLM vs. DANCER-Heuristic

Runs the **same** sampled episodes twice — once with the full LLM
cognitive layer, once with a heuristic refinement policy — and reports a
paired Wilcoxon signed-rank test on per-episode utility.

```bash
python run_bpic_experiment.py \
    --log data/BPIChallenge2019.xes \
    --ablation \
    --iters 100 \
    --ablation-types structural_failure parametric_fluctuation sovereignty_conflict
```

### 8.3 Communication-drop robustness sweep

Runs DANCER under increasing probabilities of inter-agent message loss
(0.00 to 0.40 in steps of 0.05) on a fixed pool of `structural_failure`
episodes.

```bash
python run_bpic_experiment.py \
    --log data/BPIChallenge2019.xes \
    --dropsweep \
    --drop-iters 50
```

### 8.4 Communication-complexity scaling experiment

Runs DANCER, CNP, and ζ-CNP on episodes with an increasing number of peer
agents, recording messages exchanged per episode.

```bash
python run_bpic_experiment.py \
    --log data/BPIChallenge2019.xes \
    --scaling \
    --scaling-ns "3,5,8,12,16,20" \
    --scaling-reps 10
```

---

## 9. Output artefacts

All artefacts are written to `--out-dir` (default `outputs/bpic/`):

| Command | Files | Contents |
|---|---|---|
| main campaign | `bpic_results_raw.csv` | One row per episode × system: success, rounds, messages, tokens, utility, latency, committed-proposal provenance, source `case_id`. |
| | `bpic_results_summary.csv` | Per scenario × system aggregates (completion rate, average rounds/utility/tokens/latency, unique source cases). |
| | `bpic_results_table.tex` | LaTeX summary table. |
| `--ablation` | `ablation_raw.csv` | Paired per-episode rows for `DANCER-LLM` and `DANCER-Heuristic`. |
| `--dropsweep` | `robustness_raw.csv`, `robustness_summary.csv` | Per-episode and per-drop-rate completion rates. |
| `--scaling` | `scaling_raw.csv`, `scaling_summary.csv` | Per-episode and per-(system, agent-count) message-count statistics. |

`analysis/visualization.py` consumes these CSV files to produce the figures
referenced in the accompanying paper.

---

## 10. Reproducibility notes

- All scenario sampling, agent-availability draws, and message-drop
  simulation are derived from `--seed` (default `42`) via per-episode seeded
  random number generators — re-running with the same seed and log
  reproduces the same episode set.
- Thread-pool parallelism (`--workers`) affects only the timing and
  interleaving of LLM calls; the scenario matrix itself is built
  single-threaded before execution and is unaffected by `--workers`.
- The canonical success criterion, feasibility envelopes, and utility model
  are identical across all seven systems (§1.4); only each system's
  internal coordination logic differs.

---

## 11. References

- Smith, R. G. (1980). *The Contract Net Protocol: High-Level Communication
  and Control in a Distributed Problem Solver*. IEEE Transactions on
  Computers, 29(12).
- Kephart, J. O. & Chess, D. M. (2003). *The Vision of Autonomic Computing*.
  IEEE Computer, 36(1).
- Ye, F., Shen, W., & Hao, Q. (2017). *Extended Contract Net Protocol for
  Dynamic Task Allocation in Multi-Agent Systems*. IEEE CSCWD.
- Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in
  Language Models*. ICLR.
- Wang, L. et al. (2024). *A Survey on Large Language Model based
  Autonomous Agents*. Frontiers of Computer Science, 18(6).
