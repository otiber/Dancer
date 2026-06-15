# Methodology, System Architecture, and Experimental Protocol

> **Companion document.** `README.md` describes *how to install and run* this
> repository. This document describes *what the code implements* — it maps
> every formal construct in the accompanying paper, *"DANCER:
> Decentralized LLM-Guided Negotiation for Bounded-Round Adaptive Service
> Choreography Under Open-World Disruptions,"* onto the source files that
> realise it, and details the experimental protocol used to produce the
> paper's results. It is intended for readers who want to verify, extend, or
> reproduce the study, and assumes familiarity with the paper's notation.
> For the dataset-specific pipeline (BPI Challenge 2019 → grounded
> negotiation episodes), see `DATASET.md`.

---

## 1. Purpose and scope

DANCER is evaluated as a **Decentralised Partially Observable Stochastic
Game (Dec-POSG)**: a set of autonomous *Intelligent Process Agents* (IPAs),
each holding private constraints, must reach a bounded-round consensus on a
recovery proposal after a runtime disruption to a business process, without
any agent observing another agent's private state.

This repository implements:

1. **DANCER** itself — a LangGraph state machine realising Algorithm 1 of
   the paper, including its Cognitive Layer (LLM-driven proposal generation
   and contrastive refinement) and its Deterministic Safeguard Layer.
2. **Six baseline coordination systems** spanning closed-world BPMN
   exception handling, classical and extended Contract-Net protocols,
   autonomic-computing (MAPE-K), and two alternative LLM-based
   multi-agent architectures (ReAct-MAS, LLM-Orchestra).
3. A **shared evaluation harness** (`core/`) that defines agent utility,
   judges every system's outcome under one canonical criterion, and
   enforces market-grounded feasibility constraints identically for all
   seven systems.
4. A **process-mining adapter** (`scenarios/bpic_adapter.py`) that converts
   a real-world event log (BPI Challenge 2019, by default) into
   empirically-grounded agent populations and disruption episodes.
5. An **experiment orchestrator** (`run_bpic_experiment.py`) implementing
   four experimental campaigns, and a **statistical analysis module**
   (`analysis/statistics.py`) implementing the significance tests reported
   in the paper.

---

## 2. Formal model and its realisation in code

### 2.1 Dec-POSG formulation

Each IPA `k` is a tuple `⟨θ_k, C_k, π_θ, G⟩` (Definition 1), where:

- `θ_k = (max_cost, max_delay_days, availability)` is the agent's **private
  constraint vector** — never disclosed to other agents;
- `C_k = (w_res, w_time, w_pol)` is the agent's **utility-weight
  vector**, also private;
- `π_θ` is the **Cognitive Layer** — the stochastic LLM policy that
  proposes recovery actions;
- `G` is the shared (public) negotiation goal.

In code, this tuple is realised by `core.agents.IPAgent`
(`core/agents.py`):

```python
IPAgent(
    agent_id, role,
    max_cost, max_delay_days, availability,   # θ_k
    weights,                                    # C_k = (w_res, w_time, w_pol)
    is_active, committed_f_res,
)
```

`π_θ` is not a method of `IPAgent` — it is implemented separately, per the
Cognitive/Safeguard separation of concerns described in §3, as one of the
LLM backends in `llm/` (`MockLLM`, `HeuristicPolicyGenerator`,
`ProductionLLM`), selected via `llm/llm_factory.py`.

For experiments on the BPI Challenge 2019 log, `θ_k` and `C_k` are not
hand-authored: they are **empirically derived** per spend-area subsidiary by
`scenarios.bpic_adapter.AgentProfile` (see `DATASET.md`, §7).

### 2.2 Definition 2 — Local utility decomposition (Eqs 2–5)

Each agent evaluates a proposal `p` with a private, additive utility
function (`IPAgent.compute_local_utility`, `core/agents.py:77-79`):

```
L_k(p) = w_res · f_res(p)  +  w_time · f_time(p)  +  w_pol · f_pol(p)        (Eq. 2)
```

The three sub-functions are implemented as follows.

**Resource availability `f_res`** (Eq. 3) is *not* a function of `p` — it is
the agent's **committed availability draw**, fixed once per episode at
construction time (`core/agents.py:38-46`):

```python
if not is_active:
    committed_f_res = 0.0
elif random.random() < availability:
    committed_f_res = random.uniform(0.85, 1.0)
else:
    committed_f_res = 0.0
```

Committing this draw once — rather than re-sampling each negotiation round —
ensures that an agent's resource state (e.g. "carrier is on strike") is
**consistent across all rounds of one episode**, matching the real-world
semantics of a disruption. The committed value is serialised via
`IPAgent.to_dict()` / `from_dict()` so that every system evaluating the same
episode sees the *same* availability draw.

**Time-fit `f_time`** (Eq. 4), piecewise in the proposal's `time_days`
relative to the agent's `max_delay_days` (`core/agents.py:53-59`):

```python
if time_days <= max_delay_days:
    f_time = max(0, 1 - time_days / (2 * max_delay_days))
else:
    overshoot = (time_days - max_delay_days) / max_delay_days
    f_time = max(0, 1 - overshoot)
```

Within the deadline, `f_time` decays linearly from `1.0` (instantaneous) to
`0.5` (at exactly `max_delay_days`); beyond the deadline it decays linearly
to `0.0` at `2 × max_delay_days`.

**Policy/cost-fit `f_pol`** (Eq. 5), piecewise in `cost_normalized` relative
to `max_cost` (`core/agents.py:61-67`):

```python
if cost_normalized <= max_cost:
    f_pol = max(0, 1 - cost_normalized / (1.5 * max_cost))
else:
    overage = (cost_normalized - max_cost) / max_cost
    f_pol = max(0, 1 - 2 * overage)
```

Within budget, `f_pol` decays from `1.0` to `1/3` at exactly `max_cost`;
beyond budget it decays twice as steeply, reaching `0.0` at `1.5 ×
max_cost` overage.

**Acceptance and rejection categorisation**
(`IPAgent.evaluate_proposal`, `core/agents.py:81-114`) computes `L_k(p)`,
accepts iff `L_k(p) ≥ min_utility_k`, and — on rejection — classifies the
*binding constraint* using fixed thresholds on the sub-scores
(`f_pol < 0.25` → cost violation, `f_time < 0.25` → time violation,
`f_res < 0.25` → resource unavailable, else generic *"utility below
threshold"*). This categorical reason is what populates the **rejection
vector** of Definition 4 (§2.6).

### 2.3 Definition 3 — Aggregate score and consensus (Eq. 6)

For a proposal `p` and a set of received bids `B_p = {L_k(p)}`, the
**aggregate score** is the mean local utility:

```
S_p = (1 / |B_p|) · Σ_{k ∈ B_p} L_k(p)                                       (Eq. 6)
```

This is computed twice in the codebase, with two distinct purposes:

1. **Protocol-internal** (`dancer_protocol.aggregate_scores_node`,
   `protocols/dancer_protocol.py:238-266`): `S_p` is computed over
   *received* bids only — under simulated message loss (§3.5), fewer bids
   may arrive than there are active agents. This drives DANCER's own
   `best_score` and its consensus check `best_score ≥ τ_r`.
2. **Outcome-canonical** (`core.evaluation.canonical_outcome`,
   `core/evaluation.py:115-170`): once a system *commits* to a proposal,
   `S_p` is recomputed over the **full canonical agent set** `A*` (including
   any agent whose bid was lost, and the disrupted carrier itself — see
   §4.1). This second computation is what determines `success` for *every*
   system, including DANCER.

The distinction matters: DANCER's internal `protocol_success` /
`protocol_utility` (what DANCER *believes* it achieved, based on the bids it
actually received) can differ from its canonical `success` /
`final_utility` (what actually happened, judged against the full agent
set). Both are recorded in `RunResult` for diagnostic purposes (§4.4).

### 2.4 Eq. 7 — Temporal threshold decay

The acceptance threshold `τ_r` decays geometrically across negotiation
rounds, forcing convergence:

```
τ_{r+1} = max(τ_min, τ_r · e^{-γr})                                          (Eq. 7)
```

with `γ = 0.30` and `τ_0 = 0.75` (defaults in `run_dancer`,
`protocols/dancer_protocol.py:419-433`). `τ_min` is **scenario-specific**
(`DATASET.md`, §9): `0.40` for `structural_failure`, `0.45` for
`parametric_fluctuation` and `sovereignty_conflict`.

Iterating Eq. 7 from `τ_0 = 0.75` produces the following threshold
trajectories over the `R_max + 1 = 6` admissible rounds:

| Round `r` | `τ_r` (`τ_min = 0.40`, structural) | `τ_r` (`τ_min = 0.45`, parametric / sovereignty) |
|---|---|---|
| 0 | 0.750 | 0.750 |
| 1 | 0.750 | 0.750 |
| 2 | 0.556 | 0.556 |
| 3 | 0.400 | 0.450 |
| 4 | 0.400 | 0.450 |
| 5 | 0.400 | 0.450 |

The decay is computed in two places, both of which preserve this exact
trajectory: `check_consensus_node` (the normal path, on every round that
produced at least one valid proposal) and `validate_proposals_node`'s
empty-proposal-set branch (the F14 regeneration path — see §3.6). In both
cases `new_tau = max(τ_min, τ_r · e^{-γ·r})` is applied **before** `round` is
incremented, i.e. using the *current* round index `r` as the decay exponent
— exactly Eq. 7.

### 2.5 Definition 4 — Rejection vectors and contrastive refinement (Eq. 8)

When the best proposal `p*` of round `r` fails consensus
(`best_score < τ_r`), `check_consensus_node`
(`protocols/dancer_protocol.py:269-304`) collects a **structured rejection
vector** `V` for every agent that rejected `p*`:

```
V = { (agent_id, proposal_id, reason, L_k(p*), (f_res, f_time, f_pol)) }      (Eq. 8)
```

This is the *only* information that crosses the privacy boundary back to the
Cognitive Layer between rounds (see §3.4): the **normalised margins**
`(f_res, f_time, f_pol)` let the LLM identify *which* dimension of `p*` is
binding for the rejecting agent — without ever learning that agent's
absolute constraint values `θ_j`. `refine_proposals_node`
(`protocols/dancer_protocol.py:307-342`) passes `V` to
`llm.refine_proposals(previous_proposals, rejection_reasons, round_num,
disruption_type)`, which returns 1–2 new proposals merged into the
candidate set (deduplicated by proposal `id`).

### 2.6 Propositions 1 & 2 — bounded termination and communication complexity

**Proposition 1 (bounded termination).** Every episode terminates within
`R_max + 1 = 6` rounds, reaching `SUCCESS`, `STALL`, or `ESCALATED`. The
implementation guarantees this via the routing functions
`_route_after_validate` and `_route_after_consensus`
(`protocols/dancer_protocol.py:359-373`): the round counter `round` is
incremented on *every* pass through `check_consensus_node`, **and also** on
the F14 empty-proposal-set branch of `validate_proposals_node` — so a
pathological case in which the Cognitive Layer or the Safeguard Layer
produces *zero* valid proposals cannot stall the round counter and loop
forever. `STALL` is reached deterministically once `round > R_max`.

**Proposition 2 (linear communication complexity, `O(n)`).** The number of
messages exchanged per episode is bounded by a constant factor of the number
of active agents `n`, governed by the cap `|P| ≤ K_max` (default `5`) on the
number of live proposals per round, enforced in
`validate_proposals_node` (`protocols/dancer_protocol.py:172-179`):

```python
if len(valid) > k_max:
    kept = valid[-k_max:]                 # newest K_max proposals
    if best is not None and best not in kept:
        kept[0] = best                    # retain incumbent best
    valid = kept
```

Each round contributes at most: 1 `NEGOT_INIT` (generation broadcast) + `|P|
× n` `PEER_BID` messages + 1 `REFINE` broadcast. With `|P| ≤ K_max = 5` and
`R_max + 1 = 6` rounds, the worst-case per-episode message count is bounded
by `6 × (2 + 5n) ≈ 30n`, consistent with the paper's stated linear bound
(reported as `60n` in the paper to account for bidirectional accounting).
The empirical scaling campaign (§7.4) measures this directly.

---

## 3. The DANCER protocol as a LangGraph state machine

### 3.1 State schema

`protocols/dancer_protocol.py` defines `DANCERState` (a `TypedDict`) holding
three categories of fields:

- **Hyper-parameters**, fixed for the episode: `r_max`, `tau_initial`,
  `tau_min`, `gamma`, `k_max`, `llm_backend`, `communication_drop_rate`.
- **Runtime state**, mutated node-to-node: `round`, `tau`, `proposals`,
  `valid_proposals`, `last_raw_proposals`, `bids`, `best_proposal`,
  `best_score`, `status`, `rejection_reasons`, `messages_sent`,
  `total_tokens`, `final_utility`.
- **Episode context**, supplied by the scenario adapter:
  `agent_dicts`, `active_agent_ids`, `registry`, `disruption_type`, `goal`,
  `domain_context`.

All `Proposal` and `Bid` payloads are serialised as plain `dict`s
(`model_dump()` / re-hydrated via `Proposal(**d)`) for LangGraph's state
persistence layer.

### 3.2 Control-flow graph

`build_dancer_graph()` (`protocols/dancer_protocol.py:380-412`) wires eight
nodes:

```
            ┌─────────────┐
   entry ─▶ │   generate   │  Cognitive Layer (π_θ): proposal generation
            └──────┬───────┘
                    ▼
            ┌─────────────┐
            │   validate   │  Deterministic Safeguard Layer
            └──────┬───────┘
        ┌───────────┼────────────┐
   valid│=∅, r≤R_max│        valid≠∅
        ▼           ▼            ▼
   ┌─────────┐ ┌──────────┐ ┌──────────┐
   │ refine  │ │ escalate │ │ evaluate │  every active agent bids
   └────┬────┘ └────┬─────┘ └────┬─────┘
        │           │            ▼
        │           │      ┌───────────┐
        │           │      │ aggregate │  S_p = mean(L_k(p)) over bids
        │           │      └─────┬─────┘
        │           │            ▼
        │           │   ┌──────────────────┐
        │           │   │ check_consensus  │  S_p* ≥ τ_r ?
        │           │   └────────┬─────────┘
        │           │      ┌─────┼──────┐
        │           │  SUCCESS  STALL  RUNNING
        │           │      ▼      │       │
        │           │  ┌────────┐ │       │
        │      ◀────┴──┤ commit │ │       │
        │              └────┬───┘ │       │
        │                   ▼     ▼       │
        │                  END   escalate │
        └───────────────────────────◀─────┘
```

### 3.3 Node-by-node walkthrough

| Node | Implements (Algorithm 1) | Responsibility |
|---|---|---|
| `generate_proposals_node` | line 4 (`NEGOT_INIT`) | Cognitive Layer `π_θ` produces 2–3 candidate proposals from the lead agent's own constraints. |
| `validate_proposals_node` | lines 5–9 (Safeguard) | Discards schema-invalid, off-envelope, or hallucinated proposals; enforces `K_max`; handles empty-set regeneration (F14). |
| `evaluate_proposals_node` | lines 10–12 (`PEER_BID`) | Every active agent computes `L_k(p)` for every valid `p`, with optional simulated message loss. |
| `aggregate_scores_node` | line 13 (Eq. 6) | Computes `S_p` per proposal over received bids; selects `p* = argmax S_p`. |
| `check_consensus_node` | lines 14–17 (Eq. 7, Eq. 8) | Tests `S_{p*} ≥ τ_r`; on failure, collects rejection vectors `V` and decays `τ`. |
| `refine_proposals_node` | line 18 (Contrastive Refinement) | Cognitive Layer mutates/extends the proposal set using `V`. |
| `commit_node` | line 19 (`CONSENSUS` + `TOKEN_XFER`) | Marks `SUCCESS`, records `final_utility = best_score`. |
| `escalate_node` | line 20 | Marks `ESCALATED`, `final_utility = 0.0`. |

### 3.4 The Cognitive Layer and the information-asymmetry guarantee

Both `generate_proposals_node` and `refine_proposals_node` construct the LLM
context via an identical, narrow contract
(`protocols/dancer_protocol.py:96-106`, `315-325`):

```python
lead_only = state["agent_dicts"][:1]      # θ_lead ONLY
llm.set_context(
    agent_dicts=lead_only,
    tau_current=state["tau"], tau_min=state["tau_min"],
    domain_context=state.get("domain_context"),
    registry=state["registry"],
    n_peers=max(0, len(state["active_agent_ids"]) - 1),
)
```

The Cognitive Layer therefore receives **only**:

1. the lead agent's own `θ_lead` and weight vector;
2. the **public** registry of discoverable third-party agents (their
   capability adverts — `max_delay_days`, `availability` — are *by design*
   public, analogous to a service marketplace listing);
3. at refinement time, the structured rejection vectors `V` (Eq. 8), which
   carry only normalised margins `f_res/f_time/f_pol`, never a peer's
   absolute `θ_j`.

This is the code-level realisation of the Dec-POSG's observation function
`O_k`, which by construction excludes `θ_j` for `j ≠ k`. The privacy
contract is enforced structurally — by *what is placed in the prompt* — not
by post-hoc redaction, and the same `agent_dicts[:1]` slicing is applied
identically whether the backend is `MockLLM`, `HeuristicPolicyGenerator`, or
`ProductionLLM`.

`llm.set_context()` is implemented defensively
(`ProductionLLM.set_context`, `llm/prompts.py`) to truncate any
`agent_dicts` list it is handed to its first element, so even a future
caller passing the full agent population by mistake cannot leak peer
constraints into a prompt.

### 3.5 The Deterministic Safeguard Layer

`validate_proposals_node` (`protocols/dancer_protocol.py:119-207`) is a
**pure, deterministic function** — no LLM calls — that filters the
Cognitive Layer's raw output through four checks, applied in order:

1. **Schema validity**: `0.0 ≤ cost_normalized ≤ 1.0` and `time_days > 0.0`.
   Violations are discarded outright (no clamping/repair).
2. **Market feasibility envelope** (`core/proposal_envelopes.is_feasible`,
   §4.2): a proposal whose `(cost_normalized, time_days)` falls outside the
   public envelope for its declared `method` is discarded.
3. **Capability restoration** (audit finding F13a): under
   `structural_failure`, the disrupted resource is *vacant* by definition —
   any proposal that does not set `requires_new_agent=True` is discarded,
   since it cannot execute.
4. **Registry membership**: if `requires_new_agent=True`, `new_agent_id`
   must name an agent present in the public `registry`; otherwise the
   proposal is discarded as a hallucinated reference.

Surviving proposals are then capped at `K_max` (Proposition 2, §2.6),
preferring the most recently generated/refined proposals while always
retaining the current incumbent best.

A key design decision (audit finding F2) is what this layer *does not* do:
the original implementation additionally clamped every proposal's cost to
`min(max_cost)` across **all** peer agents — a "silent repair" that (a)
required reading every peer's private budget, violating the privacy
contract of §3.4, and (b) artificially inflated completion rates by pulling
every proposal into a feasible region regardless of what the LLM actually
proposed. This clamp has been removed; the Safeguard Layer now only
**discards**, never repairs.

### 3.6 Termination in practice: the empty-proposal-set branch (F14)

If *every* candidate proposal is discarded by the Safeguard Layer (e.g. the
LLM proposed only off-envelope or non-restorative options), the original
implementation routed directly to `escalate` at round 0 — DANCER would
forfeit the episode having consumed exactly one generation call and zero
negotiation rounds.

The current implementation (`protocols/dancer_protocol.py:184-206`,
`_route_after_validate` at line 359) instead treats this as **any other
failed consensus**: it synthesises a `SAFEGUARD`-authored rejection vector
summarising the discard reasons, decays `τ` per Eq. 7, increments `round`,
and routes to `refine_proposals_node` — *provided* `round ≤ R_max`. Only
once the round budget is exhausted does the episode escalate. This preserves
Proposition 1 (the round counter strictly increases on every pass, so
termination is still bounded by `R_max + 1`) while giving the Cognitive
Layer the chance the protocol's design intends: to *use the rejection
feedback to regenerate*, rather than treating a single bad sample as final.

---

## 4. Evaluation harness

The modules in `core/` are **shared verbatim by all seven systems**. They
encode the parts of the experimental design that must be held constant for
the cross-system comparison to be meaningful: what counts as "success," what
counts as an economically realistic proposal, and what data is recorded per
episode.

### 4.1 The canonical success criterion

`core/evaluation.py` defines `canonical_outcome(proposal, agent_dicts,
registry, tau_min, active_agent_ids, disruption_type)` →
`CanonicalOutcome(success, aggregate_utility, per_agent,
n_individually_accepting, agent_set, tau_min)`.

Given the proposal `p` that a system ultimately commits to (or `None`, if
the system fails to commit), the **canonical agent set** `A*` is
reconstructed by `build_outcome_agents`:

- all agents in `active_agent_ids` (including a disrupted carrier whose
  `committed_f_res = 0`, which **remains in `A*`** — the same dilution
  applies to every system);
- plus the registry agent named by `p.new_agent_id`, *iff*
  `p.requires_new_agent` and that id is present in `registry`.

The outcome is then:

```
S_p = (1 / |A*|) · Σ_{k ∈ A*} L_k(p)
success  ⇔  S_p ≥ τ_min   AND   p is feasible
```

where *"`p` is feasible"* combines two environment-level rules applied
identically to every system (audit finding F12):

- **(a) Envelope feasibility** — `is_feasible(p)` (§4.2): `(cost, time)`
  must lie inside the market envelope for `p.method`.
- **(b) Capability restoration** — under `structural_failure`, `p` must set
  `requires_new_agent=True`; otherwise the disrupted capability remains
  unaddressed and `p` is infeasible regardless of its utility score.

**Why a single module is necessary.** Prior to this unification (audit
finding F3), each system computed "success" under a *different* criterion:
DANCER used `mean(L_k) ≥ τ_r`; LLM-Orchestra and ReAct-MAS required
*unanimous* individual acceptance (`L_k ≥ min_utility_k` for *every* agent);
MAPE-K used a fixed `0.55` threshold; CNP/ζ-CNP used individual bid
acceptance plus a manager veto. Under `structural_failure`, the disrupted
resource agent — which has the **highest** `w_res` by construction and
`f_res = 0` — can never individually accept *any* proposal. Any criterion
requiring its unanimous acceptance therefore guarantees `0%` completion
**independent of proposal quality**. `core/evaluation.py` replaces all of
these with the single criterion above; each system's *internal* mechanics
(negotiation, voting, role-matching, planning) still decide *which*
proposal — if any — gets committed, but the **judgement** of that
commitment is now identical across systems.

### 4.2 Market feasibility envelopes

`core/proposal_envelopes.py` defines, for each of the seven
`ShippingMethod` values, a public `(cost_lo, cost_hi) × (time_lo, time_hi)`
envelope:

| Method | Cost envelope | Time envelope (days) |
|---|---|---|
| `Express_Air_Freight` | [0.55, 0.95] | [0.5, 2.0] |
| `Expedited_Ground` | [0.30, 0.65] | [1.0, 3.0] |
| `Consolidated_Rail` | [0.25, 0.60] | [2.0, 6.0] |
| `Land_Transport` | [0.30, 0.60] | [2.0, 5.0] |
| `Third_Party_Logistics` | [0.30, 0.70] | [2.0, 6.0] |
| `Alternative_Port_B` | [0.30, 0.70] | [3.0, 8.0] |
| `Standard_Shipping` | [0.10, 0.60] | [3.5, 10.0] |

`is_feasible(p)` and `infeasibility_reason(p)` test membership;
`envelope_prompt_block()` renders this table as a *"MARKET FEASIBILITY
ENVELOPES (public)"* text block injected verbatim into every LLM-based
system's prompt (DANCER, ReAct-MAS, LLM-Orchestra) — i.e. the envelopes are
**public market information**, not a private constraint of any agent.

**Why envelopes are needed** (audit finding F12): without them, sampling
LLMs at `temperature=0.7` consistently converged on degenerate near-zero
declared costs (`0.05`–`0.08`) regardless of the disruption or agent
constraints — producing instant, meaningless "consensus" that reflected
nothing about the negotiation dynamics. The envelopes ground every declared
`(cost, time)` pair in a plausible market range *per shipping method*,
making `cost_normalized` and `time_days` economically meaningful and
comparable across systems.

### 4.3 Feasibility oracle and `CR_feasible`

Some episodes are **unsolvable by construction** — no combination of method,
cost, and time can clear `τ_min` for the sampled agent population, e.g. when
a sovereignty-tightened budget is incompatible with every envelope's cost
floor. Comparing completion rates across such episodes conflates "the
protocol failed" with "no protocol could have succeeded."

`feasibility_oracle(agent_dicts, registry, tau_min, disruption_type,
active_agent_ids)` (`core/evaluation.py:51-84`) resolves this: because both
`f_pol` and `f_time` are monotone non-increasing in `cost` and `time`
respectively, each method's *optimum* lies at its envelope's
`(cost_lo, time_lo)` corner. The oracle constructs all seven corner
proposals (with `requires_new_agent=True` under `structural_failure`),
evaluates each through `canonical_outcome`, and returns `True` iff *any*
corner proposal succeeds. This is computed once per episode and recorded as
`oracle_feasible` in `RunResult`; the summary statistics report both the raw
completion rate (`CR`, over all episodes) and `CR_feasible` (completion rate
over the oracle-solvable subset).

### 4.4 `RunResult` schema and provenance

`core/models.py` defines `RunResult` — the row schema written by every
system to the output CSVs:

```python
class RunResult(BaseModel):
    scenario: str            # A_real | B_real | C_real
    system: str
    iteration: int
    success: bool            # canonical outcome (§4.1)
    rounds: int
    messages_sent: int
    total_tokens: int
    final_utility: float     # canonical S_p
    latency_seconds: float
    proposal_method: Optional[str]
    proposal_cost: Optional[float]
    proposal_time: Optional[float]
    proposal_new_agent: Optional[str]
    n_accepting: Optional[float]
    refusals: Optional[float]
```

For DANCER, `run_bpic_experiment._dancer_row` additionally augments the row
with `protocol_success` / `protocol_utility` (DANCER's own internal
`status`/`best_score` — what the protocol *believed*) alongside the
canonical `success` / `final_utility` (what the harness *judged*), plus
`n_individually_accepting` and `case_id` provenance linking the episode back
to its source XES trace (`DATASET.md`, §9).

---

## 5. Baseline systems

All baselines share `core.agents.IPAgent` (and hence `L_k(p)`,
Eq. 2–5), the same `committed_f_res` draw per episode, and the canonical
outcome criterion of §4.1. They differ only in *how* a proposal is formed
and *whether* it gets committed.

| System | Family | Mechanism | Messages / episode | Reference |
|---|---|---|---|---|
| **BPMN+** | Closed-world adaptive BPMN | One design-time-fixed boundary-event handler (`Expedited_Ground`, `cost=0.45`, `time=2.0`) for `parametric_fluctuation` only; no handler exists for `structural_failure` / `sovereignty_conflict` → immediate stall. | 2 (if handled) / 0 (stall) | — |
| **CNP** | Classical Contract Net (Smith, 1980) | Manager broadcasts one *fixed* task (`TASK_ANNOUNCE`); each contractor bids or refuses (role-mismatch ⇒ automatic refusal); manager awards or stalls subject to its own utility veto. No task reformulation. | `1 + len(contractors) + 1 = n + 1` | Smith (1980) |
| **ζ-CNP** | Extended CNP with task decomposition | As CNP, but if Phase 2 finds no eligible bidder, the manager applies **one** pre-registered decomposition (per disruption type) and re-announces subtasks — can recover some `structural_failure` cases via a warehouse + 3PL split. Still rule-based; no LLM. | up to `2 × (n + 1)` | Ye, Shen & Hao (2017) |
| **MAPE-K** | Autonomic computing loop | A pre-populated Knowledge Base mirrors DANCER's registry (same `3PL_Agent` entry); Monitor→Analyze→Plan applies heuristic cost optimisation (no LLM); Execute validates the plan against `L_k`; the KB is updated post-execution (runtime enrichment). Centralised — no peer negotiation. | 1 round | Kephart & Chess (2003) |
| **ReAct-MAS** | Unstructured LLM multi-agent | Each agent runs a Thought→Action→Observation loop. Lead proposes free-text actions; peers independently vote accept/reject (no formal `L_k` exposure to the LLM, no structured rejection schema); lead refines or commits. Round budget aligned to DANCER: `R_max + 1 = 6` evaluations (`_R_MAX = 5`, `protocols/react_mas.py:48`). | `O(R_max × n)` LLM calls | Yao et al. (2023) |
| **LLM-Orchestra** | Centralised LLM monolith | A single Orchestrator LLM receives the **complete** state of all agents — including private `θ_k` — and emits one binding decision in a single call, validated post-hoc against `L_k`. No negotiation, no registry discovery. Represents the "why not just ask one big LLM with full visibility?" design point, at the cost of violating organisational sovereignty. | 1 LLM call | — |
| **DANCER** | Decentralised LLM negotiation (this work) | §3. Bounded-round negotiation with a private Cognitive Layer, deterministic Safeguard Layer, contrastive refinement, and temporal threshold decay. | `≤ 6 × (2 + K_max·n)` (Proposition 2) | this work |

`run_bpic_experiment.SYSTEM_RUNNERS` dispatches by name to
`run_bpmn_plus`, `run_cnp`, `run_zeta_cnp`, `run_mape_k`, `run_react_mas`,
`run_llm_orchestra`, and `run_dancer` (wrapped as `_dancer_row`). For CNP and
ζ-CNP, the announced task's `required_role` is **grounded in the actually
sampled agent population** for the episode (audit finding F11,
`_grounded_role`, `run_bpic_experiment.py`) — earlier versions hardcoded a
synthetic role that did not match the empirically-derived agent roles,
causing CNP to fail 100% of episodes via `role_mismatch` regardless of the
underlying scenario.

---

## 6. LLM backends and prompt design

### 6.1 Backend selection

`llm/llm_factory.py` exposes three backends behind a common interface
(`generate_proposals`, `refine_proposals`, optional `set_context`):

| Backend | Used for | Behaviour |
|---|---|---|
| `MockLLM` (`llm/llm_mock.py`) | Offline / CI (`USE_REAL_LLM=false`) | Deterministic, rule-based proposal generation — no network calls, zero tokens. |
| `HeuristicPolicyGenerator` (`llm/llm_real.py`) | `--ablation` "DANCER-Heuristic" arm | **Generation** identical to the full system (delegates to `ProductionLLM` if an endpoint is configured, else `MockLLM`); **refinement** ignores rejection vectors entirely and applies a blind `±20%` random perturbation to `cost_normalized`/`time_days` of each previous proposal — zero LLM calls during refinement. |
| `ProductionLLM` (`llm/prompts.py`) | `USE_REAL_LLM=true` | Real chat-completion endpoint via the `openai` client; see §6.2. |

`get_llm()` resolves the backend from environment variables; `get_llm_by_name(name)`
allows the protocol to force a specific backend (`"mock"` / `"heuristic"` /
`"real"` / `"env"`) — used by the ablation campaign (§7.2) to run the *same*
sampled episodes under two backends.

### 6.2 Privacy-preserving prompt construction (audit finding F2)

`llm/prompts.py` builds every prompt from exactly three ingredients, matching
the contract of §3.4:

1. `_lead_constraints_block(lead)` — renders **only** the lead agent's own
   `θ_lead`, `weights`, and the derived `f_pol = 0` threshold
   (`cost_normalized > 1.5 × max_cost`).
2. `_registry_block(registry, ctx)` — renders public capability adverts for
   discoverable registry agents (e.g. `3PL_Agent`).
3. `_peer_visibility_note(n_peers)` — informs the LLM that `n_peers` other
   agents hold *private* constraints it cannot observe, and that it will
   learn feasibility only through rejection vectors; instructs it to
   generate proposals that **span** the cost/time trade-off space rather
   than guessing a single "optimal" point.

`_round_strategy(round_num, tau, tau_min)` supplies round-specific guidance:
Round 1 asks for *targeted* adjustment of the binding dimension; Round 2
asks for a *more aggressive* (≥20%) move or a method switch; Round ≥3
("near escalation") asks for the *most broadly feasible* proposal possible
given `τ` approaching `τ_min`.

`ProductionLLM` (`llm/llm_real.py`'s `RealLLM` class) resolves its endpoint
in priority order `DANCER_BASE_URL` → `OPENROUTER_API_KEY` →
`OPENAI_API_KEY` (raising `EnvironmentError` if none is set), issues
chat-completion calls at `temperature=0.7`, and parses the response via
`_extract_json_array` — a layered parser that handles raw JSON arrays,
`{"proposals": [...]}`-style wrappers, markdown code fences, and a final
regex fallback. `_parse_proposals` then clamps `cost_normalized` to
`[0,1]`, floors `time_days` at `0.1`, and defaults `new_agent_id` to
`"3PL_Agent"` whenever `requires_new_agent=True` but no id was supplied.

### 6.3 Domain-agnostic vocabulary (`llm/domain_context.py`)

DANCER is designed to be **generic across processes**, not specific to
supply-chain logistics. When run against a real event log, every piece of
domain vocabulary that appears in a prompt — activity names, agent role
descriptions, recovery-action phrasing, disruption narratives — is derived
from the log itself via `build_domain_context(log, net, log_type,
agent_profiles, disruption_cases)`:

- `detect_log_type(log)` heuristically classifies the log (BPIC2019,
  BPIC2017, BPIC2020, SEPSIS, Road Traffic Fines, or `"generic"`) from
  activity-name keywords;
- `real_activity_names` is extracted from the **mined Petri net's**
  transition labels (not a hardcoded list);
- `deviation_activities` is the top-10 most frequent
  `DisruptionCase.deviation_activity` values, via `Counter`;
- `recovery_actions` and `disruption_descriptions` are looked up from a
  per-log-type vocabulary table (`_DOMAIN_VOCAB`) — for BPIC 2019, these are
  SAP/procurement-specific (e.g. *"Route via 2-way matching (waive GR
  requirement)"*, *"Route via alternative vendor/subsidiary"*).

When no `DomainContext` is available (e.g. running the synthetic scenario
builders in `scenarios/scenarios_base.py`), `ProductionLLM` falls back to a
synthetic `SupplyChainContext` (`llm/prompts.py`) with generic logistics
vocabulary.

### 6.4 Anti-anchoring measure (audit finding F10)

The synthetic `SupplyChainContext.recovery_actions` descriptions are
deliberately written **without concrete numbers**. Earlier prompt versions
included numeric hints (e.g. "costs around 0.3") which acted as anchors,
collapsing sampled `(cost, time)` outputs to near-constant values across
episodes regardless of the actual scenario. `run_bpic_experiment.py
--probe-llm` runs two diagnostics before any campaign: PROBE 1 issues five
identical generation requests and checks for output diversity under
`temperature=0.7` (sampling is honoured); PROBE 2 issues requests for three
different episodes and checks that `(cost, time)` outputs are
episode-sensitive (the LLM is actually conditioning on `θ_lead`, not
emitting a memorised constant).

---

## 7. Experimental protocol and campaigns

All campaigns are orchestrated by `run_bpic_experiment.py` against a single
`BPICAdapter` instance fitted once per invocation (`_fit_adapter`). The
adapter's `iter_scenarios(n, disruption_type, with_meta=True)` yields
`(scenario_tuple, case_id)` pairs from a per-episode seeded RNG
(`random.Random(seed * 1_000_003 + idx)`), guaranteeing that **the same
episode set** is presented to every system in a given cell, and that the
entire campaign is reproducible from `--seed` (default `42`).

The three disruption-type labels used throughout the output CSVs are
`A_real` (`parametric_fluctuation`), `B_real` (`structural_failure`), and
`C_real` (`sovereignty_conflict`) (`SC_LABEL`, `run_bpic_experiment.py`).

### 7.1 Main comparison campaign

```bash
python run_bpic_experiment.py --log data/BPIChallenge2019.xes --iters 100 --seed 42
```

For each of the 3 disruption types, each of the 7 systems is run over 100
sampled episodes — **2,100 episodes** in total. `_run_cell` executes one
`(system, disruption_type)` cell via a `ThreadPoolExecutor`: LLM-driven
systems (`DANCER`, `ReAct-MAS`, `LLM-Orchestra`) use `--workers` (default 8)
concurrent threads; all other systems run single-threaded for determinism.
Each row records `oracle_feasible` (§4.3) and the source `case_id`.

**Representative results** (`outputs/bpic/bpic_results_summary.csv`, `n=100`
per cell):

| Scenario | System | Completion % | Avg. rounds | Avg. utility | Avg. tokens | Avg. latency (s) |
|---|---|---|---|---|---|---|
| A (`parametric`) | BPMN+ | 100.0 | 1.00 | 0.639 | 0 | 0.0001 |
| A | CNP | 0.0 | 1.00 | 0.000 | 0 | 0.0001 |
| A | ζ-CNP | 0.0 | 2.00 | 0.000 | 0 | 0.0001 |
| A | MAPE-K | 100.0 | 1.00 | 0.639 | 0 | 0.0002 |
| A | ReAct-MAS | 100.0 | 2.01 | 0.677 | 2698 | 35.0 |
| A | LLM-Orchestra | 100.0 | 1.00 | 0.730 | 1026 | 5.6 |
| A | **DANCER** | **100.0** | 1.10 | **0.778** | 2879 | 34.5 |
| B (`structural`) | BPMN+ | 0.0 | 0.00 | 0.000 | 0 | 0.0 |
| B | CNP | 0.0 | 1.00 | 0.000 | 0 | 0.0001 |
| B | ζ-CNP | 0.0 | 2.00 | 0.000 | 0 | 0.0001 |
| B | MAPE-K | 100.0 | 1.00 | 0.590 | 0 | 0.0002 |
| B | ReAct-MAS | 0.0 | 2.01 | 0.598 | 2681 | 34.5 |
| B | LLM-Orchestra | 100.0 | 1.00 | 0.577 | 1111 | 5.9 |
| B | **DANCER** | **100.0** | 1.49 | **0.753** | 3626 | 44.1 |
| C (`sovereignty`) | BPMN+ | 0.0 | 0.00 | 0.000 | 0 | 0.0 |
| C | CNP | 0.0 | 1.00 | 0.000 | 0 | 0.0001 |
| C | ζ-CNP | 0.0 | 2.00 | 0.000 | 0 | 0.0001 |
| C | MAPE-K | 100.0 | 1.00 | 0.630 | 0 | 0.0002 |
| C | ReAct-MAS | 100.0 | 2.00 | 0.676 | 2667 | 34.5 |
| C | LLM-Orchestra | 100.0 | 1.00 | 0.676 | 1044 | 5.5 |
| C | **DANCER** | **100.0** | 1.94 | 0.681 | 4099 | 60.3 |

All cells report `feasible_pct = 100.0` (every sampled episode is
oracle-solvable, §4.3), so `CR_feasible` equals the raw completion rate
shown above. Note that `ReAct-MAS` fails **all** of scenario B despite a
non-zero average utility (`0.598`): its committed proposals achieve `S_p`
just below `τ_min = 0.40`, or fail the capability-restoration rule of §4.1
— illustrating why utility and binary success must be read together.

### 7.2 Paired ablation: DANCER-LLM vs. DANCER-Heuristic (F7)

```bash
python run_bpic_experiment.py --log data/BPIChallenge2019.xes \
    --ablation --iters 100 \
    --ablation-types structural_failure parametric_fluctuation sovereignty_conflict
```

This campaign isolates the contribution of **Contrastive Refinement**
(§2.5). `campaign_ablation` draws **one** scenario pool per disruption type
and runs it **twice**: once with `llm_backend="real"` (label
`DANCER-LLM`, full system) and once with `llm_backend="heuristic"`
(label `DANCER-Heuristic`, blind `±20%` mutation refinement, §6.1). Because
both arms see *identical* episodes (same `agent_dicts`, same
`committed_f_res` draws, same `case_id`), any difference in outcome is
attributable solely to the refinement operator — the round-0 proposals are
generated by the *same* backend in both arms.

Per-episode `final_utility` is pivoted by `iteration` and compared with a
**paired Wilcoxon signed-rank test** (`analysis.statistics.ablation_wilcoxon`).
Output: `outputs/bpic/ablation_raw.csv` (one row per episode × arm, with the
same schema as the main campaign plus `case_id` and `oracle_feasible`).

### 7.3 Communication-drop robustness sweep (F7)

```bash
python run_bpic_experiment.py --log data/BPIChallenge2019.xes \
    --dropsweep --drop-iters 50
```

`campaign_dropsweep` fixes a single pool of `structural_failure` episodes
and runs DANCER repeatedly with `communication_drop_rate ∈ {0.00, 0.05, …,
0.40}` (9 values), `n_per_rate = --drop-iters` episodes each — **450
episodes** total. The drop rate is the per-`PEER_BID` probability of message
loss applied in `evaluate_proposals_node` (§3, `protocols/dancer_protocol.py:210-235`):
a dropped bid is still counted toward `messages_sent` (the agent computed and
transmitted it) but never reaches `aggregate_scores_node`.

**Observed result** (`outputs/bpic/robustness_summary.csv`): completion rate
remains **100% across the entire 0.00–0.40 drop-rate range**. This reflects
two compounding effects: (i) `aggregate_scores_node` averages over *received*
bids only, so `S_p` remains computable (and frequently still clears `τ_r`)
even with partial delivery; and (ii) the temporal decay of Eq. 7 lowers
`τ_r` over successive rounds, giving the protocol additional rounds to
recover from an unlucky round of drops. `outputs/bpic/robustness_raw.csv`
retains the per-episode detail (rounds, messages, utility) needed to examine
*how* convergence is reached at high drop rates, even where the binary
completion rate is saturated.

### 7.4 Communication-complexity scaling (F7)

```bash
python run_bpic_experiment.py --log data/BPIChallenge2019.xes \
    --scaling --scaling-ns "3,5,8,12,16,20" --scaling-reps 10
```

`campaign_scaling` constructs, for each `n ∈ {3,5,8,12,16,20}` and each of
`--scaling-reps` (default 10) repetitions, an `n`-agent scenario: the lead
agent is the empirical profile with maximum `w_pol`, and `n-1` additional
peers are sampled from the empirical agent-profile population. DANCER, CNP,
and ζ-CNP are each run on this scenario, recording `messages_sent`. Total:
`6 ns × 10 reps × 3 systems = 180` scenario instances ×
(per-system episodes) — **360 episodes** in the full campaign.

**Observed result** (`outputs/bpic/scaling_summary.csv`):

| `n` | CNP messages | ζ-CNP messages | DANCER messages (mean ± std) |
|---|---|---|---|
| 3 | 4 | 5 | 23.8 ± 6.3 |
| 5 | 6 | 7 | 30.6 ± 9.8 |
| 8 | 9 | 10 | 53.7 ± 26.0 |
| 12 | 13 | 14 | 63.6 ± 17.8 |
| 16 | 17 | 18 | 95.4 ± 43.5 |
| 20 | 21 | 22 | 110.4 ± 50.7 |

CNP's `n + 1` and ζ-CNP's `(n + 1) + 1` formulas (§5) are exactly linear and
deterministic (zero variance). DANCER's message count is also linear in `n`
(consistent with Proposition 2) but with substantially larger slope and
variance — the variance reflects the **stochastic number of negotiation
rounds** each episode requires before `S_{p*} ≥ τ_r`, which depends on the
sampled agent population and the Cognitive Layer's stochastic output.

---

## 8. Statistical analysis

`analysis/statistics.py` implements the significance-testing pipeline,
following IEEE TSC reporting conventions (H-statistic and `p`-value for
omnibus tests; `*`/`**`/`***` flags for `p < 0.05/0.01/0.001`; effect-size
labels *negligible/small/medium/large* at Cliff's-`δ` thresholds
`0.147/0.330/0.474`, per Romano et al. 2006):

- **`clopper_pearson_ci(successes, n)`** — exact binomial confidence
  intervals for completion rates (e.g. `100/100 → [0.964, 1.000]`), avoiding
  the normal-approximation failure at the boundary.
- **`kruskal_wallis(groups)`** — omnibus non-parametric test across all
  seven systems' per-episode `final_utility`, per disruption type.
- **`dunns_test(df, ...)`** (via `scikit-posthocs`, optional dependency) —
  pairwise post-hoc comparisons with Bonferroni correction following a
  significant Kruskal–Wallis result.
- **`pairwise_mann_whitney(groups)`** — pairwise two-sample tests with
  Bonferroni-corrected `p`-values, complementing Dunn's test.
- **`pairwise_fisher(groups_success)`** — Fisher exact tests on completion
  rate (binary outcome) for each baseline vs. DANCER.
- **`cliffs_delta(x, y)`** — non-parametric effect size for utility
  comparisons.
- **`bootstrap_ci(values, stat)`** — bootstrap confidence intervals for
  means/medians.
- **`ablation_wilcoxon(llm_utilities, heuristic_utilities)`** — paired
  Wilcoxon signed-rank test for the DANCER-LLM vs. DANCER-Heuristic ablation
  (§7.2).

`run_full_analysis(results_df)` runs the full battery over a campaign's
output DataFrame and returns a structured dict consumed by
`to_lateX_table` / `print_summary_table` for the LaTeX table emitted to
`outputs/bpic/bpic_results_table.tex`.

---

## 9. Provenance, audit trail, and threats to validity

The harness in this repository is the result of a documented audit
(`CHANGELOG_AUDIT.md`) that identified and corrected several
methodological issues in an earlier version of the evaluation. Findings
referenced by code comments throughout `core/`, `llm/`, `protocols/`, and
`scenarios/` (`F2`–`F14`) include:

- **F2** — privacy/sovereignty: prompts previously exposed all agents'
  private `θ_j` to the lead's Cognitive Layer (§3.4, §6.2).
- **F3** — unified canonical success criterion across all seven systems
  (§4.1), eliminating a structural `0%`-by-construction failure mode for
  unanimity-based criteria under `structural_failure`.
- **F4** — `K_max` cap enforced as the empirical premise of Proposition 2
  (§2.6, §3.5).
- **F7** — the ablation, drop-sweep, and scaling campaigns (§7.2–7.4) now
  execute real episodes against the harness rather than synthetic
  figure-generation scripts.
- **F8** — BPIC adapter integrity: real-log requirement, per-episode agent
  sampling (`DATASET.md`, §9).
- **F9** — exact confidence intervals, additional non-parametric tests, and
  an aligned round budget for ReAct-MAS (§5, §8).
- **F10** — anti-anchoring prompt design and the `--probe-llm` diagnostic
  (§6.4).
- **F11** — role-grounding for CNP/ζ-CNP's announced task (§5).
- **F12** — market feasibility envelopes and the capability-restoration rule
  (§4.1, §4.2).
- **F13/F14** — capability-restoration discard rule and bounded
  empty-proposal-set regeneration (§3.5, §3.6).

Three aspects were **deliberately left unchanged** (documented in
`CHANGELOG_AUDIT.md`, "Untouched on purpose"): the temporal decay formula
(Eq. 7, §2.4, which matches the paper exactly); LLM-Orchestra's full
constraint visibility (its defining architectural feature, §5); and the
piecewise utility sub-functions `f_time`/`f_pol` (Eqs. 4–5, §2.2) — the
paper's equations were updated to match this implementation, not vice versa.

**Threats to validity** that remain inherent to the design (not bugs, but
properties of the evaluation that a reader should be aware of):

- The LLM backend is non-deterministic (`temperature=0.7`); campaign-level
  results are reported as means over `n=100` episodes per cell, with exact
  binomial CIs for completion rates (§8).
- Agent populations are sampled *with replacement* from a finite pool of
  empirically-derived profiles (`DATASET.md`, §9); small disruption-type
  pools (e.g. `sovereignty_conflict`, ~29 source cases) imply some episodes
  share a source `case_id`, which is recorded for traceability.
- The synthetic-log fallback (`--allow-synthetic`) exists for development
  and CI only; it is loudly labelled and must never be the basis of reported
  results (§3.6 of `DATASET.md`).

---

## 10. References

- Smith, R. G. (1980). *The Contract Net Protocol: High-Level Communication
  and Control in a Distributed Problem Solver*. IEEE Transactions on
  Computers, 29(12).
- Kephart, J. O. & Chess, D. M. (2003). *The Vision of Autonomic Computing*.
  IEEE Computer, 36(1).
- Ye, F., Shen, W., & Hao, Q. (2017). *Extended Contract Net Protocol for
  Dynamic Task Allocation in Multi-Agent Systems*. IEEE CSCWD, 123–128.
- Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in
  Language Models*. ICLR. arXiv:2210.03629.
- Wang, L. et al. (2024). *A Survey on Large Language Model based
  Autonomous Agents*. Frontiers of Computer Science, 18(6).
- Romano, J. et al. (2006). *Appropriate Statistics for Ordinal Level Data:
  Should We Really Be Using t-test and Cohen's d for Evaluating Group
  Differences on the NSSE and Other Surveys?* Annual meeting of the Florida
  Association of Institutional Research.
