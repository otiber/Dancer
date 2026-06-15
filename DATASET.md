# Dataset Integration: From the BPI Challenge 2019 Event Log to Grounded Negotiation Episodes

> **Companion document.** `METHODOLOGY.md` describes the DANCER protocol,
> the baseline systems, and the evaluation harness. This document describes
> the **other half of the experimental grounding**: how a real-world process
> event log is turned into the agent populations and disruption scenarios
> that those systems negotiate over. Every claim below is implemented in
> `scenarios/bpic_adapter.py`.

---

## 1. Why a real event log?

A central methodological claim of the paper is that DANCER and its baselines
are compared on **empirically realistic** agent constraints and disruptions
— not on hand-tuned synthetic scenarios designed to favour any one
coordination mechanism. To support that claim, every reported experiment is
derived from a single real-world process-mining artefact: an **XES event
log**. No agent's `(max_cost, max_delay_days, availability, weights)` is
authored by hand; every value is a statistic computed over real case data.

This has two consequences that recur throughout the harness:

1. **Heterogeneity.** Different subsidiaries/resource groups in the log have
   genuinely different cost profiles, throughput times, and deviation rates
   — so the 74 candidate agent profiles extracted from BPIC 2019 are not
   interchangeable, and which three are sampled for a given episode (§9)
   materially affects its difficulty.
2. **Disruptions are mined, not scripted.** The three disruption classes
   (`parametric_fluctuation`, `structural_failure`, `sovereignty_conflict`)
   are identified from **actual non-conforming traces** in the log via
   conformance checking (§6) — they are real deviations from the discovered
   process model, not synthetic fault injections.

---

## 2. The BPI Challenge 2019 dataset

The default and recommended log is the **BPI Challenge 2019** event log
(4TU Centre for Research Data, `doi:10.4121/uuid:d06aff4b`), a SAP ERP
**purchase-order handling** process from a large multinational company, made
public for the 2019 Business Process Intelligence Challenge:

- ~**251,734** purchase-order-item cases.
- **74** distinct spend-area / subsidiary groups (`org:group` /
  `Sub spend area text` / `Spend area text`), each handling a materially
  different volume and cost profile of purchase orders.
- A high proportion of cases (paper-reported **~89.9%**) deviate from the
  textbook *"Create PO → Goods Receipt → Invoice Receipt → Clear Invoice"*
  3-way-matching path — covering missed approvals, payment blocks,
  quantity/price changes, invoices recorded before goods receipt, and
  cancellations.
- Each event carries a `cost:total` (or `Cumulative net worth (EUR)`)
  attribute, a `concept:name` activity label, an `org:resource` /
  `org:group` performer, and a timestamp — sufficient to compute per-group
  cost distributions, throughput times, and activity-frequency profiles.

The raw log (`data/BPIChallenge2019.xes`, ~700 MB) is **not** committed to
the repository; it must be downloaded separately and placed under `data/`
(see `README.md`, §7). The adapter is also pre-wired (via `log_type`) for
four other 4TU logs — BPIC 2017 (loan applications), BPIC 2020 (travel
declarations), SEPSIS (hospital cases), and Road Traffic Fines — should a
reader wish to replicate the pipeline on a different domain (§10).

---

## 3. Pipeline overview

`scenarios/bpic_adapter.BPICAdapter` implements a six-stage pipeline,
summarised in its module docstring and executed by `fit()`:

```
 XES log
    │
    ▼
 1. LOAD       pm4py.xes_importer  → EventLog  (optional subsampling)
    │
    ▼
 2. DISCOVER   Inductive Miner     → Petri net Φ = (places, transitions, arcs)
    │
    ▼
 3. CONFORM    Token-based replay  → per-trace (missing_tokens, remaining_tokens)
    │                                 + throughput times
    ▼
 4. PROFILE    per-group statistics → AgentProfile{max_cost, max_delay_days,
    │                                  availability, min_utility, weights}
    ▼
 5. INJECT     deviation → DisruptionCase{disruption_type, deviation_activity, …}
    │
    ▼
 6. EMIT       iter_scenarios() → (agent_dicts, active_ids, registry,
                                    disruption_type, tau_min)
```

`fit()` runs stages 1–4 and additionally calls `build_domain_context(...)`
(§8) to derive the LLM prompt vocabulary from the *same* mined model and
disruption cases. Stages 5–6 are driven on demand by
`iter_scenarios()`/`sample_scenarios()`, called once per experimental
campaign by `run_bpic_experiment.py`.

---

## 4. Stage 1 — Load

`BPICAdapter._load()` (`scenarios/bpic_adapter.py:372-413`):

- Imports the XES file via `pm4py.objects.log.importer.xes.importer`.
- If `--max-cases N` is set and the log exceeds `N` cases, draws a random
  sample of `N` trace indices (`random.sample`, seeded by `--seed`) and
  rebuilds the log from that subsample — used to bound runtime on the full
  ~252k-case log during development.
- **Missing-log policy (audit finding F8.5).** If `xes_path` does not exist,
  the adapter **raises `FileNotFoundError`** unless `allow_synthetic=True` is
  passed explicitly. Earlier versions silently fell back to a synthetic log,
  meaning a "real-log" campaign could run end-to-end on fabricated data with
  no signal in the output that this had happened. The synthetic path now
  prints an explicit `*** SYNTHETIC MODE ***` banner (§10) and sets
  `self.is_synthetic = True`.

---

## 5. Stage 2 — Discover

`BPICAdapter._discover()` (`scenarios/bpic_adapter.py:415-423`) mines a
**Petri net process model** `Φ = (places, transitions, arcs, im, fm)` using
`pm4py.discover_petri_net_inductive(log, noise_threshold=...)` — the
Inductive Miner algorithm with a configurable noise-filtering threshold
(`--noise-threshold`, default `0.2`). The noise threshold controls how
aggressively infrequent behaviour is filtered out of the discovered model:
a higher threshold yields a simpler model (fewer paths) at the cost of
classifying more traces as non-conforming in Stage 3.

The discovered net `Φ` serves two downstream purposes: (a) it is the
reference model against which Stage 3's conformance check is run, and (b)
its **transition labels** become the `real_activity_names` injected into LLM
prompts (§8) — so the vocabulary an LLM sees ("Record Goods Receipt", "Clear
Invoice", …) is literally the set of activities the mined model contains for
this log, not a hardcoded list.

---

## 6. Stage 3 — Conform

`BPICAdapter._conform()` (`scenarios/bpic_adapter.py:425-498`) runs
**token-based replay** (TBR) of every trace against `Φ`:

```python
self.replayed = pm4py.conformance_diagnostics_token_based_replay(
    self.log, self.net, self.im, self.fm)
```

Each trace's replay result reports `missing_tokens` (transitions the trace
fired that `Φ` could not enable — the trace did something the model did not
expect) and `remaining_tokens` (tokens left over after replay — the trace
did *not* do something the model expected). A trace is a candidate
**disruption case** if any of the following hold:

1. **TBR deviation**: `missing_tokens > 0` or `remaining_tokens > 0`.
2. **Activity-rule match**: the trace contains an activity present in
   `_BPIC2019_DISRUPTION_RULES` (§7), regardless of TBR outcome.
3. **Attribute hint**: the trace carries a `disruption_type` attribute
   (only present in synthetic logs, §10).
4. **Slow-but-conforming**: if none of the above hold but the case's
   throughput time exceeds **1.5 × the log-wide median throughput**
   (precomputed once via `pm4py.statistics.traces.generic.log.case_statistics`
   to avoid `O(n²)` recomputation), it is included and will be classified as
   `parametric_fluctuation` (§7) — an unusually slow but structurally
   conforming case is treated as a timing disruption.

For every case meeting one of these criteria, a `DisruptionCase` is
recorded: `case_id`, `disruption_type` (via `_infer_disruption_type`, §7),
`missing_tokens`, `remaining_tokens`, `throughput_days` (case duration in
days), `deviation_activity` (the first rule-matching activity, if any),
`org_group`, `item_type`, `cost_total` (summed `cost:total` /
`Cumulative net worth (EUR)` over all events in the trace), and the trace's
full `raw_attributes` dict for traceability.

`summary()` reports the resulting deviation rate
(`len(disruption_cases) / len(log) * 100`) and a per-type breakdown.

---

## 7. Stage 5 — Inject: disruption classification

> *(Numbered "5" to match the pipeline diagram; implemented as part of
> Stage 3's loop and Stage 6's scenario builder.)*

Classification is performed by `_infer_disruption_type(trace, replayed,
rules)` (`scenarios/bpic_adapter.py:167-203`), in strict priority order:

**Priority 1 — Activity-level rule match.** `_BPIC2019_DISRUPTION_RULES`
maps specific SAP activity names to disruption classes, based on the domain
semantics of the BPIC 2019 log:

| Activity | Disruption class | Rationale |
|---|---|---|
| `Set Payment Block` | `sovereignty_conflict` | An organisational policy constraint (budget/payment hold) is imposed mid-process. |
| `Change Approval for Good Receipt` | `sovereignty_conflict` | An approval-authority constraint is renegotiated. |
| `SRM: Awaiting Approval` | `sovereignty_conflict` | The item is blocked pending an organisational approval decision. |
| `Cancel Purchase Order Item` | `structural_failure` | The planned routing is broken — the item itself is withdrawn. |
| `Vendor creates invoice` | `structural_failure` | An invoice recorded *before* goods receipt violates 3-way matching — the expected execution order/structure is broken. |
| `Change Quantity` | `parametric_fluctuation` | A quantity parameter has drifted; structure intact. |
| `Change Price` | `parametric_fluctuation` | A cost parameter has drifted. |
| `Change Payment Term` | `parametric_fluctuation` | A timing parameter (payment terms) has drifted. |
| `Change Delivery Indicator` | `parametric_fluctuation` | A delivery-timing parameter has drifted. |
| `Release Purchase Order Item` | `parametric_fluctuation` | A hold on the item is released — a parameter (status) change. |

If a trace contains *any* of these activities, the **first match found**
(in event order) determines `disruption_type` and is recorded as the
`deviation_activity`.

**Priority 2 — Attribute hint.** If the trace carries a `disruption_type`
attribute (set only by `_generate_synthetic_log`, §10), it is used directly.
This path is inactive for real BPIC 2019 traces.

**Priority 3 — Token-replay signature.** For traces that deviate (Priority 1
found no rule match) but carry no attribute hint, the TBR result itself is
interpreted:

| `missing_tokens` | `remaining_tokens` | Classification | Interpretation |
|---|---|---|---|
| `> 0` | `> 0` | `structural_failure` | The trace both did something unexpected *and* left expected work undone — an agent/step is absent from its normal position. |
| `> 0` | `= 0` | `sovereignty_conflict` | The trace took an unexpected path but still completed all expected work — consistent with a policy override/block rather than a missing capability. |
| `= 0` | `> 0` | `parametric_fluctuation` | The trace is missing only some expected continuation — consistent with a stalled/delayed item. |

If none of the three priorities yields a classification, the case is
dropped from `disruption_cases`.

This three-class taxonomy is exactly the taxonomy used by DANCER's
Cognitive Layer and Safeguard Layer (`METHODOLOGY.md`, §2.1, §3.5): the same
string values (`"parametric_fluctuation"`, `"structural_failure"`,
`"sovereignty_conflict"`) flow unchanged from `DisruptionCase.disruption_type`
through `iter_scenarios()` into `DANCERState["disruption_type"]`.

---

## 8. Stage 4 — Profile: empirical agent constraint extraction

`BPICAdapter._profile_agents()` (`scenarios/bpic_adapter.py:500-610`) is
where real log statistics become the **private constraint vectors `θ_k`**
and **utility weights `C_k`** of Definition 1 (`METHODOLOGY.md`, §2.1).

### 8.1 Grouping

Each trace is assigned to a group — its candidate **agent identity** — using
the first attribute present among, in order: `org:group`, `org:resource`,
`Sub spend area text` (BPIC 2019), `Spend area text` (BPIC 2019 fallback),
`Company`, else `"UNKNOWN"`. Per group, the adapter accumulates: case count,
a list of per-case throughput times, a list of per-event costs
(`cost:total`), a deviation count (cases with `missing_tokens > 0`), and a
flat list of all activity names performed. **Groups with fewer than 3
cases are dropped** — they do not contain enough data to derive stable
statistics. On the full BPIC 2019 log this yields the **74 spend-area
profiles** referenced in §2.

### 8.2 Per-profile formulas

Two log-wide normalisation constants are computed once: `global_max_cost =
P99(all event costs)` and `global_max_tp = P99(all case throughputs)`
(99th percentile, to avoid outlier domination). Then, for each group:

| `AgentProfile` field | Formula | Interpretation |
|---|---|---|
| `max_cost` | `clip(P95(group costs) / global_max_cost, 0.10, 0.95)` | This group's 95th-percentile cost, normalised against the log-wide 99th percentile, clipped to `[0.10, 0.95]` — the group's empirical **budget ceiling** `θ_k.max_cost`. |
| `max_delay_days` | `clip(P95(group throughputs), 0.5, 30.0)` | This group's 95th-percentile case duration (days), clipped to `[0.5, 30.0]` — the group's empirical **deadline** `θ_k.max_delay_days`. |
| `availability` | `clip(1 - deviation_rate, 0.10, 0.99)` | `deviation_rate = (cases with missing_tokens>0) / case_count`. Groups whose cases deviate more often from the mined model are modelled as *less reliably available* `θ_k.availability`. |
| `min_utility` | `clip(0.5 + deviation_rate × 0.2, 0.40, 0.75)` | Groups with higher historical deviation rates are modelled as *more tolerant* (lower acceptance bar) — they have empirically "seen" more off-path proposals and accept more readily. |
| `role` | `_infer_role(group, top-20 activities)` | Keyword-matched against `_ROLE_HEURISTICS` (§8.3) → `"carrier"` \| `"warehouse"` \| `"manufacturer"` (default `"warehouse"`). |
| `weights` (`w_pol`, `w_time`, `w_res`) | see §8.4 | Derived from the *types* of activities this group performs. |
| `case_count`, `deviation_count`, `avg_throughput_days` | direct counts / `mean(group throughputs)` | Retained for diagnostics and for the registry agent's constraints (§9). |

### 8.3 Role inference

`_infer_role(agent_id, dominant_activities)` (`scenarios/bpic_adapter.py:158-164`)
lower-cases the group identifier concatenated with its 20 most frequent
activities, and matches against `_ROLE_HEURISTICS` in order:

| Keywords | Inferred role |
|---|---|
| `carrier`, `transport`, `logistics`, `shipping`, `delivery`, `3pl` | `carrier` |
| `warehouse`, `store`, `goods`, `receipt`, `gr` | `warehouse` |
| `manufacturer`, `vendor`, `supplier`, `production` | `manufacturer` |
| `finance`, `payment`, `invoice`, `clear`, `accounts` | `manufacturer` (policy-holder) |
| `procurement`, `purchase`, `buyer`, `srm` | `warehouse` |
| *(none match)* | `warehouse` (default) |

### 8.4 Weight derivation (`w_res`, `w_time`, `w_pol`)

For each group, its activity multiset is scanned for two keyword families:

- **Policy-related activities**: any activity containing `"payment"`,
  `"approval"`, `"block"`, or `"srm"` (case-insensitive). Count = `policy_acts`.
- **Time-related activities**: any activity containing `"receipt"`,
  `"goods"`, `"clear"`, or `"invoice"`. Count = `time_acts`.

Raw weights are then:

```python
w_pol  = clip(policy_acts / total_acts * 2.0, 0.15, 0.70)
w_time = clip(time_acts  / total_acts * 1.5, 0.20, 0.60)
w_res  = max(0.10, 1.0 - w_pol - w_time)
```

…and finally **renormalised to sum to 1**:
`w_x_final = round(w_x / (w_pol + w_time + w_res), 3)`.

The intuition: a group whose work is dominated by approval/payment-block
activities is modelled as primarily **policy-sensitive** (high `w_pol` —
cost compliance dominates its utility); a group dominated by
goods-receipt/invoice-clearing activities is modelled as primarily
**time-sensitive** (high `w_time`); whatever proportion remains is
attributed to **resource availability** (`w_res`).

### 8.5 From `AgentProfile` to `IPAgent`

`AgentProfile.to_ipa(committed_f_res=None)` (`scenarios/bpic_adapter.py:101-111`)
constructs an `IPAgent` directly from these five derived fields plus the
group's `agent_id` and inferred `role` — i.e. **every numeric input to
`L_k(p)` (Eqs. 2–5) for a real-log episode originates from this profiling
stage**, not from any hand-authored scenario file.

---

## 9. Stage 6 — Emit: scenario construction

`iter_scenarios(n, disruption_type, with_registry=True, with_meta=False)`
(`scenarios/bpic_adapter.py:283-320`) is the adapter's public interface,
called once per `(disruption_type, n)` request by
`run_bpic_experiment.py`. For `idx in range(n)`:

1. A **per-episode RNG** is constructed deterministically from the
   campaign seed: `rng = random.Random(seed * 1_000_003 + idx)`. This is
   the *only* source of randomness for this episode's scenario construction
   — re-running with the same `--seed` reproduces an identical episode set.
2. A `DisruptionCase` is drawn (with replacement) from the pool of cases
   matching `disruption_type` (the pool built in Stage 3, §6–7).
3. `_case_to_scenario(case, with_registry, rng)` converts the case into a
   `(agent_dicts, active_agent_ids, registry, disruption_type, tau_min)`
   tuple — the exact interface every protocol runner (`run_dancer`,
   `run_cnp`, …) consumes.
4. If `with_meta=True`, the generator yields `(scenario, case.case_id)` —
   used by `run_bpic_experiment.py` to record source-case provenance in
   every output row, and by `summary()` reporting to compute
   `unique_cases` per cell.

### 9.1 Agent-triple sampling (audit finding F8.1)

`_case_to_scenario` (`scenarios/bpic_adapter.py:614-706`) selects **three**
of the available `AgentProfile`s to populate the episode — a **lead**
(policy archetype), a **time** agent, and a **resource** agent — plus
optionally a registry agent (§9.3).

> **Why per-episode sampling matters.** An earlier version of the adapter
> selected the *single global* argmax profile by `w_pol`/`w_time`/`w_res`
> for **every** episode — i.e. the same three subsidiaries appeared in all
> 100+ episodes of a given disruption type, collapsing the claimed diversity
> of "74 empirical profiles" down to one fixed triple, and producing
> identical baseline difficulty across the entire campaign.

The current procedure, given `profiles = list(agent_profiles.values())`
with `len(profiles) ≥ 3`:

```python
q = min(8, len(profiles))                                  # pool size
by_pol  = sorted(profiles, key=lambda p: -p.weights["w_pol"])[:q]
by_time = sorted(profiles, key=lambda p: -p.weights["w_time"])[:q]
by_res  = sorted(profiles, key=lambda p: -p.weights["w_res"])[:q]

lead = rng.choice(by_pol)
time_ = rng.choice([p for p in by_time if p.agent_id != lead.agent_id] or by_time)
res  = rng.choice([p for p in by_res
                    if p.agent_id not in (lead.agent_id, time_.agent_id)] or by_res)
```

i.e. each of the three roles is sampled **uniformly from the top-8 profiles
by its corresponding dominant weight**, with the three identities forced
distinct. With `74` profiles (BPIC 2019), this yields up to `8 × 7 × 6 =
336` distinct agent triples per disruption type — a substantially richer and
more representative sample of the empirical population than a single fixed
triple, while remaining fully reproducible from `--seed`.

If fewer than 3 profiles survive Stage 4's `n_cases ≥ 3` filter (e.g. on a
heavily subsampled log via `--max-cases`), `_default_agents(disruption_type)`
supplies a minimal synthetic fallback triple (`Manufacturer` /
`Warehouse` / `Carrier`, with hand-set constraints) so the campaign can still
run — this fallback is not used in any reported BPIC 2019 result, since the
full log yields 74 profiles.

### 9.2 Per-episode availability draws

Each selected profile's `committed_f_res` (Eq. 3, `METHODOLOGY.md` §2.2) is
drawn **by the scenario builder**, not by `IPAgent.__init__`, using the same
distribution but the **episode's seeded RNG**:

```python
def _draw_committed(prof):
    if rng.random() < prof.availability:
        return round(rng.uniform(0.85, 1.0), 4)
    return 0.0
```

This value is frozen into each agent's serialised dict
(`AgentProfile.to_ipa(committed_f_res=...).to_dict()`) *before* the scenario
tuple is returned — so **every system evaluated on this episode sees the
identical availability draw** for each agent, even though each system
reconstructs its own `IPAgent` instances from the dicts independently
(`METHODOLOGY.md`, §4.1).

### 9.3 Disruption-type-specific adjustments

- **`sovereignty_conflict`**: the sampled **lead** agent's `max_cost` is
  tightened by 25% (`lead.max_cost *= 0.75`) before constructing its
  `IPAgent` — modelling the mid-process budget tightening that defines this
  disruption class. A new `AgentProfile` is constructed with this adjusted
  `max_cost` (all other fields copied via `__dict__`).
- **`structural_failure`**: the sampled **resource** agent's
  `committed_f_res` is forced to `0.0` (bypassing `_draw_committed`
  entirely) — modelling the disrupted capability being *definitely* absent
  for this episode, consistent with the capability-restoration rule of
  `METHODOLOGY.md` §4.1.
- **`parametric_fluctuation`**: no structural adjustment; the disruption is
  represented purely through the sampled case's profile statistics and the
  resulting `tau_min` (§9.5).

### 9.4 Registry agent (`3PL_Agent`)

If `with_registry=True` (the default), a single synthetic **registry
agent** is appended to `registry` (not to `active_agent_ids` — it becomes
active only if a committed proposal engages it):

```python
avg_cost = mean(p.max_cost for p in profiles)   # ≈0.60 if no profiles
registry = [IPAgent(
    agent_id="3PL_Agent", role="third_party_logistics",
    max_cost=min(0.90, avg_cost * 1.20),
    min_utility=0.40,
    max_delay_days=max(3.0, case.throughput_days * 0.8),
    availability=0.92,
    weights={"w_res": 0.40, "w_time": 0.40, "w_pol": 0.20},
).to_dict()]
```

Its `max_cost` is **20% above the average empirical budget** across all 74
profiles — modelling a third-party provider as a realistically-priced, but
not free, fallback. Its `max_delay_days` is anchored to **80% of this
specific case's observed throughput time** — a 3PL engaged to resolve *this*
case's disruption is expected to be at least as fast as the case's own
historical pace. This agent is the entity that DANCER's
capability-restoration proposals (`requires_new_agent=True,
new_agent_id="3PL_Agent"`) must reference under `structural_failure`
(`METHODOLOGY.md`, §3.5, §4.1).

### 9.5 `tau_min` assignment

The scenario's feasibility floor `tau_min` (used in Eq. 7's decay and in
`canonical_outcome`, `METHODOLOGY.md` §2.4/§4.1) is assigned purely by
disruption type:

| `disruption_type` | `tau_min` |
|---|---|
| `structural_failure` | `0.40` |
| `sovereignty_conflict` | `0.45` |
| `parametric_fluctuation` | `0.45` |

`structural_failure` receives the lower floor because the canonical agent
set `A*` always includes the disrupted resource agent with `f_res = 0`
(§9.3), which mechanically depresses `S_p` for *every* system regardless of
proposal quality — `0.40` is calibrated so that a market-envelope-feasible,
capability-restoring proposal can still clear the bar (`METHODOLOGY.md`,
Eq. 7 table, §2.4).

---

## 10. Domain-context derivation: from log to LLM vocabulary

`fit()`'s final step calls `build_domain_context(log, net, log_type,
agent_profiles, disruption_cases)` (`llm/domain_context.py`), producing a
`DomainContext` object that is threaded through `DANCERState["domain_context"]`
into every LLM prompt (`METHODOLOGY.md`, §6.3). This step ensures the
Cognitive Layer's prompts describe *this log's* process in *this log's*
terms:

- **`log_type` auto-detection** (`detect_log_type`): if `log_type` is not
  recognised, activity names are scanned for domain keywords (e.g.
  `"purchase"`/`"goods receipt"`/`"invoice"` → `bpic2019`; `"loan"`/`"credit"`
  → `bpic2017`; `"sepsis"`/`"triage"`/`"icu"` → `sepsis`; `"fine"`/`"penalty"`
  → `road_traffic`; `"travel"`/`"declaration"` → `bpic2020`).
- **`real_activity_names`**: pulled from `net`'s (Stage 2's mined Petri net)
  transition labels — the vocabulary the LLM is told the process consists of
  is the vocabulary the *mined model* actually contains.
- **`deviation_activities`**: the top-10 most common
  `DisruptionCase.deviation_activity` values across `disruption_cases`
  (Stage 5/§7), via `collections.Counter` — what the LLM is told "tends to go
  wrong" is what *actually* went wrong most often in this log.
- **`real_agent_ids`**: `list(agent_profiles.keys())` — the literal group
  identifiers from Stage 4 (e.g. real spend-area codes), so any agent the
  LLM references by ID corresponds to a real profile.
- **`avg_throughput_days`**: median of `agent_profiles[*].avg_throughput_days`.
- **`cost_unit`**: `"EUR"` if any event carries `Cumulative net worth (EUR)`
  (true for BPIC 2019), else `"normalised"`.

For `log_type="bpic2019"`, the static portion of the vocabulary
(`_DOMAIN_VOCAB["bpic2019"]` in `llm/domain_context.py`) supplies
SAP/procurement-specific framing consistent with the dataset's domain
literature:

- `domain_name`: *"Purchase Order Handling (SAP ERP)"*; `process_token`:
  *"purchase order item"*; `agent_descriptor`: *"subsidiary"*.
- Role labels: `lead_agent_role="purchasing manager"`,
  `time_agent_role="goods receipt officer"`,
  `resource_agent_role="vendor/supplier"`.
- `disruption_vocab` provides one narrative description per disruption type
  (and a `"cascading"` variant for compound disruptions), each phrased in
  terms of 3-way matching, goods receipt, payment blocks, and re-approval —
  matching the activity-rule table of §7.
- `recovery_actions` is a fixed list of **8 SAP-procurement-specific**
  recovery phrasings (e.g. *"Route via 2-way matching (waive GR
  requirement)"*, *"Route via alternative vendor/subsidiary"*, *"Apply
  consignment stock arrangement"*) — these are presented to the LLM as
  *examples* of the recovery-action space, not as a closed menu it must
  choose from verbatim.

This mechanism is what allows the *same* DANCER implementation
(`protocols/dancer_protocol.py`) to be evaluated against BPIC 2019,
BPIC 2017, or any other supported log without code changes: the protocol
logic, utility model, and safeguard layer are domain-agnostic; only the
`DomainContext` changes.

---

## 11. Mapping the dataset to the paper's formal notation

| Paper symbol | Dataset-derived value | Source |
|---|---|---|
| `θ_k.max_cost` | `AgentProfile.max_cost` — `P95(group cost) / P99(log cost)`, clipped `[0.10, 0.95]` | §8.2 |
| `θ_k.max_delay_days` | `AgentProfile.max_delay_days` — `P95(group throughput)`, clipped `[0.5, 30.0]` | §8.2 |
| `θ_k.availability` | `AgentProfile.availability` — `1 - deviation_rate`, clipped `[0.10, 0.99]` | §8.2 |
| `C_k = (w_res, w_time, w_pol)` | `AgentProfile.weights` — derived from policy-/time-keyword activity counts, normalised | §8.4 |
| `min_utility_k` | `AgentProfile.min_utility` — `0.5 + deviation_rate × 0.2`, clipped `[0.40, 0.75]` | §8.2 |
| `f_res` commitment (Eq. 3) | per-episode draw via `_draw_committed`, seeded `random.Random(seed*1_000_003+idx)` | §9.2 |
| Disruption class | `DisruptionCase.disruption_type` — activity-rule match or TBR signature | §7 |
| `τ_min` (scenario floor) | fixed per disruption class: `{0.40, 0.45, 0.45}` | §9.5 |
| Registry agent `3PL_Agent` | synthetic, anchored to `mean(max_cost) × 1.2` and `0.8 × case.throughput_days` | §9.4 |
| LLM prompt vocabulary | `DomainContext` — mined Petri net transitions, deviation activities, BPIC2019 vocabulary table | §10 |

Everything in the *left* column is a quantity the paper's formal model
treats as a parameter of the Dec-POSG; everything in the *right* column is
the concrete, reproducible computation that instantiates it for the BPIC
2019 experiments.

---

## 12. Reproducibility and seeding

A single `--seed` (default `42`) governs **all** randomness in the data
pipeline:

- `BPICAdapter.__init__` seeds both `random` and `numpy.random` with `seed`
  — affecting only the optional `--max-cases` subsampling in Stage 1.
- Every call to `iter_scenarios` derives a **fresh, independent** RNG per
  episode index: `random.Random(seed * 1_000_003 + idx)`. This RNG governs:
  (a) which `DisruptionCase` is drawn from the type-specific pool, (b) which
  three `AgentProfile`s are sampled as lead/time/resource, and (c) every
  agent's `committed_f_res` draw for this episode.
- Stages 1–4 (load/discover/conform/profile) are otherwise **deterministic**
  given the log file and `--noise-threshold` — the mined model, conformance
  results, and agent profiles do not depend on `--seed`.

Consequently: re-running `run_bpic_experiment.py` with the same `--log`,
`--noise-threshold`, `--max-cases`, and `--seed` reproduces **exactly** the
same sequence of scenario tuples for every campaign, independent of
`--workers` (thread-pool size affects only timing/interleaving of LLM calls,
not the scenario matrix, which is built single-threaded before execution —
see `README.md`, §10).

---

## 13. Other supported logs and the synthetic fallback

### 13.1 Other 4TU logs

`BPICAdapter(xes_path, log_type=...)` accepts `log_type ∈ {"bpic2019",
"bpic2017", "bpic2020", "sepsis", "road_traffic"}`. The module docstring
records the corresponding 4TU DOIs:

| `log_type` | Process | DOI |
|---|---|---|
| `bpic2019` | Purchase Order Handling (SAP ERP) | `10.4121/uuid:d06aff4b` |
| `bpic2017` | Loan Application | `10.4121/uuid:5f3067df` |
| `bpic2020` | Travel Declarations | `10.4121/uuid:52fb97d4` |
| `sepsis` | Hospital Cases | `10.4121/uuid:915d2bfb` |
| Road Traffic Fines | Traffic Fine Management | `10.4121/uuid:270fd440` |

Stages 1–4, 6, and the domain-context builder are written generically enough
to run against any of these (role heuristics, weight derivation, and
`_DOMAIN_VOCAB` entries exist for each `log_type`); the activity-level
disruption rules of §7 (`_BPIC2019_DISRUPTION_RULES`) are BPIC-2019-specific
— on other logs, classification falls back to the Priority-3 token-replay
signature (§7) and the slow-conforming heuristic (§6).

### 13.2 Synthetic log generation (development/CI only)

`_generate_synthetic_log()` (`scenarios/bpic_adapter.py:727-805`), invoked
only when `allow_synthetic=True` and `xes_path` does not exist, produces a
**5,000-case** XES log with statistical properties matching BPIC 2019:

- **60 synthetic subsidiaries** (`SUB_001`–`SUB_060`), 4 item types
  (`"3-way match, invoice before GR"`, `"2-way match"`, `"3-way match,
  invoice after GR"`, `"consignment"`), and the same **16 activity labels**
  used in `_BPIC2019_DISRUPTION_RULES` plus SRM states.
- Each case follows the base path `Create PO Item → Record Goods Receipt →
  Record Invoice Receipt → Clear Invoice`, with per-case random costs and
  inter-activity delays.
- Disruptions are **injected explicitly** and tagged via the
  `trace.attributes["disruption_type"]` hint (Priority 2, §7):
  - `P(structural_failure) ≈ 0.08` — inserts `"Vendor creates invoice"`
    before goods receipt.
  - `P(sovereignty_conflict) ≈ 0.06` — inserts `"Set Payment Block"` (with
    50% chance of a later `"Remove Payment Block"`).
  - `P(parametric_fluctuation) ≈ 0.20` — inserts `"Change Quantity"` early
    in the case.

This synthetic log is written to `xes_path` (so subsequent runs reuse it)
and is loudly labelled at load time (§4). **Results produced from the
synthetic log must never be reported as BPIC 2019 results** — its only
purpose is to allow the full pipeline (mining, conformance, profiling,
scenario emission, all seven protocol runners) to execute correctly in
environments where the ~700 MB real log is unavailable, e.g. continuous
integration.

### 13.3 Legacy synthetic scenario builders

`scenarios/scenarios_base.py` and `scenarios/scenarios_extended.py` contain
hand-authored agent factories (`_manufacturer`, `_warehouse`, …) and scenario
builders for the original paper's synthetic Scenarios A–C plus later
extensions (cascading disruptions, Byzantine/adversarial agents,
multi-constraint conflicts). They are **not imported by
`run_bpic_experiment.py`** and are retained only as a reference for the
hand-authored agent archetypes that motivated the empirical profile-extraction
approach described in this document; all results in the main, ablation,
drop-sweep, and scaling campaigns are produced via the `BPICAdapter` pipeline
described above.
