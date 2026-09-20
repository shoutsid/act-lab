# ACT Implementation Status

## The Gate-4 null clause was priced before its data exists: a withheld verdict is evidence, a cleared one is not (2026-09-20)

| | |
|---|---|
| **Why this, now** | §5.14's do-nothing clause is what authorizes or withholds Gate 4, and it is a *rule*: how much it decides is a property of the rule and the noise, not of the run. Its declared block (15000-15011) is still training, so this reading was taken **before the data exists** — the interpretation of whichever verdict lands is fixed in advance rather than chosen afterwards. |
| **The rule, read from the verdict's own constants** | `NULL_WIN_LIMIT = 10` of 12: the clause withholds Gate 4 only when the adaptive arm loses-or-ties against the null on **≥10 seeds**, with ties counting against the arm seeking authorization. So a **pass needs only 3 of 12**. |
| **The level** | Under no effect the clause fires on **79/4096 = 1.93%** of blocks — a cleared clause therefore happens **98.07%** of the time when adaptive is no better than doing nothing. |
| **The power, anchored on measured data** | At the dev-fidelity effect §5.40 just measured (the null ahead by 0.13554 nats at sd 0.11288, 12/12 fresh seeds → z = 1.20) the clause fires with probability **0.848**; at z = 1.5 it is 0.958 and at z = 2.0 it is 0.998. The anchor is *read from the committed §5.40 summary*, not assumed. |
| **The reading, stated before the run reports** | (i) A withheld Gate 4 is **strong evidence** — at the only effect size this program has measured, the clause fires ~85% of the time, so it is not a coin flip that landed against the adaptive arm. (ii) A **cleared** clause is **not evidence of value**: clearing needs 3 of 12 seeds and 98% of no-effect blocks clear it, so if the clause does not fire the value claim rests on the registered cost/band/control rules exactly as registered, and the write-up must say "the clause did not fire" rather than "adaptive beats doing nothing". |
| **Where it lives** | `scripts/null_clause_power.py` (imports `NULL_WIN_LIMIT` and the declared block from `v0_verdict`, and the anchor from the §5.40 summary, so neither can drift), artifact `artifacts/null-clause-power.json`, report §5.14. |
| **Mutants** | The clause limit relaxed to 8 fails all four tests (the analysis can no longer describe the rule the verdict applies); halving the anchor's z fails 1; hand-restating the pass bar fails 1; changing a number in the artifact's **prose** fails 1 — the artifact is compared to what the script computes *now*, whole, so a stale or hand-edited field cannot look correct while the code says otherwise (the field-by-field version of that test passed with the pass bar mutated). |
| **Result** | **1183 tests pass · 274/274 claims agree · 9/9 records reproduce.** |

## §5.40 ran, and the registered prediction held — the closure survives the deployed configuration (2026-09-20)

| | |
|---|---|
| **What ran** | The block registered on 2026-09-16 and then stopped after 5 of 24 sessions: twelve **fresh** seeds (15240-15251), each a pair session (`random, adaptive`) plus a `null` session, dev shape on the real TinyStories stream, 400 steps, three events, with the deployed treatment (`preserve_router_on_rebuild=True`, `new_expert_initialization="clone"`). **24/24 sessions, ~38 min of the local card**, run detached. |
| **The blocker it was waiting on was false** | The earlier entry records that a long run "cannot be made to outlive its launching shell in this environment", so the block could only be bought in foreground chunks. Tested rather than assumed: `setsid nohup` detaches from the launching shell — the same technique the Kaggle queue driver has used all along — and the block ran to completion in the background while other work proceeded. The registration's cost estimate (~200 s/seed) was accurate. |
| **The verdict** | **Branch (b): the closure holds under the deployed configuration.** The do-nothing arm has the **lowest final loss on 12/12 fresh seeds**: `null - adaptive` = **−0.1355** (0/12 seeds in that direction, p = 1.0), `null - random` = **−0.1457** (12/12, p = 0.0005). Both registered predictions were met — P1 fails, P2 passes — so §5.13's closure is *re-established on the configuration the project actually ships*, and the repair removes the mutating arms' disadvantage without giving either an advantage (`adaptive - random` = −0.0101, 6/12, p = 0.61 — the diagnostic's +0.0007 seen again). All eleven built-in checks hold, including the registration's own post-hoc table recomputing from committed rows. |
| **What it means for Gate 4** | The declared run's quality premise now rests on the toy factor map plus the compute-efficiency protocol alone, with **no transfer claim at all** — at dev fidelity, at a depth the toy prior favours (6, against the cell's depth >= 4), with the treatment in place, doing nothing still wins every seed. The 5.14 null clause is therefore the operative bar rather than a formality, and the most likely verdict is the small-scale one. Report §10.1/§10.2/§10.3 updated to say so. |
| **A sign convention, inverted twice** | The registration's own sentence says "deltas are `null - arm`, so negative = the arm beats doing nothing". Every other use in the program (5.39's summary and its test, and 5.40's own diagnostic narrative — "the adaptive arm's harm falls from 0.248 to 0.117" for `null - adaptive <= 0`) reads the same delta as **negative = the mutating arm is worse**. §5.39.1 was the first incident; this is the second, and it is corrected in the report rather than left standing. |
| **And the first summary printed the inverted branch** | P1 and P2 are opposite claims about one delta, so they need opposite tails; the first run tested both on the churn-harm tail, making P1's `established` mean "the null is significantly better than adaptive" — P1 *failing* — and printing **branch (a), the real-text leg reopening**, on a block where the null won every seed. The numbers were right and only the label was wrong, which is why nothing else caught it: the artifact's own `per_seed` rows contradict its branch. Fixed at the source (per-primary tails, extracted `branch_from_primaries`, the convention restated in the payload with a note that the registration's sentence inverts it), and pinned by `tests/test_deployed_vs_null_summary.py`: the convention asserted convention-free (the null is the per-seed argmin), each primary's tail, the branch re-derived from the stored comparisons, and the constructed failure. |
| **Mutants** | Re-running the original bug (P1 back on the churn-harm tail) fails **3 tests** and writes branch (a) — the false headline is now loud. Flipping one seed's adaptive loss fails 4; forcing `prediction_met` to "passes" fails 1; making the derivation ignore P1 fails 1. |
| **Result** | **1179 tests pass · 274/274 claims agree · 9/9 records reproduce.** |

## The launch's own plan was priced for a one-GPU host (2026-09-20)

| | |
|---|---|
| **What was found** | The dtype probe's plan line — the number this project quotes for how long the declared run takes — was priced with `PairSchedule`'s default of **one device**, while the session's benchmark and its runner both resolve the host's placement (`arm_slots`). On Kaggle's T4 shape that is 2 devices, so the plan a human reads was **twice** the run the launch executes: the probe printed **2.871 h per pair / 3 pairs per session / 4 sessions for the primary queue**, against the true **1.443 h / 6 / 2**. |
| **Confirmed on the host, not inferred** | `--queue selfcheck` (one shot, ~1 min of the window) reports `host devices: 2`, `one arm per device: cuda:0, cuda:1`, and *all checks passed*; the re-run probe now logs `arm placement: 2 arm(s) in 1 wave(s) of slots=2 on cuda`. |
| **The re-run, with the corrected plan** | fp32 **1,451.9 tok/s** → 4.802 h per pair → **1 pair/session, 12 sessions** (an fp32 run is not merely slower, it is 6× the sessions); fp16 **5,083.4 tok/s** → **1.443 h per pair**, **6 pairs/session registered, 5 by the runner's own start gate, i.e. 3 sessions for the primary queue (5+5+2) against the 2 the registered field reads**. fp16 is **3.5×** the fp32 rate on the same host, so the pre-committed dtype rule selects it again — nothing about the decision moved, only the schedule it is priced at. |
| **And the second half of the same gap** | The plan's capacity and the runner's capacity are different quantities, and the docs read only the first. ``full_pairs_per_session`` is §5.2's arithmetic (``floor(usable / pair time)``), which the §5.2 table's rows are computed from and which the tests pin; the runner's rule 1 adds the 45-minute publish reserve to *every* start, so it begins one fewer pair whenever that reserve binds. The runbook asserted the opposite ("the plan's own arithmetic already includes the 45-minute publish reserve"), so the operator-facing expectation was one session short. Corrected at the source instead of in prose: `session_pair_capacity()` walks the deadline rules themselves, the benchmark artifact now carries `pairs_per_session_rule1` / `sessions_for_primary_rule1` beside the registered fields, the probe's plan line prints both, and the runbook says which one to plan from. Pinned by `test_the_registered_capacity_and_the_runners_gate_differ_when_the_reserve_binds`, which also shows the reserve is the whole reason (lend the session those 45 minutes of window and the two agree). |
| **Why it mattered** | Every plan number keyed to the probe was 2× too large: the primary queue's session count, the wall-clock to the primary verdict, and the window the driver's quota gate compares against. Runtime was never affected — the runner derives the pair time itself, so sessions advance at the placement-aware rate — but the plan a reader plans from was wrong in the direction that makes the run look less affordable than it is. |
| **Fixed** | The probe's plan line now comes from one function, `probe_pair_schedule`, which prices the host it measured, while the dataclass default keeps a genuinely one-device host honest. Pinned by `test_dtypeprobe_prices_the_placement_the_runner_will_use`, which also fails if the report stops quoting the numbers it computed — the first version of that test passed with the probe reverted, because it pinned the helper but not its use. |
| **Result** | **1173 tests pass · 274/274 claims agree · 9/9 records reproduce.** |

## The Gate-4 search cannot reach the one gene the other track measured as the lever (2026-09-20)

| | |
|---|---|
| **What was found** | The two tracks of this program have never been read against each other, and read together the gap is the reachable operation set. The arch-search phase measured, at a matched ~100M budget on real text, that the loss lever is **residual-stream width** (−0.1562 nats at its factorial's exact 46,656-assignment floor), that head count is parameter-free and non-significant (−0.0177, p = 0.14), that depth is real but *shallow wins*, and that the **active-parameters hypothesis is refuted** (×4.51 spread at ρ = +0.214, p = 0.62, with the fully dense design 7th of 8). The V0 session's entire search space is `feasible_expert_mutations` — grow/prune of experts — so every candidate the declared block evaluates moves the axis that phase found *uncorrelated* with loss, and none can reach the axis it found as the lever: no operation sets `hidden_size`, head count or depth, and the only assignment to `hidden_size` in `src/act/topology/` is the restore on a rejected change. |
| **What it does not mean** | Gate 4's question is about *selection* rather than about which gene — does adaptive selection beat doing nothing on the declared block — and §10.1's depth axis says selection value exists above an onset, with §10.2's null clause as the right control for it. What a pass licenses is what is bounded instead: "adaptive control of expert capacity at 15000-15011", which is not the same claim as adaptive architecture search and cannot move the other track's recommendation, because the two search spaces share no gene. |
| **Why it matters to the next gate** | The arch phase's width effect is the largest single effect either track has measured at a matched budget, and it is the covariate this program's own open question (ii) needs in order to separate depth from capacity: `arch-zoo-100m-v3-crossed` is the program's only matched-budget separation of depth from width (ρ(depth, hidden) = 0.000, balanced 2×2) and it answered *width*. A width operation is not free engineering — it reshapes every weight in the stream, so it needs the treatment `grow_expert` already gives a new expert, with re-initialisation and a parameter-budget check — which is why it is a deliberate decision after Gate 4 rather than an edit to a running queue. |
| **Written where a reader will find it** | New §10.4 of the research report, with the four-gene table, placed after 10.3's open questions because (ii) is the question it feeds. |
| **Pinned** | `tests/test_topology_transactions.py::test_no_reachable_operation_can_move_the_gene_the_search_phase_measured` — the pool's operation vocabulary is a subset of grow/prune, and applying every candidate leaves `hidden_size`, depth, `heads`, `kv_heads` and `ffn_width` unchanged — so the claim fails loudly if an operation is ever added rather than quietly going stale. |
| **Result** | **1172 tests pass · 274/274 claims agree · 9/9 records reproduce.** |

## The declared Gate-4 entry run is in flight, on a schedule measured before the spend (2026-09-20)

| | |
|---|---|
| **What was launched** | The project's critical path, per §10.2 of the research report - "Gate 4 is prepared, amended, and unspent ... the run needs a T4" - is now running on Kaggle T4: the pre-registered §5.1 compute-efficiency protocol on the declared 285M V0 config, three queue-kernels (`session`, `control`, `null`) over seeds **15000-15011**, publishing to `act-v0-5-1`. `kaggle_queue.py --queue session --max-sessions 6` is driving the primary queue detached (`artifacts/v0-5-1-launch/session.log`); nothing had been pushed since the arch-zoo kernel. |
| **Pre-flight, before any spend** | `verdict_rehearsal.py --check` applies the real verdict to every withholding clause, both absent-queue and both short-queue cases and the label-precedence combinations, and all fourteen variants match their registered label and exit code (~1 s, no GPU; the artifact it rewrites is byte-identical, so the check is idempotent). Datasets present, quota read **25.16 h remaining**, and the code dataset was republished at `92f86c0` so the pushed kernel ran this tree rather than a stale copy. |
| **The dtype probe, which decides whether a session can advance at all** | Measured on the T4 before the queue was pushed: **fp32 1,401.5 tokens/s** -> 9.842 h per pair -> **0 pairs fit a 9-hour window**, i.e. an fp32 session would idle its deadline and burn the window producing nothing - the exit-4 loop the driver refuses to re-push on. **fp16 4,926.4 tokens/s** -> **2.871 h per pair, 3 pairs per session, 4 sessions for the primary queue.** So the pre-committed throughput-gated rule selects fp16, it is **3.5x** the fp32 rate, and §5.2's committed schedule (3.585 h per pair, 6 sessions) is conservative against the measurement rather than optimistic. **(corrected 2026-09-20: those plan numbers were priced for a one-device host — see the entry above. The measured rates stand; the pair time is 1.443 h, and at that rate the primary queue needs **3** sessions, not 6 or 2 — see `sessions_for_primary_rule1`.)** |
| **What the run decides** | A pass authorizes Gate 4; a null-clause rejection closes §5.1's selection premise at its own scale; and the cost measurement is the registered blocker on scaling. The strongest real-text fact measured so far points the other way - at diagnostic fidelity the do-nothing arm beat every mutation policy, and the rebuild repair removes only 31% of the churn harm - which is why the declared run is the decisive spend rather than a formality. |
| **Why not the §11 comparison instead** | The arch-search phase's only un-run item is measured to be undecidable at its own size (§9a: 13 candidates per set needs a 0.30-nat effect, and its coverage form is invisible at every size priced), so it cannot be the next decisive spend. |
| **Verified** | The launch is quota-gated and resumable by construction: one session per queue, atomic pairs, deadline stops with in-progress snapshots, restart-from-scratch, skip-completed idempotence, and a published checkpoint dataset as the durable copy. The driver reads the weekly quota before every push and refuses when the remaining window cannot cover one. |

## The loop's price at a real horizon double-counted its confirmation (2026-09-20)

| | |
|---|---|
| **What was found** | §10 prices the search loop at two confirmation lengths: **17.0 GPU-h with the confirmation at 1,200** and **29.6 on Kaggle / ~49 on this card** with it at a 12,000-step horizon. The second figure is wrong in composition: 16.99 + 12.57 = 29.57, so it adds the horizon confirmation to the *complete* 1,200-step loop — which already contains a confirmation of the same 24 finalists — while every sentence around it says the length is a substitution ("with the confirmation at 1,200" / "with it at the horizon"; README: "what that loop costs depends entirely on **where** it confirms"). The totals are **26.4** (13.80 + 12.57) and **45.7** (13.80 + 31.92). |
| **Why nothing caught it** | The audit encoded the same composition: its claim recomputed `100×3×0.046 + 8×3×0.133 + 24×0.524 = 29.57` and compared it against the record's 29.56, so the double count was checked *against itself* and agreed. Reverting both the record and that formula is still a non-biter by construction — which is why the composition is now checked against the **driver's own planner** instead of against prose: `--plan screen` is one screen stage and one confirmation stage, and the stage whose length varies is the confirmation. |
| **Two clauses were wrong with it** | "the confirmation alone costs more than every step before it" (§9) and "the confirmation is the largest single term in the latter two" (§10) hold only on the local card: 12.57 is **91% of** the 13.80 screening half on Kaggle, and 31.92 is 2.3× it on this card. Both are now stated that way, because the point they carry — a real horizon makes the last step set the price — survives the correction while the superlative does not. |
| **The rates under every projection were unchecked too** | Both local-card rates are pooled means over the ledgers that measured them: **0.046** over the phase's 50 runs at 400 steps, **0.133** over v2's 24 flat runs at 1,200. Neither was checked, though §10's 17, 40, 26.4, 45.7 and §9's whole price list are derived from them. Added with the halves (13.80 screening, 3.19 confirmation), the quoted 17, and the corrected horizon totals — each at the precision the record quotes. |
| **And §9a's launch pre-flight** | "39 runs ≈ 5.92 GPU-hours" for the random arm is the driver's own plan projection, now recomputed from it — 13 candidates × 3 seeds flat, 95.8M tokens at the driver's conservative 4,500 tok/s — so the arm's estimate is checked before the hours rather than after. |
| **How the plan is driven** | `arch_zoo_screen.parse_args` now takes an optional `argv`, so a plan can be built and costed without a shell and without mutating the test process's own command line. The composition claims read the plan the same way `main()` does, including the candidate count from the flag — a literal `100`/`13` in the check would have made the candidates flag an equivalent mutant. |
| **Verified** | Five mutants caught: the loop planned with its confirmation at the screening length, the plan confirming 3 survivors instead of 8, the pre-flight arm sized at 26 candidates (which failed to bite until the check read the flag), the 400-step rate quoted without its half-a-place tolerance, and the 1,200-step rate read from the wrong arm. **263 → 274 claims.** |
| **Result** | **1171 tests pass · 274/274 claims agree · 9/9 records reproduce.** |

## §9a's readiness claims had no machine check, including the one that decides the statistic (2026-09-20)

| | |
|---|---|
| **What was found** | §9a states four things that decide whether the §11 comparison can be run at all and what a result would mean: that the curated sets carry no location advantage (the non-circularity condition), that equal compute is a statement about a *budget*, that the budget-matched draw is not a designed set in disguise, and that the coverage form is invisible — which is why the *set mean* rather than the winner is the pre-registered statistic. The audit recomputed §9a's mean-form power, its budget match and its typicality percentiles, and none of these four. |
| **The percentile table, whole** | Every number reproduces from the zoo definitions at the tool's own 4,000 draws: 1,059 cells at depth 8.97/3.35 and hidden 718/143; `v1` at p83/p16/p19/p64, `v2` at p68/p96/p22/p98, `v3` at p72/p100/p42/p100 — values and percentiles both. Percentiles are compared as the tool *renders* them (whole percentages) and exactly, because a relative tolerance is the wrong test on a rank: 5e-3 of 16 is 0.08, a third of a draw. `v1`'s 16th-percentile depth spread and 19th-percentile hidden size are the two numbers the non-circularity argument rests on. |
| **The budget paragraph, whole** | The curated set's 13 candidates total **1,121,409,984** and span 70.83M–123.30M (×1.7408, mean 86.26M) — checked exactly, where the audit had taken that total on trust as the denominator of every ratio beside it. The band counterexample reproduces: the *same* seed 151 at the driver's own 0.15 draws 1,282,387,008, **1.1435×**, which is why the band is part of the launch line rather than a default. And the matched draw is not the designed sets' mechanism in disguise — its structure (7.85/3.21, 718.8/137.8) sits beside its own band's population (9.49/3.50, 715.6/143.7). |
| **The reason the mean is the statistic** | A 0.60-nat better winner — twice the effect the mean form is priced at, and four fifths of the whole 0.74-nat candidate spread — leaves the `best` statistic at **0.03 power** at 13 candidates per set, **0.02** at 26 and **0.04** at 52, every one at or below the nominal 0.05. Doubling the advantage from 0.30 to 0.60 leaves the rate where it was, **0.03 both times**, so the coverage form is unsensitive rather than merely underpriced — the structural argument (half the assignments put the improved member on either side) read off the shipped test instead of asserted. The contrast that makes it informative: the same statistic sees a set-wide 0.30-nat shift at **0.94**, and an inert statistic on that design reads 0.430 against the asserted 0.8. |
| **Two comparison rules the audit did not have** | A percentile is a rank rendered to a whole percentage, so it is compared as rendered and exactly. A value quoted to one decimal carries **half a unit in the last quoted place**, not half a percent: the matched draw's depth mean is 7.846 and the record's 7.8 is that rounded — dropping the explicit tolerance makes the claim fail, so the tolerance is load-bearing rather than decorative. Both rules are pinned in the audit's own test. |
| **How it is checked** | `check_space_typicality` rebuilds the enumerated population and re-reads the three zoo definitions; the budget additions extend `check_structure_and_budget`, whose quick path still skips the 200-seed scan; and `check_set_comparison_coverage` recomputes the power figures through the shipped tool, sharing one parallel power path with `check_power_study` — generalized to take the model and statistic instead of hard-coding `shift-all`/`mean`, so a future reading cannot be computed on the wrong one. |
| **Verified** | Eight mutants, all caught: the curated totals read from the wrong definition (six claims fail), the band counterexample drawn at the matched band, the one-decimal tolerance dropped, the draw compared against the wrong population (three of four fail), the parallel power path ignoring its model/statistic (the coverage claim fails at power **1.000**), the contrast row computed on an inert statistic, the 52-per-set row transcribed as 13, and the coverage check dropped from `CHECKS` (two tests fail). The three power substitutions fail *loudly* with a `KeyError` rather than passing silently. **212 → 263 claims**, with no measurable wall clock on either the suite or the audit — both are bounded by their slowest member. |
| **Result** | **1171 tests pass · 263/263 claims agree · 9/9 records reproduce.** |

## The fifth rule gets its reading check before its arm is bought (2026-09-19)

| | |
|---|---|
| **What was done** | The reading check covered four of the phase's five pre-registered rules, and the fifth — the §11 random-vs-manual comparison — was the one that has not run. It now runs through the same shared path, and `set_comparison_decision`'s record renders the same table. |
| **Why that ordering matters** | Wiring a check in *after* a result is how a declared threshold becomes a chosen one, which is what §11's pre-registration exists to prevent. The check is part of the declaration rather than a response to anything, and by the time §11 runs, two of the four existing rules will have shown it costs something: one of them is §2c's own attribution. |
| **The mechanism had to generalize** | §11's evidence is a two-set test built from both arms' rows, not one of the zoo outcome's own blocks, so `rule_reading_stability` now takes either the outcome-block path or a caller-supplied `read(rows)`. One shared path, one set of substituted fields, one set of thresholds. |
| **One renderer, not two** | The reading table was inline in the zoo decision and absent from the §11 renderer; it is now `reading_stability_lines`, used by both. A second spelling of the table is how two arms end up describing one check differently. |
| **Verified** | Three tests on synthetic planted-effect designs — a table rendered, a branch that moves when a per-set constant is planted on the trajectory loss, and no claim at all without a reconstruction — plus the bold marker pinned on a constructed dict so the disagreeing row stays readable. Four mutants caught, including a renderer that prints the verdict without the table. **1166 → 1170 tests.** |
| **Result** | **1170 tests pass · 212/212 claims agree · 9/9 records reproduce.** |

## Every pre-registered branch now names the loss it was read on, and one of them moves (2026-09-19)

| | |
|---|---|
| **What was done** | The reading check built for the head-pairing rule was generalized to all four pre-registered rules (`replication_decision`, `horizon_decision`, `head_pairing_decision`, `finalist_order_decision`) through one shared path, so the four cannot describe the same check in four different ways. Each names the outcome blocks it actually reads, because rebuilding a block a rule never looks at costs the record's 200,000 draws and cannot change its branch. |
| **The finding it turned up** | The crossed arm's pre-registered replication rule — the branch that says §2c's attribution replicates — lands on **ATTRIBUTION FAILS TO REPLICATE** when read on the trajectory-averaged loss. Nothing about the attribution moved: both main effects keep their signs and sit at the design's Monte-Carlo floor, and the interaction stays at its 2/2⁶ floor with all six seed contrasts agreeing. Only `|hidden| ≥ 1.5 × |depth|` stops holding — 0.2255 against 0.1814, a ratio of **1.24× against the declared 1.5**. |
| **Why the condition matters, not just the label** | Both facts share one label and are different claims. The ratio is a statement about end-of-training magnitudes, and the trajectory average weights the early phase — where depth's penalty is evidently larger — more heavily (depth grows 2.20×, width 1.22×). So the readings record the condition that moved, and the record's clause names it, rather than reporting that the attribution failed. |
| **Not a six-seed accident** | The fresh three-seed confirmation arm, which shares no seeds with the pooled set, narrows its own ratio 0.1851 vs 0.0865 → 0.2265 vs 0.1879 (**1.21×**) and lands on the same alternate branch. Two independent seed sets reaching the same reading is what makes it a property of the loss. |
| **The two branches that do not move** | The horizon arm's depth-sign headline is stable on all three readings with no condition moving, and the finalist-order branch is stable while its *second* condition flips (the screening-free blocks clear 0.05 at 1/36 on the trajectory average and not on the like-for-like pair) — the case a label-only check would have called settled. |
| **Cost** | Marginal and opt-in: 0.01 s for the horizon rule, 0.16 s for finalist-order, 2.7 s for replication and 3.5 s for head-pairing, each against the analysis it belongs to. Unconditional rather than gated on a margin looking thin, because a silent skip is indistinguishable from a branch that held. |
| **Verified** | Four tests and eight claims. Two records regenerated from their own documented commands (`arch-zoo-100m-v3-crossed`, `arch-zoo-100m-v3-crossed-confirm`); `check_record_freshness.py` catches a hand-edited reading table. Four new mutants caught, including a rule re-entering its own reading check, which was an infinite recursion until the shared path passed `reading_check=False`. **204 → 212 claims.** |
| **Result** | **1166 tests pass · 212/212 claims agree · 9/9 records reproduce.** |

## The record's own branch label now carries the loss it was read on (2026-09-19)

| | |
|---|---|
| **What was found** | The previous two stretches established that the head-paired arm's branch **reverses** on a trajectory-averaged loss, and wrote that into the findings record, the arm's NOTES and the audit - but the artifact a reader actually quotes, the arm's pooled `decision.md`, still printed the branch alone. A reader could try the second loss, get a different answer, and have no way to tell which one was pre-registered. The gap was the record, not the knowledge. |
| **The fix is in the generator, not the artifact** | `head_pairing_decision` now takes the pooled rows and, when it gets them, re-reads its own rule over the other losses those ledgers carry - same rows, only the loss field substituted - then returns the branch each reading lands on plus a sentence naming any that disagrees. The decision prints that as a table. Because the alternate readings are produced by calling the *same* branch logic, the two cannot drift apart: a change to the rule moves both. |
| **What the arm's record now says** | held-out `validation_loss` and `final_train_loss` both land on `HEAD EFFECT IS REAL AT WIDE WIDTH`; `mean_train_loss` lands on `INTERACTION ABSENT - THE THREE-SEED AGREEMENT WAS SEED LUCK` and is bolded as the reading that moves. The arm's own three-seed record is `open-floor-limited` and stable on all three, which is the informative null for this check. |
| **Not gated on the margin looking thin** | The first draft skipped the re-read when no comparison sat within 2× of its threshold. That is a silent heuristic whose failure mode is indistinguishable from a branch that held - the exact defect this stretch is fixing - so it was removed. The check costs **3.6 s** against the 6.2 s 200,000-draw analysis it belongs to, and a rule applied to an outcome *without* rows reports no stability claim rather than an unverified one. |
| **Verified** | Three tests and three claims, plus an artifact-level proof. Four mutants caught: substituting the held-out loss for every field (nothing moves), dropping the reading-dependent clause, suppressing the table, and claiming stability without rows. Stripping the table out of the committed `decision.md` makes it **stop reproducing** under `check_record_freshness.py`, so the record gate covers this too. **201 -> 204 claims.** |
| **Result** | **1162 tests pass · 204/204 claims agree · 9/9 records reproduce.** |

## Every significance headline re-read on training loss: the two that move are both negatives (2026-09-19)

| | |
|---|---|
| **What was done** | The head-paired audit left an obvious question - which *other* headline had never been re-read on the training loss - so all nine committed records were checked. Five arms now carry the split check (v2, crossed, horizon, head-paired, and this stretch's finalist-order rule); the audit filled in the four that did not. |
| **The positives hold everywhere** | v2's and the crossed arm's attribution, the horizon arm's depth sign, and the two spread-only arms (`v2-horizon` at 3× and `v3-crossed-confirm`) all keep their signs and clear their thresholds on every reading. `v2-horizon`'s spread is p ≤ 5e-06 on both; the crossed confirmation's is 1.5e-05 held-out and 5e-06 on training loss. |
| **Both readings-sensitive headlines are negatives** | `arch-zoo-100m-v1-finalists-6seed` is v1's "NOT ESTABLISHED — this experiment cannot yet tell": p = 0.1111 held-out, **0.0556** on final training loss (above the declared 0.05 by 11%), and **0.0278 = 1/36, the three-block floor**, on the trajectory average. So the verdict holds on the pair the check is designed around and is reading-dependent beyond it — the same shape as the head-paired arm's negative, in the same direction. |
| **The pooled finalist-order branch is stable; its explanation is not** | The pooled six-block rule that closed §9's open item stays **order-coupled** on all three readings — 10/7776 held-out, 82/7776 on final training loss, 2/7776 on the trajectory average, against a 1/7776 floor — because the deciding condition is that two different architectures top the six blocks, which is true in every reading. Its *second* condition, "the screening-free blocks alone clear 0.05", is 4/36 and 2/36 on the like-for-like pair but **1/36, the floor**, on the trajectory average, so the clause explaining why the pooled significance is trustworthy is the part that moves. |
| **Verified** | Two tests and six claims. Pointing the rule's declared `pooled_p` at 0.005 catches the branch-stability test, tightening `fresh_p` to 0.02 and repointing `FINALIST_ORDER_FRESH_SEEDS` at the coupled blocks both catch the screening-free test, and each captures the property the other does not. **195 -> 201 claims.** |
| **Result** | **1158 tests pass · 201/201 claims agree · 9/9 records reproduce.** |

## The head-paired arm's negative headline is the one place the split check moves a decision (2026-09-19)

| | |
|---|---|
| **What was found** | The head-paired arms are the phase's only *negative* headline - "head count is free" - and they are the only arms where **no test or claim read the ledgers under a training-loss reading at all**. Run the same check the other three arms carry and the picture changes: the parent arm's head contrast is non-significant held-out (paired p = 0.140, factorial p = 0.122) and **significant on final training loss (0.044)**, while its confirmation is the reverse (0.047 held-out, 0.159 on train loss). The four numbers straddle 0.05 in both directions. |
| **The pooled contrast is thin, and now named** | The six-seed pooled head contrast is **p = 0.0115 against `HEAD_PAIRING_RULE`'s declared 0.01** - a margin of **1.15×**, about six steps of that contrast's own 1/4096 sampling resolution. The record discussed this contrast as one whose threshold is cleared; the margin is what keeps the "uniform head effect" branch out, so it is a deciding number and is now checked. |
| **The branch holds on the like-for-like pair and reverses on a third reading** | Held-out loss and final training loss are both loss at the end of training, so they are the like-for-like pair: the rule lands on **HEAD EFFECT IS REAL AT WIDE WIDTH**, five conditions of five, on both. On `mean_train_loss` - which averages the whole trajectory, so it moves with convergence speed rather than fit - one of the six per-seed interaction contrasts is **−0.0050**, the seed-blocked interaction leaves its 2/2⁶ floor for 0.0625, and the rule reports **INTERACTION ABSENT - THE THREE-SEED AGREEMENT WAS SEED LUCK** instead. |
| **What does not move, and it is what the plan spends on** | Under all three readings the width contrast is negative and at the design's floor (−0.1617 / −0.1549 / −0.1865) and the H=896 level effect is negative with **all 12 units favouring more heads at the exact 2/2¹² floor** (−0.0553 / −0.0582 / −0.0344), while H=512 stays non-significant. So the recommendation rests on the level split, which replicates, not on the pooled contrast, which does not - and the arm's headline should be read with the margin named rather than as a null. |
| **Verified** | Four tests and seven claims. The branch-stability test fails when the rule's declared `main_p` is loosened or its label changed; the margin test fails on the loosened threshold; and flipping one of the 12 wide units moves the split from 12/0 at the floor to 11/1 at p = 0.052, which the pricing test catches. The new claims run at 20,000 draws rather than the record's 200,000 - verified to return each branch unchanged, and 4× cheaper - which took the four tests from 27s to 7.5s. **188 -> 195 claims.** |
| **Result** | **1156 tests pass · 195/195 claims agree · 9/9 records reproduce.** |

## The horizon arm carried the phase's headline and none of the arm's robustness checks (2026-09-19)

| | |
|---|---|
| **What was found** | The crossed arm's attribution is backed by two checks - a re-run on training loss and a leave-one-out over candidates - and §2c states both and tests pin both. The horizon arm, whose depth-sign result at 12,000 steps is §9's first headline, had **neither**: no test read its ledgers for robustness, and §9's horizon item gave the branch and its two qualifications without saying whether either check had been run. Neither had. |
| **The split check holds, and matters more at this length** | Re-running the identical factorial at 12,000 steps on `final_train_loss` keeps both main effects at the same signs and significant: depth **+0.0249** (p = 0.0077, against +0.0209 held-out) and width **−0.0667** at the same 1296-assignment floor. A longer schedule is where a divergence in how much each architecture overfits would widen, so this is the length where the check carries the most. |
| **One check can be carried, the other cannot** | The *spread* survives leave-one-out over all 8 candidates (worst blocked p = **0.0018**, two orders below the 0.05 threshold). The *crossed attribution* cannot be checked that way and **refuses on all eight**: at 2 candidates per cell, removing one leaves 7 rows that cannot fill four cells balanced on both seeds, so the contrast returns unavailable rather than computing over a subset no design supports. A 3-replicate design would support it; v2's 8-distinct-configuration arm did. |
| **And a check nobody had needed before shows two seeds is arithmetic** | The re-split test is two-sided, so the smallest p it can report is `2/assignments`. Two seeds give 1296 assignments and a floor of 0.001543; one seed has **36** and a floor of **2/36 = 0.0556 - above the 0.05 the rule reads against**. **No single seed can reach this branch at all**, which is why the stale one-seed record the landing left behind had no verdict to state rather than merely declining to state one. What *does* replicate per seed is the effect size (depth +0.0211/+0.0207, width −0.0618/−0.0673). |
| **Verified** | Four tests against the real ledgers, each proven to bite: removing the balance enforcement catches the refusal; reporting the floor as `1/assignments` catches the two-seed check; reading the split check on the wrong loss catches the sign/size assertion; and a constructed failure where the spread rests on one candidate (the other 7 tied within each seed) drives the leave-one-out p to **1.0** and catches the threshold. Six claims added to the record check, all recomputed from the arm's ledgers. **182 -> 188 claims.** |
| **Result** | **1152 tests pass · 188/188 claims agree · 9/9 records reproduce.** |

## The search loop's most expensive step had never been priced, because no arm had ever run it (2026-09-19)

| | |
|---|---|
| **What was found** | §10 prices the search program's loop at "~17 GPU-hours" - 100 candidates x 3 seeds screened at 400 steps plus 8 finalists x 3 seeds confirmed at **1,200** steps. §9 specifies the loop as confirming finalists **at the horizon the search actually trains for**, and that step had never been run by any arm, so it was never priced. The horizon arm now measures it at two lengths on one design, and the confirmation is the largest single term in the loop at any realistic horizon. |
| **The measured step** | Both rows are from this arm's own cost file, not extrapolated: **0.0523 h/run at 1,200 steps** and **0.524 h/run at 12,000**, over 16 runs each. The ratio is **10.016x for 10x the steps**, so cost is linear in steps between two *measured* points - which is what licenses the local-card extrapolation (0.133 x 10 = 1.33 h/run) rather than assuming it, and is worth recording because a sublinear rate would have made the confirmation cheaper than this. |
| **What the loop costs at each end** | Screening 100x3 at 400 steps is 13.80 GPU-h and confirming 8x3 at 1,200 is 3.19, giving the quoted 16.99. Confirming those 24 runs at the horizon instead costs **12.57 GPU-h on Kaggle T4 fp16 or 31.92 on this card** - so the loop is **17 GPU-h** with the confirmation at 1,200, **29.6** with it at the horizon on Kaggle, and **~49** on the local card. The confirmation is a bigger term than everything before it once the horizon is real. *(Corrected 2026-09-20: those two totals double-counted the 1,200-step confirmation - 29.6 is 16.99 + 12.57 and ~49 is 16.99 + 31.92, each adding the horizon confirmation to a loop that already contained one. The totals are 26.4 and 45.7; see the entry at the top for the composition and why the audit had agreed with it.)* |
| **The like-for-like caveat, now stated** | The "~40 GPU-hours for running all 100 at full length" that the 2.4x saving is measured against is also at 1,200 steps, and at a 12,000-step horizon that baseline is ~399 GPU-h. So the loop's saving is *larger* in ratio while costing far more in absolute terms - both of which a budget has to see, and neither of which a single "~17 GPU-hours" showed. |
| **Why this also settles a platform question** | The horizon phase ran on Kaggle rather than the local card for reasons its own NOTES gave (fp16 on Turing tensor cores); this is the price of that choice at the step where it matters most, and it is **2.54x per run** on the term that dominates the loop. Recorded here because the justification was written as a throughput ratio and is really a statement about the loop's biggest cost. |
| **Verified** | Seven claims added to the record check, all recomputed from the arm's cost file or from §10's own documented rates: both per-run costs, the linearity ratio, both platform confirmation totals, the loop total, and the platform ratio. **175 -> 182 claims.** |
| **Result** | **1148 tests pass · 182/182 claims agree · 9/9 records reproduce.** |

## The capacity sweep trained a third of its arms in order to throw them away (2026-09-19)

| | |
|---|---|
| **What was found** | `capacity_sweep.py` ran **7 trainings per (scale, seed)** and reported **4**. Every probe-selection row was configured `("random", "adaptive")` so that `probe_selection` - which governs only the adaptive arm - could tell the three rows apart, and the arm without that selection was kept only to be discarded: `adaptive` and `anti` each trained a random arm they never read, and `random` trained an adaptive arm it never read. The code's own note had flagged this as "a study-design question rather than this defect" and left it. |
| **Why it is safe, measured rather than argued** | `run_v0_session` trains each arm in its **own subprocess** from the same config and seed, so the arms are independent and removing one cannot move the other. Checked directly at the dev shape (120 steps, seed 42, argmin): the adaptive arm's validation loss is **bit-identical** - 9.031953811645508 - with and without its pair, at 22.5s against 43.8s. |
| **The end-to-end A/B** | The same 120-step command with all four arms, run both ways on the real card: **64.7s paired against 38.0s single-arm, 1.70x** - and the four rows agree exactly (`null 5.626723766326904 / random 5.6035860776901245 / adaptive 5.653473734855652 / anti 5.658488988876343`, mutations 0/1/1/1). The per-row timings show the mechanism: the three probe-selection rows cost 17.8-18.2s paired against 8.5-9.3s solo, while the null row is 9.3s either way. |
| **What changed** | `make_config` now sets `arms=(ARM_MODE[arm_name],)` - the session is asked for exactly the arm the row reports - and `ARM_MODE` is what the config is built from rather than a separate lookup, so a row cannot report an arm the session was not asked for. `ARM_CONFIGS` loses its `arms` entries. |
| **Verified** | A test pins the *config* rather than the saving, because the saving follows from the config and a pair that re-grew there would be invisible in every result file; it names one training per reported row. Caught when the pair was re-grown, and the A/B above is the independent check that the rows did not move. |
| **Result** | **1148 tests pass · 175/175 claims agree · 9/9 records reproduce.** |

## The promotion threshold silently borrows a shorter stage's noise, and the record claimed it was measured there (2026-09-19)

| | |
|---|---|
| **What was found** | Now that the horizon is measured, the search program's confirmation length is known - and the promotion threshold at that length needed a seed std that the table of measured stds does not contain. `seed_std_for_steps` takes the nearest measured length *at or below* the stage, which is the conservative direction for a short screen and is the **opposite** of what a long confirmation gets: at 12,000 steps it reuses the 1,200-step 0.0148, which is a **2.64x over-estimate** against the 0.0056 this arm's own 12,000-step rows measured. Safe, and costly: the threshold is the std times `sqrt(2 / seeds)`, so the borrow buys extra seeds rather than extra safety, and nothing in the output said so. |
| **The record asserted a provenance it did not have** | Worse than the number: the decision file's caption read "the measured per-architecture seed std **at that length**", and the table printed the std with no indication of which length it came from. A 12,000-step cut therefore read as a 12,000-step measurement. The std is now rendered with the length it was measured at whenever that is not the stage's own (`0.0148 (measured at 1200)`), and a stage measured at its own length stays unannotated so the note appears only where it is true. The new key is `std_measured_at_steps`, read from `seed_std_measured_at_steps` rather than inferred, so the two cannot drift. |
| **The measured figure is deliberately not substituted** | 0.0056 is not the same measurement as 0.0148 and swapping it in would mix designs inside a table whose whole justification is that its numbers were measured the same way: a different candidate set (the crossed factorial against the matched v2 set), a different precision (fp16 against fp32), and two seeds against three. The arm's own 1,200-step anchor reads **0.0118** where that table reads 0.0148 at the same length - which is the direct measure of how unlike the two designs' noise levels are, and the reason the substitution was refused. |
| **Verified** | Two claims added to the record check, so the factor is recomputed from the arm's own rows rather than from a printed decimal (**142 -> 175 claims** across the three stretches today): a 12,000-step threshold borrows the 1,200-step std, and that borrow over-estimates the measured noise by 2.64x. Three mutants caught - a table length no longer reporting itself, the annotation removed from the renderer, and the caption restored to its false claim. No committed record renders this table, so the renderer change produced no churn in the nine committed `decision.md` files. |
| **Result** | **1147 tests pass · 175/175 claims agree · 9/9 records reproduce under `--strict-rank`.** |

## The horizon arm landed, and the file left as its record under-read it by half (2026-09-19)

| | |
|---|---|
| **The arm** | `act-t4-archzoo` completed on Kaggle T4×2: 4.83 h wall clock, `complete=True`, **32/32 rows** (8 candidates × {1200, 12000} × seeds 42/43) at ~9.22 GPU-hours token-derived against its own pre-registered 15 GPU-h cap. This is the grid whose horizon half the first attempt lost entirely (16 rows, none at 12,000), so the fix that re-ran it worked. |
| **The branch** | The pre-registered rule lands on **shallow-wins persists**: depth **+0.0209 nats, exact re-split p = 0.003086** (floor 0.001543) and width −0.0645 nats at its own floor. The depth sign is horizon-stable to 10×, so §2c's "spend at low depth" stands at the length a real budget occupies — the reading prior work predicted would reverse does not. Two qualifications come with it: the decay over 1200→12,000 is **1.38× per doubling** against the 1.65× fitted over 400→1200, so §3's rate *over-predicts* decay when extended an order of magnitude; and the **winner changes** across 10× (`W384` → `W1536`) even though the ranking holds at Spearman 0.90. Against its own fp16 anchor the spread falls 0.3085 → 0.1059 nats while seed std falls 0.0118 → 0.0056, leaving the effect at **18.88× its own seed noise** — it does not wash out. |
| **What was wrong with the landing** | The driver imports the kernel's own `decision.md`/`ranked.json`/`search_cost.json`, which are whichever sweep finished last and are **not** the arm's record. The local re-analysis that is the record launched and was killed inside the **4.2 seconds** it runs for, so the file left in place was the kernel's *single-seed* write: it reported "PROVISIONAL — single-seed evidence" and "Run at least two seeds per architecture before deciding" about a complete two-seed arm. Nothing noticed, because the watcher's log stopped mid-line with no `exit=` line to say the attempt had failed. |
| **How that is provable rather than inferred** | `decision.md`'s mtime is **older than all four of its own ledgers** (18:32:47 against 18:32:47.8–18:32:49.2), so it was not derived from the rows that landed. That is a machine-checkable property, not a timestamp curiosity: the analysis takes 4.2 s, the ledgers arrived at 18:32:47–49 and the watcher launched at 18:33:55, so the window is exactly one killed process. |
| **Fix 1 — the analysis is unbuffered** | `python3 -u` in the watcher. The redirected stdout was *block*-buffered to a file, so a kill inside that 4-second window loses every line **and any traceback** — which is why the log had no record that anything had been tried. The failure was invisible not because it was unlogged but because the log was in a buffer that died with the process. |
| **Fix 2 — the landing is verified, not assumed** | The watcher reported on an exit status alone, and 0 says the process finished, not that it wrote. It now **re-renders the record from the arm's own documented command and requires it back byte-identical** (`check_record_freshness.py --only`), which is the same check the test suite runs. Deliberately *not* a timestamp comparison, and the weaker version was tried first: asserting `decision.md` was newer than the ledgers passed the mutation probe even when its freshness test was gutted, because an unrelated ledger-to-ledger mtime comparison in the same file satisfied a substring check — and it would also fail a *correct* record on a no-op re-run where the analysis wrote nothing. Reproducibility subsumes it and names the cause. |
| **Fix 3 — the general guard, in content rather than in mtimes** | `check_record_freshness.py` now fails any arm whose committed record was read over a **seed set no documented command reads**. mtimes do not survive a fresh clone, so the durable invariant is the one the record states about itself: the horizon arm's file claimed seeds `[43]` while its own runbook globs `[42, 43]`. The check is cheap (it reads ledgers, no analysis) and it names the cause — "the committed record was read over seeds [43], but no documented command for this arm reads those" — instead of the symptom a byte diff reports after paying for a full regeneration. Equality per candidate, not containment: an arm's directory may legitimately keep a standalone and a pooled write side by side, which the v1 record does (six seeds recorded over three in its own ledgers). |
| **The same class in the other two watchers** | Both had the identical shape (`python3 … >> "$LOG" 2>&1`) and neither had failed yet, but the reasons to fix them are specific rather than prophylactic. `run_finalist_analysis_when_sweep_exits.sh` deliberately does **not** write a `decision.md` - "writing v1's decision.md is a deliberate step taken once the branch has been read" - so its log *is* the arm's only record, and a block-buffered run would discard exactly the verdict and every condition it exists to preserve. `run_confirm_analysis_when_sweep_exits.sh` ran **two** `--write` analyses and verified neither, and - unlike the other two - never resolved its own directory, so launched from anywhere but the repository root it finds no ledgers and reports a false "ledgers incomplete" against a complete arm. |
| **Why one log kept its failure and the other kept nothing, measured** | The asymmetry is buffering, and it is checkable rather than argued: the same `print` into the same redirection, killed immediately, leaves **0 bytes** on plain `python3` and **20 bytes** under `-u`, while the same line written to `stderr` leaves 17 bytes either way. So a traceback survives a kill and ordinary stdout does not - which is exactly the two logs. The finalist watcher's kept the *full traceback* of the analyzer defect it caught (stderr); the horizon watcher's kept *nothing at all*, because its run was killed before stdout flushed and it had no traceback to fall back on. |
| **The landing machinery now has tests, and the first version of them was too weak** | None of the three watchers was covered by any test, which is how a landing could be wrong without anything noticing. Five tests now pin the properties: unbuffered analyses, a write-path watcher that confirms its landing, a resolved working directory, and the content guard's behaviour plus its wiring into `check_arm`. **The first attempt at the landing guard was a substring check for `-nt`, and it passed against a deliberately gutted freshness test** - because that same watcher computes its newest ledger with an unrelated ledger-to-ledger `-nt` comparison, so the string was present whatever happened to the record check. That is the mistake the doc-index test had already recorded once (reading prose or coincidence as code), and the fix was to assert the *repository's own* record check instead: strictly stronger, and already covered by the rest of the module. **5/5 mutants caught** afterwards - the content guard disabled, the wiring removed, `-u` dropped, the landing check replaced, and the `cd` deleted. |
| **Verified** | The new guard was proven to fire by rebuilding the *exact* defect in an isolated artifact tree — the kernel-side single-seed `ranked.json` beside the real ledgers — where it failed with the message above and **0/1 records reproduced**. Across the committed arms it passes **9/9**, and it was checked against every arm *before* being added, so it is not a check that was fitted to one case. |
| **Record extended** | `verify_record.py` gained `check_horizon_10x`: **31 claims** (142 → 173) recomputed from the arm's own ledgers — the 32-row grid, both lengths' spread and seed-noise units, the decay rate and that it is below §3's 1.65×, the branch the rule returns, both main effects and their floors, the interaction at its 2-seed floor, the Spearman and that the winner is *not* preserved, the fp16 offset and that it is smaller than the effect it carries, and the phase's own cost against its own cap. The Spearman needed an explicit tolerance: quoted to two places (0.90) against a computed 0.9048, the default 5e-3 relative tolerance is 0.0045 and the difference is 0.00476 — the record is right and the audit was too tight for a two-decimal quote, the same rounding case §4a already carries. |
| **A cost decision left as a decision** | The horizon arm's 9.22 GPU-hours are **not** folded into §10's closed 23.632 total. It opened its own phase under its own 15 GPU-h cap, declared in its own pre-registration, so summing them would invent a figure neither declaration authorised. Recorded as a separate one-row table beside the phase it did not extend. |
| **Result** | **1145 tests pass · 173/173 claims agree · 9/9 records reproduce under `--strict-rank`** (all nine now report `ranked.json identical`, where the horizon arm's would have been regenerated from the wrong rows). |

## The p-value helpers' guards, and what a surviving mutant actually means (2026-09-19)

| | |
|---|---|
| **What was found** | A 22-mutant sweep over `_blocked_permutation_p`, `blocked_permutation_test` and `paired_contrast_test` - the functions every p-value quoted anywhere in this phase passes through - left 10 alive. Four were real guards, now pinned. (a) `assignments <= cap` still **enumerates**: the contract is "when the enumeration exceeds cap this samples", so a cap equal to the count leaves the reported method as `exact`, which is what the record's exact claims rest on. (b) `samples <= 0` returns `unavailable` rather than letting the `(r + 1) / (n + 1)` correction report a confident p=1.0 from **no draws at all**. (c) `len(differences) > 1` keeps the `n - 1` division out of reach for a single pair, so a one-pair contrast reports no t rather than a t with zero degrees of freedom. (d) `d < 0` means an exact tie is not counted as evidence *against* sign agreement, which would report a unanimous contrast set as inconsistent - the opposite reading. |
| **What the other six are** | Not gaps, and the distinction is worth recording because a bare "survived" reads like a hole. One is **provably** equivalent: `resolutions <= 1_000_000` can never see equality, because `resolutions` is `2 ** len(differences)` and so always a power of two, while 1,000,000 sits strictly between 2^19 and 2^20 - checked rather than argued, `<=` and `<` are the same function there. The other five differ only on an exact float equality or a specific RNG draw: the three `>= observed - 1e-12` comparisons differ from `>` only when a statistic lands exactly on `observed - 1e-12`, a value the ledger sums can produce only by coincidence and which the epsilon exists to bracket; and the Monte-Carlo coin flip `generator.random() < 0.5` differs from `<=` only on a draw of exactly 0.5, which the Mersenne Twister can make (one value in 2^53) but no seeded test does. |
| **Verified** | **4/4 real guards caught** where all four lived beforehand, and the six unreachable ones re-run and confirmed to survive - which is the evidence for that characterisation rather than an assertion about it. This is the third and fourth equivalent mutant this vein has produced, after the sign clause and the tie branch. |
| **Where the vein ends** | All five decision rules, `verdict` and its attribution, and now the p-value primitives have been swept. Every real survivor across the four sweeps was the same defect class: a threshold or guard compared at **exact equality** with no test asserting which way it decides. That is the durable finding, not any individual fix. |

## The same boundary defect in the last two rules, found by enumerating instead of hand-picking (2026-09-19)

| | |
|---|---|
| **What was found** | Rather than hand-picking edits, every comparison operator in `finalist_order_decision` and `set_comparison_decision` was swapped in turn against its dedicated test file: 18 mutants, 6 caught, 12 survived - and 6 of those 12 were inside the docstring tables (the filter skipped lines beginning with `"` but not `|`). **All six real survivors were exact-equality boundaries**, the same class as the two entries below. |
| **The five that were real** | In the finalist-order rule: `established = p_all < pooled_p` and `fresh_clears = p_fresh < fresh_p` are strict, so a p sitting on the threshold does not establish a replicated ordering and the verdict falls to *coupled* rather than *established*; and `floor_limited = pooled_floor >= pooled_p` is inclusive, since a floor equal to the threshold cannot produce a smaller p and leaves no branch reachable. In the set comparison: `ratio_ok = abs(budget_ratio - 1.0) <= budget_deviation` - the only *inclusive* condition in the codebase, and correctly so, because parity is a band and only an exceedance is rejected - and `significant = p_value < p_threshold`. |
| **The sixth survivor is an equivalent mutant, proven rather than assumed** | `branch = "manual-better" if difference < 0 else "random-better"` puts a pair whose means are exactly equal on the random side, and swapping `<` for `<=` changes nothing. That is not a missing test. The p-value is the share of permuted statistics at least as extreme as the observed one, and *every* permutation is at least as extreme as zero, so a tied pair reports **p = 1.0** and cannot clear any threshold a p-value can be compared against; the clause needs a threshold above 1.0 to be reachable at all. The invariant is now asserted (`difference_nats == 0.0` and `p == 1.0`) rather than reasoned about and filed, because a survivor being a hole or an equivalent mutant calls for opposite responses and the difference is checkable. This is the second equivalent mutant this vein has produced, after the sign clause below. |
| **What was pinned** | Five boundaries, **5/5 caught** where all five lived beforehand. The tie clause is deliberately left unpinned, with the invariant that makes it unreachable pinned in its place. |
| **Why the ratio boundary needed care** | The existing `test_boundary_budget_ratio_is_accepted` passes a ratio of `1.025`, which reads as the boundary but is not one: `abs(1.025 - 1.0)` is `0.02499999999999991`, strictly inside the declared `0.025`, so the check passes under `<=` and `<` alike and cannot see which the rule uses. That is exactly why the mutant survived. The new test takes the deviation from the ratio's own float difference, making the comparison exactly equal by construction, and says so. |

## The verdict's action named the wrong gene, and the test only checked that both genes were named (2026-09-19)

| | |
|---|---|
| **What was found** | A 9-edit mutation sweep over `verdict()` - the function that turns the evidence bundle into the headline and the *next action* - left **5 alive**. The serious one: swapping `max(resolved, key=abs(nats))` for `min` survived. That is not cosmetic. The action is an instruction about where to spend the next phase's budget - "spend the search budget there" - so attributing the effect to the *smaller* of two significant genes reverses the instruction. The existing attribution test asserted only that `` `hidden_size` `` and `` `depth` `` both appeared in the text, which they do either way: once as the lever and once as the qualification. It could not see the two swapped, so the mutant read as green. It now asserts which way round they are. |
| **Three exact-equality boundaries** | Also unpinned, and the same class as the rule boundaries fixed in the entry below: the materiality floor (`spread <= 0.01`), the significance threshold (`p_value < 0.05`), and the per-gene attribution threshold (`row["p"] < 0.05`). All three are numbers the pipeline returns *exactly* - the floor is a declared constant and both p-values come from a discrete test - and each decides **against** spending compute, so reading one generously buys work on an undecidable result. Now pinned at equality. |
| **What was pinned** | **4/4 caught**, where all four lived beforehand: the gene swap, the floor boundary, the verdict threshold, and the per-gene threshold. The attribution test was extended rather than duplicated, since the defect was in what it asserted rather than in what it covered. Each new assertion was confirmed to fail against its own mutation before being kept. |
| **A wording looseness left alone** | At a spread of exactly 0.01 the headline reads "spread 0.0100 nats is **below** the 0.01 nat floor", which is false at equality. The refusal itself is correct and is the conservative direction; only the sentence is loose, and only at an equality a continuous statistic is unlikely to land on. Left unchanged as cosmetic rather than regenerating committed decision text for it. |

## The rules decide at exact equality, and the clause that looked like a gap is an equivalent mutant (2026-09-19)

| | |
|---|---|
| **What was found** | The phase's three pre-registered rules (`replication_decision`, `head_pairing_decision`, `horizon_decision`) are the only thing between a finished arm and a published branch, and each turns on a p-value compared to a declared threshold. A 20-edit mutation sweep over those comparisons - `<=` where the code says `<`, `>` where it says `>=`, dropping a clause - left **7 alive**. Two are the boundaries: both rules decide at *exact equality*, and nothing asserted which way. The third is the largest of the seven: `REPLICATION_RULE` - the thresholds the confirmation arm's entire verdict hangs on - is referenced by **no test at all**, so `ratio: 1.5` could become `2.0` with the whole suite green. |
| **The clause that survived, and why that is a different thing** | Dropping `and consistent` from `resolved` also survived, in *both* the replication and head-paired rules. At the arm's declared seeds this is not a test gap but an **equivalent mutant**: the clause cannot change any verdict there. The interaction p comes from a sign-flip permutation test over the per-seed contrasts, and any contrast set that *mixes* signs bottoms out at **0.0625** - the lone dissenting seed always has its own sign flipped, which only makes the observed sum larger, so at least four of the 64 sign vectors count and the smallest reportable p at six seeds is exactly twice the floor (`2/64 = 0.03125`). Since `0.0625 > 0.05`, `p < 0.05` already implies agreement at six seeds, so the clause is never the deciding one. **Seven seeds is where it becomes load-bearing:** six contrasts at -0.06 and one at +0.04 clear the declared threshold at p = 0.03125 while the signs disagree, which is the *absent* branch rather than the resolved one - the two branches the clause exists to separate. Confirmed by brute force over random mixed-sign vectors for n = 4..8, where the minimum is 2x the floor in every case. |
| **What was pinned** | Five tests. (a) A p landing *exactly* on the threshold is not significant - ``p < main_p`` is strict - and a floor *equal* to the threshold still refuses the branch, since a floor that cannot produce a smaller p makes every branch unreachable. Both are numbers the discrete exact tests actually return, not boundaries the data is guaranteed to avoid, and both are asserted in the horizon rule and the replication rule. (b) `REPLICATION_RULE` is asserted equal to its declared `{main_p: 0.01, ratio: 1.5, interaction_p: 0.05}`, for the reason the module's own comment gives: a threshold that lives only in a document gets reapplied by eye once the numbers arrive. (c) The sign clause is exercised at **seven seeds in both rules**, since both carry the same clause over the same test - with the arithmetic above recorded in each docstring so a future reader knows why six seeds cannot exercise it. |
| **Verified rather than assumed** | Each new test was run against its own mutation and confirmed to fail: dropping the consistency clause in either rule, `<=` for the replication interaction threshold, `<=` for the horizon threshold, `>` for the horizon floor, and `ratio: 1.5 -> 2.0`. Six mutations, **6/6 caught**, where the same probe left all six alive beforehand. The test file also had to be changed to import `REPLICATION_RULE`, which is itself the evidence for the finding above. |

## The standing documents moved to docs/, and the layout is now checked (2026-09-19)

| | |
|---|---|
| **Why** | The repository root carried five documents of 705, 1,236, 2,700, 3,264 and 6,756 lines. A reader or an agent arriving at the root saw five walls and no statement of which was which, and nothing put them anywhere else. `docs/` now holds the four standing documents - the plan, the report, the status log and the runbook - while `artifacts/` keeps the per-arm and per-benchmark evidence beside the data it describes, which is where that belongs. `artifacts/ARCH_SEARCH_FINDINGS.md` deliberately stayed: it is the search phase's own synthesis, not a standing document. |
| **How large they actually are, and whether that is a problem** | Measured from the history rather than guessed. `RESEARCH_REPORT.md`: 557 lines on 09-13, 1,967 on 09-14, 5,349 on 09-15, 6,755 on 09-16, 6,756 on 09-19. `IMPLEMENTATION_STATUS.md`: 210 on 09-12, 501, 1,007, 2,071, 2,578, 2,700 on 09-19. So the size is *accumulated research output*, not unbounded accretion over time: the report grew 3,382 lines in one day of active study and **one line in the three days since**. Both files grow when work lands and stop when it does not, which is the right behaviour for a results narrative and an append-only log. The forward risk is rate, not age - another 09-15 adds 3,400 lines to a file already at 6,756. |
| **What had to change** | Only four places in the repository construct a path to these documents: `tests/test_doc_index.py`, `tests/test_report_artifact_links.py`, `tests/test_fused_vs_frozen_observation.py`, and `scripts/kaggle_queue.py`'s `CODE_PATHS` - the set bundled into the Kaggle code dataset, which now lists `docs` instead of the three filenames. Everything else that mentions them is prose or a docstring. The runbook flow is unaffected; `CODE_PATHS` is read at *publish* time, not on import, so the horizon arm currently running is untouched. |
| **The convention this settles** | Documents are cited by **name**, not by path - roughly 45 files say "pre-registered in `RESEARCH_REPORT.md` section 5.30", and the plan is cited only as "plan section N". Those references stay readable and unedited, including in this append-only log; the README's Documentation table is what maps a name to its location, and it now says so explicitly. Anything that *reads* a file got a real path, which is the distinction that matters. |
| **Pinned** | `tests/test_doc_index.py` enforces the layout instead of performing it once: no markdown may sit at the repository root except `README.md` and `AGENTS.md`, and every `docs/*.md` must be named in the README index. The guard was proven to fire by dropping a stray document at the root, which it named. |

## The mixer benchmark's conclusions hold; three of its stated precisions did not (2026-09-19)

| | |
|---|---|
| **What was found** | `artifacts/mixer-benchmark/FINDINGS.md` is tracked, and it is the document that rules the module-type axis out of the search space ("not on this hardware, in this implementation"). Its table and its argument check out: every `tok/s`, `s/step` and peak-memory figure matches `mixer_throughput.json` exactly, all four OOM layouts appear as OOM in the raw log, the `all_attention` control is a real `a/a/a/a/a/a/a/a` layout, and the conclusion - GDN at 4-7x the wall clock per layer, three of nine layouts unrunnable - survives every number as written. What did **not** hold were three stated precisions. |
| **The three** | (a) "All nine layouts ... land within **0.03%** of each other on total parameters": the recorded totals are `100,725,504` / `100,762,368` / `100,799,232`, a 73,728 spread - **0.073%** of the mean, 0.051% at its furthest from it. (b) The `vs attention` column **truncated** two rows where it rounded the other two, showing 6.8x for ratios of 6.850 and 6.852 against 6.9x for 6.921 - three rows that read as different were the same to within 0.07. (c) "reproduces ... to within **1%** (4,793 vs 4,733 tok/s)": the two figures are **1.27%** apart. |
| **What else the check found** | The four OOM layouts carry no `total_params` in either `mixer_throughput.json` or `bench.log`, so their totals in the table come from the block geometry rather than from a measurement - now stated. And the 4,733 in (c) is not recorded in that directory: the only throughput committed there for that shape is this benchmark's own 4,793, and the v0 zoo records **573** for `D8_H768_E8_A2`, 8x lower. The comparison can be checked arithmetically but not reproduced from the artifacts beside it. |
| **Worth recording, not fixing** | The benchmark's own log warns on **every** pattern that there was "no token file given: using uniform random tokens. Losses on this stream measure initialization, not architecture" - and `FINDINGS.md` correctly reports no loss figure at all, and the table's conclusion rests only on throughput and memory. That is the same defect `SCREENING_AUDIT.md` refuted the v0 screening for, avoided here by not reporting the losses. Verified rather than assumed. |
| **Fixed** | The three precisions are corrected to the values the data supports, each with the arithmetic shown, and the OOM-provenance and no-loss facts are stated where a reader will meet them. |

## A tracked screening analysis recommended stopping while its own audit refuted it (2026-09-19)

| | |
|---|---|
| **What was found** | `artifacts/arch-zoo/analysis.md` is tracked and reads as a live negative result: it reports a screening table, concludes "Should We Proceed with Evolutionary Search? **NO - not yet**", and recommends 10-15 GPU-hours of follow-up on the strength of it. Its sibling `SCREENING_AUDIT.md` - also tracked, and the file `ARCH_SEARCH_FINDINGS.md` sends a reader to - records five defects, each independently sufficient to invalidate the comparison, and states that "the v0 results should not be cited as a negative result about architecture". `analysis.md` itself carried no warning and nothing linked to it, so a reader arriving there first saw a decision with no hint it had been overturned. |
| **The numbers were right; the conclusion was not** | Every figure in its synthetic table is faithful to `zoo_results.json`: all 11 losses match, the range is exactly 0.0376, and the std dev of 0.0102 is the *population* std dev (sample is 0.0107 - a convention, not an error). `SCREENING_AUDIT`'s own claims check out too: `D8_H768_E8_A2` (44.1M active) and `D10_H768_E6_A2` (53.5M active) are both recorded at exactly `201451008.0` flops/token, and `artifacts/arch-zoo-real/` does hold a `zoo_definition.json` and no results file. So this was not a stale-numbers problem. |
| **Fixed** | `analysis.md` opens with a superseded header in the form its own audit established: the reason the synthetic table ranks initialization luck - uniform random tokens at context 512, every loss between **0.43% and 0.85%** above `ln(8192) = 9.0109` (the audit rounds that to "0.4% to 0.8%") - the brief's requirement of context 1024/2048, and the two real-data losses (3.5865, 3.6657) that have no artifact behind them at all because that sweep wrote results only on completion and was interrupted. |
| **Also closed here** | The `96k tokens/s` in `_MoEFeedForward`'s docstring - the one figure of the four measured last entry that sits *above* the measurement and so can still under-provision a budget - now points at `PERFORMANCE_NOTES.md` for the measured rates. |
| **Pinned** | `tests/test_doc_index.py` gained the check that the per-arm document kinds are actually covered by the index's patterns. It failed against the index as written moments earlier, naming `artifacts/mixer-benchmark/FINDINGS.md` - a real gap in the table. The existing "every document the index names exists" check was also made less brittle rather than merely satisfied: a bare filename in prose names a kind of document, so it need only exist somewhere in the project rather than at the repository root. |

## The throughput warning was stale in the other direction, and its arithmetic mattered (2026-09-19)

| | |
|---|---|
| **What was found** | Reading the tracked per-arm notes - the set the new README index marks as not machine-checked - surfaced `artifacts/arch-zoo/PERFORMANCE_NOTES.md`, which records a **75x spread** across the repository's own stated throughput for the dev shape (9.8k in the model docstring's table, 19.6k in its prose, 96k in `_MoEFeedForward`'s, 8.7k measured there) and warns that a budget drawn from 19.6k under-provisions by 2.2x and from 96k by 11x. Budgets in this phase are the binding constraint, so the warning was re-measured rather than cited: `PYTHONPATH=src python scripts/dev_cycle.py --steps 400 --tag verify-doc-throughput` in a fresh checkpoint namespace, on the current code, fp32, batch 8 x context 512, the real TinyStories stream. |
| **The measurement** | **24,299 tokens/s end to end**, 134.9 s for 3,276,800 tokens across two arms (random 24,807, adaptive 24,386). That is **2.8x** the 8.7k the note reports for the same nominal shape and **1.24x** the 19.6k docstring figure. Recorded in the note as a re-measurement, not a refutation: the two are probably not the same measurement (the note records 617 MB peak where the model docstring records 0.90 GiB for that shape, and a 20-step benchmark need not be at steady state). |
| **Why it changed the warning's arithmetic** | On the measured scale the four figures split in two directions. 8.7k, 9.8k and 19.6k all sit *below* 24.3k, so a budget drawn from any of them over-provisions - it wastes GPU-hours, which is the safe direction. The 96k in `_MoEFeedForward`'s docstring sits 4x *above* the measurement and is the only one still able to under-provision, which is the failure the note was written about. The first draft of this entry said all three flagged figures were below the measurement; that is wrong for the 96k and was corrected before committing. |
| **Deliberately not changed** | `src/act/model/v0_moe.py`'s docstrings were left alone. Its wall-time claim is right - 2.8 minutes for a two-arm 400-step dev cycle against 2.25 minutes measured - and the only figure needing a caveat, the 96k, is already covered by the tracked note. No test pins any of these strings. |

## The capacity sweep's probe arms were recording the random arm, bit-identical at every scale (2026-09-19)

| | |
|---|---|
| **What was found** | `scripts/capacity_sweep.py` read `asdict(result.arms[0])` as the sweep arm's result. Every probe-selection arm is configured `arms=("random", "adaptive")`, so `arms[0]` is the plain random-churn arm for all three of them - and the `probe_selection` that distinguishes them (`random`/`argmin`/`argmax`) governs `arms[1]`, which was never read. The `adaptive` and `anti` rows therefore carried the `random` arm's numbers. The project's own self-check already encodes the convention: `v0_session_selfcheck.py:119` reads `adaptive, random_arm = reference.arms[1], reference.arms[0]`. Found from the scratch output's signature, not from the code: `artifacts/capacity-sweep/threshold_sweep.json` records 20M/seed 42 as `random` = `adaptive` = `anti` = `4.343077778816223` exactly, with `random` taking 57.1 s and the other two 1.5 s and 1.4 s; `sweep_results.json` beside it has 8 of 12 rows carrying the 20M row's `final_loss`, `train_flops`, `search_flops` and `cost_ratio` verbatim while `scale`, `total_params` and `active_params` differ and `train_time_s` is 0.75-0.83 s, so the 40M and 60M scales were never trained. |
| **Verified before it was changed** | One real step per sweep arm returned modes `['null']`, `['random', 'adaptive']`, `['random', 'adaptive']`, `['random', 'adaptive']`, with `arms[0].mode == 'random'` in all three probe cases and the adaptive arm's own loss (`8.689111` against `8.7015`) computed and thrown away. |
| **Fixed** | `ARM_MODE` names the session mode each sweep arm reports, and `run_one` selects by mode rather than by position, raising when the session did not return it. A mode the session returns is now the only thing that can be recorded. `main` also printed a failed run and wrote `threshold_sweep.json` regardless, so a file written from no results at all read as a sweep whose arm was merely quiet - the same shape that lost the horizon arm a session. An empty result set is now refused rather than written. |
| **Pinned** | Three tests in `tests/test_capacity_sweep.py`. The arm-recording one was proven to fail on the old behaviour before being kept (`{'adaptive': 2.0} != {'adaptive': 3.0}` - the random arm's loss). A real 120-step four-arm run now gives null `5.6267` / random `5.6036` / adaptive `5.6535` / anti `5.6585`: four distinct values where `adaptive` and `anti` were both `5.6036` before, and the two probe-selection directions now separate. |
| **Scope** | The sweep is unregistered, `artifacts/capacity-sweep/` is entirely gitignored, and no committed document cites any of its numbers (each loss value was searched for). So no live claim depends on the defective rows. What it does explain is why the `analysis.md` in that directory reports "adaptive selection is indistinguishable from random" so cleanly: for the probe-selection arms it was comparing a measurement against itself. The plain random arm is still trained inside the adaptive and anti sessions and discarded there - 2x the compute for those two rows - which is left as it is because that is a study-design choice rather than this defect. **Addressed later the same day, once it was measured rather than assumed: see the entry at the top of this log on the sweep's discarded arms - 7 trainings per (scale, seed) for 4 reported rows, and the rows are bit-identical without the pair.** |
| **What finding it cost** | The end-to-end check overwrote the gitignored scratch checkpoints `artifacts/v0-sessions/20M-seed-42-{null,random}` and created `-adaptive` and `-anti` beside them. No tracked file was touched. |

## The standing documents were accurate and unindexed, and the plan's subtitle described a different plan (2026-09-19)

| | |
|---|---|
| **What was found** | The documents are in good shape where it counts. The report's section 5 numbering is complete (`5.1`-`5.47`, each number carrying its *Pre-registration* beside its *Outcome*), all 28 "plan section N" references resolve to sections the plan has, and every cross-reference between documents points at a heading that exists. What did not exist is a map. Nothing said which document owns which question, and the front door (`README.md`) linked to exactly **one** of them. The reader is handed 46 markdown files, three of which are 2,651, 3,264 and 6,756 lines, with no statement of which is prescriptive and which is a log. Separately, `complete_implementation_plan.md` was subtitled **"Kaggle-Free Compute Edition"** while its own section 3 makes Kaggle T4x2 the primary backend, section 4 the TPU second, with section 5 the Kaggle operating model, section 39 Kaggle notebook architecture and section 82 Kaggle compute allocation - 28 Kaggle mentions and four dedicated sections. The word "edition" occurs once in the file, in that line. `git show -M 75874f6` shows the file was renamed from `... - Complete Kaggle Implementation Plan.md` and the rename's only content change was a trailing newline, so the subtitle contradicted the body in the original document too. |
| **Fixed** | `README.md` carries a Documentation section: a table giving each standing document its job and what it is authoritative for, a second table for the per-arm material (which `NOTES.md` files are *executed* by `check_record_freshness.py` and which `decision.md` files are *generated* by `arch_zoo_analyze.py`), and the rule that a load-bearing number is owned by the ledger behind it. The plan's subtitle now reads `Kaggle T4x2 primary, TPU v5e-8 secondary`, which is what its own sections say. |
| **Pinned** | `tests/test_doc_index.py` makes the map checkable instead of trusted: every standing document at the repository root must be named in the index, every document the index names must exist, and every `RESEARCH_REPORT.md` section reference and every "plan section N" reference must resolve to a heading that is actually there. Each of the four guards was proven to fire on its own defect before being kept - an unindexed root document, an index entry naming a missing file, and dangling section references in both directions. |
| **Why this class matters** | It is the same defect this phase has now produced twice more: a runbook that no longer reproduced its own record, and a reproduce block that wrote somewhere the committed artifact was not. A record that cannot be read as a map fails the same way as a record that cannot be regenerated, just at the level of the whole document set rather than one arm. |

## A scratch sweep ignored its own arguments, and the reset it opened with reset nothing (2026-09-19)

| | |
|---|---|
| **What was found** | `scripts/capacity_sweep.py` has no argument parser, so `--help` began a full four-scale sweep instead of printing the flags, and a misspelled flag was dropped while the default run went ahead. It was found by running `--help` to read those flags, which started a run. The same script opens by resetting state and named `<scale>-seed-<seed>`, while every run writes `<scale>-seed-<seed>-<arm>` - and `run_v0_session` **resumes** from an existing checkpoint directory, so the reset matched nothing and each re-run continued from the previous one's weights. |
| **Fixed** | The axes are real flags (`--scales/--arms/--seeds/--steps/--out`), each validated or refused by name, and the reset now names the directory the run is handed. Both pinned in `tests/test_capacity_sweep.py` against a faked session, because the sweep has no study summary and nothing else reads it. |
| **What finding it cost** | A `--help` invocation overwrote git-ignored scratch checkpoints under `artifacts/v0-sessions/20M-seed-42-{null,random}`. No tracked file was touched (`artifacts/*` is ignored) and nothing was deleted, because the reset named a path that never existed. |
| **Scope** | The script is unregistered and referenced by no test, doc or shell script, so its output is still not part of any record. The fix is only that it no longer does something other than what it says. |

## The finalist arm resolved v1 as coupled, and getting there exposed a record that could not be regenerated (2026-09-19)

| | |
|---|---|
| **The verdict** | 9 runs, 22.1M tokens, 1.236 GPU-hours. Pooled to six seed blocks, the pre-registered rule returns **`order-coupled`**: the pooled exact test clears at p=0.001286 (floor 1/7776) while the screening-free blocks alone neither clear the threshold (p=0.1111, floor 1/36) nor agree on a winner - two different architectures finish first across the six blocks. So the pooled significance **is** the blocks that selected the finalists: nothing was replicated, and **no ranking is carried into the search**. The search's direction comes from §2c's designed factorial, which never rested on one. |
| **The record that could not be produced** | `arch_zoo_analyze.py` raised `TypeError: string indices must be integers` on **v1's own ledgers**, so the committed `decision.md` could no longer be regenerated at all. `joint_structural_fit` emits `genes` as coefficient rows on its success path but as bare gene *names* on its rank-deficient path, and `structural_signal_by_length` indexes that slot as rows. Not a corner case: `common` is the set that ran at *every* length, which for a screening sweep is its promoted finalists (three of them), and standardised columns span only n-1 = 2 dimensions, so a fit over as many predictors as architectures is dependent by construction. `genes` now has one shape, the refusal names the constraint it hit, and the consumer reads coefficient rows only from an available fit - pinned by a synthetic saturated set *and* against v1's real ledgers. |
| **The verdict read the rule's presence as its result** | `verdict()` treated "the rule is in the outcome" as "the ordering was confirmed", so the pooled p - the very number the rule reads as the screening seeds - produced "the finalists having already been confirmed on seeds never used in screening" directly above a section saying no ranking is recommended. The headline, the action and the selection caveat's closing instruction now all follow the branch; the confirming branch's wording is unchanged, and that is pinned rather than assumed. |
| **The record is now audited both ways** | `verify_record.py` recomputes each claim from the **ledgers**, which is not the same thing as re-rendering the artifact the claim is read from - that gap is why the crash survived unnoticed. `scripts/check_record_freshness.py` closes it: it parses each arm's own `NOTES.md` runbook, *executes* it against that arm's ledgers in a scratch tree, and diffs the result. **8/8 arms reproduce byte-for-byte** - `decision.md` and `ranked.json` - from the command their own documentation gives. |
| **Five runbooks did not reproduce their records** | Every committed `decision.md` was correct; the documentation was not. `v2` documented `--experiment-id arch-zoo-100m-v1 --also arch-zoo-100m-v1-confirm`, ids from a plan that was never run under that name; `v1`, `v3-crossed` and `v4-head-paired` documented the pre-rule analysis their own confirmation superseded; `v2-horizon` and `v1-finalists-6seed` documented only their *parent's* pooled write and nothing about themselves. All corrected, and `ranked.json` refreshed to the current schema so `--strict-rank` is clean. |
| **The screen could name seeds it never ran** | `write_reports` passed the *plan's* seed union to `best_available_summary`, so a flat arm's record read `seeds: [42, 43, 45, 46, 47]` against three seeds of ledgers - contradicting its own `seeds_per_architecture`, and therefore the rendered decision, which lists the rows. `arch_zoo_analyze.py` derives the set from the results, so the two writers disagreed about what the same arm had been read over. Now derived from the rows, with the flat-plan case pinned end to end. |
| **Fast enough to run casually** | The check is one subprocess per arm, so the arms run concurrently, and a candidate whose seed set cannot match the record's is ordered last rather than costing a complete analysis. **~60 s serial to 8 s**, with the analysis itself untouched: the Monte-Carlo draws must stay bit-identical or the check could not check anything. |
| **Numbers** | **1110 tests pass**; `verify_record.py` **142/142**, up from 131 - 11 new measured claims pin the branch, both p-values, both floors, the condition results, the block-top count, and the absence of the over-claim in the written headline and action. |

## v1's own open item was opened as an arm, and the rule it pre-registered was tightened before the data (2026-09-19)

| | |
|---|---|
| **Why this arm exists** | `arch-zoo-100m-v1` closed on "NOT ESTABLISHED - this experiment cannot yet tell" and named its own fix in the same file: re-run the finalists on seeds never used in screening, which removes the selection coupling and raises power. Its exact test runs 3 blocks and floors at 1/36, and its own caveat says those 3 finalists were promoted *on the seeds it tests*, so it is mildly anti-conservative in the winner's favour. `ARCH_SEARCH_FINDINGS.md` §9 has it as an open item at ~1.3 GPU-h. This arm is that item and nothing else - no new architectures, no new length, no retuned knob. |
| **What it trains** | The three v1 finalists, genomes byte-identical to the record's, flat at 1200 steps on fresh seeds 45/46/47: 9 runs, 22.1M tokens. It is read *pooled* with v1 by `--also`, so the exact test moves from 3 blocks to 6. |
| **Where, and why not Kaggle** | The local GTX 1070 is idle, so it runs beside the Kaggle horizon arm rather than competing with it. Kaggle is ~2.9x faster per token (13,930 vs 4,873 tok/s) but that advantage **is** the fp16 path, and fp16 is ruled out here: the gaps to be resolved are 0.017-0.023 nats while the measured fp16 differential reaches 0.018, so the measured ordering would be partly a numerics artefact. On fp32 the T4 is ~1.25x the 1070, which does not pay for a publish, an import and a concurrency slot beside a running kernel. |
| **The rule, and the bug found before the data** | `FINALIST_ORDER_RULE` reads the same exact test twice - all six blocks (floor 1/7776; the enumeration is still *exact* at 6^6 = 46,656, under the 1M cap) and the three screening-free blocks alone (floor 1/36) - and gates the positive branch on every block naming the same architecture first. Planting the branches to rehearse them showed the first draft unsound rather than merely loose: `blocked_permutation_test` is **omnibus**, so a design whose screening blocks rank A first while its fresh blocks rank B first is significant on the pool while having replicated nothing, and it passed that draft. Tightened in the stricter direction, with the reason written into the pre-registration, and the case pinned as a regression test. |
| **The pool was impossible until the guard was fixed** | v1's rows predate `run_config.mixed_precision` (added with autocast in `07fc36a`) and carry no field at all; the new arm records `off`. The precision guard created in the previous section read that absence as "unknown" and refused the pool - which would have blocked this arm's entire purpose. A row without the field is fp32 *by construction* (no row predating it could be fp16 or bf16), so absence now reads as `off` while any recorded mismatch is refused exactly as before; an fp16 arm pooled with an fp32 one is still refused, and that is pinned by a test. |
| **v1 stopped asking for what was already bought** | Both of `verdict()`'s closing instructions tell the reader to re-run the finalists on seeds never used in screening - which is the arm that produced the outcome. Once the rule is in the outcome the action reports what the re-run bought and points at the rule section instead. The unconfirmed wording is byte-identical, and that is checked rather than assumed: regenerating the committed `decision.md` for v1 and for the head-paired confirmation arm reproduces both files exactly. |
| **Integrity as a standing check, not a note-time claim** | Six new claims make the arm's design integrity verifiable at any point mid-run: the finalists are read off v1's own 1200-step ledgers rather than restated, the candidate genomes are compared byte-for-byte, every row sits on a screening-free seed, the plan is flat at the declared length, the validation stream is the record's holdout rather than a re-split one, and both arms' rows read as a single precision. **125 -> 131/131** claims. |
| **The run is watched** | `scripts/run_finalist_analysis_when_sweep_exits.sh` counts rows *inside* each fresh-seed ledger - the sweep writes one row per completed run, so a lost run leaves a file that exists and looks whole - and requires v1's three screening ledgers as well, since a pool partner missing turns a six-block design into a three-block one. It logs the analysis rather than writing it: `--write` supersedes v1's committed `decision.md`, which is a deliberate step once the branch has been read. Exercised against the partial arm as it stands, where it names every missing ledger and exits 2 without analysing. |
| **Pending** | The branch itself. The verdict lands in the arm's `analysis_watcher.log`; then `--write` into v1, the numeric rows in `verify_record.py`, and `ARCH_SEARCH_FINDINGS.md` §9. |

## The horizon arm trained none of its horizon, and the admission bar that refused it was the defect (2026-09-19)

| | |
|---|---|
| **What happened** | `act-t4-archzoo` ran 0.50 h and exited `COMPLETE`, and the driver reported `archzoo complete=True`. The 1200-step fp16 anchor trained; **all sixteen 12,000-step runs failed before a step** with `tokens.npy holds 40043724 tokens, need 51153024`. The per-seed logs holding the sixteen exceptions are scratch by the repo's own ignore rule (only `NOTES.md`, the decision/ranked/search-cost/matching/distance/zoo-definition files and the `*_results.jsonl` ledgers are durable), and the kernel's first version still carries the raw output on Kaggle; the failure is recorded in the arm's `NOTES.md` Status. |
| **Root cause** | The harness admission check in `architecture_eval._prepare_streams` asked for **2x** the schedule plus the 2M validation holdout, with no stated reason. The design is feasible and the arm's own cost section said so: 12,000 steps consume 24.6M tokens and use 24,000 of the stream's 37,115 windows, so no window repeats. `required_train_tokens` now states the bar as the schedule itself (24.6M here) instead of 49.2M, plus `required_holdout_tokens`/`required_stream_tokens` so a caller can check a plan against the file on disk. A one-epoch stream is admitted; one token short is still refused. |
| **Two reporting defects** | `mode_archzoo` derived `complete` from `not stopped_at_deadline` and the exit codes, but the screen catches a per-candidate exception and exits 0 even when every run in the stage failed. `complete` is now read off the requested (length, seed) grid against the ledgers, with the absent runs named; `kaggle_queue.py` warns when a re-read arm is incomplete. The screen also had no data-feasibility check, so an unservable plan was discovered inside a kernel rather than at launch; it now measures each length against the stream header **at launch** and refuses before any training, and `--dry-run` reports the same numbers without spending the hours. |
| **The landing is rehearsed, not just guarded** | The watcher's own command was run against a simulation of the landed arm - the real fp16 anchor rows plus synthesized 12,000-step rows, 32 in all, in the gate's four files - and the record it wrote was then put through `check_record_freshness.py --strict-rank`, which reproduced it **byte-for-byte** (`decision.md` and `ranked.json`) from the arm's own NOTES runbook. That closes a risk the audit tooling itself created: the checker symlinks only `*_seed*_results.jsonl` and `zoo_definition.json`, while the pre-registered `--offset-against` reads a *second* arm's ledgers, so the offset had to resolve out of that subset alone - it does. The horizon rule also read at its **declared 12,000 steps** rather than the 1200-step anchor sitting in the same directory, which is the failure the arm's first attempt would have produced had it been analysed while partial. |
| **The launch-time refusal is pinned, not just written** | The header check is only a fact until `main` acts on it, and that wiring is what the first attempt lacked - so it is tested through `main` with a 7,190-token stream: the run aborts naming `flat 12000 needs 26,600,000 tokens, stream holds 7,190`, tells the reader to lengthen the stream or shorten the plan, and leaves **no ledger** behind for anything downstream to read as a run. The existing coverage tested `stream_feasibility` returning short and the impossible-budget refusal separately; the branch that spends the quota was the one unwired. |
| **Not invalidated** | The correction relaxes an admission test, so it cannot alter a run that already passed it; `verify_record.py` reproduces **125/125** and the suite is 1068 passed. The design, candidate set, seeds, lengths and precision are unchanged, and the arm's `NOTES.md` Status section records the failure and the correction. |
| **What the attempt produced** | The reusable fp16 anchor at 1200 steps, both seeds, batch-identical to the record (same stream and holdout). Pooled: depth **+0.0742** vs fp32 **+0.0842**, hidden **-0.1918** vs **-0.1928** - a precision offset of -0.010 and +0.001 nats against effects of 0.08-0.19, so fp16 is measured rather than assumed. |
| **Re-run** | The design re-launched unchanged (~4 GPU-h at the measured 17.2k tok/s, against the 15 GPU-h horizon cap). |
| **Found by following the failure further** | `HORIZON_STEPS` now **declares** the rule's length (12,000) instead of inferring "the longest length that has rows": with only the anchor on disk, that inference resolved to 1200 and would have labelled the anchor "shallow wins persists at this horizon" from an arm that never trained a horizon step. `--also` now **refuses to pool ledgers recorded in different precisions** (the fp16 arm against the fp32 record would have produced an effect size that is part precision offset). The horizon rule's `decision.md` rendering is pinned by a test, branch and refusal both. |
| **The last unread secondary, mechanized** | The arm pre-registers five secondaries; four already had readers (width and interaction from the crossed contrast; ordering and decay from `rank_stability` and `spread_by_length` over the arm's own two lengths, inside one precision). The precision offset had none and the number had been computed once by hand, so `precision_offset` now joins two experiments' ledgers at one length, seed-matched and per-candidate, reached with `--offset-against` and rendered as a table. It refuses an unpaired length or seed set and names architectures it could not join. Rehearsed against the real fp32 ledgers with a planted +0.02 gap, recovered to the digit. |
| **The post-run read is a script, not a step** | `scripts/run_horizon_analysis_when_kernel_exits.sh` waits for the driver, **refuses to write** unless all 32 rows exist (8 candidates x {1200, 12000} x seeds 42/43), and only then runs the pre-registered analysis. The gate was exercised against the first attempt's real partial state, where it flags the two missing 12,000-step ledgers. |

## The head axis closed, the horizon arm opened on Kaggle, and the tree slimmed (2026-09-19)

| | |
|---|---|
| **Head-paired confirmation read** | Six-seed pooled analysis lands on `HEAD_PAIRING_RULE`'s **head effect is real at wide width**: −0.0553 nats at H=896 (12/12 pairs, at the 2/2¹² floor), +0.0084 at H=512 (p 0.37), interaction +0.0636 at its 2/2⁶ floor with all six seed contrasts agreeing. `verify_record.py` recomputes the six-seed rows and the branch (125/125 claims); `ARCH_SEARCH_FINDINGS.md` §2c/§9/§10/§11 and the README are updated. Phase closed at 23.632 GPU-h against the 10–20 cap, overrun pre-declared. |
| **Literature check, zero GPU** | Width winning is consistent with Levine et al. 2020; heads-only-at-wide with the low-rank head bottleneck (Bhojanapalli et al. 2020); MoE sparsity showing nothing at 2.4M tokens with routed-model scaling work (Clark 2022, Krajewski 2024). **Shallow beating deep is the one finding prior work predicts should reverse with training** (Tay et al. 2021, Petty et al. 2023). |
| **Horizon arm, pre-registered** | `arch-zoo-100m-v3-crossed-horizon10x`: the crossed depth×width factorial (byte-identical candidates, verified by dry run) at **12,000 steps** plus a 1200-step same-precision anchor, fp16, seeds 42/43, on **Kaggle T4×2** (one sweep per card). Numeric rule in `HORIZON_RULE`/`--horizon-rule`, every branch exercised on planted and real ledgers before launch; opens a horizon phase with its own 15 GPU-h cap. |
| **Kaggle plumbing** | `kaggle_t4.py` gains an `archzoo` mode (two seeds, one per device, wall-clock deadline, ledgers into the kernel output, exit 0 with `archzoo-status.json`), `kaggle_queue.py --queue archzoo` pushes and imports the ledgers into `artifacts/`, `arch_zoo_screen.py --mixed-precision` records precision per row. Token stream published as `act-lab-tinystories-tokens`. The CLI's OAuth refresh gap is worked around in `_run`. |
| **Suite and audit made fast** | `pytest` 243 s → **35 s** (pytest-xdist, one torch thread per worker via `tests/conftest.py`, three `/tmp` races moved to `tmp_path`, test-only Monte-Carlo sample counts). `verify_record.py` 63 s → **18 s** full / 27 → 14 s quick: the search-space enumeration is memoised (a 200-seed budget scan re-enumerated 19,320 cells per seed) and the audit's checks and power runs execute in a process pool. **No statistical RNG path changed**; every recorded Monte-Carlo p reproduces. |
| **Removed** | The TypeScript console scaffold (112 tracked files: routes, auth, pglite, PWA plugins, vite/eslint/prettier, `public/`, `server/`, `migrations/`, `startup.sh`), `node_modules`, screenshots. Untouched since the initial commit and representing none of the results. |

## Section 5.46: the declared launch path could not specify the rebuild treatment (2026-09-16)

Found by auditing the launch command against the adopted sections. Section 5.16 ("new work
should set `preserve_router_on_rebuild=True`") and section 5.38.3 ("a real-scale protocol
should be specified with the treatment on") both say what the declared run should carry,
and `cli._run_pair_session` had **no way to express it**: no CLI flag, no override passed
into `declared_session_config`, and no test or paragraph pinning the declared run's rebuild
policy. The protocol was registered (§5.5, 2026-09-14) before the treatment existed
(§5.16-5.28), so an unflagged launch always ran the historical rebuild.

| | |
|---|---|
| Measured consequence (§5.39 stored rows) | churn harm **0.16449** untreated vs **0.11279** repaired - the treatment is worth **1.46x** on the quantity §5.44's clause reads |
| Compared with the effect it gates | 0.0517 nats, above the clean-arm effect (0.04042) - so the missing flag can decide Gate 4 |
| Undone now (additive, no default moved) | `--preserve-router-on-rebuild`, `--new-expert {random,clone}`, threaded into `declared_session_config` for both the declared and `--profile` shapes; `clone` without preservation refused with a one-line `SystemExit`; 3 tests pin the default, the treatment, and the refusal |
| Fell out of the audit | the published seed summary already records all 35 config fields, so **which rebuild policy a run used is checkable from its own artifacts** |
| Deliberately not done | the runbook's command is unchanged - registering the declared protocol on the treatment is a sign-off decision. `T4_RUNBOOK.md` carries a **DECISION PENDING** callout at the command so T4 cannot start without noticing |

**Closed in the same section:** the verdict never compared a published run's recorded
config against the registered protocol, so a run whose configuration differed from the
registration would have been judged as though it were the registration. `v0_verdict.py`
now has `REGISTERED_RUN_CONFIG` plus a `config_conformance` block (per field: `matches` /
`differs` / `not_recorded`, with `conforms` and a reading string). It is **descriptive
only** - no label, exit code or authorization moves - and the rehearsal pins that by
building the untreated-rebuild and pre-5.5 `prefix` runs and checking they keep their
authorizations while being flagged. A pre-field artifact reads as `not_recorded`, never as
a mismatch. The two committed real-path diagnostics correctly read as non-conforming on
nine shape/budget fields (they are depth-8, 600-step runs), with their labels unchanged.

## Section 5.47: the declared run is registered with the rebuild treatment on (2026-09-16)

Delegated decision, recorded with reasoning. The declared launch command now passes
`--preserve-router-on-rebuild --new-expert clone`, and so do the §2b control and §2c null
commands - a different rebuild policy in a paired queue would make the comparison a package
of two changes instead of the decision rule alone.

| | |
|---|---|
| Grounds | (1) the untreated rebuild is an unintended router discard kept as the default only to preserve the committed record of §5.16-5.23; (2) §5.38's branch (c), which would have forbidden this, was superseded by its own correctly-sized block (§5.38.1 passed the clean arm at n=12, perm p 0.0239, clean-comparison 12/12); (3) the gap, 1.46x on churn cost, is larger than the effect claimed |
| Not claimed | the treatment does not decide Gate 4: effect is small and right-skewed (8/12 against a 10-of-12 win bar), and §5.39 shows 69% of the churn harm remains, so the §5.14 do-nothing clause stays the binding protection |
| What changes | two config fields, applied to **every** arm; no parameter count, no probe FLOPs, so the cost prediction and band are untouched |
| What does not | no criterion moves (5%, band, harm guard, §5.5 pool/quality clauses, §5.14 null clause, completeness); seeds, shape, tokens, events, dtype and probe regime all unchanged |
| Reversible | drop the two flags before session 1 and the conformance block will record the run as untreated rather than letting it pass as treated |

Record: `RESEARCH_REPORT.md` §5.47.

## Section 5.44: the quality-premise clause was shipped in one direction only (2026-09-16)

Section 5.5 change 3 was **ADOPTED** (2026-09-14) and reads "if the pool-matched arm's
quality is indistinguishable from the adaptive arm's ... withhold the Gate 4
recommendation". Only the direction where **both** arms clear the bar shipped; the
direction where **neither** does was implemented nowhere, so it authorized Gate 4.

Nothing else in the gate looks at the adaptive arm's *quality*: the harm guard is a floor
(rejects only at >= 10 losses of 12, so 3 wins clears it), and the §5.14 clause compares
churn against do-nothing rather than adaptive against the loss-blind arm.

| adaptive vs churn | loss-blind | before | after |
|---|---|---|---|
| 6/12 (tie) | 6/12 | `PASS`, exit 0, **authorized** | `PASS_COST_ONLY_NO_DEMONSTRATED_EDGE`, exit 2 |
| 9/12 | 3/12 | `PASS`, **authorized** | withheld |
| **3/12** | 0/12 | `PASS`, **authorized** | withheld |
| 2/12 | 0/12 | harm guard rejects | unchanged |
| 10/12 | 3/12 | `PASS`, authorized | unchanged |

The tie row is exactly the state the dev-fidelity evidence predicts for the real path
(§5.13, §5.39), and the 3/12 row is the harm guard's own boundary - one win above
rejection. **The amendment** completes the clause with the same `POOL_CONTROL_WIN_LIMIT`
(no new threshold) and is strictly tightening, so no spent block is reinterpreted -
verified by re-computing both committed real-path verdicts under the pre- and
post-amendment code (identical labels both sides). Gate 4 now requires the adaptive arm to
clear the bar **and** the loss-blind arm not to; the record carries a `quality_premise`
block so the reason is stored, not inferred from the label. The proxy's boundary (it also
withholds at 9-vs-3, where adaptive is clearly better) is recorded as the reading to
contest, in `registered_text_amendments[0].boundary_reading`.

Rehearsal is now **16 variants / 9 checks**, all matching; `tests/test_v0_verdict.py`
rows at the harm-guard boundary were rewritten to pin the interaction rather than
relaxed. Teeth: removing the withholding branch fails two unit rows and makes `--check`
exit 1 with three checks false; restored, 52/52 pass. Record: `RESEARCH_REPORT.md` §5.44.
Also corrected: §5.41's verification sentence misstated `local-proof`'s re-run label.

## Section 5.45: the suite's 140 s had two measurable causes, one in production code (2026-09-16)

Measured, not guessed: `--durations=35` put ~90 s of 140 s in 25 tests and 41 s of that in
four.

| | before | after |
|---|---|---|
| one CLI error-path test paying production publish backoff (`time.sleep`, injectable and already injected by the remote tests) | 15.0 s | ~0 |
| `act/evolution/study.py::_gate_min_wins` re-summing `math.comb` per step (O(n^2), ~1.06M calls for a 1536-seed block) | 16.15 s | 0.4 ms |
| full suite, `OMP_NUM_THREADS=1` (comparable with the ~140 s baseline; unpinned it is 88.3 s) | ~140 s | **98.5 s** |

The two fixes are 31.2 s of the ~42 s reduction; the rest of the suite is real work (slowest entries are fp16 session round-trips at 9.1 s and 4.4 s).

The binomial tail is now accumulated once with the exact recurrence. Because the returned
integer is a *pre-registered* analysis threshold, equality was verified rather than
assumed: **bit-identical on 199 checks** across 12 block sizes (incl. 1536), 8 alphas and
the tail-sum identity. Record: `RESEARCH_REPORT.md` §5.45.

## Section 5.43: the declared Gate-4 verdict path is rehearsed before the spend (2026-09-16)

*(Superseded in two details by §5.44: the record now holds 16 variants, and two variants
authorize - `all_clear` and the distinguishable case.)*

`scripts/verdict_rehearsal.py` builds synthetic **declared-scale** fixtures
(285,336,576 / 134,341,632, 24,576,000 tokens per arm, all three queues) and applies the
real `compute_verdict` to 14 variants - every withholding clause in isolation, both
(16 as of §5.44)
absent-queue and both short-queue cases, and the three precedence combinations. Record
`artifacts/study-summaries/gate4-verdict-rehearsal.json`, pinned by
`tests/test_gate4_rehearsal.py` (9 tests; 5 tampers each fail, restored passes).

**Why it was worth doing before the spend:** sections 5.41 and 5.42 both found defects in
this machinery by reading it, and the verdict is otherwise applied exactly once, at the end
of a multi-hour Kaggle run - so a wiring error would have surfaced as a confidently wrong
answer with nothing to compare it against.

| | settled |
|---|---|
| **Label precedence** | Verified, not just documented: cost FAIL > harm guard > pool-matched control > null > missing control > missing null > band miss. The outranked clauses' flags are recorded, so the labels are precedence and not absence. |
| **`primary_queue_short`** | An incomplete primary queue gives an ordinary `PASS`, exit 0, and `gate_4_authorized: False` - the completeness requirement lives in the flag, not the label, so a label-only reader misreads it. Now pinned. |
| **5.41 + 5.42 hold at declared scale** | Every fixture reports `run_is_at_declared_scale` with both band readings agreeing, and both short-queue variants return `..._MISSING`. |
| **Authorizing variants are named** | `all_clear`, exit 0 (two as of §5.44, the second being the distinguishable-quality case). Cost failures exit 1 and nothing else does. |

**It caught my own unit error.** A first pass expressed both cost variants as "6x the
prediction"; the prediction is itself a percentage (0.0942%), so 6x is 0.565% - inside the
5% criterion. The machinery returned `PASS_WITH_QUALIFICATION` and `HARM_GUARD_REJECTS`
where the rehearsal had written `FAIL`, and its own check reported the drift; crossing 5%
takes a multiple above 53.1. A check now ties each variant's cost to its knob - the same
family as section 5.41, committed one section after writing it up.

**Not evidence.** No session ran and every loss is fabricated; the record carries
`synthetic: true` and says it validates the decision path rather than the hypothesis, both
pinned. `--check` exits non-zero on drift, and `T4_RUNBOOK.md` step 0 runs it before
session 1.

## Section 5.42: a short pool-matched control made its own clause unfireable (2026-09-16)

Found by comparing two sibling clause evaluators that should behave alike. `_null_block`
states in its docstring that `evaluated` "requires the queue to exist AND be complete on
the declared block; anything less withholds Gate 4 rather than reading as a null result".
`_control_block` returned `evaluated: True` with a `seeds_missing` list and no guard.

**Why that is exploitable.** The clause is `wins >= POOL_CONTROL_WIN_LIMIT` (10), so on
fewer than ten pairs it cannot fire however the loss-blind arm performs, and
`compute_verdict` reads that as "clause evaluated, did not fire"
(`control["evaluated"] and control["control_passes_gate"]`). `gate_4_authorized`
consults only the **primary** queue's missingness.

| control pairs | loss-blind arm | clause | verdict | `gate_4_authorized` | exit |
|---|---|---|---|---|---|
| 4 of 12 | wins 4/4 | **unfireable** (4 < 10) | `PASS` | **True** | 0 |

So the clause whose purpose is to stop a pool/location effect from authorizing Gate 4 was
defeated by **not finishing the control queue** - the state a deadline-stopped session
produces - with nothing warning.

**Fix, mirroring `_null_block`.** `evaluated` now requires the control complete on the
declared block, recording the missing seeds and the reason; the existing
`PASS_COST_ONLY_CONTROL_MISSING` label (exit 2, Gate 4 withheld) picks it up, so no new
label and no other registered behaviour moves. A complete control behaves exactly as
registered, including 9-of-12 staying authorized. The same fixture now reads
`PASS_COST_ONLY_CONTROL_MISSING`, `gate_4_authorized: False`, exit 2.

**Boundary pinned:** a control at 11 of 12 whose loss-blind arm loses every pair is
withheld too. Two tests, teeth verified by reverting the guard (both fail; restore passes
38/38). Recorded as report section 5.42.

**Fourth instance of one shape.** Section 5.33's `forced_operation` no-op, section 5.37's
unclassified field, section 5.41's band scale, and this: **a check whose result can be
decided by something other than the quantity it claims to measure.** All four found by
reading; none by running a study.

## Section 5.41: the Gate-4 verdict's tolerance band could only be passed at declared scale (2026-09-16)

Found by reading the two committed real-path verdicts. Both record `in_tolerance_band:
false`, and the registered rules make a band miss a qualification that "forces
derivation before Gate 4" - so for two sections they have read as a miss.

**They could not have done anything else.** `compute_verdict` takes the measured cost
from the run's booked FLOPs but derives the band from `expected_probe_cost_percent`,
which is pinned to the **declared** V0 shape (285,336,576 / 134,341,632, 24,576,000
tokens/arm). The section 5.3 block ran 600 steps on a ~12.2M-parameter model -
**10.6983x** from the declared scale - so the check was measuring the distance between
the run and the declaration, not between the measurement and the identity.

**The fix is the registration's own rationale applied.** Section 5.1 already justifies
the band's slack by saying every architectural deviation "enters the identity as a
bounded multiplicative factor on `N_active` or `D`", so the factor is applied
explicitly from the run's own numbers:
`C(run) = C_declared x (T_declared / T_measured) x (N_run / N_declared)`, which reduces
to the registered comparison exactly at declared scale.

| block | scale, measured/declared | measured/predicted **at the run's own scale** | registered `in_tolerance_band` |
|---|---|---|---|
| `real-proof` (real TinyStories, 9000-9011) | 10.6983x | **1.000286** | false |
| `local-proof` (synthetic tokens, 9000-9011) | 10.6631x | **1.000278** | false |

**The only real-path execution of the compute-efficiency protocol confirms its own cost
identity to 3 parts in 10,000** (with a +0.03% systematic overshoot on both blocks),
while both artifacts report a miss.

**Additive only.** `in_tolerance_band`, `predicted_cost_percent`, `tolerance_band_percent`
and the verdict string are unchanged - verified by re-running the verdict over both
committed blocks (`PASS_COST_ONLY_HARM_GUARD_REJECTS` in each, as committed). The new
`run_scale` block is labelled `descriptive_only`, no threshold or rule moves, and the
report does not spend the reading as a criterion. What it buys is legibility on the run
that decides the gate: a declared queue that deadline-stops, resumes, or is pre-flighted
at reduced fidelity now says which reading applies. The declared numbers were also
inlined inside `expected_probe_cost_percent`; they are now named constants
(`DECLARED_TOTAL_PARAMETERS`, `DECLARED_ACTIVE_PARAMETERS`, `PROBE_TOKENS_PER_BATCH`,
`DECLARED_EVENTS`, `DECLARED_CANDIDATES`) so the re-derivation cannot drift from the
prediction - unchanged to the last digit, `0.0941636252059042`.

**Third instance of a pattern.** Same family as section 5.33's `forced_operation` no-op
and section 5.37's unclassified field: **a check that cannot be passed on the input it is
applied to.** This one sat in the artifact that decides Gate 4. Pinned by three tests in
`tests/test_v0_verdict.py` (exact reduction at declared scale, the additive contract on a
reduced-scale fixture, and the committed section 5.3 block's own numbers).

## Section 5.40: registered, deliberately not run (2026-09-16)

The deployed-configuration adaptive-vs-null block was registered and committed
(`e7e0b0f`) and then stopped after 5 of 24 sessions (seeds 15240-15242). Two measured
reasons, not a change of mind: the registered shape costs ~200 s/seed (135 s pair + 67 s
null), i.e. ~40 min of GPU for the block; and a long run **cannot** outlive its launching
shell here - a backgrounded attempt died with the shell that started it - so it can only
be bought in foreground chunks, which is time this project has twice been told is not
worth spending. No statistics are reported from the partial rows (scratch only, no
summary, no verdict). The registration stands, its prediction is sourced from the
disclosed post-hoc diagnostic (reproducible from committed rows by `--summarize`), and
the runner is resumable for whenever the GPU time is worth it.

## Section 5.39: churn harm is a property of churn, not of the rebuild (2026-09-16)

18 paired seeds (the section 5.38/5.38.1 queues), dev fidelity, real TinyStories stream: 18 new
**null-arm** sessions (~20 min GPU, 67 s each) joined to the stored churn rows. Summary
`real-path-churn-harm.json`, pinned in `tests/test_churn_harm_summary.py`. Deltas are `null - churn`,
so **negative is the harmful direction** (see the sign correction below).

| point | mean final loss |
|---|---|
| `null` (zero mutations) | **3.53398** |
| `random_base` (historical rebuild) | 3.69847 |
| `random_clone` (repaired rebuild) | 3.64678 |

| primary | mean delta | wins | permutation p | verdict |
|---|---|---|---|---|
| `null - random_base` | **-0.16449** | **18/18** | 5.00e-05 | harm established |
| `null - random_clone` | **-0.11279** | **18/18** | 5.00e-05 | harm established |
| replication `base - clone` (not a primary) | +0.05170 | 12/18 | 0.00475 | matches 5.38.2 |

**Branch (b): harm survives the repair.** Doing nothing beat churning on **every one of 18 seeds
under both rebuilds**, and the repair recovers 31.4% of the cost without removing it (0.16449 ->
0.11279). The registered red-flag direction did not occur.

**This falsifies section 5.38.3's hope** that a treatment-specified declared run would not inherit
section 5.7's harm: it inherits 69% of it. And it re-reads section 5.38.1 - the repair is 31% of a
**net loss**, not the creation of a net gain.

**Section 5.14's do-nothing clause is now the binding bar, with a measured number.** Churn costs
0.11279 nats with the repair in place, **2.2x** the +0.0517 treatment effect, so an adaptive arm that
merely matches random churn is 0.11279 nats worse than doing nothing. The declared run's null queue
is the only thing in the protocol that detects that; it is load-bearing, not insurance.

**It also moves the named next direction off the probe.** Section 5.3 scoped the real-text failure to
the probe's candidate ordering; this block shows a large part of what any ranking is fighting is that
changing the architecture at all costs ~0.11 nats at this fidelity. Two licensed follow-ups, neither
of them another probe-horizon variant: make mutation net-positive (5.26-5.28 moved this term by a
third; 5.27 put 3/4 of the gain in the cloned body, so the residual is plausibly the pruned expert and
the layers a prune re-draws), or **mutate less** (5.7's degenerate gate and 5.11's theta sweep both
showed a controller that rejects most draws beats one that spends the budget).

**Sign correction, and both errors were mine (report section 5.39.1).** The registration wrote
"positive means churn is harmful" beside the arithmetic `null - churn`, which is the reverse of its own
wording. On the inverted predicate the branch came back `c` ("not harmful") while 18/18 seeds agreed
the other way and the sign test returned p = 1.0 - the block's own checks refused to let a branch that
contradicts every seed through. The second error was in the replication check, which compared
`clone - base` against a summary whose stored delta is `base - clone` and so reported `False` on
nothing but a sign. Both fixed, neither changed a measured loss, and the registered direction is left
visible in the report rather than overwritten.

**Accounting detail, measured not asserted:** the null arm records a no-op at all **four** cadence
slots (100/200/300/400) while `mutation_budget=3` caps the churn arms at three applied mutations. A
no-op has `search_flops` 0 and leaves the architecture hash unchanged, so it cannot move the contrast;
it is recorded because "budget-matched" is a claim.

**Length caveat:** dev fidelity (depth 6, 14.9M total / 7.6M active) where the declared config is
depth 12 at 285M, on two queues 5.38/5.38.1 already spent. It decides a branch and sets the bar; the
declared run is what measures the bar.

## Section 5.38.1: the treatment's clean effect is real, small and right-skewed (2026-09-16)

Seeds 15220-15231, 24 sessions, ~27 min on the dev GPU (the section 5.37.1 checkpointing fix paid
off here: ~66 s/session against ~90 s). Summary `real-path-clone-dev-n12.json`, pooled secondary
`real-path-clone-dev-pooled.json`, both pinned in `tests/test_real_path_block_summary.py`.

| | value |
|---|---|
| `random` arm, n=12, mean `base - clone` | **+0.04042** nats (se 0.0192, t +2.10) |
| sign-flip permutation p (declared statistic) | **0.02390 -> PRIMARY PASSES** |
| exact sign test | 8/12 against a bar of 10, p 0.194 -> fails |
| the two statistics disagree | yes |
| clean-comparison guarantee | identical event sequences **12/12**, grows 21 -> 21 |
| pooled with 5.38 (post-hoc secondary, n=18) | mean +0.05170, mean-test p **0.00375**, wins 12/18 (bar 13) |

**Honest reading: the mean improves and the per-seed win rate does not.** The effect is ~0.05 nats,
about 1.4% of the ~3.6 nat dev loss, and right-skewed - seed 15224 alone contributes +0.1883, and
without it the n=12 mean falls to +0.0270. That is section 5.31.1's signature: a positive expected
effect carried by a tail, not a reliable per-seed win. Both statements answer different questions,
and the write-up keeps them apart.

**This resolves section 5.38's branch (c) instead of repeating it.** The clean arm was failing its
test because n=6, sized from the pilot's optimistic sd, could not resolve an effect this size - not
because the mechanism fails to transfer. Branch (c) is superseded for interpretation; its numbers
stand as run. The probe arm's larger effect (+0.1316, 6/6) is still a package of rebuild plus
reselection, which this design does not apportion.

**Settled:** preserve-router plus clone improves expected final loss on the declared path at dev
fidelity, on the arm that cannot be affected by probe ranking, under a pre-registered test with the
clean-comparison guarantee holding 12/12 - so section 5.28's pipeline change now has registered
real-path support rather than only toy evidence. **Carried forward as calibration:** the effect is
small, so the real-scale protocol should be specified with the treatment on without being sold an
effect size on the strength of this run. **Still open:** the rebuild/reselection split, and Gate 4
behind the compute-efficiency measurement.

**Runner note.** `scripts/real_path_block.py` now serves both blocks by profile
(`--profile 5.38 | 5.38.1`, plus `--pool`), with runs keyed by profile so a seed cannot be reused
across blocks by accident. Its t-column is descriptive only: the project's declared statistics are
the exact sign test and the permutation test, no t p-value is recorded, and the script no longer
needs scipy (which is not installed here) to say so.

## A free 1.32x on every dev cycle, straight out of the liveness contract (2026-09-16)

Section 5.37's contract established that `activation_checkpointing` is trajectory-inert, and the dev
config defaulted it to **on** - correct for the declared 285M target (section 5.2's T4 fit arithmetic
assumes checkpointed activations) and pure recompute cost at the dev shape, where memory is not a
constraint. Measured on the dev GPU at the dev shape, 60 steps on the real stream, one arm:

| setting | wall | throughput | loss |
|---|---|---|---|
| on (the dev default until now) | 13.3 s | 20,145 tok/s | 5.721975 |
| **off** | **10.1 s** | **25,760 tok/s** | **5.721975 (bit-identical)** |

The off number matches the **documented** dev throughput (25,525 tok/s), so the handbook and the
configuration had been describing different runs, and every dev cycle - including all 12 sessions of
section 5.38 - paid about 30% more than it should have.

Fixed: the field default is now `False`, and `declared_session_config()` sets `True` **explicitly** so
editing a dev default can no longer move the declared target. Both ends pinned in
`tests/test_v0_session.py`. Section 5.38's results are unaffected (the flag is inert - which is why it
was safe to act on); only its wall time was, and 5.38.1's 24 sessions now cost ~27 min instead of ~36.
The suite also got faster (155.6 s -> 138.5 s). Recorded as report section 5.37.1.

## Section 5.38: the real-path block landed on branch (c), the red-flag branch (2026-09-16)

Six fresh seeds (15210-15215), 12 sessions, ~33 min on the development GPU, base vs clone on the dev
shape and the real TinyStories stream. Runner `scripts/real_path_block.py` (resumable and
budget-capped), summary `artifacts/study-summaries/real-path-clone-dev.json`, pinned by
`tests/test_real_path_block_summary.py`.

| primary | mean `base - clone` | t | wins | perm p | verdict |
|---|---|---|---|---|---|
| `random` arm (clean) | +0.07424 | +2.06 | 4/6 | 0.0624 | **FAIL** at `alpha/2` |
| `adaptive` arm | +0.13159 | +4.92 | 6/6 | 0.0156 | **PASS** |

**Branch (c) was pre-registered as a red flag rather than a result** - a probe-driven arm improving
decisively while the clean seeded arm does not - so this is recorded as **unresolved**, not as a
transfer. The diagnostic is why: the two settings selected the **same** mutations on the random arm on
**6/6** seeds, and **different** ones on the adaptive arm on **6/6**. The adaptive arm's gain is a
package (rebuild + reselection), and only the random arm isolates the rebuild. The reselection goes in
the predicted direction - preservation makes a grow look better to the probe, so the probe arm grows
more (5 -> 6 of 18 events) while the random arm does not move (9 -> 9).

**Section 5.36 is corrected.** It reported that every one of its six arm losses improved; on six fresh
seeds **two of the six random-arm comparisons reverse** (15214, 15215), so that uniformity was a
three-seed artifact. Its numbers should not be quoted as an effect estimate.

**The failure is a sizing failure, not a null.** The pilot's sd (0.023) understated the true
dispersion (0.088, ~3.8x), so n=6 could only resolve effects above ~0.09 nats on the clean arm. The
clean mean is in the helpful direction with 4/6 agreeing. Section 5.38.1 registers the follow-up at
**n=12 fresh seeds, `random` arm only** (~36 min) and states that no result from it may resize or
reinterpret 5.38's spent seeds.

**Two disclosures.** (1) The runner's pass predicate had an inverted sign (`mean < 0` inherited from
the 5.34 studies, where the delta was treatment minus reference); it was found after the block ran and
corrected, and changes only the `passes` boolean - the deltas, wins and both p-values are computed
before it, and the verdict was hand-checked against them. (2) The event-sequence check, the per-arm
event-change counts and the grow totals were added to the summary *after* the block as structural
diagnostics, and are reported in full rather than summarised.

**What this does not do:** it is dev fidelity, GPU (so not bit-reproducible in the section-6 sense,
which is declared in the summary), one substrate, and final held-out loss rather than capability per
FLOP - so Gate 4 is untouched by it either way.

## The config surface is now checked instead of promised (2026-09-16)

Sections 5.33 and 5.34 both found the same thing the hard way: a *declared* config field that
no code path reads under the policy the block was running, so registered compute went into a
mechanism that was never applied. Nothing was checking the config surface, so the only thing that
caught either case was reading the code afterwards.

`tests/test_config_liveness.py` now checks it. Every field of `V0SessionConfig` and
`ExperimentConfig` (35 each) is classified, and each classification is **measured** at a tiny shape:
flipping a field must move the recorded trajectory where the contract says it moves and must not
where it says it must not, with decisions, losses, validation, booked FLOPs, probe losses and the
stored record separated so that "bookkeeping only" is testable. An **unclassified field fails the
completeness guard** - the check whose absence let 5.34's forced-operation block run as a duplicate
of the forced-grow block.

Measured facts now pinned rather than assumed:

- `prune_expert_policy` and `forced_operation` are **bit-identical on the probe-guided arm running
alone**, and move the random arm's trajectory. Both are scoped facts about the probe path, not dead
fields, and wiring either into the probe path now fails the table.
- `probe_scoring` is inert when the applied rebuild already *is* the historical default - the
dependency 5.34 relied on - and live otherwise.
- `probe_landscape` changes the stored record with no change to any decision and no change to booked
FLOPs, exactly as documented.
- `validation_batches` is **not** reporting-only: it consumes the arm's RNG and moves the final loss
while leaving every selected mutation identical.
- Hazards: `optimizer_reset_steps` is masked when it collides with a mutation step on a *mutating*
arm (the rebuild already discards Adam; 5.23's `fixed`-arm control is unaffected);
`probe_force_events` is a modifier, inert without `probe_force_candidate`.
- A flip is only as good as its value: `adaptive_load_threshold` moves nothing at `0.5 -> 0.9` and
everything at `0.05 -> 0.95` (6/6 seeds), so a mid-range flip would have been misread as a dead knob.

The declared path came out clean - no field on `V0SessionConfig` reaches no decision - which is now a
standing assertion rather than an assumption. 20 tests, 9.6 s. Teeth checked by tampering: making
`argmax` behave as `argmin` fails the `probe_selection` row, and adding an unclassified field fails
the completeness guard. Recorded as report section 5.37.

## The declared path had the section 5.24 defect too (2026-09-16)

While porting section 5.34's fix, reading `rebuild_v0_moe_from_genome` showed it had the
**identical** router defect that section 5.24 fixed on the toy path: expert bodies copy by
id, but the router is `hidden_size x experts_in_layer`, whose shape moves with the expert
count, so every declared mutation re-drew that layer's learned routing function. Seven
pre-registered studies were designed around a treatment the real path never had - a process
failure worth more attention than the bug, since the two paths have separate factories and
only one got fixed.

Ported: `preserve_router` / `new_expert_initialization` on the declared factory (defaults
unchanged), the same two fields on `V0SessionConfig` (rejecting `clone` without
preservation), and **one helper** that every applied rebuild in `v0_session` goes through, so
the treatment cannot be wired into some paths and forgotten in others - the failure mode that
turned section 5.33's `prune_expert_policy` and section 5.34's `forced_operation` into no-ops.
Pinned by `tests/test_v0_router_rebuild.py`.

**Dev-scale signal on the real TinyStories stream** (3 seeds, `scripts/dev_cycle.py`, ~162 s
per cycle; NOT a registered measurement): all six arm losses improved under the treatment, and
the adaptive-minus-random gap improved on all three seeds (+0.0893 -> -0.0178, -0.0260 ->
-0.0323, +0.0519 -> +0.0498). The `random` arm is the cleaner signal - it mutates on a seeded
schedule and never probes, and it improved on all three seeds. It licenses a registered
real-path block, not a claim.

**Dev-loop facts measured, not assumed:** dev cycles do **not** parallelise (four with
`xargs -P 4` failed to finish in 590 s; sequentially they are 162 s each - the path is
launch-bound, so the GPU serialises and the cores are wasted on contention). And a *resumed*
arm prints a nonsense tokens/s (687 billion in one run, because it books the full token budget
against a near-zero wall) - read the loss columns on a resume, not the throughput column.

## Section 5.34: the probe was scoring candidates in the wrong rebuild (2026-09-16)

The probe ranks candidates by the loss of a model it builds, and it was building that
model with `rebuild_model_from_genome`'s **historical defaults** - a freshly drawn router
and a random new expert - while the event then applied the winner with the arm's own
settings (`preserve_router_on_rebuild=True`, `new_expert_initialization="clone"`). A new
opt-in field `probe_scoring="applied"` makes the probe score in the rebuild it applies.

**Result (branch (b), both depths, 384 seeds each):** the fused run improves decisively
(-0.04043 and -0.03478 nats, p 5e-05, sign test passing at both depths) and the delivered
artifact does not (-0.00825 p 0.13, +0.01047 p 0.86). Value over doing nothing roughly
doubles: -0.0467 -> -0.0871 at depth 2, -0.0566 -> -0.0913 at depth 3. The two rules chose
differently in 70.3% of events and the operation mix moves the way the mismatch predicts
(grows 220 -> 382), so the controller had been pruning mostly because the default scoring
made both candidates look worse than the models the pipeline would actually build.

**Two things this session cost that it should not have.** First, an honest correction:
sections 5.32/5.33's "the cost identity must hold exactly" is a claim about *selection*
rules and it is false for a *scoring* rule, which books the scored model's active-parameter
count. Measured mean 4.62%/3.12%, max 21.86%/14.50%, exactly zero on 83/77 seeds. Second,
the declared forced-operation mechanism block was **deleted, not reported**: on the probe
path `forced_operation` never binds (`adaptive_policy="probe_loss"` returns before the
branch that reads it), so a forced grow block and a forced prune block are the same runs -
which is exactly what the numbers said. The mechanism is answered from a partition of the
stored blocks instead, and both facts are pinned by `tests/test_probe_scoring.py`.

**Fast-iteration note.** The block sizes were amended *before* the run on the disclosed
12-seed pilot (mean -0.0749, sd 0.123): at 384 paired seeds the standard error is 0.0063, so
an effect four times smaller than the pilot's still clears `alpha/4` at over 99% power.
Declaring 1536 would have bought nothing measurable. All six block kinds together ran in
6m43s wall on the 20-core box (`xargs -P 18`, `OMP_NUM_THREADS=1` per process).

Updated: 2026-09-14

This document compares the current workspace against
`Adaptive Computational Topology - Complete Kaggle Implementation Plan.md`.
It is the implementation tracker for the narrowed first scientific milestone.
The narrative companion covering the six-study factor map (six gate-style
quality tests, one pass) is `RESEARCH_REPORT.md`; all quoted statistics there
are machine-checked against `artifacts/study-summaries/`.

## Training target: dev-fast by default (2026-09-15)

The pre-registered 285M-parameter section 5.1 target is no longer the default
model. It was the default because it was the *declared* model, and the effect
was that every development cycle was a protocol-scale run (285M parameters,
24.6M tokens per arm, an hour-plus per arm on the development GPU). The declared
target is unchanged and still what a declared run uses; it is now built
explicitly:

```python
V0SessionConfig()                 # fast dev shape: ~14.9M / 7.6M active
V0SessionConfig(declared=True)    # rejected unless it IS the declared config
declared_session_config()         # the pre-registered 285M target, validated
```

``DEV_V0_CONFIG`` (vocab 8k, context 512, hidden 320, depth 6, 8 experts of 2
active, expert width 320; ~14.9M total / 7.6M active) was chosen by measurement
on the GTX 1070, not by preference, and the measurement is the interesting part:
**at these sizes the MoE block is latency-bound, not FLOP-bound** (one kernel per
expert per pass), so the smallest candidate was the *slowest* - 10.0M parameters
ran at 3.6k tokens/s while 14.9M ran at 9.8k. Cycle time is governed by tokens
per run.

That latency-bound diagnosis was then acted on: the expert block now folds its
per-expert launches into batched matmuls (``_MoEFeedForward.forward``), keeping
the eager dispatch as ``eager_forward`` and pinning the two equal by test
(float64 agreement 1e-16). Measured on the dev shape, batch 8 x context 512:
51.8 ms/step eager vs 42.5 ms/step dense (+18%).

End to end through the real session path on the **real TinyStories stream**
(``scripts/dev_cycle.py``): each arm trains at **19.6k tokens/s** (0.21 s/step),
0.9 GiB peak, and a two-arm 400-step dev cycle (1.64M tokens/arm) takes
**2.8 minutes wall** instead of an hour.

```
PYTHONPATH=src python scripts/dev_cycle.py            # 400 steps, real text
PYTHONPATH=src python scripts/dev_cycle.py --steps 40 # quick smoke
# the two settings side by side (same seed, so neither resumes the other):
PYTHONPATH=src python scripts/dev_cycle.py --seed 15201 --tag base
PYTHONPATH=src python scripts/dev_cycle.py --seed 15201 --tag clone \
    --preserve-router --new-expert clone
```

Precision stays fp32 on the development card. Measured at the dev shape,
forward+backward+AdamW: fp32 25.5k tokens/s, bf16 20.6k (emulated below compute
capability 8.0), fp16 10.5k. The fp16 penalty is the arithmetic rate, not the
guard scaler (disabling it recovers only ~1.5%), so bf16 - which needs no
scaler - does not recover it. On the declared Turing-class target fp16 is
native and correct; bf16 remains emulated there too.

The dev budget is not a protocol measurement and must not be read as one: at
dev scale the probe is 1.5% of training FLOPs, not the declared 0.0942%, and the
Gates are unchanged (Gate 4 still requires the declared target at declared
fidelity). Nothing about the pre-registered protocol moved - only which
configuration a bare ``V0SessionConfig()`` means.

The launch CLI is unaffected: ``scripts/kaggle_queue.py --queue session`` builds
``declared_session_config()`` explicitly for a real launch (pinned by test), so a
declared queue cannot silently train a dev model. Fixed while making this change:
the CLI previously relied on the defaults *being* declared, which would have
downgraded a real launch silently once the defaults changed - the test that
caught it is `test_cli_default_queue_is_the_declared_block`.

## Status Legend

- DONE: implemented and covered by a passing check.
- PARTIAL: a prototype or simulator exists, but the research contract is not complete.
- TODO: not implemented in the real Python training path.
- DEFERRED: intentionally blocked until the first scientific gate passes.

## Current Boundary

The project is a single PyTorch research path (2026-09-19: the TypeScript
"laboratory simulator" and its app-builder scaffold - auth, database, PWA plugins,
vite/eslint/prettier configs, `public/`, `server/`, `migrations/` - were removed;
they had not been touched since the initial commit and represented none of the
results below). The Python foundation has dense and static-MoE models,
deterministic training, checkpoint resume, topology genomes, safe prune/grow, and
mutation-aware MoE rebuilds, plus versioned configuration schemas, deterministic
checksum-addressed token shards, explicit CPU/CUDA device contracts, and reusable
importance signal extraction for static-MoE experts.

The first real scientific result is now decided on three fronts. (1) The
pre-registered 384-seed Gate 3 study (seeds 1000-1383) FAILED for the legacy
cheap-signal controller: 195/384 wins (50.8%), mean delta `+0.006766`, binomial
p `0.3993` - small-sample noise, no benefit. (2) A probe-guided diagnostic upper
bound (`guided` arm) PASSED its 96-seed study (seeds 2000-2095): 59/96 wins,
mean delta `-0.041325`, binomial p `0.0158`, t p `0.00025` - proving topology
selection does express at this fidelity. (3) Making bounded probe-loss selection
the adaptive controller's default decision rule (cap 4 candidates, 2 probe
batches) PASSED the re-registered 384-seed Gate 3 re-run (seeds 3000-3383):
213/384 wins (55.5%, threshold 209), mean delta `-0.025955`, binomial p
`0.018138`, paired t p `7.4e-05`, matched budgets. The same pre-registration
included a search-cost criterion (probe FLOPs <= 5% of training FLOPs) which
FAILED at 41.9%: the quality gate is passed, compute-efficiency is not, and
amortization at realistic scale is the explicit next question rather than an
assumption. (4) A cheaper pre-registered variant (cap 2, 1 probe batch,
steps=144, seeds 4004-4383) satisfied the cost criterion (4.13% <= 5%,
confirming probe cost amortizes with horizon) but FAILED quality: 181/380 wins
(47.6%, threshold 207), mean delta `+0.011747`, binomial p `0.8351`, t p
`0.8857`.The current
map: probe-rich selection wins quality but misses toy-scale cost budgets;
probe-poor selection meets cost but loses the quality edge. (5) A same-seed
paired placement study (cheap probe, steps=144, mutations moved from steps
12/24/36 to 40/80/120; seeds 4004-4383) tested whether early placement explains
that failure: late placement was better in 201/380 pairs, mean paired delta
`-0.015666`, paired t p `0.0788` - directionally suggestive but NOT significant,
so placement does not statistically explain the horizon effect, even though it
changes 39% of probe decisions. The next study should vary probe cadence, not
placement. (6) A pre-registered probe-validity analysis (counterfactual branches
forcing candidate k in {1,2,3} at the first mutation event, seeds 3000-3383)
shows probe loss IS a statistically significant predictor of realized final loss
(mean per-seed Spearman rho `0.1057`, t(383) `3.638`, one-tailed p `1.374e-04`;
pooled z-scored Pearson r `0.1041`, z `4.09`) - but weakly: argmin concordance
`0.281` is near the 0.25 chance level, and the probe-best choice mainly avoids
the worst candidates (mean worst-branch excess avoided `0.0860` vs mean loss
left on the table `0.0601`). A weak per-event signal over only 3 mutation
events is consistent with the marginal Gate 3 pass and motivated a cadence
study. (7) That cadence study (cheap probe cap=2/1-batch, steps=144, interval=24,
budget=6 = six probe-loss events, fresh seeds 5000-5383) then FAILED HARD IN
REVERSE: 137/384 wins (35.7%, threshold 209), mean delta `+0.047014`, binomial
p `1.0`, and adaptive is significantly WORSE than random (worse-than-random
sign test z `-5.61`, two-sided p ~ `2e-08`). Accumulated greedy argmin on a
noisy single-batch probe is net harmful - consistent with a winner's-curse
mechanism: selecting the probe-argmin over many events amplifies probe-batch
noise instead of signal. (8) The pre-registered variance study (cap=2, FOUR
probe batches, 3 events, steps=144, fresh seeds 6000-6383) then FAILED its
quality gate: 179/384 wins (46.6%, threshold 209), mean delta `-0.003907`,
binomial p `0.9159`, t p `0.3453`. The mean shifted in the variance-hypothesis
direction (+0.0117 -> -0.0039) but not significantly (two-sample z ~ `-1.13`).
Per the pre-registered interpretation contract, probe variance reduction alone
does not restore the adaptive edge: candidate-set richness and/or the shorter
horizon of the single passing configuration remain the differentiating axes.
(9) The pre-registered horizon replication (the PASSING cap-4/2-batch config
re-run at steps=144, fresh seeds 8000-8383) PASSED and strengthened the
original: 231/384 wins (60.2%, threshold 209), mean delta `-0.052402`,
binomial p `4.04e-05`, paired-t p `1.07e-08`, matched budgets, probe cost
`14.0%` (amortization matches the fixed-per-event prediction of 41.9%/3).
The pass is therefore NOT horizon-specific; with seven gate-style tests the
study clears Bonferroni alpha/7 ~ 0.0071 on both statistics, and the contrast
with the cap-2/1-batch failure at the same horizon pins the active axis on
candidate richness + probe quality. (10) The pre-registered richness study
then isolated that axis at fixed nominal probe tokens (192/event): arm A
cap-4/1-batch PASSED (228/384 wins, mean delta `-0.036982`, binomial p
`1.4e-04`, t p `5.0e-05`) while same-seed arm B cap-2/2-batch FAILED
(193/384, mean delta `+0.007156`, binomial p `0.47965`), and the paired
A-vs-B comparison confirms A better in 225/384 pairs (sign z `3.37`,
two-sided p `8e-04`). Candidate richness is the active axis; probe batches
matter little within the rich-cap regime (A's delta `-0.037` vs the
2-batch replication's `-0.052`, two-sample z ~ 1.1, n.s.).Gate-style quality tests now number nine, with three passes, all from the cap-4 family.
(11a) The section 5.9 do-nothing control then RE-DERIVED study 3's toy
pass: run as a three-arm block (random, adaptive, null; same registered
config, fresh block 70000-70383, `independent_arm_rng=True`), churn harm
replicated at toy scale (null < random 280/384, p `4.7e-20`) and the
adaptive arm LOST to the null arm (117/384, mean `+0.0508`, p `1.0`) -
the ordering null < adaptive < random means study 3's edge is churn
suppression, not value creation (report section 5.9). (11b) The
section-5.6 depth-4 confirmation then faced the same control (section
5.10, fresh block 71000-71383, same three-arm design): BOTH primaries
pass - churn harm replicates (null < random 254/384, p `1.2e-10`) AND
adaptive beats the null arm (218/384, mean `-0.0297`, p `4.6e-03`,
clearing alpha/2) - so the depth-4 edge is genuine value creation
(~26% of the raw edge survives the do-nothing bar) and Gate 3's depth-4
evidence stands with a third control under it.
(11) The section 5.1 protocol machinery then ran on the REAL TinyStories token
stream at reduced fidelity (`local_proof_run.py --real-data`, vocab-8192 BPE
streams, seeds 9000-9011): the cost identity held again (measured probe cost
1.0077% vs 0.9977% predicted) and arm losses were real language losses, but
the ADAPTIVE LOST 11 OF 12 PAIRS and the pre-registered harm guard engaged
(`PASS_COST_ONLY_HARM_GUARD_REJECTS`). (12) The follow-up real-data selection
diagnostic (`scripts/real_data_diagnostic.py`, same paired seed block
9000-9007, steps=600, every variant's random arm reproducing its baseline
counterpart exactly, so only selection differed) tested both toy-winning axes
on real text and NEITHER RESTORED THE EDGE: variance (cap-4/4-batch, measured
probe cost 3.99% of training FLOPs) went 0/8 with mean delta `+0.2858`;
richness (cap-8/1-batch, 1.98%) went 1/8 with mean delta `+0.1661` - though
its adaptive arm beat the baseline adaptive arm by a mean `0.109` in absolute
terms, so the richness axis itself still operates. Critically, in 5/8 seeds
the 4-batch argmin chose bit-identically to the baseline 1-batch argmin
(final-loss delta exactly `0.0`): on real text the probe ranking is STABLY
MISALIGNED, not stochastic - the winner's-curse (noise-amplification)
mechanism is excluded and the ranking is systematically anti-predictive. The
next discriminator is an argmax control (deliberately select the probe-worst
candidate). (13) That argmax control has now RUN on the same paired block
(seeds 9000-9007): 0/8 wins, mean delta vs own random `+0.3048`, mean
adaptive vs baseline adaptive `+0.0293` - argmax also loses, and by almost
exactly as much as argmin (`+0.2858`). Both tails of the probe ranking lose
to random by the same amount while a non-degenerate control (0/3 identical
picks to argmin on seed 9000; operation balance 11 grow/13 prune vs argmin's
4/20 prune bias) confirms the rule really inverted: the probe signal is
UNINFORMATIVE on real text, not inverted - probe-loss ranking carries no
candidate-ordering information at this fidelity. Per the pre-registered
discriminator contract, the fix must come from probe horizon/tokens (more or
later probe data), not from the decision rule. Gate 4 stays blocked on
real-text evidence. (14) The diagnostic now logs per-candidate probe ranks
natively: a `probe_landscape` flag on `V0SessionConfig` (off by default,
mirroring the toy path) records every probed candidate's probe loss in each
adaptive event's history, sorted best-first, plus the selected candidate's
index; the diagnostic enables it and aggregates a per-variant `probe_rank`
block (mean selected rank, rank histogram, near-top rate) into
diagnostic.json, so future mechanism claims (pick identity, operation
balance, selected rank) are directly measurable from the published seed
summaries. (15) The section-5.3 post-argmax fix candidate, the `heldout`
variant (`V0SessionConfig.probe_data_source="heldout"`; probe windows drawn
from the held-out validation stream, only the windows' source differing,
1.01% measured cost), RAN on 2026-09-14 (revision `eaa1c45`) and its
pre-registered provenance hypothesis came out FALSIFIED (outcome b):
0/8 wins, mean delta vs own random `+0.2840` (baseline argmin `+0.2858`),
adaptive arm just `+0.0085` from the baseline adaptive arm, random arms
replicating. The manipulation was real - mean selected probe loss moved
4.4195 -> 4.2848 under held-out windows and 9/24 picks changed vs the
baseline argmin (more than `var`'s extra batches moved, 7/24) - yet
outcomes did not move: the ranking is a function of the model and candidate
set, not of which iid TinyStories windows score it. Probe-data provenance
is excluded as the mechanism. (16) The long-horizon
lever ran as the `long` variant (`V0SessionConfig.probe_finetune_steps=1`,
one AdamW lookahead step per probed candidate, per-event throwaway
generators, 3.00% measured cost) and its adaptation-blindness hypothesis
also came out FALSIFIED (outcome b): 0/8 wins, mean delta `+0.3106`,
adaptive arm `+0.0350` from baseline adaptive. The lookahead worked at the
ranking level - every landscape re-ranked, grows relaxed to 5/24, and on
seed 9000 the probe top-ranked the grow at 2 of 3 events (the 4/20 prune
bias is a ranking artifact of the snapshot probe, now confirmed) - but
re-ranking did not re-rank toward what helps; across seeds the correlation
of mean selected probe loss with final harm is positive (+0.51 long,
+0.89 heldout; 8 points, indicative). Unified verdict after five
diagnostic variants: the probe ranking is real, stable, and adaptive, but
anti-correlated with downstream value (argmin +0.28 to +0.31 everywhere,
argmax +0.30, only `rich` - the cap-8 variant whose 8/24 grow share
matches the random arms' - shows any win, and it most dilutes the
ranking). Every horizon/token reading of the section-5.3 licensed axis is
exhausted; the untested reading (more probe tokens at fixed fidelity)
would sharpen an anti-correlated signal and is not recommended without a
new mechanism hypothesis. Next direction, outside the section-5.3
contract and requiring its own pre-registration: change how the ranking
is used (layer-level coarseness, or accept/reject gating of random draws).
Full suite: 285 passed (4 lookahead tests). (17) That pre-registration is
now written (report section 5.4) and implemented as the `gated` arm:
`V0SessionConfig(arms=("gated",), gate_margin=theta)` draws a random
mutation FIRST (so the chooser's load-fallback sees the same post-training
routing statistics as the random arm), scores the two-point contrast - the
no-op reference (current model) and the drawn mutation on the same probe
batches - and accepts the draw unless it loses to the reference by more
than theta nats; a rejected draw is rolled back and replaced by the
draw's own load-based grow fallback so the mutation budget stays matched.
The built-in control is exact: the gated arm shares the random arm's init
identity and streams, and at `gate_margin=inf` reproduces the random arm
BIT-FOR-BIT (pinned by test, 7 new tests, suite at 292), including the
global-RNG contract (the accepted rebuild IS the probed model, one
rebuild per accepted event) and routing-load discipline (loads captured
before any probe forward). Gated sessions are single-arm and publish
alongside the baseline block's stored random arms. Cost: two probe
evaluations per live event, ~2% of training FLOPs. The diagnostic gained
a `gated` variant (default `theta=0`) with accept/reject-rate and
draw-minus-reference aggregation. RAN 2026-09-14 - see the Verified Checks
entry below and report section 5.7.

## Substrate Audit (2026-09-14, exploratory and UNREGISTERED)

A code-level audit of the gate path found two structural defects and one
confound. The runs below are exploratory diagnostics on the fresh block
20000-20127, not pre-registered studies (full write-up: `RESEARCH_REPORT.md`
section 9).

1. **The toy path could not build depth > 1.** `StaticMoEConfig` carried a
   single `num_experts` and `StaticMoEBlock` required
   `len(expert_ids) == config.num_experts`, so any genome with unequal
   per-layer expert counts - any real topology mutation at depth > 1 - raised
   `expert_ids must match num_experts` on rebuild. Every published gate study
   is `depth=1` (confirmed in `artifacts/study-summaries/*.json`), so the
   whole factor map covers one layer with a 5-candidate mutation space. FIXED:
   each block derives its expert count from its own ids.
2. **The candidate cap truncated a layer-major list.** The bounded probe took
   `feasible_expert_mutations(...)[:cap]`, so at depth > 1 a cap of 4 reached
   only layer 0.   `probe_candidate_policy` now selects `prefix` (historical default) or
   `spread` (deterministic per-event subset of the whole feasible set), in
   both the toy path and the V0 session (`V0SessionConfig`), so the declared
   protocol can be run either way. The REGISTERED section 5.1/5.2 config is
   unchanged at the `prefix` default, so the pre-registered protocol still
   carries the confound until the policy is switched.
3. **The arms differ in candidate distribution, not only selection.** Added
   `probe_selection=random` (choose uniformly among the SAME probed candidates
   at the SAME probe cost) in both paths, extending
   `V0SessionConfig.probe_selection` from {argmin, argmax}. At
   depth 4, 128 seeds, steps=24, cap 4/1-batch, uniform baseline: argmin
   88/128 (p `1.3e-05`), random-within-pool 80/128 (p `0.0030`), argmax
   64/128 (p `0.535`). The ordering is real (monotone 88 > 80 > 64), but a
   loss-blind control on the same pool also passes, so ~40% of the argmin edge
   is a pool/location effect. Section 2's "only the selection policy differs"
   is corrected. The factor map needs re-derivation with a pool-matched
   baseline before the section 5.1 cost protocol is run.

4. **The depth-4 mechanism passes cleanly once the coupling is fixed (a
   withdrawn claim, corrected).** Under `spread` at the registered horizon with
   independent arm RNG and full power (depth 4, steps=48, 384 seeds
   50000-50383): the registered comparison is 246/384 (mean `-0.0745`, p
   `1.9e-08`); a loss-blind choice from the *identical* pool also beats uniform
   churn (217/384, p `0.0062`), so the pool accounts for ~29% of the mean gain;
   and probe ordering adds a decisive increment over the same pool at the same
   cost (222/384, mean `-0.0529`, p `0.0013`). All three clear Bonferroni
   alpha/3. An earlier draft concluded from a low-power shared-RNG run that
   "the registered comparison fails on a non-degenerate pool" - that reading is
   WITHDRAWN (65/128 coupled vs 80/128 decoupled on the same seeds; the
   64-seed steps=48 block is consistent with the powered rate). Report section
   9.7.
5. **The toy arms are coupled through the global torch RNG.** A grow draws
   fresh expert weights from the stream a prune does not, so runs whose
   treatments choose differently perturb the random arm's later rebuilds (the
   random arm is bit-identical across rules in only 70/128 seeds on the audit
   block; 384/384 once fixed). The V0 path is not affected (arms train
   sequentially with per-arm seeding). Fix implemented as the opt-in
   `ExperimentConfig.independent_arm_rng` (default `False`, so published numbers
   reproduce unchanged). Convention: every new study pins it explicitly (as
   5.6 does) rather than relying on a flipped default, because the CLI is also
   the reproduction surface for the pre-audit studies, whose records do not
   carry the field. See report section 9.8. Re-run of study 3's own block 3000-3383: the shared
   path reproduces the committed artifact exactly (213/384, mean `-0.025955`,
   p `0.01814`) and the fix gives a slightly stronger 222/384 (mean `-0.034406`,
   p `0.00128`), so the depth-1 pass survives the fix. Report section 9.8.
6. **The depth-4 mechanism is CONFIRMED out-of-sample under a pre-registered
   design.** The section 5.6 pre-registration was committed (`b876ac1`) before
   the run and executed once on the never-observed block 60000-60383 (384
   seeds, depth 4, `spread`, `independent_arm_rng=True`, cap 4/2-batch,
   steps=48). Result PASS: argmin vs uniform churn 255/384 (p `6.1e-11`),
   paired argmin vs the identical-pool loss-blind control 236/384
   (p `4.1e-06`), pool-matched control 209/384 (p `0.046`), validity control
   (`argmax`) 173/384 (p `0.977`, does not pass), random-arm invariance
   384/384. Both decision comparisons clear Bonferroni alpha/3; the pool
   accounts for ~32% of the mean gain and the selection increment for ~68%.
   This is now the project's Gate 3 evidence - the first pass on a pool (20
   candidates over 4 layers) genuinely larger than the search cap, on the
   decoupled RNG path, and out-of-sample. Report sections 5.6 and 9.9.
   Hardware note: pin `--device cpu`; the default `auto` selects CUDA and 16
   processes sharing one GPU made the same study >5 min slower (>13 min before
   being killed vs ~2.5 min on CPU shards).
7. **The published factor map has been re-run on the decoupled path.**
   Section 9.8 showed all nine studies used the drifting baseline, so section
   9.10 declared a reproduction check (published config, seed block and
   treatment arm, only `independent_arm_rng` changed) and section 9.11 records
   the result on 144 shards (~6 min wall at 16 single-threaded cores;
   `artifacts/study-summaries/factor-map-recheck.json`). **9 of 10 verdict
   labels reproduce.** Every win count moves toward the treatment (+1 to +30,
   mean +9.2, all ten positive), so the coupling was a *directional* bias that made
   every published number - passes and nulls - understate the adaptive arm.
   Studies clearing Bonferroni alpha/9 rise from **2/9 to 4/9** (horizon
   4.0e-05 -> 1.7e-09, richness A 1.4e-04 -> 9.4e-05, probe-loss 0.0181 ->
   0.00128, guided 0.0158 -> 0.00279). The single **label flip** is variance
   reduction: 179/384 FAIL -> 209/384 PASS at exactly the alpha=0.05 threshold,
   which fails alpha/9 and is therefore a null in both directions - it must not
   be quoted as evidence that more probe batches help. Cadence's
   "significantly worse than random" claim survives but weakens
   (137/384 -> 148/384). The legacy signal study was initially excluded as not
   reconstructible, but section 9.12 recovered the field its config predates
   (`adaptive_policy="signal"` - proven by reproducing the published coupled
   record to 4.8e-7 where the default sits 1.1e-1 away), so its row is now in
   the set: 195/384 FAIL -> 201/384 FAIL, +6, reproduces. Net: no headline claim
   is overturned and no verdict needs revising except variance, but the depth-1
   factor map should now be quoted from the decoupled column.
8. **The probe-validity counterfactual analysis has a committed generator, and
   re-runs weaker.** The analysis behind report section 3.2 and figure 4 had
   **no generator in the repository** - nothing produced
   `artifacts/study-summaries/probe-validity-analysis.json` - so its numbers
   were reproducible only in principle. `scripts/probe_validity_analysis.py` is
   that entry point; it recovered the definitions from the committed file
   (including two non-obvious ones: `h1_one_tailed_p` is a **normal** tail, not
   a t tail, and `pooled_zscored_pearson_r` z-scores **within** each seed
   before pooling), reproduces all 13 stored fields from the committed
   `per_seed` block, and on a fresh 4-branch x 384-seed coupled sweep
   reproduces the committed results field for field. Decoupled re-run: mean
   per-seed Spearman rho **0.1057 -> 0.0776** (p 1.4e-04 -> 4.98e-03), pooled r
   0.1041 -> 0.0809, while the avoidance statistics barely move (concordance
   0.281 -> 0.284 vs chance 0.25, excess avoided 0.0860 -> 0.0839, left on
   table 0.0601 -> 0.0592). Branch 0 reproduces the study summary's adaptive arm
   seed for seed (max drift 1.9e-6: the identity is exact in principle, not
   portably bit-exact across environments). Not pre-registered - it is a
   controlled re-measurement of a frozen `design` string, and section 9.11
   labels it as such rather than as new evidence. Report sections 3.2, 8, 9.11.
9. **The legacy signal study is reconstructible, not unknowable.** Section 9.10
   excluded `gate3-legacy-signal-384` as not reconstructible, which was right
   about the *default* - its recorded config predates `adaptive_policy`, which
   the dataclass would fill with `probe_loss`, a different experiment - but
   wrong to conclude the value was unavailable. It is recoverable three ways:
   the transition commit `d972d4c` inserted the probe branch above an untouched
   legacy chooser (so `"signal"` *is* the old controller); the raw record
   `artifacts/gate3-384.jsonl` has no probe fields in any `mutation_history`;
   and supplying `"signal"` reproduces the published coupled record to 4.8e-7
   where the default sits 1.1e-1 away (pinned as a test). Declared in section
   9.12 before the run, then re-run decoupled: **195/384 FAIL -> 201/384 FAIL,
   +6, reproduces.** The corrected figure set now carries all nine rows. A
   related defect fixed on the way: the recheck's per-shard confirmation line
   read `record['per_seed']` (the probe-validity schema), which crashed a shard
   *after* it wrote its file and made completed shards look failed.

Iteration cost for these checks: ~1.5 s/seed on CPU at depth 4 (GPU measured
slower at 2.21 s/seed for a hidden-24 model); a 128-seed study is ~85 s.

## Milestone Comparison

| Plan milestone | Status | Evidence / gap |
|---|---|---|
| 1. Foundation and baseline contract | DONE for the first gate | Python model, optimizer, training state, topology genome, hashing, model factory, and versioned checkpoint metadata are covered. Large-model and real-data contracts remain outside this gate. |
| 2. Dense baseline | PARTIAL | Dense causal LM trains on deterministic synthetic batches and resumes parameter-for-parameter. It is a tiny gate model, not the planned 256M-350M research model; validation loss, FLOPs, and real data metrics are not yet recorded. |
| 2. Static MoE baseline | DONE for the first gate | Static top-k MoE trains through the shared runtime and reports validation loss, active parameters, parameter count, and validation FLOPs. |
| 3. Topology graph and safety | PARTIAL | Python genome serialization, validation, hashing, prune/grow, merge/split/rewire guards, expert-ID weight preservation, and accepted mutation fossils exist. Full graph-aware model execution and post-mutation causal checks remain deferred. |
| 4. Random topology control | DONE for the first gate | Python Experiment 0 provides reproducible random churn with matched cadence, mutation count, training/search accounting, and report metrics. Real data and scale remain deferred. |
| 5. Importance instrumentation | DONE for the first gate | Python records routing/load, gradient norm, parameter norm, Fisher-style EMA, temporal stability, mutation count, and genome hash at structural windows. |
| 6. Adaptive controller | PARTIAL | Python exposes a deterministic, validated budget-aware `AdaptiveController` plus Pareto frontier/population selection over quality, active parameters, and search FLOPs. The validated default reaches 10/16 wins; broader multi-generation search remains. |
| 7. Experiment ledger | DONE for the first gate | Browser ledger and Python append-only JSONL registry store config, dataset/tokenizer hashes, code revision, hardware, tokens, FLOPs, arm metrics, mutation fossils, and architecture hashes. Deterministic evaluation summaries are available for report generation. |
| 8. Final gate: adaptive > random | **PASSED under a pre-registered, out-of-sample depth-4 design.** Section 5.6 was registered before the run and executed once on the fresh block 60000-60383 (384 seeds, depth 4, `spread`, `independent_arm_rng=True`, cap 4/2-batch, steps=48): registered comparison 255/384 (p `6.1e-11`), controlled same-pool selection increment 236/384 (p `4.1e-06`), loss-blind same-pool control 209/384 (p `0.046`), `argmax` validity control does not pass (173/384), random-arm invariance 384/384 - all three decision rules clear Bonferroni alpha/3 where required. This is the first gate pass on a candidate pool (20 over 4 layers) genuinely larger than the cap, with the arm-RNG coupling fixed and out-of-sample. Earlier evidence: an exploratory powered depth-4 run (246/384, p `1.9e-08`; pool 217/384; increment 222/384, p `0.0013`), and three depth-1 passes on the shared-RNG path (study 3's block reproduces exactly at 213/384 and gives 222/384 with the fix) that should now be read as a degenerate 5-candidate-space instance. See the Substrate Audit section and report sections 9.7-9.9. | Factor map (all matched budgets, objective verdicts): cap-4/2-batch, 48 steps -> PASS 213/384, p `0.0181`, at 41.9% cost. Cap-2/1-batch, 144 steps -> FAIL 181/380, p `0.8351`, at 4.13% cost. Placement (late, same seeds) -> no effect (p `0.0788`). Cap-2/1-batch, 6 events -> 137/384, significantly WORSE than random (z `-5.61`). Cap-2/4-batch -> 179/384, p `0.9159`. Horizon replication cap-4/2-batch, 144 steps -> PASS 231/384, p `4.04e-05`, t p `1.07e-08`, cost 14.0%. Richness A cap-4/1-batch, 144 steps -> PASS 228/384, p `1.4e-04`, t p `5.0e-05`, cost 7.3%; same-seed B cap-2/2-batch -> FAIL 193/384, p `0.47965`; paired A>B 225/384 (z `3.37`). Probe validity real but weak (rho `0.11`); richness is the active axis, probe batches secondary. Nine gate-style tests, three passes; Bonferroni alpha/9 ~ 0.0056 is cleared by the 144-step replication and richness-A on both statistics. Real-data diagnostic (TinyStories, seeds 9000-9007, paired): variance cap-4/4-batch 0/8 (`+0.2858`), richness cap-8/1-batch 1/8 (`+0.1661`) - neither toy axis restores the edge on real text, and the 4-batch argmin bit-reproduced the 1-batch argmin in 5/8 seeds (ranking stably misaligned, not noisy). **Re-run on the decoupled arm-RNG path (sections 9.10-9.12, Substrate Audit items 7-9): 9 of 10 labels reproduce, every win count moves toward the treatment (+1..+30, mean +9.2), studies clearing alpha/9 rise 2/9 -> 4/9 (horizon 1.7e-09, richness A 9.4e-05, probe-loss 0.00128, guided 0.00279), and the single flip is the variance study (179/384 FAIL -> 209/384 threshold PASS, failing alpha/9, i.e. a null either way). Cadence's harm claim survives at 148/384 (from 137/384). The legacy-signal study's predated `adaptive_policy` field is recoverable and its row is in the set (195/384 FAIL -> 201/384 FAIL, reproduces), so no factor-map study now rests on the coupled path alone.** So the factor-map claims stand, with corrected effect sizes; quote them from the decoupled column. |

## Plan Sections 3-6: Compute and Kaggle Operations

| Area | Status | Remaining work |
|---|---|---|
| T4x2 PyTorch/DDP | PARTIAL | Add device selection and DDP launch helpers now discover torchrun rank/world-size, initialize an appropriate process group, and wrap models only for multi-process runs; scheduler, accumulation, autocast, and activation checkpointing are available. T4x2 validation remains. |
| TPU v5e/JAX | DEFERRED | Port only after a stable architecture wins the GPU gate. |
| Kaggle artifact datasets | DONE for the 5.2 handoff | Deterministic checkpoint, evaluation, report, and manifest layout contracts exist; the V0 5.2 per-seed handoff (`act.training.v0_handoff`: atomic publish, checksum-verified restore, bit-identical resume through the dataset round trip) is implemented, and the remote half is now code: `act.ops.kaggle_remote` wraps the optional kaggle package (injectable client, zero-copy push of the handoff directory with generated metadata, restore merges manifest-complete seed versions over matching locals and keeps local-only ones) with `--kaggle-dataset owner/slug` on the launch command - restore at session start (fail-fast before any training), publish-on-exit that never raises (outcome recorded in `remote-publish-log.json`). |
| Session resume protocol | DONE for the 5.2 handoff | Local checkpoint resume is deterministic for baselines and bit-identical for V0 sessions (verified mid-run resume: losses, FLOPs counters, architecture hashes, and mutation histories all reproduce exactly); the 5.2 versioned-dataset handoff round-trips a seed working-dir -> dataset -> working-dir bit-identically, and the remote leg now exists (`--kaggle-dataset` restore at session start, publish-on-exit; remote seed versions merge over matching locals, local-only seeds are kept, partial remote versions are ignored). |
| Repository structure | PARTIAL | Core Python data, evaluation, evolution, importance, model, topology, training, and Kaggle-artifact implementations exist with focused tests; all ten plan section 39 notebooks (`00`-`09`) now exist under `notebooks/` (deterministic builder, tested), while standalone research entry points remain. |

## Plan Sections 8-24: Model and Topology

| Area | Status | Remaining work |
|---|---|---|
| V0 256M-350M model | PARTIAL | Bounded V0 construction now runs finite forward/loss probes and reports parameter/logit shapes; target-scale memory and throughput measurements remain. |
| Attention/GDN/hybrid model core | PARTIAL | Python runtime supports dense, static MoE and the gated-delta mixer (`layer.module` selects it); GDN at the ~100M zoo scale OOMs on 8 GB (`ARCH_SEARCH_FINDINGS.md` §7). |
| Full genome contract | PARTIAL | Python genome now represents heads, modules, widths, residual paths, context, and compute ceilings and can reconstruct the V0 dense/hybrid model; MoE/shared-expert reconstruction remains. |
| Weight/topology separation | DONE for the first gate | Model and optimizer remain separate from the Python genome, while versioned checkpoints and Experiment 0 fossils package genome and mutation metadata together. |
| Importance levels and lifecycle | PARTIAL | Python exposes routing, gradient, parameter, Fisher-style EMA, magnitude, temporal-stability records, temporary ablation deltas, and append-only lifecycle fossils; population aggregation remains. |
| Expert ecology and similarity graph | PARTIAL | Device-safe expert signal extraction, functional sketches, and cosine similarity primitives exist; persistent graphs and lifecycle fossils remain. |

## Plan Sections 25-42: Data, Training, and Reproducibility

| Area | Status | Remaining work |
|---|---|---|
| Real tokenizer and shards | PARTIAL | Versioned deterministic binary token shards, bounded reads, checksums, manifests, and synthetic fixtures exist; real tokenizer/corpus adapters remain. |
| Real language-model validation | PARTIAL | Add real held-out batches; bounded perplexity and deterministic named capability probes now report weighted loss, perplexity, accuracy, and token counts, while real corpora remain. |
| Complete checkpoint format | PARTIAL | Versioned local checkpoints persist model/optimizer/training RNG state plus scheduler, genome, topology history, importance, dataset position, config, and revision metadata; compatibility validation and bounded session retention are implemented, while remote/session resume remains. |
| Experiment registry | DONE for the first gate | Append-only JSONL records include experiment/config/dataset/tokenizer hashes, hardware, tokens, FLOPs, architecture hashes, and arm results. |
| Reproducibility hashes | DONE for the first gate | Genome, config, dataset, tokenizer, and complete experiment hashes are recorded. |
| Research notebooks/scripts | PARTIAL | Dense and Experiment 0 CLIs now support baseline runs, append-only records, and multi-seed gate studies; extraction, retrain, and compare entry points remain. |

## Plan Sections 43-66: Scientific Experiments

| Experiment | Status |
|---|---|
| Experiment 0: fixed vs random vs adaptive | PARTIAL | Matched-budget Python runner, metrics, append-only single/study registry records, reproducible multi-seed CLI study, and a 6/8 probe exist; universal Gate 3 is still open. |
| Conventional dense/static-MoE frontier | PARTIAL |
| Dynamic sparse prune/grow | PARTIAL | Implemented in the Python Experiment 0 runner for safe expert prune/grow; richer topology operators remain. |
| Expert ecology clone/split/merge/death | TODO |
| Module-type evolution | DEFERRED |
| Adaptive depth | PARTIAL | Python topology mutations can safely toggle effective layer depth and change attention/GDN/hybrid module metadata; full multi-generation training comparisons remain. |
| Hierarchical search | PARTIAL | Deterministic layer/module candidate ranking now supports explicit depth penalties and inspectable priors. |
| Qwen-informed search | PARTIAL | Prior interface is available for imported architecture scores; no external Qwen artifacts are bundled or claimed. |
| Teacher initialization | PARTIAL | Explicit teacher and evolved shape-safe parameter transfer policies now exist; multi-seed retraining comparisons remain. |
| Architecture reusability and fresh initialization | PARTIAL | Fresh, evolved, and rewound initialization policies report copied/skipped tensors; architecture-equivalence studies remain. |
| Cross-seed stability and architecture equivalence | PARTIAL | Strict architecture-hash equivalence and cross-seed mean/stddev/min/max summaries now exist; full retraining studies remain. |
| Compute-efficiency scaling | PRE-REGISTERED (execution path built; pre-launch tooling done) | Real-scale measurement protocol pre-registered in `RESEARCH_REPORT.md` section 5.1: declared 285M V0 config, predicted probe cost  0.094% (band 0.047-0.189%), 5% criterion + tolerance band + harm guard on 12 seed pairs, now **15000-15011** (the registered 9000-9011 was spent by the section 5.3 diagnostic), with the primary controller on `probe_candidate_policy=spread` and the adopted pool-matched control clause; PASS authorizes Gate 4. See section 5.5 (adopted) for the four changes. The V0 training path is implemented: declared `V0MoECausalLM` reproduces 285,336,576 / 134,341,632 parameters exactly, per-layer expert genomes rebuild weight-preserving after grow/prune, and `act.training.cli --mode v0` runs resumable sessions with version-4 checkpoints (FLOPs counters, genome, generator state persisted). Session integration checks live in `scripts/v0_session_selfcheck.py` (run on demand, NOT in the pytest suite). GPU execution plan pre-committed in section 5.2: per-GPU fp32 fit ~4.5 GB of 16 GB, 3.19 GB checkpoint payloads, `max_checkpoints=2` (pair steady state ~14.9 GB of the 20 GB working dir), throughput-gated dtype rule (fp32 floor 2,500 tokens/s, fp16 fallback), parameterized session schedule with atomic pairs and deadline/publish rules, and the versioned-dataset checkpoint handoff (~45 GB ledger). The 5.2 pre-launch tooling is now code: `scripts/v0_throughput_benchmark.py` (declared-model step-loop benchmark emitting the dtype decision and schedule artifact) and `act.training.session_plan` (registered schedule/deadline/working-dir arithmetic, unit-tested against the 5.2 tables); declared sessions now train with checkpointed activations as specified, with resume re-verified bit-identical; `act.training.v0_runner.run_pair_session` ties benchmark numbers, deadline rules, session training, and the dataset handoff into the per-seed session loop (atomic pairs, in-progress snapshots on deadline stops, restart-from-scratch, skip-completed idempotence), and `--mode pair-session` exposes it as the single launch command with the dtype rule enforced against the benchmark artifact; the fp16 contingency trains under autocast + GradScaler with scaler state in the version-5 checkpoint contract. |

## Plan Sections 67-105: Scale and Final Deliverables

These remain TODO or DEFERRED until the first gate passes:

- full metric and visualization report (PARTIAL: the four pre-registered
  factor-map figures now exist under `artifacts/study-summaries/figures/`,
  generated reproducibly from the versioned summaries; trajectory plots and
  real-scale metric visualizations remain);
- topology fossils and trajectory plots;
- Multi-generation Pareto population search and richer diversity preservation;
- T4x2 scaling to 256M-350M;
- Qwen analyzer and architecture priors;
- fresh/rewound/teacher retraining;
- 500M-1B scaling;
- TPU port;
- paper-quality comparisons and final research report;
- versioned Kaggle datasets and reproducible notebooks (PARTIAL: all ten
  plan section 39 notebooks - `00_environment` plus `01`-`03` per section
  104 and `04_importance`, `05_adaptive`, `06_search`, `07_extract`,
  `08_retrain`, `09_compare` - now exist under `notebooks/`, emitted by the
  deterministic builder `scripts/build_notebooks.py` (byte-identical
  re-runs, unit-tested in `tests/test_notebooks.py`, every cell executed
  end-to-end); the remote dataset publishing is now code as well
  (`act.ops.kaggle_remote` + `--kaggle-dataset`, below) with the remote
  round trip now live-verified with real credentials); the §5.1 execution
  is operator-scripted in `T4_RUNBOOK.md` with the verdict machine-checked
  by `scripts/v0_verdict.py` - the only open piece is the T4 run itself;

## Immediate Execution Order

1. DONE: add Python metrics for validation loss, parameters, active parameters, training FLOPs,
   search FLOPs, validation FLOPs, and architecture distance.
2. DONE for the first gate: expand the Python checkpoint to include genome, mutation history,
   optimizer/scheduler, dataset position, and importance/controller state.
3. DONE for the first gate: implement a Python structural clock and random-churn controller
   using the existing prune/grow and rebuild APIs.
4. DONE for the first gate: implement cheap Python importance records: routing/load, gradient norm,
   parameter norm, Fisher-style running estimate, and temporal stability.
5. PARTIAL: use persisted routing records in the adaptive prune/grow controller while retaining
   the slower 48-step / 12-step structural clock and exact random budget. Signal sweeps did not
   yet improve the 6/8 default result.
6. DONE: add validation loss, active-parameter, and validation-FLOPs reporting to the runner.
7. RESOLVED (PASSED on quality, cost open): the cap-4/2-batch probe-loss
   controller passed the 384-seed Gate 3 re-run (213/384, p `0.018138`) at 41.9%
   search cost; a cheaper cap-2/1-batch variant at steps=144 then met the cost
   budget (4.13%) but failed quality (181/380, p `0.8351`) on fresh seeds
   4004-4383.
- `PYTHONPATH=src python -m act.training.cli --mode pair-session ...
  --kaggle-dataset owner/slug` -> the remote publish-on-exit wrapper
  (`act.ops.kaggle_remote`), **live-verified against the real Kaggle API
  with credentials**: publish -> download -> bit-identical seed handoff
  round trip (all manifest checksums re-verified on the returned copy),
  and the registered one-line launch command completes restore ->
  skip-complete -> exit publish with exit 0. The live probes established
  the upload layout the adapter implements: Kaggle datasets are flat, but
  the service auto-extracts uploaded tar/zip blobs preserving internal
  paths, and the package's default directory handling is unusable for
  nested trees (`skip` drops them silently; `tar` archives contents
  without a name prefix, so multiple seeds would collide) - the adapter
  therefore stages one tar per seed with paths rooted at `seed-NNNN/...`
  and pushes them flat. A nonexistent dataset (session 1) is the normal
  bootstrap state on restore (warn and train from scratch; atomic pairs
  make it safe) and falls back to `dataset_create_new` on publish; any
  other error is fail-fast at launch and never-raising at exit. The test
  dataset was deleted after the round trip; unit tests still inject a
  fake client, so the kaggle package is never needed by the suite.
8. RESOLVED (Gate 3 re-derived and PASSED on a non-degenerate substrate): the
   substrate audit qualified all three depth-1 passes (5-candidate single-layer
   space, depth > 1 unbuildable, a loss-blind same-pool choice reproducing part
   of the effect), so Gate 3 was re-derived rather than assumed. The section 5.6
   pre-registration was committed before the run and executed once on the fresh
   block 60000-60383: registered comparison 255/384 (p `6.1e-11`) and the
   controlled same-pool selection increment 236/384 (p `4.1e-06`), both clearing
   Bonferroni alpha/3, with the pool-matched control clearing only alpha
   (209/384, p `0.046`) and the `argmax` validity control not passing. Gate 3 is
   now established on a pool larger than the cap, out-of-sample, with the
   arm-RNG coupling fixed (report sections 5.6, 9.9; Substrate Audit item 6).
   NEXT (compute-efficiency): cap-4/1-batch cuts probe cost to
   7.3% at toy scale and amortizes with horizon, but real-scale measurement is
   the plan's deferred compute-efficiency milestone and gates Gate 4 (adaptive >
   static MoE). Controller default stays cap-4/2-batch; cap-4/1-batch is the
   validated cost-saving option. The real-scale measurement is now
   **pre-registered** (`RESEARCH_REPORT.md` section 5.1): declared V0 config
   (285,336,576 total / 134,341,632 active parameters), predicted probe cost
   0.094% of training FLOPs under the measured per-event identity, decision
   rules fixed (5% criterion + factor-2 tolerance band + anti-adaptivity harm
   guard) (+ the section 5.5 pool-matched control and, per section 5.14, the
   null queue). Running it as specified - without retuning knobs against the
   result - is the entry condition for Gate 4. **Section 5.39 gives the null
   queue a measured bar (2026-09-16)**: at dev fidelity on the real stream,
   doing nothing beat churning on 18/18 seeds under both the historical rebuild
   and the section 5.35 repair (-0.16449 and -0.11279 nats, permutation p
   5e-05), so the repair removes 31% of the churn cost rather than all of it and
   an adaptive arm that merely matches random is 0.11279 nats worse than doing
   nothing. The do-nothing clause is load-bearing, not insurance, and the
   declared run must carry the null queue. **The protocol is now REGISTERED AS
   REVISED**: `RESEARCH_REPORT.md` section 5.5 was adopted in full
   (2026-09-14) and implemented, so the launch is unblocked and the only
   remaining prerequisite is the T4 hardware. The revision was needed for
   three reasons. The registered prefix controller reaches only layer 0 at the
   declared depth; the pool-matched control arm was the missing quality check
   (its premise is now backed by the registered section 5.6 PASS); and **the
   declared seed block 9000-9011 was already spent**, so the protocol could no
   longer be a blind measurement - the section 5.3 diagnostic ran the full
   queue through the same `run_pair_session` at reduced fidelity and a 5.1
   verdict was computed over it (`artifacts/real-proof/datasets/
   v0-5-1-verdict.json`: measured cost `1.0077%`,
   `PASS_COST_ONLY_HARM_GUARD_REJECTS`, `gate_4_authorized: false`, adaptive
   losing 11 of 12 pairs), meaning the harm guard - one of the verdict
   branches - was no longer blind. Running that queue again at full fidelity
   after seeing a harm-guard rejection is the "measure again until it passes"
   pattern the protocol exists to prevent. Adoption moved the queue to the
   unused block **15000-15011** (`v0_runner.DECLARED_SEEDS`), made the primary
   controller `spread` and added the pool-matched control as a second session
   (both exposed as `--probe-candidate-policy` / `--probe-selection` on
   `--mode pair-session`), and gave `scripts/v0_verdict.py` the control block,
   the interpretation clause (`POOL_CONTROL_WIN_LIMIT` = 10, the same bar the
   harm guard uses) and a `--control-root` flag. A missing control withholds
   Gate 4 rather than being silently skipped. The section 5.2
   pre-launch tooling (throughput benchmark, session scheduler, deadline rules,
   working-dir ledger, versioned-dataset handoff) is now code and unit-tested,
   and the pieces are tied together by `act.training.v0_runner.run_pair_session`,
   the per-seed session loop over the declared queue 15000-15011 (atomic pairs,
   rule-1 gating, cooperative deadline stops with in-progress snapshots,
   restart-from-scratch, skip-completed idempotence). The launch command is
   `python -m act.training.cli --mode pair-session`: it reads the session-1
   benchmark artifact, re-derives and cross-checks the dtype decision,
   applies the registered max_checkpoints=2 pin, and runs the session loop
   with the dataset handoff. The registered fp16 contingency is executable:
   fp16 sessions train under autocast with a GradScaler (checkpoint contract
   v5 carries the scaler state; fp16 resume verified bit-identical, version-4
   payloads still load in fp32). What remains before launch is running the
   benchmark on the T4 to fix the measured dtype and schedule; the whole
   execution is now operator-scripted in `T4_RUNBOOK.md` (pre-flight,
   benchmark, the one-line launch command, queue rules, machine-checked
   verdict via `scripts/v0_verdict.py`, contingencies, artifact trail), and
   `scripts/v0_verdict.py` computes the pre-registered §5.1 verdict
   (aggregate cost vs 5% criterion, tolerance band from
   `expected_probe_cost_percent`, harm-guard tally, `gate_4_authorized`) so
   no verdict number is hand-computed. The launch
   path is now reproducible-notebook-ready: all ten plan section 39
   notebooks (`00_environment` through `09_compare`) exist under
   `notebooks/` (deliverable 2), so a session can verify the compute
   contract before `--mode pair-session` and reproduce the baseline, churn,
   gate-study, importance, search, extraction, retraining, and comparison
   claims from the same committed revision.

## Verified Checks

- **Section 5.1 T4 pre-flight DONE on a real Kaggle T4 (2026-09-15).** The
  runbook's pre-flight 4 and section 1 ran as a script kernel on a
  `NvidiaTeslaT4`: the declared config reproduced 285,336,576 total /
  134,341,632 active parameters, and the benchmark (`--steps 50`, forward +
  backward + AdamW with checkpointed activations) measured **1,958.7
tokens/s -> `dtype_decision: fp16`** (below the 2,500 floor, the registered
  contingency). Derived schedule: `pair_time_hours` 3.585,
  `full_pairs_per_session` 2, `sessions_for_primary` 6, `working_dir_headroom_gb`
  +5.12. The artifact is `artifacts/v0-benchmark.json` (gitignored) and was
  downloaded from the kernel output, so the launch command's input exists and
  is not a hand-edited file.
- **Two schedule defects found and fixed 2026-09-15; they nearly cancel, so
  the recorded total was accidentally right for the wrong reasons.**
  (a) *The schedule was built on the rule's input, not the runtime dtype.*
  Section 5.2's rule reads the **fp32** rate to choose a dtype, and the
  schedule was parameterized from that same number while the sessions train
  fp16. Measured on the T4 (`--queue dtypeprobe`): fp32 **2,066.3** tokens/s
  against fp16 **4,694.9**, i.e. 2.27x. The benchmark now measures the decided
  dtype too and records `schedule_tokens_per_second` +
  `tokens_per_second_by_dtype` beside the fp32 `measured_tokens_per_second`,
  and the launch command schedules from the former while still enforcing the
  pre-committed rule against the latter.
  (b) *A pair's arms are trained sequentially, not one per GPU.*
  `pair_time_hours` was documented as "one per GPU in parallel ... the wall
  time is one arm's training time", and `PairSchedule.arms` was left unused in
  that formula - the arithmetic tell. `run_v0_session` trains `config.arms`
  with a sequential generator on one device, measured on a real pair (arm wall
  times 0.531 s + 0.298 s added rather than overlapping), so every pair cost
  twice what the clock believed. That is the dangerous direction: the deadline
  rule would start pairs it could not finish, and section 5.2 restarts an
  interrupted pair from scratch. The multiplicity is now in the formula and in
  the runner's own deadline arithmetic.  Corrected plan, re-measured by a pre-flight on 2026-09-15 (fp32 2,124.2 /
  fp16 5,433.2, pair 2.61 h, 3 pairs/session): primary 12 x 2.61 = ~31 GPU-h,
  control ~31, null (one arm) 12 x 1.36 = ~16, so ~79 GPU-h - the recorded
  "2c ~0.3" does not follow from the runner, since an arm's token budget is
  `batch x context x steps` independent of arm count. Dtype stays an execution
  detail touching no pre-registered criterion.
  **The same pre-flight answered the device question: `gpu_inventory` reports
  two `Tesla T4`s and `nvidia-smi` shows both idle.** So the pair model's
  "one per GPU" is reachable, the second T4 has been idle for the entire run,
  and running each arm on its own device is the remaining change that takes
  the pair to 1.36 h / 6 pairs per session / 2 sessions for the primary queue -
  ~47 GPU-h for the whole run instead of 79. That is the parallelism section
  5.2 already specifies, so implementing it aligns execution with the
  registration rather than changing it. The benchmark also refuses
 to write an artifact
  whose schedule fits no pair: on a non-T4 GPU the decided fp16 measured 4.2x
  *slower* than fp32, and the artifact described a session with
  `full_pairs_per_session: 0` - billing a whole window to publish nothing,
  silently, because the deadline rules never start a pair the clock cannot
  finish.
- **Section 5.2's "one arm per GPU" is implemented (2026-09-15).** The two-T4
  measurement above made the registered parallelism reachable, so a pair whose
  host has a device per arm now trains each arm in its own process
  (`src/act/training/arm_worker.py`, driven by `v0_session._train_arms`):
  `arm_devices` places arm *i* on `cuda:i` when the host has that many devices
  and on one device otherwise, `arm_slots` reports the same placement as a
  concurrency, and `PairSchedule.pair_time_hours` is charged per *wave* of
  arms - so the plan, the runner's own deadline arithmetic and the execution all
  read the placement from one definition instead of three copies.
  A process per arm rather than threads because model init consumes global
  torch RNG and `torch.manual_seed` is global. A **deadline stop crosses the
  boundary as a stop**, not as a failure: the child exits a distinct status and
  records the step it stopped at, and the parent re-raises
  `V0SessionInterrupted`, so the sequential path's in-progress snapshot still
  happens (`[6/6]` in the battery checks this). Without it a stop would have
  been recorded as `training_failed`, losing the snapshot and reporting an
  orderly deadline as a crash. A pair that raises for any *other* reason is
  recorded as `training_failed` and stops the queue instead of crashing the
  session: the durable copy is the exit publish, which a crash never reaches,
  so an OOM at seed 6 would otherwise discard the five pairs before it. The
  driver reads that action from the session log and exits 1 rather than
  pushing a session that would repeat the failure. Correctness is equivalence: the
  arms share nothing but the seed, so `scripts/v0_session_selfcheck.py`'s fifth
  check runs both paths and requires bit-identical arm results (losses,
  architecture hashes, FLOPs, mutation histories) through the process runner
  *and* through `run_v0_session`. Every arm now records the device it used in
  its result, so the published per-seed summary carries the placement as
  evidence rather than assertion. This changes no pre-registered criterion -
  it is the configuration section 5.2 prices; the sequential path remains the
  default wherever the host lacks the devices.
  **Confirmed on the target host.** The two-device path cannot be exercised on
  the project's one-GPU box, so `--queue selfcheck` was added as a one-shot job
  that runs the battery in a Kaggle kernel. On a real two-T4 kernel
  (`sid123123234/act-t4-selfcheck`, 2026-09-15, ~2 min of GPU) it reported
  `host devices: 2` and `one arm per device: cuda:0, cuda:1` with every check
  passing - the placement section 5.2 prices, measured on the hardware that has
  it, before any session assumed it. Nothing else moved: the same tiny pairs
  produce the same losses, FLOPs and architecture hashes however the arms are
  scheduled.
- **A checkpoint is never left truncated at the name retention finds.**
  `save_v0_checkpoint` wrote straight to `checkpoint-NNNNNN.pt`. A declared
  checkpoint is ~3.4 GB, so the write takes seconds, and a process killed
  inside it (the OOM killer, a 12-hour kernel timeout) would leave a truncated
  file at the final name - which `retain_checkpoints` takes for the newest,
  `publish_seed` checksums into a `complete` version, and restore's checksum
  verification cannot distinguish from a good one. Then the seed's published
  state is unrestorable while every check says it is a complete version, and
  nothing downstream loads checkpoints (the verdict reads summaries), so no
  reader would notice. The write now goes to a sibling `.partial` name and is
  renamed into place, and a test kills the write and asserts the final name is
  still the intact checkpoint (teeth-checked: reverting the rename fails it).
  The battery's newest-checkpoint helper now calls the production
  `newest_arm_checkpoint` instead of ordering by mtime, so a resume check
  cannot compare a different file than the run would resume from.
- **A provisional verdict on a partial block is not the driver's exit code.**
  The verdict script deliberately computes a provisional verdict on an
  incomplete declared block (it records `seeds_missing`, prints a warning and
  withholds Gate 4), which is useful as a progress report - but it returns 0
  for a provisional `PASS`, and the driver mirrors that code as the run's
  outcome, so a scripted caller could read a partial block as the registered
  decision. The driver now checks `queue_complete` in the artifact and returns
  3 (`no verdict produced`) when it is false, printing the reason and the
  number of missing seeds. The script's semantics are untouched - its own tests
  still pin the provisional PASS/exit 0 - because the ambiguity is in the
  driver's contract about the *run*, not in the analysis.
- **The driver reports whether a session's pairs survived.** The exit publish
  is the only durable copy, so its outcome is the number that matters in an
  unattended multi-hour drive - and it was previously visible only to someone
  who downloaded the kernel output by hand. The driver now reads
  `remote-publish-log.json` (which `mode_session` keeps beside the session log)
  after every session, prints the status and seed count, and stops when it says
  `failed`: re-training a session's pairs to re-attempt the push costs a whole
  window, and the usual cause is transient, which is why the push already
  retried. The name is pinned against `kaggle_remote.PUBLISH_LOG_NAME` by a
  test, because a drifted constant would read as "nothing published".
- The whole run is now reproducible code, not operator lore:
  `scripts/kaggle_t4.py` drives every session kind as a script kernel
  (`ACT_T4_MODE` = `metadata` / `probe` / `preflight` / `session` / `control` /
  `null`), generates the push metadata, copies the attached code dataset into
  the writable working dir, re-runs the benchmark in-session (a fresh session
  has no `artifacts/`, and the launch command refuses a stale artifact), runs
  the registered command unchanged, and keeps the session/publish logs in the
  kernel output. It absorbs two Kaggle behaviours that each cost a failed
  run: attached datasets mount **nested** (`/kaggle/input/datasets/<owner>/<slug>/`
  on the current runtime, so the code root is found by searching the mount
  tree for `pyproject.toml` + `src/act` rather than assuming a depth), and
  `gpu_type` is **not** an accelerator pin - only `machine_shape`
  (`NvidiaTeslaT4`) is honoured, so the previous `gpu_type: t4` push ran
  whatever Kaggle assigned. Pinned by 11 tests in `tests/test_kaggle_t4.py`
  (every mount layout, the exact per-queue launch argv for sections 2/2b/2c,
  disjoint datasets/roots, and the metadata key set).
- **The Kaggle dataset API authenticates inside a kernel without secrets**
  (probed 2026-09-15 as `ACT_T4_MODE=probe`): `KaggleApi().authenticate()`
  succeeds from the session's own identity with no `KAGGLE_USERNAME`,
  `KAGGLE_KEY`, or `kaggle.json`, and `dataset_list(mine=True)` returns the
  account's datasets. So the section 2 restore/publish legs need no User
  Secrets - only `enable_internet: true`, which the generated metadata sets.
- **Quota plan for the section 5.1 queues** (measured, not assumed). Superseded
  twice, and the corrected figure is ~47 GPU-hours against the free tier's
  30 GPU-h/week - about two weeks of sessions, not three. First the 3.585 h
  pair time came from the fp32 rate the dtype *rule* reads while the arms train
  fp16 (2.27x pessimistic), then it charged two waves of arms where one arm per
  device is available on the two-T4 shape (2x). At the measured fp16 rate with
  one arm per device a pair is 1.36 h, so each of the three queues is ~16
  GPU-hours (12 pairs) and two sessions each at `--hours 9` (the registered
  usable window) place six completed pairs. Both queues and the null queue bill
  only the time they actually train: the deadline rules stop a session early
  rather than fill the window.
- **The section 4 verdict is reachable after a Kaggle run, and its exit codes
  are now unambiguous.** That command reads *local* dataset roots, which are
  empty once the queues live in published Kaggle datasets, so
  `ACT_T4_MODE=verdict` restores all three queues (each session publishes the
  whole handoff directory, so the newest published version is cumulative) and
  then runs the registered verdict, keeping the verdict JSON in the kernel
  output; it is a CPU kernel, so it never queues behind a GPU. Auditing that
  path exposed a real defect: `scripts/v0_verdict.py` exited **1** both for
  the pre-registered `FAIL` (primary cost > 5%) and for every
  input-validation error (`raise SystemExit("...")` defaults to 1), so over
  an unattended multi-session run a wrong `--dataset-root` would have been
  recorded as "the amortization hypothesis is false". Validation errors now
  raise `InputError` and exit the distinct code **3**, writing no verdict;
  0/1/2 keep their registered meanings exactly. Verified end-to-end: a
  nonexistent root exits 3 with `no verdict produced: no handoff dataset
  under ...` and no JSON written, while the three-queue command composes over
  the reduced-fidelity `real-diag` roots (`rich` / `argmax` / `null`) and
  writes a coherent verdict. Pinned by 2 new verdict tests and a driver test
  asserting exit 3 is not treated as a verdict code.
- **The re-push loop is mechanised** (`scripts/kaggle_queue.py`), because the
  three queues are ~47 GPU-h against a 30 GPU-h week (two sessions each at the
  measured rate, with one arm per device) and "re-run the same command until
  the queue completes" should not depend on someone remembering - nor on
  someone remembering to stop when the week's quota is nearly gone. It
  stages the kernel, pushes it, waits for a terminal status, reads the
  session's own log, and re-pushes until the declared block is complete (exit
  0 complete, 1 session failed, 2 session budget exhausted). Progress comes
  from the log, not the dataset: the runner labels every declared seed
  `skipped_complete` or `completed` as it walks the queue, so the count is
  cumulative and costs no multi-gigabyte download per session. Staging also
  encodes a trap that would have silently wasted a session - a Kaggle script
  kernel receives *only* `code_file` (the CLI sends that file's text and
  nothing else from the push folder, and the metadata has no environment
  field), so the staged entry script is generated with the queue mode pinned
  into its `__main__` guard, and staging fails loudly if that guard moves.
  Verified by executing a staged entry with no `ACT_T4_MODE` set: it resolves
  its pinned mode and reaches the mount check instead of the mode error.
  Pinned by 16 tests.
- **Disk budget: the run as documented could not have completed.** Measured on
  the T4 runtime (2026-09-15, `ACT_T4_MODE=probe`): `/kaggle/working` is a
  20 GB loopback mount (19.5 GiB usable) while the container's root overlay has
  7.9 TB with 1.1 TB free. A completed seed publishes its final checkpoint per
  arm *including the AdamW moments* (`save_v0_checkpoint` stores `model` plus
  `optimizer` state, ~3.19 GiB per arm), so ~6.4 GiB per seed and ~77 GiB over
  the declared block - against a 19.5 GiB working mount, and the section 5.2
  ledger (14.9 GiB steady state + 5.1 GiB headroom) accounts for the *session*
  alone, not the accumulating dataset root. A session would therefore have hit
  `ENOSPC` during its second pair, and because the per-pair local publish is
  called unguarded the kernel would have aborted *before* the exit publish -
  publishing nothing, despite a completed pair. Fixed by pointing the handoff
  roots at the overlay (`/tmp/act-lab-<mode>/artifacts/...`, mode-scoped because
  the overlay is host-shared) instead of the working directory: nothing in
  those roots needs to survive a session, since the published dataset is
  restored at session start and pushed again at exit, and the 20 GB working
  directory is left holding only the code copy and the small audit files.
  `report_resources()` now logs disk headroom and the handoff-root size at the
  start and end of every session, so the growth is measured rather than
  assumed.
- **A failed local handoff publish no longer discards completed work.** The
  exit publish never raises by design (a broken push must not destroy a
  session); the per-pair local publish needed the mirror-image guarantee, since
  it was called unguarded and any failure aborted the kernel before the exit
  publish could push the seeds that *had* completed. A failure is now recorded
  as `action: publish_failed` (never `completed`, so skip-completed and the
  queue accounting stay honest), it stops the queue rather than spending more
  GPU on a substrate that cannot publish, and the session exit-publishes
  everything already finished; the failed pair restarts from scratch next
  session. Pinned by a runner test that fails the second publish and asserts
  the first seed stays `complete`, the second does not, and the third is never
  attempted.
- **The overlay work root is collision-proof and self-cleaning.** Because the
  root lives on a host-shared overlay, it is scoped by mode *and* by run:
  `KAGGLE_CONTAINER_NAME` carries the run identity (probed 2026-09-15:
  `kaggle_<hash>-<number>-webtier`), hashed into the directory name. The queues
  are driven one session at a time by design - quota, not parallelism, is the
  constraint - so this only matters when an overlap happens by accident, and it
  makes that harmless rather than corrupting a directory another kernel is
  using. The root is deleted at session end once the exit publish reports
  anything other than `failed`, and kept when it fails (the overlay copy is
  then the only copy of that session's completions); outside a kernel the root
  is the repository itself, so the reclaim is a hard no-op there and is tested
  to be one.
- **Pushing a kernel does not supersede a running session** (measured
  2026-09-15): a second `kaggle kernels push` left the first session running,
  and the burn rate against the weekly quota line went 1.00x -> 2.00x while
  only the latest version is reported by `kernels status`. This was hit for
  real: a corrected session was pushed over a running one and both burned GPU
  for ~15 min (~0.8 GPU-h) before it was caught by sampling the quota line
  twice and differencing it - a burn rate is the only way to see a second
  session. `kaggle kernels delete <owner>/<slug> --yes` does stop the runs
  (verified: burn fell to 0.00x), but it stops every version, so recovery is
  delete-then-push-once; stopping a single version is impossible (the CLI's
  versioned form raises in argument parsing and the API has no version
  parameter). Prevented in code now: `kaggle_queue.should_adopt` makes the
  driver adopt a live session instead of adding a second, pinned by a test.
- **The measured rate was not tied to a device count, and that is now fixed.**
  Pre-flight logged `nvidia-smi` as `stdout[-800:]`, which is the *processes*
  table, so the GPU inventory table never reached the log; `machine_shape`
  cannot settle it either (free-form string, no SDK enum, and the account's own
  "T4 x2" notebook declares the same `NvidiaTeslaT4`), and training reaches the
  GPU only through `resolve_device("auto")` - plain `cuda`, i.e. device 0 - so
  a second device would be idle and invisible. `act.training.runtime
  .gpu_inventory()` now reports every visible device (`count` + `names`); the
  benchmark writes it into `artifacts/v0-benchmark.json` beside the rate and
  pre-flight logs it, so the next session settles how many T4s the shape hands
  out. It deliberately reports only host facts, not which device trained, so
  it cannot contradict the artifact's own `device` (a `--device cpu` smoke run
  on a CUDA host). If `count` is 2, the registered pairs are two independent
  arms and could train one per device for the same tokens - a *schedule*
  change (re-derives `full_pairs_per_session` from a re-run section 5.2
  benchmark), not an assumption. Pinned by 2 tests in
  `tests/test_training_distributed.py`.
- **The queue driver is quota-aware, and `--hours` now actually reaches the
  kernel.** `kaggle_queue.py` reads `kaggle quota --format json` before each
  push and refuses when remaining GPU hours are below the declared window
  (exit 3), because a session cut off by quota still bills every hour it used
  and produces nothing - the dominant failure mode of a ~47 GPU-hour plan
  against a 30 GPU-h/week line. An unreadable quota line never blocks (it can
  only spend less), and a live session is adopted rather than gated. Fixing
  this exposed the flag it depends on: `--hours` set `ACT_T4_HOURS` in the
  *driver's* process, which a script kernel never sees, so the kernel silently
  used its own default (masked only because that default is also 9). The
  window is now pinned into the staged kernel body beside the mode, so the
  gate and the session compare the same number. Pinned by 8 tests in
  `tests/test_kaggle_queue.py` (quota parsing incl. the TPU row and
  cannot-tell cases, the gate's boundary, a drive-level refusal that asserts
  no push happened, and a positive control that a coverable week still
  pushes).
- **The documented verdict command works, and the section 4 path is verified
  end to end.** `T4_RUNBOOK.md` section 3 told operators to run
  `kaggle_queue.py --queue verdict`, but `--queue` only accepted the three
  queues, so the command that produces the final answer failed with an
  argparse usage error - exactly once, after the whole quota was spent.
  `--queue verdict` (and `--queue preflight`) are now one-shot jobs: staged,
  pushed once, waited on, and reported, with no re-push loop and no quota gate.
  The verdict's registered code is mirrored as the driver's exit code (0 PASS,
  1 FAIL, 2 PASS with qualification, 3 no verdict), and to make that possible
  the decision now travels in `v0-5-1-verdict.json` as `exit_code` - the
  Kaggle-side runner exits 0 whatever the decision, so a reader could
  otherwise only re-derive PASS/FAIL from the verdict string.
  Verified by running the real thing: three full 12-seed queues produced by
  `run_pair_session` with the smoke profile, then `v0_verdict.py` over them -
  `queue_complete` true, 12 seeds complete, no strays, the pool-matched
  control and null blocks both evaluated, `measured_cost_percent` 52.2% (far
  outside the 5% criterion for a tiny config, hence the FAIL), artifact
  `exit_code` 1 matching the process exit code 1. The reader/writer coupling
  is now pinned by a test, because the existing verdict tests hand-build their
  summaries and never fed the script a summary the runner wrote.
- **The remote publish round trip is now pinned, not just asserted.** The
  adapter's docstring records that the upload layout was established by live
  probes because the *naive* upload silently produced an empty dataset and
  per-directory tarring flattened every seed to the archive root where they
  collided - but the staging that fixes it had **no test coverage at all**, and
  its failure mode is a corrupt dataset that still reports `published`. (That
  is now caught twice over: the driver's no-progress guard would also fire,
  because a later session would re-train seeds that were never really
  published.) `stage_push` is now a module-level function free of the API
  client, so it is testable without authenticating, and a test drives the
  whole joint - stage, upload, service-side extraction, download, merge - into
  a fresh root, asserting one archive per seed rooted at `seed-NNNN/`, both
  seeds restored, and every checksum from the pre-upload manifest verified
  through `restore_seed`. Confirmed to have teeth: dropping the arcname prefix
  reproduces the collision and fails the test.
- **The exit publish is retried, and can no longer raise on its own first
  step.** Two defects in the one push whose failure discards a session: the
  metadata write ran *before* the `try`, so a full disk (it writes into the
  handoff folder) made the "never raises" exit path raise - turning a finished
  session into a crashed one; and the push itself was attempted exactly once,
  so a transient API or network error was recorded as a failure. The push is
  now attempted up to `PUBLISH_ATTEMPTS` (3) with backoff, inside the same
  guard as the metadata write, and the recorded error names every attempt so a
  deterministic failure is distinguishable from a flaky one. Also corrected
  the docs' claim that a later session re-pushes a failed container's handoff:
  the handoff roots live in the container overlay, so a failed exit publish
  loses that session's pairs with no local copy left to push.
- **The driver refuses to repeat a session that advanced nothing.** A session
  whose clock never fit a pair reports *success* while writing no progress
  (rule 1 idles it), and the loop would push again - up to `--max-sessions`
  times, each repeat billing quota for the same nothing. The driver now
  compares cumulative completed seeds across sessions and stops with exit 4,
  reporting the quota the fruitless session consumed (differencing the quota
  line, the same measurement that caught two sessions running at once). This
  is the driver's own version of the burn it exists to prevent. Pinned by 3
  tests in `tests/test_kaggle_queue.py`, including a positive control that a
  session which *does* advance keeps the loop going.
- **The kernel runs the mounted code dataset, and nothing checked it.** The
  staged kernel body is embedded from the working tree, but `src/`, `scripts/`
  and `tests/` are copied from the attached `act-lab-code` dataset, so a lagging
  dataset silently executes old training code while the handoff manifest
  records a `code_revision` that did not run. Measured 2026-09-15: the dataset
  was last published at 13:14, a commit behind everything fixed afterwards - so
  the running session did not contain `c84b27b`'s guarded handoff publish, and
  2040ab1's `gpu_inventory` import would have crashed the next session against
  the older dataset. `kaggle_queue.ensure_code_dataset` runs before any push:
  it pins the dataset to `git rev-parse HEAD`, refuses a dirty or non-git tree
  (no reproducible revision), republishes when the dataset's `revision.txt`
  marker differs, and is checked before quota is spent rather than after.
  The staged dataset is the tracked Python side only - deliberately excluding
  `artifacts/`, because a session must start with no `artifacts/` directory so
  the launch command cannot read a benchmark artifact it did not measure, and
  excluding the web side (`public/`, `server/`, vite/ts configs) which the
  kernel never imports. Two follow-on defects found and fixed 2026-09-15: the
  tree is uploaded as one archive because Kaggle's uploader silently drops
  every subdirectory (a folder-shaped push published a dataset with no `src/`
  while still reporting success), and the revision marker is now written into
  the *staging* folder rather than the repo - a marker left at the repo root is
  an untracked file, so the publish refused the very next publish. The
  post-publish read-back is polled: a download issued right after a version
  push can serve the previous version, which is indistinguishable from a
  rejected push. Pinned by the code-dataset tests in
  `tests/test_kaggle_queue.py`, including that `artifacts/` never ships, that
  the archive carries the package tree, and that staging leaves the source tree
  untouched.
- **The kernel could not import its own code.** `kaggle_t4._bootstrap()`
  copies the mounted dataset into the writable working dir and returns an env
  dict with `PYTHONPATH=<work>/src` - but that dict only ever configured the
  *subprocesses* each mode launches. The pre-flight, the dtype probe and the
  verdict import `act` in this very process right after bootstrapping, so they
  died on `ModuleNotFoundError: No module named 'act'` (measured 2026-09-15,
  the dtype probe, 8 s in). The pre-flight and the verdict had it from the
  start; `2040ab1` added the same in-process import to the session mode, which
  would have killed every session pushed after it. `_bootstrap` now puts the
  copied `src/` on `sys.path` as well, and a test imports a module that only
  exists in the fake mount to prove the copied tree is importable in-process.
  The session that is running predates the session-mode import, which is why it
  is unaffected.
- **The session pipeline is proven end to end, not just in pieces.** The
  registered launch command was run at reduced fidelity on CPU
  (`--profile --seeds 15000-15001`) and its own `session-log.json` fed back
  through the driver: a fresh session logs `completed` per pair, a re-run
  against the published version logs `skipped_complete`, and both read as
  `(2, 12)` cumulative progress. That round trip is what the driver's
  "is this queue finished?" decision rests on - if a resumed session did not
  count, the driver would re-push sessions with nothing left to do - and it
  also exercises the new overlay dataset root, the atomic publish (a real
  `seed-*/manifests/manifest.json` and checksum-verified restore), and the
  resume lookup. Pinned by 1 integration test in `tests/test_kaggle_queue.py`
  (plus the 12 hand-built `queue_progress` cases).
- **Every kernel-side environment knob is now pinned into the staged body.**
  A script kernel receives only `code_file`, so anything set on the pushing
  side never arrives, and the kernel falls back to its own default. Three
  instances existed and all three only appeared to work because the default
  matched: the mode (fixed earlier), `--hours`, and - the one that would have
  been destructive - `ACT_T4_OWNER`/`ACT_T4_SLUG`, which `Queue.dataset`
  builds *inside the kernel*, so an override would have published the session
  to the default dataset while the driver believed otherwise. Staging now
  emits all of them from one helper (`pinned_environment`), and refuses if any
  of the constants is renamed in the body. Verified by executing the staged
  guard with the driver's environment set and the kernel's cleared: it
  resolves `someone-else/other-dataset` rather than the default. Pinned by a
  test that stages under an override and asserts the metadata agrees.
- `PYTHONPATH=src python -m pytest -q` -> all unit tests pass (272, including the
  26 report-figure data-assembly tests, 14 session-plan tests pinning the
  section 5.2 schedule/dtype tables, 10 handoff tests covering atomic
  publish, checksum rejection, and bit-identical resume through the dataset
  round trip, 9 runner tests covering deadline gating, cooperative
  interrupt + in-progress snapshot, restart-from-scratch, and
  skip-completed idempotence, 22 notebook-builder tests pinning determinism,
  nbformat validity, and the section 104 claims, 18 verdict tests pinning
  every branch of the section 5.1 decision rules (the 5% criterion, the
  [0.5, 2]x tolerance band, the 10-of-12 harm guard and its label
  precedence over a band miss, queue completeness
  gating `gate_4_authorized`, stray-seed exclusion, unmatched-budget and
  missing-arm aborts, and the verdict read through the real publish
  pipeline), 9 real-data session tests (window sampling, vocab guard,
  config validation, real-data training below the synthetic floor,
  bit-identical real-data resume, and the publish round trip), and fp16/CLI tests covering
  the fp16 session
  round trip, the version-4 compatibility path, and the pair-session launch
  command's artifact/dtype guards); session runs are deliberately excluded
  from the suite (they are integration checks, too slow for per-commit runs) and
  live in `scripts/v0_session_selfcheck.py`, which verifies matched budgets,
  FLOPs booking, bit-identical mid-run resume, and the version-4 checkpoint
  contract on a tiny config.
- `PYTHONPATH=src python scripts/build_notebooks.py` -> writes the ten
  notebooks (00-09) byte-identically; `--check` verifies the committed
  `.ipynb` files never drift from the builder output; every code cell of
  all ten notebooks was executed in order (cell-runner pattern, like the
  self-check): 00 reproduced the declared parameter counts, 01 proved
  train/resume/seed-reproducibility, 02 proved grow/rebuild/prune weight
  and hash round trips plus the registry append, 03 and 09 ran 4-seed
  gate studies to their recorded verdicts (demo-sized, not the powered
  384-seed studies), 04 extracted signals after a real backward pass and
  verified ablation restores parameters, 05 ran the three-arm matched-
  budget experiment with inspectable mutation histories, 06 exercised
  variants/prior ranking/frontier/Pareto selection, 07 froze and published
  the evolved genome through the checksummed manifest, and 08 compared
  fresh/rewound/teacher initializations with exact copied/skipped counts.
- `PYTHONPATH=src python -W error::UserWarning -c ...` ablation check ->
  the temporary-ablation helper now detaches losses before `float()`
  conversion (no tensor-scalar warning); covered by the notebook cell
  executions above.
- `PYTHONPATH=src python scripts/v0_session_selfcheck.py` -> re-passed after the
  declared sessions switched to checkpointed activations (resume stays
  bit-identical; recompute is deterministic because blocks carry no dropout),
  and after the fp16 path landed (step [3/4] verifies fp16 resume is
  bit-identical with scaler state restored, and that fp16 books the same
  FLOPs counters as fp32).
- `PYTHONPATH=src python scripts/v0_session_selfcheck.py --device cuda` ->
  **the declared pairing now has coverage.** Every fp16 check above ran on CPU
  (where `GradScaler("cpu")` is the native configuration) and the documented
  local-GPU pair-session run was fp32, so nothing had ever trained fp16 on
  CUDA - while the declared run does exactly that, and every session after the
  first resumes a CUDA fp16 checkpoint. The self-check now takes `--device`
  (a module constant, so no call site can compare a checkpoint from one device
  against a resume on another and pass while testing nothing) and the whole
  battery passes on CUDA: fp32 and fp16 resumes both bit-identical, fp16 FLOPs
  counters unchanged, checkpoint contract intact. The hardcoded
  `GradScaler("cpu")` in `run_v0_session` was also probed directly and is
  behaviourally identical to `GradScaler("cuda")` on CUDA tensors - infs
  detected, step skipped, scale backed off 65536 -> 32768 - so it is not a
  defect and was deliberately left alone rather than changing the numeric path
  mid-run.
- `PYTHONPATH=src python -m act.training.cli --mode pair-session --profile ...`
  -> the launch command runs the whole chain end-to-end on the local GPU:
  it consumed a real benchmark artifact (smoke profile, 26,987 tokens/s,
  fp32 decision), gated and completed all 12 declared seed pairs, published
  each to the dataset handoff, and wrote the session log; the stale-artifact
  and dtype-override guards were exercised by unit tests.
- `PYTHONPATH=src python scripts/v0_throughput_benchmark.py` -> the declared-model
  benchmark runs end-to-end on the local GPU (285,336,576 / 134,341,632 exact
  parameters, ~1,736 tokens/s with activation checkpointing -> dtype rule
  selects fp16 below the 2,500 floor) and writes `artifacts/v0-benchmark.json`;
  `--profile` smoke-checks on tiny configs or CPU. The T4 decision comes from
  the T4 run.
- `PYTHONPATH=src python scripts/local_proof_run.py` -> the local rehearsal of
  the full protocol loop: benchmarks the reduced model (12.55M/6.26M params,
  ~15.2k tokens/s on the local 1070, fp32 above the floor), then runs the
  production `run_pair_session` over the declared queue 9000-9011 (atomic
  pairs, skip-completed, deadline rules) into the local-proof dataset root and
  closes with `scripts/v0_verdict.py`. Completed 2026-09-13: 12/12 pairs,
  PASS_WITH_QUALIFICATION (cost 1.0044% <= 5%; declared band correctly not
  met by a reduced model; harm guard clear at 7/12). Re-runnable; `--dry-run`
  prints the plan without training.
- `PYTHONPATH=src python scripts/local_proof_run.py --real-data` -> the
  section 5.1 protocol machinery on the REAL TinyStories token stream (vocab
  8192 streams built by `train_real_lm.py`; `V0SessionConfig.data_path`/
  `val_data_path` route training, probe, and validation draws through random
  windows into the streams - shapes, generators, checkpointed state, and
  FLOPs booking identical to the synthetic path, so the identity and the
  bit-identical resume contract hold; unit-pinned in `tests/test_v0_real_data.py`).
  Completed 2026-09-14, 12/12 pairs: verdict
  PASS_COST_ONLY_HARM_GUARD_REJECTS - the cost identity held again (measured
  1.0077% vs 0.9977% predicted, <1% relative deviation), arm losses were real
  language losses (~3.4-3.6 nats vs the 9.01 synthetic floor), but the
  ADAPTIVE LOST 11 OF 12 PAIRS on real text at this scale: single-batch
  probe-loss argmin is anti-adaptive here,  the study-5 winner's-curse
  mechanism re-emerging on real language structure (adaptive and random arms
  chose the same operation sequences - only the chosen experts differed, so
  the probe's per-candidate ranking is what misleads). The machine-checked
  harm guard engaged exactly as pre-registered and the verdict label now
  takes the harm-guard precedence over a band miss (unit-pinned). This is a
  real-data negative result the toy studies could not show: selection value
  on real text needs richer probes or longer horizons, and Gate 4 stays
  blocked. Scratch roots `artifacts/real-proof/{sessions,datasets}`; verdict
  and session summary committed as evidence. (The follow-up diagnostic below
  revises the "richer probes" reading: neither richer probe batches nor a
  richer candidate set fixed it.)
- `PYTHONPATH=src python scripts/real_data_diagnostic.py` -> the selection
  diagnostic following the real-data harm-guard rejection: both toy-winning
  axes re-tested on the real TinyStories stream over the same paired seed
  block 9000-9007 (steps=600, production `run_pair_session`, 8 pairs per
  variant, ~5.2 min/pair on the local GPU). `var` (cap-4/4-batch; measured
  probe cost 3.99% of training FLOPs) 0/8 adaptive wins, mean delta vs own
  random `+0.2858`; `rich` (cap-8/1-batch; 1.98%) 1/8, mean delta
  `+0.1661`, mean adaptive vs baseline adaptive `-0.1094` (an absolute
  improvement over the baseline adaptive arm, still a relative loss). In
  5/8 seeds the 4-batch argmin chose bit-identically to the baseline
  1-batch argmin (final-loss delta exactly `0.0`), and every variant's
  random arm replicated its baseline counterpart - only selection differed.
  Conclusion: on real text the probe ranking is stably misaligned rather
  than noisy - variance (study 6) is excluded as the mechanism and richness
  (studies 8/9) does not transfer; next discriminator is an argmax control.
  Report: `artifacts/real-diag/diagnostic.json` (run log alongside);
  diagnostic only, not a pre-registered gate, 8 seeds unpowered.
- `PYTHONPATH=src python scripts/real_data_diagnostic.py --variants argmax`
  -> the pre-registered argmax control from the diagnostic's
  stably-misaligned finding, now code and run: `V0SessionConfig` gained
  `probe_selection` (`"argmin"` default | `"argmax"`), flipping only the
  decision rule in `_adaptive_event` - same candidates, same probe batches,
  same booked probe FLOPs (measured 1.01% vs baseline ~1.0%), and the
  mutation-history policy string (`adaptive_probe_argmin`/`..._argmax`)
  audits which rule produced each event. On the same seeds 9000-9007:
  argmax 0/8 adaptive wins, mean delta vs own random `+0.3048` (argmin
  baseline: `+0.2858`), mean adaptive vs baseline adaptive `+0.0293`;
  non-degenerate (0/3 identical picks to argmin on seed 9000) with the
  operation balance inverted (argmin chose 20 prune/4 grow, argmax
  11 prune/13 grow). Both tails of the probe ranking lose to random by
  nearly the same amount -> the probe signal is uninformative on real
  text, not inverted; per the discriminator contract the fix must come
  from probe horizon/tokens, not the decision rule. Full suite: 274
  passed (2 new argmax tests pin validation and the inverted-decision
  path). Report: `artifacts/real-diag/diagnostic.json` now carries all
  three variants; the empty `run-argmax.log` from a detached-launch
  failure was removed (the argmax run executed in the foreground).
- `PYTHONPATH=src python scripts/real_data_diagnostic.py --force` -> per-
  candidate probe ranks are now logged natively: `V0SessionConfig` gained a
  `probe_landscape` flag (bookkeeping only; off by default, mirroring the
  toy path's instrumentation; three new tests pin the sorted-ascending
  landscape, the argmax tail selection, and the off-by-default contract).
  The diagnostic enables it for every variant and aggregates a per-variant
  `probe_rank` block into `diagnostic.json` (event count, mean selected
  rank, rank histogram, near-top rate), so mechanism claims are directly
  measurable from the published seed summaries. Full suite: 277 passed.
  Re-running requires `--force` (the runner skips completed pairs; previous
  results are moved aside with a timestamp suffix). Existing artifacts
  pre-date the instrumentation and report `events_with_landscape: 0`.
- `PYTHONPATH=src python scripts/real_data_diagnostic.py --variants heldout`
  -> RUN 2026-09-14 (with `long`, revision `eaa1c45`, 8 pairs, ~1.2 h): the
  section-5.3 post-argmax fix candidate. `V0SessionConfig` gained
  `probe_data_source` ("train" default | "heldout"), pointing the probe
  windows at the held-out validation stream (same dedicated generator, same
  draw discipline, so the bit-identical resume contract holds; validation
  fail-fast requires `val_data_path` for heldout). Only the probe windows'
  source stream differs from the baseline adaptive arm - candidates, probe
  batch count, and booked probe FLOPs are baseline-matched (measured 1.01%).
  Four tests pin config validation, the real stream switch at the first
  event, cost parity, and bit-identical resume (281 at design time).
  Outcome: heldout 0/8 adaptive wins, mean delta vs own random `+0.2840`,
  random arms replicate - the pre-registered provenance hypothesis is
  FALSIFIED (outcome b) with the manipulation verified real (selected probe
  losses shifted 4.4195 -> 4.2848; 9/24 picks changed, more than `var`'s
  extra batches moved).
- `PYTHONPATH=src python scripts/real_data_diagnostic.py --variants long
  --force` -> DESIGNED, pre-registered, and RUN 2026-09-14 (same block,
  revision `eaa1c45`): the long-horizon probe as a one-step lookahead.
  `V0SessionConfig.probe_finetune_steps` (0 default | R > 0) fine-tunes
  each probed candidate for R AdamW steps on freshly drawn training batches
  (fresh optimizer, per-event throwaway generators, bit-identical resume
  pinned by test); search FLOPs book the adaptation on the full probed
  model (measured 3.00%). Outcome: 0/8 adaptive wins, mean delta vs own
  random `+0.3106` - the adaptation-blindness hypothesis is FALSIFIED
  (outcome b). Mechanism evidence: landscapes re-ranked after adaptation,
  the operation balance relaxed (5/24 grows) and seed 9000's probe
  top-ranked the grow at 2 of 3 events - the baseline's 4/20 prune bias is
  a snapshot-probe ranking artifact - yet per-seed deltas stayed in the
  harm band, and mean selected probe loss correlates POSITIVELY with final
  harm across seeds (long +0.51, heldout +0.89; 8 points, indicative).
  Both scorecards, the unified five-variant verdict, and the next-direction
  note (change how the ranking is used; own pre-registration required) are
  recorded in report section 5.3. Full suite: 285 passed.
- `PYTHONPATH=src python scripts/real_data_diagnostic.py --variants null`
  -> RAN 2026-09-14 (revision `9e5e448`; pre-registered in report section
  5.8, outcome in the same section): the do-nothing control arm
  (`V0SessionConfig.arms=("null",)`) - single-arm like gated, sharing the
  random arm's init identity and training stream, recording each event as
  a no-op, booking zero probe FLOPs. Result: **outcome (a) - churn harm is
  REAL**: null 8/8 wins vs the baseline random arms, mean delta `-0.1217`
  (per-seed -0.08 to -0.20, all negative). Built-in checks held: zero
  mutations / zero search FLOPs in 8/8 seeds, and the replication check is
  EXACT - max |null - gated| = 0.000000 on all eight seeds, the null arm
  reproducing the gated run bit-for-bit as the section-5.7 degeneracy
  reading predicted. Consequences: every section-5.3 variant's loss is
  partly churn harm; the theta sweep's control should be the null arm ("do
  accepted draws beat doing nothing?"); the honest real-text ladder for
  Gate 4 is null < random < adaptive. Verdict machine-checked by
  `scripts/null_verdict.py`. Full suite: 338 passed (6 null tests).: accept/reject gating of random draws - the
  probe's ABSOLUTE signal (does this single mutation hurt the current
  model?) as a per-event gate on the random arm's draw, abandoning the
  anti-predictive ordering entirely. Result: **8/8 wins, mean delta vs own
  random `-0.1217`** (adaptive arm `-0.3972` from the baseline adaptive
  arm; first negative real-text delta of any treatment) at 0.50% probe
  cost - but degenerate: accept rate 0/24, all 24 draws harmful by probe
  (mean draw-minus-reference `+0.1065`), all 24 decisions `reject_noop`
  (the load-fallback's `< 8` guard cannot fire at `expert_count = 8`),
  `mutation_count` 0 in 8/8 seeds, and draws bit-identical to the random
  arms' own operations (24/24 - the built-in control held). The win belongs
  to the NULL arm - zero matched-budget churn beats random churn on real
  text - not to gate discrimination: outcome (a)'s letter fired, its
  content ("absolute-harm gating carries real signal") was never exercised.
  First direct measurement of real-text churn harm, and a partial
  explanation of every earlier variant's loss (all spent the full mutation
  budget). Follow-ups need their own pre-registration: a do-nothing control
  arm (now pre-registered as report section 5.8 and implemented as
  `V0SessionConfig.arms=("null",)` - RAN, outcome (a), see below), and an
  intermediate-theta sweep (theta `0.05`/`0.1` would have
  accepted 11/24 and 21/24 of the observed draws). Single-arm sessions
  compared against the baseline block's stored random arms; per-event
  `gate_decision` records and both probe losses made the mechanism
  measurable as designed. Full suite: 332 passed.
- **Section 5.11 theta sweep (RAN 2026-09-14, revision `9bd9323`)**: the
  intermediate-theta sweep against the null arm on the fresh block
  9008-9017 (10 seeds; `scripts/theta_sweep.py`: the null control plus
  `gated` at theta {0.02, 0.05, 0.1}). Family verdict: **NO theta
  clears** (0/10 wins at every theta, p 1.0, bar 9/10 at Bonferroni
  alpha/3) - the pre-registered falsification branch fires and the
  probe-on-real-text program is CLOSED at this fidelity. Mean
  gated-minus-null delta is monotonically harmful in acceptance rate:
  +0.0027 at 6.7% accepted, +0.0236 at 43.3%, +0.0713 at 76.7% - a clean
  dose-response of churn harm. Mechanism verified from the published
  seed summaries: seeds accepting nothing are bit-identical to the null
  arm (8/10 at theta 0.02), and identical accept/reject sequences across
  thetas give bit-identical losses (9008/9015/9017) - the sweep measured
  exactly the cost of accepted churn. Secondary: the published random
  arm at 9008 (loss 4.1279 vs block-mates ~3.3) is itself a churn
  catastrophe of the section-5.8 kind. Verdict committed at
  `artifacts/real-diag/theta-sweep/verdict.json`. Next direction
  pre-registered as report section 5.12: the depth x candidate-policy
  2x2 (the 5.9-vs-5.10 contrast varies two registered fields at once).
  Full suite: 349 passed.
- **Section 5.12 pre-registration (designed, not yet run)**: the depth x
  candidate-policy 2x2 completes the factorial behind the 5.9/5.10
  contrast - depth 1 + spread (fresh block 72000-72383) and depth 4 +
  prefix (fresh block 73000-73383), same three arms and the same
  primary contract, executed as new profiles (`depth1-spread`,
  `depth4-prefix`) of `scripts/null_toy_study.py` with committed
  summaries and pinning tests like 5.9/5.10.
- **Section 5.12 2x2 (RAN 2026-09-14/15, revision `142ad44`)**: both
  cells completed 384/384 seeds, built-in checks held, verdicts
  machine-checked, summaries committed and pinned by test (suite at
  355). Result: **depth, not the candidate policy, separates churn
  suppression from value creation.** depth 1 + spread: primary 1 PASS
  (289/384), primary 2 FAIL (131/384, +0.0475) - replicates 5.9;
  depth 4 + prefix: primary 1 PASS (266/384), primary 2 PASS (225/384,
  -0.0371, p 4.4e-04) - replicates 5.10. The depth-4 edge does not
  require the spread policy and spread alone does nothing at depth 1;
  the 5.10 "richness axis" attribution is revised to depth-driven, and
  the candidate policy is exchangeable within the registered
  cap-4/2-batch budget.
- **Section 5.13 pre-registration (designed, not yet run)**: the
  routing-statistics heuristic arm - the section-5.11 contract's named
  non-probe direction: deterministic grow at the argmax-routing-load
  layer every event, no chooser draw, no probes, zero search FLOPs
  beyond the rebuild. Implemented as `V0SessionConfig(arms=("heuristic",))`
  (single-arm, shares the random arm's init identity), wired into the
  diagnostic as the `heuristic` variant, 7 new tests (suite at 362).
  Runs on the theta sweep's block 9008-9017 against its published null
  arms; primary: sign test vs null at Bonferroni alpha/2 (bar 9/10).
- **Section 5.13 outcome (RAN 2026-09-15, revision `0ef29de`)**: the
  heuristic LOSES to doing nothing on every seed - 0/10 wins, mean
  delta +0.1218 (per-seed +0.049 to +0.410, all positive), p 1.0.
  Built-in checks held: 30/30 grows at the measured argmax-load layer
  (L5 x10, L7 x6, L4 x5, L0 x4, L6 x3, L3 x2), rebuild-only search
  FLOPs. **The real-data topology program is CLOSED**: probe-ordered
  (5.3), probe-gated (5.4/5.11), and load-directed (5.13) churn all
  lose to doing nothing; the null arm is the only winning policy on
  real text at this fidelity. The project's one remaining open gate is
  Gate 4 (the section-5.5 compute-efficiency protocol, block
  15000-15011, unspent).
- **Section 5.15 pre-registered (2026-09-15, revision `1e4528d`)**: the
  **depth dose-response** - the two missing cells (depth 2 on fresh block
  74000-74383, depth 3 on 75000-75383, prefix policy) that locate where
  value creation onset begins on the depth axis the 5.12 2x2 identified.
  Same three-arm machinery, same primaries, Bonferroni alpha/2; four
  pre-registered branches (monotone onset between 1-2 / 3-4 / 2-3, or a
  non-monotone response suspending the depth-axis reading).
- **Section 5.15 outcome (RAN 2026-09-15, revision `bbad439`)**: branch
  (b) - onset between depth 3 and 4. Depth 2: adaptive vs null 161/384
  (+0.0208); depth 3: 188/384 (+0.0046); depth 4 was already 225/384
  (-0.0371). Monotone in wins and mean, sign flip just above depth 3,
  churn harm clears at every depth (p < 1e-9 everywhere). Committed
  summaries `null-control-depth2-384.json` / `null-control-depth3-384.json`
  with a monotone dose-response pinning test (suite at 387). Also:
  smoke-tested the full three-queue Gate 4 launch path (primary/control/
  null through the real CLI in profile mode + the four-clause verdict) -
  clean.
- **Section 5.14 pre-registered (2026-09-15, revision `4e26de9`)**: the
  **do-nothing clause for Gate 4**. The 5.1/5.5 verdict structure predates
  the null arm - its harm guard compares adaptive against RANDOM churn
  only, so a full-fidelity run could pass while adaptive still loses to
  doing nothing (the exact 5.9 trap). The amendment: a null-session queue
  at declared fidelity over the SAME block 15000-15011 (runbook step 2c,
  `--session-arms null`, own roots, ~90 s/seed), and two new verdict
  clauses - `PASS_COST_ONLY_NULL_REJECTS` (adaptive loses-or-ties >= 10/12
  vs null) and `PASS_COST_ONLY_NULL_MISSING` (the queue is mandatory), both
  exit-2 withholdings leaving the cost PASS standing. Clause order: harm
  guard > pool control > null > control-missing > null-missing > band.
  Implemented: `scripts/v0_verdict.py --null-root`, CLI `--session-arms`,
  10 new tests (suite at 372). BONUS FIX: the pair-session `--seeds` parser
  now accepts the runbook's documented `15000-15011` range form (it crashed
  on it before - a latent bug the new CLI tests exposed). The T4 runbook
  now has three queues (primary, control, null) and a 4-clause verdict
  command.
- **Section 5.16 pre-registered (2026-09-15, revision `e62a732`)**: **is the
  discovered topology a reusable artifact?** The plan's headline claim
  (section 102: extract the topology as a "reusable computational genome")
  has never been tested, because every Gate 3 comparison scores a topology
  *while it is trained*, so weights and structure are confounded. Deliverable
  10 ("Fresh-initialization results") was TODO. This is the cheapest decisive
  version: per seed one three-arm discovery block (the registered 5.10 config,
  depth 4 + spread, arm order keeping random/adaptive at indices 0/1 so the
  discovery runs are the registered ones, with the never-mutating `fixed` arm
  appended - it consumes no global RNG, pinned by test), then each arm's final
  genome transported: rebuilt with fresh weights
  (`initialize_retraining(mode="fresh")`) and retrained 48 steps on the same
  per-seed data schedule with mutation FROZEN. New machinery:
  `ArmResult.genome` (plan section 11 weight/topology separation, made usable -
  the discovered topology is now in the record and JSON-round-trips),
  `src/act/evolution/reuse.py` (`transport_genome`), and
  `scripts/reusability_study.py` (run/verdict/summary, 20-way sharding); block
  81000-81383, fresh. 13 new tests. Transport validity is the load-bearing
  built-in check: transporting the unmutated genome must reproduce that arm's
  own run exactly.
- **Section 5.16 outcome (RAN 2026-09-15, revision `e62a732`, block
  81000-81383)**: registered **branch (c) - both primaries fail**: the
  adaptive advantage at toy fidelity lives in the trained weights and/or
  initialization, NOT in the structure the controller chose. Transport
  validity held perfectly (384/384 exact reproductions, max |delta| 0.0).
  P1 transport-adaptive < transport-random: 212/384, mean -0.1016, p 2.32e-02
  - **exactly on** the sibling studies' bar (212) but below the bar this
  section declared (220 at the declared alpha 0.0025, see the 5.16.1
  amendment: the declaration was internally inconsistent and was resolved
  AGAINST the hypothesis). P2 transport-adaptive < transport-fixed: 211/384,
  mean -0.0866, p 2.94e-02 - fails at either bar. Two by-products that matter:
  (1) the discovery half replicates 5.10's depth-4 ordering OUT OF SAMPLE on a
  fresh block (adaptive beats random 259/384, p 3.5e-12; beats fixed 216/384);
  (2) discarding the weights costs the edge its *consistency* but not its
  *mean* (win rate 259/384 -> 212/384 while the mean barely moves), and a
  seed's discovery ranking predicts its transport ranking only 201/384 times
  (52%). Not capacity-confounded: adaptive topologies are usually smaller
  (lower active params in 264/384) and on the 90 capacity-matched seeds they
  beat random only 40/90 (bar 59). Committed summary
  `artifacts/study-summaries/reusability-depth4-384.json`
  (`study_f0334893263366c2`), pinned by `tests/test_reusability_summary.py`
  (which also pins the threshold resolution and the branch label).
  Consequence: the plan's section-55 initialization-mode study
  (evolved/rewound/fresh/teacher on one genome) is the next step, NOT the
  section-52 scale-up.
- **Section 5.17 pre-registered (2026-09-15, revision `e62a732`)** and
  **outcome (RAN 2026-09-15, block 82000-82383)**: plan RQ6 / section 55 - **is
  initialization or topology the important artifact?** Registered **branch (d):
  both the structure AND its initialization carry value.** Five transports per
  seed of three discovered genomes under both section-55 treatments
  (`rewound` = restore every component that existed at initialization to its
  initial weights **by MoE expert id**; grown experts keep the fresh draw;
  `fresh` = transfer nothing). P1 rewound < fresh `215/384` (mean -0.0806,
  p 1.08e-02, PASS); P2 rewound-adaptive < rewound-fixed `274/384`
  (mean -0.1123, p 1.46e-17, PASS). **This revises section 5.16's branch (c):**
  the topology has standalone value only as the *pair* (structure + the
  initialization its components actually had) - the same P2 comparison with a
  fresh draw fails again on this independent block (`198/384`, mean -0.0317,
  reproducing 5.16's `211/384`). Plan section 11's "final topology + original
  initialization ... without ambiguity" is empirically load-bearing, and the
  section-52 scale-up is triggered with the initialization policy as part of the
  declared artifact. Two pre-registered checks keep it honest: rewinding the
  *random* arm's topology does nothing (`189/384`, mean +0.0424), so the effect
  is specific to the chosen topology rather than a generic "restore good
  weights" procedure; and P1 concentrates where rewinding is total (pure-prune
  seeds `123/213`, p 1.4e-02, vs seeds with grown experts `92/171`, p 0.18),
  while P2 holds in both subgroups. Not capacity-confounded: the adaptive
  topology is *smaller* in 352/384 seeds. Unregistered observation (labelled as
  such in the report): the discovery run itself is worse than a clean retrain of
  its own topology from the original initialization (clean retrain wins
  `271/384`, mean A-B +0.0961). NEW MACHINERY: identity-aware rewinding -
  `initialize_retraining(..., renames=...)` and
  `reuse.expert_renames`/`initial_model`/`transport_genome(initialization=...)`.
  This fixes a latent bug: the pre-existing `rewound`/`evolved` copy matched
  parameters by name and shape, so rewinding a pruned topology would have handed
  a survivor the *pruned* expert's initialization (shape-compatible, silent); a
  test now asserts survivor identity on parameter values and fails if the
  mapping is dropped. A pre-run correction (report section 5.17.1) fixed
 5.17's own built-in check, which claimed B == C for pure-prune genomes - false,
  because a pruned architecture is a different architecture, so its fresh draw
  differs; B == C holds only where the topology is unchanged. Caught by the
  smoke test on a spent block before the registered block ran. Committed summary
  `artifacts/study-summaries/initialization-depth4-384.json`
  (`study_aef9c55774ab2c04`), pinned by `tests/test_initialization_summary.py`.
- **Section 5.18 pre-registered and outcome (RAN 2026-09-15, block 83000-83383,
  donor block 84000-84383, revision `137f6d0`)**: **is the genome portable, or
  irreducibly paired with its own initialization?** Registered **branch (d):
  BOTH** - and both predictions recorded before the run were wrong in the same,
  informative way. Three transports per seed: `O` = adaptive genome rewound to
  its own initial model; `D` = adaptive genome rewound to a **donor** initial
  model (seed+1000, a declared donor block); `F` = unmutated genome rewound to
  the donor's initial model, which copies every tensor and so **is** the donor
  model trained on this seed. `D` and `F` therefore share one initialization and
  differ only in topology. P1 pairing (difference-in-differences) `237/384`
  (mean -0.0830, p 2.54e-06, PASS); P2 portability (`D` < `F`) `234/384`
  (mean -0.0465, p 1.06e-05, PASS) - **the project's first support at any
  fidelity for plan section 102's "reusable computational genome" claim**.
  **The synthesis across 5.16-5.18.2: every initialization-MATCHED
  topology-vs-initial comparison wins** (own/own `293/384` here and `274/384` in
  5.17; donor/donor `234/384`) **while both MISMATCHED ones fail** (5.16 P2
  `211/384`, 5.17 `C_adaptive_vs_B_fixed` `198/384`). Sections 5.16's branch (c)
  and 5.17's failure rows were not measuring a weak topology but a control whose
  initialization was not matched to the treatment's: the seed-level
  quality difference between two draws has stdev `0.86`, five times the `~0.1`
  topology effect, which is what swamps the comparison. A pre-run correction
  (**5.18.1**, chosen from the spent smoke block for variance reasons, and whose
  smoke-block mean pointed *against* the hypothesis) replaced the contaminated
  raw contrast with the difference-in-differences; the raw contrast is still
  reported and still fails (`199/384`, mean -0.1452, p 0.254). NEW MACHINERY:
  `transport_genome(..., rewind_source=...)` for donor initializations, plus a
  test pinning that rewinding an unmutated genome to a donor yields the donor's
  state dict exactly. Committed summary
  `artifacts/study-summaries/portability-depth4-384.json`
  (`study_757e78b90de8a510`), pinned by `tests/test_portability_summary.py`
  (suite at 523).
- **Section 5.19 RAN (2026-09-15, blocks 85000-85383 depth 2 / 86000-86383 depth
  3, donors +2000).** Its committed summaries were missing while its numbers
  were already being cited; regenerated from the surviving shards as
  `portability-depth2-384.json` (`study_436e3cfb83e13fc0`) and
  `portability-depth3-384.json` (`study_ea20b7b1ace9534b`). Registered verdict:
  **branch (c)** - both primaries pass at both depths - so the matched-init
  topology value is NOT depth-limited and section 5.15's depth-4 onset is a
  property of probe-based selection, not the substrate (redirect to controller
  research; section-52 scale-up deprioritised). Recorded trap: the script's
  automatic `outcome` label uses section 5.18's branch letters, so the depth-2
  record says "branch (b)" while 5.19's registered reading is (c); pinned by
  `test_section_5_19_auto_label_letters_index_section_5_18_not_5_19`.
- **Section 5.20 RAN (2026-09-15, blocks 89000-89383 depth 2 / 91000-91383
  depth 3, `scripts/specificity_study.py`).** The selection-specificity control
  the earlier studies lacked: three rewound transports per seed on the seed's own
  initial model. **P1 `A_own` < `R_own`: 243/384 (depth 2) and 248/384 (depth 3),
  both PASS**; P2 `A_own` < `F_own` replicates at 264/384 and 265/384. Registered
  verdict **branch (a) - the shallow-depth topology value is selection-specific**
  (my declared prediction that P1 would fail at depth 2 was wrong). The
  random-churn control is neutral at depth 3 (mean -0.0012) and only nominally
  non-zero at depth 2 (mean -0.0213, 29% of the effect, failing the Bonferroni
  bar); capacity does not explain it (random-churn topologies are never the
  initial size in 0/384 seeds, yet gain nothing). Summaries
  `specificity-depth2-384.json` (`study_f10d1b1cd3812bc1`) and
  `specificity-depth3-384.json` (`study_b6455e54f8801fd8`), pinned by
  `tests/test_specificity_summary.py`. **Amendment 5.20.1:** a one-seed-in-384
  chance coincidence of the two arms' genomes tripped an all-384-differ guard and
  withheld the depth-2 block; the guard now requires coincidences to be *exact
  ties* and *rare*, which changes no statistic - the declared reading passes with
  the tie counted as a non-win. Architecture hashes are carried per seed so the
  guard is recomputable from the artifact.
- **Sections 5.21 (observation) and 5.22 (RAN 2026-09-15, blocks 92000-92383
  depth 2 / 93000-93383 depth 3).** Reanalysing three committed blocks showed
  each mutating arm's fused run losing to a frozen retrain of its own final
  genome at every depth, and - the part section 5.17.2 could not see, since it
  checked only the adaptive arm - losing for the **random** arm too (292/384,
  293/384, 258/384). That makes the ~0.1 nat penalty a property of *mutating a
  live run*, about twice the size of the topology signal these studies measure,
  which is a sufficient explanation for the weak-to-negative fused comparison at
  shallow depth. Recorded as 5.21 explicitly **unregistered** (a reanalysis of
  blocks run for other reasons), then 5.22 pre-registered, committed, and run:
  **branch (a), both primaries passing at both depths** (296/384 and 291/384
  adaptive; 286/384 and 300/384 random). **Adopted consequence: two-stage
  delivery (discover -> freeze the genome -> retrain from the original
  initialization) is the project's pipeline.** *SUPERSEDED in part by sections
  5.23-5.28 - see below: that consequence was earned on the then-current mutation
  path, which 5.24/5.26/5.27 decomposed into a router discard (a defect) and the
  fresh new expert (invertible by cloning), and 5.28 showed Freeze-and-retrain is
  a REGRESSION in the drawn regime this was adopted from. The pipeline is now
  clone-and-preserve, keeping the fused run.* Section 5.15's depth
  dose-response is re-read as being about the *fused* process. Two fresh blocks
  also replicated 5.20's primaries (256/255 and 257/272). Unseparated mechanism:
  a frozen retrain has a fresh optimizer and trains a pruned topology, so
  neither optimizer history nor per-step FLOPs are matched - declared in the
  pre-registration as uncontrolled. Summaries
  `artifacts/study-summaries/frozen-delivery-depth{2,3}-384.json`
  (`study_bc1a4fdfe99fde63`, `study_f23d93bf49e18a57`), pinned by
  `tests/test_specificity_summary.py`; the observation is pinned to its source
  summaries by `tests/test_fused_vs_frozen_observation.py`, which also fails if
  the report drops the "unregistered" label.
- **Section 5.23 RAN (2026-09-15, blocks 94000-94383 depth 2 / 95000-95383 depth
  3, `scripts/optimizer_reset_study.py`).** The confound section 5.22 left open,
  decomposed with a new opt-in knob `ExperimentConfig.optimizer_reset_steps`
  (empty by default, so no published number moves): a twin run whose genome,
  weights, seed and data schedule are identical to the fixed arm's and which
  differs **only** by discarding Adam's moments at steps 12/24/36. Registered
  verdict **branch (c)**: resets cost nothing - P1 fails at both depths *in the
  exonerating direction* (the reset twin was slightly BETTER, `F` winning only
  157/384 and 105/384, mean +0.005 / +0.016) - while the rebuild alone accounts
  for the whole penalty (`U` < `R`: 299/384 and 311/384, mean -0.1069 / -0.1276,
  against section 5.22's total -0.0996 / -0.1219 on independent blocks). **So
  section 5.22's delivery change stands unqualified and the cheap alternative is
  closed off: preserving optimizer state across mutations would buy nothing.**
  Prediction declared as correct here after four consecutive understatements, and
  the pre-registration disclosed that it was informed by a spent-block smoke that
  pointed against the mechanism the design was built to detect. What remains
  unseparated: the reset twin keeps the *initial* architecture (so `U` vs `R` is
  a between-topology contrast), and a grow's fresh expert initialization is not
  isolated at all - it is now the leading candidate. Summaries
  `artifacts/study-summaries/optimizer-reset-depth{2,3}-384.json`
  (`study_e0f1f2629785dc17`, `study_09573ffbfb828091`), pinned by
  `tests/test_optimizer_reset_summary.py` (which also pins that the rebuild's
  cost is 0.8-1.2x the 5.22 penalty, so the decomposition cannot silently stop
  closing) and `tests/test_optimizer_reset_control.py`.
- **Section 5.24 RAN (2026-09-15, blocks 96000-96383 depth 2 / 97000-97383 depth
  3, `scripts/router_preservation_study.py`) - and it found a REAL DEFECT.** A
  layer's router weight is `hidden x experts`, so its shape changes with the
  expert count and `rebuild_model_from_genome` (which carries only same-shape
  keys) **re-draws the mutated layer's entire learned routing head at random at
  every mutation** - while the experts' own weights survive by id and unmutated
  layers' routers are copied. Verified directly, not inferred. Every fused run to
  date therefore discarded a layer's learned routing function three times.
  `rebuild_model_from_genome(preserve_router=True)` copies surviving rows by id
  and leaves only a grown expert's row fresh. Registered verdict **branch (b)
  - partial**: preserving the routing head removes **55.1% / 54.3%** of the
  penalty (`Rp` < `R`: 264/384 and 270/384, mean -0.0542 / -0.0694) and a
  residual `-0.0441` / `-0.0583` survives (`Rp_own` < `Rp`: 254/384, 258/384).
  **So the dominant term of the effect section 5.22 changed the delivery path
  for was an avoidable implementation defect, not an intrinsic cost of mutating
  a live run.** The study is perfectly paired and that was checked, not assumed:
  both settings make the same architectural choices, so final genomes are
  identical in 384/384 seeds and the frozen references are bit-identical in
  384/384 (`R_own == Rp_own` exactly). **The `preserve_router=False` default is
  deliberately retained** even though the discard is unintended, because every
  committed number in sections 5.16-5.23 was measured under it and flipping it
  would silently invalidate the record; new work should set
  `preserve_router_on_rebuild=True`. Prediction (branch (b)) was right, second in
  a row, with the report stating plainly that the two spent-block smokes
  disagreed with each other. Remaining: the residual `~0.05` is either a pruned
  expert's lost capacity or a grown expert's fresh initialization, which this
  design cannot separate. Summaries
  `artifacts/study-summaries/router-preservation-depth{2,3}-384.json`
  (`study_18c56d8bf539e2f3`, `study_65103c230c610378`), pinned by
  `tests/test_router_preservation_summary.py` (which also pins the defect itself,
  teeth-checked by flipping the default).
- **Section 5.25 RAN (2026-09-15, blocks 98000-98383 prune / 99000-99383 grow,
  `scripts/residual_attribution_study.py`).** Attributes section 5.24's residual
  by *forcing* the operation rather than splitting a mix: new opt-in
  `ExperimentConfig.forced_operation` makes every non-adaptive event the named
  operation, and a forced operation the genome cannot perform is left
  unperformed rather than swapped for the fallback grow (so a profile that stops
  mutating fails its budget check instead of measuring the wrong thing).
  Registered verdict **branch (a) - the residual is BOTH operations**: the prune
  block 241/384 (mean -0.0297, p 3.25e-07) and the grow block 241/384 (mean
  -0.0476, p 3.25e-07), bar 212 at `alpha/2`. **The grow costs ~1.6x the prune**
  (0.0159 vs 0.0099 per forced event) - the opposite of the declared prediction,
  which expected the grow to be free because it only adds capacity. Fifth
  understated prediction in the sequence, and the first where the miss was in the
  *comparison* rather than in whether an effect exists. An additive check roughly
  holds: per-event costs predict `0.0386` for a 50/50 mix against section 5.24's
  measured `0.0441`, on different blocks and substrates. **Consequence:** the
  remaining lever is the *initialization of a new expert*, exactly plan section
  51's teacher treatment - now the best-motivated untested experiment in the
  programme, and still blocked on Qwen-derived artifacts this repo does not
  bundle. Summaries
  `artifacts/study-summaries/residual-{prune,grow}-384.json`
  (`study_68591d55f3a56db7`, `study_c7ec09294da043fc`), pinned by
  `tests/test_residual_attribution_summary.py` and
  `tests/test_forced_operation_control.py`.
- **Section 5.26 RAN (2026-09-15, blocks 103000-103383 depth 2 / 104000-104383
  depth 3, `scripts/new_expert_init_study.py`).** Answers section 5.25's closing
  question - is the residual penalty the *freshly drawn* new expert? - with
  something the model already contains instead of the blocked Qwen teacher: new
  opt-in `ExperimentConfig.new_expert_initialization="clone"` gives a grown expert
  the layer's **busiest surviving expert**, body **and router row**, so it starts
  as a working twin rather than as noise. Cloning is defined as copying a router
  row, so the config *refuses* `clone` with `preserve_router_on_rebuild=False`
  rather than silently cloning nothing. Registered verdict **branch (a)**:
  **P1 cloning helps** 315/384 (mean -0.1005, p 5.9e-39) at depth 2 and 307/384
  (-0.1036, p 6.4e-34) at depth 3; **P2 residual penalty FAILED IN THE REVERSED
  DIRECTION** - the penalty goes from **-0.0406/-0.0292** (fresh draw) to
  **+0.0600/+0.0745** (cloned), so cloning does not merely remove the penalty, it
  **inverts it**. The full ordering is the same at both depths,
  **`C < F < C_own < R`**: the clone-grow fused run (14.2414 / 13.3819) beats the
  unmutated fixed arm (14.2887 / 13.4362; 296/384, 291/384), the frozen retrain
  of its own final genome (14.3014 / 13.4564; 258/384, 253/384) and the historical
  random-grow run (14.3419 / 13.4855). The design was perfectly paired by
  construction - the forced grow picks its layer from the seed's own RNG and
  cloning consumes no randomness - so **the two settings chose the same final
  genome in 384/384 seeds at both depths and `C_own == R_own` bit-identically**,
  both asserted as built-in checks. **Consequence: section 5.22's two-stage
  `discover -> freeze -> retrain` pipeline is a REGRESSION on this substrate**,
  not a safeguard; the recommendation for new work is to *keep the fused run*
  with the router preserved and new experts cloned, which also makes plan section
  102's genome extraction unnecessary for performance and gives plan section 51's
  teacher treatment a *bound* (it can only pay off where a teacher beats the
  model's own busiest expert). One open mechanism, declared unresolved: cloning
  copies a trained body *and* a router row, and this design does not separate
  them. Sixth understated prediction (declared P2 pass, branch (b)). Summaries
  `artifacts/study-summaries/new-expert-init-{depth2,depth3}-384.json`
  (`study_99a12c16f910e9c0`, `study_ee833b6b284195f9`), pinned by
  `tests/test_new_expert_init.py` and `tests/test_new_expert_init_summary.py`.
  **Amendment 5.26.1:** the pre-registered branch list was written in the
  opposite letter order to section 5.24's convention and to the study's own
  script; the letters are reconciled to the script, which changes no statistic,
  no threshold and no reading (P2 failed in the exonerating direction, so the
  penalised reading was never available).
- **Section 5.27 RAN (2026-09-15, blocks 105000-105383 depth 2 / 106000-106383
  depth 3, `scripts/clone_mechanism_study.py`).** Resolves the mechanism section
  5.26 declared unresolved: cloning does two things at once (a trained *body*, and
  a copied *router row*), and a third setting separates them. New opt-in
  `new_expert_initialization="clone_body"` is identical to `"clone"` except the
  grown expert keeps a freshly drawn row, so `B - R` is the body's contribution and
  `C - B` the routing affinity's, on one genome. Registered verdict **branch (a),
  both contribute**: **P1 the body helps** 295/384 (mean -0.0744, p 3.4e-27) at
  depth 2 and 300/384 (-0.0844, p 7.0e-30) at depth 3; **P2 routing affinity adds**
  242/384 (-0.0264, p 1.9e-07) and 238/384 (-0.0220, p 1.5e-06). The split is
  **body 73.8%/79.4%, row 26.2%/20.6%** of the total clone gain, and the two terms
  partition it exactly. The ordering is four-deep and the same at both depths -
  **`C < B < F < own < R`** - so the body clone *alone* already beats the unmutated
  arm and the frozen retrain, and the row is a real, cheap second term. Three-way
  pairing asserted, not assumed: same final genome, bit-identical frozen reference
  and an identical fixed arm in **384/384 seeds**. **Consequence:** section 5.26's
  "twin" framing was right but imprecise - the body is the larger half, the row is
  what makes it a twin. It also sharpens plan section 51: a teacher must beat the
  model's own busiest expert **as a body source** for the larger term. Prediction
  was correct (second in a row) and came from the disclosed 4-seed smoke rather
  than from a prior. Declared **not** settled: no `clone_random` arm (any survivor
  vs the busiest), and the layer ends with a tied twin pair whose cost at larger
  `active_experts` is untested. Summaries
  `artifacts/study-summaries/clone-mechanism-{depth2,depth3}-384.json`
  (`study_3bcc9e86b575b7a6`, `study_872d33bdd99496a8`), pinned by
  `tests/test_clone_mechanism_summary.py` and `tests/test_new_expert_init.py`.
  Note: the toy studies stay on **CPU deliberately** - measured 2.6x faster than
  this 1070 at this shape (5.26 vs 2.04 s/seed), and every committed number in
  sections 5.16-5.27 was measured on CPU, so a device switch would stop the record
  reproducing. The real-data legs do use the GPU.
- **Section 5.28 RAN (2026-09-15, blocks 107000-107383 depth 2 / 108000-108383
  depth 3, `scripts/unforced_clone_study.py`).** Closes the gap that mattered most:
  sections 5.24/5.26/5.27 all measured on **forced** operations, but section 5.22's
  freeze-and-retrain pipeline was adopted from the **mixed, unforced** regime. Here
  `forced_operation` is left None, so the mix is drawn (622 grows / 530 prunes at
  depth 2, 548 / 604 at depth 3 - checked, so this cannot be a forced block in
  disguise). Registered verdict **branch (a)**: **P1 clone helps** 276/384 (mean
  -0.0720, p 2.3e-18) and 252/384 (-0.0651, p 4.6e-10); **P2 no residual penalty**
  FAILS in the exonerating direction again (**+0.0243** / **+0.0113**), against a
  historical penalty of -0.0477 / -0.0538. **The claim is sharper than the forced
  blocks could make it: the historical path leaves the run worse than NEVER
  MUTATING** (`F` < `R`: 251/384, -0.0444, p 8.9e-10; 236/384, -0.0403,
  p 4.1e-06), while **cloning makes mutating pay** (`C` < `F`: 238/384, -0.0275,
  p 1.5e-06; 237/384, -0.0248, p 2.5e-06). The frozen retrain ties the unmutated
  arm (186/384, p 0.75; 207/384, p 0.069), as a mixed draw should. Means
  **`C` < `own` ≈ `F` < `R`** at both depths. **Pairing was recorded, not assumed**
  - a drawn prune can fall back to a load-based grow, which reads weights cloning
  changes - and it held anyway: identical final genome **384/384**, identical
  operation mix **384/384**, `C_own == R_own` bit-identically, even though 69/72
  seeds had a layer at the 2-expert floor. Third correct prediction in a row.
  **Consequence: section 5.22's pipeline is contradicted in the regime it was
  adopted from**, and this is now a requirement rather than a preference - the
  recommended pipeline is **clone-and-preserve, keeping the fused run**
  (beats retrain 214/384 and 206/384, beats not mutating 238/384 and 237/384).
  Not tested and named as the next axis: the probe-based **adaptive** selection
  path; arm-genericity of the process penalty is cited from section 5.22, not
  re-measured. Summaries
  `artifacts/study-summaries/unforced-clone-{depth2,depth3}-384.json`
  (`study_5ece35fb3fbc8724`, `study_f8bc02ca0f048a79`), pinned by
  `tests/test_unforced_clone_summary.py`.
- **Section 5.29 RAN (2026-09-15, blocks 109000-109383 depth 2 / 110000-110383
  depth 3, `scripts/adaptive_clone_study.py`).** Tests the pipeline recommendation
  on the **probe-guided adaptive** arm - the discovery path the project actually
  uses, not the `random` arm section 5.28 used. Registered verdict **branch (c)**
  (P1 134/384 and 131/384 FAIL; P2 221/384 and 232/384 PASS), and the prediction
  (branch (a)) was WRONG - the fourth miss in a row. Pairing was not assumed: the
  probe scores candidates with the **default** rebuild while the *selected* one is
  applied with the arm's flags, so the runs can diverge. A pre-run 16-seed
  measurement found 2/16 divergent; the blocks came in at **45/384 and 40/384**.
  **But both registered verdicts are artifacts, not findings.** The adaptive arm
  grows in only **18%** of events, so **205/384 and 207/384 seeds contain NO grow**
  and cloning is then a bit-identical no-op (`C - R == 0` exactly, 0 wins). A
  full-block sign test scores those as non-wins for P1 and hands P2 the full
  historical penalty. Stratified post-hoc on whether a grow occurred, both readings
  reverse: **`C < R` 134/179 (75%) p 9.1e-12 and 131/177 (74%) p 6.0e-11**, with
  the penalty gone in the grow stratum (+0.0126 / +0.0016) and intact in the no-grow
  stratum (-0.0416 / -0.0633). This is the project's first statistic whose
  *validity* was destroyed by a structural no-op rather than by noise. Summaries
  `artifacts/study-summaries/adaptive-clone-{depth2,depth3}-384.json`
  (`study_59e969bbd7767f2d`, `study_c28ad3ce4784f3b9`). **Amendment 5.29.1** records
  that the stratified reading is post-hoc and not outcome-dependent, and
  **supersedes branch (c) for interpretation** - its numbers stand, its reading does
  not.
- **Section 5.30 RAN (2026-09-15, blocks 111000-111383 depth 2 / 112000-112383
  depth 3, `scripts/adaptive_clone_stratified_study.py`).** The pre-registered
  stratified re-test on fresh blocks, with the stratum declared in advance and
  **stratified on the HISTORICAL run's drawn mix** so the setting under test cannot
  decide who enters it. Registered verdict **branch (a), replicated at both
  depths**: grow stratum 195 and 179 seeds, **`C < R` 142/195 (73%) mean -0.0542
  p 7.1e-11 and 130/179 (73%) mean -0.0466 p 5.9e-10** (bars 114 and 105), penalty
  gone there (P2 fails: -0.0064 / +0.0144). **The artifact replicated too:** the
  full-block statistic fails on the same runs (**142/384, 130/384**) while the
  stratified one passes at p ~1e-11 - same evidence, opposite verdicts, and the
  only difference is whether the ~200 seeds that could not respond are counted.
  Positive control held: in the no-grow stratum `C == R` bit-identically and
  `C_own - C == R_own - R` exactly. **Synthesis:** the discovery path prunes 82% of
  events, so cloning repairs the grow term (-0.054/-0.047 where it binds) and
  leaves the prune term untouched, where the historical penalty stands at
  -0.055/-0.074; means are `own < C < R < F` at both depths, so the frozen retrain
  still wins overall on the prune term alone. **Section 5.28's "keep the fused run"
  holds for the `random` arm (half grows) and does NOT hold as stated for
  discovery** - cloning is necessary but not sufficient, and the residual deficit
  on prune-dominated discovery runs is a separate open problem. Summaries
  `artifacts/study-summaries/adaptive-clone-stratified-{depth2,depth3}-384.json`
  (`study_54cff7dc8d8b6f11`, `study_da528e81c3204901`), pinned by
  `tests/test_adaptive_clone_stratified_summary.py`. **Methodological conclusion
  (5.30.1):** record how often a treatment can bind before choosing the statistic,
  stratify on a pre-treatment variable, and include a positive control that the
  treatment is a no-op where it should be.
- **Section 5.31 RAN (2026-09-15, blocks 113000-113383 depth 2 / 114000-114383
  depth 3, `scripts/prune_target_study.py`).** Asks whether *which* expert a prune
  removes is a lever. New opt-in `ExperimentConfig.prune_expert_policy`
  (`"random"` default / `"busiest"` / `"quietest"`, ranked by the model's own
  recorded routing load). Registered verdict **branch (d)** - the sign test failed
  both primaries while the means ordered `Q < R < B` at both depths at 53-54% win
  rates. Summaries `artifacts/study-summaries/prune-target-depth{2,3}-384.json`
  (`study_41bbf67d9ae9bce2`, `study_131929958f11abc2`), pinned by
  `tests/test_prune_target_policy.py`, which also re-runs a committed section 5.25
  block from its own stored config and requires **bit-identical** losses - the
  guarantee that adding a defaulted knob did not move the published record.
  **Reproducibility finding recorded there and in the report:** reproduction is
  bit-exact at 1 and 4 torch threads but **not at 2** (last digits move), so
  "reproduces" is only defined at a fixed thread count and all committed numbers
  are `OMP_NUM_THREADS=1`. **5.31.1:** the binding constraint was the *statistic* -
  at a 53% win rate the paired sign test needs ~1396 seeds for what a mean test
  resolves with ~380, and this block had 384.
- **Section 5.32 RAN (2026-09-15, blocks 115000-116535 depth 2 / 117000-118535
  depth 3, `scripts/prune_target_powered_study.py`).** Section 5.31's question
  re-registered with the statistic declared in advance (one-sided paired sign-flip
  permutation test on the mean, 20k permutations, `alpha/4 = 0.0125`) on a ~4x
  block sized from 5.31's observed effect and spread. Registered verdict **branch
  (a), both primaries pass at both depths**: `fused(Q) < fused(R)` **-0.01133
  p 2.5e-04** and **-0.01250 p 1.0e-04**; `fused(R) < fused(B)` **-0.01678
  p 5.0e-05** and **-0.02435 p 5.0e-05**. **Both statistics agree everywhere in
  both blocks** (sign tests 813/1536, 823/1536, 845/1536, 846/1536 against a bar of
  813), which is 5.31.1's power claim measured rather than argued - the effect was
  always there, and 5.31's branch (d) was its statistic. Loss means order `Q < R <
  B` on the fused column and `F` sits between at depth 2 / above at depth 3.
  **Mechanism, and it is not the penalty framing's:** the lever acts on the fused
  trajectory, not the retrained topology - `own(Q) - own(R)` is **+0.0034 / +0.0008**
  (slightly *worse*) against fused -0.0113 / -0.0125, so the penalty secondary
  **fails** (p 1.00 both depths). Recovered share of the historical prune penalty
  is **27.6% / 41.6%** by targeting the quietest, and targeting the busiest *adds*
  **40.8% / 81.0%**. **Consequence: the lever is conditional on the delivery
  pipeline** - under 5.28's keep-the-fused-run recommendation it is free gain;
  under freeze-and-retrain the delivered artifact is unchanged. Against `F`,
  `fused(Q)` beats never mutating at depth 3 (`-0.01603`, p 5.0e-05) but not at
  depth 2 (`+0.00609`), so 5.30's depth-2 deficit survives. Summaries
  `artifacts/study-summaries/prune-target-powered-depth{2,3}-1536.json`
  (`study_ab45f014e8cb5e50`, `study_55d7527c71843ced`), pinned by
  `tests/test_prune_target_powered_summary.py` (which also pins the statistic's
  own calibration on symmetric heavy-tailed noise, and re-derives the stored
  comparisons from the stored per-seed rows). **5.32
  correction:** 5.31's prose read the sign-flipped `penalty = own - fused` column
  as agreeing with the fused column; it cannot, and 5.32's null own-column
  measurement is the corrected reading.
- **Section 5.33 RAN (2026-09-15, blocks 119000-120535 depth 2 / 121000-122535
  depth 3, `scripts/probe_selection_study.py`).** Two findings prompted it, both
  from section 5.32: its lever was worth 27.6-41.6% of the prune penalty on the
  **non-probe** paths, and its consequence ("the discovery controller should prefer
  the quietest expert") is **wrong**, because `prune_expert_policy` is inert on the
  probe-guided path - `_mutate_by_probe` enumerates every expert through
  `feasible_expert_mutations` and returns before the policy branch, measured as
  bit-identical losses across all three policies under `adaptive_policy="probe_loss"`
  and pinned by `test_the_knob_is_inert_on_the_probe_guided_discovery_path`. So the
  discovery controller already owns the lever; the open question is whether its own
  choice beats the cheap rule. New opt-in `probe_selection="quietest"` (both probe
  paths, so they cannot drift) isolates the TARGET: it fires only when the probe's
  own pick is a prune, and then takes the least-loaded expert in the probed pool.
  Registered verdict **branch (a) at both depths**: `fused(A)` vs `fused(M)`
  **-0.04118 p 5e-05** and **-0.04848 p 5e-05**; `fused(A)` vs `fused(Q)`
  **-0.03120 p 5e-05** and **-0.03443 p 5e-05**; both clear the sign test too. The
  ordering is three-deep and that is the interpretable part: against the same pool at
  the same probe cost the probe beats random by 0.0412/0.0485, the load rule beats
  random by 0.00998/0.0140, so the load rule recovers only **24.2%/29.0%** of the
  probe's advantage - it is not empty, and it is not enough. **Mechanism differs from
  5.32:** here `own(A) - own(Q)` is -0.03571/-0.05414, comparable to the fused
  contrast, so the probe buys a better run AND a better artifact, where 5.32's lever
  moved only the fused column. Against `F`, mutation pays in this configuration
  (`A` -0.04537/-0.05685, `Q` -0.01418/-0.02242). Treatment bound in 6110/6144 and
  6091/6144 seed-selection pairs; four distinct final genomes in 1253/1536 and
  1300/1536. Summaries `artifacts/study-summaries/probe-selection-depth{2,3}-1536.json`
  (`study_dbd63572517fd575`, `study_a24a62de8ce82d39`), pinned by
  `tests/test_probe_selection_summary.py` and
  `tests/test_probe_selection_quietest.py` (the rule's isolation contract: the
  selection must come from the probed pool, and must leave the operation alone when
  the probe's own pick was not a prune). Prediction correct, the fifth of the last
  six; the magnitude was the surprise - I expected the load rule to be comparable.
- **Report-artifact link guard (2026-09-15).** Section 5.19's missing summaries
  were a defect *class*, not an incident: a result can be cited widely while its
  artifact does not exist. `tests/test_report_artifact_links.py` now fails if any
  `study_...` id cited in `RESEARCH_REPORT.md` has no committed summary, or if
  any committed summary is referenced by neither the report nor this file
  (teeth-checked both ways). Full audit at the time of writing: **0 cited but
  missing, 0 orphaned** - section 5.19 was the only occurrence.
- `PYTHONPATH=src python scripts/train_real_lm.py` (and inference via
  `scripts/real_lm_infer.py`) -> the real-data leg the plan's data sections
  deferred: TinyStories (real English corpus) downloaded, a BPE tokenizer
  (vocab 8192) trained on the corpus, 40,043,724 real tokens encoded, and the
  project's own `V0MoECausalLM` (20,863,872 total / 10,247,040 active
  parameters) trained on GPU for 6,000 steps / 49.2M tokens (~45 min on the
  local 1070 at ~18k tokens/s). Final held-out validation loss 1.713
  (perplexity 5.55) against the uniform floor 9.011 (ppl 8192) - the model
  really learned the language. Inference produces fluent on-genre story text
  (greedy decoding deterministic; out-of-domain text scores ppl ~4000, i.e.
  the model  knows its domain). Report and samples in
  `artifacts/real-lm/report.json`; corpus/tokens/checkpoints are local
  scratch (gitignored).
- `PYTHONPATH=src python scripts/real_lm_chat.py` -> interactive streaming
  chat over the trained checkpoint: user lines are appended to the rolling
  story context (the model reacts to what you typed, verified: a "brave
  knight" line produced a lion-roar continuation), tokens print as they are
  generated, and session commands (/new with optional seed, /temp, /topk,
  /len, /show, /save, /quit) manage the loop; `--once` reads one seed line
  from stdin for piped use. Reuses `real_lm_infer.load_model` so chat and
  one-shot inference always agree on config, checkpoint, and decode.
- `PYTHONPATH=src python scripts/generate_figures.py [--set decoupled]` -> writes
  the four factor-map figures (fig1 win-rate forest, fig2 quality-vs-cost, fig3
  delta distributions, fig4 probe-rank heatmap), byte-identical across re-runs.
  `--set decoupled` plots the corrected section 9.11 record into
  `artifacts/study-summaries/decoupled/figures/` from summaries emitted by
  `factor_map_recheck.py --summaries` (each re-verified by `check_summary` at
  plot time); that set carries all nine rows and plots its own figure 4, from
  the decoupled counterfactual analysis produced by
  `scripts/probe_validity_analysis.py` (section 9.11).
- `PYTHONPATH=src python scripts/probe_validity_analysis.py --verify` ->
  recomputes every field of the committed probe-validity `results` block from
  its `per_seed` block and checks it (13/13 fields agree). Without `--verify`:
  `--shard i --shards n --out DIR [--coupled]` runs the 4-branch counterfactual
  sweep for a seed slice, and `--analyze '<DIR>/*.json' [--out FILE]` merges
  shards, cross-checks branch 0 against the named study summary, and writes the
  document in the committed schema. Ran coupled (reproduces the committed
  results) and decoupled (writes
  `artifacts/study-summaries/decoupled/probe-validity-analysis.json`) -
  section 9.11.
- `PYTHONPATH=src python scripts/factor_map_recheck.py --list` -> prints the
  recheck set (the nine gate studies, the legacy signal one reconstructed rather
  than excluded, plus the guided bound). Membership is decided by whether a
  summary records a gate verdict, not by having a config and a seed block: the
  depth, portability and probe-scoring studies carry both and record a different
  schema, so the `config and seeds` test grew the set from ten to fifty-six and
  `--list` raised `KeyError: 'gate_passed'` instead of printing it;
  `--study NAME --shard i --shards n --out DIR` re-runs a
  shard's slice with the published config and only `independent_arm_rng`
  changed, and `--compare '<DIR>/*.json'>` applies the declared label-agreement
  rule and can write  the result artifact, plus `--summaries DIR` to emit the corrected study
  summaries in the committed schema (validated by `check_summary`, so a summary
  that does not recompute cannot reach a figure). Ran as declared: 144 shards,
  ~6 min wall, `artifacts/study-summaries/factor-map-recheck.json` and
  `artifacts/study-summaries/decoupled/` (section 9.11). Unit-pinned in
  `tests/test_factor_map_recheck.py` (recheck set
  membership, only-the-RNG-changes reconstruction, shard merge order, label
  agreement and flip detection, threshold recomputation from n, and incomplete
  seed sets never reporting agreement).
- `PYTHONPATH=src python scripts/v0_verdict.py --dataset-root <root>` -> computes
  the pre-registered §5.1 verdict from the published seed summaries (verified
  against a live-published dataset and now unit-pinned in
  `tests/test_v0_verdict.py`: correctly aggregates per-arm FLOPs under the
  toy accounting, excludes stray complete seeds outside the declared block with
  a warning, refuses unmatched mutation budgets or missing arms, and returns
  FAIL with the missing-seed list until the queue completes; the harm-guard
  boundary (reject at >= 10 adaptive losses) is asserted on both sides; the
  adopted section 5.5 pool-matched clause is pinned over
  `POOL_CONTROL_WIN_LIMIT` - a control clearing the same bar withholds Gate 4
  (`PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES`), one win below it does not,
  and a **missing** control withholds Gate 4 too
  (`PASS_COST_ONLY_CONTROL_MISSING`) rather than reading as "no pool effect").
  `--control-root` supplies the §2b control session; the declared block is
  overridable with `--seeds "A-B"` (or a comma list) without loosening any
  criterion. The CLIs now expose the adopted protocol
  (`--probe-candidate-policy` / `--probe-selection` on `--mode pair-session`),
  both defaulting to the registered values.
  Exit codes:
  0 = PASS (Gate 4 authorized only with all 12 declared seeds complete),
  1 = FAIL, 2 = PASS with qualification (band miss or harm-guard rejection).
- `PYTHONPATH=src python -m pytest -q` -> 301 passed (adds 9 substrate-audit
  tests: per-layer expert counts build and rebuild at depth > 1; prefix is
  confined to the first layer while spread reaches past it, in both the toy
  path and the V0 session; the three `probe_selection` rules differ, are
  deterministic, and are recorded in the mutation history; and a V0 resume
  that re-executes probe events redraws the identical spread subset
  bit-identically).
- Reduced-dim V0 session check (exploratory, seeds 30000-30011, depth 6 x 4
  experts, cap 4, steps=120): the declared code path's prefix policy probes
  only layers L0-L1 while `spread` probes all six - the structural difference
  confirmed outside the toy path. No adaptive edge appears at this fidelity
  (3/12, 5/12, 5/12; mean deltas ~1e-03), so the confound is not quantified
  there. Consequence recorded in `RESEARCH_REPORT.md` section 9.6: the
  section 5.1 T4 run should not launch on the current pre-registration until
  the candidate policy and a pool-matched control arm are decided.
- Substrate audit runs (exploratory, block 20000-20127): depth-4 cap-4 studies
  pass on both `prefix` and `spread` (47/64 and 48/64 at steps=24); the
  probe-rank gradient is 47/37/35 for best/middle/worst; and the pool-matched
  random control (80/128, p `0.0030`) reproduces most of the argmin edge
  (88/128, p `1.3e-05`) that argmax (64/128) does not. See
  `RESEARCH_REPORT.md` section 9.
- `PYTHONPATH=src python -m compileall -q src/act tests` -> passed.
- The frontend checks that stood here (`npm run typecheck`, `npm run build`,
  `npm test -- --run` -> 185 passed, including the grok-pwa-plugin and preview
  test-port fixes) were dropped with the TypeScript console they exercised; it
  was removed as unmaintained and representing none of the results. See the
  removal record near the top of this document, and git history for the logs.
- Python Experiment 0 mutation-budget test -> passed.
- Python registry/CLI round trip -> passed.
- Initial Gate 3 trial -> adaptive lost to random on seeds 1, 2, 3, and 42.
- Four-seed gate test -> adaptive won 3/4 seeds with equal three-mutation budgets.
- Eight-seed generalization probe -> adaptive won 6/8 seeds; Gate 3 remained open pending a stronger pass.
- Sixteen-seed baseline -> adaptive won 10/16 seeds, with matched budgets and mean loss delta `-0.032691`.
- Expert-signal sweep -> random expert selection remained strongest at 6/8; routing, gradient,
  Fisher, and parameter-norm choices were retained as configurable probes but not selected by default.
- Gate-study CLI -> `PYTHONPATH=src python -m act.evolution.cli --seeds 1,2,3,42 --steps 48`
   emits a deterministic multi-seed JSON summary.
- Gate-study registry round trip -> the same CLI appends exactly one `gate_study` JSONL record.
- Gate-study budget invariant -> study execution and registry records enforce equal random/adaptive
   mutation counts for every seed.
- Research invariants -> passed in the latest checks. (The browser console this
  line also covered was removed with the frontend, so nothing checks it now.)
- Fresh 16-seed Gate 3 study (`1,2,3,42,5,6,7,8,9,10,11,12,13,14,15,16`) -> 9/16 adaptive wins at
  48 steps, mean paired delta `-0.008127`, population standard deviation `0.162910`, and matched
  budgets; the scaling gate remains open.
- **384-seed pre-registered Gate 3 study (seeds 1000-1383, steps=48, budgets matched)**
  -> 195/384 adaptive wins (50.8%), mean paired delta `+0.006766`, stddev `0.131115`,
  exact binomial p `0.3993`, paired-t p `0.8437` (directed: treatment helps = negative
  delta); Gate 3 FAILED for the cheap-signal controller. Reproducible
  (`study_id` `48163d824f2d6852`, recorded in `artifacts/gate3-384.jsonl`).
- **96-seed guided-arm diagnostic (seeds 2000-2095, steps=48, budgets matched)**
  -> 59/96 guided wins (61.5%), mean delta `-0.041325`, binomial p `0.015774`,
  paired-t p `0.00025`; the probe-loss-greedy upper bound PASSED under the same
  verdict machinery (`study_id` `d2b76921ac40d39b`, `artifacts/guided-diagnostic-96.jsonl`).
- Gate-study verdict fields (`gate_alpha`, `gate_p_value`, `gate_min_wins`,
  `gate_passed`, `t_statistic`, `t_p_value_one_tailed`, `treatment`) added to
  `GateStudyResult` and gate-study registry records; `study_hash` now hashes the
  gate-decision inputs rather than float aggregates; the t p-value direction was
  corrected (helpful treatment = negative delta = P(T <= t)); covered by new unit tests.
- New opt-in `guided` arm (`--arms random,guided`, `--treatment guided`) with exact
  mutation-candidate enumeration, probe-loss selection, probe cost in `search_flops`,
  and probe metadata in mutation history.
- **Controller decision rule replaced**: the adaptive arm now defaults to bounded
  probe-loss selection (`adaptive_policy="probe_loss"`, `probe_candidate_cap=4`,
  `probe_batch_count=2`) via a shared probe-mutation engine; the legacy
  routing/gradient/Fisher rule remains available as `adaptive_policy="signal"`.
  Studies and registry records now include `search_flops_by_arm` and
  `treatment_search_overhead_ratio` for explicit search-cost accounting.
- **384-seed probe-loss Gate 3 re-run (seeds 3000-3383, steps=48, budgets matched)**
  -> 213/384 adaptive wins (55.5%; 209 required), mean delta `-0.025955`, binomial
  p `0.018138`, directed t p `7.4e-05`; quality PASSED, pre-registered <=5% search
  cost FAILED at 41.9% (`study_id` `27fe7e53d0533f29`,
  `artifacts/gate3-probe-loss-384.jsonl`).
- **380-seed cheap-probe study (seeds 4004-4383, steps=144, cap=2, 1 probe batch,
  budgets matched)** -> cost criterion PASSED (4.13% <= 5%; probe cost amortizes
  with horizon) but quality FAILED: 181/380 wins (47.6%), mean delta `+0.011747`,
  binomial p `0.8351`, t p `0.8857` (`study_id` `41f4b9f7cf6d82e0`,
  `artifacts/gate3-cheap-probe-380.jsonl`, summary in `artifacts/study-summaries/`).
  Confound noted: probe budget and horizon changed together vs the passing
  cap-4/2-batch/48-step study.
- **380-seed same-seed placement study (cheap probe, steps=144, interval=40 ->
  mutations at 40/80/120; seeds 4004-4383 identical to the early-placement run)**
  -> late placement better in 201/380 pairs, mean paired delta `-0.015666`,
  paired t p `0.0788`; late-study vs random alone: 198/380, binomial p `0.2208`.
  Placement changes 39% of probe decisions but does not statistically explain
  the horizon effect (`study_id` `34a9a5216a8aac36`,
  `artifacts/gate3-late-placement-380.jsonl`, summary in `artifacts/study-summaries/`).
- Added `--mutation-interval` and `--mutation-budget` to the experiment CLI
  (were config-only).
- **384-seed variance study (seeds 6000-6383, cap=2 candidates, 4 probe batches,
  3 events, steps=144, budgets matched)** -> quality FAILED: 179/384 wins (46.6%,
  threshold 209), mean delta `-0.003907`, binomial p `0.9159`, t p `0.3453`;
  probe cost 16.0% (descriptive). Probe variance reduction alone does not restore
  the adaptive edge (`study_id` `cdebaf2ed9f23f54`,
  `artifacts/gate3-variance-384.jsonl`, summary in `artifacts/study-summaries/`).
- **384-seed horizon replication (seeds 8000-8383, the passing cap-4/2-batch
  config at steps=144, budgets matched)** -> quality PASSED decisively: 231/384
  wins (60.2%, threshold 209), mean delta `-0.052402`, binomial p `4.04e-05`,
  t p `1.07e-08`; probe cost 14.0% (41.9%/3, matching the amortization model);
  clears Bonferroni alpha/7 ~ 0.0071 on both statistics
  (`study_id` `162362a15b1d6aee`, `artifacts/gate3-horizon-384.jsonl`, summary in
  `artifacts/study-summaries/`).
- **384-seed richness study (seeds 7000-7383, fixed nominal probe tokens of
  192/event, 3 events, steps=144, budgets matched)** -> arm A cap-4/1-batch
  PASSED: 228/384 wins (59.4%), mean delta `-0.036982`, binomial p `1.4e-04`,
  t p `5.0e-05`, cost 7.3% (`study_id` `994282a17955597e`); same-seed arm B
  cap-2/2-batch FAILED: 193/384, mean delta `+0.007156`, binomial p `0.47965`,
  cost 8.1% (`study_id` `de2f894d75eaedac`); paired A>B in 225/384 pairs
  (sign z `3.37`, two-sided p `8e-04`). Candidate richness is the active axis
  (`artifacts/gate3-rich-384.jsonl`, `artifacts/gate3-narrow-384.jsonl`,
  summaries in `artifacts/study-summaries/`).
- **384-seed cadence study (seeds 5000-5383, steps=144, interval=24, budget=6,
  cheap probe cap=2/1-batch, budgets matched)** -> 137/384 adaptive wins (35.7%,
  threshold 209), mean delta `+0.047014`, binomial p `1.0`; adaptive significantly
  WORSE than random (z `-5.61`, two-sided p ~ `2e-08`); probe cost 8.41%
  (descriptive only - the 5% cap was not part of this pre-registration)
  (`study_id` `964e3f08e13a3710`, `artifacts/gate3-cadence-384.jsonl`, summary in
  `artifacts/study-summaries/`).
- **Probe-landscape instrumentation** (`--probe-landscape`, `--probe-force-candidate`,
  `--probe-force-events`) records every candidate's probe loss per mutation event
  and supports deterministic counterfactual branch runs; forced index 0 is verified
  trajectory-identical to greedy selection. Probe-validity counterfactual study
  (384 seeds x 4 candidate branches at event 1) recorded in
  `artifacts/study-summaries/probe-validity-analysis.json`.
- **Report figures generated** (deliverable 12): the four pre-registered figures
  (`artifacts/study-summaries/figures/`) are produced by `scripts/generate_figures.py`
  from the versioned summaries via `act.evolution.figures_data`; the script
  recomputes every verdict (binomial p, pass flag, mean delta, probe-cost
  identity, argmin concordance) from raw per-seed records before plotting and
  fails on any mismatch. Re-runs are byte-identical. Data assembly is unit-tested
  in `tests/test_report_figures.py` (26 tests; statistics helpers pinned to
  scipy reference values); run `PYTHONPATH=src python scripts/generate_figures.py`
  (requires matplotlib, not needed for the test suite).
- Gate studies now fail fast with a clear error when the `random` baseline arm is
  missing from `--arms`.
- The two frontend fixes once recorded here (`injectGrokPwaHead` preferring the
  document title for `og:title`, and preview test ports matching `PREVIEW_PORT`)
  are spent: the code they patched was removed with the console.

## Environment Notes

- Git was initialized on 2026-09-12; public project baseline committed as `90416f3`.
- The app uses port `4173` because port `8080` is occupied by an unrelated local service.
- The installed PyTorch CUDA build does not support the local GTX 1070, so Python validation
  currently runs on CPU with a warning.
- The installed PyTorch CUDA build (2.14.0+cu126) runs kernels on the local GTX 1070
  (sm_61): `torch.cuda.is_available()` is true, matmul launches succeed, and the local
  proof run (below) trained on GPU at ~15k tokens/s. Earlier "CPU only" notes are stale
  for the current environment.
- Operational hazard (learned 2026-09-13): `V0SessionConfig` defaults are the DECLARED protocol
  dimensions, so a partial config silently instantiates the full 285M-parameter model on CPU
  (~1.1-3.4 GB checkpoints; a test run once accumulated 19 GB in /tmp before hitting the disk
  quota and hanging). Any reduced-fidelity session - tests, self-check, smoke - must set
  vocab/context/hidden/depth/expert dims explicitly; `scripts/v0_session_selfcheck.py` and the
  test suite both do. The `declared=True` guard rejects undeclared dimensions before any
  training work starts.
- Operational note (learned 2026-09-13): background processes must be launched with
  `setsid nohup ... < /dev/null &` from the terminal tool - a plain `nohup ... &` child is
  killed when the tool call that spawned it returns.
- Local proof run (2026-09-13): `scripts/local_proof_run.py` executed the section 5.1
  protocol machinery end-to-end on the local GPU with the production runner at reduced
  dims (12.55M total / 6.26M active parameters; vocab 8k, context 512, depth 8, 8 experts
  of 2 active, cap-4/1-batch primary controller; E=3; steps=600; batch 8) over the declared
  queue 9000-9011. All 12 pairs published complete; `scripts/v0_verdict.py` returned
  PASS_WITH_QUALIFICATION: measured probe cost 1.0044% (primary <=5% PASS), the declared
  285M band [0.0471%, 0.1883%] correctly not applied to a reduced model, harm guard clear
  (7/12 adaptive wins - unpowered at this scale, as expected for the cost-only protocol).
  The section 5.1 FLOPs identity S = E*k*Bp*2*N_active reproduced exactly (search FLOPs
  bit-match the formula), measured cost within 0.4% relative of the reduced-dims identity
  prediction (1.0014% vs 0.9977% on seed 9000) - the identity is dimension-independent,
  which is precisely what the tolerance band is designed to test at declared scale.
  Scratch roots: `artifacts/local-proof/{sessions,datasets}` (gitignored); verdict JSON at
  `artifacts/local-proof/datasets/v0-5-1-verdict.json`.
- `startup.sh` has a separate preview-process exit-code issue to clean up; it is not part of
  the scientific implementation and should not be solved by killing the unrelated 8080 service.
