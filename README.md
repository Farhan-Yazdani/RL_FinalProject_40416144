# RL Final Project — Dynamic Maze — Reproduction Guide

This covers everything built so far: the environment, all three
algorithms, cross-algorithm comparison, and transfer learning. The
GUI, `tests/`, `main.py`, and `experiments/run_experiments.py` are
not included in this drop (still in progress).

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run every command below from the project root (the folder containing
`environments/`, `agents/`, `experiments/`, `transfer/`, `gui/`).
Everything is invoked as a module (`python -m ...`) so the internal
imports (e.g. `from environments.maze import ...`) resolve correctly.

## 2. Generate the maps (only needed once)

The three maps included in this drop (`environments/maps/*.json`)
were generated with `student_id="40416144"`, which resolves to
`base_seed=4, maze_size=15` — the values stated in the project spec.
To regenerate them from scratch (fully deterministic, byte-identical
output every time):

```bash
python -m environments.generate_maps --student-id 40416144
```

Optional flags (defaults shown match the spec and the shipped maps):
`--maps-dir environments/maps`, `--similar-change-fraction 0.175`,
`--different-change-fraction 0.4`, `--different-new-penalties 4`.

## 3. Run Value Iteration

```bash
python -m agents.value_iteration \
  --student-id 40416144 --map-name source \
  --gamma 0.95 --reward-version shaped \
  --run-id vi_matched_shaped
```

Change `--gamma` (spec requires ≥3 values) and re-run with a distinct
`--run-id` for each, e.g. `--gamma 0.7`, `--gamma 0.9`, `--gamma 0.99`.

Outputs:
- `results/raw_data/value_iteration/<run_id>/config.json` — resolved config
- `results/raw_data/value_iteration/<run_id>/metrics.csv` — per-iteration delta (convergence curve)
- `results/models/value_iteration/<run_id>/{V,policy,Q}.npy`

**Note:** omit `--max-energy` to use the default
(`2 * num_traversable_cells`, here 382), which is what the Q-Learning/
SARSA runs below use by default too — keep these consistent across
algorithms if you want to compare them later (see step 6).

## 4. Run Q-Learning

```bash
python -m agents.q_learning \
  --student-id 40416144 --map-name source \
  --n-episodes 8000 --alpha 0.2 --gamma 0.95 \
  --eps-schedule exponential --eps-start 1.0 --eps-end 0.02 \
  --reward-version shaped \
  --run-id ql_test3
```

This took ~54s and reached 100% success (last 100 episodes) in local
testing. To compare epsilon-decay schedules, also run with
`--eps-schedule linear`. To compare reward versions, also run with
`--reward-version sparse` (note: sparse reward converges much slower —
took only ~8% success at 3000 episodes in testing; consider more
episodes or a larger `--alpha` if using sparse alone).

Outputs: `results/raw_data/q_learning/<run_id>/{config.json,metrics.csv,events.log}`,
`results/models/q_learning/<run_id>/Q.npy`.

## 5. Run SARSA(λ)

Must be run at all 4 required λ values:

```bash
for lam in 0.0 0.3 0.7 0.9; do
  python -m agents.sarsa_lambda \
    --student-id 40416144 --map-name source \
    --lam $lam --n-episodes 4000 --alpha 0.15 --gamma 0.95 \
    --eps-schedule exponential --reward-version shaped \
    --run-id sarsa_lambda${lam}_test
done
```

Each takes 30–95s (higher λ is slower — more eligibility-trace entries
stay active per step). All four reached 100% success in testing, with
λ=0.9 converging slower and to a slightly worse final policy than
λ=0/0.3/0.7 — a good empirical basis for the report's "which λ is
best" question.

Outputs: `results/raw_data/sarsa_lambda/<run_id>/{config.json,metrics.csv,events.log,trace_detail.json}`,
`results/models/sarsa_lambda/<run_id>/Q.npy`.

`trace_detail.json` logs δ and the eligibility-trace norm/max for the
last full episode of training — exactly the "log δ and E for at least
one short episode" requirement.

## 6. Compare the three algorithms

**Important:** for a fair comparison, all three algorithms must be run
with the *same* `max_energy` (and ideally the same `--reward-version`).
The commands above all use the default `max_energy=382`, so they're
already consistent with each other.

```bash
python -m experiments.compare_runs \
  --map-name source \
  --vi-run-id vi_matched_shaped \
  --q-learning-run-id ql_test3 \
  --sarsa-run-id sarsa_lambda0.7_test \
  --output-json results/raw_data/comparison_vi_ql_sarsa.json
```

This prints raw policy-agreement (over the *entire* nominal state
space) and a reachable-states-only agreement (restricted to
`(x,y,k,energy)` combinations actually visited during training — the
raw number is misleading here since `energy` resets every episode, so
most `(position, energy)` combinations are never visited under any
sensible policy; see the `reachable_states_analysis` field).

## 7. Run transfer learning

Requires a completed Q-Learning source run (step 4) as input:

```bash
python -m transfer.transfer_learning \
  --student-id 40416144 --source-map source \
  --source-run-id ql_test3 \
  --n-episodes 1500 --alpha 0.15 --gamma 0.95 \
  --eps-schedule exponential --eps-start 0.5 --eps-end 0.02
```

This runs all 4 scenarios (scratch / full / scaled×3 β values /
selective) on both target maps (`transfer_similar`,
`transfer_different`) — 12 runs total, each ~15-25s at 1500 episodes.

Outputs per scenario:
`results/raw_data/transfer_learning/transfer_<target>_<scenario>[_beta<b>]/{config.json,metrics.csv}`,
`results/models/transfer_learning/.../{Q_initial,Q_final}.npy`.

`config.json` includes a `negative_transfer_example` field (may be
`null` if none was found in that particular run — it was found
reliably for the `transfer_different_full` scenario in testing) with
the specific state, its transferred Q-values, and why they were wrong
for the new map.

A summary file is also written to
`results/raw_data/transfer_learning/<target>_summary.json` with
initial/final success rate per scenario.

## 8. Regenerate visual outputs

```bash
python -m visualization.render_outputs \
  --map-name source --algorithm q_learning --run-id ql_test3
```

(swap in your own `--algorithm`/`--run-id`; output defaults to
`results/figures/<algorithm>/<run_id>/`, or pass `--output-dir` to
override).

This writes a single 2x2 figure (`policy_and_value_k0_k1.png`): value
heatmaps for `k=0`/`k=1` on the top row and policy arrows for
`k=0`/`k=1` on the bottom row. Instead of one fixed energy slice, each
`(x, y, k)` position uses the energy level visited most during
training (from the run's `events.log`); positions without a
visitation record fall back to `--energy`, which defaults to
`max_energy // 2`.

Other renderer functions available in `gui/renderer.py`:
`render_visitation_map`, `render_policy_disagreement_map`,
`render_qvalue_diff_map`, `render_agent_path` — each takes the arrays
already saved under `results/models/` and `results/raw_data/`.
<!--
## 9. Launch the interactive GUI (needs a display)

```bash
python -m visualization.app --student-id 40416144 --map-name source \
  --policy-npy results/models/value_iteration/vi_matched_shaped/policy.npy
```

Controls: mouse-click Start/Stop/Resume/Reset/Re-run/Policy-toggle/
Speed±, or press Space to toggle animation and R to reset. Omit
`--policy-npy` to run in random-action "manual" mode instead of
policy-eval mode.

## Notes on reproducibility

- All map generation and env sampling uses seeded
  `np.random.default_rng`, never global numpy random state — same
  `--seed`/`--student-id` always reproduces the same result.
- Every run writes a full `config.json` snapshot — nothing here is a
  bare hyperparameter; every number used is traceable to a specific
  run's config file.
- If you're running multiple seeds for the
  final report, keep every run (don't discard failed/worse ones) and
  aggregate mean±std — `experiments/analysis.py`'s
  `stability_across_runs()` does this for you given a list of loaded
  metrics DataFrames.
-->


## 9. Launch the interactive GUI

```bash
python -m gui.app
```

<details>
<summary><strong>GUI workflow, controls, and things that will bite you if you skip this (click to expand)</strong></summary>

<!-- fig: results/figures/gui/full_window_overview.png -->
![full_window_overview.png](./results/figures/gui/full_window_looking_for_key.png)


**Workflow:** in the sidebar, pick a **Map** and **Algorithm**, set
hyperparameters, then click **Build / Apply** — nothing happens from the
dropdowns alone until you click it. Leave **Learned Model** as *"None (train
fresh)"* to train live from scratch, or pick a completed run from that
dropdown to import a saved `Q`/`V`/`policy` from `results/models/` and replay
it (this auto-fills the sidebar from that run's `config.json` and forces
**Eval** mode — an imported model does not keep training). Use **Start /
Stop / Resume / Reset / Re-run** to drive it: Reset restarts only the current
episode, Re-run rebuilds the whole session from scratch. The GUI is a live
*visualization/demo* tool — it is not the source of the project's reported
numbers; those come from the batch CLI scripts in §3–§7.

**Things you'll misuse if nobody tells you:**
- **Value Iteration's first "Start" click doesn't animate anything** — it
  runs the full Bellman sweep in a background thread first (can take
  15–20+ seconds); click Start again once it reports convergence to see the
  greedy rollout.
- **`total_episodes` in the GUI does not stop training at that count** —
  it only shapes the ε-decay curve. A live Q-Learning/SARSA session will run
  indefinitely past it at `eps_end`. Only the CLI's `--n-episodes` is a real
  cap.
- **The Visitation Count tab for Value Iteration will look almost empty
  even after convergence** — VI's "live" stepping is one greedy rollout, not
  exploration, so don't expect it to resemble Q-Learning/SARSA's map.
- **Analytics tabs show a single fixed `(k, energy)` slice** (set via the
  "Analytics Slice" controls) — at extreme energy values the visitation/value
  views can look sparse or arbitrary simply because that slice was rarely or
  never visited, not because anything is broken.
- **The "Policy Disagreement" tab is currently a non-functional
  placeholder** — it's visible but never updates in this drop.

Full details, every control explained, and the complete list of
non-obvious/nonsensical-looking results are in
[`gui/gui_guide.md`](gui/gui_guide.md).

</details>