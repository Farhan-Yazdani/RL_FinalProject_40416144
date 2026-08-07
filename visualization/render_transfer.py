"""Transfer-learning charts for the report, generated from saved run data.

Implements the "Transfer learning" required visual output of
final_project.md ("Difference in Q-values or policy before and after
transfer") and provides the learning-curve evidence for the report's
transfer section. Reads only ``results/raw_data/transfer_learning`` and
``results/models/transfer_learning`` (``CODING_STYLE.md`` 1.8); never
retrains anything.

Two kinds of charts:

- ``curves`` -- smoothed per-episode reward of every transfer scenario
  (scratch, full, scaled with the three β values, selective) on one or
  both target maps, for the "initial performance / learning speed /
  final performance" comparison of final_project.md.
- ``qdiff`` -- heatmap of ``max_{k,e,a} |Q_after - Q_before|`` between
  the transferred initialization (``Q_initial.npy``) and the continued
  training result (``Q_final.npy``), for the "Difference in Q-values
  before and after transfer" required visual output.

Usage
-----
::

    python -m visualization.render_transfer --kind curves
    python -m visualization.render_transfer --kind qdiff --target transfer_different --scenario full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environments.generator import WALL, load_map
from visualization import renderer as rd


def _load_metrics(run_id: str) -> pd.DataFrame:
    """Load a transfer run's per-episode metrics.csv."""
    path = Path("results/raw_data/transfer_learning") / run_id / "metrics.csv"
    if not path.exists():
        raise SystemExit(f"no metrics.csv at {path}")
    return pd.read_csv(path)


def _scenario_label(run_id: str) -> str:
    """Human label for a transfer run id, e.g. ``scaled (β=0.5)``."""
    scenario = run_id.split("_")[-1]
    if scenario.startswith("beta"):
        return f"scaled ($\\beta$={scenario[4:]})"
    return scenario


def plot_transfer_curves(output_dir: Path,
                         targets=("transfer_similar", "transfer_different")) -> Path:
    """Plot smoothed reward curves of all scenarios for one or both targets.

    Parameters
    ----------
    targets : tuple of str, default=("transfer_similar", "transfer_different")
        Target-map names to plot (one panel each).
    output_dir : pathlib.Path
        Directory to save the figure into.

    Returns
    -------
    pathlib.Path
        Path of the saved ``transfer_reward_curves.png``.
    """
    scenario_run_ids = {
        "scratch": f"transfer_{targets[0]}_scratch",
        "full": f"transfer_{targets[0]}_full",
        "scaled_beta0.25": f"transfer_{targets[0]}_scaled_beta0.25",
        "scaled_beta0.5": f"transfer_{targets[0]}_scaled_beta0.5",
        "scaled_beta0.75": f"transfer_{targets[0]}_scaled_beta0.75",
        "selective": f"transfer_{targets[0]}_selective",
    }

    fig, axes = plt.subplots(1, len(targets), figsize=(13, 4.6),
                             constrained_layout=True, sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        for scenario in scenario_run_ids:
            run_id = f"transfer_{target}_{scenario}"
            metrics = _load_metrics(run_id)
            reward = metrics["reward"].rolling(100, min_periods=25).mean()
            label = "scratch" if scenario == "scratch" else _scenario_label(f"x_{scenario}")
            ax.plot(metrics["episode"], reward, label=label, linewidth=1.4)
        ax.set_xlabel("Episode")
        ax.set_title("similar target (15–20 % walls moved)"
                     if target == "transfer_similar"
                     else "different target (≥35 % change, key moved)")
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("Reward (rolling 100-episode mean)")
    fig.suptitle("Transfer learning: reward curves per scenario")

    out = output_dir / "transfer_reward_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_qvalue_diff(target: str, scenario: str, output_dir: Path,
                     map_name: str = None) -> Path:
    """Render the before/after Q-value change map for one transfer scenario.

    Parameters
    ----------
    target : str
        Target map name (e.g. ``transfer_different``).
    scenario : str
        Scenario run suffix (e.g. ``full`` or ``scaled_beta0.5``).
    output_dir : pathlib.Path
        Directory to save the figure into.
    map_name : str, optional
        Map whose wall mask to use (defaults to ``target``).

    Returns
    -------
    pathlib.Path
        Path of the saved ``qvalue_diff_<target>_<scenario>.png``.
    """
    run_id = f"transfer_{target}_{scenario}"
    model_dir = Path("results/models/transfer_learning") / run_id
    Q_before = np.load(model_dir / "Q_initial.npy")
    Q_after = np.load(model_dir / "Q_final.npy")

    map_spec = load_map("environments/maps", map_name or target)
    wall_mask = map_spec.grid == WALL

    fig = rd.render_qvalue_diff_map(
        Q_before, Q_after, wall_mask,
        title=f"Q-value change after continued training ({target} / {scenario})",
    )
    out = output_dir / f"qvalue_diff_{target}_{scenario}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv=None):
    """CLI entry point: render one kind of transfer chart."""
    parser = argparse.ArgumentParser(
        description="Render transfer-learning charts from saved run data."
    )
    parser.add_argument("--kind", type=str, required=True,
                        choices=["curves", "qdiff"])
    parser.add_argument("--target", type=str, default="transfer_different",
                        help="Target map for qdiff (e.g. transfer_different).")
    parser.add_argument("--scenario", type=str, default="full",
                        help="Scenario for qdiff (e.g. full, scaled_beta0.5, selective).")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("results/figures/transfer_learning")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "curves":
        out = plot_transfer_curves(output_dir=output_dir)
    else:
        out = plot_qvalue_diff(args.target, args.scenario, output_dir)

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
