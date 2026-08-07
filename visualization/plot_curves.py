"""Learning-curve and convergence-curve charts generated from saved metrics.

Produces the report's statistical charts entirely from the ``metrics.csv``
files persisted in ``results/raw_data`` (never retrains anything,
``CODING_STYLE.md`` 1.8). Three kinds of charts:

- ``ql_eps`` -- smoothed per-episode reward curves of two Q-Learning
  runs (e.g. exponential vs. linear epsilon decay), for the
  "at least two decay schedules must be implemented and compared"
  requirement of final_project.md.
- ``sarsa_lambda`` -- smoothed reward curves of the four SARSA(λ) runs
  (λ = 0, 0.3, 0.7, 0.9), for the "which λ is best" question.
- ``vi_gamma`` -- Value Iteration convergence (per-iteration Bellman
  delta, log scale) for several γ values, for the "effect of at least
  three different discount factor values" requirement.

Usage
-----
::

    python -m visualization.plot_curves --kind ql_eps --run-ids ql_test3 ql_linear_test
    python -m visualization.plot_curves --kind sarsa_lambda
    python -m visualization.plot_curves --kind vi_gamma --gammas 0.7 0.9 0.95 0.99
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load_metrics(algorithm: str, run_id: str) -> pd.DataFrame:
    """Load a run's per-episode metrics.csv into a DataFrame."""
    path = Path("results/raw_data") / algorithm / run_id / "metrics.csv"
    if not path.exists():
        raise SystemExit(f"no metrics.csv at {path}")
    return pd.read_csv(path)


def plot_ql_eps_curves(run_ids: list, output_dir: Path) -> Path:
    """Plot smoothed reward curves for several Q-Learning epsilon schedules.

    Parameters
    ----------
    run_ids : list of str
        Q-Learning run ids (one curve each, labeled by their
        ``--eps-schedule`` from config.json).
    output_dir : pathlib.Path
        Directory to save the figure into.

    Returns
    -------
    pathlib.Path
        Path of the saved ``ql_eps_decay_reward_curves.png``.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for run_id in run_ids:
        metrics = _load_metrics("q_learning", run_id)
        cfg_dir = Path("results/raw_data/q_learning") / run_id
        schedule = "exponential"
        with open(cfg_dir / "config.json") as f:
            import json
            schedule = json.load(f)["eps_schedule"]
        reward = metrics["reward"].rolling(200, min_periods=50).mean()
        ax.plot(metrics["episode"], reward, label=f"{schedule} decay ({run_id})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (rolling 200-episode mean)")
    ax.set_title("Q-Learning: reward under exponential vs. linear $\\epsilon$ decay")
    ax.legend()
    fig.tight_layout()
    out = output_dir / "ql_eps_decay_reward_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_sarsa_lambda_curves(run_ids: list, output_dir: Path) -> Path:
    """Plot smoothed reward curves for the four SARSA(λ) runs.

    Parameters
    ----------
    run_ids : list of str
        SARSA run ids in increasing-λ order (one curve each).
    output_dir : pathlib.Path
        Directory to save the figure into.

    Returns
    -------
    pathlib.Path
        Path of the saved ``sarsa_lambda_reward_curves.png``.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for run_id in run_ids:
        metrics = _load_metrics("sarsa_lambda", run_id)
        reward = metrics["reward"].rolling(200, min_periods=50).mean()
        ax.plot(metrics["episode"], reward, label=f"$\\lambda$={run_id.split('lambda')[-1].split('_')[0]}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (rolling 200-episode mean)")
    ax.set_title("SARSA(λ): learning curves for λ = 0, 0.3, 0.7, 0.9")
    ax.legend()
    fig.tight_layout()
    out = output_dir / "sarsa_lambda_reward_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_vi_gamma_convergence(gammas: list, run_id_by_gamma, output_dir: Path) -> Path:
    """Plot Value Iteration per-iteration convergence delta for several γ values.

    Parameters
    ----------
    gammas : list of float
        Discount factors to plot.
    run_id_by_gamma : dict
        Maps each γ value to its run id.
    output_dir : pathlib.Path
        Directory to save the figure into.

    Returns
    -------
    pathlib.Path
        Path of the saved ``vi_gamma_convergence.png``.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for gamma in gammas:
        metrics = _load_metrics("value_iteration", run_id_by_gamma[gamma])
        ax.semilogy(
            metrics["iteration"],
            metrics["delta"],
            label=f"$\\gamma$={gamma} (converged in {len(metrics)} iterations)",
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("$\\max_s |V_{k+1}(s) - V_k(s)|$ (log scale)")
    ax.set_title("Value Iteration: convergence for different $\\gamma$")
    ax.legend()
    fig.tight_layout()
    out = output_dir / "vi_gamma_convergence.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv=None):
    """CLI entry point: render one kind of metrics-based chart.

    Parameters
    ----------
    argv : list of str, optional
        Argument list (defaults to ``sys.argv[1:]`` if ``None``).

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Render learning / convergence curves from saved metrics.csv files."
    )
    parser.add_argument("--kind", type=str, required=True,
                        choices=["ql_eps", "sarsa_lambda", "vi_gamma"])
    parser.add_argument("--run-ids", type=str, nargs="+", default=None,
                        help="Run ids for ql_eps (default: ql_test3 ql_linear_test) "
                             "or sarsa_lambda (default: the four sarsa_lambda*_test runs).")
    parser.add_argument("--gammas", type=float, nargs="+", default=None,
                        help="γ values for vi_gamma (default: 0.7 0.9 0.95 0.99).")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args(argv)

    if args.kind == "ql_eps":
        run_ids = args.run_ids or ["ql_test3", "ql_linear_test"]
        output_dir = Path(args.output_dir) if args.output_dir else Path("results/figures/q_learning")
    elif args.kind == "sarsa_lambda":
        run_ids = args.run_ids or [f"sarsa_lambda{lam}_test" for lam in ("0.0", "0.3", "0.7", "0.9")]
        output_dir = Path(args.output_dir) if args.output_dir else Path("results/figures/sarsa_lambda")
    else:  # vi_gamma
        gammas = args.gammas or [0.7, 0.9, 0.95, 0.99]
        run_id_by_gamma = {
            0.7: "vi_gamma0.7", 0.9: "vi_gamma0.9",
            0.95: "vi_matched_shaped", 0.99: "vi_gamma0.99",
        }
        output_dir = Path(args.output_dir) if args.output_dir else Path("results/figures/value_iteration")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "ql_eps":
        out = plot_ql_eps_curves(run_ids, output_dir)
    elif args.kind == "sarsa_lambda":
        out = plot_sarsa_lambda_curves(run_ids, output_dir)
    else:  # vi_gamma
        out = plot_vi_gamma_convergence(gammas, run_id_by_gamma, output_dir)

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
