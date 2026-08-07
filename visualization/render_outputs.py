"""Render the value heatmap and policy-arrows figures for a fixed
(k, energy) slice of a saved Value Iteration (or any Q-table-based)
run, and save them under results/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from environments.generator import load_map, WALL
from visualization import renderer as rd


def main(argv=None):
    """CLI entry point: load a saved V/policy and render the two static figures.

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
        description="Render value-heatmap and policy-arrows figures from a saved run."
    )
    parser.add_argument("--map-name", type=str, default="source")
    parser.add_argument("--algorithm", type=str, default="value_iteration",
                         help="Sub-directory under results/models/ the run lives in "
                              "(e.g. value_iteration, q_learning, sarsa_lambda).")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--k", type=int, default=0, help="Key-state slice (0 or 1).")
    parser.add_argument("--energy", type=int, required=True,
                         help="Energy-state slice to render (must be <= run's max_energy).")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Defaults to results/figures/<algorithm>/<run_id>/")
    args = parser.parse_args(argv)

    map_spec = load_map("environments/maps", args.map_name)
    wall_mask = map_spec.grid == WALL

    model_dir = Path("results/models") / args.algorithm / args.run_id
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/figures") / args.algorithm / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if (model_dir / "V.npy").exists():
        V = np.load(model_dir / "V.npy")
        fig = rd.render_value_heatmap(
            V[:, :, args.k, args.energy], wall_mask,
            title=f"V, k={args.k}, energy={args.energy}",
        )
        out_path = output_dir / f"value_heatmap_k{args.k}_e{args.energy}.png"
        fig.savefig(out_path, dpi=120)
        print(f"wrote {out_path}")

    if (model_dir / "policy.npy").exists():
        policy = np.load(model_dir / "policy.npy")
        terminal_mask = np.zeros_like(wall_mask)
        terminal_mask[map_spec.goal] = True
        fig = rd.render_policy_arrows(
            policy[:, :, args.k, args.energy], wall_mask, terminal_mask,
            title=f"Policy, k={args.k}, energy={args.energy}",
        )
        out_path = output_dir / f"policy_k{args.k}_e{args.energy}.png"
        fig.savefig(out_path, dpi=120)
        print(f"wrote {out_path}")
    elif (model_dir / "Q.npy").exists():
        from experiments.analysis import extract_policy_from_Q
        Q = np.load(model_dir / "Q.npy")
        policy = extract_policy_from_Q(Q)
        terminal_mask = np.zeros_like(wall_mask)
        terminal_mask[map_spec.goal] = True
        fig = rd.render_policy_arrows(
            policy[:, :, args.k, args.energy], wall_mask, terminal_mask,
            title=f"Policy (from Q), k={args.k}, energy={args.energy}",
        )
        out_path = output_dir / f"policy_k{args.k}_e{args.energy}.png"
        fig.savefig(out_path, dpi=120)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
