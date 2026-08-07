"""Render a single 2x2 figure from a saved Q-table-based run and save it
under results/figures/.

Layout (both key states in one figure, so there is no ``--k`` flag):

    +-------------------------+-------------------------+
    |  Value heatmap, k=0     |  Value heatmap, k=1     |
    |  (no key)               |  (has key)              |
    +-------------------------+-------------------------+
    |  Policy arrows, k=0     |  Policy arrows, k=1     |
    |  (no key)               |  (has key)              |
    +-------------------------+-------------------------+

Instead of picking a single fixed ``(k, energy)`` slice, each
``(x, y, k)`` position uses the energy level that was visited *most
during training* (from the run's ``visitation_counts.npy`` compact
artifact, or its ``events.log`` under ``results/raw_data``). Positions
without a visitation record fall back to ``--energy``, which defaults
to ``max_energy // 2`` -- a far more representative default than an
arbitrary fixed slice, since energy decreases monotonically along a
trajectory and always resets to ``max_energy`` on episode start, so
only a thin position-correlated band of the energy axis is ever
visited.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agents.policy_extraction import derive_v_and_policy_from_Q
from environments.generator import load_map, WALL
from visualization import renderer as rd


def main(argv=None):
    """CLI entry point: load a saved Q-table and render the 2x2 figure.

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
        description="Render a 4-panel figure (value heatmaps and policy arrows "
                    "for both key states) from a saved run."
    )
    parser.add_argument("--map-name", type=str, default="source")
    parser.add_argument("--algorithm", type=str, default="value_iteration",
                         help="Sub-directory under results/models/ the run lives in "
                              "(e.g. value_iteration, q_learning, sarsa_lambda).")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--energy", type=int, default=None,
                         help="Fallback energy level used only for positions with "
                              "no most-visited-energy record (defaults to "
                              "max_energy // 2). Ignored wherever the run's "
                              "visitation_counts.npy / events.log provides a "
                              "most-visited energy.")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Defaults to results/figures/<algorithm>/<run_id>/")
    args = parser.parse_args(argv)

    map_spec = load_map("environments/maps", args.map_name)
    wall_mask = map_spec.grid == WALL

    model_dir = Path("results/models") / args.algorithm / args.run_id
    events_log = Path("results/raw_data") / args.algorithm / args.run_id / "events.log"

    output_dir = Path(args.output_dir) if args.output_dir else Path("results/figures") / args.algorithm / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    q_path = model_dir / "Q.npy"
    if not q_path.exists():
        raise SystemExit(
            f"No Q.npy found in {model_dir}. This renderer derives both the "
            "value heatmap and the policy from the run's Q-table."
        )

    Q = np.load(q_path)
    V, policy = derive_v_and_policy_from_Q(
        Q, events_log_path=events_log, default_energy=args.energy
    )

    terminal_mask = np.zeros_like(wall_mask)
    terminal_mask[map_spec.goal] = True
    key_mask = np.zeros_like(wall_mask)
    key_mask[map_spec.key_pos] = True

    fig = rd.render_combined_panels(
        V, policy, wall_mask,
        terminal_mask=terminal_mask,
        key_mask=key_mask,
        suptitle=f"{args.algorithm} / {args.run_id}",
    )
    out_path = output_dir / "policy_and_value_k0_k1.png"
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
