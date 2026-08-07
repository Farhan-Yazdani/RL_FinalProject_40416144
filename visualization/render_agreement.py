"""Color-coded policy-agreement maps between a model-free run and Value Iteration.

Implements the final_project.md comparison requirement that differences
between the model-free greedy policy and the VI reference "be shown on a
color-coded map", and collects the concrete example states needed to
answer analytical question 5 ("find three states where the model-free
policy differs from the VI policy, and analyze the cause").

Both policies are derived per-position (one action per ``(x, y, k)``)
from their saved Q-tables via ``agents.policy_extraction`` -- for the
model-free run using each position's most-visited energy level (from the
compact ``visitation_counts.npy``, or ``events.log``), and for VI using
the same half-energy fallback the 4-panel renderer uses. Reads only from
``results/models`` and ``results/raw_data`` (``CODING_STYLE.md`` 1.8).

Usage
-----
::

    python -m visualization.render_agreement --algorithm q_learning --run-id ql_test3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.policy_extraction import derive_v_and_policy_from_Q
from environments.generator import DOOR, GOAL, KEY, NORMAL, PENALTY, WALL, load_map
from visualization import renderer as rd

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


def collect_example_states(agreement_mask, mf_policy, vi_policy, map_spec) -> list:
    """Describe up to ``n`` disagreement positions for the report's analysis.

    Parameters
    ----------
    agreement_mask : ndarray of shape (X, Y, 2), dtype=bool
        ``True`` where the two policies agree.
    mf_policy, vi_policy : ndarray of shape (X, Y, 2), dtype=int
        Model-free and VI per-position greedy actions.
    map_spec : MapSpec
        The map, for the local cell-type context of each position.

    Returns
    -------
    list of dict
        One entry per disagreement state with keys ``"x"``, ``"y"``,
        ``"k"``, ``"vi_action"``, ``"model_free_action"``, ``"cell"``
        (grid value at (x, y)) and ``"neighbors"`` (grid values of the
        four orthogonal neighbours).
    """
    examples = []
    for x in range(map_spec.maze_size):
        for y in range(map_spec.maze_size):
            if map_spec.grid[x, y] == WALL:
                continue
            for k in range(2):
                if agreement_mask[x, y, k]:
                    continue
                nx = map_spec.grid[x, max(0, y - 1)]
                sx = map_spec.grid[x, min(map_spec.maze_size - 1, y + 1)]
                wx = map_spec.grid[max(0, x - 1), y]
                ex = map_spec.grid[min(map_spec.maze_size - 1, x + 1), y]
                examples.append({
                    "x": x, "y": y, "k": k,
                    "vi_action": ACTION_NAMES[int(vi_policy[x, y, k])],
                    "model_free_action": ACTION_NAMES[int(mf_policy[x, y, k])],
                    "cell": int(map_spec.grid[x, y]),
                    "neighbors": [int(nx), int(sx), int(wx), int(ex)],
                })
    return examples


def main(argv=None):
    """CLI entry point: render agreement maps + example-state JSON for one run.

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
        description="Render policy-agreement maps between a model-free run and Value Iteration."
    )
    parser.add_argument("--map-name", type=str, default="source")
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=["q_learning", "sarsa_lambda"])
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--vi-run-id", type=str, default="vi_matched_shaped")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args(argv)

    map_spec = load_map("environments/maps", args.map_name)
    wall_mask = map_spec.grid == WALL

    mf_dir = Path("results/models") / args.algorithm / args.run_id
    vi_dir = Path("results/models/value_iteration") / args.vi_run_id
    mf_events = Path("results/raw_data") / args.algorithm / args.run_id / "events.log"
    vi_events = Path("results/raw_data") / "value_iteration" / args.vi_run_id / "events.log"

    Q_mf = np.load(mf_dir / "Q.npy")
    Q_vi = np.load(vi_dir / "Q.npy")

    _, mf_policy = derive_v_and_policy_from_Q(Q_mf, events_log_path=mf_events)
    _, vi_policy = derive_v_and_policy_from_Q(Q_vi, events_log_path=vi_events)

    agreement = vi_policy == mf_policy  # shape (X, Y, 2)

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("results/figures") / args.algorithm / args.run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = rd.render_policy_disagreement_panels(
        agreement, wall_mask,
        title=f"{args.algorithm}/{args.run_id} vs VI/{args.vi_run_id}",
    )
    fig_path = output_dir / "policy_disagreement_k0_k1.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    examples = collect_example_states(agreement, mf_policy, vi_policy, map_spec)
    n_non_wall = int(np.sum(~wall_mask))
    details = {
        "n_non_wall_positions": n_non_wall,
        "agreement_fraction_k0": float(np.sum(agreement[:, :, 0])) / max(1, n_non_wall),
        "agreement_fraction_k1": float(np.sum(agreement[:, :, 1])) / max(1, n_non_wall),
        "n_example_disagreement_states": len(examples),
        "examples": examples[:50],
    }
    details_path = output_dir / "policy_agreement_details.json"
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2)

    print(
        f"wrote {fig_path}\nwrote {details_path}\n"
        f"per-position agreement: k=0 {details['agreement_fraction_k0']:.3f}, "
        f"k=1 {details['agreement_fraction_k1']:.3f} "
        f"({len(examples)} disagreement states)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
