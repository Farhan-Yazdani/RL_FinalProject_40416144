"""Render the state-visitation map from the compact ``visitation_counts.npy``.

Satisfies the "Visitation map" required visual output of final_project.md
("Number of visits to each state during training"). Reads the compact,
trackable ``visitation_counts.npy`` artifact written by
``experiments.condense_events`` (falling back to ``events.log`` if the
compact artifact is missing), sums over the energy axis, and renders one
log-scaled heatmap per key state.

Usage
-----
::

    python -m visualization.render_visitation --algorithm q_learning --run-id ql_test3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agents.policy_extraction import _visitation_counts_from_events_log
from environments.generator import WALL, load_map
from visualization import renderer as rd
from visualization.render_agreement import plt


def main(argv=None):
    """CLI entry point: render the visitation map for one run.

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
        description="Render the per-state visitation map of a trained run."
    )
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=["q_learning", "sarsa_lambda"])
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--map-name", type=str, default="source")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args(argv)

    map_spec = load_map("environments/maps", args.map_name)
    wall_mask = map_spec.grid == WALL
    shape = (map_spec.grid.shape[0], map_spec.grid.shape[1], 2)

    raw_dir = Path("results/raw_data") / args.algorithm / args.run_id
    counts_path = raw_dir / "visitation_counts.npy"
    events_path = raw_dir / "events.log"
    if counts_path.exists():
        counts = np.load(counts_path)
    else:
        counts = _visitation_counts_from_events_log(events_path, shape)
        if counts is None:
            raise SystemExit(
                f"no visitation_counts.npy or events.log at {raw_dir}"
            )
    if counts.ndim != 4 or counts.shape[:2] != shape[:2] or counts.shape[2] != 2:
        raise SystemExit(
            f"unexpected visitation_counts shape {counts.shape}; expected "
            f"(X, Y, 2, E)."
        )

    visitation_by_k = counts.sum(axis=-1)  # sum over energy

    fig = rd.render_visitation_panels(visitation_by_k, wall_mask)
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("results/figures") / args.algorithm / args.run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "visitation_k0_k1.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
