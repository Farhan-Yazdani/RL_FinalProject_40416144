"""Produce compact, trackable analytics artifacts from a run's full events.log.

The full ``events.log`` written by ``agents/q_learning.py`` /
``agents/sarsa_lambda.py`` (one JSON line per environment step) grows to
~130 MB for a typical run and is therefore not tracked in git. Running
this tool after a training run replaces it with two small, trackable
artifacts inside ``results/raw_data/<algorithm>/<run_id>/``:

- ``condensed_events.log`` -- one JSON line *per episode* (aggregated
  move-level event counters + time-stamped notable events; typically
  < 1 MB). Written by ``experiments.analysis.condense_events_log``.
- ``visitation_counts.npy`` -- the ``(X, Y, 2, max_energy+1)`` per-state
  visitation-count array (dtype=int64, ~1.4 MB for a 15x15 maze).

These two files let every downstream analytical tool -- the 4-panel
policy/value figures (``visualization.render_outputs``), the visitation
map, the reachable-states-aware agreement metric, and the policy
extraction in ``agents.policy_extraction`` -- run without the
multi-hundred-MB raw log being present (each reads the compact artifact
first and only falls back to ``events.log`` if it is missing).

Usage
-----
::

    python -m experiments.condense_events --algorithm q_learning --run-id ql_test3

The tool is read-only with respect to training: it consumes the already
written ``events.log`` and adds only the two compact artifacts next to
it (``CODING_STYLE.md`` 1.8).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from environments.generator import load_map
from experiments.analysis import condense_events_log, visitation_count_from_events


def main(argv=None):
    """CLI entry point: condense one run's events.log into compact artifacts.

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
        description="Condense a run's events.log into compact, trackable artifacts."
    )
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=["q_learning", "sarsa_lambda"])
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--map-name", type=str, default="source")
    args = parser.parse_args(argv)

    raw_dir = Path("results") / "raw_data" / args.algorithm / args.run_id
    events_path = raw_dir / "events.log"
    if not events_path.exists():
        print(f"error: {events_path} not found (nothing to condense).",
              file=__import__("sys").stderr)
        return 1

    with open(raw_dir / "config.json") as f:
        config = json.load(f)
    map_spec = load_map("environments/maps", args.map_name)
    state_space_shape = (
        map_spec.grid.shape[0],
        map_spec.grid.shape[1],
        2,
        config["max_energy"] + 1,
    )

    condensed_path = raw_dir / "condensed_events.log"
    records = condense_events_log(events_path, condensed_path)

    counts = visitation_count_from_events(events_path, state_space_shape)
    np.save(raw_dir / "visitation_counts.npy", counts)

    raw_bytes = events_path.stat().st_size
    condensed_bytes = condensed_path.stat().st_size
    print(
        f"{args.algorithm}/{args.run_id}: {raw_bytes / 1e6:.1f} MB events.log "
        f"-> {condensed_bytes / 1e6:.1f} MB condensed_events.log "
        f"({len(records)} episodes) + visitation_counts.npy "
        f"({counts.shape}, {counts.nbytes / 1e6:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
