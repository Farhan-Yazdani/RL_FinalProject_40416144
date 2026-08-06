"""Compare a completed Value Iteration run against Q-Learning and
SARSA(lambda) runs: policy agreement, runtime, samples, memory, path
quality. Reads only from results/raw_data and results/models (per
CODING_STYLE.md 1.8 -- never retrains anything).
"""

from __future__ import annotations

import argparse
import json

from environments.generator import WALL, load_map
from experiments.analysis import compare_algorithms, load_run


def main(argv=None):
    """CLI entry point: load three runs and print the full comparison table.

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
        description="Compare Value Iteration, Q-Learning, and SARSA(lambda) runs."
    )
    parser.add_argument("--map-name", type=str, default="source")
    parser.add_argument("--vi-run-id", type=str, required=True)
    parser.add_argument("--q-learning-run-id", type=str, required=True)
    parser.add_argument("--sarsa-run-id", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None,
                         help="Optional path to also write the comparison result as JSON "
                              "(agreement_mask arrays are dropped from this output).")
    args = parser.parse_args(argv)

    map_spec = load_map("environments/maps", args.map_name)
    wall_mask = map_spec.grid == WALL

    vi_run = load_run("value_iteration", args.vi_run_id)
    ql_run = load_run("q_learning", args.q_learning_run_id)
    sarsa_run = load_run("sarsa_lambda", args.sarsa_run_id)

    ql_events = f"results/raw_data/q_learning/{args.q_learning_run_id}/events.log"
    sarsa_events = f"results/raw_data/sarsa_lambda/{args.sarsa_run_id}/events.log"

    result = compare_algorithms(
        vi_run, ql_run, sarsa_run, wall_mask,
        q_learning_events_log_path=ql_events,
        sarsa_events_log_path=sarsa_events,
    )

    print(f"VI runtime: {result['vi_runtime_seconds']:.4f}s, "
          f"iterations: {result['vi_n_iterations']}")
    print()
    for name in ("q_learning_vs_vi", "sarsa_vs_vi"):
        print(f"{name}:")
        for k, v in result[name].items():
            if k != "agreement_mask":
                print(f"  {k}: {v}")
        print()

    if args.output_json:
        serializable = {
            "vi_runtime_seconds": result["vi_runtime_seconds"],
            "vi_n_iterations": result["vi_n_iterations"],
            "q_learning_vs_vi": {k: v for k, v in result["q_learning_vs_vi"].items() if k != "agreement_mask"},
            "sarsa_vs_vi": {k: v for k, v in result["sarsa_vs_vi"].items() if k != "agreement_mask"},
        }
        with open(args.output_json, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Comparison written to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
