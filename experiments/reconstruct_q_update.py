"""Manually reconstruct one real Q-Learning update from the run's event log.

Satisfies the final_project.md requirement: "At least one real Q-update
must be selected from the log file and manually reconstructed in the
report." Instead of presenting final Q-table numbers (which say nothing
about the *update* that produced them), this tool **replays the run's
event log from episode 0** through the selected ``(episode, step)``,
applying exactly the update from ``agents/q_learning.py``:

    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s,a)]

Because the log records every transition's ``state``, ``action``,
``reward``, and ``next_state``, and Q is initialized to zeros, the
replay reproduces the exact Q-table state that existed *before* the
selected step (the only piece not stored in the log is the ``done``
flag, which is fully determined by the episode boundary: the last step
of every episode is a ``done`` step). The tool prints the logged step
and the full arithmetic of its update. Running ``--verify`` additionally
replays the *entire* log and checks that the replayed Q-table matches
the saved ``Q.npy``, which validates both the replay and the logging.

Usage
-----
::

    python -m experiments.reconstruct_q_update --algorithm q_learning --run-id ql_test3
    python -m experiments.reconstruct_q_update --algorithm q_learning --run-id ql_test3 --episode 9 --step 243 --verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


def find_first_goal_step(condensed_path):
    """Return the ``(episode, step)`` of the first goal-reached event.

    Parameters
    ----------
    condensed_path : str or pathlib.Path
        Path to the run's ``condensed_events.log``.

    Returns
    -------
    tuple or None
        ``(episode, step)`` of the first ``goal_reached`` event, or
        ``None`` if the run never reached the goal.
    """
    with open(condensed_path) as f:
        for line in f:
            record = json.loads(line)
            for item in record["notable"]:
                if item["event"] == "goal_reached":
                    return record["episode"], item["step"]
    return None


def replay_to_step(events_log_path, Q_shape, alpha, gamma, stop_episode, stop_step,
                   saved_Q=None, verify_full=False):
    """Replay Q-Learning updates from the log; return before/after values at the target step.

    Parameters
    ----------
    events_log_path : str or pathlib.Path
        Full ``events.log`` (one JSON object per line).
    Q_shape : tuple of int
        ``(X, Y, 2, E, A)`` shape of the run's Q-table.
    alpha, gamma : float
        Learning rate and discount factor (from config.json).
    stop_episode, stop_step : int
        Target step whose update is reconstructed.
    saved_Q : ndarray, optional
        The run's saved ``Q.npy``, used only for the full-replay
        verification when ``verify_full`` is set.
    verify_full : bool
        If ``True``, replay the entire log from a zero table and report
        the max absolute difference against ``saved_Q`` (validates both
        the logging and this replay).

    Returns
    -------
    dict
        The logged row, ``Q_before`` value at ``Q[s,a]``, the bootstrap
        value ``max_a' Q(s', a')``, the computed TD error, the resulting
        ``Q_after`` value, whether the target step was ``done``, and
        (when ``verify_full``) the max absolute difference between the
        fully replayed Q-table and ``saved_Q``.
    """
    Q = np.zeros(Q_shape, dtype=np.float64)
    before_value = after_value = td_error = td_target = bootstrap = None
    logged_row = None
    logged_done = None
    max_dev_full = None

    def apply_update(row, done):
        nonlocal before_value, after_value, td_error, td_target, bootstrap, logged_row, logged_done
        s = row["state"]
        a = row["action"]
        r = row["reward"]
        s_next = row["next_state"]
        idx = (s[0], s[1], s[2], s[3], a)
        q_before = float(Q[idx])
        best_next = 0.0 if done else float(np.max(Q[s_next[0], s_next[1], s_next[2], s_next[3]]))
        target = r + gamma * best_next
        error = target - q_before
        Q[idx] = q_before + alpha * error
        if (row["episode"], row["step"]) == (stop_episode, stop_step):
            before_value, bootstrap, td_target = q_before, best_next, target
            td_error, after_value = error, float(Q[idx])
            logged_row, logged_done = row, done

    with open(events_log_path) as f:
        prev = None
        for line in f:
            cur = json.loads(line)
            if prev is not None:
                apply_update(prev, done=(cur["episode"] != prev["episode"]))
            prev = cur
        if prev is not None:
            apply_update(prev, done=True)

    if verify_full:
        max_dev_full = float(np.max(np.abs(Q - saved_Q))) if saved_Q is not None else None

    return {
        "logged_row": logged_row,
        "Q_before": before_value,
        "bootstrap": bootstrap,
        "td_target": td_target,
        "td_error": td_error,
        "Q_after": after_value,
        "done": logged_done,
        "max_dev_full": max_dev_full,
    }


def main(argv=None):
    """CLI entry point: reconstruct one real Q-update from a run's log.

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
        description="Reconstruct one real Q-Learning update from the event log."
    )
    parser.add_argument("--algorithm", type=str, default="q_learning",
                        choices=["q_learning"])
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--episode", type=int, default=None,
                        help="Episode of the target step (default: first goal-reached episode).")
    parser.add_argument("--step", type=int, default=None,
                        help="Step index within that episode (default: the goal-reached step).")
    parser.add_argument("--verify", action="store_true",
                        help="Replay the whole log and compare with the saved Q.npy.")
    args = parser.parse_args(argv)

    raw_dir = Path("results/raw_data") / args.algorithm / args.run_id
    events_path = raw_dir / "events.log"
    if not events_path.exists():
        raise SystemExit(
            f"{events_path} not found. Full-log replay requires the raw events.log "
            "(produced by `python -m agents.{args.algorithm}` or a fresh training run)."
        )
    with open(raw_dir / "config.json") as f:
        config = json.load(f)
    Q = np.load(Path("results/models") / args.algorithm / args.run_id / "Q.npy")

    if args.episode is None or args.step is None:
        target = find_first_goal_step(raw_dir / "condensed_events.log")
        if target is None:
            raise SystemExit("no goal-reached step found; pass --episode/--step explicitly")
        episode, step = target
    else:
        episode, step = args.episode, args.step

    result = replay_to_step(
        events_path, Q.shape, config["alpha"], config["gamma"], episode, step,
        saved_Q=Q if args.verify else None, verify_full=args.verify,
    )
    row = result["logged_row"]
    if row is None:
        raise SystemExit(f"no logged step at episode={episode}, step={step}")

    print("Logged transition (from events.log):")
    print(f"  episode={row['episode']}  step={row['step']}")
    print(f"  s={row['state']}  a={row['action']} ({ACTION_NAMES[row['action']]})")
    print(f"  r={row['reward']:+.1f}  event={row['event']}")
    print(f"  s'={row['next_state']}")
    print(f"  done={result['done']}")
    print()
    print("Manual reconstruction of the update:")
    print(f"  Q(s,a) before        = {result['Q_before']:.6f}")
    print(f"  max_a' Q(s',a')      = {result['bootstrap']:.6f}")
    print(f"  r + gamma*max Q(s')  = {result['td_target']:.6f}")
    print(f"  TD error             = {result['td_error']:.6f}")
    print(f"  Q(s,a) after         = {result['Q_after']:.6f}")
    if result["max_dev_full"] is not None:
        print()
        print(f"Full-log replay verification: max |replayed Q - saved Q.npy| "
              f"= {result['max_dev_full']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
