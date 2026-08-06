"""Comparison metrics between Value Iteration, Q-Learning, and SARSA(lambda).

Implements the "Comparing the Three Algorithms" section of
final_project.md: policy agreement against the VI reference, runtime,
sample counts, run-to-run stability, memory usage, and path quality.
Reads only from ``results/raw_data`` / ``results/models`` -- never
retrains anything itself (``CODING_STYLE.md`` 1.8/2.4).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from environments.generator import WALL, load_map


def policy_agreement_mask(reference_policy: np.ndarray, other_policy: np.ndarray, wall_mask: np.ndarray):
    """Compute per-state agreement between a learned policy and the VI reference.

    Parameters
    ----------
    reference_policy : ndarray of shape (X, Y, 2, E)
        Value Iteration's greedy policy (ground truth).
    other_policy : ndarray of shape (X, Y, 2, E)
        Greedy policy extracted from a Q-Learning or SARSA(lambda) Q-table.
    wall_mask : ndarray of shape (X, Y), dtype=bool
        ``True`` where the cell is a wall; those (x, y, *, *) states
        are excluded from the agreement computation since they are
        never legally occupied.

    Returns
    -------
    agreement_mask : ndarray of shape (X, Y, 2, E), dtype=bool
        ``True`` where the two policies choose the same action, at
        non-wall positions. Wall positions are set to ``False`` (they
        contribute 0 to the numerator and are excluded from the
        denominator via ``valid_mask``).
    valid_mask : ndarray of shape (X, Y, 2, E), dtype=bool
        ``True`` at every non-wall position (broadcast over k, energy).
    agreement_fraction : float
        ``sum(agreement_mask) / sum(valid_mask)`` -- percentage of
        legally-occupiable states where the two policies agree.

    Notes
    -----
    Fully vectorized (``CODING_STYLE.md`` 2.1); the returned
    ``agreement_mask`` is exactly what a color-coded disagreement map
    should render (see ``gui/renderer.py``).
    """
    valid_2d = ~wall_mask
    valid_mask = np.broadcast_to(
        valid_2d[:, :, None, None], reference_policy.shape
    )
    agreement_mask = (reference_policy == other_policy) & valid_mask
    agreement_fraction = float(np.sum(agreement_mask) / max(1, np.sum(valid_mask)))
    return agreement_mask, valid_mask, agreement_fraction


def extract_policy_from_Q(Q: np.ndarray) -> np.ndarray:
    """Greedily extract a policy array from a learned Q-table.

    Parameters
    ----------
    Q : ndarray of shape (X, Y, 2, E, A)
        Learned action-value table (from Q-Learning or SARSA(lambda)).

    Returns
    -------
    ndarray of shape (X, Y, 2, E), dtype=int
        ``argmax`` over the action axis.
    """
    return np.argmax(Q, axis=-1).astype(np.int64)


def sample_count_from_metrics(metrics: pd.DataFrame) -> int:
    """Total environment steps (samples) consumed by a training run.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Per-episode metrics with a ``steps`` column.

    Returns
    -------
    int
        Sum of ``steps`` across all episodes -- the total number of
        environment interactions (samples) the algorithm required.
    """
    return int(metrics["steps"].sum())


def stability_across_runs(list_of_metrics: list) -> dict:
    """Aggregate run-to-run stability across multiple seeds of the same config.

    Parameters
    ----------
    list_of_metrics : list of pandas.DataFrame
        One per-episode metrics DataFrame per seed/run, all for the
        same hyperparameter configuration.

    Returns
    -------
    dict
        ``"final_reward_mean"``, ``"final_reward_std"``,
        ``"final_success_rate_mean"``, ``"final_success_rate_std"``:
        mean/std (across runs) of each run's last-100-episode average
        reward and success rate. Per ``CODING_STYLE.md`` 1.8, this
        aggregates *all* provided runs (mean +/- std) -- callers must
        not discard failed runs or report only the best seed.
    """
    final_rewards = [m["reward"].tail(100).mean() for m in list_of_metrics]
    final_success = [m["success"].tail(100).mean() for m in list_of_metrics]
    return {
        "final_reward_mean": float(np.mean(final_rewards)),
        "final_reward_std": float(np.std(final_rewards)),
        "final_success_rate_mean": float(np.mean(final_success)),
        "final_success_rate_std": float(np.std(final_success)),
        "n_runs": len(list_of_metrics),
    }


def memory_usage_bytes(*arrays: np.ndarray) -> int:
    """Total memory footprint of one or more saved model arrays.

    Parameters
    ----------
    *arrays : ndarray
        Any number of arrays (e.g. V-table, Q-table, policy array).

    Returns
    -------
    int
        Sum of ``.nbytes`` across all provided arrays.
    """
    return int(sum(a.nbytes for a in arrays))


def path_quality(metrics: pd.DataFrame, window: int = 100) -> dict:
    """Summarize path quality (steps-to-goal, success rate) over the last ``window`` episodes.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Per-episode metrics with ``steps`` and ``success`` columns.
    window : int, default=100
        Number of final episodes to average over.

    Returns
    -------
    dict
        ``"avg_steps_on_success"`` (mean steps among successful
        episodes only, or NaN if none succeeded), ``"success_rate"``,
        ``"avg_wall_collisions"``, ``"avg_penalty_entries"`` over the
        final ``window`` episodes.
    """
    tail = metrics.tail(window)
    successful = tail[tail["success"]]
    return {
        "avg_steps_on_success": float(successful["steps"].mean()) if len(successful) else float("nan"),
        "success_rate": float(tail["success"].mean()),
        "avg_wall_collisions": float(tail["wall_collisions"].mean()),
        "avg_penalty_entries": float(tail["penalty_entries"].mean()),
    }


def load_run(algorithm: str, run_id: str, results_root: Path = Path("results")) -> dict:
    """Load a completed run's config, metrics, and saved model arrays.

    Parameters
    ----------
    algorithm : str
        One of ``"value_iteration"``, ``"q_learning"``, ``"sarsa_lambda"``.
    run_id : str
        The run's identifier (matches the directory name under
        ``results/raw_data/<algorithm>/`` and ``results/models/<algorithm>/``).
    results_root : pathlib.Path, default=Path("results")
        Root of the results directory tree.

    Returns
    -------
    dict
        ``"config"`` (dict), ``"metrics"`` (DataFrame or None for VI,
        which logs iteration deltas instead of episodes), and
        ``"models"`` (dict of str -> ndarray, whatever ``.npy`` files
        exist in the run's model directory).

    Notes
    -----
    Reads exclusively from ``results/raw_data`` and ``results/models``
    -- this is the reproducibility contract from ``CODING_STYLE.md``
    1.8: any comparison or chart must be regenerable from persisted
    logs without rerunning training.
    """
    raw_dir = results_root / "raw_data" / algorithm / run_id
    model_dir = results_root / "models" / algorithm / run_id

    with open(raw_dir / "config.json") as f:
        config = json.load(f)

    metrics = None
    metrics_path = raw_dir / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)

    models = {}
    if model_dir.exists():
        for npy_path in model_dir.glob("*.npy"):
            models[npy_path.stem] = np.load(npy_path)

    return {"config": config, "metrics": metrics, "models": models}


def visitation_count_from_events(events_log_path, state_space_shape) -> np.ndarray:
    """Build a per-state visitation count array from a saved events.log file.

    Parameters
    ----------
    events_log_path : str or pathlib.Path
        Path to a ``events.log`` file (one JSON object per line, as
        written by ``agents/q_learning.py`` / ``agents/sarsa_lambda.py``).
    state_space_shape : tuple of int
        ``(X, Y, 2, E+1)`` shape to build the visitation array in.

    Returns
    -------
    ndarray of shape state_space_shape, dtype=int
        Number of times each ``(x, y, k, energy)`` state was the
        *pre-transition* state of a logged step. Used both as a
        required "Visitation map" visual output (see
        ``gui/renderer.py``) and to build a reachable-states mask for
        a more meaningful policy-agreement metric (see Notes on
        :func:`compare_algorithms`'s raw full-space agreement being
        dominated by never-visited (position, energy) combinations,
        since energy decreases monotonically along any trajectory and
        always starts at max_energy on reset).
    """
    counts = np.zeros(state_space_shape, dtype=np.int64)
    with open(events_log_path) as f:
        for line in f:
            row = json.loads(line)
            x, y, k, e = row["state"]
            counts[x, y, k, e] += 1
    return counts


def reachable_states_mask(visitation_counts: np.ndarray, min_visits: int = 1) -> np.ndarray:
    """Boolean mask of states visited at least ``min_visits`` times during training.

    Parameters
    ----------
    visitation_counts : ndarray
        Output of :func:`visitation_count_from_events`.
    min_visits : int, default=1
        Minimum visitation count to consider a state "reachable" /
        meaningfully trained.

    Returns
    -------
    ndarray, same shape as ``visitation_counts``, dtype=bool
        ``True`` where the state was visited at least ``min_visits``
        times.

    Notes
    -----
    Restricting policy-agreement computations to this mask (rather
    than the full nominal state space) is important for this
    environment specifically: because ``energy`` decreases
    monotonically along a trajectory and always resets to
    ``max_energy``, only a thin, position-correlated slice of the
    full ``(x, y, k, energy)`` space is ever actually visited under
    any reasonable policy. Reporting raw full-space agreement without
    this restriction conflates "the learned policy disagrees with VI"
    with "this (position, energy) combination was simply never
    trained," which is a materially different and less interesting
    finding -- and one worth surfacing explicitly in the report as
    part of the required "likely cause of the discrepancy" analysis.
    """
    return visitation_counts >= min_visits


def compare_algorithms(
    vi_run: dict,
    q_learning_run: dict,
    sarsa_run: dict,
    wall_mask: np.ndarray,
    q_learning_events_log_path=None,
    sarsa_events_log_path=None,
) -> dict:
    """Produce the full cross-algorithm comparison table required by the spec.

    Parameters
    ----------
    vi_run : dict
        Output of :func:`load_run` for the Value Iteration run
        (must include ``models["policy"]``).
    q_learning_run : dict
        Output of :func:`load_run` for a Q-Learning run (must include
        ``models["Q"]`` and non-None ``metrics``).
    sarsa_run : dict
        Output of :func:`load_run` for a SARSA(lambda) run (same
        shape requirements as ``q_learning_run``).
    wall_mask : ndarray of shape (X, Y), dtype=bool
        Map's wall mask, for excluding illegal states from agreement
        percentages.
    q_learning_events_log_path : str or pathlib.Path, optional
        Path to the Q-Learning run's ``events.log``. If provided, an
        additional agreement metric restricted to actually-visited
        (reachable) states is computed alongside the raw full-space
        one (see :func:`reachable_states_mask` for why this matters).
    sarsa_events_log_path : str or pathlib.Path, optional
        Same as ``q_learning_events_log_path`` but for the SARSA(lambda) run.

    Returns
    -------
    dict
        Nested comparison dict with keys ``"q_learning_vs_vi"`` and
        ``"sarsa_vs_vi"``, each containing ``"agreement_fraction"``
        (raw, full state space), ``"agreement_mask"`` (for later
        color-coded rendering), ``"reachable_states_analysis"`` (None
        if no events log path was given, else a dict with
        ``"n_reachable_states"`` and
        ``"agreement_fraction_reachable_only"``), ``"runtime_seconds"``,
        ``"n_samples"``, ``"memory_bytes"``, and ``"path_quality"``,
        plus top-level ``"vi_runtime_seconds"``/``"vi_n_iterations"``
        for reference.
    """
    vi_policy = vi_run["models"]["policy"]
    state_shape = vi_policy.shape

    def _one_comparison(run, events_log_path=None):
        Q = run["models"]["Q"]
        policy = extract_policy_from_Q(Q)
        mask, valid_mask, agreement_fraction = policy_agreement_mask(vi_policy, policy, wall_mask)

        reachable_result = None
        if events_log_path is not None and Path(events_log_path).exists():
            visits = visitation_count_from_events(events_log_path, state_shape)
            reach_mask = reachable_states_mask(visits, min_visits=1)
            restricted_valid = valid_mask & reach_mask
            restricted_agree = mask & reach_mask
            reachable_fraction = float(
                np.sum(restricted_agree) / max(1, np.sum(restricted_valid))
            )
            reachable_result = {
                "n_reachable_states": int(np.sum(reach_mask & valid_mask)),
                "agreement_fraction_reachable_only": reachable_fraction,
            }

        return {
            "agreement_fraction": agreement_fraction,
            "agreement_mask": mask,
            "reachable_states_analysis": reachable_result,
            "runtime_seconds": run["config"].get("runtime_seconds"),
            "n_samples": sample_count_from_metrics(run["metrics"]),
            "memory_bytes": memory_usage_bytes(Q),
            "path_quality": path_quality(run["metrics"]),
        }

    return {
        "vi_runtime_seconds": vi_run["config"].get("runtime_seconds"),
        "vi_n_iterations": vi_run["config"].get("n_iterations"),
        "q_learning_vs_vi": _one_comparison(
            q_learning_run, q_learning_events_log_path
        ),
        "sarsa_vs_vi": _one_comparison(sarsa_run, sarsa_events_log_path),
    }


def find_disagreement_examples(
    vi_policy: np.ndarray,
    other_policy: np.ndarray,
    wall_mask: np.ndarray,
    k: int,
    energy: int,
    n_examples: int = 3,
) -> list:
    """Find example states where a learned policy disagrees with the VI reference.

    Parameters
    ----------
    vi_policy : ndarray of shape (X, Y, 2, E)
        Value Iteration's greedy policy.
    other_policy : ndarray of shape (X, Y, 2, E)
        Learned (Q-Learning/SARSA) greedy policy.
    wall_mask : ndarray of shape (X, Y), dtype=bool
        Wall mask, to exclude illegal states.
    k : int
        Key-state slice to search within (0 or 1).
    energy : int
        Energy-state slice to search within.
    n_examples : int, default=3
        Number of example disagreement states to return.

    Returns
    -------
    list of dict
        Each dict has ``"x"``, ``"y"``, ``"vi_action"``, ``"other_action"``
        for one disagreeing state, for the report's requirement to
        "analyze three example states with respect to local structure."
    """
    disagree = (vi_policy[:, :, k, energy] != other_policy[:, :, k, energy]) & (~wall_mask)
    xs, ys = np.where(disagree)
    examples = []
    for i in range(min(n_examples, len(xs))):
        x, y = int(xs[i]), int(ys[i])
        examples.append(
            {
                "x": x,
                "y": y,
                "vi_action": int(vi_policy[x, y, k, energy]),
                "other_action": int(other_policy[x, y, k, energy]),
            }
        )
    return examples
