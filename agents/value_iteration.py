"""Value Iteration agent. Implements Algorithm 1 (model-based) from
final_project.md.

Uses the exact transition model from ``environments.maze.transition_probabilities``
directly -- the same function the environment samples from in
``MazeEnv.step`` -- so the model VI plans over is identical to the one
generating episodes for Q-Learning/SARSA(lambda) (``CODING_STYLE.md`` 1.1).

No ready-made RL/solver library is used; the Bellman backup and greedy
policy extraction are implemented from scratch below.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from environments import generator as gen
from environments.generator import WALL, derive_seed_and_size, load_map
from environments.maze import (
    ACTIONS,
    EnvConfig,
    State,
    default_max_energy,
    default_step_cap,
    transition_probabilities,
)


@dataclass(frozen=True)
class VIConfig:
    """Resolved Value Iteration run configuration.

    Parameters
    ----------
    student_id : str
        Student ID the map/seed were derived from (recorded for
        traceability, not re-derived here).
    map_name : str
        Name of the map file under ``environments/maps/`` to load.
    gamma : float
        Discount factor in [0, 1].
    theta : float
        Convergence threshold: sweep stops when max|V_new - V_old| < theta.
    max_iterations : int
        Hard cap on sweeps, in case of non-convergence.
    max_energy : int
        Max energy value defining the state space's energy dimension.
    reward_version : {"sparse", "shaped"}
        Which reward function to plan under.
    """

    student_id: str
    map_name: str
    gamma: float
    theta: float
    max_iterations: int
    max_energy: int
    reward_version: str


def build_state_index(maze_size: int, max_energy: int):
    """Build the bijection between ``State`` tuples and flat array indices.

    Parameters
    ----------
    maze_size : int
        Grid side length.
    max_energy : int
        Maximum energy value.

    Returns
    -------
    shape : tuple of int
        ``(maze_size, maze_size, 2, max_energy + 1)`` -- the ndarray
        shape used to index directly by ``(x, y, k, energy)`` without
        a separate flattening step (documented bijection per
        ``CODING_STYLE.md`` 2.1: state tuple <-> array index is simply
        direct multi-dimensional indexing here).

    Notes
    -----
    V-tables and Q-tables in this project are ``ndarray`` objects
    shaped exactly like this, indexed as ``V[x, y, k, energy]`` /
    ``Q[x, y, k, energy, action]`` -- never a ``dict``.
    """
    return (maze_size, maze_size, 2, max_energy + 1)


def build_sparse_transition_and_reward(
    map_spec,
    env_config: EnvConfig,
    reward_fn,
):
    """Materialize the transition/reward model as fixed-width sparse arrays.

    Parameters
    ----------
    map_spec : environments.generator.MapSpec
        Validated map to build the model over.
    env_config : EnvConfig
        Environment configuration (defines the energy dimension and
        energy-cost bookkeeping).
    reward_fn : callable
        ``reward_fn(map_spec, state, action, next_state, event) -> float``,
        one of ``environments.maze.REWARD_FNS``.

    Returns
    -------
    next_idx : ndarray of shape (X, Y, 2, E, A, 3, 4), dtype=int64
        ``next_idx[x,y,k,e,a,i,:]`` is the ``(x2,y2,k2,e2)`` index
        tuple of the ``i``-th (of up to 3) possible successor states
        for outcome ``i`` of taking action ``a`` in state
        ``(x,y,k,e)``. Padded with the state itself (self-loop, prob
        0) if fewer than 3 distinct outcomes apply.
    next_prob : ndarray of shape (X, Y, 2, E, A, 3), dtype=float64
        Probability of each of the (up to 3) outcomes; unused padding
        slots have probability 0.
    next_reward : ndarray of shape (X, Y, 2, E, A, 3), dtype=float64
        Immediate reward for each of the (up to 3) outcomes.
    wall_mask : ndarray of shape (X, Y), dtype=bool
        ``True`` where the cell is a wall.

    Notes
    -----
    A dense ``(S, A, S)`` transition tensor is infeasible here: with
    ``maze_size=15`` and a realistic ``max_energy`` in the hundreds,
    ``|S| = X*Y*2*(E+1)`` reaches ~70,000+ states, making a dense
    ``S x A x S`` tensor hundreds of gigabytes. This function instead
    exploits that :func:`environments.maze.transition_probabilities`
    returns at most 3 successor outcomes per ``(s, a)`` pair (the
    intended action plus two perpendicular deviations), and stores
    only those -- a fixed-width sparse format that vectorizes cleanly
    with ``np.take_along_axis``/advanced indexing in
    :func:`bellman_update` while staying memory-tractable. This is
    still built by calling the single canonical
    :func:`environments.maze.transition_probabilities` for every
    ``(state, action)`` pair, so the model is guaranteed consistent
    with ``MazeEnv.step``'s sampling (``CODING_STYLE.md`` 1.1).
    """
    size = map_spec.maze_size
    max_energy = env_config.max_energy
    n_actions = len(ACTIONS)
    n_outcomes = 3

    next_idx = np.zeros((size, size, 2, max_energy + 1, n_actions, n_outcomes, 4), dtype=np.int64)
    next_prob = np.zeros((size, size, 2, max_energy + 1, n_actions, n_outcomes), dtype=np.float64)
    next_reward = np.zeros((size, size, 2, max_energy + 1, n_actions, n_outcomes), dtype=np.float64)
    wall_mask = map_spec.grid == WALL

    from environments.maze import apply_energy_cost

    for x in range(size):
        for y in range(size):
            if wall_mask[x, y]:
                continue
            for k in (0, 1):
                for e in range(max_energy + 1):
                    s = State(x, y, k, e)
                    for a in ACTIONS:
                        if e == 0:
                            # Absorbing: self-loop, prob 0, reward 0.
                            for i in range(n_outcomes):
                                next_idx[x, y, k, e, a, i] = (x, y, k, e)
                            continue
                        outcomes = transition_probabilities(map_spec, s, a)
                        for i, (prob, raw_next, event) in enumerate(outcomes):
                            next_s = apply_energy_cost(raw_next, event, env_config)
                            next_idx[x, y, k, e, a, i] = (
                                next_s.x, next_s.y, next_s.k, next_s.energy
                            )
                            next_prob[x, y, k, e, a, i] = prob
                            next_reward[x, y, k, e, a, i] = reward_fn(
                                map_spec, s, a, next_s, event
                            )

    return next_idx, next_prob, next_reward, wall_mask


def build_transition_and_reward_tensors(
    map_spec,
    env_config: EnvConfig,
    reward_fn,
):
    """Materialize the full transition/reward model as dense ndarrays.

    Parameters
    ----------
    map_spec : environments.generator.MapSpec
        Validated map to build the model over.
    env_config : EnvConfig
        Environment configuration (defines the energy dimension and
        energy-cost bookkeeping).
    reward_fn : callable
        ``reward_fn(map_spec, state, action, next_state, event) -> float``,
        one of ``environments.maze.REWARD_FNS``.

    Returns
    -------
    P : ndarray of shape (X, Y, 2, E, A, X, Y, 2, E)
        ``P[x,y,k,e,a,x2,y2,k2,e2]`` = probability of transitioning to
        ``(x2,y2,k2,e2)`` from ``(x,y,k,e)`` under action ``a``.
    R : ndarray of shape (X, Y, 2, E, A)
        Expected immediate reward, integrated over the transition
        distribution.
    wall_mask : ndarray of shape (X, Y), dtype=bool
        ``True`` where the cell is a wall.

    Notes
    -----
    **Only safe to call for small state spaces** (e.g. small
    ``max_energy`` in tests/demos): memory is ``O((X*Y*2*E)^2 * A)``.
    For real runs with realistic ``max_energy``, use
    :func:`build_sparse_transition_and_reward` and
    :func:`bellman_update` instead -- this dense version is kept only
    for unit tests that want to cross-check the sparse path against a
    ground-truth dense tensor on a tiny toy energy range.
    """
    size = map_spec.maze_size
    max_energy = env_config.max_energy
    shape = build_state_index(size, max_energy)
    n_actions = len(ACTIONS)

    P = np.zeros(shape + (n_actions,) + shape, dtype=np.float64)
    R = np.zeros(shape + (n_actions,), dtype=np.float64)
    wall_mask = map_spec.grid == WALL

    from environments.maze import apply_energy_cost

    for x in range(size):
        for y in range(size):
            if wall_mask[x, y]:
                continue
            for k in (0, 1):
                for e in range(max_energy + 1):
                    s = State(x, y, k, e)
                    for a in ACTIONS:
                        if e == 0:
                            continue
                        outcomes = transition_probabilities(map_spec, s, a)
                        expected_r = 0.0
                        for prob, raw_next, event in outcomes:
                            next_s = apply_energy_cost(raw_next, event, env_config)
                            expected_r += prob * reward_fn(map_spec, s, a, next_s, event)
                            P[x, y, k, e, a, next_s.x, next_s.y, next_s.k, next_s.energy] += prob
                        R[x, y, k, e, a] = expected_r

    return P, R, wall_mask


def bellman_update(V, next_idx, next_prob, next_reward, gamma):
    """Perform one synchronous Bellman backup over all states.

    Parameters
    ----------
    V : ndarray of shape (X, Y, 2, E)
        Current value function estimate.
    next_idx : ndarray of shape (X, Y, 2, E, A, 3, 4)
        Successor-state index tuples, from
        :func:`build_sparse_transition_and_reward`.
    next_prob : ndarray of shape (X, Y, 2, E, A, 3)
        Successor-state probabilities.
    next_reward : ndarray of shape (X, Y, 2, E, A, 3)
        Immediate reward for each successor outcome.
    gamma : float
        Discount factor in [0, 1].

    Returns
    -------
    V_new : ndarray of shape (X, Y, 2, E)
        Updated value function after one Bellman sweep.
    delta : float
        Maximum absolute change ``max(abs(V_new - V))``, used for the
        convergence check.

    Notes
    -----
    Implements the update
    ``V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V_k(s')]``
    as specified in the project's Value Iteration section. Vectorized
    via fancy indexing over the fixed-width (<=3 outcomes) sparse
    representation: ``V[next_idx[...,0], next_idx[...,1], next_idx[...,2],
    next_idx[...,3]]`` gathers all successor values in one call, then
    the outcome axis is summed (weighted by probability) and the
    action axis is maxed -- no explicit Python loop over states
    (``CODING_STYLE.md`` 2.1).
    """
    v_next = V[next_idx[..., 0], next_idx[..., 1], next_idx[..., 2], next_idx[..., 3]]
    # v_next, next_prob, next_reward: shape (X, Y, 2, E, A, 3)
    q_outcomes = next_prob * (next_reward + gamma * v_next)
    Q = np.sum(q_outcomes, axis=-1)  # shape (X, Y, 2, E, A)
    V_new = np.max(Q, axis=-1)
    delta = float(np.max(np.abs(V_new - V)))
    return V_new, delta


def extract_greedy_policy(V, next_idx, next_prob, next_reward, gamma):
    """Extract the greedy policy w.r.t. a converged (or current) value function.

    Parameters
    ----------
    V : ndarray of shape (X, Y, 2, E)
        Value function to act greedily with respect to.
    next_idx : ndarray of shape (X, Y, 2, E, A, 3, 4)
        Successor-state index tuples.
    next_prob : ndarray of shape (X, Y, 2, E, A, 3)
        Successor-state probabilities.
    next_reward : ndarray of shape (X, Y, 2, E, A, 3)
        Immediate reward for each successor outcome.
    gamma : float
        Discount factor.

    Returns
    -------
    policy : ndarray of shape (X, Y, 2, E), dtype=int
        ``policy[x,y,k,e] = argmax_a Q(s,a)``.
    Q : ndarray of shape (X, Y, 2, E, A)
        The action-value tensor the policy was extracted from (also
        useful for later policy-agreement comparisons against
        Q-Learning/SARSA).
    """
    v_next = V[next_idx[..., 0], next_idx[..., 1], next_idx[..., 2], next_idx[..., 3]]
    q_outcomes = next_prob * (next_reward + gamma * v_next)
    Q = np.sum(q_outcomes, axis=-1)
    policy = np.argmax(Q, axis=-1).astype(np.int64)
    return policy, Q


def run_value_iteration(
    map_spec,
    env_config: EnvConfig,
    reward_fn,
    gamma: float,
    theta: float,
    max_iterations: int,
):
    """Run Value Iteration to (near-)convergence.

    Parameters
    ----------
    map_spec : environments.generator.MapSpec
        Validated map.
    env_config : EnvConfig
        Environment configuration (energy dimension, etc.).
    reward_fn : callable
        Reward function, one of ``environments.maze.REWARD_FNS``.
    gamma : float
        Discount factor in [0, 1].
    theta : float
        Convergence threshold on max value change between sweeps.
    max_iterations : int
        Hard cap on sweeps.

    Returns
    -------
    V : ndarray of shape (X, Y, 2, E)
        Converged (or best-effort) value function.
    policy : ndarray of shape (X, Y, 2, E), dtype=int
        Greedy policy w.r.t. ``V``.
    Q : ndarray of shape (X, Y, 2, E, A)
        Action-value tensor policy was extracted from.
    n_iterations : int
        Number of Bellman sweeps actually performed.
    runtime_seconds : float
        Wall-clock time for the sweep loop (excludes tensor-building
        time, which is reported separately by the caller).
    deltas : list of float
        Per-sweep convergence delta, for convergence-curve plotting.
    """
    shape = build_state_index(map_spec.maze_size, env_config.max_energy)
    V = np.zeros(shape, dtype=np.float64)

    next_idx, next_prob, next_reward, wall_mask = build_sparse_transition_and_reward(
        map_spec, env_config, reward_fn
    )

    deltas = []
    start = time.perf_counter()
    n_iterations = 0
    for i in range(max_iterations):
        V, delta = bellman_update(V, next_idx, next_prob, next_reward, gamma)
        deltas.append(delta)
        n_iterations = i + 1
        if delta < theta:
            break
    runtime_seconds = time.perf_counter() - start

    policy, Q = extract_greedy_policy(V, next_idx, next_prob, next_reward, gamma)
    return V, policy, Q, n_iterations, runtime_seconds, deltas


def _resolve_run_dir(algorithm: str, run_id: str) -> dict:
    """Build and create the standard results sub-directories for a run.

    Parameters
    ----------
    algorithm : str
        Algorithm name (e.g. ``"value_iteration"``).
    run_id : str
        Deterministic, informative run identifier.

    Returns
    -------
    dict
        Keys ``"raw_data"``, ``"models"``, ``"figures"`` mapping to
        created ``Path`` objects under ``results/``.
    """
    root = Path("results")
    dirs = {
        "raw_data": root / "raw_data" / algorithm / run_id,
        "models": root / "models" / algorithm / run_id,
        "figures": root / "figures" / algorithm / run_id,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def main(argv=None):
    """CLI entry point for a single Value Iteration run.

    Parameters
    ----------
    argv : list of str, optional
        Argument list (defaults to ``sys.argv[1:]`` if ``None``).

    Returns
    -------
    int
        Process exit code (0 on success, non-zero on invalid args).
    """
    parser = argparse.ArgumentParser(description="Run Value Iteration on the maze.")
    parser.add_argument("--student-id", type=str, default="40")
    parser.add_argument("--map-name", type=str, default="source")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--theta", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--max-energy", type=int, default=None)
    parser.add_argument("--reward-version", type=str, default="sparse", choices=["sparse", "shaped"])
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args(argv)

    if not (0.0 <= args.gamma <= 1.0):
        print(f"error: --gamma must be in [0, 1], got {args.gamma}", file=sys.stderr)
        return 1

    from environments.maze import REWARD_FNS

    map_spec = load_map("environments/maps", args.map_name)
    base_seed, maze_size = derive_seed_and_size(args.student_id)
    assert maze_size == map_spec.maze_size, (
        f"Derived maze_size={maze_size} does not match loaded map "
        f"size={map_spec.maze_size}; check --student-id / --map-name."
    )

    max_energy = args.max_energy if args.max_energy is not None else default_max_energy(map_spec)
    step_cap = default_step_cap(map_spec)
    env_config = EnvConfig(
        max_energy=max_energy,
        step_cap=step_cap,
        reward_version=args.reward_version,
    )
    reward_fn = REWARD_FNS[args.reward_version]

    run_id = args.run_id or f"vi_{args.map_name}_gamma{args.gamma}_seed{base_seed}"
    dirs = _resolve_run_dir("value_iteration", run_id)

    V, policy, Q, n_iter, runtime_s, deltas = run_value_iteration(
        map_spec, env_config, reward_fn, args.gamma, args.theta, args.max_iterations
    )

    config_snapshot = {
        "student_id": args.student_id,
        "base_seed": base_seed,
        "maze_size": maze_size,
        "map_name": args.map_name,
        "gamma": args.gamma,
        "theta": args.theta,
        "max_iterations": args.max_iterations,
        "max_energy": max_energy,
        "step_cap": step_cap,
        "reward_version": args.reward_version,
        "n_iterations": n_iter,
        "runtime_seconds": runtime_s,
        "run_id": run_id,
    }
    with open(dirs["raw_data"] / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    import pandas as pd
    pd.DataFrame({"iteration": np.arange(1, len(deltas) + 1), "delta": deltas}).to_csv(
        dirs["raw_data"] / "metrics.csv", index=False
    )

    np.save(dirs["models"] / "V.npy", V)
    np.save(dirs["models"] / "policy.npy", policy)
    np.save(dirs["models"] / "Q.npy", Q)

    print(
        f"Value Iteration converged in {n_iter} iterations "
        f"({runtime_s:.4f}s), final delta={deltas[-1]:.6g}. "
        f"Results written to {dirs['raw_data']} / {dirs['models']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
