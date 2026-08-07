"""SARSA(lambda) agent. Implements Algorithm 3 (on-policy, eligibility
traces) from final_project.md.

Trace type: **replacing** traces are used (see :func:`update_eligibility_trace`
docstring for the justification), applied uniformly for all
lambda in {0, 0.3, 0.7, 0.9} as required by the spec.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from environments.generator import derive_seed_and_size, load_map
from environments.maze import (
    ACTIONS,
    EnvConfig,
    Event,
    MazeEnv,
    State,
    default_max_energy,
    default_step_cap,
)
from agents.q_learning import epsilon_greedy, EPSILON_SCHEDULES
from agents.policy_extraction import save_derived_policy
 


VALID_LAMBDAS = (0.0, 0.3, 0.7, 0.9)


TRACE_MIN_THRESHOLD = 1e-4


def update_eligibility_trace(E: dict, s: State, a: int, gamma: float, lam: float) -> dict:
    """Decay all active trace entries and bump the current (s, a) trace (replacing traces).

    Parameters
    ----------
    E : dict of (tuple of int) -> float
        Current *sparse* eligibility trace, mapping ``(x, y, k, energy, a)``
        index tuples with non-negligible trace value to that value.
        Mutated in place (see Notes).
    s : State
        State visited on this step.
    a : int
        Action taken on this step.
    gamma : float
        Discount factor.
    lam : float
        Trace decay parameter (the "lambda" of SARSA(lambda)).

    Returns
    -------
    E : dict of (tuple of int) -> float
        The same dict, updated in place, returned for convenience.
        Entries that decay below :data:`TRACE_MIN_THRESHOLD` are
        dropped so the dict stays proportional to the number of
        *recently and repeatedly* visited (state, action) pairs, not
        the full state-action space.

    Notes
    -----
    Implements
    ``E_t(s,a) = gamma * lambda * E_{t-1}(s,a) + 1{s=s_t, a=a_t}``
    from the project's SARSA(lambda) section, using **replacing**
    traces: the visited ``(s, a)`` entry is *set* to 1 after decay
    (rather than *incremented* by 1, which is the accumulating-trace
    variant). Replacing traces are used here because the maze allows
    the agent to revisit the same state under a stochastic policy
    within a single episode (e.g. bouncing near a wall or an
    ineffective detour); accumulating traces would let such a state's
    eligibility grow past 1 and disproportionately amplify the credit
    assigned to a state the agent visited repeatedly by mistake rather
    than by productive routing, which is undesirable given the
    wall-collision dynamics explicitly in this environment.

    A **sparse dict** is used instead of a dense ``(X,Y,2,E,A)`` numpy
    array for tractability: with a realistic ``max_energy`` in the
    hundreds, the full state-action space is on the order of 10^5-10^6
    entries, and decaying + touching every entry on *every single
    environment step* (needed for the dense ``Q += alpha*delta*E``
    formulation) makes training infeasibly slow. Eligibility traces
    are near-zero for the vast majority of states at any given step
    (only recently-visited states have appreciable trace), so a dict
    keyed by visited-state-action index, dropped once its value decays
    below :data:`TRACE_MIN_THRESHOLD`, is mathematically equivalent to
    the dense formulation up to that truncation and is what makes
    training tractable. This is still the single, clearly named place
    trace mutation happens (in-place accumulator, per
    ``CODING_STYLE.md`` 2.2's documented carve-out for eligibility
    traces specifically).
    """
    decay = gamma * lam
    if decay == 0.0:
        E.clear()
    else:
        to_delete = []
        for key in E:
            E[key] *= decay
            if E[key] < TRACE_MIN_THRESHOLD:
                to_delete.append(key)
        for key in to_delete:
            del E[key]
    E[(s.x, s.y, s.k, s.energy, a)] = 1.0
    return E


@dataclass(frozen=True)
class SarsaLambdaConfig:
    """Resolved SARSA(lambda) run configuration."""

    student_id: str
    map_name: str
    alpha: float
    gamma: float
    lam: float
    eps_start: float
    eps_end: float
    eps_schedule: str
    n_episodes: int
    max_energy: int
    step_cap: int
    reward_version: str
    seed: int


def train_sarsa_lambda(
    map_spec,
    env_config: EnvConfig,
    alpha: float,
    gamma: float,
    lam: float,
    eps_start: float,
    eps_end: float,
    eps_schedule_name: str,
    n_episodes: int,
    rng: np.random.Generator,
    trace_episode_log_index: int = 0,
):
    """Run SARSA(lambda) training for ``n_episodes`` episodes.

    Parameters
    ----------
    map_spec : environments.generator.MapSpec
        Validated map to train on.
    env_config : EnvConfig
        Environment configuration.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    lam : float
        Eligibility trace decay parameter; must be in
        :data:`VALID_LAMBDAS`.
    eps_start : float
        Initial exploration rate.
    eps_end : float
        Final/floor exploration rate.
    eps_schedule_name : {"linear", "exponential"}
        Which decay schedule to use.
    n_episodes : int
        Number of training episodes.
    rng : numpy.random.Generator
        Seeded generator driving the environment and epsilon-greedy
        action selection.
    trace_episode_log_index : int, default=0
        Episode index to record a full step-by-step
        delta/eligibility-trace trace for (per the spec's requirement
        to log and interpret delta/E over a short episode). Only the
        *first* ``done`` episode at or after this index is logged in
        full detail, to keep the returned structure bounded.

    Returns
    -------
    Q : ndarray of shape (X, Y, 2, E, A)
        Learned action-value table.
    episode_metrics : pandas.DataFrame
        One row per episode: ``episode``, ``reward``, ``steps``,
        ``success``, ``wall_collisions``, ``penalty_entries``, ``epsilon``.
    events_log : list of dict
        Structured per-step event log across all episodes.
    trace_detail : list of dict
        Step-by-step ``(state, action, reward, delta, trace_norm)``
        for the logged short episode (see ``trace_episode_log_index``),
        satisfying the spec's requirement to show delta/E evolution
        for at least one short episode.

    Notes
    -----
    Implements the on-policy SARSA(lambda) update:
    ``delta_t = r_{t+1} + gamma*Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)``,
    ``Q <- Q + alpha * delta_t * E``, with ``E`` updated via
    :func:`update_eligibility_trace`. Unlike Q-Learning's ``max`` over
    next actions, the bootstrap here uses ``Q(s_{t+1}, a_{t+1})`` for
    the *actual* next action ``a_{t+1}`` sampled from the same
    epsilon-greedy behavior policy generating the trajectory -- this
    is what makes SARSA on-policy.
    """
    if lam not in VALID_LAMBDAS:
        raise ValueError(f"lam must be one of {VALID_LAMBDAS}, got {lam}")

    env = MazeEnv(map_spec, env_config, rng)
    shape = (map_spec.maze_size, map_spec.maze_size, 2, env_config.max_energy + 1, len(ACTIONS))
    Q = np.zeros(shape, dtype=np.float64)
    schedule_fn = EPSILON_SCHEDULES[eps_schedule_name]

    episode_rows = []
    events_log = []
    trace_detail = []
    trace_logged = False

    for ep in range(n_episodes):
        epsilon = schedule_fn(ep, n_episodes, eps_start, eps_end)
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon, rng)
        E: dict = {}

        ep_reward = 0.0
        ep_steps = 0
        wall_collisions = 0
        penalty_entries = 0
        success = False

        log_this_episode = (ep >= trace_episode_log_index) and not trace_logged

        while True:
            res = env.step(a)
            s_next, r, done, event = res

            a_next = None if done else epsilon_greedy(Q, s_next, epsilon, rng)

            q_sa = Q[s.x, s.y, s.k, s.energy, a]
            q_next = 0.0 if done else Q[s_next.x, s_next.y, s_next.k, s_next.energy, a_next]
            delta = r + gamma * q_next - q_sa

            E = update_eligibility_trace(E, s, a, gamma, lam)
            for (ex, ey, ek, ee, ea), trace_val in E.items():
                Q[ex, ey, ek, ee, ea] += alpha * delta * trace_val

            if log_this_episode:
                trace_detail.append(
                    {
                        "step": ep_steps,
                        "state": s,
                        "action": a,
                        "reward": r,
                        "delta": float(delta),
                        "trace_norm": float(sum(E.values())),
                        "trace_max": float(max(E.values())) if E else 0.0,
                    }
                )

            events_log.append(
                {
                    "episode": ep,
                    "step": ep_steps,
                    "state": s,
                    "action": a,
                    "next_state": s_next,
                    "event": event.value,
                    "reward": r,
                }
            )

            ep_reward += r
            ep_steps += 1
            if event == Event.WALL_COLLISION:
                wall_collisions += 1
            elif event == Event.PENALTY_CELL:
                penalty_entries += 1
            elif event == Event.GOAL_REACHED:
                success = True

            s = s_next
            a = a_next
            if done:
                if log_this_episode:
                    trace_logged = True
                break

        episode_rows.append(
            {
                "episode": ep,
                "reward": ep_reward,
                "steps": ep_steps,
                "success": success,
                "wall_collisions": wall_collisions,
                "penalty_entries": penalty_entries,
                "epsilon": epsilon,
            }
        )

    return Q, pd.DataFrame(episode_rows), events_log, trace_detail


def _resolve_run_dir(algorithm: str, run_id: str) -> dict:
    """Build and create the standard results sub-directories for a run."""
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
    """CLI entry point for a single SARSA(lambda) training run."""
    parser = argparse.ArgumentParser(description="Train SARSA(lambda) on the maze.")
    parser.add_argument("--student-id", type=str, default="40")
    parser.add_argument("--map-name", type=str, default="source")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lam", type=float, default=0.7, choices=list(VALID_LAMBDAS))
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-schedule", type=str, default="exponential", choices=list(EPSILON_SCHEDULES))
    parser.add_argument("--n-episodes", type=int, default=4000)
    parser.add_argument("--max-energy", type=int, default=None)
    parser.add_argument("--reward-version", type=str, default="shaped", choices=["sparse", "shaped"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args(argv)

    if args.lam not in VALID_LAMBDAS:
        print(f"error: --lam must be one of {VALID_LAMBDAS}, got {args.lam}", file=sys.stderr)
        return 1

    map_spec = load_map("environments/maps", args.map_name)
    base_seed, maze_size = derive_seed_and_size(args.student_id)
    assert maze_size == map_spec.maze_size

    max_energy = args.max_energy if args.max_energy is not None else default_max_energy(map_spec)
    step_cap = default_step_cap(map_spec)
    env_config = EnvConfig(max_energy=max_energy, step_cap=step_cap, reward_version=args.reward_version)

    rng = np.random.default_rng(args.seed)
    start_time = time.perf_counter()
    Q, episode_metrics, events_log, trace_detail = train_sarsa_lambda(
        map_spec, env_config, args.alpha, args.gamma, args.lam,
        args.eps_start, args.eps_end, args.eps_schedule,
        args.n_episodes, rng,
        trace_episode_log_index=max(0, args.n_episodes - 5),
    )
    runtime_seconds = time.perf_counter() - start_time

    run_id = args.run_id or f"sarsa_lambda{args.lam}_{args.map_name}_seed{args.seed}"
    dirs = _resolve_run_dir("sarsa_lambda", run_id)

    config_snapshot = {
        "student_id": args.student_id,
        "base_seed": base_seed,
        "maze_size": maze_size,
        "map_name": args.map_name,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "lambda": args.lam,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "eps_schedule": args.eps_schedule,
        "n_episodes": args.n_episodes,
        "max_energy": max_energy,
        "step_cap": step_cap,
        "reward_version": args.reward_version,
        "seed": args.seed,
        "runtime_seconds": runtime_seconds,
        "run_id": run_id,
        "trace_type": "replacing",
    }
    with open(dirs["raw_data"] / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    episode_metrics.to_csv(dirs["raw_data"] / "metrics.csv", index=False)

    with open(dirs["raw_data"] / "events.log", "w") as f:
        for row in events_log:
            f.write(
                json.dumps(
                    {
                        "episode": row["episode"],
                        "step": row["step"],
                        "state": list(row["state"]),
                        "action": row["action"],
                        "next_state": list(row["next_state"]),
                        "event": row["event"],
                        "reward": row["reward"],
                    }
                )
                + "\n"
            )

    with open(dirs["raw_data"] / "trace_detail.json", "w") as f:
        json.dump(
            [
                {**row, "state": list(row["state"])}
                for row in trace_detail
            ],
            f,
            indent=2,
        )

    np.save(dirs["models"] / "Q.npy", Q)
    events_log_path = dirs["raw_data"] / "events.log"
    policy = save_derived_policy(Q, dirs["models"], events_log_path=events_log_path)
 
    success_rate_last_100 = episode_metrics["success"].tail(100).mean()
    print(
        f"SARSA(lambda={args.lam}) trained for {args.n_episodes} episodes "
        f"({runtime_seconds:.2f}s). Success rate (last 100 eps): "
        f"{success_rate_last_100:.1%}. Results in {dirs['raw_data']} / {dirs['models']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
