"""Q-Learning agent. Implements Algorithm 2 (off-policy, epsilon-greedy
with decay) from final_project.md.
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

from environments import generator as gen
from environments.generator import derive_seed_and_size, load_map
from environments.maze import (
    ACTIONS,
    EnvConfig,
    Event,
    MazeEnv,
    REWARD_FNS,
    State,
    default_max_energy,
    default_step_cap,
)


def epsilon_greedy(Q, state: State, epsilon: float, rng: np.random.Generator) -> int:
    """Sample an action under an epsilon-greedy behavior policy.

    Parameters
    ----------
    Q : ndarray of shape (X, Y, 2, E, A)
        Current action-value table.
    state : State
        Current state ``(x, y, k, energy)``.
    epsilon : float
        Exploration probability in [0, 1].
    rng : numpy.random.Generator
        Seeded generator for the random choice.

    Returns
    -------
    int
        Chosen action index, one of :data:`environments.maze.ACTIONS`.

    Notes
    -----
    Pure function: same ``(Q, state, epsilon)`` plus the same RNG draw
    always yields the same action (``CODING_STYLE.md`` 2.2).
    """
    if rng.random() < epsilon:
        return int(rng.integers(0, len(ACTIONS)))
    q_values = Q[state.x, state.y, state.k, state.energy]
    return int(np.argmax(q_values))


def td_update(Q, s: State, a: int, r: float, s_next: State, alpha: float, gamma: float, done: bool):
    """Compute the Q-Learning (off-policy) TD update for one transition.

    Parameters
    ----------
    Q : ndarray of shape (X, Y, 2, E, A)
        Current action-value table (not mutated).
    s : State
        State before the transition.
    a : int
        Action taken.
    r : float
        Reward received.
    s_next : State
        State after the transition.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    done : bool
        Whether ``s_next`` is a terminal state for this episode; if
        so, the bootstrap term ``max_a' Q(s_next, a')`` is omitted
        (treated as 0), matching standard episodic TD-control.

    Returns
    -------
    Q_new : ndarray of shape (X, Y, 2, E, A)
        A *copy* of ``Q`` with only ``Q[s, a]`` updated -- pure
        function, no in-place mutation of the input array
        (``CODING_STYLE.md`` 2.2). Callers in the hot training loop
        may instead mutate in place for performance; see
        :func:`train_q_learning` which does so explicitly and
        documents that choice.

    Notes
    -----
    Implements
    ``Q(s,a) <- Q(s,a) + alpha [r + gamma * max_a' Q(s',a') - Q(s,a)]``
    from the project's Q-Learning section. This is the *off-policy*
    target: the bootstrap uses ``max`` over next actions regardless of
    which action the behavior policy would actually take next (unlike
    SARSA's on-policy bootstrap).
    """
    Q_new = Q.copy()
    best_next = 0.0 if done else np.max(Q[s_next.x, s_next.y, s_next.k, s_next.energy])
    td_target = r + gamma * best_next
    idx = (s.x, s.y, s.k, s.energy, a)
    Q_new[idx] = Q[idx] + alpha * (td_target - Q[idx])
    return Q_new


# --------------------------------------------------------------------------
# Epsilon decay schedules
# --------------------------------------------------------------------------

def linear_decay(episode: int, total_episodes: int, eps_start: float, eps_end: float) -> float:
    """Linear epsilon decay schedule.

    Parameters
    ----------
    episode : int
        Current episode index (0-based).
    total_episodes : int
        Total number of training episodes.
    eps_start : float
        Epsilon at episode 0.
    eps_end : float
        Epsilon at the final episode (and beyond).

    Returns
    -------
    float
        ``eps_start + (eps_end - eps_start) * min(1, episode / total_episodes)``.
    """
    frac = min(1.0, episode / max(1, total_episodes))
    return eps_start + (eps_end - eps_start) * frac


def exponential_decay(episode: int, total_episodes: int, eps_start: float, eps_end: float) -> float:
    """Exponential epsilon decay schedule.

    Parameters
    ----------
    episode : int
        Current episode index (0-based).
    total_episodes : int
        Total number of training episodes (defines the decay rate so
        epsilon reaches approximately ``eps_end`` by the final episode).
    eps_start : float
        Epsilon at episode 0.
    eps_end : float
        Asymptotic epsilon floor.

    Returns
    -------
    float
        ``eps_end + (eps_start - eps_end) * exp(-decay_rate * episode)``,
        where ``decay_rate`` is chosen so the schedule reaches within
        1% of ``eps_end`` by ``total_episodes``.
    """
    decay_rate = -np.log(0.01) / max(1, total_episodes)
    return eps_end + (eps_start - eps_end) * np.exp(-decay_rate * episode)


EPSILON_SCHEDULES = {
    "linear": linear_decay,
    "exponential": exponential_decay,
}


@dataclass(frozen=True)
class QLearningConfig:
    """Resolved Q-Learning run configuration."""

    student_id: str
    map_name: str
    alpha: float
    gamma: float
    eps_start: float
    eps_end: float
    eps_schedule: str
    n_episodes: int
    max_energy: int
    step_cap: int
    reward_version: str
    seed: int


def train_q_learning(
    map_spec,
    env_config: EnvConfig,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_schedule_name: str,
    n_episodes: int,
    rng: np.random.Generator,
):
    """Run Q-Learning training for ``n_episodes`` episodes.

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
    eps_start : float
        Initial exploration rate.
    eps_end : float
        Final/floor exploration rate.
    eps_schedule_name : {"linear", "exponential"}
        Which decay schedule to use (see :data:`EPSILON_SCHEDULES`).
    n_episodes : int
        Number of training episodes.
    rng : numpy.random.Generator
        Seeded generator driving the environment and epsilon-greedy
        action selection.

    Returns
    -------
    Q : ndarray of shape (X, Y, 2, E, A)
        Learned action-value table.
    episode_metrics : pandas.DataFrame
        One row per episode with columns: ``episode``, ``reward``,
        ``steps``, ``success`` (bool), ``wall_collisions``,
        ``penalty_entries``, ``epsilon``.
    events_log : list of dict
        Structured per-step event log across all episodes (``episode``,
        ``step``, ``state``, ``action``, ``event``, ``reward``), per
        the CLI/logging contract (``CODING_STYLE.md`` 1.4/2.4).

    Notes
    -----
    The per-step TD update is applied via direct in-place mutation of
    a single ``Q`` array threaded through the (inherently sequential)
    episode loop, rather than calling the pure :func:`td_update` and
    reassigning -- copying the full ``(X,Y,2,E,A)`` table on every
    single environment step would be prohibitively slow for
    realistic ``max_energy``. This mutation is isolated behind the one
    clearly-named update line below, consistent with
    ``CODING_STYLE.md`` 2.2's carve-out for documented in-place
    accumulators. :func:`td_update` remains available, tested, and
    used as the reference implementation (e.g. by ``tests/``) for
    what this inline update must match.
    """
    env = MazeEnv(map_spec, env_config, rng)
    shape = (map_spec.maze_size, map_spec.maze_size, 2, env_config.max_energy + 1, len(ACTIONS))
    Q = np.zeros(shape, dtype=np.float64)
    schedule_fn = EPSILON_SCHEDULES[eps_schedule_name]

    episode_rows = []
    events_log = []

    for ep in range(n_episodes):
        epsilon = schedule_fn(ep, n_episodes, eps_start, eps_end)
        s = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        wall_collisions = 0
        penalty_entries = 0
        success = False

        while True:
            a = epsilon_greedy(Q, s, epsilon, rng)
            res = env.step(a)
            s_next, r, done, event = res

            best_next = 0.0 if done else np.max(Q[s_next.x, s_next.y, s_next.k, s_next.energy])
            idx = (s.x, s.y, s.k, s.energy, a)
            td_error = r + gamma * best_next - Q[idx]
            Q[idx] = Q[idx] + alpha * td_error

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
            if done:
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

    return Q, pd.DataFrame(episode_rows), events_log


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
    """CLI entry point for a single Q-Learning training run."""
    parser = argparse.ArgumentParser(description="Train Q-Learning on the maze.")
    parser.add_argument("--student-id", type=str, default="40")
    parser.add_argument("--map-name", type=str, default="source")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-schedule", type=str, default="linear", choices=list(EPSILON_SCHEDULES))
    parser.add_argument("--n-episodes", type=int, default=2000)
    parser.add_argument("--max-energy", type=int, default=None)
    parser.add_argument("--reward-version", type=str, default="sparse", choices=["sparse", "shaped"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args(argv)

    if not (0.0 <= args.eps_start <= 1.0) or not (0.0 <= args.eps_end <= 1.0):
        print("error: --eps-start and --eps-end must be in [0, 1]", file=sys.stderr)
        return 1
    if args.eps_end > args.eps_start:
        print("error: --eps-end must be <= --eps-start", file=sys.stderr)
        return 1

    map_spec = load_map("environments/maps", args.map_name)
    base_seed, maze_size = derive_seed_and_size(args.student_id)
    assert maze_size == map_spec.maze_size

    max_energy = args.max_energy if args.max_energy is not None else default_max_energy(map_spec)
    step_cap = default_step_cap(map_spec)
    env_config = EnvConfig(max_energy=max_energy, step_cap=step_cap, reward_version=args.reward_version)

    rng = np.random.default_rng(args.seed)
    start_time = time.perf_counter()
    Q, episode_metrics, events_log = train_q_learning(
        map_spec, env_config, args.alpha, args.gamma,
        args.eps_start, args.eps_end, args.eps_schedule,
        args.n_episodes, rng,
    )
    runtime_seconds = time.perf_counter() - start_time

    run_id = args.run_id or (
        f"qlearning_{args.map_name}_{args.eps_schedule}_seed{args.seed}"
    )
    dirs = _resolve_run_dir("q_learning", run_id)

    config_snapshot = {
        "student_id": args.student_id,
        "base_seed": base_seed,
        "maze_size": maze_size,
        "map_name": args.map_name,
        "alpha": args.alpha,
        "gamma": args.gamma,
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

    np.save(dirs["models"] / "Q.npy", Q)

    success_rate_last_100 = episode_metrics["success"].tail(100).mean()
    print(
        f"Q-Learning trained for {args.n_episodes} episodes "
        f"({runtime_seconds:.2f}s). Success rate (last 100 eps): "
        f"{success_rate_last_100:.1%}. Results in {dirs['raw_data']} / {dirs['models']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
