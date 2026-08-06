"""Transfer learning for Q-Learning. Implements the "Transfer Learning
Section" of final_project.md: train on a source map, then initialize
target-environment training from the source Q-table under 4 scenarios.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from environments.generator import WALL, derive_seed_and_size, load_map
from environments.maze import (
    ACTIONS,
    EnvConfig,
    default_max_energy,
    default_step_cap,
)
from agents.q_learning import EPSILON_SCHEDULES, epsilon_greedy, train_q_learning


BETA_VALUES = (0.25, 0.50, 0.75)
SCENARIOS = ("scratch", "full", "scaled", "selective")


def _local_neighborhood_signature(grid: np.ndarray, x: int, y: int) -> tuple:
    """Wall/non-wall signature of the 4-neighborhood around ``(x, y)``.

    Parameters
    ----------
    grid : ndarray of shape (size, size)
        Cell-type grid.
    x, y : int
        Position to inspect.

    Returns
    -------
    tuple of bool
        ``(is_wall_here, up_is_wall, down_is_wall, left_is_wall, right_is_wall)``,
        with out-of-bounds neighbors treated as walls. Used by the
        "selective" transfer scenario to decide whether a state's
        local structure is unchanged between source and target maps.
    """
    size = grid.shape[0]

    def _is_wall(px, py):
        if not (0 <= px < size and 0 <= py < size):
            return True
        return bool(grid[px, py] == WALL)

    return (
        _is_wall(x, y),
        _is_wall(x, y - 1),
        _is_wall(x, y + 1),
        _is_wall(x - 1, y),
        _is_wall(x + 1, y),
    )


def unchanged_neighborhood_mask(source_grid: np.ndarray, target_grid: np.ndarray) -> np.ndarray:
    """Mask of ``(x, y)`` positions whose local 4-neighborhood is identical
    between the source and target maps.

    Parameters
    ----------
    source_grid : ndarray of shape (size, size)
        Source map's cell-type grid.
    target_grid : ndarray of shape (size, size)
        Target map's cell-type grid (same shape as ``source_grid``).

    Returns
    -------
    ndarray of shape (size, size), dtype=bool
        ``True`` at positions whose wall/non-wall neighborhood
        signature (see :func:`_local_neighborhood_signature`) matches
        between the two maps. Used by the "selective" transfer
        scenario: only Q-values for states at these unchanged
        positions are transferred, since the local structure they were
        learned under still applies in the target environment.
    """
    size = source_grid.shape[0]
    mask = np.zeros((size, size), dtype=bool)
    for x in range(size):
        for y in range(size):
            sig_src = _local_neighborhood_signature(source_grid, x, y)
            sig_tgt = _local_neighborhood_signature(target_grid, x, y)
            mask[x, y] = sig_src == sig_tgt
    return mask


def initialize_transfer_q_table(
    scenario: str,
    Q_source: np.ndarray,
    target_shape: tuple,
    beta: float = 0.5,
    unchanged_mask: np.ndarray = None,
) -> np.ndarray:
    """Initialize a target-environment Q-table under one of the 4 transfer scenarios.

    Parameters
    ----------
    scenario : {"scratch", "full", "scaled", "selective"}
        Which transfer scenario to apply.
    Q_source : ndarray of shape (X, Y, 2, E_src, A)
        Q-table learned in the source environment.
    target_shape : tuple of int
        ``(X, Y, 2, E_tgt, A)`` shape for the target environment's
        Q-table (``E_tgt`` may differ from ``E_src`` if max_energy
        differs between source/target configs; in that case only the
        overlapping energy range is transferred and the rest is
        zero-initialized).
    beta : float, default=0.5
        Scaling factor for the "scaled" scenario; must be in
        :data:`BETA_VALUES` when that scenario is used.
    unchanged_mask : ndarray of shape (X, Y), dtype=bool, optional
        Required for the "selective" scenario (see
        :func:`unchanged_neighborhood_mask`); ``(x, y)`` positions
        where ``True`` have their full ``Q_source[x, y]`` slice copied,
        all other positions are zero-initialized.

    Returns
    -------
    ndarray of shape target_shape
        Initial Q-table for target-environment training.

    Notes
    -----
    Implements
    ``Q_T^(0)(s,a) = beta * Q_S(s,a), beta in {0.25, 0.50, 0.75}``
    for the "scaled" scenario, and the "scratch"/"full"/"selective"
    scenarios described in the Transfer Learning section, as **one**
    parameterized function rather than four copy-pasted scripts
    (``CODING_STYLE.md`` 1.6). Pure function: does not mutate
    ``Q_source`` (``CODING_STYLE.md`` 2.2).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")

    Q_target = np.zeros(target_shape, dtype=np.float64)
    e_overlap = min(Q_source.shape[3], target_shape[3])

    if scenario == "scratch":
        return Q_target

    if scenario == "full":
        Q_target[:, :, :, :e_overlap, :] = Q_source[:, :, :, :e_overlap, :]
        return Q_target

    if scenario == "scaled":
        if beta not in BETA_VALUES:
            raise ValueError(f"beta must be one of {BETA_VALUES}, got {beta}")
        Q_target[:, :, :, :e_overlap, :] = beta * Q_source[:, :, :, :e_overlap, :]
        return Q_target

    if scenario == "selective":
        if unchanged_mask is None:
            raise ValueError("selective scenario requires unchanged_mask")
        Q_target[:, :, :, :e_overlap, :] = np.where(
            unchanged_mask[:, :, None, None, None],
            Q_source[:, :, :, :e_overlap, :],
            0.0,
        )
        return Q_target

    raise AssertionError("unreachable")


def find_negative_transfer_example(
    Q_transferred: np.ndarray,
    source_grid: np.ndarray,
    target_grid: np.ndarray,
    k: int,
    energy: int = None,
) -> dict:
    """Find a candidate negative-transfer state: a wall in the target
    where the transferred policy's greedy action walks straight into it.

    Parameters
    ----------
    Q_transferred : ndarray of shape (X, Y, 2, E, A)
        Q-table as initialized by :func:`initialize_transfer_q_table`
        (before any continued training in the target environment).
    source_grid : ndarray of shape (size, size)
        Source map's cell-type grid.
    target_grid : ndarray of shape (size, size)
        Target map's cell-type grid.
    k : int
        Key-state slice to search within.
    energy : int, optional
        Single energy-state slice to search within. If ``None``
        (default), searches across a spread of energy levels (since
        transferred Q-values are typically non-zero -- i.e. actually
        trained in the source run -- for only a thin, trajectory-
        correlated slice of the energy dimension; fixing one arbitrary
        energy value risks landing entirely on unvisited/zero entries).

    Returns
    -------
    dict or None
        If found: ``"x"``, ``"y"``, ``"energy"``, ``"greedy_action"``,
        ``"q_values"`` (list, the 4 transferred Q-values at this
        state), and ``"reason"`` (a short human-readable
        explanation). ``None`` if no such example is found across the
        searched slices.

    Notes
    -----
    A concrete, minimal example of negative transfer: a state where
    the source environment's structure justified a particular action
    (e.g. "go right, there's open space"), but the target environment
    placed a new wall exactly there, so blindly trusting the
    transferred Q-values would walk the agent into a wall it had never
    encountered during source training. This satisfies the spec's
    requirement to show "the Q-values of that state, the structural
    change in the environment" for at least one negative-transfer
    example.
    """
    from environments.maze import ACTION_DELTAS

    size = source_grid.shape[0]
    max_e = Q_transferred.shape[3] - 1
    energy_candidates = [energy] if energy is not None else list(range(max_e, -1, -max(1, max_e // 40)))

    for e in energy_candidates:
        for x in range(size):
            for y in range(size):
                if target_grid[x, y] == WALL or source_grid[x, y] == WALL:
                    continue
                q_values = Q_transferred[x, y, k, e]
                if np.all(q_values == 0):
                    continue
                greedy_a = int(np.argmax(q_values))
                dx, dy = ACTION_DELTAS[greedy_a]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < size and 0 <= ny < size):
                    continue
                target_is_wall_now = target_grid[nx, ny] == WALL
                source_was_wall = source_grid[nx, ny] == WALL
                if target_is_wall_now and not source_was_wall:
                    return {
                        "x": x,
                        "y": y,
                        "energy": e,
                        "greedy_action": greedy_a,
                        "q_values": q_values.tolist(),
                        "reason": (
                            f"Transferred policy greedily walks from ({x},{y}) "
                            f"toward ({nx},{ny}), which was open in the source "
                            f"map but is now a wall in the target map."
                        ),
                    }
    return None


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


def train_target_with_transfer(
    target_map_spec,
    env_config: EnvConfig,
    scenario: str,
    Q_source: np.ndarray,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_schedule_name: str,
    n_episodes: int,
    rng: np.random.Generator,
    beta: float = 0.5,
    unchanged_mask: np.ndarray = None,
):
    """Continue Q-Learning training on a target map from a transferred Q-table.

    Parameters
    ----------
    target_map_spec : environments.generator.MapSpec
        Target environment map (similar or different).
    env_config : EnvConfig
        Environment configuration (assumed same max_energy as source
        run, for a clean Q-table shape match; see
        :func:`initialize_transfer_q_table`'s handling if not).
    scenario : {"scratch", "full", "scaled", "selective"}
        Transfer scenario to initialize from.
    Q_source : ndarray
        Source-environment Q-table.
    alpha, gamma, eps_start, eps_end, eps_schedule_name, n_episodes : see
        ``agents.q_learning.train_q_learning``.
    rng : numpy.random.Generator
        Seeded generator.
    beta : float, default=0.5
        Only used for the "scaled" scenario.
    unchanged_mask : ndarray, optional
        Only used for the "selective" scenario.

    Returns
    -------
    Q_final : ndarray
        Q-table after continued training on the target map.
    episode_metrics : pandas.DataFrame
        Per-episode metrics, as in :func:`agents.q_learning.train_q_learning`.
    Q_initial : ndarray
        The transferred (pre-training) Q-table, returned so callers
        can compute negative-transfer examples and before/after diffs.

    Notes
    -----
    This reuses :func:`agents.q_learning.train_q_learning`'s inner
    loop by temporarily monkey-patching its zero-initialization --
    instead, to keep things simple and avoid patching, this function
    re-implements the same training loop but seeds ``Q`` from
    :func:`initialize_transfer_q_table` instead of zeros. The update
    rule and epsilon-greedy behavior are otherwise identical to
    :func:`agents.q_learning.train_q_learning`.
    """
    from environments.maze import Event, MazeEnv

    target_shape = (
        target_map_spec.maze_size,
        target_map_spec.maze_size,
        2,
        env_config.max_energy + 1,
        len(ACTIONS),
    )
    Q = initialize_transfer_q_table(scenario, Q_source, target_shape, beta, unchanged_mask)
    Q_initial = Q.copy()

    env = MazeEnv(target_map_spec, env_config, rng)
    schedule_fn = EPSILON_SCHEDULES[eps_schedule_name]

    episode_rows = []
    for ep in range(n_episodes):
        epsilon = schedule_fn(ep, n_episodes, eps_start, eps_end)
        s = env.reset()
        ep_reward, ep_steps = 0.0, 0
        wall_collisions, penalty_entries = 0, 0
        success = False

        while True:
            a = epsilon_greedy(Q, s, epsilon, rng)
            res = env.step(a)
            s_next, r, done, event = res

            best_next = 0.0 if done else np.max(Q[s_next.x, s_next.y, s_next.k, s_next.energy])
            idx = (s.x, s.y, s.k, s.energy, a)
            Q[idx] = Q[idx] + alpha * (r + gamma * best_next - Q[idx])

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
                "episode": ep, "reward": ep_reward, "steps": ep_steps,
                "success": success, "wall_collisions": wall_collisions,
                "penalty_entries": penalty_entries, "epsilon": epsilon,
            }
        )

    return Q, pd.DataFrame(episode_rows), Q_initial


def main(argv=None):
    """CLI entry point: run all 4 transfer scenarios on both target maps."""
    parser = argparse.ArgumentParser(description="Run transfer learning experiments.")
    parser.add_argument("--student-id", type=str, default="40")
    parser.add_argument("--source-map", type=str, default="source")
    parser.add_argument("--source-run-id", type=str, required=True,
                         help="run_id of a completed q_learning source-environment run")
    parser.add_argument("--n-episodes", type=int, default=1500)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--eps-start", type=float, default=0.5)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--eps-schedule", type=str, default="exponential", choices=list(EPSILON_SCHEDULES))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    source_map = load_map("environments/maps", args.source_map)
    base_seed, maze_size = derive_seed_and_size(args.student_id)

    source_run_dir = Path("results/raw_data/q_learning") / args.source_run_id
    with open(source_run_dir / "config.json") as f:
        source_config = json.load(f)
    Q_source = np.load(Path("results/models/q_learning") / args.source_run_id / "Q.npy")
    max_energy = source_config["max_energy"]
    env_config = EnvConfig(max_energy=max_energy, step_cap=source_config["step_cap"],
                            reward_version=source_config["reward_version"])

    for target_name in ("transfer_similar", "transfer_different"):
        target_map = load_map("environments/maps", target_name)
        unchanged_mask = unchanged_neighborhood_mask(source_map.grid, target_map.grid)

        results_summary = {}
        for scenario in SCENARIOS:
            betas_to_run = BETA_VALUES if scenario == "scaled" else (None,)
            for beta in betas_to_run:
                rng = np.random.default_rng(args.seed)
                Q_final, metrics, Q_initial = train_target_with_transfer(
                    target_map, env_config, scenario, Q_source,
                    args.alpha, args.gamma, args.eps_start, args.eps_end,
                    args.eps_schedule, args.n_episodes, rng,
                    beta=beta if beta is not None else 0.5,
                    unchanged_mask=unchanged_mask,
                )

                run_id = f"transfer_{target_name}_{scenario}"
                if beta is not None:
                    run_id += f"_beta{beta}"
                dirs = _resolve_run_dir("transfer_learning", run_id)

                metrics.to_csv(dirs["raw_data"] / "metrics.csv", index=False)
                np.save(dirs["models"] / "Q_final.npy", Q_final)
                np.save(dirs["models"] / "Q_initial.npy", Q_initial)

                neg_example = find_negative_transfer_example(
                    Q_initial, source_map.grid, target_map.grid, k=0
                )

                config_snapshot = {
                    "student_id": args.student_id, "base_seed": base_seed,
                    "target_map": target_name, "scenario": scenario, "beta": beta,
                    "source_run_id": args.source_run_id,
                    "n_episodes": args.n_episodes, "alpha": args.alpha, "gamma": args.gamma,
                    "eps_start": args.eps_start, "eps_end": args.eps_end,
                    "eps_schedule": args.eps_schedule, "seed": args.seed,
                    "max_energy": max_energy, "run_id": run_id,
                    "negative_transfer_example": neg_example,
                }
                with open(dirs["raw_data"] / "config.json", "w") as f:
                    json.dump(config_snapshot, f, indent=2)

                initial_perf = metrics["success"].head(50).mean()
                final_perf = metrics["success"].tail(100).mean()
                results_summary[run_id] = {
                    "initial_performance": float(initial_perf),
                    "final_performance": float(final_perf),
                }
                print(f"{run_id}: initial_success={initial_perf:.1%}, final_success={final_perf:.1%}")

        summary_path = Path("results/raw_data/transfer_learning") / f"{target_name}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
