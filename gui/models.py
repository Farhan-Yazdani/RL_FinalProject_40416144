"""Shared data model for the GUI, mirroring the real backend exactly.

Cell-type ints, action indices, and event tags below are copied to
match ``environments.generator`` / ``environments.maze`` verbatim (not
re-derived), so grids and events coming from the real backend can be
rendered directly without a translation layer. State is the real
4-tuple ``(x, y, k, energy)`` from ``environments.maze.State`` --
this project's mandatory extra feature is **limited energy**.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional, Tuple

import numpy as np


class CellType(IntEnum):
    """Static maze cell types -- values copied from ``environments.generator``.

    0=normal, 1=wall, 2=penalty, 3=start, 4=key, 5=door, 6=goal.
    """

    NORMAL = 0
    WALL = 1
    PENALTY = 2
    START = 3
    KEY = 4
    DOOR = 5
    GOAL = 6


class Action(IntEnum):
    """Action indices -- copied from ``environments.maze`` (``ACTIONS``)."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class EventTag(str, Enum):
    """Event tags -- copied from ``environments.maze.Event`` verbatim."""

    NORMAL_MOVE = "normal_move"
    WALL_COLLISION = "wall_collision"
    PENALTY_CELL = "penalty_cell"
    KEY_OBTAINED = "key_obtained"
    DOOR_BLOCKED = "door_blocked"
    DOOR_PASSED = "door_passed"
    GOAL_REACHED = "goal_reached"
    STEP_CAP_REACHED = "step_cap_reached"
    ENERGY_DEPLETED = "energy_depleted"


class RunMode(str, Enum):
    """Train (live learning) vs. eval (frozen greedy rollout)."""

    TRAIN = "train"
    EVAL = "eval"


class AlgorithmName(str, Enum):
    """Registry keys for the three required algorithms."""

    VALUE_ITERATION = "value_iteration"
    Q_LEARNING = "q_learning"
    SARSA_LAMBDA = "sarsa_lambda"


class MapName(str, Enum):
    """Map names -- must match the filenames under ``environments/maps/``
    exactly as written by ``generate_maps.py`` / ``environments.generator``.
    """

    SOURCE = "source"
    TRANSFER_SIMILAR = "transfer_similar"
    TRANSFER_DIFFERENT = "transfer_different"


@dataclass(frozen=True)
class LiveStats:
    """Snapshot of the live info panel required by the GUI spec."""

    episode: int = 0
    step: int = 0
    reward: float = 0.0
    epsilon: Optional[float] = None
    key_status: bool = False
    energy: Optional[int] = None
    recent_success_rate: float = 0.0
    mode: RunMode = RunMode.TRAIN
    last_event: Optional[EventTag] = None


@dataclass(frozen=True)
class MazeRenderState:
    """Everything the renderer needs to draw one frame.

    Parameters
    ----------
    grid : ndarray of shape (N, N)
        ``CellType`` values for every cell (``map_spec.grid``).
    agent_state : tuple
        The real ``(x, y, k, energy)`` state tuple.
    max_energy : int
        This run's configured max energy, for the energy readout.
    door_open : bool
        Whether the key has been collected (``k == 1``).
    """

    grid: np.ndarray
    agent_state: Tuple[Any, ...]
    max_energy: int
    door_open: bool = False


def success_rate(outcomes: np.ndarray, window: int = 50) -> float:
    """Compute the recent success rate over a trailing window.

    Parameters
    ----------
    outcomes : ndarray of shape (num_episodes,), dtype bool
        ``True`` where an episode reached the goal.
    window : int, default=50
        Number of most recent episodes to average over.

    Returns
    -------
    float
        Fraction of successful episodes in the trailing window, or
        ``0.0`` if ``outcomes`` is empty.
    """
    if outcomes.size == 0:
        return 0.0
    return float(np.mean(outcomes[-window:]))
