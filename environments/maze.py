"""Dynamic maze environment: state, canonical transition model, rewards.

Implements Algorithm-agnostic parts of ``final_project.md``:
"Problem Definition and Maze Environment", "MDP Modeling and Reward
Function", and the mandatory extra feature (limited energy).

State representation
---------------------
``s = (x, y, k, energy)``

- ``x, y``      : agent position, ``0 <= x, y < maze_size``
- ``k``         : 0 or 1, whether the key has been collected
- ``energy``    : remaining energy, ``0 <= energy <= max_energy``

This is Markov: given ``(x, y, k, energy)`` and an action, the next
state's distribution does not depend on how the agent got there. The
energy counter is folded into the state (rather than tracked as
external history) precisely so this holds -- see
``CODING_STYLE.md`` 1.2.

Canonical transition function
------------------------------
:func:`transition_probabilities` is the *single* place stochastic
dynamics (0.8 intended / 0.1 each perpendicular deviation) are
defined. Both :meth:`MazeEnv.step` (via sampling) and
``agents/value_iteration.py`` (via exact probabilities) call into
this function, per ``CODING_STYLE.md`` 1.1 ("This must live in one
canonical transition function used by *both* the environment step and
Value Iteration's model -- never re-derived twice.").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, NamedTuple, Optional

import numpy as np

from environments.generator import (
    DOOR,
    GOAL,
    KEY,
    NORMAL,
    PENALTY,
    START,
    WALL,
    MapSpec,
    num_traversable_cells,
)

# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)
ACTION_DELTAS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}
# The two actions perpendicular to a given action (for the 0.1/0.1
# stochastic deviation).
PERPENDICULAR = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}

P_INTENDED = 0.8
P_DEVIATION = 0.1  # each of the two perpendicular directions


class State(NamedTuple):
    """Immutable, hashable Markov state ``(x, y, k, energy)``.

    Notes
    -----
    A ``NamedTuple`` rather than a mutable object, so it can be used
    directly as a dict/array key and passed through pure functions
    safely (``CODING_STYLE.md`` 1.2).
    """

    x: int
    y: int
    k: int
    energy: int


class Event(str, Enum):
    """Loggable per-step event tags (``CODING_STYLE.md`` 1.4 / spec
    "Minimum Loggable Events", extended with ``ENERGY_DEPLETED`` for
    the mandatory limited-energy feature).
    """

    NORMAL_MOVE = "normal_move"
    WALL_COLLISION = "wall_collision"
    PENALTY_CELL = "penalty_cell"
    KEY_OBTAINED = "key_obtained"
    DOOR_BLOCKED = "door_blocked"
    DOOR_PASSED = "door_passed"
    GOAL_REACHED = "goal_reached"
    STEP_CAP_REACHED = "step_cap_reached"
    ENERGY_DEPLETED = "energy_depleted"


@dataclass(frozen=True)
class EnvConfig:
    """Immutable environment configuration.

    Parameters
    ----------
    max_energy : int
        Starting/maximum energy value; also the upper bound for the
        ``energy`` field of :class:`State`.
    energy_cost_per_step : int, default=1
        Energy consumed per action, regardless of outcome.
    penalty_extra_energy_cost : int, default=0
        Additional energy consumed specifically when entering a
        penalty cell, on top of ``energy_cost_per_step``. Defaults to
        0 to keep the energy mechanic isolated from the reward
        function's own penalty handling (see project consult notes).
    step_cap : int
        Maximum steps per episode before ``STEP_CAP_REACHED`` fires.
    reward_version : {"sparse", "shaped"}
        Which reward function variant to use.
    """

    max_energy: int
    energy_cost_per_step: int = 1
    penalty_extra_energy_cost: int = 0
    step_cap: int = 300
    reward_version: str = "sparse"


def default_step_cap(map_spec: MapSpec) -> int:
    """Compute the spec-suggested episode step cap for a map.

    Parameters
    ----------
    map_spec : MapSpec
        The map to compute the cap for.

    Returns
    -------
    int
        ``3 * num_traversable_cells(map_spec.grid)``, per the spec's
        "Episode Cap and Termination Condition" section. Callers
        should still record the *actually used* value in
        ``config.json`` rather than relying on this being re-derived
        implicitly (``CODING_STYLE.md`` 1.4).
    """
    return 3 * num_traversable_cells(map_spec.grid)


def default_max_energy(map_spec: MapSpec) -> int:
    """Compute a reasonable default max energy for a map.

    Parameters
    ----------
    map_spec : MapSpec
        The map to compute the default for.

    Returns
    -------
    int
        ``2 * num_traversable_cells(map_spec.grid)``, generous enough
        that a near-optimal policy comfortably succeeds while a
        wasteful policy (excess wall bumps, backtracking) can run out
        of energy. This is a *default*; it is expected to be varied
        and recorded in ``config.json`` like any other hyperparameter.
    """
    return 2 * num_traversable_cells(map_spec.grid)


def transition_probabilities(
    map_spec: MapSpec,
    state: State,
    action: int,
) -> list:
    """Canonical stochastic transition model: outcomes and their probabilities.

    This is the single source of truth for maze dynamics. It is used
    directly, exactly, by both :meth:`MazeEnv.step` (which samples one
    outcome from this list) and ``agents/value_iteration.py`` (which
    sums over every outcome for the Bellman backup) -- per
    ``CODING_STYLE.md`` 1.1, this function must never be re-derived
    twice.

    Parameters
    ----------
    map_spec : MapSpec
        Static map (grid + special cell coordinates) the transition
        is computed against.
    state : State
        Current state ``(x, y, k, energy)``.
    action : int
        One of :data:`ACTIONS`.

    Returns
    -------
    list of (float, State, Event)
        Each tuple is ``(probability, next_state, event)`` for one
        possible outcome of taking ``action`` in ``state``. Probabilities
        sum to 1.0. ``next_state.energy`` is *not* decremented here --
        energy bookkeeping is applied uniformly by the caller (see
        :func:`apply_energy_cost`) so this function stays a pure
        spatial/logical transition model independent of the energy
        config (``energy_cost_per_step`` etc. can vary without
        touching this function).

    Notes
    -----
    Implements the "Transition dynamics" subsection of
    ``final_project.md``: intended action w.p. 0.8, each perpendicular
    deviation w.p. 0.1. A move into a wall (from any of the three
    candidate directions) leaves the agent in place and is tagged
    ``WALL_COLLISION``. A move into the closed door while ``k == 0``
    is also blocked in place and tagged ``DOOR_BLOCKED``.
    """
    size = map_spec.maze_size
    x, y, k, energy = state

    candidates = [
        (P_INTENDED, action),
        (P_DEVIATION, PERPENDICULAR[action][0]),
        (P_DEVIATION, PERPENDICULAR[action][1]),
    ]

    outcomes = []
    for prob, act in candidates:
        dx, dy = ACTION_DELTAS[act]
        nx, ny = x + dx, y + dy

        if not (0 <= nx < size and 0 <= ny < size):
            # Out of bounds counts as a wall collision.
            outcomes.append((prob, State(x, y, k, energy), Event.WALL_COLLISION))
            continue

        cell = map_spec.grid[nx, ny]

        if cell == WALL:
            outcomes.append((prob, State(x, y, k, energy), Event.WALL_COLLISION))
            continue

        if cell == DOOR and k == 0:
            outcomes.append((prob, State(x, y, k, energy), Event.DOOR_BLOCKED))
            continue

        new_k = k
        event = Event.NORMAL_MOVE

        if cell == KEY and k == 0:
            new_k = 1
            event = Event.KEY_OBTAINED
        elif cell == DOOR and k == 1:
            event = Event.DOOR_PASSED
        elif cell == PENALTY:
            event = Event.PENALTY_CELL
        elif cell == GOAL:
            event = Event.GOAL_REACHED

        outcomes.append((prob, State(nx, ny, new_k, energy), event))

    return outcomes


def apply_energy_cost(
    next_state: State,
    event: Event,
    config: EnvConfig,
) -> State:
    """Deduct energy for a step outcome, clipped to ``[0, max_energy]``.

    Parameters
    ----------
    next_state : State
        Spatial/key next state from :func:`transition_probabilities`,
        with its original (pre-cost) ``energy`` field.
    event : Event
        The event tag for this transition (penalty cells may cost
        extra energy per :attr:`EnvConfig.penalty_extra_energy_cost`).
    config : EnvConfig
        Environment configuration.

    Returns
    -------
    State
        ``next_state`` with ``energy`` reduced by the step cost
        (and, if applicable, the penalty surcharge), clipped to
        ``[0, config.max_energy]``.
    """
    cost = config.energy_cost_per_step
    if event == Event.PENALTY_CELL:
        cost += config.penalty_extra_energy_cost
    new_energy = int(np.clip(next_state.energy - cost, 0, config.max_energy))
    return State(next_state.x, next_state.y, next_state.k, new_energy)


# --------------------------------------------------------------------------
# Reward functions
# --------------------------------------------------------------------------

# Shared reward constants, exposed so config.json snapshots can record
# the exact values and the report can quote/justify them.
SPARSE_REWARDS = {
    "step_cost": -1.0,
    "wall_collision": -5.0,
    "door_blocked": -5.0,
    "penalty_cell": -10.0,
    "key_obtained": 50.0,
    "goal_reached": 200.0,
    "energy_depleted": -100.0,
    "step_cap_reached": -50.0,
}

SHAPED_REWARDS = {
    **SPARSE_REWARDS,
    # Shaping terms layered on top of the sparse base reward.
    "distance_shaping_scale": 2.0,  # reward per unit reduction in Manhattan
                                     # distance to the current sub-goal
                                     # (key, then door/goal)
    "safe_passage_bonus": 1.0,  # small bonus for a normal_move adjacent to
                                # a penalty cell without entering it
    "wasted_move_penalty": -2.0,  # extra penalty for a wall_collision, to
                                  # discourage repeated bumping beyond the
                                  # base step cost
}


def _manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _subgoal_position(map_spec: MapSpec, k: int) -> tuple:
    """Return the agent's current navigational sub-goal position.

    Parameters
    ----------
    map_spec : MapSpec
        Map providing ``key_pos`` and ``goal`` coordinates.
    k : int
        Whether the key has been collected.

    Returns
    -------
    tuple of int
        ``key_pos`` if the key has not yet been collected, else
        ``goal``. Used only by the shaped reward's distance term.
    """
    return map_spec.key_pos if k == 0 else map_spec.goal


def sparse_reward_fn(
    map_spec: MapSpec,
    state: State,
    action: int,
    next_state: State,
    event: Event,
) -> float:
    """Sparse reward: small per-step cost, large reward on key/goal.

    Parameters
    ----------
    map_spec : MapSpec
        Static map (unused directly here, present for interface
        parity with :func:`shaped_reward_fn` and future extensions).
    state : State
        State before the transition.
    action : int
        Action taken.
    next_state : State
        State after the transition.
    event : Event
        Event tag for this transition, as returned by
        :func:`transition_probabilities`.

    Returns
    -------
    float
        Reward for this transition. Per "Version 1 -- Sparse Reward":
        most of the reward comes from ``key_obtained``/``goal_reached``,
        with a small constant cost per move.

    Notes
    -----
    Implements the reward interface
    ``reward_fn(state, action, next_state, event) -> float`` required
    by ``CODING_STYLE.md`` 1.4, so sparse/shaped variants are
    swappable via config without editing algorithm code.
    """
    r = SPARSE_REWARDS["step_cost"]
    if event == Event.WALL_COLLISION:
        r += SPARSE_REWARDS["wall_collision"]
    elif event == Event.DOOR_BLOCKED:
        r += SPARSE_REWARDS["door_blocked"]
    elif event == Event.PENALTY_CELL:
        r += SPARSE_REWARDS["penalty_cell"]
    elif event == Event.KEY_OBTAINED:
        r += SPARSE_REWARDS["key_obtained"]
    elif event == Event.GOAL_REACHED:
        r += SPARSE_REWARDS["goal_reached"]
    elif event == Event.ENERGY_DEPLETED:
        r += SPARSE_REWARDS["energy_depleted"]
    elif event == Event.STEP_CAP_REACHED:
        r += SPARSE_REWARDS["step_cap_reached"]
    return r


def shaped_reward_fn(
    map_spec: MapSpec,
    state: State,
    action: int,
    next_state: State,
    event: Event,
) -> float:
    """Shaped reward: sparse base plus intermediate shaping signals.

    Parameters
    ----------
    map_spec : MapSpec
        Static map, used to compute Manhattan distance to the current
        sub-goal (key, then goal) for the distance-shaping term.
    state : State
        State before the transition.
    action : int
        Action taken.
    next_state : State
        State after the transition.
    event : Event
        Event tag for this transition.

    Returns
    -------
    float
        Reward for this transition. Per "Version 2 -- Reward
        Shaping": adds a term for moving closer to the current
        sub-goal, a small bonus for safely passing near a penalty
        cell without entering it, and an extra penalty for wasted
        (wall-collision) moves, on top of the sparse base reward.

    Notes
    -----
    The distance-shaping term is potential-based in spirit (reward
    proportional to *reduction* in distance to sub-goal, not absolute
    distance), which limits (but per the spec's own required analysis,
    does not necessarily eliminate) unwanted loop-exploiting behavior;
    this must be checked empirically and reported per final_project.md
    ("show whether the shaped reward ... introduced unwanted
    behaviors such as looping movement").
    """
    r = sparse_reward_fn(map_spec, state, action, next_state, event)

    if event == Event.WALL_COLLISION:
        r += SHAPED_REWARDS["wasted_move_penalty"]
        return r

    sub_before = _subgoal_position(map_spec, state.k)
    sub_after = _subgoal_position(map_spec, next_state.k)
    dist_before = _manhattan((state.x, state.y), sub_before)
    dist_after = _manhattan((next_state.x, next_state.y), sub_after)
    if event != Event.KEY_OBTAINED:
        # Reward reduction in distance to the *current* sub-goal;
        # skip on the key-obtaining step itself since the sub-goal
        # target switches (key -> goal) at that exact transition.
        r += SHAPED_REWARDS["distance_shaping_scale"] * (dist_before - dist_after)

    if event == Event.NORMAL_MOVE:
        size = map_spec.maze_size
        nx, ny = next_state.x, next_state.y
        adjacent_to_penalty = False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ax, ay = nx + dx, ny + dy
            if 0 <= ax < size and 0 <= ay < size and map_spec.grid[ax, ay] == PENALTY:
                adjacent_to_penalty = True
                break
        if adjacent_to_penalty:
            r += SHAPED_REWARDS["safe_passage_bonus"]

    return r


REWARD_FNS: dict = {
    "sparse": sparse_reward_fn,
    "shaped": shaped_reward_fn,
}


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

class StepResult(NamedTuple):
    """Result of one :meth:`MazeEnv.step` call.

    Attributes
    ----------
    next_state : State
        Resulting state.
    reward : float
        Reward for this transition, from the configured reward
        function.
    done : bool
        Whether the episode has terminated.
    event : Event
        Event tag for this transition (may be overridden to
        ``ENERGY_DEPLETED`` or ``STEP_CAP_REACHED`` if those
        termination conditions fire on this step).
    """

    next_state: State
    reward: float
    done: bool
    event: Event


class MazeEnv:
    """Stochastic dynamic maze environment (gym-like step API).

    Parameters
    ----------
    map_spec : MapSpec
        Validated, serialized map to run on (loaded via
        ``environments.generator.load_map``).
    config : EnvConfig
        Environment configuration (energy, step cap, reward version).
    rng : numpy.random.Generator
        Seeded generator used for sampling stochastic transitions.
        Never uses global numpy random state (``CODING_STYLE.md`` 2.1).

    Notes
    -----
    This class is a thin, side-effect-isolated wrapper around the pure
    functions :func:`transition_probabilities`, :func:`apply_energy_cost`,
    and the reward functions in :data:`REWARD_FNS`. All actual dynamics
    logic lives in those pure functions so it can be unit-tested and
    reused identically by ``agents/value_iteration.py``
    (``CODING_STYLE.md`` 2.2).
    """

    def __init__(self, map_spec: MapSpec, config: EnvConfig, rng: np.random.Generator):
        self.map_spec = map_spec
        self.config = config
        self.rng = rng
        self.reward_fn: Callable = REWARD_FNS[config.reward_version]
        self._state: Optional[State] = None
        self._step_count = 0

    def reset(self) -> State:
        """Reset the environment to the map's start state, full energy.

        Returns
        -------
        State
            Initial state ``(start_x, start_y, 0, max_energy)``.
        """
        x, y = self.map_spec.start
        self._state = State(x, y, 0, self.config.max_energy)
        self._step_count = 0
        return self._state

    def step(self, action: int) -> StepResult:
        """Advance the environment by one action.

        Parameters
        ----------
        action : int
            One of :data:`ACTIONS`.

        Returns
        -------
        StepResult
            ``(next_state, reward, done, event)``. ``event`` reflects
            whichever of the loggable event tags applies, with
            ``ENERGY_DEPLETED`` / ``STEP_CAP_REACHED`` taking priority
            as *termination* reasons when they coincide with the step
            that triggers them.

        Raises
        ------
        RuntimeError
            If called before :meth:`reset`.
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step().")

        outcomes = transition_probabilities(self.map_spec, self._state, action)
        probs = [o[0] for o in outcomes]
        idx = self.rng.choice(len(outcomes), p=probs)
        _, raw_next_state, event = outcomes[idx]

        next_state = apply_energy_cost(raw_next_state, event, self.config)
        self._step_count += 1

        done = False
        if event == Event.GOAL_REACHED:
            done = True
        elif next_state.energy <= 0:
            event = Event.ENERGY_DEPLETED
            done = True
        elif self._step_count >= self.config.step_cap:
            event = Event.STEP_CAP_REACHED
            done = True

        reward = self.reward_fn(self.map_spec, self._state, action, next_state, event)

        self._state = next_state
        return StepResult(next_state, reward, done, event)

    @property
    def state(self) -> State:
        """Current state (read-only view; use :meth:`step`/:meth:`reset` to change it)."""
        if self._state is None:
            raise RuntimeError("Must call reset() before accessing state.")
        return self._state

    def state_space_shape(self) -> tuple:
        """Shape of the full ``(x, y, k, energy)`` state space.

        Returns
        -------
        tuple of int
            ``(maze_size, maze_size, 2, max_energy + 1)``, i.e. the
            shape any per-state numpy array (V-table, visitation
            counts, etc.) for this environment should use.
        """
        size = self.map_spec.maze_size
        return (size, size, 2, self.config.max_energy + 1)

    def all_states(self):
        """Iterate every state in the full state space (including unreachable ones).

        Yields
        ------
        State
            Every ``(x, y, k, energy)`` combination. Callers needing
            only *legal* (non-wall) states should filter using
            ``self.map_spec.grid[x, y] != WALL``.
        """
        size = self.map_spec.maze_size
        for x in range(size):
            for y in range(size):
                for k in (0, 1):
                    for energy in range(self.config.max_energy + 1):
                        yield State(x, y, k, energy)

    def is_terminal(self, state: State) -> bool:
        """Whether ``state`` is a terminal (goal) state for planning purposes.

        Parameters
        ----------
        state : State
            State to check.

        Returns
        -------
        bool
            ``True`` iff the agent's position is the goal cell. Used
            by Value Iteration to fix ``V(terminal) = 0`` / exclude
            terminal states from the max-over-actions Bellman update,
            since no action is taken after reaching the goal.

        Notes
        -----
        Energy-depletion and step-cap are *episode-ending* conditions
        for sampled rollouts (Q-Learning/SARSA episodes), but are not
        "terminal states" in the MDP sense used by Value Iiteration's
        state space -- VI reasons over the full stationary state
        space and does not have a notion of elapsed steps. Energy
        reaching 0 is instead handled structurally: actions from an
        ``energy == 0`` state simply cannot occur (energy is clipped
        to >= 0 and reset per-episode), so those states are
        effectively absorbing/unreachable-from in a single VI sweep
        and do not need special-casing beyond this goal check.
        """
        return (state.x, state.y) == self.map_spec.goal
