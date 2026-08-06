"""Adapter layer between the GUI and the real backend modules.

This is the only file in ``gui/`` that imports ``environments``,
``agents``, ``experiments``, or ``transfer``. It never re-derives
their logic -- every stochastic/learning computation calls the real
function (``MazeEnv.step``, ``epsilon_greedy``,
``update_eligibility_trace``, ``run_value_iteration``,
``extract_policy_from_Q``, ``initialize_transfer_q_table``, ...)
directly. The one exception is the Q-Learning TD update itself:
``agents.q_learning.td_update`` is a pure function that returns a
full copy of the Q-table per call (appropriate for its role as the
tested reference implementation), so :class:`QLearningSession` applies
the identical formula in place instead, for the same reason
``agents.q_learning.train_q_learning``'s own hot loop does and
:class:`SarsaLambdaSession` below already does for its trace update --
copying a few-hundred-thousand-element array on every single GUI
timer tick has no benefit here since nothing else holds a reference
to the previous array.
Its only job is turning "one GUI timer tick" into "one call into the
real backend" and translating the result into
:mod:`gui.models` types the widgets already know how to draw.

Three live "sessions" are provided, one per algorithm, all exposing
the same small ``step(mode)`` / ``render_state()`` / ``live_stats()``
surface so :mod:`gui.controller` can drive any of them identically:

- :class:`ValueIterationSession` -- VI has no per-step trajectory of
  its own (training *is* one global Bellman sweep, run via a
  background thread), so its "live" stepping is a greedy rollout of
  the converged policy through the real ``MazeEnv``.
- :class:`QLearningSession` -- live, incremental, one real
  environment step per tick, using the exact
  ``agents.q_learning.epsilon_greedy`` / ``td_update`` functions the
  batch trainer's own hot loop is built from.
- :class:`SarsaLambdaSession` -- same idea, using
  ``agents.q_learning.epsilon_greedy`` and
  ``agents.sarsa_lambda.update_eligibility_trace``.

None of this replaces ``experiments/run_experiments.py`` for actual
reported results -- it is a real-time *visualization* of the same
algorithms, useful for demoing/debugging, while the authoritative,
reproducible numbers still come from the batch CLI scripts writing to
``results/``.

This module also discovers and loads previously-completed batch runs
(:func:`list_model_run_ids`, :func:`load_model_arrays`,
:func:`apply_loaded_arrays`), so a session can be built from a
*trained* Q/V/policy array instead of always starting from zero.
Discovery and array-loading read only ``results/models/``; the one
exception is :func:`load_run_config`, which reads
``results/raw_data/<algorithm>/<run_id>/config.json`` -- see that
function's docstring for exactly why that's unavoidable with the
current file layout.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

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
from agents.q_learning import EPSILON_SCHEDULES, epsilon_greedy
from agents.sarsa_lambda import TRACE_MIN_THRESHOLD, update_eligibility_trace
from agents.value_iteration import run_value_iteration
from environments.maze import REWARD_FNS
from experiments.analysis import extract_policy_from_Q

from gui.models import EventTag, LiveStats, MazeRenderState, RunMode, success_rate


@dataclass(frozen=True)
class BackendConfig:
    """Everything needed to load a map and build a matching environment.

    Parameters
    ----------
    student_id : str
        Used only to assert the loaded map's ``maze_size`` matches
        ``derive_seed_and_size`` -- the maps themselves are pre-generated
        by ``generate_maps.py`` and loaded by name, never regenerated
        here.
    map_name : str
        One of ``"source"``, ``"transfer_similar"``, ``"transfer_different"``.
    maps_dir : str
        Directory containing the saved ``<map_name>.json`` files.
    reward_version : {"sparse", "shaped"}
        Reward function variant.
    max_energy : int, optional
        Overrides ``default_max_energy(map_spec)`` if given.
    seed : int
        RNG seed for the environment's stochastic transitions.
    """

    student_id: str
    map_name: str
    maps_dir: str = "environments/maps"
    reward_version: str = "sparse"
    max_energy: Optional[int] = None
    seed: int = 0


def build_environment(config: BackendConfig) -> MazeEnv:
    """Load a saved map and construct a real :class:`MazeEnv` for it.

    Parameters
    ----------
    config : BackendConfig
        Resolved configuration (see above).

    Returns
    -------
    MazeEnv
        Ready to ``reset()``/``step()``.

    Raises
    ------
    AssertionError
        If the loaded map's size doesn't match
        ``derive_seed_and_size(config.student_id)`` -- surfaced early
        and loudly rather than producing confusing downstream shape
        errors.
    """
    map_spec = load_map(config.maps_dir, config.map_name)
    _, maze_size = derive_seed_and_size(config.student_id)
    assert maze_size == map_spec.maze_size, (
        f"Derived maze_size={maze_size} from student_id={config.student_id!r} "
        f"does not match loaded map size={map_spec.maze_size} "
        f"for {config.map_name!r}."
    )
    max_energy = config.max_energy if config.max_energy is not None else default_max_energy(map_spec)
    step_cap = default_step_cap(map_spec)
    env_config = EnvConfig(
        max_energy=max_energy, step_cap=step_cap, reward_version=config.reward_version
    )
    rng = np.random.default_rng(config.seed)
    env = MazeEnv(map_spec, env_config, rng)
    env.reset()
    return env


MODELS_ROOT = Path("results/models")
RAW_DATA_ROOT = Path("results/raw_data")


def list_model_run_ids(algorithm: str, models_root: Path = MODELS_ROOT) -> list:
    """List completed run-ids with saved model arrays for ``algorithm``.

    Parameters
    ----------
    algorithm : str
        One of ``"value_iteration"``, ``"q_learning"``, ``"sarsa_lambda"``
        (matches the sub-directory name under ``models_root``).
    models_root : pathlib.Path, default=Path("results/models")
        Root of the saved-model directory tree.

    Returns
    -------
    list of str
        Sorted run-ids (sub-directory names) that contain at least one
        ``.npy`` file. Reads **only** ``results/models/`` -- no other
        directory is touched by this function.
    """
    algo_dir = models_root / algorithm
    if not algo_dir.exists():
        return []
    run_ids = []
    for run_dir in sorted(algo_dir.iterdir()):
        if run_dir.is_dir() and any(run_dir.glob("*.npy")):
            run_ids.append(run_dir.name)
    return run_ids


def load_model_arrays(algorithm: str, run_id: str, models_root: Path = MODELS_ROOT) -> dict:
    """Load every saved ``.npy`` array for one completed run.

    Parameters
    ----------
    algorithm : str
        Sub-directory under ``models_root`` (see :func:`list_model_run_ids`).
    run_id : str
        Run identifier (matches a sub-directory name).
    models_root : pathlib.Path, default=Path("results/models")
        Root of the saved-model directory tree.

    Returns
    -------
    dict of str -> ndarray
        Keyed by filename stem (e.g. ``"V"``, ``"policy"``, ``"Q"``).
        Reads **only** ``results/models/<algorithm>/<run_id>/`` -- no
        other directory is touched by this function.
    """
    run_dir = models_root / algorithm / run_id
    arrays = {}
    for npy_path in run_dir.glob("*.npy"):
        arrays[npy_path.stem] = np.load(npy_path)
    return arrays


def load_run_config(algorithm: str, run_id: str, raw_data_root: Path = RAW_DATA_ROOT) -> Optional[dict]:
    """Load a completed run's resolved config (map, hyperparameters, ...).

    Parameters
    ----------
    algorithm : str
        Sub-directory under ``raw_data_root``.
    run_id : str
        Run identifier.
    raw_data_root : pathlib.Path, default=Path("results/raw_data")
        Root of the raw-data directory tree.

    Returns
    -------
    dict or None
        Parsed ``config.json`` contents, or ``None`` if it doesn't
        exist for this run.

    Notes
    -----
    **This is the one function in this module that reads outside**
    ``results/models/``. The array files under ``results/models/``
    contain only the bare learned numbers (V/Q/policy) -- no record of
    which map, hyperparameters, or reward version produced them (and
    that information is not recoverable from the arrays: array
    *shape* does encode ``maze_size``/``max_energy``, since those set
    the array dimensions, but *which map* -- ``source`` vs.
    ``transfer_similar`` vs. ``transfer_different``, all the same
    size -- and every hyperparameter (alpha, gamma, epsilon schedule,
    lambda, reward version, ...) leaves no trace in the array shape at
    all). That information only exists in
    ``results/raw_data/<algorithm>/<run_id>/config.json``, written by
    the corresponding batch CLI script. Importing a learned model into
    a *matching* environment, and displaying the hyperparameters it
    was actually trained with, both require reading it. If a strict
    "results/models/ only" data source is wanted in a future revision,
    the relevant subset of this config would need to be duplicated
    into ``results/models/`` at save time instead.
    """
    config_path = raw_data_root / algorithm / run_id / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return json.load(f)


def apply_loaded_arrays(session, algorithm: str, arrays: dict) -> None:
    """Overwrite a freshly-constructed session's arrays with loaded ones.

    Parameters
    ----------
    session : QLearningSession | SarsaLambdaSession | ValueIterationSession
        A session just constructed the normal (zero-initialized) way.
    algorithm : str
        Which kind of session ``session`` is (selects which attributes
        to overwrite).
    arrays : dict of str -> ndarray
        Output of :func:`load_model_arrays`.

    Notes
    -----
    For :class:`ValueIterationSession`, this also sets
    ``is_converged = True`` so the session skips straight to a greedy
    rollout of the loaded policy instead of running a fresh sweep --
    the whole point of importing a saved run is to *not* recompute it.
    For :class:`QLearningSession`/:class:`SarsaLambdaSession`, only
    ``Q`` needs overwriting: :meth:`policy`/:meth:`value_function`
    already derive from ``self.Q`` on demand, so nothing else needs to
    change for those to reflect the loaded model.
    """
    if algorithm == "value_iteration":
        if "V" in arrays:
            session.V = arrays["V"]
        if "policy" in arrays:
            session.policy_array = arrays["policy"]
        if "Q" in arrays:
            session.Q = arrays["Q"]
        session.is_converged = True
    else:
        if "Q" in arrays:
            session.Q = arrays["Q"]


def _render_state(env: MazeEnv) -> MazeRenderState:
    s = env.state
    return MazeRenderState(
        grid=env.map_spec.grid,
        agent_state=tuple(s),
        max_energy=env.config.max_energy,
        door_open=bool(s.k),
    )


class _LiveSessionBase:
    """Shared bookkeeping (episode/step counters, success history)."""

    def __init__(self, env: MazeEnv) -> None:
        self.env = env
        self.episode = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.wall_collisions = 0
        self.penalty_entries = 0
        self._outcomes = np.zeros(0, dtype=bool)
        self.last_event: Optional[EventTag] = None
        # Per-state visitation counts, shape (X, Y, 2, max_energy+1),
        # incremented once per environment step at the *pre-transition*
        # state -- mirrors experiments.analysis.visitation_count_from_events'
        # convention, so live and batch-run visitation maps are counted
        # the same way. Feeds the "Visitation Count" analytics tab.
        shape = (*env.map_spec.grid.shape, 2, env.config.max_energy + 1)
        self.visitation_counts = np.zeros(shape, dtype=np.int64)

    def render_state(self) -> MazeRenderState:
        return _render_state(self.env)

    def _record_step(self, s: State, reward: float, event) -> None:
        self.visitation_counts[s.x, s.y, s.k, s.energy] += 1
        self.step_count += 1
        self.episode_reward += reward
        self.last_event = EventTag(event.value)
        if event == Event.WALL_COLLISION:
            self.wall_collisions += 1
        elif event == Event.PENALTY_CELL:
            self.penalty_entries += 1

    def _end_episode(self, success: bool) -> None:
        self._outcomes = np.append(self._outcomes, success)
        self.episode += 1
        self.step_count = 0
        self.episode_reward = 0.0
        self.wall_collisions = 0
        self.penalty_entries = 0
        self.env.reset()

    def visitation_counts_2d(self, k: int) -> np.ndarray:
        """Visitation counts for key-state ``k``, summed over energy.

        Parameters
        ----------
        k : int
            Key-held slice (0 or 1).

        Returns
        -------
        ndarray of shape (X, Y)
            ``visitation_counts[:, :, k, :].sum(axis=-1)``. Summed
            over energy rather than sliced at one fixed value: energy
            decreases monotonically along any trajectory and always
            resets to max_energy on episode start, so any single fixed
            energy value is visited by only a thin, position-correlated
            slice of states (see ``experiments.analysis.reachable_states_mask``'s
            docstring for the same observation in the batch-analysis
            code) -- summing over energy gives a much more informative
            "where has the agent actually been" picture than any one
            energy slice would.
        """
        return self.visitation_counts[:, :, k, :].sum(axis=-1)

    def live_stats(self, mode: RunMode, epsilon: Optional[float]) -> LiveStats:
        s = self.env.state
        return LiveStats(
            episode=self.episode,
            step=self.step_count,
            reward=self.episode_reward,
            epsilon=epsilon,
            key_status=bool(s.k),
            energy=s.energy,
            recent_success_rate=success_rate(self._outcomes),
            mode=mode,
            last_event=self.last_event,
        )


class QLearningSession(_LiveSessionBase):
    """Live, incremental Q-Learning driven one real environment step at a time.

    Uses ``agents.q_learning.epsilon_greedy`` directly for action
    selection, and applies the exact TD-update formula from
    ``agents.q_learning.td_update`` in place (see :meth:`step`'s
    Notes for why) -- so live playback matches the real algorithm
    rather than a GUI-only reimplementation.
    """

    def __init__(self, env: MazeEnv, alpha: float, gamma: float,
                 eps_start: float, eps_end: float, eps_schedule: str,
                 total_episodes: int, seed: int = 0) -> None:
        super().__init__(env)
        shape = (*env.map_spec.grid.shape, 2, env.config.max_energy + 1, len(ACTIONS))
        self.Q = np.zeros(shape, dtype=np.float64)
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.schedule_fn = EPSILON_SCHEDULES[eps_schedule]
        self.total_episodes = total_episodes
        self.rng = np.random.default_rng(seed)

    def current_epsilon(self) -> float:
        return float(self.schedule_fn(self.episode, self.total_episodes,
                                       self.eps_start, self.eps_end))

    def step(self, mode: RunMode) -> None:
        """Advance one environment step; update ``self.Q`` if training.

        Notes
        -----
        Applies the exact Q-Learning update formula from
        ``agents.q_learning.td_update`` in place, rather than calling
        that function directly. ``td_update`` is a pure function that
        returns a full copy of the ``(X, Y, 2, E, A)`` Q-table on every
        call -- correct and appropriate for its role as the tested,
        importable reference implementation, but calling it once per
        GUI timer tick would copy the whole table (hundreds of
        thousands of floats at realistic ``max_energy``) every tick
        for no benefit here, since nothing else holds a reference to
        the old array. ``agents.q_learning.train_q_learning`` avoids
        this the same way in its own hot loop, and
        ``SarsaLambdaSession`` below already follows this same
        in-place pattern for its eligibility-trace update -- this
        keeps the two sessions consistent. The formula itself is
        unchanged and matches ``td_update`` exactly.
        """
        epsilon = 0.0 if mode == RunMode.EVAL else self.current_epsilon()
        s = self.env.state
        a = epsilon_greedy(self.Q, s, epsilon, self.rng)
        s_next, r, done, event = self.env.step(a)

        if mode == RunMode.TRAIN:
            best_next = 0.0 if done else np.max(self.Q[s_next.x, s_next.y, s_next.k, s_next.energy])
            idx = (s.x, s.y, s.k, s.energy, a)
            self.Q[idx] = self.Q[idx] + self.alpha * (r + self.gamma * best_next - self.Q[idx])

        self._record_step(s, r, event)
        success = event == Event.GOAL_REACHED
        if done:
            self._end_episode(success)

    def policy(self) -> np.ndarray:
        """Greedy policy extracted from the current (possibly untrained) Q."""
        return extract_policy_from_Q(self.Q)

    def value_function(self) -> np.ndarray:
        """``max_a Q(s, a)`` per state, for the value-heatmap tab."""
        return np.max(self.Q, axis=-1)


class SarsaLambdaSession(_LiveSessionBase):
    """Live, incremental SARSA(lambda) driven one real environment step at a time.

    Uses ``agents.q_learning.epsilon_greedy`` for action selection and
    ``agents.sarsa_lambda.update_eligibility_trace`` for the trace update
    (replacing traces, as chosen and justified in ``agents/sarsa_lambda.py``).
    The delta/Q update itself is the same two-line formula
    ``train_sarsa_lambda`` uses in its own hot loop (that formula is not
    factored into a standalone importable function there either, so it is
    reproduced here identically rather than duplicated-and-diverged).
    """

    def __init__(self, env: MazeEnv, alpha: float, gamma: float, lam: float,
                 eps_start: float, eps_end: float, eps_schedule: str,
                 total_episodes: int, seed: int = 0) -> None:
        super().__init__(env)
        shape = (*env.map_spec.grid.shape, 2, env.config.max_energy + 1, len(ACTIONS))
        self.Q = np.zeros(shape, dtype=np.float64)
        self.E: dict = {}
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.schedule_fn = EPSILON_SCHEDULES[eps_schedule]
        self.total_episodes = total_episodes
        self.rng = np.random.default_rng(seed)
        self.last_delta: Optional[float] = None
        self._current_action: Optional[int] = None

    def current_epsilon(self) -> float:
        return float(self.schedule_fn(self.episode, self.total_episodes,
                                       self.eps_start, self.eps_end))

    def step(self, mode: RunMode) -> None:
        """Advance one environment step; update ``self.Q``/``self.E`` if training."""
        epsilon = 0.0 if mode == RunMode.EVAL else self.current_epsilon()
        s = self.env.state
        if self._current_action is None:
            self._current_action = epsilon_greedy(self.Q, s, epsilon, self.rng)
        a = self._current_action

        s_next, r, done, event = self.env.step(a)
        a_next = None if done else epsilon_greedy(self.Q, s_next, epsilon, self.rng)

        if mode == RunMode.TRAIN:
            q_sa = self.Q[s.x, s.y, s.k, s.energy, a]
            q_next = 0.0 if done else self.Q[s_next.x, s_next.y, s_next.k, s_next.energy, a_next]
            delta = r + self.gamma * q_next - q_sa
            self.last_delta = float(delta)
            self.E = update_eligibility_trace(self.E, s, a, self.gamma, self.lam)
            for (ex, ey, ek, ee, ea), trace_val in self.E.items():
                self.Q[ex, ey, ek, ee, ea] += self.alpha * delta * trace_val

        self._record_step(s, r, event)
        success = event == Event.GOAL_REACHED
        self._current_action = a_next
        if done:
            self.E = {}
            self._current_action = None
            self._end_episode(success)

    def policy(self) -> np.ndarray:
        return extract_policy_from_Q(self.Q)

    def value_function(self) -> np.ndarray:
        return np.max(self.Q, axis=-1)


class ValueIterationSession(_LiveSessionBase):
    """Value Iteration: one background Bellman-sweep, then a greedy rollout.

    VI has no online trajectory of its own -- "training" is
    ``run_value_iteration``'s global sweep over the full state space,
    run once on a background thread so the GUI event loop doesn't
    freeze. Once converged, :meth:`step` just walks the real
    ``MazeEnv`` greedily under the resulting policy, exactly like an
    eval-mode rollout of the other two algorithms.
    """

    def __init__(self, env: MazeEnv, gamma: float, theta: float,
                 max_iterations: int, reward_version: str) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.reward_version = reward_version
        self.V: Optional[np.ndarray] = None
        self.policy_array: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.n_iterations: Optional[int] = None
        self.runtime_seconds: Optional[float] = None
        self.deltas: Optional[list] = None
        self._sweep_thread: Optional[threading.Thread] = None
        self.is_converged = False
        self.is_running = False

    def start_sweep_async(self, on_done) -> None:
        """Run the full Bellman-sweep on a background thread.

        Parameters
        ----------
        on_done : callable
            Called with no arguments on the *calling* thread once the
            sweep finishes (the caller, i.e. :mod:`gui.controller`, is
            expected to bounce this back onto the Qt event loop via a
            signal rather than touching widgets from this thread).
        """
        if self.is_running:
            return
        self.is_running = True

        def _run():
            reward_fn = REWARD_FNS[self.reward_version]
            V, policy, Q, n_iter, runtime_s, deltas = run_value_iteration(
                self.env.map_spec, self.env.config, reward_fn,
                self.gamma, self.theta, self.max_iterations,
            )
            self.V, self.policy_array, self.Q = V, policy, Q
            self.n_iterations, self.runtime_seconds, self.deltas = n_iter, runtime_s, deltas
            self.is_converged = True
            self.is_running = False
            on_done()

        self._sweep_thread = threading.Thread(target=_run, daemon=True)
        self._sweep_thread.start()

    def step(self, mode: RunMode) -> None:
        """Greedy rollout step under the converged policy (any mode, once ready)."""
        if not self.is_converged:
            return
        s = self.env.state
        a = int(self.policy_array[s.x, s.y, s.k, s.energy])
        s_next, r, done, event = self.env.step(a)
        self._record_step(s, r, event)
        success = event == Event.GOAL_REACHED
        if done:
            self._end_episode(success)

    def policy(self) -> Optional[np.ndarray]:
        return self.policy_array

    def value_function(self) -> Optional[np.ndarray]:
        return self.V

    def live_stats(self, mode: RunMode, epsilon: Optional[float]) -> LiveStats:
        stats = super().live_stats(mode, epsilon=None)
        return stats
