"""Simulation controller: drives a real backend session for the GUI.

Owns the animation timer and the start/stop/resume/reset/re-run
control contract. Talks to whichever
:mod:`gui.backend_adapter` session is currently attached
(:class:`~gui.backend_adapter.QLearningSession`,
:class:`~gui.backend_adapter.SarsaLambdaSession`, or
:class:`~gui.backend_adapter.ValueIterationSession`) through their
shared ``step(mode)`` / ``render_state()`` / ``live_stats()`` surface,
and to the rest of the GUI purely through Qt signals.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from gui.models import RunMode


class SimulationController(QObject):
    """Owns the train/eval loop and emits render-ready state each step.

    Signals
    -------
    frame_ready : MazeRenderState
        Emitted once per environment step.
    stats_changed : LiveStats
        Emitted once per environment step.
    episode_finished : bool
        Success/failure, emitted at the end of every episode.
    vi_sweep_started :
        Emitted the instant a Value Iteration sweep is kicked off (on
        the calling/main thread, *before* the background thread does
        any work) so the GUI can show immediate feedback -- the sweep
        itself commonly takes 15-20+ seconds at realistic max_energy
        values, during which nothing else changes on screen, which
        reads as a frozen/unresponsive app without this signal.
    vi_sweep_finished :
        Emitted once a Value Iteration sweep (started via
        :meth:`start`) completes on its background thread.
    periodic_refresh :
        Emitted every :attr:`_refresh_every` steps regardless of
        episode boundaries. Without this, the analytics tabs (value
        heatmap, policy arrows, visitation count) only ever update at
        the end of an episode -- early in training, episodes commonly
        run long before the agent finds the goal, during which the
        maze animation is visibly stepping but every analytics tab
        stays frozen on its initial (all-zero) state, which reads as
        "the analytics don't work" even though training is happening.
    """

    frame_ready = pyqtSignal(object)
    stats_changed = pyqtSignal(object)
    episode_finished = pyqtSignal(bool)
    vi_sweep_started = pyqtSignal()
    vi_sweep_finished = pyqtSignal()
    periodic_refresh = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._session = None
        self._session_factory: Optional[Callable[[], object]] = None
        self._is_vi = False
        self._mode = RunMode.TRAIN
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._steps_per_tick = 1
        self._refresh_every = 25
        self._steps_since_refresh = 0

    # -- session wiring ---------------------------------------------------

    def set_session(self, session, session_factory: Callable[[], object], is_vi: bool) -> None:
        """Attach a freshly-built session (from ``gui.app``'s "Apply" action).

        Parameters
        ----------
        session : QLearningSession | SarsaLambdaSession | ValueIterationSession
            The session to drive.
        session_factory : callable
            Zero-arg callable that builds a brand-new equivalent
            session (fresh Q-table/episode count), used by
            :meth:`rerun`.
        is_vi : bool
            Whether ``session`` is a :class:`ValueIterationSession`
            (its ``start()`` semantics differ: one background sweep
            instead of a per-tick loop).
        """
        self._timer.stop()
        self._session = session
        self._session_factory = session_factory
        self._is_vi = is_vi
        self._emit_frame_and_stats()

    def set_mode(self, mode: RunMode) -> None:
        """Switch between train (learning updates) and eval (greedy, frozen)."""
        self._mode = mode

    def set_speed(self, steps_per_tick: int, interval_ms: int = 60) -> None:
        """Configure animation speed (steps advanced per timer tick)."""
        self._steps_per_tick = max(1, steps_per_tick)
        self._timer.setInterval(interval_ms)

    # -- controls: start/stop/resume/reset/re-run --------------------------

    def start(self) -> None:
        """Start/resume stepping, or (for VI) kick off the background sweep."""
        if self._session is None:
            raise RuntimeError("No session attached; build one via the GUI's Apply button.")
        if self._is_vi:
            if not self._session.is_converged and not self._session.is_running:
                self.vi_sweep_started.emit()
                self._session.start_sweep_async(self._on_vi_sweep_done)
            else:
                self._timer.start()
            return
        self._timer.start()

    def stop(self) -> None:
        """Pause stepping (state is preserved; call start() to resume)."""
        self._timer.stop()

    def resume(self) -> None:
        """Alias for :meth:`start`, kept distinct for GUI-button clarity."""
        self.start()

    def reset(self) -> None:
        """Reset the current episode only (keeps the learned Q-table/policy)."""
        self._timer.stop()
        if self._session is not None:
            self._session.env.reset()
            self._session.episode = 0
            self._session.step_count = 0
            self._session.episode_reward = 0.0
            self._emit_frame_and_stats()

    def rerun(self) -> None:
        """Restart the whole experiment: a brand-new session (Q reset to zero)."""
        self._timer.stop()
        if self._session_factory is not None:
            was_vi = self._is_vi
            self._session = self._session_factory()
            self._is_vi = was_vi
            self._emit_frame_and_stats()

    # -- stepping -----------------------------------------------------------

    def _tick(self) -> None:
        for _ in range(self._steps_per_tick):
            self._single_step()

    def _single_step(self) -> None:
        session = self._session
        prev_episode = session.episode
        session.step(self._mode)
        self._emit_frame_and_stats()
        if session.episode != prev_episode:
            success = bool(session._outcomes[-1]) if session._outcomes.size else False
            self.episode_finished.emit(success)
            self._steps_since_refresh = 0
        else:
            self._steps_since_refresh += 1
            if self._steps_since_refresh >= self._refresh_every:
                self._steps_since_refresh = 0
                self.periodic_refresh.emit()

    def _on_vi_sweep_done(self) -> None:
        self.vi_sweep_finished.emit()
        self._emit_frame_and_stats()

    # -- signal emission helpers ---------------------------------------------

    def _emit_frame_and_stats(self) -> None:
        if self._session is None:
            return
        self.frame_ready.emit(self._session.render_state())
        epsilon = None
        if hasattr(self._session, "current_epsilon"):
            epsilon = self._session.current_epsilon()
        self.stats_changed.emit(self._session.live_stats(self._mode, epsilon))
