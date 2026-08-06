"""Main application window for the Dynamic Maze RL project.

Assembles the maze animation frame (:class:`gui.renderer.MazeCanvas`)
and the analytics frame (:class:`gui.analytics.AnalyticsPanel`) as
independent, toggleable panes in a ``QSplitter``, plus every control
required by ``final_project.md`` / ``CODING_STYLE.md`` 1.7, wired to
the **real** backend through :mod:`gui.backend_adapter` -- no mocks.

Because the real V/Q/policy arrays are 4D (``x, y, k, energy``), this
window also owns the "Key held" / "Energy" slice selectors that pick
which 2D cross-section the maze overlay and analytics tabs show.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.analytics import AnalyticsPanel
from gui.backend_adapter import (
    BackendConfig,
    QLearningSession,
    SarsaLambdaSession,
    ValueIterationSession,
    build_environment,
)
from gui.controller import SimulationController
from gui.models import AlgorithmName, CellType, LiveStats, MapName, RunMode
from gui.renderer import MazeCanvas


class MainWindow(QMainWindow):
    """Top-level window wiring controls, the controller, and both frames."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dynamic Maze RL — GUI")
        self.resize(1250, 780)

        self.controller = SimulationController(self)
        self.maze_canvas = MazeCanvas()
        self.analytics_panel = AnalyticsPanel()
        self._current_env = None

        self._build_layout()
        self._connect_controller_signals()

    # -- layout construction --------------------------------------------------

    def _build_layout(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.maze_canvas)
        splitter.addWidget(self.analytics_panel)
        splitter.setSizes([650, 600])
        root.addWidget(splitter, stretch=1)
        root.addWidget(self._build_controls_panel())

    def _build_controls_panel(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self._build_selection_group())
        layout.addWidget(self._build_hyperparam_group())
        layout.addWidget(self._build_mode_group())
        layout.addWidget(self._build_playback_group())
        layout.addWidget(self._build_slice_group())
        layout.addWidget(self._build_overlay_group())
        layout.addWidget(self._build_frame_visibility_group())
        layout.addWidget(self._build_live_info_group())
        layout.addWidget(self._build_export_group())
        layout.addStretch(1)

        # A plain QVBoxLayout of this many group boxes has no way to
        # shrink gracefully: if the sidebar's available height is ever
        # less than the stack's natural total height (which happens
        # when maximizing on some screen/DPI combinations, since a
        # maximized window's *width* can grow a lot more than its
        # *height*), Qt compresses every widget below its preferred
        # size instead of showing a scrollbar -- which is exactly what
        # made "Algorithm/Environment" and "Hyperparameters" become
        # illegibly small. Wrapping the content in a QScrollArea lets
        # every group box keep its natural size unconditionally; a
        # vertical scrollbar appears instead of any squeezing whenever
        # the viewport is shorter than the content.
        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(300)
        return scroll

    def _build_selection_group(self) -> QGroupBox:
        box = QGroupBox("Algorithm / Environment")
        grid = QGridLayout(box)

        self.student_id_edit = QLineEdit("40416144")
        grid.addWidget(QLabel("Student ID:"), 0, 0)
        grid.addWidget(self.student_id_edit, 0, 1)

        self.maps_dir_edit = QLineEdit("environments/maps")
        grid.addWidget(QLabel("Maps dir:"), 1, 0)
        grid.addWidget(self.maps_dir_edit, 1, 1)

        self.env_combo = QComboBox()
        self.env_combo.addItems([m.value for m in MapName])
        grid.addWidget(QLabel("Map:"), 2, 0)
        grid.addWidget(self.env_combo, 2, 1)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems([a.value for a in AlgorithmName])
        self.algo_combo.currentTextChanged.connect(self._on_algorithm_changed)
        grid.addWidget(QLabel("Algorithm:"), 3, 0)
        grid.addWidget(self.algo_combo, 3, 1)

        self.reward_version_combo = QComboBox()
        self.reward_version_combo.addItems(["sparse", "shaped"])
        grid.addWidget(QLabel("Reward:"), 4, 0)
        grid.addWidget(self.reward_version_combo, 4, 1)

        apply_btn = QPushButton("Build / Apply")
        apply_btn.clicked.connect(self._on_apply_clicked)
        grid.addWidget(apply_btn, 5, 0, 1, 2)
        return box

    def _build_hyperparam_group(self) -> QGroupBox:
        box = QGroupBox("Hyperparameters")
        grid = QGridLayout(box)
        self.alpha_edit = QLineEdit("0.1")
        self.gamma_edit = QLineEdit("0.95")
        self.eps_start_edit = QLineEdit("1.0")
        self.eps_end_edit = QLineEdit("0.05")
        self.lambda_edit = QLineEdit("0.7")
        self.total_episodes_edit = QLineEdit("2000")
        self.theta_edit = QLineEdit("1e-4")
        self.max_iterations_edit = QLineEdit("2000")
        self.eps_schedule_combo = QComboBox()
        self.eps_schedule_combo.addItems(["linear", "exponential"])

        rows = [
            ("alpha:", self.alpha_edit), ("gamma:", self.gamma_edit),
            ("eps_start:", self.eps_start_edit), ("eps_end:", self.eps_end_edit),
            ("eps_schedule:", self.eps_schedule_combo),
            ("lambda (SARSA):", self.lambda_edit),
            ("total_episodes:", self.total_episodes_edit),
            ("theta (VI):", self.theta_edit),
            ("max_iterations (VI):", self.max_iterations_edit),
        ]
        for i, (label, widget) in enumerate(rows):
            grid.addWidget(QLabel(label), i, 0)
            grid.addWidget(widget, i, 1)
        return box

    def _build_mode_group(self) -> QGroupBox:
        box = QGroupBox("Mode")
        layout = QHBoxLayout(box)
        self.train_radio = QRadioButton("Train")
        self.eval_radio = QRadioButton("Eval")
        self.train_radio.setChecked(True)
        self.train_radio.toggled.connect(self._on_mode_changed)
        layout.addWidget(self.train_radio)
        layout.addWidget(self.eval_radio)
        return box

    def _build_playback_group(self) -> QGroupBox:
        box = QGroupBox("Playback")
        layout = QVBoxLayout(box)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.resume_button = QPushButton("Resume")
        self.reset_button = QPushButton("Reset")
        self.rerun_button = QPushButton("Re-run")
        for btn in (self.start_button, self.stop_button, self.resume_button,
                    self.reset_button, self.rerun_button):
            buttons.addWidget(btn)
        layout.addLayout(buttons)

        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self.controller.stop)
        self.resume_button.clicked.connect(self.controller.resume)
        self.reset_button.clicked.connect(self._on_reset_clicked)
        self.rerun_button.clicked.connect(self._on_rerun_clicked)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(1)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.speed_slider)
        layout.addLayout(speed_row)
        self._on_speed_changed(self.speed_slider.value())

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #b35c00; font-style: italic;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        return box

    def _build_slice_group(self) -> QGroupBox:
        """(k, energy) slice selector for the 4D V/Q/policy arrays."""
        box = QGroupBox("Analytics Slice (k, energy)")
        layout = QHBoxLayout(box)
        self.key_held_checkbox = QCheckBox("Key held (k=1)")
        self.key_held_checkbox.toggled.connect(self._on_slice_changed)
        layout.addWidget(self.key_held_checkbox)
        layout.addWidget(QLabel("Energy:"))
        self.energy_spinbox = QSpinBox()
        self.energy_spinbox.setRange(0, 100000)
        self.energy_spinbox.valueChanged.connect(self._on_slice_changed)
        layout.addWidget(self.energy_spinbox)
        return box

    def _build_overlay_group(self) -> QGroupBox:
        box = QGroupBox("Overlays")
        layout = QVBoxLayout(box)
        self.policy_overlay_checkbox = QCheckBox("Show policy arrows")
        self.policy_overlay_checkbox.toggled.connect(self._on_overlay_toggle)
        self.value_overlay_checkbox = QCheckBox("Show value heatmap")
        self.value_overlay_checkbox.toggled.connect(self._on_overlay_toggle)
        self.path_trace_checkbox = QCheckBox("Show visited path")
        self.path_trace_checkbox.toggled.connect(self.maze_canvas.toggle_path_trace)
        layout.addWidget(self.policy_overlay_checkbox)
        layout.addWidget(self.value_overlay_checkbox)
        layout.addWidget(self.path_trace_checkbox)
        return box

    def _build_frame_visibility_group(self) -> QGroupBox:
        box = QGroupBox("Panels")
        layout = QVBoxLayout(box)
        self.show_maze_checkbox = QCheckBox("Maze animation")
        self.show_maze_checkbox.setChecked(True)
        self.show_maze_checkbox.toggled.connect(self.maze_canvas.setVisible)
        self.show_analytics_checkbox = QCheckBox("Analytics panel")
        self.show_analytics_checkbox.setChecked(True)
        self.show_analytics_checkbox.toggled.connect(self.analytics_panel.setVisible)
        layout.addWidget(self.show_maze_checkbox)
        layout.addWidget(self.show_analytics_checkbox)
        return box

    def _build_live_info_group(self) -> QGroupBox:
        box = QGroupBox("Live Info")
        grid = QGridLayout(box)
        self.episode_label = QLabel("0")
        self.step_label = QLabel("0")
        self.reward_label = QLabel("0.0")
        self.epsilon_label = QLabel("—")
        self.key_label = QLabel("No")
        self.energy_label = QLabel("—")
        self.success_rate_label = QLabel("0%")
        rows = [
            ("Episode:", self.episode_label), ("Step:", self.step_label),
            ("Reward:", self.reward_label), ("Epsilon:", self.epsilon_label),
            ("Key held:", self.key_label), ("Energy:", self.energy_label),
            ("Success rate:", self.success_rate_label),
        ]
        for i, (label_text, value_widget) in enumerate(rows):
            grid.addWidget(QLabel(label_text), i, 0)
            grid.addWidget(value_widget, i, 1)
        return box

    def _build_export_group(self) -> QGroupBox:
        box = QGroupBox("Export")
        layout = QVBoxLayout(box)
        save_maze_btn = QPushButton("Save maze frame…")
        save_maze_btn.clicked.connect(self._on_save_maze_frame)
        save_analytics_btn = QPushButton("Save all analytics figures…")
        save_analytics_btn.clicked.connect(self._on_save_analytics)
        layout.addWidget(save_maze_btn)
        layout.addWidget(save_analytics_btn)
        return box

    # -- controller <-> widgets wiring ---------------------------------------

    def _connect_controller_signals(self) -> None:
        self.controller.frame_ready.connect(self.maze_canvas.set_render_state)
        self.controller.stats_changed.connect(self._on_stats_changed)
        self.controller.episode_finished.connect(self._on_episode_finished)
        self.controller.periodic_refresh.connect(self._refresh_analytics_and_overlays)
        self.controller.vi_sweep_started.connect(self._on_vi_sweep_started)
        self.controller.vi_sweep_finished.connect(self._on_vi_sweep_finished)

    def _on_stats_changed(self, stats: LiveStats) -> None:
        self.episode_label.setText(str(stats.episode))
        self.step_label.setText(str(stats.step))
        self.reward_label.setText(f"{stats.reward:.2f}")
        self.epsilon_label.setText("—" if stats.epsilon is None else f"{stats.epsilon:.3f}")
        self.key_label.setText("Yes" if stats.key_status else "No")
        self.energy_label.setText("—" if stats.energy is None else str(stats.energy))
        self.success_rate_label.setText(f"{stats.recent_success_rate * 100:.1f}%")

    def _on_episode_finished(self, success: bool) -> None:
        self._refresh_analytics_and_overlays()

    def _on_vi_sweep_started(self) -> None:
        self.status_label.setText(
            "Running Value Iteration sweep in the background — this can "
            "take 15-20+ seconds at the default max_energy. The window "
            "will stay responsive; a message will appear when it's done."
        )
        self.start_button.setEnabled(False)

    def _on_vi_sweep_finished(self) -> None:
        session = getattr(self.controller, "_session", None)
        n_iter = getattr(session, "n_iterations", None)
        runtime_s = getattr(session, "runtime_seconds", None)
        if n_iter is not None and runtime_s is not None:
            self.status_label.setText(
                f"Sweep converged in {n_iter} iterations ({runtime_s:.1f}s). "
                f"Click Start again to watch a greedy rollout of the policy."
            )
        else:
            self.status_label.setText("Sweep converged. Click Start again for a greedy rollout.")
        self.start_button.setEnabled(True)
        self._refresh_analytics_and_overlays()

    def _on_mode_changed(self, checked: bool) -> None:
        self.controller.set_mode(RunMode.TRAIN if self.train_radio.isChecked() else RunMode.EVAL)

    def _on_speed_changed(self, value: int) -> None:
        self.controller.set_speed(steps_per_tick=value, interval_ms=max(10, 200 - value * 9))

    def _on_reset_clicked(self) -> None:
        self.controller.reset()
        self.maze_canvas.clear_path_trace()

    def _on_rerun_clicked(self) -> None:
        self.controller.rerun()
        self.maze_canvas.clear_path_trace()

    def _on_start_clicked(self) -> None:
        try:
            self.controller.start()
        except RuntimeError as exc:
            QMessageBox.warning(self, "No session", str(exc))

    def _on_save_maze_frame(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save maze frame", "maze_frame.png",
                                               "PNG Image (*.png)")
        if path:
            self.maze_canvas.save_image(path)

    def _on_save_analytics(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if directory:
            self.analytics_panel.save_all(directory)

    def _on_algorithm_changed(self, name: str) -> None:
        is_sarsa = name == AlgorithmName.SARSA_LAMBDA.value
        self.lambda_edit.setEnabled(is_sarsa)

    def _on_slice_changed(self, *_args) -> None:
        self._refresh_analytics_and_overlays()

    def _on_overlay_toggle(self, *_args) -> None:
        self.maze_canvas.toggle_policy_overlay(self.policy_overlay_checkbox.isChecked())
        self.maze_canvas.toggle_value_overlay(self.value_overlay_checkbox.isChecked())
        self._refresh_analytics_and_overlays()

    # -- (re)building the backend session --------------------------------------

    def _current_slice(self) -> tuple:
        """Return the ``(k, energy)`` slice currently selected in the sidebar."""
        return (1 if self.key_held_checkbox.isChecked() else 0, self.energy_spinbox.value())

    def _on_apply_clicked(self) -> None:
        try:
            env = build_environment(BackendConfig(
                student_id=self.student_id_edit.text().strip(),
                map_name=self.env_combo.currentText(),
                maps_dir=self.maps_dir_edit.text().strip(),
                reward_version=self.reward_version_combo.currentText(),
            ))
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to build environment", str(exc))
            return

        self._current_env = env
        self.energy_spinbox.setMaximum(env.config.max_energy)
        self.energy_spinbox.setValue(env.config.max_energy)

        factory = self._make_session_factory(env)
        try:
            session = factory()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Failed to build agent", str(exc))
            return

        is_vi = self.algo_combo.currentText() == AlgorithmName.VALUE_ITERATION.value
        self.controller.set_session(session, factory, is_vi)
        self.maze_canvas.clear_path_trace()
        self._refresh_analytics_and_overlays()

    def _make_session_factory(self, env):
        algo = self.algo_combo.currentText()
        alpha, gamma = float(self.alpha_edit.text()), float(self.gamma_edit.text())
        eps_start, eps_end = float(self.eps_start_edit.text()), float(self.eps_end_edit.text())
        eps_schedule = self.eps_schedule_combo.currentText()
        total_episodes = int(self.total_episodes_edit.text())

        if algo == AlgorithmName.Q_LEARNING.value:
            return lambda: QLearningSession(
                env, alpha, gamma, eps_start, eps_end, eps_schedule, total_episodes
            )
        if algo == AlgorithmName.SARSA_LAMBDA.value:
            lam = float(self.lambda_edit.text())
            return lambda: SarsaLambdaSession(
                env, alpha, gamma, lam, eps_start, eps_end, eps_schedule, total_episodes
            )
        # value_iteration
        theta = float(self.theta_edit.text())
        max_iterations = int(self.max_iterations_edit.text())
        reward_version = self.reward_version_combo.currentText()
        return lambda: ValueIterationSession(env, gamma, theta, max_iterations, reward_version)

    # -- analytics/overlay refresh (slices the real 4D arrays) -----------------

    def _refresh_analytics_and_overlays(self) -> None:
        session = getattr(self.controller, "_session", None)
        if session is None or self._current_env is None:
            return
        k, energy = self._current_slice()
        wall_mask = self._current_env.map_spec.grid == CellType.WALL.value

        value_fn = session.value_function() if hasattr(session, "value_function") else None
        if value_fn is not None:
            energy = min(energy, value_fn.shape[3] - 1)
            v_slice = value_fn[:, :, k, energy]
            self.analytics_panel.update_value_heatmap(v_slice)
            self.maze_canvas.set_value_overlay(v_slice)

        policy = session.policy() if hasattr(session, "policy") else None
        if policy is not None:
            energy = min(energy, policy.shape[3] - 1)
            p_slice = policy[:, :, k, energy]
            self.analytics_panel.update_policy_arrows(p_slice, wall_mask)
            self.maze_canvas.set_policy_overlay(p_slice)

        if hasattr(session, "visitation_counts_2d"):
            counts_2d = session.visitation_counts_2d(k)
            self.analytics_panel.update_visitation_map(counts_2d)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone GUI launch."""
    parser = argparse.ArgumentParser(description="Dynamic Maze RL — GUI")
    parser.add_argument("--window-width", type=int, default=1250)
    parser.add_argument("--window-height", type=int, default=780)
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    """Standalone entry point: run from the project root as

    ``python -m gui.app`` (so ``environments``/``agents``/``experiments``
    resolve as siblings of ``gui``).
    """
    args = parse_args(argv)
    app = QApplication(sys.argv[:1])
    window = MainWindow()
    window.resize(args.window_width, args.window_height)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
