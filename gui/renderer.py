"""Step-by-step maze rendering widget.

Draws the real ``map_spec.grid`` (cell-type ints matching
``environments.generator`` exactly) plus the agent's real
``(x, y, k, energy)`` state. The mandatory extra feature for this
project is **limited energy**, shown as a live readout rather than an
extra marker on the grid (energy has no fixed cell -- it's a resource,
not a location).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QLineF, QPointF, QRectF
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget

from gui.models import Action, CellType, MazeRenderState

_CELL_COLORS = {
    CellType.WALL: QColor(45, 45, 48),
    CellType.NORMAL: QColor(235, 235, 235),
    CellType.PENALTY: QColor(220, 120, 100),
    CellType.START: QColor(140, 200, 140),
    CellType.KEY: QColor(240, 200, 80),
    CellType.DOOR: QColor(120, 140, 220),
    CellType.GOAL: QColor(90, 170, 90),
}

_AGENT_COLOR = QColor(20, 20, 20)
_POLICY_ARROW_COLOR = QColor(30, 30, 30, 220)
_VALUE_HEATMAP_ALPHA = 215
_ENERGY_LOW_COLOR = QColor(200, 40, 40)
_ENERGY_OK_COLOR = QColor(120, 0, 120)


def _value_to_color(value: float, v_min: float, v_max: float) -> QColor:
    """Map a scalar value to a red-to-green heat colour (semi-transparent).

    Notes
    -----
    Uses a wider, more saturated red -> green ramp (low blue, larger
    R/G deltas across the range) plus a higher alpha than a first
    pass at this function used, since the previous narrower,
    lower-opacity blend looked washed out once composited over the
    cell's own base fill color underneath it.
    """
    t = 0.5 if v_max <= v_min else float(np.clip((value - v_min) / (v_max - v_min), 0.0, 1.0))
    r = int(235 * (1 - t) + 25 * t)
    g = int(25 * (1 - t) + 205 * t)
    b = 20
    color = QColor(r, g, b)
    color.setAlpha(_VALUE_HEATMAP_ALPHA)
    return color


class MazeCanvas(QWidget):
    """Widget that draws the maze, agent, energy readout, and overlays."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 440)
        self._render_state: Optional[MazeRenderState] = None
        self._policy_overlay: Optional[np.ndarray] = None  # 2D slice (x, y) -> action
        self._value_overlay: Optional[np.ndarray] = None   # 2D slice (x, y) -> value
        self._path_trace: list = []
        self._show_policy = False
        self._show_value = False
        self._show_visited_path = False

    # -- public API -----------------------------------------------------

    def set_render_state(self, state: MazeRenderState) -> None:
        """Replace the current frame and repaint."""
        self._render_state = state
        self._path_trace.append(state.agent_state[:2])
        self.update()

    def clear_path_trace(self) -> None:
        """Drop the recorded agent path (called on reset/re-run)."""
        self._path_trace = []
        self.update()

    def set_policy_overlay(self, policy_2d: Optional[np.ndarray]) -> None:
        """Set a 2D ``(x, y) -> action`` slice, already sliced by k/energy."""
        self._policy_overlay = policy_2d
        self.update()

    def set_value_overlay(self, value_2d: Optional[np.ndarray]) -> None:
        """Set a 2D ``(x, y) -> value`` slice, already sliced by k/energy."""
        self._value_overlay = value_2d
        self.update()

    def toggle_policy_overlay(self, on: bool) -> None:
        self._show_policy = on
        self.update()

    def toggle_value_overlay(self, on: bool) -> None:
        self._show_value = on
        self.update()

    def toggle_path_trace(self, on: bool) -> None:
        self._show_visited_path = on
        self.update()

    def save_image(self, path: str) -> None:
        """Save the current frame to ``path`` (png by extension)."""
        self.grab().save(path)

    # -- Qt paint event ---------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override name)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._render_state is None:
            painter.fillRect(self.rect(), QColor(250, 250, 250))
            painter.drawText(self.rect(), Qt.AlignCenter, "No maze loaded")
            return

        grid = self._render_state.grid
        n_x, n_y = grid.shape
        header_h = 22
        # Independent x/y cell size (rather than a single square size
        # clamped to the smaller dimension) so the grid fills the
        # entire widget with no wasted margin, regardless of the
        # widget's aspect ratio. Cells become rectangular instead of
        # perfectly square when the widget isn't square, but every
        # color/label/overlay is otherwise unchanged.
        cell_w = self.width() / n_x
        cell_h = (self.height() - header_h) / n_y
        ox, oy = 0.0, float(header_h)

        self._paint_energy_bar(painter, header_h)
        self._paint_cells(painter, grid, cell_w, cell_h, ox, oy)
        if self._show_value and self._value_overlay is not None:
            self._paint_value_overlay(painter, grid, cell_w, cell_h, ox, oy)
        if self._show_visited_path:
            self._paint_path_trace(painter, cell_w, cell_h, ox, oy)
        if self._show_policy and self._policy_overlay is not None:
            self._paint_policy_overlay(painter, grid, cell_w, cell_h, ox, oy)
        self._paint_agent(painter, cell_w, cell_h, ox, oy)

    # -- private drawing helpers ------------------------------------------

    def _paint_energy_bar(self, painter, header_h) -> None:
        state = self._render_state
        energy = state.agent_state[3]
        frac = 0.0 if state.max_energy <= 0 else max(0.0, min(1.0, energy / state.max_energy))
        color = _ENERGY_LOW_COLOR if frac < 0.15 else _ENERGY_OK_COLOR
        bar_rect = QRectF(4, 3, (self.width() - 8) * frac, header_h - 8)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(230, 230, 230))
        painter.drawRect(QRectF(4, 3, self.width() - 8, header_h - 8))
        painter.setBrush(color)
        painter.drawRect(bar_rect)
        painter.setPen(QColor(20, 20, 20))
        painter.setFont(QFont("Sans", 9))
        painter.drawText(QRectF(0, 0, self.width(), header_h), Qt.AlignCenter,
                          f"Energy: {energy} / {state.max_energy}")

    def _paint_cells(self, painter, grid, cell_w, cell_h, ox, oy) -> None:
        # grid is indexed grid[x, y] (backend convention: x horizontal,
        # y vertical, UP decreases y) -- iterate the same way agent/
        # overlay drawing below does, so a cell and the agent standing
        # on it always land at the same screen position.
        n_x, n_y = grid.shape
        painter.setBrush(Qt.NoBrush)
        label_font_size = max(6, min(10, int(min(cell_w, cell_h) * 0.22)))
        label_font = QFont("Sans", label_font_size)
        for x in range(n_x):
            for y in range(n_y):
                cell_type = CellType(int(grid[x, y]))
                rect = QRectF(ox + x * cell_w, oy + y * cell_h, cell_w, cell_h)
                painter.fillRect(rect, _CELL_COLORS[cell_type])
                # NoBrush is essential here: drawRect() fills using the
                # painter's *current* brush as well as stroking with the
                # current pen. Without resetting it, whatever brush was
                # last set elsewhere (e.g. _paint_energy_bar's solid
                # purple energy-bar brush) silently overwrites the
                # fillRect() color above on every single cell, which is
                # exactly the "everything renders as one solid color"
                # bug this guards against.
                painter.setPen(QPen(QColor(200, 200, 200), 1))
                painter.drawRect(rect)
                # Special cells spell out what they're for (rather than
                # a bare single letter) so their purpose is clear at a
                # glance; font size scales with cell size so the words
                # still fit as the grid gets denser.
                if cell_type == CellType.START:
                    painter.setPen(QColor(20, 20, 20))
                    painter.setFont(label_font)
                    painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, "Start")
                elif cell_type == CellType.KEY and not self._render_state.door_open:
                    painter.setPen(QColor(20, 20, 20))
                    painter.setFont(label_font)
                    painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, "Key")
                elif cell_type == CellType.DOOR:
                    status = "Open" if self._render_state.door_open else "Closed"
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(label_font)
                    painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, f"Door\n{status}")
                elif cell_type == CellType.GOAL:
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(label_font)
                    painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, "Goal")

    def _paint_value_overlay(self, painter, grid, cell_w, cell_h, ox, oy) -> None:
        values = self._value_overlay
        finite = values[np.isfinite(values)]
        v_min, v_max = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        n_x, n_y = grid.shape
        for x in range(n_x):
            for y in range(n_y):
                if CellType(int(grid[x, y])) == CellType.WALL:
                    continue
                rect = QRectF(ox + x * cell_w, oy + y * cell_h, cell_w, cell_h)
                painter.fillRect(rect, _value_to_color(values[x, y], v_min, v_max))

    def _paint_path_trace(self, painter, cell_w, cell_h, ox, oy) -> None:
        painter.setPen(QPen(QColor(30, 30, 200, 140), 2))
        painter.setBrush(QColor(30, 30, 200, 140))
        for (x, y) in self._path_trace:
            center = (ox + x * cell_w + cell_w / 2, oy + y * cell_h + cell_h / 2)
            painter.drawEllipse(QRectF(center[0] - 2, center[1] - 2, 4, 4))

    def _paint_policy_overlay(self, painter, grid, cell_w, cell_h, ox, oy) -> None:
        painter.setPen(QPen(_POLICY_ARROW_COLOR, 2))
        painter.setBrush(_POLICY_ARROW_COLOR)
        arrow_len_x = cell_w * 0.3
        arrow_len_y = cell_h * 0.3
        head_size = min(cell_w, cell_h) * 0.13
        deltas = {
            Action.UP: (0, -1), Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0), Action.RIGHT: (1, 0),
        }
        n_x, n_y = grid.shape
        for x in range(n_x):
            for y in range(n_y):
                if CellType(int(grid[x, y])) == CellType.WALL:
                    continue
                action = int(self._policy_overlay[x, y])
                cx, cy = ox + x * cell_w + cell_w / 2, oy + y * cell_h + cell_h / 2
                dx, dy = deltas[Action(action)]
                tip_x, tip_y = cx + dx * arrow_len_x, cy + dy * arrow_len_y
                painter.drawLine(QLineF(cx, cy, tip_x, tip_y))
                # Small pointy arrowhead at the tip, matching the
                # matplotlib "Final Policy" analytics tab's arrow
                # style rather than a bare line. Actions are
                # axis-aligned unit deltas, so the perpendicular
                # direction for the two base corners is just (-dy, dx).
                perp_x, perp_y = -dy, dx
                base_x, base_y = tip_x - dx * head_size, tip_y - dy * head_size
                left = QPointF(base_x + perp_x * head_size * 0.6, base_y + perp_y * head_size * 0.6)
                right = QPointF(base_x - perp_x * head_size * 0.6, base_y - perp_y * head_size * 0.6)
                painter.drawPolygon(QPolygonF([QPointF(tip_x, tip_y), left, right]))

    def _paint_agent(self, painter, cell_w, cell_h, ox, oy) -> None:
        x, y = self._render_state.agent_state[0], self._render_state.agent_state[1]
        center = (ox + x * cell_w + cell_w / 2, oy + y * cell_h + cell_h / 2)
        painter.setBrush(_AGENT_COLOR)
        painter.setPen(Qt.NoPen)
        radius = min(cell_w, cell_h) * 0.3
        painter.drawEllipse(QRectF(center[0] - radius, center[1] - radius, radius * 2, radius * 2))
