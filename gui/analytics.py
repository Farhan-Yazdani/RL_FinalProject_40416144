"""Analytics panel: the matplotlib half of the GUI.

Real V/Q/policy arrays from ``agents/*`` are 4D, shaped
``(X, Y, 2, max_energy+1[, A])`` (see ``agents/value_iteration.py``'s
``build_state_index``). This panel only ever draws 2D ``(X, Y)``
slices -- callers (``gui.app``) are responsible for slicing at a
chosen ``(k, energy)`` before calling any ``update_*`` method here,
via the "Key held" / "Energy" controls in the side panel. Keeping the
slicing decision in ``app.py`` (not here) means this panel stays a
generic 2D-array renderer, reusable for any per-state scalar/action
field regardless of how many extra state dimensions a future feature
adds.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QWidget


class _FigureTab(QWidget):
    """A single QTabWidget page wrapping one matplotlib Figure."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self._draw_placeholder()

    def reset_axes(self):
        """Clear the whole figure and recreate a single fresh subplot.

        Returns
        -------
        matplotlib.axes.Axes
            The newly created axes (also stored as ``self.ax``).

        Notes
        -----
        ``self.ax.clear()`` alone is *not* enough when a previous draw
        added a colorbar: ``figure.colorbar(...)`` creates its own
        separate axes on the figure (not a child of ``self.ax``), so
        clearing only ``self.ax`` leaves that colorbar axes in place.
        Every subsequent update then adds *another* colorbar next to
        it -- e.g. toggling the "Key held" checkbox repeatedly grows
        an extra vertical color-scale bar each time. Clearing the
        whole figure and recreating one subplot avoids this by
        construction: there is never more than one axes (and at most
        one colorbar) on the figure after any update.
        """
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        return self.ax

    def _draw_placeholder(self) -> None:
        self.ax.clear()
        self.ax.set_title(self.title)
        self.ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                      transform=self.ax.transAxes, color="gray")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw_idle()

    def save(self, path: str) -> None:
        self.figure.savefig(path)


class AnalyticsPanel(QWidget):
    """Tabbed container for all required analytics figures (2D slices only)."""

    TAB_NAMES = (
        "value_heatmap",
        "policy_arrows",
        "visitation_map",
        "policy_disagreement",
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        self._tabs: dict[str, _FigureTab] = {}
        titles = {
            "value_heatmap": "Value Heatmap",
            "policy_arrows": "Final Policy",
            "visitation_map": "Visitation Count",
            "policy_disagreement": "Policy Disagreement",
        }
        for name in self.TAB_NAMES:
            tab = _FigureTab(titles[name])
            self._tabs[name] = tab
            self.tab_widget.addTab(tab, titles[name])

    # -- public update API (all inputs are 2D (X, Y) slices) ---------------

    def update_value_heatmap(self, values_2d: np.ndarray, title: str = "Value Heatmap") -> None:
        """Redraw the value-function / max-Q heatmap.

        Parameters
        ----------
        values_2d : ndarray of shape (X, Y)
            ``V[:, :, k, energy]`` or ``max_a Q[:, :, k, energy, a]``
            for the caller's chosen ``(k, energy)`` slice.
        """
        tab = self._tabs["value_heatmap"]
        tab.reset_axes()
        im = tab.ax.imshow(values_2d.T, cmap="viridis", origin="upper")
        tab.ax.set_title(title)
        tab.figure.colorbar(im, ax=tab.ax, fraction=0.046)
        tab.canvas.draw_idle()

    def update_policy_arrows(self, policy_2d: np.ndarray, walls_2d: np.ndarray,
                              terminal_mask_2d: Optional[np.ndarray] = None) -> None:
        """Redraw the final greedy-policy arrow field.

        Parameters
        ----------
        policy_2d : ndarray of shape (X, Y)
            Greedy action index (0=up,1=down,2=left,3=right) per cell,
            for the caller's chosen ``(k, energy)`` slice.
        walls_2d : ndarray of shape (X, Y), dtype bool
            ``True`` where the cell is a wall -- drawn as a filled
            dark square (matching the maze canvas's wall color)
            rather than left blank, so it's visually clear the cell
            has no policy arrow *because* it's a wall, not because
            data is missing.
        terminal_mask_2d : ndarray of shape (X, Y), dtype bool, optional
            ``True`` at the goal position.
        """
        tab = self._tabs["policy_arrows"]
        tab.reset_axes()
        deltas = {0: (0, 0.4), 1: (0, -0.4), 2: (-0.4, 0), 3: (0.4, 0)}
        n_x, n_y = policy_2d.shape
        for x in range(n_x):
            for y in range(n_y):
                if walls_2d[x, y]:
                    tab.ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                                                facecolor="#2b2b2b", edgecolor="none", zorder=0))
                    continue
                if terminal_mask_2d is not None and terminal_mask_2d[x, y]:
                    tab.ax.plot(x, y, marker="*", color="red", markersize=10)
                    continue
                dx, dy = deltas[int(policy_2d[x, y])]
                tab.ax.arrow(x, y, dx, dy, head_width=0.15, color="black")
        tab.ax.set_xlim(-0.5, n_x - 0.5)
        tab.ax.set_ylim(-0.5, n_y - 0.5)
        tab.ax.set_title("Final Policy")
        tab.ax.set_aspect("equal")
        tab.canvas.draw_idle()

    def update_visitation_map(self, counts_2d: np.ndarray) -> None:
        """Redraw the state-visitation count heatmap.

        Parameters
        ----------
        counts_2d : ndarray of shape (X, Y)
            Visitation counts, e.g. summed over ``k``/``energy`` from
            ``experiments.analysis.visitation_count_from_events``, or
            a single slice.
        """
        tab = self._tabs["visitation_map"]
        tab.reset_axes()
        im = tab.ax.imshow(counts_2d.T, cmap="magma", origin="upper")
        tab.ax.set_title("Visitation Count")
        tab.figure.colorbar(im, ax=tab.ax, fraction=0.046)
        tab.canvas.draw_idle()

    def update_policy_disagreement(self, reference_policy_2d: np.ndarray,
                                    candidate_policy_2d: np.ndarray,
                                    walls_2d: np.ndarray) -> float:
        """Redraw the policy-disagreement map vs. the VI reference.

        Parameters
        ----------
        reference_policy_2d : ndarray of shape (X, Y)
            Value Iteration's greedy policy slice (the baseline).
        candidate_policy_2d : ndarray of shape (X, Y)
            Q-Learning or SARSA(lambda)'s greedy policy slice.
        walls_2d : ndarray of shape (X, Y), dtype bool
            Wall mask, excluded from the agreement statistic.

        Returns
        -------
        float
            Percentage of non-wall states where the two policies agree.
        """
        tab = self._tabs["policy_disagreement"]
        tab.reset_axes()
        disagreement = (reference_policy_2d != candidate_policy_2d).astype(float)
        disagreement = np.where(walls_2d, np.nan, disagreement)
        im = tab.ax.imshow(disagreement.T, cmap="RdYlGn_r", vmin=0, vmax=1, origin="upper")
        tab.ax.set_title("Policy Disagreement (red = differs from VI)")
        tab.figure.colorbar(im, ax=tab.ax, fraction=0.046)
        tab.canvas.draw_idle()
        valid = ~walls_2d
        agreement_pct = 100.0 * float(np.mean(
            reference_policy_2d[valid] == candidate_policy_2d[valid]
        )) if valid.any() else 0.0
        return agreement_pct

    def save_tab(self, tab_name: str, path: str) -> None:
        """Save one named tab's figure to disk."""
        if tab_name not in self._tabs:
            raise ValueError(f"Unknown analytics tab: {tab_name!r}")
        self._tabs[tab_name].save(path)

    def save_all(self, directory: str) -> None:
        """Save every tab's figure into ``directory`` using its tab name."""
        import os
        os.makedirs(directory, exist_ok=True)
        for name, tab in self._tabs.items():
            tab.save(os.path.join(directory, f"{name}.png"))
