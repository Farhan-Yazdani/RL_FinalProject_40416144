"""Pure rendering functions for the maze GUI and visual analytics outputs.

Implements the "Visual Outputs" requirements of final_project.md:
value heatmap, policy arrows, visitation map, final path, policy
disagreement map, pre/post-transfer Q-value diff. Kept separate from
``gui/app.py`` (the interactive PyQt/Tkinter/Pygame controller) per
``CODING_STYLE.md`` 2.2's side-effect isolation: everything here is a
pure function that takes arrays and returns a matplotlib Figure, with
no widget/event-loop code.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

from environments.generator import DOOR, GOAL, KEY, NORMAL, PENALTY, START, WALL

CELL_COLORS = {
    NORMAL: "#f5f5f0",
    WALL: "#2b2b2b",
    PENALTY: "#e07a5f",
    START: "#81b29a",
    KEY: "#f2cc8f",
    DOOR: "#6d597a",
    GOAL: "#3d405b",
}

ACTION_ARROWS = {
    0: (0, -0.35),  # UP
    1: (0, 0.35),   # DOWN
    2: (-0.35, 0),  # LEFT
    3: (0.35, 0),   # RIGHT
}


def render_grid(ax, grid: np.ndarray, agent_pos=None):
    """Draw the static maze grid (walls, cells, key/door/goal) onto ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    grid : ndarray of shape (size, size)
        Cell-type grid.
    agent_pos : tuple of int, optional
        ``(x, y)`` position to draw the agent marker at.

    Returns
    -------
    matplotlib.axes.Axes
        The same ``ax``, for chaining.

    Notes
    -----
    Every cell type required by the spec ("Walls, normal cells,
    penalty cells, the key, the door, the goal, the agent ... must all
    have visually distinct markers") gets a distinct fill color from
    :data:`CELL_COLORS`. The limited-energy feature is not spatially
    localized, so it is shown via the live info panel / energy bar in
    ``gui/app.py`` rather than a grid marker here.
    """
    size = grid.shape[0]
    for x in range(size):
        for y in range(size):
            color = CELL_COLORS[int(grid[x, y])]
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="#cccccc", linewidth=0.5))

    if agent_pos is not None:
        ax.plot(agent_pos[0] + 0.5, agent_pos[1] + 0.5, marker="o", markersize=14,
                 markerfacecolor="#e63946", markeredgecolor="white", markeredgewidth=1.5, zorder=10)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def _heatmap_image(ax, V_slice: np.ndarray, wall_mask: np.ndarray, vmin=None, vmax=None):
    """Draw a value-heatmap slice onto ``ax`` and return the mappable.

    Shared by :func:`render_value_heatmap` and
    :func:`render_combined_panels` so both get an identical map. Uses
    the same coordinate convention as the policy axes: cell ``(0, 0)``
    sits at the top-left (data y=0 at the top). ``origin="upper"``
    already renders the array with the y-axis inverted, so no manual
    inversion is needed -- but callers must not re-set ``ylim``
    afterwards, which would silently un-invert it.
    """
    size = V_slice.shape[0]
    display = np.ma.array(V_slice.T, mask=wall_mask.T)
    return ax.imshow(display, cmap="viridis", origin="upper",
                     extent=[0, size, size, 0], vmin=vmin, vmax=vmax)


def render_value_heatmap(V_slice: np.ndarray, wall_mask: np.ndarray, title: str = "Value Function"):
    """Render a value-function heatmap for a fixed (k, energy) slice.

    Parameters
    ----------
    V_slice : ndarray of shape (X, Y)
        Value function values at a fixed ``(k, energy)``.
    wall_mask : ndarray of shape (X, Y), dtype=bool
        Wall positions, masked out (shown as black) in the heatmap.
    title : str, default="Value Function"
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the heatmap, ready to save via
        ``fig.savefig(...)``.

    Notes
    -----
    Satisfies the "Value heatmap" required visual output ("V or
    max-Q value for all valid states").
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = _heatmap_image(ax, V_slice, wall_mask)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="V(s)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig


def _draw_policy_arrows(ax, policy_slice: np.ndarray, wall_mask: np.ndarray,
                        terminal_mask: np.ndarray = None, key_mask: np.ndarray = None):
    """Draw one policy-arrows panel onto ``ax`` (shared by the single
    and the 2x2 combined renderers).

    Walls are filled black, the optional key cell is filled with the
    key color so it stands out from every other traversable cell, the
    goal is marked with a star, and every remaining cell gets the
    arrow for its greedy action.
    """
    size = policy_slice.shape[0]
    for x in range(size):
        for y in range(size):
            if wall_mask[x, y]:
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#2b2b2b"))
                continue
            if key_mask is not None and key_mask[x, y]:
                ax.add_patch(Rectangle((x, y), 1, 1,
                                       facecolor=CELL_COLORS[KEY],
                                       edgecolor="#cccccc", linewidth=0.5))
            if terminal_mask is not None and terminal_mask[x, y]:
                ax.plot(x + 0.5, y + 0.5, marker="*", markersize=16, color="#f2cc8f")
                continue
            dx, dy = ACTION_ARROWS[int(policy_slice[x, y])]
            ax.arrow(x + 0.5 - dx / 2, y + 0.5 - dy / 2, dx, dy,
                      head_width=0.15, head_length=0.12, fc="#3d405b", ec="#3d405b")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.invert_yaxis()


def render_policy_arrows(policy_slice: np.ndarray, wall_mask: np.ndarray, terminal_mask: np.ndarray = None,
                           title: str = "Policy", key_mask: np.ndarray = None):
    """Render policy arrows (best action per state) for a fixed (k, energy) slice.

    Parameters
    ----------
    policy_slice : ndarray of shape (X, Y)
        Greedy action index per position.
    wall_mask : ndarray of shape (X, Y), dtype=bool
        Wall positions, skipped (no arrow drawn).
    terminal_mask : ndarray of shape (X, Y), dtype=bool, optional
        Positions to mark with a distinct terminal-state glyph
        (typically the goal cell) instead of an arrow.
    key_mask : ndarray of shape (X, Y), dtype=bool, optional
        Positions to fill with the key color so the key cell is
        visually distinct from every other traversable cell.
    title : str, default="Policy"
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one arrow per non-wall, non-terminal cell.

    Notes
    -----
    Satisfies the "Final policy" required visual output ("Arrow for
    best action, plus markers for terminal states").
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_policy_arrows(ax, policy_slice, wall_mask, terminal_mask, key_mask)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def render_combined_panels(V: np.ndarray, policy: np.ndarray, wall_mask: np.ndarray,
                           terminal_mask: np.ndarray = None, key_mask: np.ndarray = None,
                           titles: tuple = None, suptitle: str = None):
    """Render a single 2x2 figure with value heatmaps and policy arrows for both key states.

    Top row: value heatmap for ``k=0`` (left) and ``k=1`` (right).
    Bottom row: policy arrows for ``k=0`` (left) and ``k=1`` (right).
    The two heatmaps share one colorbar and one value range so the key
    states are directly comparable.

    Parameters
    ----------
    V : ndarray of shape (X, Y, 2)
        Value per ``(x, y, k)`` position (already aggregated over the
        energy axis by the caller, e.g. via
        ``agents.policy_extraction.derive_v_and_policy_from_Q``).
    policy : ndarray of shape (X, Y, 2)
        Greedy action index per ``(x, y, k)`` position.
    wall_mask : ndarray of shape (X, Y), dtype=bool
        Wall positions, shown as black.
    terminal_mask : ndarray of shape (X, Y), dtype=bool, optional
        Goal positions, marked with a star instead of an arrow.
    key_mask : ndarray of shape (X, Y), dtype=bool, optional
        Key cell positions, filled with the key color.
    titles : tuple of str, optional
        The four axis titles (defaults to k=0/k=1 value and policy
        labels).
    suptitle : str, optional
        Optional figure-level title.

    Returns
    -------
    matplotlib.figure.Figure
        The 2x2 figure, ready to save via ``fig.savefig(...)``.
    """
    if titles is None:
        titles = (
            "Value, no key (k=0)",
            "Value, has key (k=1)",
            "Policy, no key (k=0)",
            "Policy, has key (k=1)",
        )
    valid = ~wall_mask
    finite = V[valid]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)

    im0 = _heatmap_image(axes[0, 0], V[:, :, 0], wall_mask, vmin, vmax)
    _heatmap_image(axes[0, 1], V[:, :, 1], wall_mask, vmin, vmax)
    fig.colorbar(im0, ax=[axes[0, 0], axes[0, 1]], label="V(s)")

    _draw_policy_arrows(axes[1, 0], policy[:, :, 0], wall_mask, terminal_mask, key_mask)
    _draw_policy_arrows(axes[1, 1], policy[:, :, 1], wall_mask, terminal_mask, key_mask)

    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    if suptitle:
        fig.suptitle(suptitle)

    return fig


def render_visitation_map(visitation_slice: np.ndarray, title: str = "State Visitation Counts"):
    """Render a heatmap of visitation counts for a fixed (k, energy) slice.

    Parameters
    ----------
    visitation_slice : ndarray of shape (X, Y)
        Visitation counts (e.g. summed over energy, or a single
        slice; caller decides which aggregation to pass in).
    title : str, default="State Visitation Counts"
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with a log-scaled visitation heatmap (log scale since
        visitation counts are typically highly skewed toward a few
        frequently-traversed corridor cells).

    Notes
    -----
    Satisfies the "Visitation map" required visual output ("Number of
    visits to each state during training").
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    display = np.log1p(visitation_slice.T)
    im = ax.imshow(display, cmap="magma", origin="upper")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log(1 + visits)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig


def render_visitation_panels(visitation_by_k: np.ndarray, wall_mask: np.ndarray,
                             k_labels=("k=0 (no key)", "k=1 (has key)"),
                             title="State Visitation Counts During Training"):
    """Render a 1x2 log-scaled visitation heatmap for both key states.

    Parameters
    ----------
    visitation_by_k : ndarray of shape (X, Y, 2)
        Visitation counts per position for each key state (the caller
        is expected to have summed over the energy axis already).
    wall_mask : ndarray of shape (X, Y), dtype=bool
        Wall positions, shown as black.
    k_labels : tuple of str, default=("k=0 (no key)", "k=1 (has key)")
        Subplot titles for the two key states.
    title : str, default="State Visitation Counts During Training"
        Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
        One figure with two log(1+visits) heatmaps, walls masked out.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    for ax, k, label in zip(axes, range(2), k_labels):
        display = np.ma.array(np.log1p(visitation_by_k[:, :, k]), mask=wall_mask)
        im = ax.imshow(display.T, cmap="magma", origin="upper")
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, label="log(1 + visits)")
    return fig


def render_policy_disagreement_map(agreement_mask_slice: np.ndarray, wall_mask: np.ndarray,
                                     title: str = "Policy Agreement with VI Reference"):
    """Render a color-coded map of where a learned policy agrees/disagrees with VI.

    Parameters
    ----------
    agreement_mask_slice : ndarray of shape (X, Y), dtype=bool
        ``True`` where the learned policy's greedy action matches the
        VI reference policy, at a fixed (k, energy) slice.
    wall_mask : ndarray of shape (X, Y), dtype=bool
        Wall positions, shown as black.
    title : str, default="Policy Agreement with VI Reference"
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Two-color (agree / disagree) map, walls masked out.

    Notes
    -----
    Satisfies the "Policy difference" required visual output ("States
    agreeing/disagreeing with the reference policy"), and the spec's
    requirement that comparison-agreement differences "be shown on a
    color-coded map."
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    display = np.ma.array(agreement_mask_slice.T.astype(float), mask=wall_mask.T)
    cmap = ListedColormap(["#e07a5f", "#81b29a"])  # red=disagree, green=agree
    im = ax.imshow(display, cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig


def render_policy_disagreement_panels(agreement_mask: np.ndarray, wall_mask: np.ndarray,
                                       k_labels=("k=0 (no key)", "k=1 (has key)"),
                                       title="Policy Agreement with VI Reference"):
    """Render a 1x2 color-coded map of policy agreement for both key states.

    Parameters
    ----------
    agreement_mask : ndarray of shape (X, Y, 2), dtype=bool
        ``True`` where the learned policy's greedy action matches the
        VI reference policy at each ``(x, y, k)`` position.
    wall_mask : ndarray of shape (X, Y), dtype=bool
        Wall positions, shown as black.
    k_labels : tuple of str, default=("k=0 (no key)", "k=1 (has key)")
        Subplot titles for the two key states.
    title : str, default="Policy Agreement with VI Reference"
        Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
        One figure with the two per-key-state agreement maps (green =
        agree, red = disagree), walls masked out. Each subplot title
        also reports the agreement fraction restricted to non-wall
        positions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    n_non_wall = int(np.sum(~wall_mask))
    for ax, k, label in zip(axes, range(2), k_labels):
        agree_fraction = float(np.sum(agreement_mask[:, :, k])) / max(1, n_non_wall)
        display = np.ma.array(agreement_mask[:, :, k].T.astype(float), mask=wall_mask.T)
        ax.imshow(display, cmap=ListedColormap(["#e07a5f", "#81b29a"]),
                  origin="upper", vmin=0, vmax=1)
        ax.set_title(f"{label}  --  agreement {agree_fraction:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def render_qvalue_diff_map(Q_before: np.ndarray, Q_after: np.ndarray, wall_mask: np.ndarray,
                             title: str = "Q-Value Change (post - pre transfer)"):
    """Render the max-|Q-value change| per position between two Q-tables.

    Parameters
    ----------
    Q_before : ndarray of shape (X, Y, 2, E, A)
        Q-table before transfer/continued training (e.g. transferred
        initialization).
    Q_after : ndarray of shape (X, Y, 2, E, A)
        Q-table after continued training on the target environment.
    wall_mask : ndarray of shape (X, Y), dtype=bool
        Wall positions, shown as black.
    title : str, default="Q-Value Change (post - pre transfer)"
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Heatmap of ``max_{k,e,a} |Q_after - Q_before|`` per ``(x, y)``.

    Notes
    -----
    Satisfies the "Transfer learning" required visual output
    ("Difference in Q-values or policy before and after transfer").
    """
    diff = np.max(np.abs(Q_after - Q_before), axis=(2, 3, 4))
    fig, ax = plt.subplots(figsize=(6, 6))
    display = np.ma.array(diff.T, mask=wall_mask.T)
    im = ax.imshow(display, cmap="coolwarm", origin="upper")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="max |delta Q|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig


def render_agent_path(grid: np.ndarray, path: list, title: str = "Agent Final Path"):
    """Render the maze with the agent's traversed path overlaid.

    Parameters
    ----------
    grid : ndarray of shape (size, size)
        Cell-type grid.
    path : list of tuple of int
        Sequence of ``(x, y)`` positions visited, in order.
    title : str, default="Agent Final Path"
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Maze rendering with a connected line + markers along ``path``.

    Notes
    -----
    Satisfies the "Agent's final path" required visual output.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    render_grid(ax, grid)
    if path:
        xs = [p[0] + 0.5 for p in path]
        ys = [p[1] + 0.5 for p in path]
        ax.plot(xs, ys, color="#e63946", linewidth=2, marker=".", markersize=6, zorder=5)
        ax.plot(xs[0], ys[0], marker="o", markersize=12, color="#81b29a", zorder=6)
        ax.plot(xs[-1], ys[-1], marker="X", markersize=12, color="#3d405b", zorder=6)
    ax.set_title(title)
    fig.tight_layout()
    return fig
