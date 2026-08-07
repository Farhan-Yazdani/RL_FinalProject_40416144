"""Derive a well-defined, VI-comparable ``policy.npy`` from a learned
Q-table for Q-Learning / SARSA(lambda) runs.

Why this is needed
-------------------
``Q.npy`` has shape ``(X, Y, 2, max_energy+1, A)``. A naive
``np.argmax(Q, axis=-1)`` is *not* a well-defined policy the way it is
for Value Iteration's ``Q``, because VI's ``Q`` is computed from the
full transition model for *every* state, while Q-Learning/SARSA(lambda)
only ever update ``Q[x, y, k, energy, :]`` for states the agent
actually visited during training. Because ``energy`` decreases
monotonically within an episode and always resets to ``max_energy`` on
``reset()``, only a thin, position-correlated slice of the
``(x, y, k, energy)`` space is ever visited (see
``experiments.analysis.reachable_states_mask``'s docstring for the
same observation). For every unvisited ``(x, y, k, energy)`` combo,
``Q[x, y, k, energy, :]`` is exactly ``[0, 0, 0, 0]``, and
``argmax`` of an all-zero row always returns action ``0`` (``UP``) --
a meaningless artifact of ``argmax``'s tie-breaking, not a learned
decision. Rendering/comparing that directly (as
``gui/render_outputs.py`` and ``experiments/analysis.py`` do today via
a single fixed ``--energy`` slice) silently produces a mostly-fake
"all UP" policy outside the thin trained band.

Fix implemented here
---------------------
For each ``(x, y, k)`` position, borrow the greedy action from
whichever *trained* (non-all-zero) energy row is available, preferring
the energy level the agent actually experienced most at that position
(from the run's ``events.log`` visitation counts). If a position has
no visitation record at all (never reached during training), fall
back to the trained row nearest ``max_energy // 2`` ("half energy"),
which is a much more representative default than any arbitrary fixed
slice, since a policy's action at a position rarely depends on the
exact remaining energy except very close to depletion. The resulting
per-position action is then broadcast across the full energy axis, so
the output keeps the same ``(X, Y, 2, E+1)`` shape ``value_iteration``
saves (`render_outputs.py --k --energy`, the GUI's slice selectors,
and ``experiments/analysis.py``'s agreement computations all keep
working unchanged) while every entry reflects a real, trained decision
rather than an argmax-of-zeros artifact.

This module is intentionally standalone (imported by both
``agents/q_learning.py`` and ``agents/sarsa_lambda.py``) so the same
derivation logic is never duplicated (``CODING_STYLE.md`` 2.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _visitation_counts_from_events_log(events_log_path, shape) -> Optional[np.ndarray]:
    """Build a ``(X, Y, 2, E+1)`` visitation-count array from ``events.log``.

    Parameters
    ----------
    events_log_path : str or pathlib.Path
        Path to the run's ``events.log`` (one JSON object per line,
        with a ``"state": [x, y, k, energy]`` field), as written by
        ``agents/q_learning.py`` / ``agents/sarsa_lambda.py``.
    shape : tuple of int
        ``(X, Y, 2, E+1)`` shape to build the counts array in (must
        match the run's ``Q`` array's first four dimensions).

    Returns
    -------
    ndarray of shape ``shape``, dtype=int64, or None
        Visitation counts per state, or ``None`` if the events log
        does not exist (e.g. called on a run predating this feature,
        or from a context without the raw events log on disk).
    """
    import json

    path = Path(events_log_path)
    if not path.exists():
        return None
    counts = np.zeros(shape, dtype=np.int64)
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            x, y, k, e = row["state"]
            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= e < shape[3]:
                counts[x, y, k, e] += 1
    return counts


def derive_policy_from_Q(
    Q: np.ndarray,
    events_log_path=None,
    default_energy: Optional[int] = None,
) -> np.ndarray:
    """Derive a well-defined ``(X, Y, 2, E+1)`` greedy policy from a learned Q-table.

    Parameters
    ----------
    Q : ndarray of shape (X, Y, 2, E+1, A)
        Learned action-value table from Q-Learning or SARSA(lambda).
    events_log_path : str or pathlib.Path, optional
        Path to the run's ``events.log``. When given, the trained
        energy row nearest to each position's *most-visited* energy
        level is used. When omitted (or the file doesn't exist), every
        position falls back to the trained row nearest
        ``default_energy``.
    default_energy : int, optional
        Energy level to prefer when no visitation information is
        available for a position (or no events log is given at all).
        Defaults to ``max_energy // 2`` (half energy), per the
        module's stated fallback.

    Returns
    -------
    ndarray of shape (X, Y, 2, E+1), dtype=int64
        A greedy policy: for every ``(x, y, k)``, one action index is
        chosen from a *trained* (non-all-zero) ``Q[x, y, k, e, :]`` row
        -- never an untrained all-zero row -- and broadcast across the
        full energy axis, so this array can be sliced by
        ``[:, :, k, energy]`` exactly like Value Iteration's
        ``policy.npy`` anywhere downstream (``gui/render_outputs.py``,
        ``gui/backend_adapter.py``, ``experiments/analysis.py``).

    Notes
    -----
    A position with *no* trained row at any energy level (i.e. the
    agent's trajectory never reached that ``(x, y)`` for that
    key-state at all -- e.g. an unreachable wall-locked pocket, or a
    key-state combination never explored) keeps action ``0`` for every
    energy level at that position, identical to plain ``argmax``. This
    is unavoidable without a model (there is genuinely no learned
    information to report there); the fix here only concerns positions
    that *were* trained, just not at every energy value, which is the
    overwhelming majority of the "wrong" entries a naive fixed-slice
    ``argmax`` would produce.
    """
    n_x, n_y, n_k, n_e, n_a = Q.shape
    if default_energy is None:
        default_energy = n_e // 2
    default_energy = int(np.clip(default_energy, 0, n_e - 1))

    # A row is "trained" iff it isn't exactly all-zero (untouched
    # initialization). This is a cheap, vectorized proxy for "was this
    # (x, y, k, energy) ever updated" that needs no extra bookkeeping
    # beyond the Q-table itself.
    trained_mask = np.any(Q != 0.0, axis=-1)  # shape (X, Y, 2, E)

    visitation = None
    if events_log_path is not None:
        visitation = _visitation_counts_from_events_log(events_log_path, (n_x, n_y, n_k, n_e))

    # Preferred energy level per (x, y, k): the most-visited energy if
    # we have visitation data and the position was visited at all,
    # else the global default_energy fallback.
    if visitation is not None and visitation.sum() > 0:
        preferred_energy = np.argmax(visitation, axis=-1)  # shape (X, Y, 2)
        ever_visited = np.any(visitation > 0, axis=-1)      # shape (X, Y, 2)
        preferred_energy = np.where(ever_visited, preferred_energy, default_energy)
    else:
        preferred_energy = np.full((n_x, n_y, n_k), default_energy, dtype=np.int64)

    policy = np.zeros((n_x, n_y, n_k, n_e), dtype=np.int64)

    # For each (x, y, k), search outward from the preferred energy for
    # the nearest trained row; broadcast that row's greedy action
    # across the whole energy axis for this position. A Python loop
    # over (x, y, k) is used here (not a hot training loop -- this
    # runs once, after training, per CODING_STYLE.md 2.1's carve-out
    # for non-inherently-sequential-but-cheap post-processing); the
    # per-position search itself is O(E) worst case and this whole
    # function is O(X*Y*2*E).
    for x in range(n_x):
        for y in range(n_y):
            for k in range(n_k):
                trained_here = trained_mask[x, y, k]  # shape (E,)
                if not trained_here.any():
                    # No trained row anywhere for this position/key
                    # state: nothing learned to report (see Notes).
                    continue
                pref_e = int(preferred_energy[x, y, k])
                trained_energies = np.flatnonzero(trained_here)
                nearest_e = trained_energies[
                    np.argmin(np.abs(trained_energies - pref_e))
                ]
                action = int(np.argmax(Q[x, y, k, nearest_e]))
                policy[x, y, k, :] = action

    return policy


def save_derived_policy(
    Q: np.ndarray,
    models_dir,
    events_log_path=None,
    default_energy: Optional[int] = None,
) -> np.ndarray:
    """Derive and save ``policy.npy`` next to an already-saved ``Q.npy``.

    Parameters
    ----------
    Q : ndarray of shape (X, Y, 2, E+1, A)
        Learned Q-table (already saved by the caller as ``Q.npy``).
    models_dir : str or pathlib.Path
        The run's ``results/models/<algorithm>/<run_id>/`` directory
        (same directory ``Q.npy`` was written to).
    events_log_path : str or pathlib.Path, optional
        Path to the run's ``events.log``, for visitation-aware energy
        selection (see :func:`derive_policy_from_Q`).
    default_energy : int, optional
        Fallback energy level; defaults to half of ``max_energy``.

    Returns
    -------
    ndarray of shape (X, Y, 2, E+1)
        The derived policy (also written to ``<models_dir>/policy.npy``).
    """
    policy = derive_policy_from_Q(Q, events_log_path=events_log_path, default_energy=default_energy)
    np.save(Path(models_dir) / "policy.npy", policy)
    return policy
