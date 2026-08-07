"""Seed/size derivation and deterministic maze map generation.

Implements spec section "Dedicated Seed and Map Generation" and the
structural constraints from "Problem Definition and Maze Environment"
in ``final_project.md``. Also implements the BFS-based map validator
required there and in the Transfer Learning section (both target maps
must be BFS-validated the same way as the source map).

Cell type encoding (stored as small ints in the grid array)
-------------------------------------------------------------
0 : normal
1 : wall
2 : penalty
3 : start
4 : key
5 : door (closed door; treated as impassable until key is held)
6 : goal

Notes
-----
This module is intentionally free of any RL-algorithm logic. It only
produces and validates static map data (grids + special-cell
coordinates) that ``environments.maze`` consumes to build the dynamic
environment (state space, transition function, reward function).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Cell type constants (see module docstring for encoding).
NORMAL = 0
WALL = 1
PENALTY = 2
START = 3
KEY = 4
DOOR = 5
GOAL = 6

CELL_NAMES = {
    NORMAL: "normal",
    WALL: "wall",
    PENALTY: "penalty",
    START: "start",
    KEY: "key",
    DOOR: "door",
    GOAL: "goal",
}

MIN_OBSTACLE_FRACTION = 0.15
MIN_PENALTY_CELLS = 5

# Number of door cells guarding the goal. Any value in {1, 2, 3, 4} is
# valid (a cell has at most 4 neighbors); doors are interchangeable --
# holding the key, the agent may enter the goal through *any* of them,
# not a specific one, so this is a redundant/wider chokepoint rather
# than a multi-stage lock. See _place_special_cells / _seal_goal_behind_doors.
DEFAULT_N_DOORS = 2


@dataclass(frozen=True)
class MapSpec:
    """Immutable, serializable description of a validated maze map.

    Parameters
    ----------
    name : str
        Identifier for this map (used as the filename stem under
        ``environments/maps/``).
    student_id : str
        Raw student ID string the seed/size were derived from.
    base_seed : int
        Second-to-last digit of ``student_id`` (see
        :func:`derive_seed_and_size`).
    maze_size : int
        Grid is ``maze_size x maze_size``.
    grid : ndarray of shape (maze_size, maze_size), dtype=int8
        Cell type for every position (see module-level constants).
    start : tuple of int
        ``(x, y)`` coordinate of the start cell.
    key_pos : tuple of int
        ``(x, y)`` coordinate of the key cell.
    door_positions : tuple of tuple of int
        ``(x, y)`` coordinates of every closed-door cell guarding the
        goal (see :func:`_place_special_cells`). All doors are
        interchangeable: once the agent holds the key, entering the
        goal through *any* of them succeeds -- this is a redundant
        chokepoint (a wider gate), not a multi-stage lock requiring a
        specific sequence.
    goal : tuple of int
        ``(x, y)`` coordinate of the goal cell.
    generation_attempt : int
        Number of deterministic regeneration attempts needed before a
        valid map was produced (0 means the first draw was valid).
    """

    name: str
    student_id: str
    base_seed: int
    maze_size: int
    grid: np.ndarray
    start: tuple
    key_pos: tuple
    door_positions: tuple
    goal: tuple
    generation_attempt: int

    @property
    def door_pos(self) -> tuple:
        """Single "primary" door coordinate, for callers that only
        need one representative door cell (e.g. a legacy single-door
        assumption). Returns ``door_positions[0]``. Prefer
        ``door_positions`` directly for anything that should account
        for *all* doors (rendering, reachability, sealing checks).
        """
        return self.door_positions[0]

    def to_serializable(self) -> dict:
        """Convert this ``MapSpec`` to a JSON-serializable dict.

        Returns
        -------
        dict
            All fields with ``grid`` converted to a nested list and
            coordinate tuples converted to lists.
        """
        d = asdict(self)
        d["grid"] = self.grid.astype(int).tolist()
        d["start"] = list(self.start)
        d["key_pos"] = list(self.key_pos)
        d["door_positions"] = [list(p) for p in self.door_positions]
        d["goal"] = list(self.goal)
        return d

    @staticmethod
    def from_serializable(d: dict) -> "MapSpec":
        """Reconstruct a ``MapSpec`` from :meth:`to_serializable` output.

        Parameters
        ----------
        d : dict
            Dict as produced by :meth:`to_serializable`.

        Returns
        -------
        MapSpec
            Reconstructed map specification with ``grid`` as an
            ``ndarray`` and coordinates as tuples.
        """
        return MapSpec(
            name=d["name"],
            student_id=d["student_id"],
            base_seed=d["base_seed"],
            maze_size=d["maze_size"],
            grid=np.array(d["grid"], dtype=np.int8),
            start=tuple(d["start"]),
            key_pos=tuple(d["key_pos"]),
            door_positions=tuple(tuple(p) for p in d["door_positions"]),
            goal=tuple(d["goal"]),
            generation_attempt=d["generation_attempt"],
        )


def derive_seed_and_size(student_id: str) -> tuple:
    """Derive the base seed and maze size from a student ID.

    Implements the formula from the "Dedicated Seed and Map
    Generation" section of ``final_project.md``:
    ``b = int(StudentID[-2])``, ``N = 15 + (b mod 4)``.

    Parameters
    ----------
    student_id : str
        Student ID string. Must have at least 2 characters.

    Returns
    -------
    base_seed : int
        The second-to-last digit of ``student_id``.
    maze_size : int
        ``15 + (base_seed % 4)``.

    Notes
    -----
    This is the single canonical place this formula may be evaluated;
    every other module must import and call this function rather than
    recomputing or hardcoding the result (see ``CODING_STYLE.md`` 1.1,
    1.3). For this project's ``student_id`` this evaluates to
    ``base_seed = 4``, ``maze_size = 15``.
    """
    if len(student_id) < 2:
        raise ValueError(
            f"student_id must have at least 2 characters, got {student_id!r}"
        )
    base_seed = int(student_id[-2])
    maze_size = 15 + (base_seed % 4)
    return base_seed, maze_size


def _neighbors4(pos: tuple, size: int):
    """Yield in-bounds 4-connected neighbors of ``pos`` on a ``size`` grid."""
    x, y = pos
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            yield (nx, ny)


def bfs_reachable(
    grid: np.ndarray,
    start: tuple,
    goal: tuple,
    door_passable: bool = False,
) -> bool:
    """Check reachability from ``start`` to ``goal`` treating walls
    (and, unless ``door_passable``, the closed door) as impassable.

    Parameters
    ----------
    grid : ndarray of shape (size, size)
        Cell-type grid (see module-level constants).
    start : tuple of int
        ``(x, y)`` starting coordinate.
    goal : tuple of int
        ``(x, y)`` target coordinate.
    door_passable : bool, default=False
        If ``False`` (the default), ``DOOR`` cells are treated as
        impassable, matching the real dynamics of an agent that does
        not yet hold the key (used for the start->key check). If
        ``True``, ``DOOR`` cells are treated like any other open
        cell, matching the real dynamics of an agent that already
        holds the key (used for the key->goal check, since the goal
        is now sealed behind the door and would otherwise be
        unreachable by construction -- see
        :func:`is_goal_sealed_behind_door`).

    Returns
    -------
    bool
        ``True`` iff a 4-connected path free of ``WALL`` cells (and
        free of ``DOOR`` cells when ``door_passable`` is ``False``)
        exists from ``start`` to ``goal``.

    Notes
    -----
    This is a plain deterministic BFS used purely for structural map
    validation, per the spec's explicit note that "BFS is only for
    map validation and is not a substitute for the learning agent."
    ``door_passable`` lets the *same* BFS routine be reused for both
    halves of the validation despite them needing opposite door
    semantics, rather than hardcoding the door as always-blocking
    (which would make the key->goal check impossible to satisfy once
    the goal is deliberately sealed behind the door -- see
    :func:`_seal_goal_behind_door`).
    """
    size = grid.shape[0]
    visited = np.zeros((size, size), dtype=bool)
    visited[start] = True
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            return True
        for nxt in _neighbors4(cur, size):
            if visited[nxt]:
                continue
            cell = grid[nxt]
            if cell == WALL:
                continue
            if cell == DOOR and not door_passable:
                continue
            visited[nxt] = True
            queue.append(nxt)
    return bool(visited[goal])


def is_goal_sealed_behind_doors(
    grid: np.ndarray, goal: tuple, door_positions: tuple
) -> bool:
    """Check that the doors are a mandatory chokepoint guarding the goal.

    Parameters
    ----------
    grid : ndarray of shape (size, size)
        Cell-type grid.
    goal : tuple of int
        Goal coordinate.
    door_positions : tuple of tuple of int
        Coordinates of every door guarding the goal.

    Returns
    -------
    bool
        ``True`` iff every in-bounds 4-connected neighbor of ``goal``
        is either one of ``door_positions`` or a ``WALL`` cell -- i.e.
        the *only* way to step onto the goal is through one of the
        doors (doors are interchangeable, so any one of them
        suffices). This is a pure structural check (no BFS/traversal),
        so it is cheap to call repeatedly and works regardless of grid
        size, number of doors, or where the goal sits (corner, edge,
        or interior all fall out of :func:`_neighbors4` returning
        fewer/more in-bounds neighbors -- e.g. a corner goal with
        ``n_doors=2`` may have *zero* sealing walls if both of its two
        neighbors are doors, which is still correctly "sealed" since
        every entrance is a door).

    Notes
    -----
    This is what turns the doors from decorative into load-bearing:
    without this check, ``final_project.md``'s door mechanic could be
    satisfied by a map where the goal has other, wall-free approaches
    that never touch any door. Used both by :func:`validate_map` (so
    *no* map -- source or transfer target -- can ever be accepted
    without this property) and by :func:`_seal_goal_behind_doors` /
    :func:`generate_transfer_target` to confirm the invariant survived
    a mutation.
    """
    size = grid.shape[0]
    door_set = set(door_positions)
    for nb in _neighbors4(goal, size):
        if nb in door_set:
            continue
        if grid[nb] != WALL:
            return False
    return True


def validate_map(
    grid: np.ndarray,
    start: tuple,
    key_pos: tuple,
    goal: tuple,
    door_positions: tuple,
) -> bool:
    """Validate structural constraints and reachability of a candidate map.

    Parameters
    ----------
    grid : ndarray of shape (size, size)
        Candidate cell-type grid.
    start : tuple of int
        Start coordinate.
    key_pos : tuple of int
        Key coordinate.
    goal : tuple of int
        Goal coordinate.
    door_positions : tuple of tuple of int
        Coordinates of every door guarding the goal. The goal must be
        reachable *only* through one of these cells (see
        :func:`is_goal_sealed_behind_doors`) -- passed explicitly
        (rather than re-derived) so the check is anchored to whatever
        door positions the caller actually placed.

    Returns
    -------
    bool
        ``True`` iff:
        (1) at least ``MIN_OBSTACLE_FRACTION`` of cells are walls,
        (2) at least ``MIN_PENALTY_CELLS`` cells are penalty cells,
        (3) the goal is sealed behind the doors (every non-wall
            neighbor of goal is one of ``door_positions``),
        (4) a path free of walls *and every closed door* exists
            start -> key_pos (the agent cannot hold the key yet, so
            it cannot use any door as a shortcut), and
        (5) a path free of walls, with doors passable, exists
            key_pos -> goal (the agent holds the key by this point,
            so at least one legitimate route through a door must
            exist).

    Notes
    -----
    Used both at generation time (to accept/reject/regenerate a
    candidate map) and importable directly by ``tests/`` per
    ``CODING_STYLE.md`` 2.5. This function is the single authority for
    "is this map acceptable" -- both fresh generation
    (:func:`generate_map`) and transfer-target mutation
    (:func:`generate_transfer_target`) must pass through it, so the
    door-as-chokepoint guarantee can never be silently lost by a code
    path that forgets to re-check it after moving walls around.
    """
    size = grid.shape[0]
    total_cells = size * size

    n_walls = int(np.sum(grid == WALL))
    if n_walls < MIN_OBSTACLE_FRACTION * total_cells:
        return False

    n_penalty = int(np.sum(grid == PENALTY))
    if n_penalty < MIN_PENALTY_CELLS:
        return False

    if not is_goal_sealed_behind_doors(grid, goal, door_positions):
        return False

    if not bfs_reachable(grid, start, key_pos, door_passable=False):
        return False
    if not bfs_reachable(grid, key_pos, goal, door_passable=True):
        return False

    return True


def _place_special_cells(
    grid: np.ndarray, rng: np.random.Generator, n_doors: int = DEFAULT_N_DOORS
) -> dict:
    """Pick and stamp start/key/doors/goal onto ``grid`` (mutates and returns coords).

    Parameters
    ----------
    grid : ndarray of shape (size, size)
        Grid to mutate in place; must already have walls placed as
        ``WALL`` and everything else as ``NORMAL``.
    rng : numpy.random.Generator
        Seeded generator used for all random placement choices.
    n_doors : int, default=DEFAULT_N_DOORS
        Number of door cells to place around the goal. Must be
        between 1 and 4 inclusive (a cell has at most 4 neighbors).

    Returns
    -------
    dict or None
        ``None`` if fewer than ``n_doors`` free neighbors of the goal
        are available for this candidate -- the caller must discard
        the candidate and retry with the next deterministic seed
        rather than silently placing fewer doors than requested
        (which would make the door count inconsistent across runs) or
        falling back to some unrelated cell (which used to produce a
        fake "door" that wasn't even adjacent to the goal).
        Otherwise, a dict with keys ``"start"``, ``"key_pos"``,
        ``"door_positions"`` (a tuple of ``n_doors`` coordinates),
        ``"goal"``. Every door is a genuine free neighbor of the goal
        (so "passing through a door" is a real, enforced final gate --
        see :func:`_seal_goal_behind_doors`), and the key is placed in
        the opposite half of the grid from the start to force
        traversal.

    Raises
    ------
    ValueError
        If ``n_doors`` is not in ``{1, 2, 3, 4}``.
    """
    if not 1 <= n_doors <= 4:
        raise ValueError(f"n_doors must be between 1 and 4, got {n_doors}")

    size = grid.shape[0]
    free_mask = grid == NORMAL
    free_cells = list(zip(*np.where(free_mask)))
    free_cells = [(int(x), int(y)) for x, y in free_cells]
    if len(free_cells) < 4:
        raise RuntimeError("Not enough free cells to place special cells.")

    rng.shuffle(free_cells)

    # Start in the "first half" (by Manhattan distance from origin) to
    # bias key/goal toward being spatially separated from it.
    free_cells.sort(key=lambda c: c[0] + c[1])
    start = free_cells[0]

    # Goal: pick the farthest-from-start cell that structurally has
    # *at least* n_doors neighbors (a corner has 2, an edge has 3, an
    # interior cell has 4 -- see _neighbors4). Without this filter the
    # plain "farthest cell" heuristic below almost always lands on a
    # grid corner (2 neighbors), which makes n_doors > 2 impossible
    # for essentially every seed -- wasting the entire regeneration
    # budget rather than just picking a slightly less extreme goal
    # cell that can actually support the requested door count.
    goal_candidates = [
        c
        for c in reversed(free_cells)
        if c != start and sum(1 for _ in _neighbors4(c, size)) >= n_doors
    ]
    if not goal_candidates:
        return None
    goal = goal_candidates[0]

    # Doors: must be genuine free neighbors of goal, since they are
    # about to become the *only* way onto the goal (see
    # _seal_goal_behind_doors). Deterministic order comes from
    # _neighbors4 plus the already-shuffled free_cells, so this is
    # reproducible given the seed. If goal doesn't have at least
    # n_doors free neighbors (e.g. it landed against a cluster of
    # walls, in a corner with n_doors > 2, or on an edge with
    # n_doors > 3), there is no way to place exactly n_doors real
    # chokepoint doors for this candidate -- signal failure so the
    # deterministic regeneration loop in generate_map/
    # generate_transfer_target draws a new candidate instead.
    door_positions = tuple(
        nb for nb in _neighbors4(goal, size) if grid[nb] == NORMAL and nb != start
    )[:n_doors]
    if len(door_positions) < n_doors:
        return None
    door_set = set(door_positions)

    remaining = [
        c for c in free_cells if c != start and c != goal and c not in door_set
    ]
    # Key roughly in the middle third of the sorted-by-distance list,
    # so it typically requires real traversal from start but isn't
    # placed pathologically next to the goal.
    mid = len(remaining) // 2
    key_pos = remaining[mid] if remaining else remaining[0]

    grid[start] = START
    grid[key_pos] = KEY
    for d in door_positions:
        grid[d] = DOOR
    grid[goal] = GOAL

    return {
        "start": start,
        "key_pos": key_pos,
        "door_positions": door_positions,
        "goal": goal,
    }


def _seal_goal_behind_doors(
    grid: np.ndarray,
    goal: tuple,
    door_positions: tuple,
    start: tuple,
    key_pos: tuple,
) -> bool:
    """Wall off every neighbor of ``goal`` except ``door_positions``, in place.

    Parameters
    ----------
    grid : ndarray of shape (size, size)
        Grid to mutate in place. Assumed to already have ``start``,
        ``key_pos``, ``door_positions``, and ``goal`` stamped in.
    goal : tuple of int
        Goal coordinate.
    door_positions : tuple of tuple of int
        Door coordinates; each must already be a 4-connected neighbor
        of ``goal`` (guaranteed by :func:`_place_special_cells`).
    start : tuple of int
        Start coordinate; protected from being overwritten.
    key_pos : tuple of int
        Key coordinate; protected from being overwritten.

    Returns
    -------
    bool
        ``True`` if sealing succeeded. ``False`` if ``start`` or
        ``key_pos`` happens to be a neighbor of ``goal`` that isn't
        one of ``door_positions`` -- in that case the goal cannot be
        sealed without destroying a protected cell, so the caller must
        discard this candidate and regenerate rather than silently
        overwrite ``start``/``key_pos`` or silently leave a gap in the
        seal.

    Notes
    -----
    This function is what makes the doors a *mandatory* chokepoint
    rather than "just normal cells among the goal's neighbors." It
    works uniformly regardless of maze size, number of doors, or
    where the goal landed (corner/edge/interior all fall out of
    :func:`_neighbors4` naturally returning 2/3/4 neighbors -- e.g. a
    corner goal with ``n_doors=2`` has nothing left to wall, which is
    correct: every entrance is already a door). It is idempotent --
    calling it again on an already-sealed goal is a no-op other than
    re-asserting existing walls, which
    :func:`generate_transfer_target` relies on as a post-mutation
    safety net.
    """
    size = grid.shape[0]
    door_set = set(door_positions)
    to_wall = []
    for nb in _neighbors4(goal, size):
        if nb in door_set:
            continue
        if nb == start or nb == key_pos:
            return False
        to_wall.append(nb)
    for nb in to_wall:
        grid[nb] = WALL
    return True


def _generate_candidate(
    size: int, rng: np.random.Generator, n_doors: int = DEFAULT_N_DOORS
) -> tuple:
    """Draw one random candidate grid + special cells (not yet validated).

    Parameters
    ----------
    size : int
        Grid side length.
    rng : numpy.random.Generator
        Seeded generator for this attempt.
    n_doors : int, default=DEFAULT_N_DOORS
        Number of door cells to place around the goal.

    Returns
    -------
    grid : ndarray of shape (size, size), or None
        Candidate grid with walls, penalties, and special cells
        stamped in, with the goal sealed behind the doors. ``None`` if
        this candidate could not produce ``n_doors`` viable door
        placements or could not be sealed without overwriting
        ``start``/``key_pos`` (see :func:`_place_special_cells` and
        :func:`_seal_goal_behind_doors`) -- the caller (``generate_map``
        / ``generate_transfer_target``) must treat this the same as a
        failed :func:`validate_map` call and move on to the next
        deterministic attempt.
    coords : dict or None
        As returned by :func:`_place_special_cells`, or ``None`` iff
        ``grid`` is ``None``.
    """
    total_cells = size * size
    grid = np.full((size, size), NORMAL, dtype=np.int8)

    # Slightly above the 15% floor so validation has margin after
    # special-cell placement can't remove walls but rounding could
    # otherwise put us right at the boundary.
    n_walls = int(np.ceil(MIN_OBSTACLE_FRACTION * total_cells)) + rng.integers(0, 3)
    wall_flat_idx = rng.choice(total_cells, size=n_walls, replace=False)
    wall_coords = [(int(i % size), int(i // size)) for i in wall_flat_idx]
    for c in wall_coords:
        grid[c] = WALL

    free_mask = grid == NORMAL
    free_flat_idx = np.where(free_mask.flatten())[0]
    n_penalty = max(MIN_PENALTY_CELLS, MIN_PENALTY_CELLS + int(rng.integers(0, 3)))
    n_penalty = min(n_penalty, len(free_flat_idx))
    penalty_flat_idx = rng.choice(free_flat_idx, size=n_penalty, replace=False)
    for i in penalty_flat_idx:
        x, y = int(i % size), int(i // size)
        grid[x, y] = PENALTY

    coords = _place_special_cells(grid, rng, n_doors=n_doors)
    if coords is None:
        return None, None

    sealed = _seal_goal_behind_doors(
        grid,
        coords["goal"],
        coords["door_positions"],
        coords["start"],
        coords["key_pos"],
    )
    if not sealed:
        return None, None

    return grid, coords


def generate_map(
    student_id: str,
    name: str = "source",
    max_attempts: int = 200,
    n_doors: int = DEFAULT_N_DOORS,
) -> MapSpec:
    """Deterministically generate a BFS-validated map for ``student_id``.

    Parameters
    ----------
    student_id : str
        Student ID used to derive ``base_seed``/``maze_size`` via
        :func:`derive_seed_and_size`.
    name : str, default="source"
        Map name, used as the on-disk filename stem and stored in the
        returned :class:`MapSpec`.
    max_attempts : int, default=200
        Maximum deterministic regeneration attempts before raising.
    n_doors : int, default=DEFAULT_N_DOORS
        Number of interchangeable door cells to place around the
        goal. Must be between 1 and 4. Larger values are stricter
        (more free neighbors of the goal are required), so they may
        need more regeneration attempts -- see :func:`_place_special_cells`.

    Returns
    -------
    MapSpec
        A validated map specification.

    Raises
    ------
    RuntimeError
        If no valid map is found within ``max_attempts`` tries.

    Notes
    -----
    Regeneration on validation failure is deterministic: attempt
    ``k`` seeds its RNG with ``base_seed * 10_000 + k`` (via
    ``np.random.default_rng``), per ``CODING_STYLE.md`` 1.3 ("increment
    a generation counter fed into the same seeded RNG -- never by
    manual/interactive fixing").
    """
    base_seed, maze_size = derive_seed_and_size(student_id)

    for attempt in range(max_attempts):
        rng = np.random.default_rng(base_seed * 10_000 + attempt)
        grid, coords = _generate_candidate(maze_size, rng, n_doors=n_doors)
        if grid is None:
            # This attempt couldn't place n_doors real door-neighbors
            # for the goal, or couldn't seal it without overwriting
            # start/key -- discard and try the next deterministic seed.
            continue
        if validate_map(
            grid,
            coords["start"],
            coords["key_pos"],
            coords["goal"],
            coords["door_positions"],
        ):
            return MapSpec(
                name=name,
                student_id=student_id,
                base_seed=base_seed,
                maze_size=maze_size,
                grid=grid,
                start=coords["start"],
                key_pos=coords["key_pos"],
                door_positions=coords["door_positions"],
                goal=coords["goal"],
                generation_attempt=attempt,
            )

    raise RuntimeError(
        f"Failed to generate a valid map for student_id={student_id!r} "
        f"within {max_attempts} deterministic attempts."
    )


def save_map(map_spec: MapSpec, maps_dir: Path) -> Path:
    """Serialize ``map_spec`` to ``<maps_dir>/<name>.json``.

    Parameters
    ----------
    map_spec : MapSpec
        Map to serialize.
    maps_dir : pathlib.Path
        Destination directory (created if missing).

    Returns
    -------
    pathlib.Path
        Path to the written JSON file.
    """
    maps_dir = Path(maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)
    out_path = maps_dir / f"{map_spec.name}.json"
    with open(out_path, "w") as f:
        json.dump(map_spec.to_serializable(), f, indent=2)
    return out_path


def load_map(maps_dir: Path, name: str) -> MapSpec:
    """Load a previously saved map by name.

    Parameters
    ----------
    maps_dir : pathlib.Path
        Directory containing ``<name>.json``.
    name : str
        Map name (filename stem).

    Returns
    -------
    MapSpec
        Reconstructed map specification.
    """
    path = Path(maps_dir) / f"{name}.json"
    with open(path) as f:
        d = json.load(f)
    return MapSpec.from_serializable(d)


def generate_transfer_target(
    source: MapSpec,
    name: str,
    change_fraction: float,
    move_key_or_goal: bool,
    n_new_penalties: int,
    max_attempts: int = 200,
) -> MapSpec:
    """Generate a BFS-validated transfer-target map derived from ``source``.

    Implements the "Target Environments" subsection of the Transfer
    Learning section: a *similar* target moves ~15-20% of obstacles
    with start/key/goal fixed, and a *different* target changes
    >=35% of obstacles, moves the key or goal, and adds new penalty
    cells.

    Parameters
    ----------
    source : MapSpec
        The already-validated source map to derive from.
    name : str
        Name for the new map (e.g. ``"transfer_similar"`` or
        ``"transfer_different"``).
    change_fraction : float
        Fraction of current wall cells to relocate to new random free
        positions (e.g. ``0.175`` for the similar target, ``0.4`` for
        the different target).
    move_key_or_goal : bool
        If ``True``, relocate the key (and, deterministically based
        on the RNG draw, possibly the goal) to a new free cell -- used
        for the "different" target only.
    n_new_penalties : int
        Number of additional penalty cells to stamp onto currently
        normal cells.
    max_attempts : int, default=200
        Maximum deterministic regeneration attempts before raising.

    Returns
    -------
    MapSpec
        A validated transfer-target map, with the same ``student_id``/
        ``base_seed``/``maze_size`` as ``source`` (only layout differs).

    Raises
    ------
    RuntimeError
        If no valid variant is found within ``max_attempts`` tries.

    Notes
    -----
    Regeneration on validation failure follows the same deterministic
    pattern as :func:`generate_map`: attempt ``k`` uses
    ``np.random.default_rng(source.base_seed * 10_000 + 5000 + k)``, a
    seed offset disjoint from the source map's own attempt range so
    the two never coincide.
    """
    size = source.maze_size

    # The walls immediately guarding the goal (every neighbor of goal
    # that isn't one of source.door_positions) are what make the
    # doors a mandatory chokepoint (see _seal_goal_behind_doors). They
    # must never be candidates for relocation here, or a
    # "similar"/"different" target could silently reopen an alternate
    # route onto the goal that bypasses every door entirely.
    goal_seal_neighbors = {
        nb
        for nb in _neighbors4(source.goal, size)
        if nb not in source.door_positions
    }

    for attempt in range(max_attempts):
        rng = np.random.default_rng(source.base_seed * 10_000 + 5000 + attempt)
        grid = source.grid.copy()

        wall_coords = list(zip(*np.where(grid == WALL)))
        wall_coords = [
            (int(x), int(y))
            for x, y in wall_coords
            if (int(x), int(y)) not in goal_seal_neighbors
        ]
        # ceil (rather than round) so integer-count rounding never pulls
        # the realized fraction below the requested change_fraction --
        # important for the 15-20% / >=35% obstacle-change requirements
        # in the Transfer Learning section, which are floors on the
        # *realized* fraction, not targets to round toward.
        n_to_move = max(1, int(np.ceil(change_fraction * len(wall_coords))))
        n_to_move = min(n_to_move, len(wall_coords))
        move_idx = rng.choice(len(wall_coords), size=n_to_move, replace=False)
        cells_to_clear = [wall_coords[i] for i in move_idx]
        for c in cells_to_clear:
            grid[c] = NORMAL

        # Exclude protected special cells *and* the just-cleared cells
        # themselves from the new-wall candidate pool -- otherwise a
        # cleared cell can be immediately re-selected as a "new" wall
        # position, silently turning a move into a no-op and making
        # the realized change_fraction lower than requested.
        protected = {source.start, source.key_pos, source.goal} | set(
            source.door_positions
        )
        excluded = protected | set(cells_to_clear)
        free_mask = grid == NORMAL
        free_coords = [
            (int(x), int(y))
            for x, y in zip(*np.where(free_mask))
            if (int(x), int(y)) not in excluded
        ]
        rng.shuffle(free_coords)
        n_new_walls = min(len(cells_to_clear), len(free_coords))
        for c in free_coords[:n_new_walls]:
            grid[c] = WALL
        free_coords = free_coords[n_new_walls:]

        key_pos = source.key_pos
        goal = source.goal
        if move_key_or_goal and free_coords:
            grid[key_pos] = NORMAL
            key_pos = free_coords.pop()
            grid[key_pos] = KEY

        n_added = 0
        for c in free_coords:
            if n_added >= n_new_penalties:
                break
            grid[c] = PENALTY
            n_added += 1

        # Defensive re-seal: the exclusion above already keeps the
        # move/new-wall steps from touching the chokepoint walls, and
        # key/goal relocation can't land on them either (they're WALL
        # cells, never in the NORMAL-only free_coords pool). This call
        # is a no-op in the current code path, but it's what protects
        # this function if move_key_or_goal is ever extended to also
        # relocate the goal itself -- in which case it correctly
        # signals failure (via the start/key_pos check) instead of
        # producing a target map with an unsealed goal.
        if not _seal_goal_behind_doors(
            grid, goal, source.door_positions, source.start, key_pos
        ):
            continue

        if validate_map(grid, source.start, key_pos, goal, source.door_positions):
            return MapSpec(
                name=name,
                student_id=source.student_id,
                base_seed=source.base_seed,
                maze_size=size,
                grid=grid,
                start=source.start,
                key_pos=key_pos,
                door_positions=source.door_positions,
                goal=goal,
                generation_attempt=attempt,
            )

    raise RuntimeError(
        f"Failed to generate a valid transfer target {name!r} "
        f"within {max_attempts} deterministic attempts."
    )


def num_traversable_cells(grid: np.ndarray) -> int:
    """Count cells the agent could ever legally occupy.

    Parameters
    ----------
    grid : ndarray of shape (size, size)
        Cell-type grid.

    Returns
    -------
    int
        Number of cells whose type is not ``WALL``. The closed door
        counts as traversable once the key is obtained, so it is
        included.
    """
    return int(np.sum(grid != WALL))
