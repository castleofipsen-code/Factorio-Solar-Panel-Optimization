"""Coordinate-only Stage-A search scored by the unchanged Stage-B root LP.

Stage A is represented by exactly 24 integer coordinates:

    (row, column) for the two non-fixed substations, and
    (row, column) for each of ten medium electric poles.

There are no Stage-A placement binaries and no learned bound objective.  A
standalone feasibility oracle enforces the same 5+10 network restrictions used
by the current coverage search.  Feasible coordinate layouts are evaluated by
the existing fixed-network Stage-B packing LP, whose exact root bound is the
only optimization score.

The search first exhausts the 9 x 54 movable-substation combinations for the
best known medium layouts.  Its main search then makes coherent global mosaics
that replace four to nine spatial roles at once using hundreds of historically
feasible coordinate families.  Separate coarse spatial islands receive local
destroy-and-repair improvement.  Stage-B equality duals are used only as valid
upper bounds for pruning; every ranking decision, substation sweep, and
incumbent improvement is based on an exact Stage-B solve.
"""

from __future__ import annotations

import argparse
import ast
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import itertools
import math
from pathlib import Path
import random
import threading
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

import objectives
import parameters


GRID = 50
DD = GRID * GRID
NETWORK_SIZE = 2 * DD
FIXED_CORNER_SUBSTATIONS = (
    (8, 8),
    (8, GRID - 10),
    (GRID - 10, 8),
)
TOP_SUBSTATION_COORDINATES = tuple((8, column) for column in range(20, 29))
INNER_SUBSTATION_COORDINATES = tuple(
    (row, column)
    for row in range(26, 32)
    for column in range(34, 43)
)
ROBOPORT_ROOT = (23, 23)
EXACT_SUBSTATIONS = 5
EXACT_MEDIUMS = 10
STAGE_B_NETWORK_INDICES = np.concatenate(
    [
        np.arange(2 * DD, 3 * DD, dtype=int),
        np.arange(4 * DD, 5 * DD, dtype=int),
    ]
)


def coordinate_index(coordinate: tuple[int, int]) -> int:
    return coordinate[0] * GRID + coordinate[1]


def index_coordinate(index: int) -> tuple[int, int]:
    return divmod(int(index), GRID)


def medium_color(coordinate: tuple[int, int]) -> int:
    row, column = coordinate
    return 2 * (row % 2) + column % 2


@dataclass(frozen=True, order=True)
class CoordinateLayout:
    """Canonical 24-integer Stage-A state."""

    top_substation: tuple[int, int]
    inner_substation: tuple[int, int]
    medium_poles: tuple[tuple[int, int], ...]

    @classmethod
    def create(
        cls,
        top_substation,
        inner_substation,
        medium_poles,
    ) -> "CoordinateLayout":
        return cls(
            tuple(map(int, top_substation)),
            tuple(map(int, inner_substation)),
            tuple(sorted(tuple(map(int, pole)) for pole in medium_poles)),
        )

    @property
    def substations(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(FIXED_CORNER_SUBSTATIONS + (
            self.top_substation,
            self.inner_substation,
        )))

    @property
    def coordinate_vector(self) -> tuple[int, ...]:
        values = [*self.top_substation, *self.inner_substation]
        for coordinate in self.medium_poles:
            values.extend(coordinate)
        return tuple(values)

    @property
    def key(self) -> tuple[int, ...]:
        return (
            coordinate_index(self.top_substation),
            coordinate_index(self.inner_substation),
            *(coordinate_index(coordinate) for coordinate in self.medium_poles),
        )

    @property
    def medium_key(self) -> tuple[int, ...]:
        return tuple(coordinate_index(coordinate) for coordinate in self.medium_poles)

    @property
    def selected_network_indices(self) -> np.ndarray:
        return np.asarray(
            [
                *(coordinate_index(coordinate) for coordinate in self.substations),
                *(
                    DD + coordinate_index(coordinate)
                    for coordinate in self.medium_poles
                ),
            ],
            dtype=int,
        )

    def network_vector(self) -> np.ndarray:
        network = np.zeros(NETWORK_SIZE)
        network[self.selected_network_indices] = 1
        return network


class CoordinateStageAFeasibility:
    """Exact direct checker for the current coordinate Stage-A constraints."""

    def __init__(self):
        self.top_set = set(TOP_SUBSTATION_COORDINATES)
        self.inner_set = set(INNER_SUBSTATION_COORDINATES)
        self.substation_allowed = np.ones(DD, dtype=bool)
        self.medium_allowed = np.ones(DD, dtype=bool)
        self._apply_roboport_clearance_and_gap_bounds()

        self.substation_floor_masks = tuple(
            self._floor_coverage_mask(index, 8, 9) for index in range(DD)
        )
        self.medium_floor_masks = tuple(
            self._floor_coverage_mask(index, 3, 3) for index in range(DD)
        )
        self.all_floor_placements = (1 << DD) - 1

        allowed_indices = np.flatnonzero(self.medium_allowed)
        self.medium_indices_by_color = tuple(
            tuple(
                int(index)
                for index in allowed_indices
                if medium_color(index_coordinate(int(index))) == color
            )
            for color in range(4)
        )

    @staticmethod
    def _footprint(index: int, size: int) -> set[int]:
        row, column = index_coordinate(index)
        return {
            ((row + dr) % GRID) * GRID + ((column + dc) % GRID)
            for dr in range(size)
            for dc in range(size)
        }

    @staticmethod
    def _floor_coverage_mask(index: int, left: int, right: int) -> int:
        """Building roots whose wrapped 5x5 footprint intersects this coverage."""
        row, column = index_coordinate(index)
        covered_rows = {
            (row + offset) % GRID
            for offset in range(-left, right + 1)
        }
        covered_columns = {
            (column + offset) % GRID
            for offset in range(-left, right + 1)
        }
        building_rows = {
            (covered_row - offset) % GRID
            for covered_row in covered_rows
            for offset in range(5)
        }
        building_columns = {
            (covered_column - offset) % GRID
            for covered_column in covered_columns
            for offset in range(5)
        }

        mask = 0
        for building_row in building_rows:
            for building_column in building_columns:
                mask |= 1 << (building_row * GRID + building_column)
        return mask

    def _apply_roboport_clearance_and_gap_bounds(self):
        root_row, root_column = ROBOPORT_ROOT

        # Medium poles touching the roboport remain legal.  The one-tile-away
        # outer ring is forbidden, exactly matching the current MAE model.
        offset = 2
        for row in range(root_row - offset, root_row + 4 + offset):
            for column in range(root_column - offset, root_column + 4 + offset):
                inside_inner_ring = (
                    root_row - offset + 1 <= row < root_row + 4 + offset - 1
                    and root_column - offset + 1
                    <= column
                    < root_column + 4 + offset - 1
                )
                if not inside_inner_ring and 0 <= row < GRID and 0 <= column < GRID:
                    self.medium_allowed[row * GRID + column] = False

        # The corresponding substation-root ring for forbidden gap one.
        sub_outer_row_lo = max(0, root_row - 1 - 2)
        sub_outer_row_hi = min(GRID, root_row + 4 + 1 + 1)
        sub_outer_col_lo = max(0, root_column - 1 - 2)
        sub_outer_col_hi = min(GRID, root_column + 4 + 1 + 1)
        for row in range(sub_outer_row_lo, sub_outer_row_hi):
            for column in range(sub_outer_col_lo, sub_outer_col_hi):
                inside_inner_ring = (
                    sub_outer_row_lo + 1 <= row < sub_outer_row_hi - 1
                    and sub_outer_col_lo + 1 <= column < sub_outer_col_hi - 1
                )
                if not inside_inner_ring:
                    self.substation_allowed[row * GRID + column] = False

        # The central 4x4 roboport footprint is always physically clear.  A
        # substation is forbidden when any tile of its wrapped 2x2 footprint
        # intersects it; a medium pole occupies its root tile.
        central_tiles = {
            row * GRID + column
            for row in range(root_row, root_row + 4)
            for column in range(root_column, root_column + 4)
        }
        for index in range(DD):
            if self._footprint(index, 2) & central_tiles:
                self.substation_allowed[index] = False
        self.medium_allowed[np.fromiter(central_tiles, dtype=int)] = False

    @staticmethod
    def _wire_connected(
        parent_kind: str,
        parent_index: int,
        child_kind: str,
        child_index: int,
    ) -> bool:
        parent_row, parent_column = index_coordinate(parent_index)
        child_row, child_column = index_coordinate(child_index)
        parent_offset = 1.0 if parent_kind == "sub" else 0.5
        child_offset = 1.0 if child_kind == "sub" else 0.5
        radius = 18.0 if parent_kind == child_kind == "sub" else 9.0
        row_distance = child_row + child_offset - parent_row - parent_offset
        column_distance = (
            child_column + child_offset - parent_column - parent_offset
        )
        return row_distance * row_distance + column_distance * column_distance <= (
            radius * radius + 1e-12
        )

    def check(self, layout: CoordinateLayout) -> tuple[bool, str]:
        if layout.top_substation not in self.top_set:
            return False, "top substation is outside its 9-coordinate window"
        if layout.inner_substation not in self.inner_set:
            return False, "inner substation is outside its 54-coordinate window"
        if len(layout.medium_poles) != EXACT_MEDIUMS:
            return False, "medium-pole count is not ten"
        if len(set(layout.medium_poles)) != EXACT_MEDIUMS:
            return False, "duplicate medium-pole coordinates"

        substations = layout.substations
        if len(set(substations)) != EXACT_SUBSTATIONS:
            return False, "duplicate substation coordinates"
        substation_indices = tuple(map(coordinate_index, substations))
        medium_indices = tuple(map(coordinate_index, layout.medium_poles))

        if any(not (0 <= row < GRID and 0 <= column < GRID) for row, column in layout.medium_poles):
            return False, "medium-pole coordinate is outside the grid"
        if any(not self.substation_allowed[index] for index in substation_indices):
            return False, "substation violates roboport clearance/gap bounds"
        if any(not self.medium_allowed[index] for index in medium_indices):
            return False, "medium pole violates roboport clearance/gap bounds"

        color_counts = np.bincount(
            [medium_color(coordinate) for coordinate in layout.medium_poles],
            minlength=4,
        )
        if (color_counts[0] - color_counts[3]) % 3:
            return False, "medium colors 0 and 3 fail modulo-three completion"
        if (color_counts[1] - color_counts[2]) % 3:
            return False, "medium colors 1 and 2 fail modulo-three completion"

        # Stage B would be infeasible if fixed network footprints overlap, so
        # reject those coordinate states before invoking the packing LP.
        occupied_tiles: set[int] = set()
        for index in substation_indices:
            footprint = self._footprint(index, 2)
            if occupied_tiles & footprint:
                return False, "overlapping substation footprints"
            occupied_tiles.update(footprint)
        for index in medium_indices:
            if index in occupied_tiles:
                return False, "medium pole overlaps a substation footprint"
            occupied_tiles.add(index)

        covered_floor = 0
        for index in substation_indices:
            covered_floor |= self.substation_floor_masks[index]
        for index in medium_indices:
            covered_floor |= self.medium_floor_masks[index]
        if covered_floor != self.all_floor_placements:
            return False, "network does not hit every wrapped 5x5 floor placement"

        nodes = [*(('sub', index) for index in substation_indices)]
        nodes.extend(('med', index) for index in medium_indices)
        root_index = min(substation_indices)
        for child_kind, child_index in nodes:
            if child_kind == "sub" and child_index == root_index:
                continue
            has_parent = any(
                parent_index < child_index
                and self._wire_connected(
                    parent_kind,
                    parent_index,
                    child_kind,
                    child_index,
                )
                for parent_kind, parent_index in nodes
            )
            if not has_parent:
                return False, f"{child_kind} at {index_coordinate(child_index)} has no lower-rank parent"

        return True, "ok"


def add_matrix_constraints(model, matrix, variables, lower, upper):
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    equal = finite_lower & finite_upper & np.isclose(lower, upper)
    lower_only = finite_lower & ~equal
    upper_only = finite_upper & ~equal
    if np.any(equal):
        model.addMConstr(matrix[equal], variables, "=", upper[equal])
    if np.any(lower_only):
        model.addMConstr(matrix[lower_only], variables, ">", lower[lower_only])
    if np.any(upper_only):
        model.addMConstr(matrix[upper_only], variables, "<", upper[upper_only])


class ExactStageBLP:
    """Unchanged Stage-B packing LP, parameterized by coordinate Stage A."""

    def __init__(
        self,
        initial_network,
        time_limit,
        periodic_electric_coverage=True,
    ):
        result = objectives.construct_matrix_coverage_fixed_network(
            GRID,
            initial_network[:DD],
            initial_network[DD:NETWORK_SIZE],
            min_power=8000,
            roboport_substitution_factor=0,
            periodic_electric_coverage=periodic_electric_coverage,
        )
        A, b_lb, b_ub, n, c, lb, ub, _ = result
        A = sp.csr_matrix(A)
        lb = np.asarray(lb, dtype=float).copy()
        ub = np.asarray(ub, dtype=float).copy()
        lb[STAGE_B_NETWORK_INDICES] = 0
        ub[STAGE_B_NETWORK_INDICES] = 1

        self.model = gp.Model("coordinate_stage_b_root_lp")
        self.model.Params.OutputFlag = 0
        self.model.Params.Threads = 1
        self.model.Params.Method = 2
        self.model.Params.Crossover = 0
        self.model.Params.Presolve = 2
        self.model.Params.TimeLimit = time_limit
        self.variables = self.model.addMVar(
            n,
            lb=lb,
            ub=ub,
            obj=c,
            vtype=GRB.CONTINUOUS,
            name="stage_b_x",
        )
        self.model.ModelSense = GRB.MINIMIZE
        add_matrix_constraints(
            self.model,
            A,
            self.variables,
            np.asarray(b_lb),
            np.asarray(b_ub),
        )
        self.network_fixing = self.model.addConstr(
            self.variables[STAGE_B_NETWORK_INDICES] == initial_network,
            name="coordinate_network",
        )
        self.model.update()

    def evaluate(self, layout: CoordinateLayout):
        network = layout.network_vector()
        self.model.reset()
        self.network_fixing.RHS = network
        self.model.update()
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            return math.nan, None, float(self.model.Runtime), int(self.model.Status)
        bound = -float(self.model.ObjVal)
        equality_dual = np.asarray(self.network_fixing.Pi, dtype=float).copy()
        return bound, equality_dual, float(self.model.Runtime), int(self.model.Status)


class ParallelStageBEvaluator:
    def __init__(
        self,
        initial_network,
        workers,
        time_limit,
        periodic_electric_coverage=True,
    ):
        self.initial_network = initial_network
        self.time_limit = time_limit
        self.periodic_electric_coverage = periodic_electric_coverage
        self.local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def _evaluate(self, layout):
        evaluator = getattr(self.local, "evaluator", None)
        if evaluator is None:
            evaluator = ExactStageBLP(
                self.initial_network,
                self.time_limit,
                self.periodic_electric_coverage,
            )
            self.local.evaluator = evaluator
        return layout, *evaluator.evaluate(layout)

    def evaluate(self, layouts):
        return list(self.executor.map(self._evaluate, layouts))

    def close(self):
        self.executor.shutdown(wait=True)


@dataclass
class ExactEvaluation:
    layout: CoordinateLayout
    bound: float
    runtime: float
    equality_dual: np.ndarray


class DualUpperBoundPool:
    """Small in-memory pool of globally valid Stage-B affine upper planes."""

    def __init__(self, maximum_cuts=512, safety=0.02):
        self.maximum_cuts = maximum_cuts
        self.safety = safety
        self.constants: list[float] = []
        self.duals: list[np.ndarray] = []
        self.constant_array = None
        self.dual_matrix = None

    def add(self, layout, bound, equality_dual):
        selected = layout.selected_network_indices
        self.constants.append(float(bound + np.sum(equality_dual[selected])))
        self.duals.append(np.asarray(equality_dual, dtype=np.float32))
        if len(self.constants) > self.maximum_cuts:
            self.constants.pop(0)
            self.duals.pop(0)
        self.constant_array = None
        self.dual_matrix = None

    def _arrays(self):
        if self.dual_matrix is None:
            self.constant_array = np.asarray(self.constants, dtype=float)
            self.dual_matrix = np.stack(self.duals)
        return self.constant_array, self.dual_matrix

    def upper_bound(self, layout):
        if not self.constants:
            return math.inf
        selected = layout.selected_network_indices
        constants, duals = self._arrays()
        values = constants - np.sum(duals[:, selected], axis=1)
        return float(np.min(values)) + self.safety


def load_layout(path: Path) -> CoordinateLayout:
    substations = []
    medium_poles = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith("x["):
                continue
            close = line.find("]")
            index = int(line[2:close])
            if index >= NETWORK_SIZE:
                break
            if float(line.split()[1]) <= 0.5:
                continue
            if index < DD:
                substations.append(index_coordinate(index))
            else:
                medium_poles.append(index_coordinate(index - DD))

    fixed = set(FIXED_CORNER_SUBSTATIONS)
    movable = [coordinate for coordinate in substations if coordinate not in fixed]
    top = [coordinate for coordinate in movable if coordinate in TOP_SUBSTATION_COORDINATES]
    inner = [coordinate for coordinate in movable if coordinate in INNER_SUBSTATION_COORDINATES]
    if len(substations) != EXACT_SUBSTATIONS or len(top) != 1 or len(inner) != 1:
        raise ValueError(f"{path} is not a current five-substation coordinate layout")
    return CoordinateLayout.create(top[0], inner[0], medium_poles)


def discover_seed_layouts(checker, count):
    coverage_root = Path("coverage/50x50")
    summary_paths = set(coverage_root.rglob("stage_b_lp_screen*.csv"))
    summary_paths.update(coverage_root.rglob("stage_b_lp_bounds*.csv"))
    summary_paths.update(coverage_root.rglob("stage_b_lp_verification*.csv"))
    summary_paths.update(coverage_root.rglob("stage_b_exact_bounds.csv"))
    summary_paths.update(coverage_root.rglob("stage_b_bound_tests_*/summary.csv"))

    path_bounds = {}
    for summary_path in sorted(summary_paths):
        try:
            with summary_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        bound = float(row["positive_bound"])
                        candidate_path = Path(row["path"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not np.isfinite(bound) or not str(candidate_path):
                        continue
                    if not candidate_path.exists():
                        continue
                    previous = path_bounds.get(candidate_path)
                    if previous is None or bound > previous:
                        path_bounds[candidate_path] = bound
        except OSError:
            continue

    explicit_paths = [
        (
            Path(
            "coverage/50x50/20260719_093707_mae/"
            "incumbent_20260719_093707_mae_2.sol"
            ),
            8304.903830,
        ),
        (
            Path(
            "coverage/50x50/20260719_162603_benders/"
            "incumbent_20260719_162603_benders_126.sol"
            ),
            8303.941038,
        ),
    ]
    for path, bound in explicit_paths:
        if path.exists():
            path_bounds[path] = max(bound, path_bounds.get(path, -math.inf))

    records = []
    seen_layouts = set()
    for path, recorded_bound in sorted(
        path_bounds.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        try:
            layout = load_layout(path)
        except (OSError, ValueError, IndexError):
            continue
        feasible, _ = checker.check(layout)
        if not feasible or layout.key in seen_layouts:
            continue
        seen_layouts.add(layout.key)
        records.append((layout, recorded_bound, path))
        if len(records) >= max(count * 8, count + 32):
            break

    # Prefer distinct medium-pole families before filling with close variants.
    selected = []
    selected_keys = set()
    medium_keys = set()
    for record in records:
        if record[0].medium_key not in medium_keys:
            selected.append(record)
            selected_keys.add(record[0].key)
            medium_keys.add(record[0].medium_key)
            if len(selected) == count:
                return selected
    for record in records:
        if record[0].key not in selected_keys:
            selected.append(record)
            selected_keys.add(record[0].key)
            if len(selected) == count:
                break
    return selected


def discover_global_layout_pool(checker, seed_records, count):
    """Load a broad set of previously verified coordinate families.

    High-bound layouts provide useful role definitions, while deliberately
    distant layouts provide different basins for coherent crossover.  Both
    coordinate runs and exactly screened solver runs are read; no learned
    score or Stage-B approximation is imported.
    """
    if count <= 0:
        return []
    seed_layouts = [record[0] for record in seed_records]
    seed_medium_sets = [set(layout.medium_key) for layout in seed_layouts]
    records = {}

    def keep_record(layout, bound):
        previous = records.get(layout.medium_key)
        if previous is None or bound > previous[0]:
            records[layout.medium_key] = (bound, layout)

    paths = sorted(
        Path("coverage/50x50").glob("*_coordinate/coordinate_evaluations.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        try:
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        values = tuple(ast.literal_eval(row["coordinate_vector"]))
                        bound = float(row["positive_bound"])
                        layout = CoordinateLayout.create(
                            values[0:2],
                            values[2:4],
                            zip(values[4::2], values[5::2]),
                        )
                    except (KeyError, TypeError, ValueError, SyntaxError):
                        continue
                    keep_record(layout, bound)
        except OSError:
            continue

    # The large restarted search is the newest source of genuinely
    # different medium families.  Earlier versions of the coordinate search
    # never saw these records because they only loaded coordinate_evaluations.
    # Read only exact Stage-B screens and retain the strongest substation
    # realization available for each medium family.
    screened_paths = set(Path("coverage/50x50").rglob("stage_b_lp_bounds*.csv"))
    screened_paths.update(Path("coverage/50x50").rglob("stage_b_lp_screen*.csv"))
    screened_paths.update(
        Path("coverage/50x50").rglob("stage_b_lp_verification*.csv")
    )
    for summary_path in sorted(screened_paths):
        try:
            with summary_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        bound = float(row["positive_bound"])
                        candidate_path = Path(row["path"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not np.isfinite(bound) or not candidate_path.exists():
                        continue
                    try:
                        layout = load_layout(candidate_path)
                    except (OSError, ValueError, IndexError):
                        continue
                    keep_record(layout, bound)
        except OSError:
            continue

    # Exhaustive family sweeps may find a much better movable-substation pair
    # than the source .sol used to discover the medium family.  Reconstruct
    # that exact best realization directly from the resumable family summary.
    for summary_path in sorted(
        Path("coverage/50x50").rglob("medium_family_substation_summary.csv")
    ):
        try:
            with summary_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        bound = float(row["best_bound"])
                        top = tuple(ast.literal_eval(row["best_top_substation"]))
                        inner = tuple(ast.literal_eval(row["best_inner_substation"]))
                        medium_key = tuple(ast.literal_eval(row["medium_key"]))
                        layout = CoordinateLayout.create(
                            top,
                            inner,
                            (index_coordinate(index) for index in medium_key),
                        )
                    except (KeyError, TypeError, ValueError, SyntaxError):
                        continue
                    if np.isfinite(bound):
                        keep_record(layout, bound)
        except OSError:
            continue

    # Seeds must remain available even if there is no historical coordinate
    # run yet.  Keep their recorded bound only for source-pool selection.
    for layout, bound, _ in seed_records:
        keep_record(layout, bound)

    candidates = []
    for bound, layout in records.values():
        feasible, _ = checker.check(layout)
        if not feasible:
            continue
        medium_set = set(layout.medium_key)
        nearest_seed_changes = min(
            EXACT_MEDIUMS - len(medium_set & seed_set)
            for seed_set in seed_medium_sets
        ) if seed_medium_sets else EXACT_MEDIUMS
        candidates.append((bound, nearest_seed_changes, layout))

    # Reserve one quarter for strong known geometry and use the rest for the
    # farthest available basins.  Deduplication is by medium family, so this
    # pool is not inflated by substation variants.
    selected = []
    selected_mediums = set()
    strong_count = min(count // 4, len(candidates))
    for _, _, layout in sorted(candidates, key=lambda item: item[0], reverse=True):
        if layout.medium_key in selected_mediums:
            continue
        selected.append(layout)
        selected_mediums.add(layout.medium_key)
        if len(selected) >= strong_count:
            break
    for _, _, layout in sorted(
        candidates,
        key=lambda item: (item[1], item[0]),
        reverse=True,
    ):
        if layout.medium_key in selected_mediums:
            continue
        selected.append(layout)
        selected_mediums.add(layout.medium_key)
        if len(selected) >= count:
            break
    return selected


def valid_color_patterns(total=EXACT_MEDIUMS):
    return tuple(
        (m0, m1, m2, total - m0 - m1 - m2)
        for m0 in range(total + 1)
        for m1 in range(total - m0 + 1)
        for m2 in range(total - m0 - m1 + 1)
        if (m0 - (total - m0 - m1 - m2)) % 3 == 0
        and (m1 - m2) % 3 == 0
    )


class GlobalMosaicGenerator:
    """Make coherent large jumps by recombining complete spatial roles.

    Medium poles are canonically sorted by grid index.  Across feasible
    layouts, each rank has a stable broad role in the directed network.  A
    rank mosaic replaces four to nine roles at once, preserving coherent
    coverage/connectivity far more often than independent random coordinates.
    """

    def __init__(
        self,
        checker,
        rng,
        source_layouts,
        reference_layouts,
        minimum_seed_changes=4,
        maximum_sources=4_000,
    ):
        self.checker = checker
        self.rng = rng
        self.minimum_seed_changes = minimum_seed_changes
        self.maximum_sources = maximum_sources
        self.sources = {
            layout.medium_key: layout for layout in source_layouts
        }
        self.reference_medium_sets = tuple(
            set(layout.medium_key) for layout in reference_layouts
        )

    def add_sources(self, layouts):
        for layout in layouts:
            if layout.medium_key in self.sources:
                self.sources[layout.medium_key] = layout
            elif len(self.sources) < self.maximum_sources:
                self.sources[layout.medium_key] = layout

    def _far_enough(self, medium_poles):
        if self.minimum_seed_changes <= 0 or not self.reference_medium_sets:
            return True
        medium_set = {
            coordinate_index(coordinate) for coordinate in medium_poles
        }
        return min(
            EXACT_MEDIUMS - len(medium_set & reference)
            for reference in self.reference_medium_sets
        ) >= self.minimum_seed_changes

    def _rank_mosaic(self, parents):
        if len(parents) == 2:
            left, right = parents
            donor_positions = set(
                self.rng.sample(
                    range(EXACT_MEDIUMS),
                    self.rng.randint(4, 9),
                )
            )
            return [
                right.medium_poles[position]
                if position in donor_positions
                else left.medium_poles[position]
                for position in range(EXACT_MEDIUMS)
            ]

        # A less frequent three-parent mosaic creates genuinely new role
        # combinations without independently randomizing coordinates.
        choices = [self.rng.randrange(len(parents)) for _ in range(EXACT_MEDIUMS)]
        if len(set(choices)) < 2:
            choices[self.rng.randrange(EXACT_MEDIUMS)] = 1
        return [
            parents[parent_number].medium_poles[position]
            for position, parent_number in enumerate(choices)
        ]

    def _block_translation(self, parent):
        start = self.rng.randrange(0, EXACT_MEDIUMS - 3)
        stop = self.rng.randrange(start + 4, EXACT_MEDIUMS + 1)
        row_delta = 2 * self.rng.randint(-6, 6)
        column_delta = 2 * self.rng.randint(-6, 6)
        if row_delta == 0 and column_delta == 0:
            row_delta = 2
        medium_poles = list(parent.medium_poles)
        for position in range(start, stop):
            row, column = medium_poles[position]
            medium_poles[position] = (
                row + row_delta,
                column + column_delta,
            )
        return medium_poles

    def generate(self, maximum, excluded_mediums, attempt_factor=100):
        pool = list(self.sources.values())
        if len(pool) < 2 or maximum <= 0:
            return []
        results = {}
        attempts = 0
        maximum_attempts = max(maximum, maximum * attempt_factor)
        while len(results) < maximum and attempts < maximum_attempts:
            attempts += 1
            draw = self.rng.random()
            if draw < 0.82:
                parents = self.rng.sample(pool, 2)
                medium_poles = self._rank_mosaic(parents)
            elif draw < 0.94 and len(pool) >= 3:
                parents = self.rng.sample(pool, 3)
                medium_poles = self._rank_mosaic(parents)
            else:
                parents = [self.rng.choice(pool)]
                medium_poles = self._block_translation(parents[0])

            if len(set(medium_poles)) != EXACT_MEDIUMS:
                continue
            if not all(
                0 <= row < GRID and 0 <= column < GRID
                for row, column in medium_poles
            ):
                continue
            if not self._far_enough(medium_poles):
                continue
            medium_key = tuple(
                sorted(coordinate_index(coordinate) for coordinate in medium_poles)
            )
            if medium_key in excluded_mediums or medium_key in results:
                continue

            # Test the parents' actual substation realizations first.  This is
            # only a feasibility choice; alternative substations are considered
            # later, after the real Stage-B bound is known.
            substation_choices = [
                (parent.top_substation, parent.inner_substation)
                for parent in parents
            ]
            self.rng.shuffle(substation_choices)
            for top, inner in substation_choices:
                layout = CoordinateLayout.create(top, inner, medium_poles)
                feasible, _ = self.checker.check(layout)
                if feasible:
                    results[medium_key] = layout
                    break
        print(
            f"global mosaic generator: {len(results)} feasible distant "
            f"medium families in {attempts} attempts from {len(pool)} sources",
            flush=True,
        )
        return list(results.values())


class CoordinateRepairGenerator:
    """Generate feasible layouts by repairing deliberate holes in a parent.

    The old generator sampled arbitrary coordinates and rejected almost every
    draw.  Here the retained network is analysed first.  Replacement poles are
    restricted to coordinates that already have a lower-ranked wire parent,
    and are ranked by the floor holes and disconnected retained nodes they
    repair.  The final exact checker remains authoritative.
    """

    def __init__(self, checker, rng, pool_per_color=72, repair_jobs=14):
        self.checker = checker
        self.rng = rng
        self.pool_per_color = pool_per_color
        self.repair_jobs = repair_jobs

    @staticmethod
    def _color_signature(layout):
        return tuple(
            int(value)
            for value in np.bincount(
                [medium_color(pole) for pole in layout.medium_poles],
                minlength=4,
            )
        )

    def _has_lower_parent(self, child_kind, child_index, nodes):
        return any(
            parent_index < child_index
            and self.checker._wire_connected(
                parent_kind,
                parent_index,
                child_kind,
                child_index,
            )
            for parent_kind, parent_index in nodes
        )

    def _partial_state(self, parent, retained):
        substation_indices = tuple(map(coordinate_index, parent.substations))
        retained_indices = tuple(map(coordinate_index, retained))
        nodes = [*(('sub', index) for index in substation_indices)]
        nodes.extend(('med', index) for index in retained_indices)
        root_index = min(substation_indices)
        orphans = tuple(
            (kind, index)
            for kind, index in nodes
            if not (kind == "sub" and index == root_index)
            and not self._has_lower_parent(kind, index, nodes)
        )

        covered = 0
        for index in substation_indices:
            covered |= self.checker.substation_floor_masks[index]
        for index in retained_indices:
            covered |= self.checker.medium_floor_masks[index]

        substation_tiles = set()
        for index in substation_indices:
            substation_tiles.update(self.checker._footprint(index, 2))
        return (
            nodes,
            orphans,
            self.checker.all_floor_placements & ~covered,
            substation_tiles,
        )

    def _candidate_pool(
        self,
        color,
        retained,
        removed,
        nodes,
        orphans,
        uncovered,
        substation_tiles,
        donor_coordinates,
        guidance_dual,
        limit,
    ):
        occupied = set(retained)
        entries = []
        for index in self.checker.medium_indices_by_color[color]:
            coordinate = index_coordinate(index)
            if coordinate in occupied or index in substation_tiles:
                continue
            if not self._has_lower_parent("med", index, nodes):
                continue

            repair_bits = 0
            for orphan_number, (kind, orphan_index) in enumerate(orphans):
                if (
                    index < orphan_index
                    and self.checker._wire_connected(
                        "med", index, kind, orphan_index
                    )
                ):
                    repair_bits |= 1 << orphan_number
            floor_mask = self.checker.medium_floor_masks[index]
            floor_gain = (floor_mask & uncovered).bit_count()
            distance = min(
                abs(coordinate[0] - old[0]) + abs(coordinate[1] - old[1])
                for old in removed
            )
            dual_preference = (
                -float(guidance_dual[DD + index])
                if guidance_dual is not None
                else 0.0
            )
            score = (
                repair_bits.bit_count(),
                floor_gain,
                dual_preference,
                int(coordinate in donor_coordinates),
                -distance,
                self.rng.random(),
            )
            entries.append(
                (score, coordinate, index, floor_mask, repair_bits)
            )

        entries.sort(key=lambda entry: entry[0], reverse=True)
        if len(entries) <= limit:
            return [entry[1:] for entry in entries]

        # Keep most of the best targeted coordinates, but reserve part of the
        # pool for broad exploration so every repair job is not deterministic.
        deterministic = max(1, 3 * limit // 4)
        selected = entries[:deterministic]
        remainder = entries[deterministic:]
        selected.extend(
            self.rng.sample(remainder, min(limit - deterministic, len(remainder)))
        )
        selected.sort(key=lambda entry: entry[0], reverse=True)
        return [entry[1:] for entry in selected]

    def _repair_destroy_set(
        self,
        parent,
        destroyed_positions,
        donor_coordinates,
        guidance_dual,
        excluded_keys,
        maximum,
        pool_limit,
    ):
        destroyed = set(destroyed_positions)
        removed = [
            pole
            for position, pole in enumerate(parent.medium_poles)
            if position in destroyed
        ]
        retained = [
            pole
            for position, pole in enumerate(parent.medium_poles)
            if position not in destroyed
        ]
        nodes, orphans, uncovered, substation_tiles = self._partial_state(
            parent, retained
        )
        required_colors = sorted(medium_color(pole) for pole in removed)
        pools = {}
        for color in set(required_colors):
            pools[color] = self._candidate_pool(
                color,
                retained,
                removed,
                nodes,
                orphans,
                uncovered,
                substation_tiles,
                donor_coordinates,
                guidance_dual,
                pool_limit,
            )
            if not pools[color]:
                return []

        suffix_floor = [0] * (len(required_colors) + 1)
        suffix_orphans = [0] * (len(required_colors) + 1)
        for position in range(len(required_colors) - 1, -1, -1):
            color = required_colors[position]
            floor_union = 0
            orphan_union = 0
            for _, _, floor_mask, repair_bits in pools[color]:
                floor_union |= floor_mask
                orphan_union |= repair_bits
            suffix_floor[position] = suffix_floor[position + 1] | floor_union
            suffix_orphans[position] = (
                suffix_orphans[position + 1] | orphan_union
            )

        all_orphans = (1 << len(orphans)) - 1
        results = {}
        chosen = []
        chosen_coordinates = set(retained)
        last_index_by_color = {}

        def visit(position, floor_coverage, repaired_orphans):
            if len(results) >= maximum:
                return
            if uncovered & ~(floor_coverage | suffix_floor[position]):
                return
            if all_orphans & ~(
                repaired_orphans | suffix_orphans[position]
            ):
                return
            if position == len(required_colors):
                if uncovered & ~floor_coverage:
                    return
                if repaired_orphans != all_orphans:
                    return
                layout = CoordinateLayout.create(
                    parent.top_substation,
                    parent.inner_substation,
                    [*retained, *chosen],
                )
                if layout.key in excluded_keys or layout.key in results:
                    return
                feasible, _ = self.checker.check(layout)
                if feasible:
                    results[layout.key] = layout
                return

            color = required_colors[position]
            previous_same_color = last_index_by_color.get(color, -1)
            for coordinate, index, floor_mask, repair_bits in pools[color]:
                if coordinate in chosen_coordinates or index <= previous_same_color:
                    continue
                chosen.append(coordinate)
                chosen_coordinates.add(coordinate)
                old_last = last_index_by_color.get(color)
                last_index_by_color[color] = index
                visit(
                    position + 1,
                    floor_coverage | floor_mask,
                    repaired_orphans | repair_bits,
                )
                if old_last is None:
                    del last_index_by_color[color]
                else:
                    last_index_by_color[color] = old_last
                chosen_coordinates.remove(coordinate)
                chosen.pop()
                if len(results) >= maximum:
                    break

        visit(0, 0, 0)
        return list(results.values())

    def _blend_parents(self, parent, donors, excluded_keys, maximum):
        results = {}
        parent_signature = self._color_signature(parent)
        for donor in donors:
            if (
                donor.medium_key == parent.medium_key
                or self._color_signature(donor) != parent_signature
            ):
                continue
            pools = {
                color: tuple(
                    set(
                        pole
                        for pole in (*parent.medium_poles, *donor.medium_poles)
                        if medium_color(pole) == color
                    )
                )
                for color in range(4)
            }
            for _ in range(40):
                medium_poles = []
                possible = True
                for color, count in enumerate(parent_signature):
                    if len(pools[color]) < count:
                        possible = False
                        break
                    medium_poles.extend(self.rng.sample(pools[color], count))
                if not possible:
                    break
                layout = CoordinateLayout.create(
                    parent.top_substation,
                    parent.inner_substation,
                    medium_poles,
                )
                if layout.key in excluded_keys or layout.key in results:
                    continue
                feasible, _ = self.checker.check(layout)
                if feasible:
                    results[layout.key] = layout
                    if len(results) >= maximum:
                        return list(results.values())
        return list(results.values())

    def generate(
        self,
        parent,
        donors,
        excluded_keys,
        maximum,
        strength=1,
        guidance_dual=None,
    ):
        results = {}

        def add(layouts):
            for layout in layouts:
                if layout.key not in excluded_keys and layout.key not in results:
                    results[layout.key] = layout
                    if len(results) >= maximum:
                        break

        crossover_budget = max(2, maximum // 5)
        add(
            self._blend_parents(
                parent,
                donors,
                excluded_keys,
                crossover_budget,
            )
        )

        single_budget = max(4, maximum // 4)
        single_positions = list(range(EXACT_MEDIUMS))
        self.rng.shuffle(single_positions)
        for position in single_positions:
            if len(results) >= crossover_budget + single_budget:
                break
            add(
                self._repair_destroy_set(
                    parent,
                    (position,),
                    set().union(*(set(donor.medium_poles) for donor in donors)),
                    guidance_dual,
                    excluded_keys | set(results),
                    min(6, crossover_budget + single_budget - len(results)),
                    self.pool_per_color,
                )
            )

        pair_budget = max(8, maximum - len(results))
        pair_jobs = list(itertools.combinations(range(EXACT_MEDIUMS), 2))
        self.rng.shuffle(pair_jobs)
        donor_coordinates = set().union(
            *(set(donor.medium_poles) for donor in donors)
        )
        for destroyed_positions in pair_jobs[: self.repair_jobs]:
            if len(results) >= maximum:
                break
            add(
                self._repair_destroy_set(
                    parent,
                    destroyed_positions,
                    donor_coordinates,
                    guidance_dual,
                    excluded_keys | set(results),
                    min(8, maximum - len(results), pair_budget),
                    self.pool_per_color,
                )
            )

        # After sustained stagnation, make a few genuine three-pole jumps.
        if strength >= 2 and len(results) < maximum:
            triple_jobs = list(itertools.combinations(range(EXACT_MEDIUMS), 3))
            self.rng.shuffle(triple_jobs)
            for destroyed_positions in triple_jobs[: max(2, self.repair_jobs // 3)]:
                if len(results) >= maximum:
                    break
                add(
                    self._repair_destroy_set(
                        parent,
                        destroyed_positions,
                        donor_coordinates,
                        guidance_dual,
                        excluded_keys | set(results),
                        min(6, maximum - len(results)),
                        min(36, self.pool_per_color),
                    )
                )

        return list(results.values())


class CoordinateSearch:
    def __init__(self, args, checker, seeds, global_layouts):
        self.args = args
        self.checker = checker
        self.rng = random.Random(args.seed)
        self.seed_records = seeds
        self.cache: dict[tuple[int, ...], ExactEvaluation] = {}
        self.seen_keys: set[tuple[int, ...]] = set()
        self.feasible_archive: dict[tuple[int, ...], CoordinateLayout] = {}
        self.archive_scores: dict[tuple[int, ...], float] = {}
        self.parent_uses: dict[tuple[int, ...], int] = {}
        self.cut_pool = DualUpperBoundPool(args.maximum_dual_cuts)
        self.best: ExactEvaluation | None = None
        self.evaluation_number = 0
        self.incumbent_number = 0
        self.swept_mediums = set()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{timestamp}_coordinate"
        self.folder = Path("coverage/50x50") / self.run_name
        self.folder.mkdir(parents=True, exist_ok=True)
        self.csv_handle = (self.folder / "coordinate_evaluations.csv").open(
            "w", newline="", buffering=1
        )
        self.writer = csv.writer(self.csv_handle)
        self.writer.writerow(
            [
                "evaluation_id",
                "phase",
                "generation",
                "status",
                "positive_bound",
                "lp_runtime",
                "dual_upper_before_solve",
                "is_new_best",
                "top_substation",
                "inner_substation",
                "medium_poles",
                "coordinate_vector",
                "solution_path",
            ]
        )

        initial_network = seeds[0][0].network_vector()
        self.evaluator = ParallelStageBEvaluator(
            initial_network,
            args.workers,
            args.lp_seconds,
        )
        self.mutator = CoordinateRepairGenerator(
            checker,
            self.rng,
            pool_per_color=args.repair_pool_per_color,
            repair_jobs=args.repair_jobs_per_parent,
        )
        self.global_generator = GlobalMosaicGenerator(
            checker,
            self.rng,
            [*(record[0] for record in seeds), *global_layouts],
            [record[0] for record in seeds],
            minimum_seed_changes=args.global_min_seed_changes,
            maximum_sources=max(args.global_source_layouts * 4, 1_000),
        )

    def close(self):
        self.evaluator.close()
        self.csv_handle.close()

    def _remember_layout(self, layout, exact_bound=None):
        """Keep one useful substation realization of each medium family."""
        medium_key = layout.medium_key
        score = -math.inf if exact_bound is None else float(exact_bound)
        previous_score = self.archive_scores.get(medium_key, -math.inf)
        if medium_key in self.feasible_archive and score <= previous_score:
            return
        if (
            medium_key not in self.feasible_archive
            and len(self.feasible_archive) >= self.args.archive_size
        ):
            # Prefer replacing a heavily reused family, preserving the archive
            # as a source of genuinely new parents during long runs.
            sample = self.rng.sample(
                list(self.feasible_archive),
                min(64, len(self.feasible_archive)),
            )
            victim = max(sample, key=lambda key: self.parent_uses.get(key, 0))
            del self.feasible_archive[victim]
            self.archive_scores.pop(victim, None)
            self.parent_uses.pop(victim, None)
        self.feasible_archive[medium_key] = layout
        self.archive_scores[medium_key] = score

    def _write_solution(self, evaluation):
        path = self.folder / (
            f"incumbent_{self.run_name}_{self.incumbent_number}.sol"
        )
        self.incumbent_number += 1
        network = evaluation.layout.network_vector()
        with path.open("w") as handle:
            handle.write("# Coordinate Stage-A solution\n")
            handle.write(
                f"# Exact Stage-B LP bound = {evaluation.bound:.16g}\n"
            )
            handle.write(
                "# Coordinate vector = "
                + " ".join(map(str, evaluation.layout.coordinate_vector))
                + "\n"
            )
            for index, value in enumerate(network):
                handle.write(f"x[{index}] {value:.16g}\n")
        return path

    def _record_result(
        self,
        layout,
        bound,
        dual,
        runtime,
        status,
        upper_before,
        phase,
        generation,
    ):
        is_new_best = bool(
            np.isfinite(bound)
            and (
                self.best is None
                or bound > self.best.bound + self.args.improvement_tolerance
            )
        )
        evaluation = None
        solution_path = ""
        if np.isfinite(bound):
            evaluation = ExactEvaluation(layout, bound, runtime, dual)
            self.cache[layout.key] = evaluation
            self._remember_layout(layout, bound)
            self.cut_pool.add(layout, bound, dual)
            if is_new_best:
                self.best = evaluation
                path = self._write_solution(evaluation)
                solution_path = str(path).replace("\\", "/")
                print(
                    f"NEW COORDINATE BEST {bound:.6f}: "
                    f"top={layout.top_substation} inner={layout.inner_substation} "
                    f"mediums={layout.medium_poles}",
                    flush=True,
                )

        self.writer.writerow(
            [
                self.evaluation_number,
                phase,
                generation,
                status,
                bound,
                runtime,
                upper_before,
                int(is_new_best),
                layout.top_substation,
                layout.inner_substation,
                layout.medium_poles,
                layout.coordinate_vector,
                solution_path,
            ]
        )
        self.evaluation_number += 1
        return evaluation

    def evaluate_candidates(
        self,
        layouts,
        phase,
        generation=0,
        maximum_exact=None,
    ):
        unique = {}
        for layout in layouts:
            if layout.key in self.seen_keys or layout.key in unique:
                continue
            unique[layout.key] = layout
            self.seen_keys.add(layout.key)
            self._remember_layout(layout)
        if not unique:
            return []

        scored = [
            (self.cut_pool.upper_bound(layout), self.rng.random(), layout)
            for layout in unique.values()
        ]
        best_bound = self.best.bound if self.best is not None else -math.inf
        scored = [
            item for item in scored if item[0] > best_bound - 1e-6
        ]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        competitive_count = len(scored)
        if maximum_exact is not None:
            scored = scored[:maximum_exact]

        results = []
        for start in range(0, len(scored), self.args.evaluation_batch):
            # Recheck upper bounds after every exact batch because each result
            # contributes another globally valid affine plane.
            batch_scored = []
            current_best = self.best.bound if self.best is not None else -math.inf
            for _, tie_break, layout in scored[start:start + self.args.evaluation_batch]:
                upper = self.cut_pool.upper_bound(layout)
                if upper > current_best - 1e-6:
                    batch_scored.append((upper, tie_break, layout))
            if not batch_scored:
                continue

            evaluated = self.evaluator.evaluate(
                [item[2] for item in batch_scored]
            )
            for (layout, bound, dual, runtime, status), (upper, _, _) in zip(
                evaluated,
                batch_scored,
            ):
                evaluation = self._record_result(
                    layout,
                    bound,
                    dual,
                    runtime,
                    status,
                    upper,
                    phase,
                    generation,
                )
                if evaluation is not None:
                    results.append(evaluation)
                if (
                    self.best is not None
                    and self.best.bound
                    >= self.args.target - self.args.improvement_tolerance
                ):
                    return results
        print(
            f"{phase} generation {generation}: {len(unique)} new feasible, "
            f"{competitive_count} dual-competitive, {len(results)} exact LPs",
            flush=True,
        )
        return results

    def evaluate_seeds(self):
        layouts = [record[0] for record in self.seed_records]
        print(f"exactly evaluating {len(layouts)} coordinate seeds", flush=True)
        return self.evaluate_candidates(layouts, "seed")

    def sweep_substations(self, base_evaluation, generation=0):
        medium_key = base_evaluation.layout.medium_key
        if medium_key in self.swept_mediums:
            return []
        self.swept_mediums.add(medium_key)
        candidates = []
        for top in TOP_SUBSTATION_COORDINATES:
            for inner in INNER_SUBSTATION_COORDINATES:
                layout = CoordinateLayout.create(
                    top,
                    inner,
                    base_evaluation.layout.medium_poles,
                )
                feasible, _ = self.checker.check(layout)
                if feasible and layout.key not in self.seen_keys:
                    candidates.append(layout)
        print(
            f"substation sweep for medium family {medium_key}: "
            f"{len(candidates)} feasible coordinate combinations",
            flush=True,
        )
        return self.evaluate_candidates(
            candidates,
            "substation_sweep",
            generation,
        )

    def population(self):
        family_best = {}
        for evaluation in self.cache.values():
            medium_key = evaluation.layout.medium_key
            previous = family_best.get(medium_key)
            if previous is None or evaluation.bound > previous.bound:
                family_best[medium_key] = evaluation
        evaluations = sorted(
            family_best.values(),
            key=lambda evaluation: evaluation.bound,
            reverse=True,
        )
        selected = []
        selected_mediums = set()
        elite_target = max(4, self.args.population // 3)
        for evaluation in evaluations:
            medium_key = evaluation.layout.medium_key
            if medium_key in selected_mediums:
                continue
            selected.append(evaluation.layout)
            selected_mediums.add(medium_key)
            if len(selected) >= min(elite_target, self.args.population):
                break

        # Preserve the strongest exactly evaluated representative of each
        # coarse spatial island.  This lets distant global basins improve
        # locally without competing directly with the 8304 basin for every
        # parent slot.
        island_best = {}
        for evaluation in evaluations:
            signature = tuple(
                (
                    coordinate[0] // self.args.island_bin_size,
                    coordinate[1] // self.args.island_bin_size,
                )
                for coordinate in evaluation.layout.medium_poles
            )
            previous = island_best.get(signature)
            if previous is None or evaluation.bound > previous.bound:
                island_best[signature] = evaluation
        archive_candidates = [
            evaluation.layout
            for evaluation in island_best.values()
            if evaluation.layout.medium_key not in selected_mediums
        ]
        if len(archive_candidates) > self.args.archive_parent_sample:
            # First retain the stronger half of the sample, then add a random
            # half so low-bound but genuinely different islands are not lost.
            archive_candidates.sort(
                key=lambda layout: family_best[layout.medium_key].bound,
                reverse=True,
            )
            strong = archive_candidates[: self.args.archive_parent_sample // 2]
            remainder = archive_candidates[len(strong):]
            archive_candidates = [
                *strong,
                *self.rng.sample(
                    remainder,
                    min(
                        self.args.archive_parent_sample - len(strong),
                        len(remainder),
                    ),
                ),
            ]
        selected_sets = [set(layout.medium_key) for layout in selected]
        while archive_candidates and len(selected) < self.args.population:
            tie_breaks = [self.rng.random() for _ in archive_candidates]

            def diversity_score(position):
                layout = archive_candidates[position]
                coordinates = set(layout.medium_key)
                distance = min(
                    EXACT_MEDIUMS - len(coordinates & existing)
                    for existing in selected_sets
                ) if selected_sets else EXACT_MEDIUMS
                return (
                    distance,
                    family_best[layout.medium_key].bound,
                    -self.parent_uses.get(layout.medium_key, 0),
                    tie_breaks[position],
                )

            best_position = max(
                range(len(archive_candidates)),
                key=diversity_score,
            )
            layout = archive_candidates.pop(best_position)
            selected.append(layout)
            selected_mediums.add(layout.medium_key)
            selected_sets.append(set(layout.medium_key))
        return selected

    def generate_offspring(
        self,
        population,
        generation,
        stagnation,
        maximum=None,
    ):
        candidates = {}
        maximum = self.args.offspring if maximum is None else maximum
        if maximum <= 0 or not population:
            return []
        strength = min(3, 1 + stagnation // 4)
        parent_order = list(population)
        self.rng.shuffle(parent_order)
        quota = max(8, math.ceil(maximum / len(parent_order)))
        for parent in parent_order:
            if len(candidates) >= maximum:
                break
            self.parent_uses[parent.medium_key] = (
                self.parent_uses.get(parent.medium_key, 0) + 1
            )
            donor_pool = [
                layout
                for layout in population
                if layout.medium_key != parent.medium_key
            ]
            donors = self.rng.sample(
                donor_pool,
                min(self.args.donors_per_parent, len(donor_pool)),
            )
            parent_evaluation = self.cache.get(parent.key)
            generated = self.mutator.generate(
                parent,
                donors,
                self.seen_keys | set(candidates),
                min(quota, maximum - len(candidates)),
                strength,
                None if parent_evaluation is None else parent_evaluation.equality_dual,
            )
            for layout in generated:
                candidates[layout.key] = layout

        print(
            f"generation {generation}: generated {len(candidates)} feasible "
            f"local destroy/repair layouts from {len(parent_order)} "
            f"diverse parents; strength={strength}, archive="
            f"{len(self.feasible_archive)}",
            flush=True,
        )
        return list(candidates.values())

    def sweep_actual_families(
        self,
        evaluations,
        generation,
        family_limit=None,
    ):
        family_limit = (
            self.args.substation_repair_families
            if family_limit is None
            else family_limit
        )
        if family_limit <= 0 or not evaluations:
            return []

        family_best = {}
        for evaluation in evaluations:
            medium_key = evaluation.layout.medium_key
            previous = family_best.get(medium_key)
            if previous is None or evaluation.bound > previous.bound:
                family_best[medium_key] = evaluation

        swept_results = []
        swept = 0
        for evaluation in sorted(
            family_best.values(),
            key=lambda record: record.bound,
            reverse=True,
        ):
            if evaluation.layout.medium_key in self.swept_mediums:
                continue
            print(
                f"generation {generation}: actual-bound substation sweep "
                f"for {evaluation.bound:.6f} medium family",
                flush=True,
            )
            swept_results.extend(self.sweep_substations(evaluation, generation))
            swept += 1
            if (
                swept >= family_limit
                or self.best.bound
                >= self.args.target - self.args.improvement_tolerance
            ):
                break
        return swept_results

    def run(self):
        self.evaluate_seeds()
        if self.best is None:
            raise RuntimeError("No coordinate seed had a feasible Stage-B root LP.")

        seed_evaluations = sorted(
            self.cache.values(),
            key=lambda evaluation: evaluation.bound,
            reverse=True,
        )
        swept = 0
        if self.args.substation_families > 0:
            for evaluation in seed_evaluations:
                if evaluation.layout.medium_key in self.swept_mediums:
                    continue
                self.sweep_substations(evaluation)
                swept += 1
                if (
                    swept >= self.args.substation_families
                    or self.best.bound
                    >= self.args.target - self.args.improvement_tolerance
                ):
                    break

        stagnation = 0
        previous_best = self.best.bound
        for generation in range(1, self.args.generations + 1):
            if self.best.bound >= self.args.target - self.args.improvement_tolerance:
                break
            global_layouts = self.global_generator.generate(
                self.args.global_offspring,
                set(self.feasible_archive),
                self.args.global_attempt_factor,
            )
            self.global_generator.add_sources(global_layouts)
            global_evaluations = self.evaluate_candidates(
                global_layouts,
                "global_mosaic",
                generation,
                maximum_exact=self.args.exact_per_generation,
            )

            # Improve within many independently seeded spatial islands.  This
            # is deliberately secondary to the global restart channel.
            population = self.population()
            local_layouts = self.generate_offspring(
                population,
                generation,
                stagnation,
                maximum=self.args.offspring,
            )
            self.global_generator.add_sources(local_layouts)
            local_evaluations = self.evaluate_candidates(
                local_layouts,
                "island_repair",
                generation,
                maximum_exact=self.args.exact_per_generation,
            )

            # Substations are now varied only after the medium family has a
            # measured Stage-B result.  No dual-ranked speculative variants.
            global_sweep_count = max(
                1,
                3 * self.args.substation_repair_families // 4,
            ) if self.args.substation_repair_families else 0
            self.sweep_actual_families(
                global_evaluations,
                generation,
                global_sweep_count,
            )
            self.sweep_actual_families(
                local_evaluations,
                generation,
                self.args.substation_repair_families - global_sweep_count,
            )

            if self.best.bound > previous_best + self.args.improvement_tolerance:
                previous_best = self.best.bound
                stagnation = 0
            else:
                stagnation += 1
            print(
                f"generation {generation} complete: exact evaluations="
                f"{self.evaluation_number}, cached={len(self.cache)}, "
                f"best={self.best.bound:.6f}, stagnation={stagnation}",
                flush=True,
            )

        print(
            f"coordinate search complete: evaluations={self.evaluation_number}; "
            f"best={self.best.bound:.6f}; target={self.args.target:.6f}; "
            f"output={self.folder}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=8316.0)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--substation-families", type=int, default=12)
    parser.add_argument("--generations", type=int, default=10_000)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--layouts-per-medium-family", type=int, default=2)
    parser.add_argument("--global-source-layouts", type=int, default=3_000)
    parser.add_argument("--global-offspring", type=int, default=1_200)
    parser.add_argument("--global-attempt-factor", type=int, default=100)
    parser.add_argument("--global-min-seed-changes", type=int, default=4)
    parser.add_argument("--island-bin-size", type=int, default=8)
    parser.add_argument("--offspring", type=int, default=400)
    # Stage B is cheap relative to generating valid coordinate layouts.  The
    # dual envelope is retained only for mathematically safe pruning; its
    # ranking proved too weak to choose a small sample reliably, so the
    # default cap deliberately exceeds a normal generation.
    parser.add_argument("--exact-per-generation", type=int, default=2_000)
    parser.add_argument("--evaluation-batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lp-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-dual-cuts", type=int, default=512)
    parser.add_argument("--local-parents", type=int, default=8)
    parser.add_argument("--local-radius", type=int, default=4)
    parser.add_argument("--archive-size", type=int, default=30_000)
    parser.add_argument("--archive-parent-sample", type=int, default=2_000)
    parser.add_argument("--donors-per-parent", type=int, default=6)
    parser.add_argument("--repair-pool-per-color", type=int, default=72)
    parser.add_argument("--repair-jobs-per-parent", type=int, default=14)
    parser.add_argument("--substation-repair-families", type=int, default=8)
    parser.add_argument("--improvement-tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2909)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for name in (
        "seed_count",
        "population",
        "offspring",
        "exact_per_generation",
        "evaluation_batch",
        "workers",
        "local_parents",
        "local_radius",
        "archive_size",
        "archive_parent_sample",
        "donors_per_parent",
        "repair_pool_per_color",
        "repair_jobs_per_parent",
        "global_source_layouts",
        "global_offspring",
        "global_attempt_factor",
        "island_bin_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (
        args.substation_families < 0
        or args.substation_repair_families < 0
        or not 0 <= args.global_min_seed_changes <= EXACT_MEDIUMS
        or args.generations < 0
    ):
        raise ValueError("Substation-family and generation counts must be nonnegative")

    parameters.GRID_SIZE = GRID
    build_start = time.perf_counter()
    checker = CoordinateStageAFeasibility()
    seeds = discover_seed_layouts(checker, args.seed_count)
    print(
        f"coordinate Stage A ready in {time.perf_counter() - build_start:.2f}s: "
        f"24 integer coordinates, {len(seeds)} valid distinct seeds",
        flush=True,
    )
    if not seeds:
        raise ValueError("No valid coordinate seed layouts were discovered.")
    for number, (layout, recorded_bound, path) in enumerate(seeds[:10]):
        print(
            f"seed {number}: recorded={recorded_bound:.6f} "
            f"top={layout.top_substation} inner={layout.inner_substation} "
            f"colors={np.bincount([medium_color(p) for p in layout.medium_poles], minlength=4).tolist()} "
            f"path={path}",
            flush=True,
        )

    if args.validate_only:
        print("coordinate validation complete; no optimization requested", flush=True)
        return

    global_layouts = discover_global_layout_pool(
        checker,
        seeds,
        args.global_source_layouts,
    )
    print(
        f"loaded {len(global_layouts)} distinct historical coordinate "
        "families for global mosaics",
        flush=True,
    )
    search = CoordinateSearch(args, checker, seeds, global_layouts)
    try:
        search.run()
    finally:
        search.close()


if __name__ == "__main__":
    main()
