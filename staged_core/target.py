"""Target-directed free-coordinate search for an exact 8316 packing.

The ordinary Stage-B power LP is a useful bound, but it smooths together the
two packing requirements.  At the 8316 target the geometry is much sharper:
198 3x3 solar panels and 168 2x2 accumulators must exactly cover every tile
not occupied by the 5+10 electrical network and the fixed 4x4 roboport.

This search scores networks by the minimum fractional exact-cover defect for
that target.  It explores coordinated chain, branch, shear, and basin-splice
moves that can cross a valley in the ordinary power bound.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import combinations
import math
from pathlib import Path
import random
import threading
import time
from types import SimpleNamespace

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from staged_core.history import (
    DEFAULT_COVERAGE_ROOT,
    candidate_layouts,
    select_basins,
)
from staged_core.coordinate import ExactStageBLP
from staged_core.network import (
    DD,
    EXACT_MEDIUMS,
    EXACT_SUBSTATIONS,
    GRID,
    CoordinateDestroyRepair,
    FreeCoordinateLayout,
    FreePeriodicOracle,
    index_coordinate,
    write_binary_solution,
    write_model_semantics,
)


TARGET_SOLAR = 198
TARGET_ACCUMULATORS = 168
TARGET_POWER = 8316.0
ROBOT_ROOT = (23, 23)
ROBOT_SIZE = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-root",
        type=Path,
        default=DEFAULT_COVERAGE_ROOT,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-seed-bound", type=float, default=8310.0)
    parser.add_argument("--basin-separation", type=int, default=3)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--parents", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=768)
    parser.add_argument("--per-parent", type=int, default=320)
    parser.add_argument(
        "--generations",
        type=int,
        default=0,
        help="Generation cap for validation; zero runs until stopped/target.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def destroy_repair_args():
    """Arguments used by the established coordinate repair constructor."""
    return SimpleNamespace(
        translation_radius=8,
        coordinate_pool=700,
        seam_pool=240,
        local_radius=5,
        shallow_edge_bonus=14.0,
        axial_edge_penalty=6.0,
        target_axial_edges=4,
        target_shallow_edges=4,
        shallow_excess_penalty=10.0,
        moderate_edge_penalty=6.0,
        deep_edge_penalty=16.0,
        degree_two_bonus=8.0,
        leaf_penalty=6.0,
        construction_noise=35.0,
        construction_elite=18,
        dual_guidance_weight=0.0,
        exact_repair_pool=18,
        repair_steps=28,
        repair_temperature=250.0,
        local_destroy=3,
        discovery_min_changes=4,
        discovery_max_changes=9,
        walk_step_factor=3,
        snake_min_poles=3,
        snake_max_poles=6,
        snake_attempts=3,
        improvement_move_limit=6,
        construction_attempts=10,
        basin_snake_probability=0.75,
        improvement_snake_probability=0.65,
        discovery_snake_probability=0.75,
        translation_probability=0.0,
        walk_probability=0.75,
        scaffold_probability=0.65,
    )


def placement_matrix(size: int) -> sp.csr_matrix:
    rows = []
    columns = []
    for root in range(DD):
        row, column = divmod(root, GRID)
        for row_offset in range(size):
            for column_offset in range(size):
                tile = (
                    ((row + row_offset) % GRID) * GRID
                    + (column + column_offset) % GRID
                )
                rows.append(tile)
                columns.append(root)
    return sp.csr_matrix(
        (
            np.ones(len(rows)),
            (rows, columns),
        ),
        shape=(DD, DD),
    )


SOLAR_PLACEMENT = placement_matrix(3)
ACCUMULATOR_PLACEMENT = placement_matrix(2)
ROBOT_INDICES = tuple(
    (ROBOT_ROOT[0] + row_offset) * GRID
    + ROBOT_ROOT[1]
    + column_offset
    for row_offset in range(ROBOT_SIZE)
    for column_offset in range(ROBOT_SIZE)
)


def mask_indices(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def packing_parameters(
    layout: FreeCoordinateLayout,
    oracle: FreePeriodicOracle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupied = 0
    for index in layout.substation_indices:
        occupied |= oracle.substation_footprints[index]
    for index in layout.medium_indices:
        occupied |= oracle.medium_footprints[index]
    for index in ROBOT_INDICES:
        occupied |= 1 << index

    free = np.ones(DD)
    free[np.fromiter(mask_indices(occupied), dtype=int)] = 0
    if int(np.sum(1 - free)) != 46:
        raise ValueError("The target network must occupy exactly 46 tiles.")

    electric = oracle.electric_union(layout)
    electric_vector = np.zeros(DD)
    electric_vector[
        np.fromiter(mask_indices(electric), dtype=int)
    ] = 1
    solar_eligible = (
        np.asarray(SOLAR_PLACEMENT.T @ electric_vector).ravel() > 0
    )
    accumulator_eligible = (
        np.asarray(
            ACCUMULATOR_PLACEMENT.T @ electric_vector
        ).ravel()
        > 0
    )
    return free, solar_eligible, accumulator_eligible


class TargetDefectLP:
    """Reusable LP measuring distance from the exact 198+168 cover."""

    def __init__(self):
        identity = sp.eye(DD, format="csr")
        cover = sp.hstack(
            [
                SOLAR_PLACEMENT,
                ACCUMULATOR_PLACEMENT,
                identity,
                -identity,
            ],
            format="csr",
        )
        counts = sp.lil_matrix((2, 4 * DD))
        counts[0, :DD] = 1
        counts[1, DD:2 * DD] = 1

        self.model = gp.Model("target_8316_defect_lp")
        self.model.Params.OutputFlag = 0
        self.model.Params.Threads = 1
        self.model.Params.Method = 2
        self.model.Params.Crossover = 0
        self.model.Params.Presolve = 2
        objective = np.zeros(4 * DD)
        objective[2 * DD:] = 1
        upper = np.full(4 * DD, GRB.INFINITY)
        upper[:2 * DD] = 1
        self.variables = self.model.addMVar(
            4 * DD,
            lb=0,
            ub=upper,
            obj=objective,
            vtype=GRB.CONTINUOUS,
        )
        self.tile_constraints = self.model.addMConstr(
            cover,
            self.variables,
            "=",
            np.ones(DD),
        )
        self.model.addMConstr(
            counts.tocsr(),
            self.variables,
            "=",
            np.asarray([TARGET_SOLAR, TARGET_ACCUMULATORS]),
        )
        self.model.ModelSense = GRB.MINIMIZE
        self.model.update()

    def evaluate(
        self,
        layout: FreeCoordinateLayout,
        oracle: FreePeriodicOracle,
    ) -> tuple[float, int, float]:
        free, solar_eligible, accumulator_eligible = packing_parameters(
            layout,
            oracle,
        )
        self.model.reset()
        self.variables[:DD].UB = solar_eligible.astype(float)
        self.variables[DD:2 * DD].UB = accumulator_eligible.astype(float)
        self.tile_constraints.RHS = free
        self.model.update()
        self.model.optimize()
        status = int(self.model.Status)
        runtime = float(self.model.Runtime)
        if status != GRB.OPTIMAL:
            return math.inf, status, runtime
        return float(self.model.ObjVal), status, runtime

    def close(self):
        self.model.dispose()


class ParallelTargetEvaluator:
    def __init__(self, oracle: FreePeriodicOracle, workers: int):
        self.oracle = oracle
        self.local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def _evaluate(self, layout):
        evaluator = getattr(self.local, "evaluator", None)
        if evaluator is None:
            evaluator = TargetDefectLP()
            self.local.evaluator = evaluator
        defect, status, runtime = evaluator.evaluate(layout, self.oracle)
        return layout, defect, status, runtime

    def evaluate(self, layouts):
        return list(self.executor.map(self._evaluate, layouts))

    def close(self):
        self.executor.shutdown(wait=True)


class ExactTargetPacking:
    """Binary exact-cover feasibility check used only after LP defect zero."""

    def __init__(self, workers: int):
        matrix = sp.vstack(
            [
                sp.hstack(
                    [SOLAR_PLACEMENT, ACCUMULATOR_PLACEMENT],
                    format="csr",
                ),
                sp.hstack(
                    [
                        sp.csr_matrix(np.ones((1, DD))),
                        sp.csr_matrix((1, DD)),
                    ],
                    format="csr",
                ),
                sp.hstack(
                    [
                        sp.csr_matrix((1, DD)),
                        sp.csr_matrix(np.ones((1, DD))),
                    ],
                    format="csr",
                ),
            ],
            format="csr",
        )
        self.model = gp.Model("target_8316_exact_cover")
        self.model.Params.OutputFlag = 1
        self.model.Params.Threads = workers
        self.model.Params.Presolve = 2
        self.model.Params.MIPFocus = 1
        self.variables = self.model.addMVar(
            2 * DD,
            lb=0,
            ub=1,
            obj=0,
            vtype=GRB.BINARY,
            name="packing",
        )
        rhs = np.concatenate(
            [
                np.ones(DD),
                [TARGET_SOLAR, TARGET_ACCUMULATORS],
            ]
        )
        self.constraints = self.model.addMConstr(
            matrix,
            self.variables,
            "=",
            rhs,
        )
        self.model.ModelSense = GRB.MINIMIZE
        self.model.update()

    def solve(self, layout, oracle):
        free, solar_eligible, accumulator_eligible = packing_parameters(
            layout,
            oracle,
        )
        self.model.reset()
        self.variables[:DD].UB = solar_eligible.astype(float)
        self.variables[DD:].UB = accumulator_eligible.astype(float)
        rhs = np.concatenate(
            [
                free,
                [TARGET_SOLAR, TARGET_ACCUMULATORS],
            ]
        )
        self.constraints.RHS = rhs
        self.model.update()
        self.model.optimize()
        if not self.model.SolCount:
            return None
        values = np.rint(self.variables.X).astype(int)
        return values[:DD], values[DD:]

    def close(self):
        self.model.dispose()


def shifted_index(index: int, row_shift: int, column_shift: int) -> int:
    row, column = divmod(index, GRID)
    return (
        ((row + row_shift) % GRID) * GRID
        + (column + column_shift) % GRID
    )


def replace_group(
    layout: FreeCoordinateLayout,
    kind: str,
    selected,
    replacements,
) -> FreeCoordinateLayout:
    if kind == "med":
        retained = set(layout.medium_indices) - set(selected)
        return FreeCoordinateLayout.create(
            layout.substation_indices,
            (*retained, *replacements),
        )
    retained = set(layout.substation_indices) - set(selected)
    return FreeCoordinateLayout.create(
        (*retained, *replacements),
        layout.medium_indices,
    )


def connected_subsets(layout, oracle, minimum=2, maximum=6):
    mediums = layout.medium_indices
    adjacency = {
        index: {
            other
            for other in mediums
            if other != index
            and oracle.pair_edge("med", index, "med", other) is not None
        }
        for index in mediums
    }
    for size in range(minimum, min(maximum, len(mediums)) + 1):
        for subset in combinations(mediums, size):
            selected = set(subset)
            reached = {subset[0]}
            frontier = [subset[0]]
            while frontier:
                current = frontier.pop()
                for neighbor in adjacency[current] & selected - reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
            if len(reached) == size:
                yield subset


def simple_paths(layout, oracle, maximum=7, limit=320):
    mediums = layout.medium_indices
    adjacency = {
        index: tuple(
            other
            for other in mediums
            if other != index
            and oracle.pair_edge("med", index, "med", other) is not None
        )
        for index in mediums
    }
    paths = set()

    def visit(path):
        if len(paths) >= limit:
            return
        if len(path) >= 3:
            forward = tuple(path)
            reverse = tuple(reversed(path))
            paths.add(min(forward, reverse))
        if len(path) >= maximum:
            return
        for neighbor in adjacency[path[-1]]:
            if neighbor not in path:
                visit([*path, neighbor])

    for start in mediums:
        visit([start])
        if len(paths) >= limit:
            break
    return tuple(paths)


def add_feasible(
    proposals,
    candidate,
    oracle,
    seen,
    cap,
):
    if (
        len(proposals) >= cap
        or candidate.key in seen
        or candidate.key in proposals
    ):
        return
    if oracle.diagnose(candidate).feasible:
        proposals[candidate.key] = candidate


def structured_candidates(
    parent,
    donors,
    oracle,
    rng,
    cap,
    radius,
    seen,
):
    proposals = {}
    offsets = [
        (row_shift, column_shift)
        for row_shift in range(-radius, radius + 1)
        for column_shift in range(-radius, radius + 1)
        if (
            (row_shift or column_shift)
            and max(abs(row_shift), abs(column_shift)) <= radius
        )
    ]
    rng.shuffle(offsets)

    # Translate connected medium-pole segments.  Moving a whole segment keeps
    # its internal skewed geometry and crosses distance 4+ in a useful way.
    subsets = list(connected_subsets(parent, oracle))
    rng.shuffle(subsets)
    for subset in subsets:
        for row_shift, column_shift in offsets:
            replacements = [
                shifted_index(index, row_shift, column_shift)
                for index in subset
            ]
            candidate = replace_group(
                parent,
                "med",
                subset,
                replacements,
            )
            add_feasible(
                proposals,
                candidate,
                oracle,
                seen,
                cap,
            )
            if len(proposals) >= cap:
                return tuple(proposals.values())

    # Shear a connected path perpendicular to its overall direction.  This
    # deliberately turns aligned runs into doglegs and snaking chains.
    paths = list(simple_paths(parent, oracle))
    rng.shuffle(paths)
    for path in paths:
        start_row, start_column = index_coordinate(path[0])
        end_row, end_column = index_coordinate(path[-1])
        row_span = end_row - start_row
        column_span = end_column - start_column
        perpendiculars = (
            ((0, 1), (0, -1))
            if abs(row_span) >= abs(column_span)
            else ((1, 0), (-1, 0))
        )
        for perpendicular in perpendiculars:
            levels = np.rint(
                np.linspace(-2, 2, len(path))
            ).astype(int)
            replacements = [
                shifted_index(
                    index,
                    int(level * perpendicular[0]),
                    int(level * perpendicular[1]),
                )
                for index, level in zip(path, levels)
            ]
            candidate = replace_group(
                parent,
                "med",
                path,
                replacements,
            )
            add_feasible(
                proposals,
                candidate,
                oracle,
                seen,
                cap,
            )
            if len(proposals) >= cap:
                return tuple(proposals.values())

    # Translate one substation with a small attached medium-pole branch.
    for substation in parent.substation_indices:
        attached = [
            medium
            for medium in parent.medium_indices
            if oracle.pair_edge(
                "sub",
                substation,
                "med",
                medium,
            )
            is not None
        ]
        for branch_size in range(2, min(4, len(attached)) + 1):
            branches = list(combinations(attached, branch_size))
            rng.shuffle(branches)
            for branch in branches:
                selected = (substation, *branch)
                for row_shift, column_shift in offsets[:24]:
                    new_substation = shifted_index(
                        substation,
                        row_shift,
                        column_shift,
                    )
                    new_mediums = [
                        shifted_index(
                            index,
                            row_shift,
                            column_shift,
                        )
                        for index in branch
                    ]
                    retained_substations = (
                        set(parent.substation_indices) - {substation}
                    )
                    retained_mediums = (
                        set(parent.medium_indices) - set(branch)
                    )
                    candidate = FreeCoordinateLayout.create(
                        (*retained_substations, new_substation),
                        (*retained_mediums, *new_mediums),
                    )
                    add_feasible(
                        proposals,
                        candidate,
                        oracle,
                        seen,
                        cap,
                    )
                    if len(proposals) >= cap:
                        return tuple(proposals.values())

    # Splice coordinate groups from structurally distinct good basins.
    if donors:
        for _ in range(1200):
            donor = rng.choice(donors)
            medium_count = rng.randint(3, 7)
            removed_mediums = rng.sample(
                list(parent.medium_indices),
                medium_count,
            )
            retained_mediums = (
                set(parent.medium_indices) - set(removed_mediums)
            )
            donor_mediums = [
                index
                for index in donor.medium_indices
                if index not in retained_mediums
            ]
            if len(donor_mediums) < medium_count:
                continue
            inserted_mediums = rng.sample(
                donor_mediums,
                medium_count,
            )

            substations = parent.substation_indices
            if rng.random() < 0.45:
                substation_count = rng.randint(1, 3)
                removed_substations = rng.sample(
                    list(parent.substation_indices),
                    substation_count,
                )
                retained_substations = (
                    set(parent.substation_indices)
                    - set(removed_substations)
                )
                donor_substations = [
                    index
                    for index in donor.substation_indices
                    if index not in retained_substations
                ]
                if len(donor_substations) < substation_count:
                    continue
                inserted_substations = rng.sample(
                    donor_substations,
                    substation_count,
                )
                substations = (
                    *retained_substations,
                    *inserted_substations,
                )
            candidate = FreeCoordinateLayout.create(
                substations,
                (*retained_mediums, *inserted_mediums),
            )
            add_feasible(
                proposals,
                candidate,
                oracle,
                seen,
                cap,
            )
            if len(proposals) >= cap:
                break
    return tuple(proposals.values())


def expanded_parent_candidates(
    parent,
    donors,
    oracle,
    random_seed,
    cap,
    radius,
    seen,
    guidance_dual=None,
    discovery_probability=0.35,
):
    """Generate one parent's macros with an independent repair engine."""
    rng = random.Random(random_seed)
    proposals = {
        candidate.key: candidate
        for candidate in structured_candidates(
            parent,
            donors,
            oracle,
            rng,
            max(1, cap // 2),
            radius,
            seen,
        )
    }
    destroy_repair = CoordinateDestroyRepair(
        oracle,
        destroy_repair_args(),
        rng,
    )
    attempts = 0
    while attempts < cap * 3 and len(proposals) < cap:
        attempts += 1
        donor = rng.choice(donors) if donors else None
        discovery = rng.random() < discovery_probability
        candidate = destroy_repair.generate(
            parent,
            donor,
            discovery=discovery,
            guidance_dual=guidance_dual,
            basin=not discovery,
            move_limit=6,
        )
        if (
            candidate is None
            or candidate.key in seen
            or candidate.key in proposals
        ):
            continue
        if oracle.diagnose(candidate).feasible:
            proposals[candidate.key] = candidate
    return tuple(proposals.values())


_PROCESS_ORACLE = None


def expanded_parent_candidates_process(task):
    """Process-pool entry point; each worker constructs its oracle once."""
    global _PROCESS_ORACLE
    if _PROCESS_ORACLE is None:
        _PROCESS_ORACLE = FreePeriodicOracle()
    if len(task) == 8:
        (
            parent,
            donors,
            random_seed,
            cap,
            radius,
            seen,
            guidance_dual,
            discovery_probability,
        ) = task
    elif len(task) == 7:
        (
            parent,
            donors,
            random_seed,
            cap,
            radius,
            seen,
            guidance_dual,
        ) = task
        discovery_probability = 0.35
    else:
        (
            parent,
            donors,
            random_seed,
            cap,
            radius,
            seen,
        ) = task
        guidance_dual = None
        discovery_probability = 0.35
    return expanded_parent_candidates(
        parent,
        donors,
        _PROCESS_ORACLE,
        random_seed,
        cap,
        radius,
        seen,
        guidance_dual,
        discovery_probability,
    )


@dataclass
class Record:
    layout: FreeCoordinateLayout
    defect: float
    stage_b_bound: float | None = None


def select_population(records, size, rng):
    best_by_key = {}
    for record in records:
        previous = best_by_key.get(record.layout.key)
        if previous is None or record.defect < previous.defect:
            best_by_key[record.layout.key] = record
    ranked = sorted(
        best_by_key.values(),
        key=lambda record: (
            record.defect,
            -(
                record.stage_b_bound
                if record.stage_b_bound is not None
                else -math.inf
            ),
        ),
    )
    selected = []
    for separation in (3, 2, 1, 0):
        for record in ranked:
            if record in selected:
                continue
            if separation and any(
                record.layout.relative_distance(other.layout) < separation
                for other in selected
            ):
                continue
            selected.append(record)
            if len(selected) >= size:
                return selected
    rng.shuffle(ranked)
    return selected[:size]


def write_target_packing(path, layout, solar, accumulators):
    network = layout.network_vector()
    with path.open("w") as handle:
        handle.write("# Exact target 8316 Stage-B packing\n")
        handle.write(
            "# Model semantics = v2 wire_centers=physical "
            "electric_coverage=periodic standalone_connectivity=True\n"
        )
        for index in range(DD):
            handle.write(f"x[{index}] {int(solar[index])}\n")
        for index in range(DD):
            handle.write(
                f"x[{DD + index}] {int(accumulators[index])}\n"
            )
        for index in range(DD):
            handle.write(
                f"x[{2 * DD + index}] {int(network[index])}\n"
            )
        robot = np.zeros(DD, dtype=int)
        robot[ROBOT_ROOT[0] * GRID + ROBOT_ROOT[1]] = 1
        for index in range(DD):
            handle.write(f"x[{3 * DD + index}] {int(robot[index])}\n")
        for index in range(DD):
            handle.write(
                f"x[{4 * DD + index}] "
                f"{int(network[DD + index])}\n"
            )
        handle.write(f"x[{5 * DD}] {TARGET_POWER}\n")


def main():
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8.")
    if min(
        args.population,
        args.parents,
        args.candidates,
        args.per_parent,
    ) <= 0:
        raise ValueError("Search counts must be positive.")
    if args.generations < 0:
        raise ValueError("--generations cannot be negative.")

    random_seed = (
        args.seed
        if args.seed is not None
        else time.time_ns() & 0x7FFF_FFFF
    )
    rng = random.Random(random_seed)
    oracle = FreePeriodicOracle()
    coverage_root = args.coverage_root.resolve()
    seeds = select_basins(
        candidate_layouts(
            coverage_root,
            args.minimum_seed_bound,
        ),
        args.basin_separation,
        0,
    )
    if not seeds:
        raise ValueError("No strong free-coordinate seed was found.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        args.output.resolve()
        if args.output is not None
        else coverage_root / f"{timestamp}_target_8316_lns"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_model_semantics(output, oracle)
    progress_path = output / "target_8316_progress.csv"
    progress_handle = progress_path.open("w", newline="", buffering=1)
    writer = csv.writer(progress_handle)
    writer.writerow(
        [
            "evaluation_id",
            "generation",
            "target_defect",
            "stage_b_bound",
            "distance_from_best",
            "runtime",
            "is_new_best",
            "substations",
            "medium_poles",
            "solution_path",
        ]
    )

    evaluator = ParallelTargetEvaluator(oracle, args.workers)
    generator_pool = ProcessPoolExecutor(max_workers=args.workers)
    exact_stage_b = ExactStageBLP(
        seeds[0][1].network_vector(),
        GRB.INFINITY,
    )
    exact_target = ExactTargetPacking(args.workers)
    try:
        seed_layouts = [entry[1] for entry in seeds]
        seed_results = evaluator.evaluate(seed_layouts)
        records = []
        evaluation_id = 0
        for (layout, defect, status, runtime), seed_entry in zip(
            seed_results,
            seeds,
        ):
            if status != GRB.OPTIMAL or not math.isfinite(defect):
                continue
            record = Record(
                layout=layout,
                defect=float(defect),
                stage_b_bound=float(seed_entry[0]),
            )
            records.append(record)
            writer.writerow(
                [
                    evaluation_id,
                    0,
                    defect,
                    seed_entry[0],
                    "",
                    runtime,
                    0,
                    layout.substations,
                    layout.medium_poles,
                    seed_entry[2],
                ]
            )
            evaluation_id += 1
            print(
                f"target seed bound={seed_entry[0]:.6f} "
                f"defect={defect:.9f} "
                f"distance={layout.relative_distance(seed_layouts[0])}",
                flush=True,
            )
        if not records:
            raise RuntimeError("No target-defect seed solved.")

        records.sort(key=lambda record: record.defect)
        best = records[0]
        best_bound, _, _, best_status = exact_stage_b.evaluate(best.layout)
        if best_status == GRB.OPTIMAL:
            best.stage_b_bound = float(best_bound)
        best_path = output / "best_target_8316_network.sol"
        write_binary_solution(
            best_path,
            best.layout,
            best.stage_b_bound or math.nan,
            f"target defect seed {best.defect:.9f}",
            oracle.diagnose(best.layout),
            oracle,
        )
        population = select_population(
            records,
            args.population,
            rng,
        )
        seen = {record.layout.key for record in records}
        print(
            f"TARGET-8316 START best_defect={best.defect:.9f} "
            f"stage_b={best.stage_b_bound:.6f} "
            f"seeds={len(records)} workers={args.workers} "
            f"random_seed={random_seed}",
            flush=True,
        )

        generation = 0
        stagnation = 0
        while True:
            generation += 1
            ranked_population = sorted(
                population,
                key=lambda record: record.defect,
            )
            elite_count = min(
                len(ranked_population),
                max(1, round(args.parents * 0.65)),
            )
            parents = ranked_population[:elite_count]
            remaining = [
                record
                for record in ranked_population[elite_count:]
            ]
            rng.shuffle(remaining)
            parents.extend(
                remaining[: max(0, args.parents - len(parents))]
            )
            radius = min(6, 3 + stagnation // 12)
            proposals = {}
            donor_layouts = [
                record.layout for record in ranked_population
            ]
            parent_cap = min(
                args.per_parent,
                max(
                    32,
                    math.ceil(
                        1.5 * args.candidates / max(1, len(parents))
                    ),
                ),
            )
            tasks = [
                (
                    parent.layout,
                    tuple(
                        layout
                        for layout in donor_layouts
                        if layout != parent.layout
                    ),
                    rng.randrange(2**31),
                    parent_cap,
                    radius,
                    frozenset(seen),
                )
                for parent in parents
            ]
            generated_groups = list(
                generator_pool.map(
                    expanded_parent_candidates_process,
                    tasks,
                )
            )
            for generated in generated_groups:
                for candidate in generated:
                    proposals[candidate.key] = candidate

            layouts = list(proposals.values())
            rng.shuffle(layouts)
            if len(layouts) > args.candidates:
                # Reserve half for large moves, then use a random sample to
                # avoid reintroducing a geometry-only hill climb.
                large = [
                    layout
                    for layout in layouts
                    if min(
                        layout.distance(parent.layout)
                        for parent in parents
                    )
                    >= 4
                ]
                rng.shuffle(large)
                selected = large[: args.candidates // 2]
                selected_keys = {layout.key for layout in selected}
                remainder = [
                    layout
                    for layout in layouts
                    if layout.key not in selected_keys
                ]
                rng.shuffle(remainder)
                layouts = (
                    selected
                    + remainder[: args.candidates - len(selected)]
                )
            if not layouts:
                stagnation += 1
                print(
                    f"target generation {generation}: no new exact-feasible "
                    f"macro layout radius={radius}",
                    flush=True,
                )
                continue

            seen.update(layout.key for layout in layouts)
            results = evaluator.evaluate(layouts)
            new_records = []
            improved_generation = False
            for layout, defect, status, runtime in results:
                if status != GRB.OPTIMAL or not math.isfinite(defect):
                    continue
                is_new_best = defect < best.defect - 1e-7
                stage_b_bound = None
                solution_path = ""
                if is_new_best:
                    (
                        stage_b_bound,
                        _,
                        _,
                        stage_b_status,
                    ) = exact_stage_b.evaluate(layout)
                    if stage_b_status != GRB.OPTIMAL:
                        stage_b_bound = math.nan
                    record = Record(
                        layout,
                        float(defect),
                        float(stage_b_bound),
                    )
                    best = record
                    improved_generation = True
                    numbered = output / (
                        f"incumbent_{evaluation_id:07d}_"
                        f"defect_{defect:.9f}_"
                        f"bound_{stage_b_bound:.6f}.sol"
                    )
                    write_binary_solution(
                        numbered,
                        layout,
                        stage_b_bound,
                        f"target exact-cover defect {defect:.9f}",
                        oracle.diagnose(layout),
                        oracle,
                    )
                    write_binary_solution(
                        best_path,
                        layout,
                        stage_b_bound,
                        f"target exact-cover defect {defect:.9f}",
                        oracle.diagnose(layout),
                        oracle,
                        required=False,
                    )
                    solution_path = str(numbered).replace("\\", "/")
                    print(
                        f"NEW TARGET-DEFECT BEST {defect:.9f} "
                        f"stage_b={stage_b_bound:.6f} "
                        f"generation={generation}",
                        flush=True,
                    )
                    print(
                        f"  substations={layout.substations}",
                        flush=True,
                    )
                    print(
                        f"  medium_poles={layout.medium_poles}",
                        flush=True,
                    )
                    if defect <= 1e-7:
                        print(
                            "LP TARGET DEFECT ZERO: starting exact binary "
                            "198+168 cover.",
                            flush=True,
                        )
                        packing = exact_target.solve(layout, oracle)
                        if packing is not None:
                            solar, accumulators = packing
                            write_target_packing(
                                output / "target_8316_stage_b.sol",
                                layout,
                                solar,
                                accumulators,
                            )
                            write_binary_solution(
                                output / "target_8316_stage_a.sol",
                                layout,
                                TARGET_POWER,
                                "exact 198 solar + 168 accumulators",
                                oracle.diagnose(layout),
                                oracle,
                                required=False,
                            )
                            print(
                                "TARGET 8316 EXACT PACKING FOUND "
                                f"output={output}",
                                flush=True,
                            )
                            return
                record = Record(
                    layout,
                    float(defect),
                    (
                        None
                        if stage_b_bound is None
                        else float(stage_b_bound)
                    ),
                )
                new_records.append(record)
                writer.writerow(
                    [
                        evaluation_id,
                        generation,
                        defect,
                        (
                            ""
                            if stage_b_bound is None
                            else stage_b_bound
                        ),
                        layout.distance(best.layout),
                        runtime,
                        int(is_new_best),
                        layout.substations,
                        layout.medium_poles,
                        solution_path,
                    ]
                )
                evaluation_id += 1

            population = select_population(
                [*population, *new_records],
                args.population,
                rng,
            )
            stagnation = 0 if improved_generation else stagnation + 1
            large_count = sum(
                min(
                    layout.distance(parent.layout)
                    for parent in parents
                )
                >= 4
                for layout in layouts
            )
            print(
                f"target generation {generation}: exact={len(new_records)} "
                f"large_moves={large_count}/{len(layouts)} "
                f"population={len(population)} "
                f"best_defect={best.defect:.9f} "
                f"best_stage_b={best.stage_b_bound:.6f} "
                f"stagnation={stagnation} radius={radius}",
                flush=True,
            )
            if args.generations and generation >= args.generations:
                print(
                    f"target validation stopped after {generation} "
                    f"generation(s); output={output}",
                    flush=True,
                )
                return
    finally:
        generator_pool.shutdown(wait=True, cancel_futures=True)
        evaluator.close()
        exact_target.close()
        exact_stage_b.model.dispose()
        progress_handle.close()


if __name__ == "__main__":
    main()
