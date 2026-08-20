"""Search networks by unsupported mass in an exact 198+168 fractional cover."""

from __future__ import annotations

import argparse
import ast
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import random
import threading
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from staged_core.history import (
    DEFAULT_COVERAGE_ROOT,
    candidate_layouts,
    read_header,
    select_basins,
)
from staged_core.coordinate import ExactStageBLP
from staged_core.network import (
    DD,
    FreeCoordinateLayout,
    FreePeriodicOracle,
    write_binary_solution,
    write_model_semantics,
)
from staged_core.target import (
    ACCUMULATOR_PLACEMENT,
    SOLAR_PLACEMENT,
    ExactTargetPacking,
    expanded_parent_candidates_process,
    packing_parameters,
    write_target_packing,
)


TARGET_SOLAR = 198
TARGET_ACCUMULATORS = 168


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-root",
        type=Path,
        default=DEFAULT_COVERAGE_ROOT,
    )
    parser.add_argument(
        "--seed-sol",
        type=Path,
        action="append",
        default=[],
        help=(
            "Explicit Stage-A .sol seed; repeat for multiple seeds. "
            "These seeds are added before the historical seed scan."
        ),
    )
    parser.add_argument(
        "--provided-seeds-only",
        action="store_true",
        help=(
            "Skip the coverage-root and historical scans and start only "
            "from --seed-sol layouts."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum-seed-bound", type=float, default=8310.0)
    parser.add_argument("--basin-separation", type=int, default=3)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--parents", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--per-parent", type=int, default=128)
    parser.add_argument("--base-radius", type=int, default=3)
    parser.add_argument("--solar-penalty", type=float, default=4.0)
    parser.add_argument("--penalty-seed-limit", type=int, default=192)
    parser.add_argument("--maximum-seed-penalty", type=float, default=8.0)
    parser.add_argument(
        "--guided-fraction",
        type=float,
        default=0.70,
        help="Fraction of each exact LP batch selected by LP guidance.",
    )
    parser.add_argument("--generations", type=int, default=0)
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help=(
            "Wall-clock budget; zero is unlimited. The current generation "
            "is completed before stopping."
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


class ExactCoverPenaltyLP:
    def __init__(self, solar_penalty):
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
        self.solar_penalty = solar_penalty
        self.model = gp.Model("target_exact_cover_penalty_lp")
        self.model.Params.OutputFlag = 0
        self.model.Params.Threads = 1
        self.model.Params.Method = 1
        self.model.Params.Presolve = 2
        self.variables = self.model.addMVar(
            2 * DD,
            lb=0,
            ub=1,
            vtype=GRB.CONTINUOUS,
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

    def evaluate(self, layout, oracle):
        free, solar_eligible, accumulator_eligible = packing_parameters(
            layout,
            oracle,
        )
        objective = np.concatenate(
            [
                self.solar_penalty
                * (~solar_eligible).astype(float),
                (~accumulator_eligible).astype(float),
            ]
        )
        self.model.reset()
        self.variables.Obj = objective
        self.constraints.RHS = np.concatenate(
            [
                free,
                [TARGET_SOLAR, TARGET_ACCUMULATORS],
            ]
        )
        self.model.update()
        self.model.optimize()
        status = int(self.model.Status)
        if status != GRB.OPTIMAL:
            return math.inf, status, float(self.model.Runtime)
        return (
            float(self.model.ObjVal),
            status,
            float(self.model.Runtime),
        )

    def evaluate_detailed(self, layout, oracle):
        """Return the optimum cover as well as its scalar penalty."""
        free, solar_eligible, accumulator_eligible = packing_parameters(
            layout,
            oracle,
        )
        objective = np.concatenate(
            [
                self.solar_penalty
                * (~solar_eligible).astype(float),
                (~accumulator_eligible).astype(float),
            ]
        )
        self.model.reset()
        self.variables.Obj = objective
        self.constraints.RHS = np.concatenate(
            [
                free,
                [TARGET_SOLAR, TARGET_ACCUMULATORS],
            ]
        )
        self.model.update()
        self.model.optimize()
        status = int(self.model.Status)
        if status != GRB.OPTIMAL:
            return (
                math.inf,
                status,
                float(self.model.Runtime),
                None,
            )
        return (
            float(self.model.ObjVal),
            status,
            float(self.model.Runtime),
            np.asarray(self.variables.X, dtype=float).copy(),
        )


class ParallelPenaltyEvaluator:
    def __init__(self, oracle, workers, solar_penalty):
        self.oracle = oracle
        self.solar_penalty = solar_penalty
        self.local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def _evaluate(self, layout):
        evaluator = getattr(self.local, "evaluator", None)
        if evaluator is None:
            evaluator = ExactCoverPenaltyLP(self.solar_penalty)
            self.local.evaluator = evaluator
        score, status, runtime = evaluator.evaluate(
            layout,
            self.oracle,
        )
        return layout, score, status, runtime

    def _evaluate_detailed(self, layout):
        evaluator = getattr(self.local, "evaluator", None)
        if evaluator is None:
            evaluator = ExactCoverPenaltyLP(self.solar_penalty)
            self.local.evaluator = evaluator
        score, status, runtime, cover = evaluator.evaluate_detailed(
            layout,
            self.oracle,
        )
        return layout, score, status, runtime, cover

    def evaluate(self, layouts):
        return list(self.executor.map(self._evaluate, layouts))

    def evaluate_detailed(self, layouts):
        return list(self.executor.map(self._evaluate_detailed, layouts))

    def close(self):
        self.executor.shutdown(wait=True)


def _mask_incidence(masks) -> sp.csr_matrix:
    """Sparse rows indicating the electric tiles reached by each root."""
    indices = []
    indptr = [0]
    for mask in masks:
        while mask:
            bit = mask & -mask
            indices.append(bit.bit_length() - 1)
            mask ^= bit
        indptr.append(len(indices))
    return sp.csr_matrix(
        (
            np.ones(len(indices), dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(DD, DD),
    )


@dataclass
class PenaltyProfile:
    cover: np.ndarray
    solar_mass: float
    accumulator_mass: float
    solar_root_count: int
    accumulator_root_count: int
    guidance_dual: np.ndarray


class PenaltyGuidance:
    """Turn an LP cover into move guidance for the coordinate repairer."""

    def __init__(self, oracle, solar_penalty):
        self.oracle = oracle
        self.solar_penalty = float(solar_penalty)
        self.substation_electric = _mask_incidence(
            oracle.substation_electric_masks
        )
        self.medium_electric = _mask_incidence(
            oracle.medium_electric_masks
        )

    def profile(self, layout, cover) -> PenaltyProfile:
        _, solar_eligible, accumulator_eligible = packing_parameters(
            layout,
            self.oracle,
        )
        solar = np.maximum(np.asarray(cover[:DD]), 0.0)
        accumulators = np.maximum(np.asarray(cover[DD:]), 0.0)
        unsupported_solar = solar * (~solar_eligible)
        unsupported_accumulators = (
            accumulators * (~accumulator_eligible)
        )
        solar_mass = float(np.sum(unsupported_solar))
        accumulator_mass = float(np.sum(unsupported_accumulators))

        # Each unsupported placement votes for every tile that could make it
        # electrically eligible.  Existing coordinate repair already knows
        # how to preserve coverage/connectivity/color/tileability; this vector
        # tells it which feasible coordinates attack the current LP defect.
        tile_pressure = np.asarray(
            SOLAR_PLACEMENT
            @ (self.solar_penalty * unsupported_solar)
            + ACCUMULATOR_PLACEMENT @ unsupported_accumulators
        ).ravel()
        desirability = np.concatenate(
            [
                np.asarray(
                    self.substation_electric @ tile_pressure
                ).ravel(),
                np.asarray(
                    self.medium_electric @ tile_pressure
                ).ravel(),
            ]
        )
        # CoordinateDestroyRepair interprets lower duals as more attractive.
        guidance_dual = -desirability.astype(float, copy=False)
        return PenaltyProfile(
            cover=np.asarray(cover, dtype=float),
            solar_mass=solar_mass,
            accumulator_mass=accumulator_mass,
            solar_root_count=int(np.count_nonzero(unsupported_solar > 1e-7)),
            accumulator_root_count=int(
                np.count_nonzero(unsupported_accumulators > 1e-7)
            ),
            guidance_dual=guidance_dual,
        )

    def fixed_cover_penalty(self, layout, profile) -> float:
        """Cheap ranking proxy before reoptimizing a candidate's cover LP."""
        _, solar_eligible, accumulator_eligible = packing_parameters(
            layout,
            self.oracle,
        )
        solar = profile.cover[:DD]
        accumulators = profile.cover[DD:]
        return float(
            self.solar_penalty * np.sum(solar[~solar_eligible])
            + np.sum(accumulators[~accumulator_eligible])
        )


@dataclass
class Record:
    layout: object
    penalty: float
    stage_b_bound: float | None = None
    profile: PenaltyProfile | None = None


def select_population(records, size):
    distinct = {}
    for record in records:
        previous = distinct.get(record.layout.key)
        if previous is None or record.penalty < previous.penalty:
            distinct[record.layout.key] = record
    ranked = sorted(
        distinct.values(),
        key=lambda record: (
            record.penalty,
            -(
                record.stage_b_bound
                if record.stage_b_bound is not None
                else -math.inf
            ),
        ),
    )
    # Keep a compact, unconditional penalty elite so neutral/equal-penalty
    # walks are not discarded merely because they are geometrically close.
    # The remaining slots enforce basin diversity.
    elite_slots = min(len(ranked), max(1, round(0.30 * size)))
    selected = ranked[:elite_slots]
    if len(selected) >= size:
        return selected[:size]
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
    return selected


def historical_penalty_seeds(
    coverage_root,
    maximum_penalty,
    limit,
):
    """Recover strong layouts even when no numbered .sol was written."""
    distinct = {}
    for path in coverage_root.rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or ())
                if not {
                    "unsupported_penalty",
                    "substations",
                    "medium_poles",
                }.issubset(fields):
                    continue
                for row in reader:
                    try:
                        penalty = float(row["unsupported_penalty"])
                        if (
                            not math.isfinite(penalty)
                            or penalty > maximum_penalty + 1e-9
                        ):
                            continue
                        layout = FreeCoordinateLayout.create(
                            ast.literal_eval(row["substations"]),
                            ast.literal_eval(row["medium_poles"]),
                        )
                    except (TypeError, ValueError, SyntaxError):
                        continue
                    try:
                        bound = float(row.get("stage_b_bound", ""))
                    except (TypeError, ValueError):
                        bound = math.nan
                    if not math.isfinite(bound):
                        bound = math.nan
                    previous = distinct.get(layout.key)
                    previous_bound = (
                        previous[1]
                        if previous is not None
                        and math.isfinite(previous[1])
                        else -math.inf
                    )
                    if (
                        previous is None
                        or penalty < previous[0] - 1e-9
                        or (
                            abs(penalty - previous[0]) <= 1e-9
                            and bound > previous_bound
                        )
                    ):
                        distinct[layout.key] = (
                            penalty,
                            bound,
                            layout,
                            path,
                        )
        except (OSError, UnicodeError, csv.Error):
            continue

    ranked = sorted(
        distinct.values(),
        key=lambda item: (
            item[0],
            -(item[1] if math.isfinite(item[1]) else -math.inf),
        ),
    )
    elite_slots = min(len(ranked), max(1, round(0.30 * limit)))
    selected = ranked[:elite_slots]
    if len(selected) >= limit:
        return selected[:limit]
    for separation in (4, 3, 2, 1, 0):
        for item in ranked:
            if item in selected:
                continue
            if separation and any(
                item[2].relative_distance(other[2]) < separation
                for other in selected
            ):
                continue
            selected.append(item)
            if len(selected) >= limit:
                return selected
    return selected


def main():
    args = parse_args()
    if not 1 <= args.workers <= 20:
        raise ValueError("--workers must be between 1 and 20.")
    if min(
        args.population,
        args.parents,
        args.candidates,
        args.per_parent,
    ) <= 0:
        raise ValueError("Search counts must be positive.")
    if (
        args.solar_penalty <= 0
        or args.generations < 0
        or args.seconds < 0
        or not 1 <= args.base_radius <= 6
        or args.penalty_seed_limit <= 0
        or args.maximum_seed_penalty <= 0
        or not 0.25 <= args.guided_fraction <= 0.90
    ):
        raise ValueError(
            "Penalty/search fractions must be positive and within bounds."
        )
    if args.provided_seeds_only and not args.seed_sol:
        raise ValueError("--provided-seeds-only requires --seed-sol.")

    random_seed = (
        args.seed
        if args.seed is not None
        else time.time_ns() & 0x7FFF_FFFF
    )
    rng = random.Random(random_seed)
    deadline = (
        math.inf
        if args.seconds == 0
        else time.monotonic() + args.seconds
    )
    coverage_root = args.coverage_root.resolve()
    seeds = []
    seed_keys = {seed[1].key for seed in seeds}
    for requested_path in args.seed_sol:
        path = requested_path.resolve()
        parsed = read_header(path)
        if parsed is None:
            raise ValueError(f"Could not read seed solution header: {path}")
        bound, layout = parsed
        if layout.key in seed_keys:
            continue
        seeds.append((bound, layout, path, 0))
        seed_keys.add(layout.key)
    history_added = 0
    if not args.provided_seeds_only:
        scanned = select_basins(
            candidate_layouts(
                coverage_root,
                args.minimum_seed_bound,
            ),
            args.basin_separation,
            0,
        )
        for bound, layout, path, nearest in scanned:
            if layout.key in seed_keys:
                continue
            seeds.append((bound, layout, path, nearest))
            seed_keys.add(layout.key)

        # Target-specific incumbents may deliberately cross below the
        # ordinary Stage-B cutoff.  Always retain them on broad restarts.
        for path in coverage_root.rglob(
            "best_exact_cover_penalty_network.sol"
        ):
            parsed = read_header(path)
            if parsed is None:
                continue
            bound, layout = parsed
            if layout.key in seed_keys:
                continue
            seeds.append((bound, layout, path, 0))
            seed_keys.add(layout.key)

        history = historical_penalty_seeds(
            coverage_root,
            args.maximum_seed_penalty,
            args.penalty_seed_limit,
        )
        for _, bound, layout, path in history:
            if layout.key in seed_keys:
                continue
            seeds.append((bound, layout, path, 0))
            seed_keys.add(layout.key)
            history_added += 1
    if not seeds:
        raise ValueError("No qualifying seed layouts.")
    print(
        f"penalty seed scan: total={len(seeds)} "
        f"historical_progress_added={history_added}",
        flush=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        args.output.resolve()
        if args.output is not None
        else coverage_root / f"{timestamp}_exact_cover_penalty"
    )
    output.mkdir(parents=True, exist_ok=True)
    progress_handle = (output / "exact_cover_penalty_progress.csv").open(
        "w",
        newline="",
        buffering=1,
    )
    writer = csv.writer(progress_handle)
    writer.writerow(
        [
            "evaluation_id",
            "generation",
            "unsupported_penalty",
            "unsupported_solar_mass",
            "unsupported_accumulator_mass",
            "unsupported_solar_roots",
            "unsupported_accumulator_roots",
            "stage_b_bound",
            "distance_from_best",
            "runtime",
            "is_new_best",
            "substations",
            "medium_poles",
            "solution_path",
        ]
    )

    oracle = FreePeriodicOracle()
    write_model_semantics(output, oracle)
    guidance = PenaltyGuidance(oracle, args.solar_penalty)
    evaluator = ParallelPenaltyEvaluator(
        oracle,
        args.workers,
        args.solar_penalty,
    )
    generator_pool = ProcessPoolExecutor(max_workers=args.workers)
    exact_stage_b = ExactStageBLP(
        seeds[0][1].network_vector(),
        GRB.INFINITY,
        periodic_electric_coverage=oracle.true_periodic_coverage,
    )
    exact_target = ExactTargetPacking(args.workers)
    try:
        seed_results = evaluator.evaluate_detailed(
            [seed[1] for seed in seeds]
        )
        records = []
        evaluation_id = 0
        for (
            layout,
            penalty,
            status,
            runtime,
            cover,
        ), (bound, _, path, _) in zip(seed_results, seeds):
            if (
                status != GRB.OPTIMAL
                or not math.isfinite(penalty)
                or cover is None
            ):
                continue
            profile = guidance.profile(layout, cover)
            stage_b_bound = (
                float(bound) if math.isfinite(bound) else None
            )
            record = Record(
                layout,
                float(penalty),
                stage_b_bound,
                profile,
            )
            records.append(record)
            writer.writerow(
                [
                    evaluation_id,
                    0,
                    penalty,
                    profile.solar_mass,
                    profile.accumulator_mass,
                    profile.solar_root_count,
                    profile.accumulator_root_count,
                    "" if stage_b_bound is None else stage_b_bound,
                    "",
                    runtime,
                    0,
                    layout.substations,
                    layout.medium_poles,
                    path,
                ]
            )
            evaluation_id += 1
            print(
                f"penalty seed stage_b="
                f"{stage_b_bound if stage_b_bound is not None else 'unknown'} "
                f"unsupported={penalty:.9f} "
                f"solar={profile.solar_mass:.6f} "
                f"accumulators={profile.accumulator_mass:.6f}",
                flush=True,
            )
        if not records:
            raise RuntimeError("No seed exact-cover penalty LP solved.")
        records.sort(key=lambda record: record.penalty)
        best = records[0]
        population = select_population(records, args.population)
        seen = {record.layout.key for record in records}
        write_binary_solution(
            output / "best_exact_cover_penalty_network.sol",
            best.layout,
            best.stage_b_bound or math.nan,
            f"exact-cover unsupported penalty {best.penalty:.9f}",
            oracle.diagnose(best.layout),
            oracle,
        )
        print(
            f"EXACT-COVER-PENALTY START best={best.penalty:.9f} "
            f"stage_b="
            f"{best.stage_b_bound if best.stage_b_bound is not None else 'unknown'} "
            f"workers={args.workers} seeds={len(records)} "
            f"random_seed={random_seed}",
            flush=True,
        )

        generation = 0
        stagnation = 0
        while True:
            if time.monotonic() >= deadline:
                print(
                    f"PENALTY TIME BUDGET reached before generation "
                    f"{generation + 1}",
                    flush=True,
                )
                return
            generation += 1
            ranked = sorted(
                population,
                key=lambda record: record.penalty,
            )
            elite_count = min(
                len(ranked),
                max(1, round(args.parents * 0.65)),
            )
            parents = ranked[:elite_count]
            remainder = ranked[elite_count:]
            rng.shuffle(remainder)
            parents.extend(
                remainder[: max(0, args.parents - len(parents))]
            )
            donor_layouts = [record.layout for record in ranked]
            radius = min(
                6,
                args.base_radius + stagnation // 10,
            )
            parent_cap = min(
                args.per_parent,
                max(
                    32,
                    math.ceil(
                        1.5 * args.candidates / len(parents)
                    ),
                ),
            )
            tasks = [
                (
                    parent.layout,
                    tuple(
                        donor
                        for donor in donor_layouts
                        if donor != parent.layout
                    ),
                    rng.randrange(2**31),
                    parent_cap,
                    radius,
                    frozenset(seen),
                    (
                        None
                        if parent.profile is None
                        else parent.profile.guidance_dual
                    ),
                )
                for parent in parents
            ]
            generated = generator_pool.map(
                expanded_parent_candidates_process,
                tasks,
            )
            proposals = {}
            proxy_scores = {}
            for parent, group in zip(parents, generated):
                for candidate in group:
                    proposals[candidate.key] = candidate
                    if parent.profile is not None:
                        proxy = guidance.fixed_cover_penalty(
                            candidate,
                            parent.profile,
                        )
                        previous = proxy_scores.get(candidate.key)
                        if previous is None or proxy < previous:
                            proxy_scores[candidate.key] = proxy
            layouts = list(proposals.values())
            rng.shuffle(layouts)
            if len(layouts) > args.candidates:
                guided_count = round(
                    args.guided_fraction * args.candidates
                )
                ranked_guided = sorted(
                    layouts,
                    key=lambda layout: (
                        proxy_scores.get(layout.key, math.inf),
                        rng.random(),
                    ),
                )
                layouts = ranked_guided[:guided_count]
                selected = {layout.key for layout in layouts}
                rest = [
                    layout
                    for layout in ranked_guided[guided_count:]
                    if layout.key not in selected
                ]

                # Keep a deliberate basin-crossing slice.  This prevents the
                # exact proxy from collapsing the search onto one LP basis.
                discovery_count = (
                    args.candidates - guided_count
                ) * 2 // 3
                distant = [
                    layout
                    for layout in rest
                    if min(
                        layout.distance(parent.layout)
                        for parent in parents
                    )
                    >= 4
                ]
                rng.shuffle(distant)
                layouts.extend(distant[:discovery_count])
                selected.update(layout.key for layout in layouts)
                rest = [
                    layout
                    for layout in rest
                    if layout.key not in selected
                ]
                rng.shuffle(rest)
                layouts.extend(
                    rest[: args.candidates - len(layouts)]
                )
            if not layouts:
                stagnation += 1
                continue
            seen.update(layout.key for layout in layouts)
            results = evaluator.evaluate_detailed(layouts)
            new_records = []
            improved = False
            for layout, penalty, status, runtime, cover in results:
                if (
                    status != GRB.OPTIMAL
                    or not math.isfinite(penalty)
                    or cover is None
                ):
                    continue
                profile = guidance.profile(layout, cover)
                is_new_best = penalty < best.penalty - 1e-7
                stage_b_bound = None
                solution_path = ""
                if is_new_best:
                    (
                        stage_b_bound,
                        _,
                        _,
                        bound_status,
                    ) = exact_stage_b.evaluate(layout)
                    if bound_status != GRB.OPTIMAL:
                        stage_b_bound = math.nan
                    best = Record(
                        layout,
                        float(penalty),
                        float(stage_b_bound),
                        profile,
                    )
                    improved = True
                    numbered = output / (
                        f"incumbent_{evaluation_id:07d}_"
                        f"penalty_{penalty:.9f}_"
                        f"bound_{stage_b_bound:.6f}.sol"
                    )
                    write_binary_solution(
                        numbered,
                        layout,
                        stage_b_bound,
                        f"exact-cover unsupported penalty {penalty:.9f}",
                        oracle.diagnose(layout),
                        oracle,
                    )
                    write_binary_solution(
                        output
                        / "best_exact_cover_penalty_network.sol",
                        layout,
                        stage_b_bound,
                        f"exact-cover unsupported penalty {penalty:.9f}",
                        oracle.diagnose(layout),
                        oracle,
                        required=False,
                    )
                    solution_path = str(numbered)
                    print(
                        f"NEW EXACT-COVER PENALTY BEST "
                        f"{penalty:.9f} stage_b={stage_b_bound:.6f} "
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
                # Every zero-penalty LP basis needs a binary test.  Testing
                # only the first zero would incorrectly discard later zero
                # basins when the first fractional cover is not integral.
                if penalty <= 1e-7:
                    print(
                        "ZERO PENALTY: running exact binary 8316 test",
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
                            8316.0,
                            "exact-cover penalty zero, exact packing",
                            oracle.diagnose(layout),
                            oracle,
                            required=False,
                        )
                        print(
                            f"TARGET 8316 FOUND output={output}",
                            flush=True,
                        )
                        return
                    print(
                        "ZERO PENALTY was fractional-only; continuing",
                        flush=True,
                    )
                new_records.append(
                    Record(
                        layout,
                        float(penalty),
                        (
                            None
                            if stage_b_bound is None
                            else float(stage_b_bound)
                        ),
                        profile,
                    )
                )
                writer.writerow(
                    [
                        evaluation_id,
                        generation,
                        penalty,
                        profile.solar_mass,
                        profile.accumulator_mass,
                        profile.solar_root_count,
                        profile.accumulator_root_count,
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
            )
            stagnation = 0 if improved else stagnation + 1
            print(
                f"penalty generation {generation}: "
                f"exact={len(new_records)} "
                f"best={best.penalty:.9f} "
                f"solar={best.profile.solar_mass:.6f} "
                f"accumulators={best.profile.accumulator_mass:.6f} "
                f"stage_b="
                f"{best.stage_b_bound if best.stage_b_bound is not None else 'unknown'} "
                f"stagnation={stagnation} radius={radius}",
                flush=True,
            )
            if args.generations and generation >= args.generations:
                return
            if time.monotonic() >= deadline:
                print(
                    f"PENALTY TIME BUDGET reached after generation "
                    f"{generation}",
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
