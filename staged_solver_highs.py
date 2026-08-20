"""HiGHS staged A+B search over network quality and packability.

Stage A proposes physically valid periodic 5-substation, 10-medium-pole
networks.  Stage B evaluates every proposal twice: once with the continuous
power bound and once with the unsupported-cover penalty for an exact
198-solar, 168-accumulator packing.  Penalty is guidance, never a hard filter.

The durable archive retains the Pareto front, independent objective elites,
and distance-separated basin representatives.  New basin representatives are
recycled as parents automatically.  When the penalty reaches zero, a binary
Stage-B model verifies and saves the complete packing.
"""

from __future__ import annotations

import argparse
import ast
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

from staged_core.history_highs import (
    DEFAULT_COVERAGE_ROOT,
    REPRESENTATIVE_NAMES,
    read_header,
)
from staged_core.highs_evaluators import (
    ExactTargetPackingHighs,
    INFEASIBLE,
    OPTIMAL,
    ParallelPenaltyEvaluatorHighs,
    ParallelStageBEvaluatorHighs,
)
from staged_core.network_highs import (
    FreeCoordinateLayout,
    FreePeriodicOracle,
    model_semantics,
    progress_matches_model_semantics,
    write_binary_solution,
    write_model_semantics,
)
from staged_core.target_highs import (
    expanded_parent_candidates_process,
    write_target_packing,
)


# Solver setting. Change this value directly, or override it with --workers.
# THREAD COUNT: this is the maximum number of simultaneous worker processes.
thread_count = 20


PENALTY_SOURCE = re.compile(
    r"^# Source = exact-cover unsupported penalty ([0-9.eE+-]+)"
)
TRACKED_SLACKS = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
EXPORT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = EXPORT_ROOT / "results"
PROGRESS_COLUMNS = (
    "evaluation_id",
    "generation",
    "phase",
    "stage_b_bound",
    "unsupported_penalty",
    "nearest_basin_distance",
    "is_pareto",
    "is_basin_representative",
    "geometry_signature",
    "substations",
    "medium_poles",
    "source",
    "solution_path",
)


def portable_source_path(path):
    """Describe a seed without recording a user-specific absolute path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(EXPORT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel HiGHS Pareto search for high-bound, low-penalty, "
            "structurally diverse 5+10 networks."
        )
    )
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
        help="Explicit seed solution; may be repeated.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=thread_count,
        help=(
            "Total active worker budget. Proposal generation uses this many "
            "processes; concurrent bound/penalty evaluation splits it."
        ),
    )
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--parents", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=640)
    parser.add_argument("--per-parent", type=int, default=96)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument(
        "--discovery-share",
        type=float,
        default=0.60,
        help="Share of parent tasks using aggressive basin discovery.",
    )
    parser.add_argument("--basin-separation", type=int, default=4)
    parser.add_argument("--basin-archive", type=int, default=256)
    parser.add_argument("--donors", type=int, default=64)
    parser.add_argument("--minimum-seed-bound", type=float, default=8295.0)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument(
        "--seed-scan-limit",
        type=int,
        default=4000,
        help="Maximum historical candidates validated under current semantics.",
    )
    parser.add_argument("--solar-penalty", type=float, default=4.0)
    parser.add_argument(
        "--generations",
        type=int,
        default=0,
        help=(
            "Zero runs until interrupted or, unless "
            "--continue-after-target is set, an exact target is found."
        ),
    )
    parser.add_argument(
        "--continue-after-target",
        action="store_true",
        help=(
            "Keep searching after exact 8316 packings are found; every "
            "distinct exact packing is persisted under targets/."
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@dataclass(frozen=True)
class Record:
    layout: FreeCoordinateLayout
    penalty: float
    bound: float
    generation: int = 0
    source: str = ""


def layout_token(layout: FreeCoordinateLayout) -> str:
    return hashlib.blake2s(
        repr(layout.key).encode("ascii"),
        digest_size=6,
    ).hexdigest()


def geometry_signature(
    record: Record,
    oracle: FreePeriodicOracle,
) -> tuple[int, ...]:
    return oracle.medium_geometry(record.layout).signature


def penalty_from_header(path: Path) -> float | None:
    try:
        with path.open(errors="replace") as handle:
            for _ in range(16):
                line = handle.readline()
                if not line:
                    break
                match = PENALTY_SOURCE.match(line.strip())
                if match:
                    return float(match.group(1))
    except (OSError, ValueError):
        pass
    return None


def _read_seed(path: Path):
    parsed = read_header(path)
    if parsed is None:
        return None
    bound, layout = parsed
    if not math.isfinite(bound):
        return None
    return float(bound), layout, path


def historical_seed_candidates(
    coverage_root: Path,
    explicit_paths: list[Path],
    minimum_bound: float,
    scan_limit: int,
    desired_count: int,
    oracle: FreePeriodicOracle,
):
    """Find physical-valid historical seeds without trusting their scores."""
    print(
        f"seed scan: traversing {coverage_root} (one pass; progress every "
        f"5000 .sol files)",
        flush=True,
    )
    explicit = []
    for raw_path in explicit_paths:
        path = raw_path.resolve()
        parsed = _read_seed(path)
        if parsed is None:
            raise ValueError(f"Could not read seed solution: {path}")
        if not oracle.diagnose(parsed[1]).feasible:
            raise ValueError(
                f"Explicit seed is infeasible under the current physical/"
                f"periodic model: {path}"
            )
        explicit.append(parsed)

    by_key = {layout.key: item for item in explicit for layout in (item[1],)}
    historical = {}
    files_seen = 0
    headers_read = 0
    for path in coverage_root.rglob("*.sol"):
        files_seen += 1
        if files_seen % 5000 == 0:
            print(
                f"seed scan: files={files_seen} eligible_headers={headers_read}",
                flush=True,
            )
        name = path.name
        if not (
            name in REPRESENTATIVE_NAMES
            or name.startswith("seed_")
            or name.startswith("incumbent_")
            or name in {"best_penalty.sol", "best_bound.sol"}
            or path.parent.name
            in {"seed_solutions", "pareto", "basins", "above_8316"}
        ):
            continue
        parsed = _read_seed(path)
        if parsed is None or parsed[0] + 1e-9 < minimum_bound:
            continue
        headers_read += 1
        previous = historical.get(parsed[1].key)
        if previous is None or parsed[0] > previous[0]:
            historical[parsed[1].key] = parsed

    ranked = sorted(
        historical.values(),
        key=lambda item: item[0],
        reverse=True,
    )
    print(
        f"seed scan: traversal complete files={files_seen} "
        f"distinct_eligible={len(ranked)}; validating corrected semantics",
        flush=True,
    )
    examined = 0
    for item in ranked:
        if scan_limit and examined >= scan_limit:
            break
        examined += 1
        if examined % 100 == 0:
            print(
                f"seed scan: validated={examined}/{min(len(ranked), scan_limit or len(ranked))} "
                f"physical_valid={len(by_key)}",
                flush=True,
            )
        _, layout, _ = item
        if layout.key in by_key:
            continue
        if not oracle.diagnose(layout).feasible:
            continue
        by_key[layout.key] = item
        # More than eight shells per requested seed is ample for the later
        # distance/topology preselection, and avoids validating thousands of
        # near-duplicates from the same historical run.
        if len(by_key) >= max(desired_count * 8, 256):
            break
    valid = list(by_key.values())
    valid.sort(key=lambda item: item[0], reverse=True)
    print(
        f"seed scan: examined={examined} physical_valid={len(valid)} "
        f"explicit={len(explicit)} (historical bounds will be rescored)",
        flush=True,
    )
    return valid, {item[1].key for item in explicit}


def preselect_seed_candidates(
    candidates,
    explicit_keys,
    count,
    oracle,
):
    """Retain a high-bound core and a much larger diverse seed shell."""
    if not candidates:
        return []
    count = min(count, len(candidates))
    selected = []
    selected_keys = set()

    def add(item):
        if item[1].key in selected_keys or len(selected) >= count:
            return
        selected.append(item)
        selected_keys.add(item[1].key)

    for item in candidates:
        if item[1].key in explicit_keys:
            add(item)
    for item in candidates[: max(4, count // 4)]:
        add(item)

    # First preserve unseen topology signatures, then demand progressively
    # weaker coordinate separation until the requested seed count is filled.
    signatures = {
        oracle.medium_geometry(item[1]).signature for item in selected
    }
    for item in candidates:
        signature = oracle.medium_geometry(item[1]).signature
        if signature not in signatures:
            add(item)
            signatures.add(signature)
    for separation in (8, 6, 5, 4, 3, 2, 1, 0):
        for item in candidates:
            if item[1].key in selected_keys:
                continue
            if separation and any(
                item[1].relative_distance(other[1]) < separation
                for other in selected
            ):
                continue
            add(item)
            if len(selected) >= count:
                return selected
    return selected


def dominates(left: Record, right: Record) -> bool:
    return (
        left.penalty <= right.penalty + 1e-8
        and left.bound >= right.bound - 1e-8
        and (
            left.penalty < right.penalty - 1e-8
            or left.bound > right.bound + 1e-8
        )
    )


def pareto_front(records) -> list[Record]:
    """Compute the two-objective front in O(n log n)."""
    distinct = {}
    for record in records:
        previous = distinct.get(record.layout.key)
        if previous is None or dominates(record, previous):
            distinct[record.layout.key] = record
    ordered = sorted(
        distinct.values(),
        key=lambda record: (record.penalty, -record.bound),
    )
    front = []
    best_bound = -math.inf
    for record in ordered:
        if record.bound > best_bound + 1e-8:
            front.append(record)
            best_bound = record.bound
    return front


def _round_robin(streams):
    streams = [list(stream) for stream in streams if stream]
    positions = [0] * len(streams)
    while streams:
        next_streams = []
        next_positions = []
        for stream, position in zip(streams, positions):
            if position < len(stream):
                yield stream[position]
                position += 1
            if position < len(stream):
                next_streams.append(stream)
                next_positions.append(position)
        streams, positions = next_streams, next_positions


def basin_representatives(
    records,
    oracle,
    separation,
    maximum,
) -> list[Record]:
    values = list(records)
    if not values or maximum <= 0:
        return []
    front = pareto_front(values)
    by_geometry = {}
    for record in values:
        signature = geometry_signature(record, oracle)
        previous = by_geometry.get(signature)
        if previous is None or (
            record.bound,
            -record.penalty,
        ) > (previous.bound, -previous.penalty):
            by_geometry[signature] = record
    streams = (
        sorted(front, key=lambda record: (-record.bound, record.penalty)),
        sorted(values, key=lambda record: (-record.bound, record.penalty)),
        sorted(values, key=lambda record: (record.penalty, -record.bound)),
        sorted(
            by_geometry.values(),
            key=lambda record: (-record.bound, record.penalty),
        ),
    )
    priority = []
    keys = set()
    for record in _round_robin(streams):
        if record.layout.key not in keys:
            priority.append(record)
            keys.add(record.layout.key)

    selected = []
    for record in priority:
        if any(
            record.layout.relative_distance(other.layout) < separation
            for other in selected
        ):
            continue
        selected.append(record)
        if len(selected) >= maximum:
            break
    return selected


def _append_stream(selected, selected_keys, stream, quota, separation):
    added = 0
    for record in stream:
        if record.layout.key in selected_keys:
            continue
        if separation and any(
            record.layout.relative_distance(other.layout) < separation
            for other in selected
        ):
            continue
        selected.append(record)
        selected_keys.add(record.layout.key)
        added += 1
        if added >= quota:
            break


def select_population(records, basins, size) -> list[Record]:
    """Build a soft multi-stream population; no penalty cutoff is applied."""
    values = list(records)
    if len(values) <= size:
        return values
    front = pareto_front(values)
    penalty_ranked = sorted(
        values,
        key=lambda record: (record.penalty, -record.bound),
    )
    bound_ranked = sorted(
        values,
        key=lambda record: (-record.bound, record.penalty),
    )
    best_penalty = penalty_ranked[0].penalty
    band_stream = []
    band_keys = set()
    for slack in TRACKED_SLACKS[1:]:
        eligible = [
            record
            for record in values
            if record.penalty <= best_penalty + slack + 1e-8
        ]
        for record in sorted(
            eligible,
            key=lambda item: (-item.bound, item.penalty),
        )[: max(4, size // 12)]:
            if record.layout.key not in band_keys:
                band_stream.append(record)
                band_keys.add(record.layout.key)

    selected = []
    selected_keys = set()
    quotas = (
        (basins, max(1, round(size * 0.35))),
        (front, max(1, round(size * 0.25))),
        (bound_ranked, max(1, round(size * 0.15))),
        (penalty_ranked, max(1, round(size * 0.15))),
        (band_stream, max(1, round(size * 0.10))),
    )
    for separation in (3, 2, 1, 0):
        for stream, quota in quotas:
            remaining = size - len(selected)
            if remaining <= 0:
                return selected
            _append_stream(
                selected,
                selected_keys,
                stream,
                min(quota, remaining),
                separation,
            )
        if len(selected) >= size:
            return selected
    _append_stream(
        selected,
        selected_keys,
        _round_robin((basins, front, bound_ranked, penalty_ranked)),
        size - len(selected),
        0,
    )
    return selected


def choose_parents(
    population,
    basins,
    count,
    discovery_share,
    rng,
):
    """Return (record, discovery_probability) parent tasks."""
    count = min(count, len({record.layout.key for record in population + basins}))
    discovery_count = min(count, round(count * discovery_share))
    discovery_pool = list(basins)
    rng.shuffle(discovery_pool)
    # Keep some quality in the discovery stream without making it homogeneous.
    discovery_pool = list(
        _round_robin(
            (
                discovery_pool,
                sorted(basins, key=lambda r: (-r.bound, r.penalty)),
                sorted(basins, key=lambda r: (r.penalty, -r.bound)),
            )
        )
    )
    parents = []
    keys = set()
    for record in discovery_pool:
        if record.layout.key in keys:
            continue
        parents.append((record, 0.82))
        keys.add(record.layout.key)
        if len(parents) >= discovery_count:
            break

    exploit_stream = _round_robin(
        (
            pareto_front(population),
            sorted(population, key=lambda r: (-r.bound, r.penalty)),
            sorted(population, key=lambda r: (r.penalty, -r.bound)),
        )
    )
    for record in exploit_stream:
        if record.layout.key in keys:
            continue
        parents.append((record, 0.25))
        keys.add(record.layout.key)
        if len(parents) >= count:
            break
    return parents


def select_donors(population, basins, maximum):
    donors = []
    keys = set()
    streams = (
        basins,
        pareto_front(population),
        sorted(population, key=lambda r: (-r.bound, r.penalty)),
        sorted(population, key=lambda r: (r.penalty, -r.bound)),
    )
    for record in _round_robin(streams):
        if record.layout.key in keys:
            continue
        donors.append(record.layout)
        keys.add(record.layout.key)
        if len(donors) >= maximum:
            break
    return tuple(donors)


def metric_best(records, best_penalty):
    result = {}
    values = list(records)
    for slack in TRACKED_SLACKS:
        eligible = [
            record
            for record in values
            if record.penalty <= best_penalty + slack + 1e-8
        ]
        result[slack] = (
            max(eligible, key=lambda record: record.bound)
            if eligible
            else None
        )
    return result


def save_record(path, record, label, oracle, *, required=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_binary_solution(
        path,
        record.layout,
        record.bound,
        (
            f"Parallel HiGHS Pareto bound/penalty search; "
            f"penalty={record.penalty:.9f}; {label}"
        ),
        oracle.diagnose(record.layout),
        oracle,
        required=required,
    )
    return path


def _atomic_json(path: Path, payload):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_state(
    path,
    *,
    args,
    oracle,
    random_seed,
    generation,
    evaluation_id,
    stagnation,
):
    _atomic_json(
        path,
        {
            "version": 1,
            "random_seed": random_seed,
            "generation": generation,
            "evaluation_id": evaluation_id,
            "stagnation": stagnation,
            "solar_penalty": args.solar_penalty,
            "model_semantics": model_semantics(oracle),
            "updated": datetime.now().astimezone().isoformat(),
        },
    )


def load_state(path, args):
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if abs(float(state["solar_penalty"]) - args.solar_penalty) > 1e-12:
        raise ValueError(
            "Cannot resume with a different --solar-penalty; use a new output."
        )
    return state


def load_progress(path: Path) -> dict[tuple[int, ...], Record]:
    records = {}
    if not path.exists():
        return records
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {
            "stage_b_bound",
            "unsupported_penalty",
            "substations",
            "medium_poles",
        }.issubset(reader.fieldnames):
            raise ValueError(f"Unrecognized progress format: {path}")
        for row in reader:
            try:
                layout = FreeCoordinateLayout.create(
                    ast.literal_eval(row["substations"]),
                    ast.literal_eval(row["medium_poles"]),
                )
                record = Record(
                    layout,
                    float(row["unsupported_penalty"]),
                    float(row["stage_b_bound"]),
                    int(row.get("generation") or 0),
                    row.get("source") or "resume",
                )
            except (ValueError, TypeError, SyntaxError):
                continue
            if math.isfinite(record.penalty) and math.isfinite(record.bound):
                records[layout.key] = record
    return records


def _atomic_csv(path, columns, rows):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    os.replace(temporary, path)


def load_target_checks(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def refresh_exact_target_report(output: Path, checks: dict[str, dict]):
    exact = [
        item
        for item in checks.values()
        if item.get("status") == "exact"
    ]
    exact.sort(key=lambda item: float(item["bound"]), reverse=True)
    _atomic_csv(
        output / "exact_targets.csv",
        (
            "stage_b_bound",
            "unsupported_penalty",
            "generation",
            "token",
            "substations",
            "medium_poles",
            "stage_a_path",
            "stage_b_path",
        ),
        (
            (
                item["bound"],
                item["penalty"],
                item["generation"],
                item["token"],
                item["substations"],
                item["medium_poles"],
                item["stage_a_path"],
                item["stage_b_path"],
            )
            for item in exact
        ),
    )


def verify_zero_penalty_records(
    records,
    *,
    output,
    oracle,
    exact_target,
    checks,
):
    """Binary-test every previously unchecked zero-penalty Stage-A layout."""
    checks_path = output / "target_checks.json"
    terminal = {"exact", "fractional_only"}
    newly_exact = []
    zero_records = sorted(
        (
            record
            for record in records
            if record.penalty <= 1e-7
        ),
        key=lambda record: record.bound,
        reverse=True,
    )
    for record in zero_records:
        token = layout_token(record.layout)
        previous = checks.get(token)
        if previous is not None and previous.get("status") in terminal:
            continue
        print(
            f"ZERO PENALTY candidate token={token} "
            f"bound={record.bound:.6f}; running exact binary 198/168 test",
            flush=True,
        )
        packing = exact_target.solve(record.layout, oracle)
        solver_status = int(exact_target.status)
        base = {
            "token": token,
            "bound": record.bound,
            "penalty": record.penalty,
            "generation": record.generation,
            "substations": record.layout.substations,
            "medium_poles": record.layout.medium_poles,
            "solver_status": solver_status,
            "checked_at": datetime.now().astimezone().isoformat(),
        }
        if packing is None:
            base["status"] = (
                "fractional_only"
                if solver_status == int(INFEASIBLE)
                else "unresolved"
            )
            checks[token] = base
            _atomic_json(checks_path, checks)
            print(
                f"ZERO PENALTY token={token} was not integral "
                f"(status={solver_status})",
                flush=True,
            )
            continue

        solar, accumulators = packing
        target_dir = output / "targets"
        target_dir.mkdir(parents=True, exist_ok=True)
        stage_a = target_dir / (
            f"target_bound_{record.bound:.6f}_{token}_stage_a.sol"
        )
        stage_b = target_dir / (
            f"target_bound_{record.bound:.6f}_{token}_stage_b.sol"
        )
        write_target_packing(stage_b, record.layout, solar, accumulators)
        save_record(
            stage_a,
            record,
            "distinct zero-penalty exact 198/168 packing",
            oracle,
            required=True,
        )
        base.update(
            {
                "status": "exact",
                "stage_a_path": str(stage_a.relative_to(output)),
                "stage_b_path": str(stage_b.relative_to(output)),
            }
        )
        previous_best = max(
            (
                float(item["bound"])
                for item in checks.values()
                if item.get("status") == "exact"
            ),
            default=-math.inf,
        )
        checks[token] = base
        _atomic_json(checks_path, checks)
        refresh_exact_target_report(output, checks)
        newly_exact.append(record)
        if record.bound > previous_best + 1e-8:
            write_target_packing(
                output / "target_8316_stage_b.sol",
                record.layout,
                solar,
                accumulators,
            )
            save_record(
                output / "target_8316_stage_a.sol",
                record,
                "highest-bound exact target found so far",
                oracle,
                required=True,
            )
        print(
            f"NEW EXACT 8316 SETUP token={token} "
            f"bound={record.bound:.6f} total_exact="
            f"{sum(item.get('status') == 'exact' for item in checks.values())}",
            flush=True,
        )
    refresh_exact_target_report(output, checks)
    total_exact = sum(
        item.get("status") == "exact" for item in checks.values()
    )
    return newly_exact, total_exact


def refresh_archive_reports(output, archive, basins, oracle):
    front = sorted(
        pareto_front(archive),
        key=lambda record: (record.penalty, -record.bound),
    )
    columns = (
        "stage_b_bound",
        "unsupported_penalty",
        "generation",
        "geometry_signature",
        "substations",
        "medium_poles",
    )
    _atomic_csv(
        output / "pareto_front.csv",
        columns,
        (
            (
                record.bound,
                record.penalty,
                record.generation,
                geometry_signature(record, oracle),
                record.layout.substations,
                record.layout.medium_poles,
            )
            for record in front
        ),
    )
    _atomic_csv(
        output / "basin_archive.csv",
        columns,
        (
            (
                record.bound,
                record.penalty,
                record.generation,
                geometry_signature(record, oracle),
                record.layout.substations,
                record.layout.medium_poles,
            )
            for record in basins
        ),
    )


def evaluate_both(
    layouts,
    bound_evaluator,
    penalty_evaluator,
    coordinator,
):
    started = time.monotonic()
    if coordinator is None:
        bound_results = bound_evaluator.evaluate(layouts)
        penalty_results = penalty_evaluator.evaluate(layouts)
    else:
        bound_future = coordinator.submit(bound_evaluator.evaluate, layouts)
        penalty_future = coordinator.submit(penalty_evaluator.evaluate, layouts)
        bound_results = bound_future.result()
        penalty_results = penalty_future.result()
    bounds = {
        layout.key: float(bound)
        for layout, bound, _, _, status in bound_results
        if status == OPTIMAL and math.isfinite(bound)
    }
    penalties = {
        layout.key: float(penalty)
        for layout, penalty, status, _ in penalty_results
        if status == OPTIMAL and math.isfinite(penalty)
    }
    records = [
        (layout, bounds[layout.key], penalties[layout.key])
        for layout in layouts
        if layout.key in bounds and layout.key in penalties
    ]
    return records, time.monotonic() - started


def validate_args(args):
    if not 1 <= args.workers <= 32:
        raise ValueError("--workers must be between 1 and 32.")
    if min(
        args.population,
        args.parents,
        args.candidates,
        args.per_parent,
        args.seed_count,
        args.basin_archive,
        args.donors,
    ) <= 0:
        raise ValueError("Search sizes must be positive.")
    if args.seed_scan_limit < 0 or args.generations < 0:
        raise ValueError("Scan and generation limits must be nonnegative.")
    if not 1 <= args.radius <= 8:
        raise ValueError("--radius must be between 1 and 8.")
    if not 0.0 <= args.discovery_share <= 1.0:
        raise ValueError("--discovery-share must be between 0 and 1.")
    if args.basin_separation <= 0 or args.solar_penalty <= 0:
        raise ValueError("Basin separation and solar penalty must be positive.")


def main():
    args = parse_args()
    validate_args(args)
    coverage_root = args.coverage_root.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        args.output.resolve()
        if args.output is not None
        else RESULTS_ROOT / f"staged_highs_{timestamp}"
    )
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "bound_penalty_progress.csv"
    state_path = output / "state.json"
    oracle = FreePeriodicOracle()

    if progress_path.exists():
        if not progress_matches_model_semantics(progress_path, oracle):
            raise RuntimeError(
                "Existing progress uses different model semantics; choose a "
                "new --output directory."
            )
    else:
        write_model_semantics(output, oracle)

    state = load_state(state_path, args)
    archive_by_key = load_progress(progress_path)
    resuming = bool(archive_by_key)
    if state is not None:
        random_seed = int(state["random_seed"])
        generation = int(state.get("generation", 0))
        evaluation_id = int(state.get("evaluation_id", len(archive_by_key)))
        stagnation = int(state.get("stagnation", 0))
    else:
        random_seed = (
            args.seed
            if args.seed is not None
            else time.time_ns() & 0x7FFF_FFFF
        )
        generation = max(
            (record.generation for record in archive_by_key.values()),
            default=0,
        )
        evaluation_id = len(archive_by_key)
        stagnation = 0

    seed_items = []
    if not resuming:
        seed_candidates, explicit_keys = historical_seed_candidates(
            coverage_root,
            args.seed_sol,
            args.minimum_seed_bound,
            args.seed_scan_limit,
            args.seed_count,
            oracle,
        )
        seed_items = preselect_seed_candidates(
            seed_candidates,
            explicit_keys,
            args.seed_count,
            oracle,
        )
        if not seed_items:
            raise RuntimeError(
                "No physical-valid seed layouts found. Supply one or more "
                "--seed-sol paths or lower --minimum-seed-bound."
            )

    initial_layout = (
        next(iter(archive_by_key.values())).layout
        if archive_by_key
        else seed_items[0][1]
    )
    if args.workers == 1:
        bound_workers = penalty_workers = 1
        coordinator = None
    else:
        bound_workers = max(1, args.workers // 2)
        penalty_workers = args.workers - bound_workers
        coordinator = ThreadPoolExecutor(max_workers=2)
    bound_evaluator = ParallelStageBEvaluatorHighs(
        initial_layout.network_vector(),
        bound_workers,
        math.inf,
    )
    penalty_evaluator = ParallelPenaltyEvaluatorHighs(
        oracle,
        penalty_workers,
        args.solar_penalty,
    )
    generator_pool = ProcessPoolExecutor(max_workers=args.workers)
    exact_target = ExactTargetPackingHighs(args.workers)
    target_checks = load_target_checks(output / "target_checks.json")

    new_progress = not progress_path.exists()
    progress_handle = progress_path.open(
        "a",
        newline="",
        encoding="utf-8",
        buffering=1,
    )
    writer = csv.DictWriter(progress_handle, fieldnames=PROGRESS_COLUMNS)
    if new_progress:
        writer.writeheader()

    try:
        if not resuming:
            layouts = [item[1] for item in seed_items]
            source_by_key = {
                item[1].key: portable_source_path(item[2])
                for item in seed_items
            }
            seed_results, elapsed = evaluate_both(
                layouts,
                bound_evaluator,
                penalty_evaluator,
                coordinator,
            )
            for layout, bound, penalty in seed_results:
                record = Record(
                    layout,
                    penalty,
                    bound,
                    0,
                    source_by_key.get(layout.key, "historical seed"),
                )
                archive_by_key[layout.key] = record
                writer.writerow(
                    {
                        "evaluation_id": evaluation_id,
                        "generation": 0,
                        "phase": "rescored_seed",
                        "stage_b_bound": bound,
                        "unsupported_penalty": penalty,
                        "nearest_basin_distance": "",
                        "is_pareto": "",
                        "is_basin_representative": "",
                        "geometry_signature": geometry_signature(record, oracle),
                        "substations": layout.substations,
                        "medium_poles": layout.medium_poles,
                        "source": record.source,
                        "solution_path": "",
                    }
                )
                evaluation_id += 1
            if not archive_by_key:
                raise RuntimeError("Neither exact seed evaluator returned a result.")
            print(
                f"rescored {len(archive_by_key)}/{len(layouts)} seeds in "
                f"{elapsed:.2f}s",
                flush=True,
            )

        archive = list(archive_by_key.values())
        basins = basin_representatives(
            archive,
            oracle,
            args.basin_separation,
            args.basin_archive,
        )
        population = select_population(archive, basins, args.population)
        best_penalty_record = min(
            archive,
            key=lambda record: (record.penalty, -record.bound),
        )
        best_bound_record = max(
            archive,
            key=lambda record: (record.bound, -record.penalty),
        )
        tracked = metric_best(archive, best_penalty_record.penalty)
        save_record(
            output / "best_penalty.sol",
            best_penalty_record,
            "global penalty best",
            oracle,
        )
        save_record(
            output / "best_bound.sol",
            best_bound_record,
            "global Stage-B bound best",
            oracle,
        )
        for slack, record in tracked.items():
            if record is not None:
                save_record(
                    output / f"best_bound_penalty_plus_{slack:g}.sol",
                    record,
                    f"best bound within penalty slack {slack:g}",
                    oracle,
                )
        refresh_archive_reports(output, archive, basins, oracle)
        write_state(
            state_path,
            args=args,
            oracle=oracle,
            random_seed=random_seed,
            generation=generation,
            evaluation_id=evaluation_id,
            stagnation=stagnation,
        )
        print(
            f"PARETO {'RESUME' if resuming else 'START'} "
            f"generation={generation} archive={len(archive)} "
            f"population={len(population)} basins={len(basins)} "
            f"penalty={best_penalty_record.penalty:.6f} "
            f"penalty_bound={best_penalty_record.bound:.6f} "
            f"best_bound={best_bound_record.bound:.6f} workers={args.workers} "
            f"proposal={args.workers} bound={bound_workers} "
            f"penalty={penalty_workers} random_seed={random_seed}",
            flush=True,
        )

        _, total_exact_targets = verify_zero_penalty_records(
            archive,
            output=output,
            oracle=oracle,
            exact_target=exact_target,
            checks=target_checks,
        )
        if total_exact_targets:
            print(
                f"exact target archive contains {total_exact_targets} "
                f"setup(s); continue_after_target="
                f"{args.continue_after_target}",
                flush=True,
            )
            if not args.continue_after_target:
                return 0

        attempted = set(archive_by_key)
        while args.generations == 0 or generation < args.generations:
            generation += 1
            rng = random.Random(
                random_seed ^ (generation * 0x9E3779B1)
            )
            old_front_keys = {
                record.layout.key for record in pareto_front(archive)
            }
            old_basin_keys = {record.layout.key for record in basins}
            old_best_penalty = best_penalty_record
            old_best_bound = best_bound_record
            old_tracked = tracked

            parents = choose_parents(
                population,
                basins,
                args.parents,
                args.discovery_share,
                rng,
            )
            donors = select_donors(population, basins, args.donors)
            radius = min(8, args.radius + stagnation // 10)
            parent_cap = min(
                args.per_parent,
                max(
                    24,
                    math.ceil(
                        1.35 * args.candidates / max(1, len(parents))
                    ),
                ),
            )
            local_seen = frozenset(record.layout.key for record in population)
            tasks = []
            discovery_tasks = 0
            for parent, discovery_probability in parents:
                if discovery_probability > 0.5:
                    discovery_tasks += 1
                tasks.append(
                    (
                        parent.layout,
                        tuple(
                            donor
                            for donor in donors
                            if donor.key != parent.layout.key
                        ),
                        rng.randrange(2**31),
                        parent_cap,
                        radius,
                        local_seen,
                        None,
                        discovery_probability,
                    )
                )

            proposal_started = time.monotonic()
            proposals = {}
            for group in generator_pool.map(
                expanded_parent_candidates_process,
                tasks,
            ):
                for layout in group:
                    if layout.key not in attempted:
                        proposals[layout.key] = layout
            layouts = list(proposals.values())
            rng.shuffle(layouts)
            if len(layouts) > args.candidates:
                layouts = layouts[: args.candidates]
            proposal_elapsed = time.monotonic() - proposal_started
            if not layouts:
                stagnation += 1
                write_state(
                    state_path,
                    args=args,
                    oracle=oracle,
                    random_seed=random_seed,
                    generation=generation,
                    evaluation_id=evaluation_id,
                    stagnation=stagnation,
                )
                print(
                    f"pareto generation {generation}: no unseen proposals "
                    f"parents={len(parents)} discovery={discovery_tasks} "
                    f"radius={radius} stagnation={stagnation}",
                    flush=True,
                )
                continue
            attempted.update(layout.key for layout in layouts)

            evaluated, evaluation_elapsed = evaluate_both(
                layouts,
                bound_evaluator,
                penalty_evaluator,
                coordinator,
            )
            new_records = [
                Record(layout, penalty, bound, generation, "generated")
                for layout, bound, penalty in evaluated
            ]
            if not new_records:
                stagnation += 1
                write_state(
                    state_path,
                    args=args,
                    oracle=oracle,
                    random_seed=random_seed,
                    generation=generation,
                    evaluation_id=evaluation_id,
                    stagnation=stagnation,
                )
                print(
                    f"pareto generation {generation}: both exact LPs "
                    f"accepted 0/{len(layouts)} proposals "
                    f"stagnation={stagnation}",
                    flush=True,
                )
                continue
            for record in new_records:
                archive_by_key[record.layout.key] = record
            archive = list(archive_by_key.values())
            new_front = pareto_front(archive)
            front_keys = {record.layout.key for record in new_front}
            basins = basin_representatives(
                archive,
                oracle,
                args.basin_separation,
                args.basin_archive,
            )
            basin_keys = {record.layout.key for record in basins}
            new_basin_keys = basin_keys - old_basin_keys
            best_penalty_record = min(
                archive,
                key=lambda record: (record.penalty, -record.bound),
            )
            best_bound_record = max(
                archive,
                key=lambda record: (record.bound, -record.penalty),
            )
            tracked = metric_best(archive, best_penalty_record.penalty)

            solution_paths = {}
            for record in new_records:
                token = layout_token(record.layout)
                if record.layout.key in front_keys - old_front_keys:
                    path = output / "pareto" / (
                        f"generation_{generation:06d}_penalty_"
                        f"{record.penalty:.6f}_bound_{record.bound:.6f}_"
                        f"{token}.sol"
                    )
                    save_record(path, record, "new Pareto-front member", oracle)
                    solution_paths[record.layout.key] = path
                if record.layout.key in new_basin_keys:
                    path = output / "basins" / (
                        f"generation_{generation:06d}_penalty_"
                        f"{record.penalty:.6f}_bound_{record.bound:.6f}_"
                        f"{token}.sol"
                    )
                    save_record(path, record, "new diverse basin seed", oracle)
                    solution_paths.setdefault(record.layout.key, path)
                if record.bound >= 8316.0 - 1e-8:
                    path = output / "above_8316" / (
                        f"bound_{record.bound:.6f}_penalty_"
                        f"{record.penalty:.6f}_{token}.sol"
                    )
                    save_record(path, record, "all Stage-B bound >= 8316", oracle)
                    solution_paths.setdefault(record.layout.key, path)

            penalty_improved = (
                best_penalty_record.penalty
                < old_best_penalty.penalty - 1e-8
            )
            bound_improved = (
                best_bound_record.bound > old_best_bound.bound + 1e-7
            )
            if penalty_improved:
                save_record(
                    output / "best_penalty.sol",
                    best_penalty_record,
                    "new global penalty best",
                    oracle,
                )
                print(
                    f"NEW PENALTY BEST {best_penalty_record.penalty:.9f} "
                    f"bound={best_penalty_record.bound:.6f} "
                    f"generation={generation}",
                    flush=True,
                )
            if bound_improved:
                save_record(
                    output / "best_bound.sol",
                    best_bound_record,
                    "new global Stage-B bound best",
                    oracle,
                )
                print(
                    f"NEW BOUND BEST {best_bound_record.bound:.6f} "
                    f"penalty={best_bound_record.penalty:.6f} "
                    f"generation={generation}",
                    flush=True,
                )

            tradeoff_improved = False
            for slack, record in tracked.items():
                previous = old_tracked.get(slack)
                if record is None:
                    continue
                if (
                    previous is None
                    or record.layout.key != previous.layout.key
                    or record.bound > previous.bound + 1e-7
                ):
                    tradeoff_improved = True
                    save_record(
                        output / f"best_bound_penalty_plus_{slack:g}.sol",
                        record,
                        f"best bound within penalty slack {slack:g}",
                        oracle,
                    )

            for record in new_records:
                nearest = min(
                    (
                        record.layout.relative_distance(other.layout)
                        for other in basins
                        if other.layout.key != record.layout.key
                    ),
                    default=15,
                )
                solution_path = solution_paths.get(record.layout.key)
                writer.writerow(
                    {
                        "evaluation_id": evaluation_id,
                        "generation": generation,
                        "phase": "mixed_discovery_pareto",
                        "stage_b_bound": record.bound,
                        "unsupported_penalty": record.penalty,
                        "nearest_basin_distance": nearest,
                        "is_pareto": int(record.layout.key in front_keys),
                        "is_basin_representative": int(
                            record.layout.key in basin_keys
                        ),
                        "geometry_signature": geometry_signature(record, oracle),
                        "substations": record.layout.substations,
                        "medium_poles": record.layout.medium_poles,
                        "source": record.source,
                        "solution_path": (
                            str(solution_path.relative_to(output))
                            if solution_path is not None
                            else ""
                        ),
                    }
                )
                evaluation_id += 1
            progress_handle.flush()

            population = select_population(archive, basins, args.population)
            meaningful = (
                penalty_improved
                or bound_improved
                or tradeoff_improved
                or bool(new_basin_keys)
            )
            stagnation = 0 if meaningful else stagnation + 1
            refresh_archive_reports(output, archive, basins, oracle)
            write_state(
                state_path,
                args=args,
                oracle=oracle,
                random_seed=random_seed,
                generation=generation,
                evaluation_id=evaluation_id,
                stagnation=stagnation,
            )
            newly_exact, total_exact_targets = verify_zero_penalty_records(
                new_records,
                output=output,
                oracle=oracle,
                exact_target=exact_target,
                checks=target_checks,
            )
            print(
                f"pareto generation {generation}: proposed={len(layouts)} "
                f"exact={len(new_records)} proposal_s={proposal_elapsed:.2f} "
                f"exact_s={evaluation_elapsed:.2f} "
                f"penalty={best_penalty_record.penalty:.6f} "
                f"best_bound={best_bound_record.bound:.6f} "
                f"frontier={len(new_front)} basins={len(basins)} "
                f"new_basins={len(new_basin_keys)} "
                f"exact_targets={total_exact_targets} "
                f"discovery_parents={discovery_tasks}/{len(parents)} "
                f"stagnation={stagnation} radius={radius}",
                flush=True,
            )
            if newly_exact and not args.continue_after_target:
                print(f"TARGET 8316 FOUND output={output}", flush=True)
                return 0
    except KeyboardInterrupt:
        print(
            f"\nStopped by Ctrl-C. State is durable; rerun the same command "
            f"with --output \"{output}\".",
            flush=True,
        )
        return 130
    finally:
        progress_handle.close()
        exact_target.close()
        penalty_evaluator.close()
        bound_evaluator.close()
        generator_pool.shutdown(wait=True, cancel_futures=True)
        if coordinator is not None:
            coordinator.shutdown(wait=True, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
