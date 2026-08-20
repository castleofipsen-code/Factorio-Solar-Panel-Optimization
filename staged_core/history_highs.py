"""Exhaust exact distance-three neighborhoods of strong distinct basins."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import re
import subprocess
import sys

from staged_core.network_highs import FreeCoordinateLayout


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVERAGE_ROOT = ROOT / "staged_seeds"
REFINER = ROOT / "refine_free_coordinate_distance_beam.py"
PAIR_REFINER = ROOT / "refine_free_coordinate_incumbent.py"
BOUND_PREFIX = "# Exact Stage-B LP bound = "
SUBSTATION_PREFIX = "# Substations = "
MEDIUM_PREFIX = "# Medium poles = "
COORDINATE = re.compile(r"\((\d+),\s*(\d+)\)")
REPRESENTATIVE_NAMES = {
    "best_free_coordinate.sol",
    "best_free_coordinate_refined.sol",
    "best_free_coordinate_triple_refined.sol",
    "best_free_coordinate_distance_refined.sol",
    "best_exact_cover_penalty_network.sol",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-root",
        type=Path,
        default=DEFAULT_COVERAGE_ROOT,
    )
    parser.add_argument("--minimum-bound", type=float, default=8310.0)
    parser.add_argument("--basin-separation", type=int, default=3)
    parser.add_argument(
        "--basin-count",
        type=int,
        default=0,
        help="Maximum basin count; zero keeps every qualifying basin.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=100_000)
    parser.add_argument("--exact-batch", type=int, default=64)
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument(
        "--skip-rank",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--rebuild-distance-two",
        action="store_true",
        help=(
            "Exhaust the radius-limited single/pair layer before expanding "
            "distance three."
        ),
    )
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def coordinates(line: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(row), int(column))
        for row, column in COORDINATE.findall(line)
    )


def read_header(
    path: Path,
) -> tuple[float, FreeCoordinateLayout] | None:
    bound = None
    substations = None
    medium_poles = None
    try:
        with path.open(encoding="utf-8") as handle:
            for _ in range(12):
                line = handle.readline()
                if not line:
                    break
                if line.startswith(BOUND_PREFIX):
                    bound = float(line[len(BOUND_PREFIX):].strip())
                elif line.startswith(SUBSTATION_PREFIX):
                    substations = coordinates(line)
                elif line.startswith(MEDIUM_PREFIX):
                    medium_poles = coordinates(line)
    except (OSError, UnicodeError, ValueError):
        return None
    if bound is None or substations is None or medium_poles is None:
        return None
    try:
        layout = FreeCoordinateLayout.create(substations, medium_poles)
    except ValueError:
        return None
    return bound, layout


def candidate_layouts(
    coverage_root: Path,
    minimum_bound: float,
) -> list[tuple[float, FreeCoordinateLayout, Path]]:
    distinct: dict[
        tuple[int, ...],
        tuple[float, FreeCoordinateLayout, Path],
    ] = {}
    paths = (
        path
        for path in coverage_root.rglob("*.sol")
        if (
            path.name in REPRESENTATIVE_NAMES
            or path.name.startswith("seed_")
        )
    )
    for path in paths:
        parsed = read_header(path)
        if parsed is None:
            continue
        bound, layout = parsed
        if not math.isfinite(bound):
            continue
        if bound + 1e-9 < minimum_bound:
            continue
        previous = distinct.get(layout.key)
        if previous is None or bound > previous[0]:
            distinct[layout.key] = (bound, layout, path)
    return sorted(distinct.values(), key=lambda item: item[0], reverse=True)


def select_basins(
    candidates: list[tuple[float, FreeCoordinateLayout, Path]],
    separation: int,
    count: int,
) -> list[tuple[float, FreeCoordinateLayout, Path, int]]:
    selected = []
    for bound, layout, path in candidates:
        nearest = min(
            (
                layout.relative_distance(other_layout)
                for _, other_layout, _, _ in selected
            ),
            default=15,
        )
        if nearest < separation:
            continue
        selected.append((bound, layout, path, nearest))
        if count and len(selected) >= count:
            break
    return selected


def main() -> int:
    args = parse_args()
    coverage_root = args.coverage_root.resolve()
    if args.basin_separation <= 0:
        raise ValueError("--basin-separation must be positive.")
    if args.basin_count < 0 or args.workers <= 0:
        raise ValueError("Counts must be nonnegative and workers positive.")
    if args.radius <= 0 or args.beam_width <= 0:
        raise ValueError("Radius and beam width must be positive.")

    candidates = candidate_layouts(coverage_root, args.minimum_bound)
    basins = select_basins(
        candidates,
        args.basin_separation,
        args.basin_count,
    )
    print(
        f"Selected {len(basins)} distinct basins from "
        f"{len(candidates)} qualifying layouts "
        f"(bound>={args.minimum_bound:.3f}, "
        f"relative separation>={args.basin_separation}).",
        flush=True,
    )
    for rank, (bound, _, path, nearest) in enumerate(basins, 1):
        print(
            f"  basin {rank:02d}: {bound:.6f} "
            f"nearest={nearest} seed={path.relative_to(ROOT)}",
            flush=True,
        )
    if args.list_only:
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        args.output.resolve()
        if args.output is not None
        else coverage_root / f"{timestamp}_basin_distance3_sweep"
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "basin_distance3_manifest.csv"
    with manifest.open("w", newline="", buffering=1) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "seed_bound",
                "nearest_selected_basin",
                "seed_path",
                "distance_two_status",
                "status",
                "output_path",
            ]
        )
        failures = 0
        for rank, (bound, _, seed_path, nearest) in enumerate(basins, 1):
            if rank < args.start_rank or rank in args.skip_rank:
                writer.writerow(
                    [
                        rank,
                        bound,
                        nearest,
                        seed_path,
                        "skipped",
                        "skipped",
                        "",
                    ]
                )
                continue
            basin_root = output / f"basin_{rank:02d}_{bound:.6f}"
            distance_two_status = "reused"
            parent_arguments = []
            if args.rebuild_distance_two:
                distance_two_output = basin_root / "distance_2"
                pair_command = [
                    sys.executable,
                    str(PAIR_REFINER),
                    "--seed-sol",
                    str(seed_path),
                    "--workers",
                    str(args.workers),
                    "--seconds",
                    "0",
                    "--enumeration-seconds",
                    "0",
                    "--pair-radius",
                    str(args.radius),
                    "--exact-batch",
                    str(args.exact_batch),
                    "--target",
                    "inf",
                    "--output",
                    str(distance_two_output),
                ]
                print(
                    f"\n=== basin {rank}/{len(basins)} "
                    f"distance 2 seed={bound:.6f} ===",
                    flush=True,
                )
                pair_result = subprocess.run(
                    pair_command,
                    cwd=ROOT,
                    check=False,
                )
                distance_two_status = (
                    "complete" if pair_result.returncode == 0 else "failed"
                )
                if pair_result.returncode != 0:
                    failures += 1
                    writer.writerow(
                        [
                            rank,
                            bound,
                            nearest,
                            seed_path,
                            distance_two_status,
                            "not_run",
                            basin_root,
                        ]
                    )
                    continue
                parent_arguments = [
                    "--parent-progress",
                    str(distance_two_output / "refinement_progress.csv"),
                ]

            basin_output = basin_root / "distance_3"
            command = [
                sys.executable,
                str(REFINER),
                "--seed-sol",
                str(seed_path),
                *parent_arguments,
                "--all-parent-root",
                str(coverage_root),
                "--all-history-root",
                str(coverage_root),
                "--target-distance",
                "3",
                "--workers",
                str(args.workers),
                "--beam-width",
                str(args.beam_width),
                "--radius",
                str(args.radius),
                "--exact-batch",
                str(args.exact_batch),
                "--seconds",
                "0",
                "--output",
                str(basin_output),
            ]
            print(
                f"\n=== basin {rank}/{len(basins)} "
                f"seed={bound:.6f} ===",
                flush=True,
            )
            result = subprocess.run(command, cwd=ROOT, check=False)
            status = "complete" if result.returncode == 0 else "failed"
            failures += result.returncode != 0
            writer.writerow(
                [
                    rank,
                    bound,
                    nearest,
                    seed_path,
                    distance_two_status,
                    status,
                    basin_root,
                ]
            )
    print(
        f"Basin distance-three sweep finished: "
        f"{len(basins) - failures} complete, {failures} failed, "
        f"output={output}",
        flush=True,
    )
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
