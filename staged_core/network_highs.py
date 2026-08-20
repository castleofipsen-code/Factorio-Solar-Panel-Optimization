"""Free-coordinate 5+10 Stage-A search with exact periodic tileability.

This solver removes every fixed substation and medium-pole position.  Its
search state is only the 30 integer coordinates of five substations and ten
medium poles.  A direct oracle checks the current Stage-A coverage, color,
physical-clearance, connectivity, and periodic-tileability rules before the
unchanged Stage-B root LP is evaluated.

Every persisted solution is written in the established dense 5,006-variable
Stage-A format:

    x[0:2500]       substation placement binaries
    x[2500:5000]    medium-pole placement binaries
    x[5000:5006]    color witnesses a, b, c, d, X, q

The old fixed-medium constructor is deliberately not changed.  A genuinely
free periodic layout cannot be validated against that constructor because it
intentionally need not contain its three fixed corner poles; this file's
direct oracle is the feasibility authority for the freed formulation.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from datetime import datetime
import errno
import json
from math import gcd
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np

from staged_core.coloring import color_witness


GRID = 50
DD = GRID * GRID
NETWORK_SIZE = 2 * DD
BINARY_SOLUTION_SIZE = NETWORK_SIZE + 6
EXACT_SUBSTATIONS = 5
EXACT_MEDIUMS = 10
MODEL_SEMANTICS_VERSION = 2
CENTRAL_LOW = GRID // 2 - 2
CENTRAL_HIGH = GRID // 2 + 2
CENTRAL_INDICES = tuple(
    row * GRID + column
    for row in range(CENTRAL_LOW, CENTRAL_HIGH)
    for column in range(CENTRAL_LOW, CENTRAL_HIGH)
)
X_LINE = re.compile(
    r"^\s*x\[(\d+)\]\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _unique_temporary_path(path: Path) -> Path:
    return path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )


def _replace_with_retry(
    temporary: Path,
    path: Path,
    *,
    required: bool,
) -> bool:
    """Survive transient filesystem locks during an atomic replacement."""
    delay = 0.05
    attempts = 12
    last_error = None
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return True
        except OSError as error:
            last_error = error
            transient = (
                getattr(error, "winerror", None) in {5, 32, 33}
                or error.errno
                in {errno.EACCES, errno.EBUSY, errno.EPERM}
            )
            if not transient or attempt + 1 == attempts:
                break
            time.sleep(delay)
            delay = min(1.0, 2.0 * delay)

    if required:
        assert last_error is not None
        raise last_error
    print(
        f"warning: could not refresh convenience file {path}; "
        f"complete replacement left at {temporary}: {last_error}",
        flush=True,
    )
    return False


def model_semantics(oracle: "FreePeriodicOracle") -> dict[str, object]:
    return {
        "version": MODEL_SEMANTICS_VERSION,
        "wire_centers": (
            "physical" if oracle.physical_wire_offsets else "legacy_binary"
        ),
        "electric_coverage": (
            "periodic" if oracle.true_periodic_coverage else "legacy_clipped"
        ),
        "standalone_connectivity_required": (
            not oracle.periodic_only_connectivity
        ),
    }


def write_model_semantics(
    folder: Path,
    oracle: "FreePeriodicOracle",
) -> Path:
    """Record the feasibility semantics beside progress files."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model_semantics.json"
    payload = json.dumps(model_semantics(oracle), indent=2, sort_keys=True) + "\n"
    temporary = _unique_temporary_path(path)
    temporary.write_text(payload, encoding="utf-8")
    _replace_with_retry(temporary, path, required=True)
    return path


def progress_matches_model_semantics(
    progress_path: Path,
    oracle: "FreePeriodicOracle",
) -> bool:
    """Legacy progress must not suppress evaluation under corrected rules."""
    path = progress_path.parent / "model_semantics.json"
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return recorded == model_semantics(oracle)


def coordinate_index(coordinate: tuple[int, int]) -> int:
    row, column = coordinate
    return int(row) * GRID + int(column)


def index_coordinate(index: int) -> tuple[int, int]:
    return divmod(int(index), GRID)


def medium_color(index: int) -> int:
    row, column = index_coordinate(index)
    return 2 * (row % 2) + column % 2


def _indices_from_coordinates(values) -> tuple[int, ...]:
    indices = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            index = int(value)
        else:
            index = coordinate_index(tuple(value))
        indices.append(index)
    return tuple(sorted(indices))


@dataclass(frozen=True, order=True)
class FreeCoordinateLayout:
    """Canonical coordinate state for five substations and ten medium poles."""

    substation_indices: tuple[int, ...]
    medium_indices: tuple[int, ...]

    @classmethod
    def create(cls, substations, medium_poles) -> "FreeCoordinateLayout":
        return cls(
            _indices_from_coordinates(substations),
            _indices_from_coordinates(medium_poles),
        )

    @property
    def substations(self) -> tuple[tuple[int, int], ...]:
        return tuple(map(index_coordinate, self.substation_indices))

    @property
    def medium_poles(self) -> tuple[tuple[int, int], ...]:
        return tuple(map(index_coordinate, self.medium_indices))

    @property
    def key(self) -> tuple[int, ...]:
        return (*self.substation_indices, *self.medium_indices)

    @property
    def relative_key(self) -> tuple[int, ...]:
        """Canonical key modulo a whole-layout toroidal translation."""
        canonical = None
        for anchor in self.substation_indices:
            anchor_row, anchor_column = index_coordinate(anchor)

            def shifted(index):
                row, column = index_coordinate(index)
                return (
                    (row - anchor_row) % GRID
                ) * GRID + (column - anchor_column) % GRID

            key = (
                *sorted(shifted(index) for index in self.substation_indices),
                *sorted(shifted(index) for index in self.medium_indices),
            )
            if canonical is None or key < canonical:
                canonical = key
        return self.key if canonical is None else canonical

    @property
    def coordinate_vector(self) -> tuple[int, ...]:
        values: list[int] = []
        for index in (*self.substation_indices, *self.medium_indices):
            values.extend(index_coordinate(index))
        return tuple(values)

    @property
    def selected_network_indices(self) -> np.ndarray:
        return np.asarray(
            [
                *self.substation_indices,
                *(DD + index for index in self.medium_indices),
            ],
            dtype=int,
        )

    def network_vector(self) -> np.ndarray:
        network = np.zeros(NETWORK_SIZE, dtype=float)
        network[self.selected_network_indices] = 1.0
        return network

    def distance(self, other: "FreeCoordinateLayout") -> int:
        """Number of selected objects replaced, separated by object type."""
        common_substations = len(
            set(self.substation_indices) & set(other.substation_indices)
        )
        common_mediums = len(
            set(self.medium_indices) & set(other.medium_indices)
        )
        return (
            EXACT_SUBSTATIONS
            + EXACT_MEDIUMS
            - common_substations
            - common_mediums
        )

    def relative_distance(self, other: "FreeCoordinateLayout") -> int:
        """Minimum object-replacement distance over whole-layout shifts."""
        shifts = {(0, 0)}
        for left_indices, right_indices in (
            (self.substation_indices, other.substation_indices),
            (self.medium_indices, other.medium_indices),
        ):
            for left in left_indices:
                left_row, left_column = index_coordinate(left)
                for right in right_indices:
                    right_row, right_column = index_coordinate(right)
                    shifts.add(
                        (
                            (right_row - left_row) % GRID,
                            (right_column - left_column) % GRID,
                        )
                    )

        other_substations = set(other.substation_indices)
        other_mediums = set(other.medium_indices)
        maximum_common = 0
        for row_shift, column_shift in shifts:
            def shifted(index):
                row, column = index_coordinate(index)
                return (
                    (row + row_shift) % GRID
                ) * GRID + (column + column_shift) % GRID

            common = sum(
                shifted(index) in other_substations
                for index in self.substation_indices
            )
            common += sum(
                shifted(index) in other_mediums
                for index in self.medium_indices
            )
            maximum_common = max(maximum_common, common)
        return EXACT_SUBSTATIONS + EXACT_MEDIUMS - maximum_common


@dataclass(frozen=True)
class MediumGeometry:
    """Shape and topology summary of the medium-pole subgraph."""

    edges: int
    axial: int
    shallow: int
    moderate: int
    deep: int
    degree_two: int
    leaves: int
    branches: int

    @property
    def signature(self) -> tuple[int, ...]:
        return (
            self.axial,
            self.shallow,
            self.moderate,
            self.deep,
            self.degree_two,
            self.leaves,
            self.branches,
        )


@dataclass(frozen=True)
class PeriodicCertificate:
    quotient_components: int
    cell_components: int
    edge_count: int
    winding_rank: int
    lattice_index: int | None
    windings: tuple[tuple[int, int], ...]

    @property
    def exact_periodic(self) -> bool:
        return (
            self.quotient_components == 1
            and self.winding_rank == 2
            and self.lattice_index == 1
        )


@dataclass(frozen=True)
class OracleResult:
    feasible: bool
    reason: str
    floor_holes: int
    central_holes: int
    physical_conflicts: int
    forbidden_positions: int
    duplicate_positions: int
    color_defect: int
    certificate: PeriodicCertificate

    @property
    def damage(self) -> float:
        certificate = self.certificate
        lattice_damage = 0
        if certificate.quotient_components > 1:
            lattice_damage += 1_500 * (certificate.quotient_components - 1)
        if certificate.winding_rank == 0:
            lattice_damage += 1_200
        elif certificate.winding_rank == 1:
            lattice_damage += 700
        elif certificate.lattice_index != 1:
            lattice_damage += 350 * min(
                10,
                max(1, int(certificate.lattice_index or 2) - 1),
            )
        cell_damage = max(0, certificate.cell_components - 1) * 900
        return float(
            self.floor_holes * 10
            + self.central_holes * 250
            + self.physical_conflicts * 5_000
            + self.forbidden_positions * 5_000
            + self.duplicate_positions * 10_000
            + self.color_defect * 1_000
            + cell_damage
            + lattice_damage
        )


class FreePeriodicOracle:
    """Direct feasibility checker for a completely free 5+10 network."""

    def __init__(
        self,
        *,
        physical_wire_offsets: bool = True,
        periodic_only_connectivity: bool = False,
        true_periodic_coverage: bool = True,
    ):
        self.physical_wire_offsets = bool(physical_wire_offsets)
        self.periodic_only_connectivity = bool(periodic_only_connectivity)
        self.true_periodic_coverage = bool(true_periodic_coverage)
        self.all_floor = (1 << DD) - 1
        self.central_mask = sum(1 << index for index in CENTRAL_INDICES)

        self.substation_footprints = tuple(
            self._footprint_mask(index, 2) for index in range(DD)
        )
        self.medium_footprints = tuple(1 << index for index in range(DD))
        self.substation_electric_masks = tuple(
            self._electric_mask(index, 8, 9) for index in range(DD)
        )
        self.medium_electric_masks = tuple(
            self._electric_mask(index, 3, 3) for index in range(DD)
        )
        self.substation_floor_masks = tuple(
            self._floor_coverage_mask(index, 8, 9) for index in range(DD)
        )
        self.medium_floor_masks = tuple(
            self._floor_coverage_mask(index, 3, 3) for index in range(DD)
        )

        self.substation_allowed = np.ones(DD, dtype=bool)
        self.medium_allowed = np.ones(DD, dtype=bool)
        for index in range(DD):
            if self.substation_footprints[index] & self.central_mask:
                self.substation_allowed[index] = False
            if self.medium_footprints[index] & self.central_mask:
                self.medium_allowed[index] = False

        allowed_mediums = np.flatnonzero(self.medium_allowed)
        self.medium_indices_by_color = tuple(
            tuple(
                int(index)
                for index in allowed_mediums
                if medium_color(int(index)) == color
            )
            for color in range(4)
        )
        self.valid_medium_color_patterns = tuple(
            pattern
            for pattern in self._all_medium_color_patterns()
            if self._pattern_has_color_witness(pattern)
        )

    @staticmethod
    def _footprint_mask(index: int, size: int) -> int:
        row, column = index_coordinate(index)
        mask = 0
        for dr in range(size):
            for dc in range(size):
                wrapped_row = (row + dr) % GRID
                wrapped_column = (column + dc) % GRID
                mask |= 1 << (wrapped_row * GRID + wrapped_column)
        return mask

    def _electric_mask(self, index: int, left: int, right: int) -> int:
        """Tiles reached by one pole in the periodic 50x50 array."""
        row, column = index_coordinate(index)
        mask = 0
        if self.true_periodic_coverage:
            covered_rows = (
                (row + offset) % GRID
                for offset in range(-left, right + 1)
            )
            covered_columns = tuple(
                (column + offset) % GRID
                for offset in range(-left, right + 1)
            )
        else:
            covered_rows = range(
                max(0, row - left),
                min(GRID, row + right + 1),
            )
            covered_columns = tuple(
                range(
                    max(0, column - left),
                    min(GRID, column + right + 1),
                )
            )
        for covered_row in covered_rows:
            for covered_column in covered_columns:
                mask |= 1 << (covered_row * GRID + covered_column)
        return mask

    def _floor_coverage_mask(self, index: int, left: int, right: int) -> int:
        """Roots of wrapped 5x5 floors hit by one network object."""
        if self.true_periodic_coverage:
            row, column = index_coordinate(index)
            covered_rows = {
                (row + offset) % GRID
                for offset in range(-left, right + 1)
            }
            covered_columns = {
                (column + offset) % GRID
                for offset in range(-left, right + 1)
            }
        else:
            electric = self._electric_mask(index, left, right)
            covered_indices = tuple(_bit_indices(electric))
            covered_rows = {covered // GRID for covered in covered_indices}
            covered_columns = {covered % GRID for covered in covered_indices}

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

    @staticmethod
    def _all_medium_color_patterns():
        for color_0 in range(EXACT_MEDIUMS + 1):
            for color_1 in range(EXACT_MEDIUMS - color_0 + 1):
                for color_2 in range(
                    EXACT_MEDIUMS - color_0 - color_1 + 1
                ):
                    color_3 = (
                        EXACT_MEDIUMS - color_0 - color_1 - color_2
                    )
                    yield (color_0, color_1, color_2, color_3)

    @staticmethod
    def _pattern_has_color_witness(pattern) -> bool:
        return (
            (pattern[0] - pattern[3]) % 3 == 0
            and (pattern[1] - pattern[2]) % 3 == 0
        )

    @staticmethod
    def color_counts(layout: FreeCoordinateLayout) -> tuple[int, ...]:
        counts = [0, 0, 0, 0]
        for index in layout.medium_indices:
            if 0 <= index < DD:
                counts[medium_color(index)] += 1
        return tuple(counts)

    @staticmethod
    def color_defect(counts: tuple[int, ...]) -> int:
        if sum(counts) != EXACT_MEDIUMS:
            return abs(sum(counts) - EXACT_MEDIUMS) + 4

        def residue_distance(value):
            residue = value % 3
            return min(residue, (-residue) % 3)

        return residue_distance(counts[0] - counts[3]) + residue_distance(
            counts[1] - counts[2]
        )

    def coverage_union(self, layout: FreeCoordinateLayout) -> int:
        covered = 0
        for index in layout.substation_indices:
            if 0 <= index < DD:
                covered |= self.substation_floor_masks[index]
        for index in layout.medium_indices:
            if 0 <= index < DD:
                covered |= self.medium_floor_masks[index]
        return covered

    def electric_union(self, layout: FreeCoordinateLayout) -> int:
        covered = 0
        for index in layout.substation_indices:
            if 0 <= index < DD:
                covered |= self.substation_electric_masks[index]
        for index in layout.medium_indices:
            if 0 <= index < DD:
                covered |= self.medium_electric_masks[index]
        return covered

    def _doubled_center(self, kind: str, index: int) -> tuple[int, int]:
        row, column = index_coordinate(index)
        if kind == "sub":
            offset = 2 if self.physical_wire_offsets else 0
        else:
            offset = 1
        return 2 * row + offset, 2 * column + offset

    @staticmethod
    def _wrapped_component(raw: int) -> tuple[int, int]:
        doubled_period = 2 * GRID
        if raw > GRID:
            return raw - doubled_period, -1
        if raw < -GRID:
            return raw + doubled_period, 1
        return raw, 0

    def pair_edge(
        self,
        left_kind: str,
        left_index: int,
        right_kind: str,
        right_index: int,
    ) -> tuple[int, int] | None:
        """Return the target-cell shift for the unique wrapped wire edge."""
        left_row, left_column = self._doubled_center(
            left_kind,
            left_index,
        )
        right_row, right_column = self._doubled_center(
            right_kind,
            right_index,
        )
        row_distance, row_shift = self._wrapped_component(
            right_row - left_row
        )
        column_distance, column_shift = self._wrapped_component(
            right_column - left_column
        )
        radius = (
            36
            if left_kind == right_kind == "sub"
            else 18
        )
        if (
            row_distance * row_distance
            + column_distance * column_distance
            <= radius * radius
        ):
            return row_shift, column_shift
        return None

    def medium_edge_shape(
        self,
        left_index: int,
        right_index: int,
    ) -> str | None:
        """Classify one connected medium edge by its smaller tile offset."""
        shift = self.pair_edge(
            "med",
            left_index,
            "med",
            right_index,
        )
        if shift is None:
            return None
        left_row, left_column = index_coordinate(left_index)
        right_row, right_column = index_coordinate(right_index)
        row_delta = abs(
            right_row + shift[0] * GRID - left_row
        )
        column_delta = abs(
            right_column + shift[1] * GRID - left_column
        )
        minor = min(row_delta, column_delta)
        if minor == 0:
            return "axial"
        if minor <= 2:
            return "shallow"
        if minor <= 4:
            return "moderate"
        return "deep"

    def medium_geometry(
        self,
        layout: FreeCoordinateLayout,
    ) -> MediumGeometry:
        degrees = [0] * len(layout.medium_indices)
        counts = {
            "axial": 0,
            "shallow": 0,
            "moderate": 0,
            "deep": 0,
        }
        for left in range(len(layout.medium_indices)):
            for right in range(left + 1, len(layout.medium_indices)):
                shape = self.medium_edge_shape(
                    layout.medium_indices[left],
                    layout.medium_indices[right],
                )
                if shape is None:
                    continue
                counts[shape] += 1
                degrees[left] += 1
                degrees[right] += 1
        return MediumGeometry(
            edges=sum(counts.values()),
            axial=counts["axial"],
            shallow=counts["shallow"],
            moderate=counts["moderate"],
            deep=counts["deep"],
            degree_two=sum(degree == 2 for degree in degrees),
            leaves=sum(degree == 1 for degree in degrees),
            branches=sum(degree >= 3 for degree in degrees),
        )

    def periodic_edges(
        self,
        layout: FreeCoordinateLayout,
    ) -> list[tuple[int, int, tuple[int, int]]]:
        nodes = [
            *(("sub", index) for index in layout.substation_indices),
            *(("med", index) for index in layout.medium_indices),
        ]
        edges = []
        for left in range(len(nodes)):
            left_kind, left_index = nodes[left]
            for right in range(left + 1, len(nodes)):
                right_kind, right_index = nodes[right]
                shift = self.pair_edge(
                    left_kind,
                    left_index,
                    right_kind,
                    right_index,
                )
                if shift is not None:
                    edges.append((left, right, shift))
        return edges

    @staticmethod
    def _component_count(
        node_count: int,
        edges,
        *,
        zero_shift_only: bool = False,
    ) -> int:
        if node_count <= 0:
            return 0
        adjacency = [[] for _ in range(node_count)]
        for left, right, shift in edges:
            if zero_shift_only and shift != (0, 0):
                continue
            adjacency[left].append(right)
            adjacency[right].append(left)
        seen = set()
        components = 0
        for root in range(node_count):
            if root in seen:
                continue
            components += 1
            stack = [root]
            seen.add(root)
            while stack:
                node = stack.pop()
                for neighbor in adjacency[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
        return components

    @staticmethod
    def _winding_lattice(
        node_count: int,
        edges: list[tuple[int, int, tuple[int, int]]],
    ) -> tuple[int, int | None, tuple[tuple[int, int], ...]]:
        if node_count <= 0:
            return 0, None, ()
        adjacency = [[] for _ in range(node_count)]
        for edge_id, (left, right, shift) in enumerate(edges):
            adjacency[left].append((right, shift, edge_id))
            adjacency[right].append(
                (left, (-shift[0], -shift[1]), edge_id)
            )

        potentials: list[tuple[int, int] | None] = [None] * node_count
        potentials[0] = (0, 0)
        stack = [0]
        while stack:
            node = stack.pop()
            potential = potentials[node]
            assert potential is not None
            for neighbor, shift, _ in adjacency[node]:
                if potentials[neighbor] is None:
                    potentials[neighbor] = (
                        potential[0] + shift[0],
                        potential[1] + shift[1],
                    )
                    stack.append(neighbor)
        if any(potential is None for potential in potentials):
            return 0, None, ()

        windings = set()
        for left, right, shift in edges:
            left_potential = potentials[left]
            right_potential = potentials[right]
            assert left_potential is not None and right_potential is not None
            winding = (
                left_potential[0] + shift[0] - right_potential[0],
                left_potential[1] + shift[1] - right_potential[1],
            )
            if winding != (0, 0):
                # Canonicalize sign only for compact diagnostics.
                if winding[0] < 0 or (
                    winding[0] == 0 and winding[1] < 0
                ):
                    winding = (-winding[0], -winding[1])
                windings.add(winding)
        ordered = tuple(sorted(windings))
        if not ordered:
            return 0, None, ()

        determinant_gcd = 0
        for position, (left_x, left_y) in enumerate(ordered):
            for right_x, right_y in ordered[:position]:
                determinant_gcd = gcd(
                    determinant_gcd,
                    abs(left_x * right_y - left_y * right_x),
                )
        if determinant_gcd == 0:
            return 1, None, ordered
        return 2, determinant_gcd, ordered

    def periodic_certificate(
        self,
        layout: FreeCoordinateLayout,
    ) -> PeriodicCertificate:
        node_count = (
            len(layout.substation_indices) + len(layout.medium_indices)
        )
        edges = self.periodic_edges(layout)
        quotient_components = self._component_count(node_count, edges)
        cell_components = self._component_count(
            node_count,
            edges,
            zero_shift_only=True,
        )
        rank, lattice_index, windings = self._winding_lattice(
            node_count,
            edges,
        )
        return PeriodicCertificate(
            quotient_components=quotient_components,
            cell_components=cell_components,
            edge_count=len(edges),
            winding_rank=rank,
            lattice_index=lattice_index,
            windings=windings,
        )

    def diagnose(self, layout: FreeCoordinateLayout) -> OracleResult:
        bad_count = (
            abs(len(layout.substation_indices) - EXACT_SUBSTATIONS)
            + abs(len(layout.medium_indices) - EXACT_MEDIUMS)
        )
        out_of_bounds = sum(
            not 0 <= index < DD
            for index in (
                *layout.substation_indices,
                *layout.medium_indices,
            )
        )
        duplicate_positions = (
            len(layout.substation_indices)
            - len(set(layout.substation_indices))
            + len(layout.medium_indices)
            - len(set(layout.medium_indices))
        )
        forbidden_positions = out_of_bounds
        forbidden_positions += sum(
            0 <= index < DD and not self.substation_allowed[index]
            for index in layout.substation_indices
        )
        forbidden_positions += sum(
            0 <= index < DD and not self.medium_allowed[index]
            for index in layout.medium_indices
        )

        occupied = 0
        physical_conflicts = 0
        for index in layout.substation_indices:
            if not 0 <= index < DD:
                continue
            footprint = self.substation_footprints[index]
            if occupied & footprint:
                physical_conflicts += 1
            occupied |= footprint
        for index in layout.medium_indices:
            if not 0 <= index < DD:
                continue
            footprint = self.medium_footprints[index]
            if occupied & footprint:
                physical_conflicts += 1
            occupied |= footprint

        floor_holes = (
            self.all_floor ^ (self.coverage_union(layout) & self.all_floor)
        ).bit_count()
        # The fixed central 4x4 roboport is one building.  Its whole
        # footprint is physically reserved above, but it is electrically
        # powered when at least one of its sixteen tiles is reached.  Older
        # versions incorrectly required electric coverage of all 16 tiles.
        central_covered = bool(
            self.central_mask & self.electric_union(layout)
        )
        central_holes = 0 if central_covered else self.central_mask.bit_count()
        counts = self.color_counts(layout)
        color_defect = self.color_defect(counts)
        certificate = self.periodic_certificate(layout)

        reasons = []
        if bad_count:
            reasons.append("wrong object count")
        if duplicate_positions:
            reasons.append("duplicate coordinate")
        if forbidden_positions:
            reasons.append("central-roboport overlap/out-of-grid")
        if physical_conflicts:
            reasons.append("network footprints overlap")
        if color_defect:
            reasons.append("medium colors have no completion")
        if floor_holes:
            reasons.append(f"{floor_holes} uncovered 5x5 roots")
        if central_holes:
            reasons.append("central roboport has no electric coverage")
        if (
            not self.periodic_only_connectivity
            and certificate.cell_components != 1
        ):
            reasons.append(
                f"cell graph has {certificate.cell_components} components"
            )
        if not certificate.exact_periodic:
            reasons.append(
                "periodic graph does not generate the full Z^2 lattice"
            )

        feasible = not reasons
        return OracleResult(
            feasible=feasible,
            reason="ok" if feasible else "; ".join(reasons),
            floor_holes=floor_holes,
            central_holes=central_holes,
            physical_conflicts=physical_conflicts,
            forbidden_positions=forbidden_positions,
            duplicate_positions=duplicate_positions,
            color_defect=color_defect,
            certificate=certificate,
        )

    def check(
        self,
        layout: FreeCoordinateLayout,
    ) -> tuple[bool, str]:
        result = self.diagnose(layout)
        return result.feasible, result.reason


def _bit_indices(mask: int):
    while mask:
        least_bit = mask & -mask
        yield least_bit.bit_length() - 1
        mask ^= least_bit


def layout_to_binary_solution(layout: FreeCoordinateLayout) -> np.ndarray:
    """Convert coordinates to the established dense 5,006-value solution."""
    if (
        len(layout.substation_indices) != EXACT_SUBSTATIONS
        or len(set(layout.substation_indices)) != EXACT_SUBSTATIONS
        or len(layout.medium_indices) != EXACT_MEDIUMS
        or len(set(layout.medium_indices)) != EXACT_MEDIUMS
        or any(
            not 0 <= index < DD
            for index in (
                *layout.substation_indices,
                *layout.medium_indices,
            )
        )
    ):
        raise ValueError("Cannot encode a malformed free-coordinate layout.")
    network = layout.network_vector()
    witnesses = color_witness(network)
    values = np.concatenate((network, witnesses))
    if values.shape != (BINARY_SOLUTION_SIZE,):
        raise AssertionError("Binary conversion did not produce 5,006 values.")
    return values


def binary_solution_to_layout(values) -> FreeCoordinateLayout:
    values = np.asarray(values, dtype=float)
    if values.shape[0] < NETWORK_SIZE:
        raise ValueError("A Stage-A binary solution needs at least 5,000 values.")
    network = np.rint(values[:NETWORK_SIZE])
    if np.any(np.abs(values[:NETWORK_SIZE] - network) > 1e-7):
        raise ValueError("Stage-A network values are not binary.")
    substations = tuple(np.flatnonzero(network[:DD] > 0.5).tolist())
    mediums = tuple(np.flatnonzero(network[DD:NETWORK_SIZE] > 0.5).tolist())
    return FreeCoordinateLayout.create(substations, mediums)


def validate_binary_roundtrip(
    values,
    oracle: FreePeriodicOracle,
) -> tuple[FreeCoordinateLayout, OracleResult]:
    values = np.asarray(values, dtype=float)
    if values.shape != (BINARY_SOLUTION_SIZE,):
        raise ValueError(
            f"Expected exactly {BINARY_SOLUTION_SIZE} values, got "
            f"{values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Binary solution contains non-finite values.")
    layout = binary_solution_to_layout(values)
    expected = layout_to_binary_solution(layout)
    if np.max(np.abs(values - expected)) > 1e-7:
        raise ValueError("Binary solution has inconsistent color witnesses.")
    result = oracle.diagnose(layout)
    if not result.feasible:
        raise ValueError(
            f"Binary solution fails the free periodic oracle: {result.reason}"
        )
    return layout, result


def write_binary_solution(
    path: Path,
    layout: FreeCoordinateLayout,
    bound: float,
    source: str,
    result: OracleResult,
    oracle: FreePeriodicOracle,
    *,
    required: bool = True,
) -> bool:
    """Write all 5,006 x-lines densely and atomically."""
    values = layout_to_binary_solution(layout)
    decoded, decoded_result = validate_binary_roundtrip(values, oracle)
    if decoded != layout or decoded_result != result:
        raise ValueError("Dense binary roundtrip changed the validated result.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary_path(path)
    with temporary.open("w") as handle:
        handle.write("# Free-coordinate periodic Stage-A solution\n")
        handle.write("# Binary format = established 5006-variable Stage A\n")
        semantics = model_semantics(oracle)
        handle.write(
            "# Model semantics = "
            f"v{semantics['version']} "
            f"wire_centers={semantics['wire_centers']} "
            f"electric_coverage={semantics['electric_coverage']} "
            "standalone_connectivity="
            f"{semantics['standalone_connectivity_required']}\n"
        )
        handle.write(f"# Exact Stage-B LP bound = {bound:.16g}\n")
        handle.write(f"# Source = {source}\n")
        handle.write(
            "# Substations = "
            + " ".join(map(str, layout.substations))
            + "\n"
        )
        handle.write(
            "# Medium poles = "
            + " ".join(map(str, layout.medium_poles))
            + "\n"
        )
        handle.write(
            "# Periodic certificate = "
            f"cell_components={result.certificate.cell_components} "
            f"quotient_components={result.certificate.quotient_components} "
            f"rank={result.certificate.winding_rank} "
            f"index={result.certificate.lattice_index} "
            f"windings={result.certificate.windings}\n"
        )
        for index, value in enumerate(values):
            handle.write(f"x[{index}] {value:.16g}\n")
    return _replace_with_retry(
        temporary,
        path,
        required=required,
    )


def read_solution_layout(path: Path) -> FreeCoordinateLayout:
    """Read either an old Stage-A 5,000/5,006 file or a Stage-B solution."""
    values: dict[int, float] = {}
    with path.open(errors="replace") as handle:
        for line in handle:
            match = X_LINE.match(line)
            if match:
                values[int(match.group(1))] = float(match.group(2))
    if not values:
        raise ValueError(f"{path} contains no indexed x values.")

    maximum_index = max(values)
    if maximum_index >= NETWORK_SIZE + 6:
        # Stage-B packing format: substations are block 2 and mediums block 4.
        substations = [
            index - 2 * DD
            for index, value in values.items()
            if 2 * DD <= index < 3 * DD and value > 0.5
        ]
        mediums = [
            index - 4 * DD
            for index, value in values.items()
            if 4 * DD <= index < 5 * DD and value > 0.5
        ]
    else:
        substations = [
            index
            for index, value in values.items()
            if 0 <= index < DD and value > 0.5
        ]
        mediums = [
            index - DD
            for index, value in values.items()
            if DD <= index < NETWORK_SIZE and value > 0.5
        ]
    return FreeCoordinateLayout.create(substations, mediums)


class CoordinateDestroyRepair:
    """Randomized coordinate destroy/repair with exact-oracle acceptance."""

    def __init__(self, oracle: FreePeriodicOracle, args, rng: random.Random):
        self.oracle = oracle
        self.args = args
        self.rng = rng
        self.guidance_dual: np.ndarray | None = None
        self.allowed_substations = tuple(
            int(index)
            for index in np.flatnonzero(oracle.substation_allowed)
        )
        self.allowed_mediums = tuple(
            int(index)
            for index in np.flatnonzero(oracle.medium_allowed)
        )

    @staticmethod
    def translate(
        layout: FreeCoordinateLayout,
        row_shift: int,
        column_shift: int,
    ) -> FreeCoordinateLayout:
        def shifted(index):
            row, column = index_coordinate(index)
            return (
                (row + row_shift) % GRID,
                (column + column_shift) % GRID,
            )

        return FreeCoordinateLayout.create(
            [shifted(index) for index in layout.substation_indices],
            [shifted(index) for index in layout.medium_indices],
        )

    def translation_candidates(
        self,
        layout: FreeCoordinateLayout,
    ) -> list[FreeCoordinateLayout]:
        candidates = []
        radius = min(self.args.translation_radius, GRID // 2)
        for row_shift in range(-radius, radius + 1):
            for column_shift in range(-radius, radius + 1):
                if row_shift == 0 and column_shift == 0:
                    continue
                candidate = self.translate(
                    layout,
                    row_shift,
                    column_shift,
                )
                if self.oracle.diagnose(candidate).feasible:
                    candidates.append(candidate)
        return candidates

    def _balanced_skew_pair_candidates(
        self,
        parent: FreeCoordinateLayout,
    ) -> list[FreeCoordinateLayout]:
        """Exact-feasible paired pole shifts that preserve color counts."""
        offsets = (
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        )
        mediums = parent.medium_indices
        candidates = {}
        for position, left in enumerate(mediums):
            left_row, left_column = index_coordinate(left)
            for right in mediums[position + 1:]:
                right_row, right_column = index_coordinate(right)
                old_colors = sorted(
                    (medium_color(left), medium_color(right))
                )
                retained = set(mediums) - {left, right}
                for row_shift, column_shift in offsets:
                    replacements = (
                        coordinate_index(
                            (
                                (left_row + row_shift) % GRID,
                                (left_column + column_shift) % GRID,
                            )
                        ),
                        coordinate_index(
                            (
                                (right_row + row_shift) % GRID,
                                (right_column + column_shift) % GRID,
                            )
                        ),
                    )
                    if (
                        replacements[0] == replacements[1]
                        or any(
                            not self.oracle.medium_allowed[index]
                            for index in replacements
                        )
                        or any(index in retained for index in replacements)
                        or sorted(map(medium_color, replacements))
                        != old_colors
                    ):
                        continue
                    candidate = FreeCoordinateLayout.create(
                        parent.substation_indices,
                        (*retained, *replacements),
                    )
                    if (
                        candidate.distance(parent) != 2
                        or candidate.key in candidates
                        or not self.oracle.diagnose(candidate).feasible
                    ):
                        continue
                    candidates[candidate.key] = candidate
        return sorted(
            candidates.values(),
            key=lambda layout: (
                self._layout_geometry_score(layout),
                self.rng.random(),
            ),
            reverse=True,
        )

    def balanced_skew_candidates(
        self,
        parent: FreeCoordinateLayout,
    ) -> list[FreeCoordinateLayout]:
        """Interleave exact one-pair and disjoint two-pair skew moves."""
        pairs = self._balanced_skew_pair_candidates(parent)
        doubles = {}
        for first in pairs:
            for second in self._balanced_skew_pair_candidates(first):
                if second.distance(parent) != 4:
                    continue
                doubles[second.key] = second
        ranked_doubles = sorted(
            doubles.values(),
            key=lambda layout: (
                self._layout_geometry_score(layout),
                self.rng.random(),
            ),
            reverse=True,
        )
        interleaved = []
        for position in range(max(len(pairs), len(ranked_doubles))):
            if position < len(pairs):
                interleaved.append(pairs[position])
            if position < len(ranked_doubles):
                interleaved.append(ranked_doubles[position])
        return interleaved

    def _footprint(self, kind: str, index: int) -> int:
        if kind == "sub":
            return self.oracle.substation_footprints[index]
        return self.oracle.medium_footprints[index]

    def _floor_mask(self, kind: str, index: int) -> int:
        if kind == "sub":
            return self.oracle.substation_floor_masks[index]
        return self.oracle.medium_floor_masks[index]

    def _electric_mask(self, kind: str, index: int) -> int:
        if kind == "sub":
            return self.oracle.substation_electric_masks[index]
        return self.oracle.medium_electric_masks[index]

    def _occupied_mask(self, substations, mediums) -> int:
        occupied = 0
        for index in substations:
            occupied |= self.oracle.substation_footprints[index]
        for index in mediums:
            occupied |= self.oracle.medium_footprints[index]
        return occupied

    def _candidate_indices(
        self,
        kind: str,
        color: int | None,
        hints,
        *,
        discovery: bool,
        seam_needed: bool,
    ) -> list[int]:
        if kind == "sub":
            source = self.allowed_substations
        elif color is None:
            source = self.allowed_mediums
        else:
            source = self.oracle.medium_indices_by_color[color]

        sample_size = min(self.args.coordinate_pool, len(source))
        if sample_size == len(source):
            candidates = list(source)
        else:
            candidates = self.rng.sample(source, sample_size)

        radius = self.args.local_radius if not discovery else 2
        for hint in hints:
            hint_row, hint_column = index_coordinate(hint)
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    row = hint_row + dr
                    column = hint_column + dc
                    if not (0 <= row < GRID and 0 <= column < GRID):
                        continue
                    index = row * GRID + column
                    if kind == "sub":
                        allowed = self.oracle.substation_allowed[index]
                    else:
                        allowed = (
                            self.oracle.medium_allowed[index]
                            and (
                                color is None
                                or medium_color(index) == color
                            )
                        )
                    if allowed:
                        candidates.append(index)

        if seam_needed or discovery:
            boundary_width = 10 if kind == "med" else 19
            boundary_source = []
            for index in source:
                row, column = index_coordinate(index)
                if (
                    row < boundary_width
                    or row >= GRID - boundary_width
                    or column < boundary_width
                    or column >= GRID - boundary_width
                ):
                    boundary_source.append(index)
            if boundary_source:
                candidates.extend(
                    self.rng.sample(
                        boundary_source,
                        min(self.args.seam_pool, len(boundary_source)),
                    )
                )
        return list(dict.fromkeys(candidates))

    def _graph_features(
        self,
        kind: str,
        index: int,
        substations,
        mediums,
    ) -> tuple[int, int, int, int, int, int, int]:
        internal_edges = 0
        row_seams = 0
        column_seams = 0
        axial = 0
        shallow = 0
        moderate = 0
        deep = 0
        for other in substations:
            shift = self.oracle.pair_edge(kind, index, "sub", other)
            if shift is None:
                continue
            if shift == (0, 0):
                internal_edges += 1
            row_seams += int(shift[0] != 0)
            column_seams += int(shift[1] != 0)
        for other in mediums:
            shift = self.oracle.pair_edge(kind, index, "med", other)
            if shift is None:
                continue
            if shift == (0, 0):
                internal_edges += 1
            row_seams += int(shift[0] != 0)
            column_seams += int(shift[1] != 0)
            if kind == "med":
                shape = self.oracle.medium_edge_shape(index, other)
                if shape == "axial":
                    axial += 1
                elif shape == "shallow":
                    shallow += 1
                elif shape == "moderate":
                    moderate += 1
                elif shape == "deep":
                    deep += 1
        return (
            internal_edges,
            row_seams,
            column_seams,
            axial,
            shallow,
            moderate,
            deep,
        )

    def _geometry_bonus(
        self,
        axial: int,
        shallow: int,
        moderate: int,
        deep: int,
    ) -> float:
        medium_degree = axial + shallow + moderate + deep
        return (
            self.args.shallow_edge_bonus * min(2, shallow)
            - self.args.axial_edge_penalty * max(0, axial - 1)
            - self.args.moderate_edge_penalty * moderate
            - self.args.deep_edge_penalty * deep
            + self.args.degree_two_bonus * int(medium_degree == 2)
        )

    def _dual_preference(self, kind: str, index: int) -> float:
        if self.guidance_dual is None:
            return 0.0
        network_index = index if kind == "sub" else DD + index
        return -float(self.guidance_dual[network_index])

    def _dual_bonuses(self, entries) -> dict[int, float]:
        """Return a bounded rank-like bonus for candidate network variables."""
        if self.guidance_dual is None or not entries:
            return {}
        values = np.asarray(
            [
                self._dual_preference(kind, index)
                for kind, index in entries
            ],
            dtype=float,
        )
        low, high = np.quantile(values, (0.10, 0.90))
        if high <= low + 1e-12:
            return {index: 0.0 for _, index in entries}
        scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
        return {
            index: self.args.dual_guidance_weight * float(value)
            for (_, index), value in zip(entries, scaled)
        }

    def _layout_dual_delta(
        self,
        parent: FreeCoordinateLayout,
        candidate: FreeCoordinateLayout,
    ) -> float:
        if self.guidance_dual is None:
            return 0.0
        return float(
            np.sum(self.guidance_dual[parent.selected_network_indices])
            - np.sum(self.guidance_dual[candidate.selected_network_indices])
        )

    def _layout_geometry_score(
        self,
        layout: FreeCoordinateLayout,
    ) -> float:
        geometry = self.oracle.medium_geometry(layout)
        return (
            self.args.shallow_edge_bonus
            * min(geometry.shallow, self.args.target_shallow_edges)
            - self.args.shallow_excess_penalty
            * max(0, geometry.shallow - self.args.target_shallow_edges)
            - self.args.axial_edge_penalty
            * abs(geometry.axial - self.args.target_axial_edges)
            - self.args.moderate_edge_penalty * geometry.moderate
            - self.args.deep_edge_penalty * geometry.deep
            + self.args.degree_two_bonus * geometry.degree_two
            - self.args.leaf_penalty * geometry.leaves
        )

    def _compatible_patterns(
        self,
        retained_mediums,
        parent: FreeCoordinateLayout,
        *,
        discovery: bool,
    ):
        retained_counts = [0, 0, 0, 0]
        for index in retained_mediums:
            retained_counts[medium_color(index)] += 1
        compatible = [
            pattern
            for pattern in self.oracle.valid_medium_color_patterns
            if all(
                pattern[color] >= retained_counts[color]
                for color in range(4)
            )
        ]
        if not compatible:
            return []
        parent_counts = self.oracle.color_counts(parent)
        compatible.sort(
            key=lambda pattern: sum(
                abs(pattern[color] - parent_counts[color])
                for color in range(4)
            ),
            reverse=discovery,
        )
        if discovery:
            upper = max(1, min(len(compatible), len(compatible) // 2 + 1))
            return compatible[:upper]
        if parent_counts in compatible:
            return [parent_counts, *(
                pattern
                for pattern in compatible
                if pattern != parent_counts
            )]
        return compatible

    def _choose_position(
        self,
        kind: str,
        color: int | None,
        substations,
        mediums,
        floor_covered: int,
        electric_covered: int,
        hints,
        *,
        discovery: bool,
    ) -> int | None:
        floor_holes = self.oracle.all_floor & ~floor_covered
        central_holes = self.oracle.central_mask & ~electric_covered
        partial = FreeCoordinateLayout.create(substations, mediums)
        partial_certificate = self.oracle.periodic_certificate(partial)
        seam_needed = (
            partial_certificate.winding_rank < 2
            or partial_certificate.lattice_index != 1
        )
        occupied = self._occupied_mask(substations, mediums)
        used = set(substations) | set(mediums)
        candidates = self._candidate_indices(
            kind,
            color,
            hints,
            discovery=discovery,
            seam_needed=seam_needed,
        )
        scored = []
        for index in candidates:
            if index in used or occupied & self._footprint(kind, index):
                continue
            floor_gain = (
                floor_holes & self._floor_mask(kind, index)
            ).bit_count()
            central_gain = (
                central_holes & self._electric_mask(kind, index)
            ).bit_count()
            (
                internal,
                row_seams,
                column_seams,
                axial,
                shallow,
                moderate,
                deep,
            ) = self._graph_features(kind, index, substations, mediums)
            hint_distance = GRID
            if hints:
                row, column = index_coordinate(index)
                hint_distance = min(
                    abs(row - index_coordinate(hint)[0])
                    + abs(column - index_coordinate(hint)[1])
                    for hint in hints
                )
            distance_term = (
                0.15 * hint_distance
                if discovery
                else -0.20 * hint_distance
            )
            score = (
                10.0 * floor_gain
                + 60.0 * central_gain
                + 28.0 * min(2, internal)
                + 45.0 * (row_seams + column_seams)
                + self._geometry_bonus(
                    axial,
                    shallow,
                    moderate,
                    deep,
                )
                + distance_term
                + self.rng.random() * self.args.construction_noise
            )
            scored.append((score, index))
        if not scored:
            return None
        dual_bonuses = self._dual_bonuses(
            [(kind, index) for _, index in scored]
        )
        scored = [
            (score + dual_bonuses.get(index, 0.0), index)
            for score, index in scored
        ]
        scored.sort(reverse=True)
        elite = scored[: min(self.args.construction_elite, len(scored))]
        weights = [
            1.0 / (1 + position)
            for position in range(len(elite))
        ]
        return self.rng.choices(
            [item[1] for item in elite],
            weights=weights,
            k=1,
        )[0]

    def _construct_from_retained(
        self,
        parent: FreeCoordinateLayout,
        retained_substations,
        retained_mediums,
        substation_hints,
        medium_hints,
        donor: FreeCoordinateLayout | None,
        *,
        discovery: bool,
    ) -> FreeCoordinateLayout | None:
        substations = list(retained_substations)
        mediums = list(retained_mediums)
        patterns = self._compatible_patterns(
            mediums,
            parent,
            discovery=discovery,
        )
        if not patterns:
            return None
        pattern = self.rng.choice(patterns[: min(12, len(patterns))])
        retained_counts = [0, 0, 0, 0]
        for index in mediums:
            retained_counts[medium_color(index)] += 1
        missing_colors = [
            color
            for color in range(4)
            for _ in range(pattern[color] - retained_counts[color])
        ]
        self.rng.shuffle(missing_colors)

        roles = [
            *("sub" for _ in range(EXACT_SUBSTATIONS - len(substations))),
            *(f"med:{color}" for color in missing_colors),
        ]
        # Substations cover much more floor, so place one first; alternate the
        # remainder to let connectivity influence both object types.
        roles.sort(
            key=lambda role: (
                0 if role == "sub" and len(substations) < 3 else 1,
                self.rng.random(),
            )
        )
        floor_covered = 0
        electric_covered = 0
        for index in substations:
            floor_covered |= self.oracle.substation_floor_masks[index]
            electric_covered |= self.oracle.substation_electric_masks[index]
        for index in mediums:
            floor_covered |= self.oracle.medium_floor_masks[index]
            electric_covered |= self.oracle.medium_electric_masks[index]

        donor_substations = (
            () if donor is None else donor.substation_indices
        )
        donor_mediums = () if donor is None else donor.medium_indices
        for role in roles:
            if role == "sub":
                kind = "sub"
                color = None
                hints = (*substation_hints, *donor_substations)
            else:
                kind = "med"
                color = int(role.split(":")[1])
                hints = tuple(
                    index
                    for index in (*medium_hints, *donor_mediums)
                    if medium_color(index) == color
                )
            index = self._choose_position(
                kind,
                color,
                substations,
                mediums,
                floor_covered,
                electric_covered,
                hints,
                discovery=discovery,
            )
            if index is None:
                return None
            if kind == "sub":
                substations.append(index)
                floor_covered |= self.oracle.substation_floor_masks[index]
                electric_covered |= self.oracle.substation_electric_masks[index]
            else:
                mediums.append(index)
                floor_covered |= self.oracle.medium_floor_masks[index]
                electric_covered |= self.oracle.medium_electric_masks[index]
        return FreeCoordinateLayout.create(substations, mediums)

    def _replacement_candidates(
        self,
        layout: FreeCoordinateLayout,
        kind: str,
        position: int,
        diagnosis: OracleResult,
        *,
        discovery: bool,
    ):
        if kind == "sub":
            old_index = layout.substation_indices[position]
            other_substations = list(layout.substation_indices)
            del other_substations[position]
            other_mediums = list(layout.medium_indices)
            color = None
        else:
            old_index = layout.medium_indices[position]
            other_substations = list(layout.substation_indices)
            other_mediums = list(layout.medium_indices)
            del other_mediums[position]
            color = medium_color(old_index)

        floor_covered = 0
        electric_covered = 0
        for index in other_substations:
            floor_covered |= self.oracle.substation_floor_masks[index]
            electric_covered |= self.oracle.substation_electric_masks[index]
        for index in other_mediums:
            floor_covered |= self.oracle.medium_floor_masks[index]
            electric_covered |= self.oracle.medium_electric_masks[index]
        floor_holes = self.oracle.all_floor & ~floor_covered
        central_holes = self.oracle.central_mask & ~electric_covered
        occupied = self._occupied_mask(other_substations, other_mediums)
        candidates = self._candidate_indices(
            kind,
            color,
            (old_index,),
            discovery=discovery,
            seam_needed=not diagnosis.certificate.exact_periodic,
        )
        approximate = []
        for index in candidates:
            if index == old_index or occupied & self._footprint(kind, index):
                continue
            if index in set(other_substations) | set(other_mediums):
                continue
            floor_after = (
                floor_holes & ~self._floor_mask(kind, index)
            ).bit_count()
            central_after = (
                central_holes & ~self._electric_mask(kind, index)
            ).bit_count()
            (
                internal,
                row_seams,
                column_seams,
                axial,
                shallow,
                moderate,
                deep,
            ) = self._graph_features(
                kind,
                index,
                other_substations,
                other_mediums,
            )
            approximate_score = (
                floor_after * 10
                + central_after * 250
                - min(2, internal) * 40
                - (row_seams + column_seams) * (
                    80 if not diagnosis.certificate.exact_periodic else 12
                )
                - self._geometry_bonus(
                    axial,
                    shallow,
                    moderate,
                    deep,
                )
                + self.rng.random() * 10
            )
            approximate.append((approximate_score, index))
        dual_bonuses = self._dual_bonuses(
            [(kind, index) for _, index in approximate]
        )
        approximate = [
            (score - dual_bonuses.get(index, 0.0), index)
            for score, index in approximate
        ]
        approximate.sort()
        for _, index in approximate[: self.args.exact_repair_pool]:
            if kind == "sub":
                candidate = FreeCoordinateLayout.create(
                    [*other_substations, index],
                    other_mediums,
                )
            else:
                candidate = FreeCoordinateLayout.create(
                    other_substations,
                    [*other_mediums, index],
                )
            yield candidate

    def _repair_full(
        self,
        raw: FreeCoordinateLayout,
        parent: FreeCoordinateLayout,
        *,
        discovery: bool,
        minimum_distance: int,
    ) -> FreeCoordinateLayout | None:
        current = raw
        current_result = self.oracle.diagnose(current)
        best = current
        best_result = current_result
        if (
            current_result.feasible
            and current.distance(parent) >= minimum_distance
        ):
            return current

        for step in range(self.args.repair_steps):
            kind = (
                "med"
                if self.rng.random() < EXACT_MEDIUMS / (
                    EXACT_SUBSTATIONS + EXACT_MEDIUMS
                )
                else "sub"
            )
            length = (
                len(current.medium_indices)
                if kind == "med"
                else len(current.substation_indices)
            )
            if not length:
                continue
            position = self.rng.randrange(length)
            exact_candidates = list(
                self._replacement_candidates(
                    current,
                    kind,
                    position,
                    current_result,
                    discovery=discovery,
                )
            )
            if not exact_candidates:
                continue
            scored = [
                (self.oracle.diagnose(candidate), candidate)
                for candidate in exact_candidates
            ]
            scored.sort(
                key=lambda item: (
                    item[0].damage,
                    -self._layout_dual_delta(parent, item[1]),
                    -self._layout_geometry_score(item[1]),
                )
            )
            candidate_result, candidate = scored[0]
            temperature = max(
                1.0,
                self.args.repair_temperature
                * (1.0 - step / max(1, self.args.repair_steps)),
            )
            improvement = current_result.damage - candidate_result.damage
            accept = improvement >= 0
            if (
                not accept
                and discovery
                and self.rng.random()
                < math.exp(max(-50.0, improvement / temperature))
            ):
                accept = True
            if accept:
                current = candidate
                current_result = candidate_result
            if candidate_result.damage < best_result.damage:
                best = candidate
                best_result = candidate_result
            if (
                candidate_result.feasible
                and candidate.distance(parent) >= minimum_distance
            ):
                return candidate
        if best_result.feasible and best.distance(parent) >= minimum_distance:
            return best
        return None

    def _seam_roles(self, layout: FreeCoordinateLayout):
        nodes = [
            *(("sub", index) for index in layout.substation_indices),
            *(("med", index) for index in layout.medium_indices),
        ]
        edges = self.oracle.periodic_edges(layout)
        row_edges = [
            edge for edge in edges if edge[2][0] != 0
        ]
        column_edges = [
            edge for edge in edges if edge[2][1] != 0
        ]
        if not row_edges or not column_edges:
            return ()
        pairs = []
        for row_edge in row_edges:
            for column_edge in column_edges:
                endpoints = {
                    row_edge[0],
                    row_edge[1],
                    column_edge[0],
                    column_edge[1],
                }
                pairs.append((len(endpoints), self.rng.random(), endpoints))
        _, _, endpoints = min(pairs)
        return tuple(nodes[node] for node in endpoints)

    @staticmethod
    def _role_accepts(role, index):
        kind, old_index = role
        return kind == "sub" or medium_color(index) == medium_color(old_index)

    def _random_shared_scaffold(self, roles, remaining_nodes):
        if len(roles) != 3:
            return None
        root_role, horizontal_role, vertical_role = roles
        root_kind = root_role[0]
        horizontal_kind = horizontal_role[0]
        vertical_kind = vertical_role[0]
        for _ in range(200):
            root_row = self.rng.randrange(0, 10)
            root_column = self.rng.randrange(0, 10)
            root = root_row * GRID + root_column
            if (
                root_kind == "sub"
                and not self.oracle.substation_allowed[root]
            ) or (
                root_kind == "med"
                and not self.oracle.medium_allowed[root]
            ):
                continue
            if not self._role_accepts(root_role, root):
                continue
            if root in {old_index for _, old_index in roles}:
                continue
            horizontal_candidates = []
            vertical_candidates = []
            for _ in range(80):
                horizontal = (
                    self.rng.randrange(0, 20) * GRID
                    + self.rng.randrange(GRID - 10, GRID)
                )
                shift = self.oracle.pair_edge(
                    root_kind,
                    root,
                    horizontal_kind,
                    horizontal,
                )
                if (
                    shift is not None
                    and shift[0] == 0
                    and shift[1] != 0
                    and self._role_accepts(horizontal_role, horizontal)
                    and horizontal
                    not in {old_index for _, old_index in roles}
                ):
                    horizontal_candidates.append(horizontal)

                vertical = (
                    self.rng.randrange(GRID - 10, GRID) * GRID
                    + self.rng.randrange(0, 20)
                )
                shift = self.oracle.pair_edge(
                    root_kind,
                    root,
                    vertical_kind,
                    vertical,
                )
                if (
                    shift is not None
                    and shift[0] != 0
                    and shift[1] == 0
                    and self._role_accepts(vertical_role, vertical)
                    and vertical
                    not in {old_index for _, old_index in roles}
                ):
                    vertical_candidates.append(vertical)
            if horizontal_candidates and vertical_candidates:
                self.rng.shuffle(horizontal_candidates)
                self.rng.shuffle(vertical_candidates)
                for horizontal in horizontal_candidates:
                    for vertical in vertical_candidates:
                        scaffold = (
                            (root_kind, root),
                            (horizontal_kind, horizontal),
                            (vertical_kind, vertical),
                        )
                        if len({index for _, index in scaffold}) != 3:
                            continue
                        occupied = 0
                        physically_valid = True
                        for kind, index in scaffold:
                            footprint = self._footprint(kind, index)
                            if occupied & footprint:
                                physically_valid = False
                                break
                            occupied |= footprint
                        if not physically_valid:
                            continue
                        # Each new seam endpoint must also attach to the
                        # retained zero-shift cell graph.  A wrapped seam edge
                        # alone is tileability, not within-cell connectivity.
                        attaches = []
                        for kind, index in scaffold:
                            attaches.append(
                                any(
                                    self.oracle.pair_edge(
                                        kind,
                                        index,
                                        other_kind,
                                        other_index,
                                    )
                                    == (0, 0)
                                    for other_kind, other_index in remaining_nodes
                                )
                            )
                        if not all(attaches):
                            continue
                        trial_substations = [
                            index
                            for kind, index in (*remaining_nodes, *scaffold)
                            if kind == "sub"
                        ]
                        trial_mediums = [
                            index
                            for kind, index in (*remaining_nodes, *scaffold)
                            if kind == "med"
                        ]
                        trial = FreeCoordinateLayout.create(
                            trial_substations,
                            trial_mediums,
                        )
                        certificate = self.oracle.periodic_certificate(trial)
                        if (
                            certificate.cell_components == 1
                            and certificate.exact_periodic
                        ):
                            return scaffold
        return None

    def _scaffold_raw(
        self,
        parent: FreeCoordinateLayout,
    ) -> FreeCoordinateLayout | None:
        roles = self._seam_roles(parent)
        if len(roles) != 3:
            return None
        # Put the shared endpoint first.  The two old seam edges share it in
        # the normal three-anchor certificate.
        role_counts = {}
        for kind, index in roles:
            count = 0
            for left, right, shift in self.oracle.periodic_edges(parent):
                if shift == (0, 0):
                    continue
                nodes = [
                    *(("sub", item) for item in parent.substation_indices),
                    *(("med", item) for item in parent.medium_indices),
                ]
                if nodes[left] == (kind, index) or nodes[right] == (kind, index):
                    count += 1
            role_counts[(kind, index)] = count
        ordered_roles = sorted(
            roles,
            key=lambda role: role_counts[role],
            reverse=True,
        )
        remaining_nodes = [
            *(("sub", index) for index in parent.substation_indices),
            *(("med", index) for index in parent.medium_indices),
        ]
        for role in ordered_roles:
            remaining_nodes.remove(role)
        scaffold = self._random_shared_scaffold(
            tuple(ordered_roles),
            tuple(remaining_nodes),
        )
        if scaffold is None:
            return None

        substations = list(parent.substation_indices)
        mediums = list(parent.medium_indices)
        for kind, index in ordered_roles:
            target = substations if kind == "sub" else mediums
            if index in target:
                target.remove(index)
        for kind, index in scaffold:
            target = substations if kind == "sub" else mediums
            target.append(index)
        return FreeCoordinateLayout.create(substations, mediums)

    def _snake_destroy_set(
        self,
        parent: FreeCoordinateLayout,
        count: int,
    ) -> tuple[int, ...]:
        """Choose a connected medium segment, favoring its axial runs."""
        mediums = parent.medium_indices
        axial_adjacency = {index: set() for index in mediums}
        all_adjacency = {index: set() for index in mediums}
        for position, left in enumerate(mediums):
            for right in mediums[position + 1:]:
                shape = self.oracle.medium_edge_shape(left, right)
                if shape is None:
                    continue
                all_adjacency[left].add(right)
                all_adjacency[right].add(left)
                if shape == "axial":
                    axial_adjacency[left].add(right)
                    axial_adjacency[right].add(left)

        components = []
        unseen = set(mediums)
        while unseen:
            root = unseen.pop()
            component = {root}
            stack = [root]
            while stack:
                current = stack.pop()
                for neighbor in all_adjacency[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            if len(component) >= count:
                components.append(component)
        if not components:
            return ()
        component = self.rng.choices(
            components,
            weights=[
                1 + sum(
                    len(axial_adjacency[index])
                    for index in candidate
                )
                for candidate in components
            ],
            k=1,
        )[0]
        weighted_seeds = [
            index
            for index in component
            for _ in range(1 + 2 * len(axial_adjacency[index]))
        ]
        selected = {self.rng.choice(weighted_seeds)}
        while len(selected) < count:
            axial_frontier = {
                neighbor
                for index in selected
                for neighbor in axial_adjacency[index]
                if neighbor in component and neighbor not in selected
            }
            connected_frontier = {
                neighbor
                for index in selected
                for neighbor in all_adjacency[index]
                if neighbor in component and neighbor not in selected
            }
            frontier = axial_frontier or connected_frontier
            if not frontier:
                return ()
            selected.add(self.rng.choice(tuple(frontier)))
        return tuple(selected)

    def _exact_skew_walk_candidate(
        self,
        parent: FreeCoordinateLayout,
        *,
        discovery: bool,
        move_limit: int | None = None,
    ) -> FreeCoordinateLayout | None:
        """Walk three to six medium poles through exact-feasible skew states."""
        minimum = max(
            3,
            self.args.snake_min_poles,
            self.args.discovery_min_changes if discovery else 3,
        )
        maximum = min(
            self.args.snake_max_poles,
            6,
            move_limit
            if not discovery and move_limit is not None
            else self.args.improvement_move_limit
            if not discovery
            else self.args.discovery_max_changes,
        )
        if minimum > maximum:
            return None

        target = self.rng.randint(minimum, maximum)
        beam_width = max(12, self.args.snake_attempts)
        parent_mediums = frozenset(parent.medium_indices)
        parent_geometry = self.oracle.medium_geometry(parent)
        # score, layout, connected active region, removed parent mediums
        beam = [
            (
                0.0,
                parent,
                frozenset(),
                frozenset(),
            )
        ]
        deepest = []

        for depth in range(1, target + 1):
            next_states = {}
            for _, current, active_region, removed_before in beam:
                current_result = self.oracle.diagnose(current)
                if not current_result.feasible:
                    continue
                current_mediums = set(current.medium_indices)
                untouched = current_mediums & parent_mediums

                for old_index in untouched:
                    old_neighbors = {
                        other
                        for other in current_mediums
                        if (
                            other != old_index
                            and self.oracle.medium_edge_shape(
                                old_index,
                                other,
                            )
                            is not None
                        )
                    }
                    if not active_region:
                        if not any(
                            self.oracle.medium_edge_shape(
                                old_index,
                                other,
                            )
                            == "axial"
                            for other in old_neighbors
                        ):
                            continue
                        anchors = old_neighbors
                    else:
                        anchors = set(active_region) - {old_index}
                        if not any(
                            self.oracle.medium_edge_shape(
                                old_index,
                                other,
                            )
                            is not None
                            for other in anchors
                        ):
                            continue

                    position = current.medium_indices.index(old_index)
                    for candidate in self._replacement_candidates(
                        current,
                        "med",
                        position,
                        current_result,
                        discovery=False,
                    ):
                        if candidate.substation_indices != parent.substation_indices:
                            continue
                        candidate_result = self.oracle.diagnose(candidate)
                        if not candidate_result.feasible:
                            continue

                        candidate_mediums = set(candidate.medium_indices)
                        inserted = candidate_mediums - current_mediums
                        removed = parent_mediums - candidate_mediums
                        if (
                            len(inserted) != 1
                            or len(removed) != depth
                            or not removed_before <= removed
                        ):
                            continue
                        new_index = next(iter(inserted))
                        if not any(
                            self.oracle.medium_edge_shape(
                                new_index,
                                anchor,
                            )
                            is not None
                            for anchor in anchors
                        ):
                            continue

                        raw_region = (
                            (set(active_region) - {old_index})
                            | {new_index}
                            | old_neighbors
                        ) & candidate_mediums
                        connected_region = {new_index}
                        frontier = [new_index]
                        while frontier:
                            current_index = frontier.pop()
                            for other in raw_region - connected_region:
                                if (
                                    self.oracle.medium_edge_shape(
                                        current_index,
                                        other,
                                    )
                                    is not None
                                ):
                                    connected_region.add(other)
                                    frontier.append(other)

                        shape_counts = {
                            "axial": 0,
                            "shallow": 0,
                            "moderate": 0,
                            "deep": 0,
                        }
                        region_list = list(connected_region)
                        for left, left_index in enumerate(region_list):
                            for right_index in region_list[left + 1:]:
                                shape = self.oracle.medium_edge_shape(
                                    left_index,
                                    right_index,
                                )
                                if shape is not None:
                                    shape_counts[shape] += 1

                        geometry = self.oracle.medium_geometry(candidate)
                        dual_delta = max(
                            -100.0,
                            min(
                                100.0,
                                self._layout_dual_delta(
                                    parent,
                                    candidate,
                                ),
                            ),
                        )
                        score = (
                            20.0 * shape_counts["shallow"]
                            + 8.0 * shape_counts["moderate"]
                            - 8.0 * shape_counts["axial"]
                            - 12.0 * shape_counts["deep"]
                            + 6.0 * (
                                parent_geometry.axial
                                - geometry.axial
                            )
                            + 3.0 * geometry.degree_two
                            - 3.0 * geometry.leaves
                            + 0.02 * dual_delta
                            + self.rng.random() * 1e-6
                        )
                        previous = next_states.get(candidate.key)
                        if previous is None or score > previous[0]:
                            next_states[candidate.key] = (
                                score,
                                candidate,
                                frozenset(connected_region),
                                frozenset(removed),
                            )

            beam = sorted(
                next_states.values(),
                key=lambda state: state[0],
                reverse=True,
            )[:beam_width]
            if not beam:
                break
            if depth >= minimum:
                deepest = beam

        if not deepest:
            return None
        skewed = []
        for state in deepest:
            candidate = state[1]
            added = set(candidate.medium_indices) - parent_mediums
            nonaxial_edges = 0
            for left, left_index in enumerate(candidate.medium_indices):
                for right_index in candidate.medium_indices[left + 1:]:
                    if (
                        left_index not in added
                        and right_index not in added
                    ):
                        continue
                    shape = self.oracle.medium_edge_shape(
                        left_index,
                        right_index,
                    )
                    if shape in {"shallow", "moderate", "deep"}:
                        nonaxial_edges += 1
            if nonaxial_edges >= min(2, len(added) - 1):
                skewed.append(state)
        if not skewed:
            return None
        elite = skewed[: min(3, len(skewed))]
        return self.rng.choice(elite)[1]

    def _shallow_snake_candidate(
        self,
        parent: FreeCoordinateLayout,
        donor: FreeCoordinateLayout | None,
        *,
        discovery: bool,
        move_limit: int | None = None,
    ) -> FreeCoordinateLayout | None:
        """Jointly rebuild a medium segment into a shallow zig-zag."""
        minimum = max(
            self.args.snake_min_poles,
            self.args.discovery_min_changes if discovery else 1,
        )
        maximum = min(
            self.args.snake_max_poles,
            EXACT_MEDIUMS,
            move_limit
            if not discovery and move_limit is not None
            else self.args.improvement_move_limit
            if not discovery
            else EXACT_MEDIUMS,
        )
        if minimum > maximum:
            return None
        minimum_distance = (
            self.args.discovery_min_changes
            if discovery
            else self.args.snake_min_poles
        )
        feasible = []
        best_raw = None
        best_result = None
        parent_geometry_score = self._layout_geometry_score(parent)
        for _ in range(self.args.snake_attempts):
            destroyed = self._snake_destroy_set(
                parent,
                self.rng.randint(minimum, maximum),
            )
            if not destroyed:
                continue
            destroyed_set = set(destroyed)
            retained_mediums = [
                index
                for index in parent.medium_indices
                if index not in destroyed_set
            ]
            raw = self._construct_from_retained(
                parent,
                parent.substation_indices,
                retained_mediums,
                (),
                destroyed,
                donor,
                discovery=discovery,
            )
            if raw is None:
                continue
            result = self.oracle.diagnose(raw)
            if (
                result.feasible
                and raw.distance(parent)
                >= minimum_distance
            ):
                geometry_delta = (
                    self._layout_geometry_score(raw)
                    - parent_geometry_score
                )
                dual_delta = self._layout_dual_delta(parent, raw)
                if discovery:
                    rank = (
                        geometry_delta,
                        raw.relative_distance(parent),
                        dual_delta,
                        self.rng.random(),
                    )
                else:
                    rank = (
                        geometry_delta,
                        dual_delta,
                        self.rng.random(),
                    )
                feasible.append((rank, raw))
            if best_result is None or result.damage < best_result.damage:
                best_raw, best_result = raw, result

        if feasible:
            feasible.sort(key=lambda item: item[0], reverse=True)
            elite = feasible[: min(3, len(feasible))]
            return self.rng.choice(elite)[1]
        if best_raw is None:
            return None
        return self._repair_full(
            best_raw,
            parent,
            discovery=discovery,
            minimum_distance=minimum_distance,
        )

    def _destroy_repair_candidate(
        self,
        parent: FreeCoordinateLayout,
        donor: FreeCoordinateLayout | None,
        *,
        discovery: bool,
        initial_raw: FreeCoordinateLayout | None = None,
    ) -> FreeCoordinateLayout | None:
        minimum_distance = (
            self.args.discovery_min_changes if discovery else 1
        )
        best_raw = initial_raw
        best_result = (
            None
            if initial_raw is None
            else self.oracle.diagnose(initial_raw)
        )
        if (
            best_raw is not None
            and best_result is not None
            and best_result.feasible
            and best_raw.distance(parent) >= minimum_distance
        ):
            return best_raw

        for _ in range(self.args.construction_attempts):
            if discovery:
                destroy_count = self.rng.randint(
                    self.args.discovery_min_changes,
                    min(
                        self.args.discovery_max_changes,
                        EXACT_SUBSTATIONS + EXACT_MEDIUMS,
                    ),
                )
            else:
                destroy_count = self.rng.randint(
                    1,
                    min(self.args.local_destroy, EXACT_SUBSTATIONS + EXACT_MEDIUMS),
                )
            all_roles = [
                *(("sub", index) for index in parent.substation_indices),
                *(("med", index) for index in parent.medium_indices),
            ]
            destroyed = self.rng.sample(all_roles, destroy_count)
            destroyed_set = set(destroyed)
            retained_substations = [
                index
                for index in parent.substation_indices
                if ("sub", index) not in destroyed_set
            ]
            retained_mediums = [
                index
                for index in parent.medium_indices
                if ("med", index) not in destroyed_set
            ]
            raw = self._construct_from_retained(
                parent,
                retained_substations,
                retained_mediums,
                [
                    index
                    for kind, index in destroyed
                    if kind == "sub"
                ],
                [
                    index
                    for kind, index in destroyed
                    if kind == "med"
                ],
                donor,
                discovery=discovery,
            )
            if raw is None or raw.distance(parent) < minimum_distance:
                continue
            result = self.oracle.diagnose(raw)
            if result.feasible:
                return raw
            if best_result is None or result.damage < best_result.damage:
                best_raw, best_result = raw, result

        if best_raw is None:
            return None
        return self._repair_full(
            best_raw,
            parent,
            discovery=discovery,
            minimum_distance=minimum_distance,
        )

    def _feasible_walk(
        self,
        parent: FreeCoordinateLayout,
        donor: FreeCoordinateLayout | None,
    ) -> FreeCoordinateLayout | None:
        target_distance = self.rng.randint(
            self.args.discovery_min_changes,
            self.args.discovery_max_changes,
        )
        current = parent
        current_distance = 0
        failed_steps = 0
        maximum_steps = max(
            target_distance * self.args.walk_step_factor,
            target_distance,
        )
        for _ in range(maximum_steps):
            candidate = self._destroy_repair_candidate(
                current,
                donor,
                discovery=False,
            )
            if candidate is None:
                failed_steps += 1
                if failed_steps >= self.args.walk_step_factor:
                    break
                continue
            distance = candidate.distance(parent)
            # The walk is allowed to move sideways but never to erase more
            # than one accumulated change.  This prevents four nominal local
            # moves from merely oscillating around the same incumbent.
            if distance + 1 < current_distance:
                continue
            current = candidate
            current_distance = distance
            if current_distance >= target_distance:
                return current
        if current_distance >= self.args.discovery_min_changes:
            return current
        return None

    def generate(
        self,
        parent: FreeCoordinateLayout,
        donor: FreeCoordinateLayout | None,
        *,
        discovery: bool,
        guidance_dual: np.ndarray | None = None,
        basin: bool = False,
        move_limit: int | None = None,
    ) -> FreeCoordinateLayout | None:
        self.guidance_dual = guidance_dual
        if not discovery:
            snake_probability = (
                self.args.basin_snake_probability
                if basin
                else self.args.improvement_snake_probability
            )
            if self.rng.random() < snake_probability:
                skew_walk = self._exact_skew_walk_candidate(
                    parent,
                    discovery=False,
                    move_limit=move_limit,
                )
                if skew_walk is not None:
                    return skew_walk
                snake = self._shallow_snake_candidate(
                    parent,
                    donor,
                    discovery=False,
                    move_limit=move_limit,
                )
                if snake is not None:
                    return snake
            return self._destroy_repair_candidate(
                parent,
                donor,
                discovery=False,
            )

        if self.rng.random() < self.args.translation_probability:
            translations = self.translation_candidates(parent)
            if translations:
                return self.rng.choice(translations)

        if self.rng.random() < self.args.discovery_snake_probability:
            skew_walk = self._exact_skew_walk_candidate(
                parent,
                discovery=True,
            )
            if skew_walk is not None:
                return skew_walk
            snake = self._shallow_snake_candidate(
                parent,
                donor,
                discovery=True,
            )
            if snake is not None:
                return snake

        if self.rng.random() < self.args.walk_probability:
            walked = self._feasible_walk(parent, donor)
            if walked is not None:
                return walked

        scaffold_raw = None
        if self.rng.random() < self.args.scaffold_probability:
            scaffold_raw = self._scaffold_raw(parent)
        return self._destroy_repair_candidate(
            parent,
            donor,
            discovery=True,
            initial_raw=scaffold_raw,
        )


@dataclass
class ExactRecord:
    layout: FreeCoordinateLayout
    bound: float
    runtime: float
    equality_dual: np.ndarray
    phase: str
    generation: int


def discover_seed_layouts(
    oracle: FreePeriodicOracle,
    explicit_paths,
    count: int,
    scan_limit: int,
):
    candidates: list[Path] = []
    seen_paths = set()

    def add(path):
        path = Path(path)
        normalized = str(path.resolve()) if path.exists() else str(path)
        if normalized not in seen_paths:
            seen_paths.add(normalized)
            candidates.append(path)

    for path in explicit_paths:
        add(path)

    preferred = Path(
        "coverage/50x50/20260730_095710_balanced_multibasin/"
        "best_balanced_multibasin.sol"
    )
    if preferred.exists():
        add(preferred)

    coverage_root = Path("coverage/50x50")
    if coverage_root.exists() and scan_limit > 0:
        discovered = list(coverage_root.rglob("best*.sol"))
        discovered.sort(
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in discovered[:scan_limit]:
            add(path)

    layouts = []
    seen_layouts = set()
    rejected = []
    for path in candidates:
        if len(layouts) >= count:
            break
        if not path.exists():
            rejected.append((path, "file does not exist"))
            continue
        try:
            layout = read_solution_layout(path)
            result = oracle.diagnose(layout)
        except (OSError, ValueError) as error:
            rejected.append((path, str(error)))
            continue
        if not result.feasible:
            rejected.append((path, result.reason))
            continue
        if layout.key in seen_layouts:
            continue
        seen_layouts.add(layout.key)
        layouts.append((layout, path, result))

    if explicit_paths:
        explicit_set = {str(Path(path)) for path in explicit_paths}
        for path, reason in rejected:
            if str(path) in explicit_set:
                print(f"rejected explicit seed {path}: {reason}", flush=True)
    return layouts


class FreeCoordinateSearch:
    def __init__(self, args, oracle, seed_records):
        self.args = args
        self.oracle = oracle
        self.seed_records = seed_records
        self.random_seed = (
            int(args.seed)
            if args.seed is not None
            else time.time_ns() & 0x7FFF_FFFF
        )
        self.rng = random.Random(self.random_seed)
        self.generator = CoordinateDestroyRepair(
            oracle,
            args,
            self.rng,
        )
        self.exact_records: dict[tuple[int, ...], ExactRecord] = {}
        self.relative_records: dict[tuple[int, ...], ExactRecord] = {}
        self.relative_counts: dict[tuple[int, ...], int] = {}
        self.signature_records: dict[tuple[int, ...], ExactRecord] = {}
        self.parent_uses: dict[tuple[int, ...], int] = {}
        self.audited_relative_keys: set[tuple[int, ...]] = set()
        self.seen_layouts: set[tuple[int, ...]] = set()
        self.best: ExactRecord | None = None
        self.evaluation_number = 0
        self.incumbent_number = 0
        self.local_seconds = 0.0
        self.discovery_seconds = 0.0
        self.local_units = 0
        self.discovery_units = 0
        self.cut_pool = DualUpperBoundPool(
            maximum_cuts=args.dual_cut_pool,
            safety=args.dual_safety,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{timestamp}_free_coordinate"
        self.folder = (
            args.output
            if args.output is not None
            else Path("coverage/50x50") / self.run_name
        )
        for _, seed_path, _ in seed_records:
            if (
                seed_path.parent.resolve() == self.folder.resolve()
                and (
                    self.folder / "free_coordinate_progress.csv"
                ).exists()
            ):
                raise ValueError(
                    "The output folder is an input run folder; choose a new "
                    "--output so its progress history is not overwritten."
                )
        self.folder.mkdir(parents=True, exist_ok=True)
        write_model_semantics(self.folder, oracle)
        self.csv_handle = (self.folder / "free_coordinate_progress.csv").open(
            "w",
            newline="",
            buffering=1,
        )
        self.writer = csv.writer(self.csv_handle)
        self.writer.writerow(
            [
                "evaluation_id",
                "phase",
                "generation",
                "positive_bound",
                "lp_runtime",
                "is_new_best",
                "distance_from_best",
                "substations",
                "medium_poles",
                "medium_colors",
                "relative_family_count",
                "relative_distance_from_best",
                "medium_edges",
                "medium_axial",
                "medium_shallow",
                "medium_moderate",
                "medium_deep",
                "medium_degree_two",
                "medium_leaves",
                "cell_components",
                "winding_rank",
                "lattice_index",
                "solution_path",
            ]
        )

        initial_network = seed_records[0][0].network_vector()
        lp_limit = (
            GRB.INFINITY
            if args.lp_seconds <= 0
            else args.lp_seconds
        )
        self.evaluator = ParallelStageBEvaluator(
            initial_network,
            args.workers,
            lp_limit,
            periodic_electric_coverage=oracle.true_periodic_coverage,
        )

    def close(self):
        self.evaluator.close()
        self.csv_handle.close()

    def _load_seen_histories(self):
        if self.args.ignore_seen_history:
            return
        history_names = (
            "free_coordinate_progress.csv",
            "refinement_progress.csv",
            "distance_refinement_progress.csv",
            "triple_refinement_progress.csv",
        )
        progress_paths = {
            seed_path.parent / history_name
            for _, seed_path, _ in self.seed_records
            for history_name in history_names
        }
        history_roots = self.args.history_root or [Path("coverage/50x50")]
        for history_root in history_roots:
            if not history_root.exists():
                continue
            for history_name in history_names:
                progress_paths.update(
                    history_root.rglob(history_name)
                )
        progress_paths = {
            path
            for path in progress_paths
            if path.parent.resolve() != self.folder.resolve()
        }
        loaded = 0
        used_paths = 0
        for progress_path in sorted(progress_paths):
            if not progress_path.exists():
                continue
            if not progress_matches_model_semantics(
                progress_path,
                self.oracle,
            ):
                continue
            used_paths += 1
            try:
                with progress_path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        try:
                            layout = FreeCoordinateLayout.create(
                                ast.literal_eval(row["substations"]),
                                ast.literal_eval(row["medium_poles"]),
                            )
                        except (
                            KeyError,
                            SyntaxError,
                            TypeError,
                            ValueError,
                        ):
                            continue
                        if (
                            len(layout.substation_indices)
                            != EXACT_SUBSTATIONS
                            or len(layout.medium_indices)
                            != EXACT_MEDIUMS
                            or layout.key in self.seen_layouts
                        ):
                            continue
                        self.seen_layouts.add(layout.key)
                        relative_key = layout.relative_key
                        self.relative_counts[relative_key] = (
                            self.relative_counts.get(relative_key, 0) + 1
                        )
                        loaded += 1
            except OSError as error:
                print(
                    f"could not reuse seen history {progress_path}: {error}",
                    flush=True,
                )
        if loaded:
            print(
                f"reused {loaded} already-evaluated layouts from "
                f"{used_paths} prior free-coordinate run(s)",
                flush=True,
            )

    def _persist(
        self,
        record: ExactRecord,
        result: OracleResult,
        *,
        latest: bool,
        best: bool,
        save_all: bool,
    ) -> str:
        source = f"{record.phase} generation {record.generation}"
        solution_path = ""
        if latest:
            latest_path = self.folder / "latest_feasible.sol"
            write_binary_solution(
                latest_path,
                record.layout,
                record.bound,
                source,
                result,
                self.oracle,
                required=False,
            )
        if save_all:
            candidate_path = self.folder / (
                f"candidate_{self.evaluation_number:07d}.sol"
            )
            write_binary_solution(
                candidate_path,
                record.layout,
                record.bound,
                source,
                result,
                self.oracle,
            )
            solution_path = candidate_path.name
        if best:
            numbered_path = self.folder / (
                f"incumbent_{self.incumbent_number:04d}_"
                f"{record.bound:.6f}.sol"
            )
            self.incumbent_number += 1
            write_binary_solution(
                numbered_path,
                record.layout,
                record.bound,
                source,
                result,
                self.oracle,
            )
            best_path = self.folder / "best_free_coordinate.sol"
            write_binary_solution(
                best_path,
                record.layout,
                record.bound,
                source,
                result,
                self.oracle,
                required=False,
            )
            solution_path = numbered_path.name
        return solution_path

    def evaluate(self, layouts, phase, generation):
        unique = []
        for layout in layouts:
            if (
                layout.key in self.seen_layouts
                or layout.key in {candidate.key for candidate in unique}
            ):
                continue
            result = self.oracle.diagnose(layout)
            if not result.feasible:
                continue
            unique.append(layout)
        if not unique:
            return []

        evaluated = self.evaluator.evaluate(unique)
        records = []
        for layout, bound, dual, runtime, status in evaluated:
            if status != GRB.OPTIMAL or not np.isfinite(bound):
                print(
                    f"Stage B skipped a candidate: status={status} "
                    f"runtime={runtime:.3f}s",
                    flush=True,
                )
                continue
            self.seen_layouts.add(layout.key)
            result = self.oracle.diagnose(layout)
            if not result.feasible:
                raise AssertionError(
                    "A layout changed after exact-oracle acceptance."
                )
            record = ExactRecord(
                layout=layout,
                bound=float(bound),
                runtime=float(runtime),
                equality_dual=np.asarray(dual, dtype=float),
                phase=phase,
                generation=generation,
            )
            if phase in {"discovery", "translation_bootstrap"}:
                self.discovery_seconds += record.runtime
            elif phase in {"improvement", "translation_audit"}:
                self.local_seconds += record.runtime
            is_new_best = (
                self.best is None
                or record.bound
                > self.best.bound + self.args.improvement_tolerance
            )
            previous_best = self.best
            if is_new_best:
                self.best = record
            self.exact_records[layout.key] = record
            self.cut_pool.add(layout, record.bound, record.equality_dual)
            relative_key = layout.relative_key
            self.relative_counts[relative_key] = (
                self.relative_counts.get(relative_key, 0) + 1
            )
            family_record = self.relative_records.get(relative_key)
            if (
                family_record is None
                or record.bound > family_record.bound
            ):
                self.relative_records[relative_key] = record
            geometry = self.oracle.medium_geometry(layout)
            signature_record = self.signature_records.get(
                geometry.signature
            )
            if (
                signature_record is None
                or record.bound > signature_record.bound
            ):
                self.signature_records[geometry.signature] = record
            distance_from_best = (
                0
                if previous_best is None
                else layout.distance(previous_best.layout)
            )
            relative_distance_from_best = (
                0
                if previous_best is None
                else layout.relative_distance(previous_best.layout)
            )
            solution_path = self._persist(
                record,
                result,
                latest=False,
                best=is_new_best,
                save_all=self.args.save_all_solutions,
            )
            self.writer.writerow(
                [
                    self.evaluation_number,
                    phase,
                    generation,
                    record.bound,
                    record.runtime,
                    int(is_new_best),
                    distance_from_best,
                    layout.substations,
                    layout.medium_poles,
                    self.oracle.color_counts(layout),
                    self.relative_counts[relative_key],
                    relative_distance_from_best,
                    geometry.edges,
                    geometry.axial,
                    geometry.shallow,
                    geometry.moderate,
                    geometry.deep,
                    geometry.degree_two,
                    geometry.leaves,
                    result.certificate.cell_components,
                    result.certificate.winding_rank,
                    result.certificate.lattice_index,
                    solution_path,
                ]
            )
            self.evaluation_number += 1
            records.append(record)
            if is_new_best:
                print(
                    f"NEW FREE-COORDINATE BEST {record.bound:.6f} "
                    f"phase={phase} generation={generation} "
                    f"distance={distance_from_best}",
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
                print(
                    "  medium_geometry="
                    f"axial={geometry.axial} shallow={geometry.shallow} "
                    f"moderate={geometry.moderate} deep={geometry.deep} "
                    f"degree2={geometry.degree_two} "
                    f"leaves={geometry.leaves}",
                    flush=True,
                )
        if records:
            latest_record = records[-1]
            self._persist(
                latest_record,
                self.oracle.diagnose(latest_record.layout),
                latest=True,
                best=False,
                save_all=False,
            )
        return records

    def _population(self):
        return sorted(
            self.exact_records.values(),
            key=lambda record: record.bound,
            reverse=True,
        )

    def _relative_population(self):
        return sorted(
            self.relative_records.values(),
            key=lambda record: record.bound,
            reverse=True,
        )

    def _signature_population(self):
        return sorted(
            self.signature_records.values(),
            key=lambda record: record.bound,
            reverse=True,
        )

    def _diverse_basin_population(self, relative_population=None):
        """High-quality parents outside the incumbent's local attractor."""
        if self.best is None:
            return []
        if relative_population is None:
            relative_population = self._relative_population()
        candidates = [
            record
            for record in relative_population
            if (
                record.layout.relative_key
                != self.best.layout.relative_key
                and record.layout.relative_distance(self.best.layout)
                >= self.args.basin_min_distance
            )
        ]
        diverse = []
        for record in candidates:
            if all(
                record.layout.relative_distance(other.layout)
                >= self.args.basin_separation
                for other in diverse
            ):
                diverse.append(record)
                if len(diverse) >= self.args.basin_parent_pool:
                    break
        return diverse

    def _choose_basin_parent(self, pool):
        """Rotate fairly through distinct basins instead of reconverging."""
        if not pool:
            return None
        minimum_uses = min(
            self.parent_uses.get(record.layout.relative_key, 0)
            for record in pool
        )
        least_used = [
            record
            for record in pool
            if (
                self.parent_uses.get(record.layout.relative_key, 0)
                == minimum_uses
            )
        ]
        # Keep the rotation quality-biased without repeatedly selecting only
        # the highest-bound member of the same attractor.
        return self.rng.choice(
            least_used[: min(4, len(least_used))]
        )

    def _parent_and_donor(
        self,
        discovery,
        *,
        population,
        relative_population,
        signature_population,
        basin_population,
        nearby_population,
    ):
        if not population:
            raise RuntimeError("No exact population is available.")
        if discovery:
            parent_mode = "discovery"
            roll = self.rng.random()
            if roll < self.args.discovery_quality_parent_share:
                pool = relative_population[
                    : min(
                        self.args.discovery_parent_pool,
                        len(relative_population),
                    )
                ]
            elif (
                roll
                < self.args.discovery_quality_parent_share
                + self.args.topology_parent_share
            ):
                pool = signature_population
            else:
                pool = relative_population
            parent_record = self.rng.choice(pool)
        else:
            assert self.best is not None
            roll = self.rng.random()
            if roll < self.args.incumbent_parent_share:
                parent_record = self.best
                parent_mode = "incumbent"
            elif roll < (
                self.args.incumbent_parent_share
                + self.args.nearby_parent_share
            ):
                if nearby_population:
                    parent_record = self.rng.choice(
                        nearby_population[
                            : min(
                                self.args.local_elites,
                                len(nearby_population),
                            )
                        ]
                    )
                    parent_mode = "nearby"
                else:
                    parent_record = self.best
                    parent_mode = "incumbent"
            else:
                parent_record = self._choose_basin_parent(
                    basin_population
                )
                if parent_record is None:
                    parent_record = self.best
                    parent_mode = "incumbent"
                else:
                    parent_mode = "basin"

            relative_key = parent_record.layout.relative_key
            self.parent_uses[relative_key] = (
                self.parent_uses.get(relative_key, 0) + 1
            )

        donor_record = None
        donor_probability = (
            1.0
            if discovery
            else self.args.improvement_donor_probability
        )
        if (
            len(relative_population) > 1
            and self.rng.random() < donor_probability
        ):
            parent = parent_record.layout
            donor_pool = [
                record
                for record in relative_population
                if record.layout.relative_key != parent.relative_key
            ]
            if not discovery:
                donor_pool = [
                    record
                    for record in donor_pool
                    if record.layout.relative_distance(parent)
                    <= self.args.nearby_donor_distance
                ]
            if donor_pool:
                sample = self.rng.sample(
                    donor_pool,
                    min(24, len(donor_pool)),
                )
                donor_record = max(
                    sample,
                    key=lambda record: (
                        record.layout.relative_distance(parent)
                        if discovery
                        else record.bound
                    ),
                )
        return parent_record, donor_record, parent_mode

    def generate_phase(self, count, discovery):
        phase = "discovery" if discovery else "improvement"
        if count <= 0:
            return []
        start = time.perf_counter()
        candidates = {}
        candidate_family_counts = {}
        attempts = 0
        population = self._population()
        relative_population = self._relative_population()
        signature_population = self._signature_population()
        basin_population = self._diverse_basin_population(
            relative_population
        )
        nearby_population = (
            []
            if self.best is None
            else [
                record
                for record in relative_population
                if (
                    record.layout.key != self.best.layout.key
                    and record.layout.relative_distance(self.best.layout)
                    <= self.args.nearby_parent_distance
                )
            ]
        )
        proposal_factor = (
            self.args.discovery_proposal_factor
            if discovery
            else self.args.improvement_proposal_factor
        )
        proposal_target = max(count, math.ceil(count * proposal_factor))
        maximum_attempts = max(
            proposal_target,
            proposal_target * self.args.attempt_factor,
        )

        def admit_candidate(
            candidate,
            parent_record,
            parent_mode,
            move_limit,
        ):
            if (
                candidate is None
                or candidate.key in self.seen_layouts
                or candidate.key in candidates
                or (
                    self.relative_counts.get(candidate.relative_key, 0)
                    + candidate_family_counts.get(
                        candidate.relative_key,
                        0,
                    )
                    >= self.args.relative_family_cap
                )
                or (
                    not discovery
                    and (
                        candidate.distance(parent_record.layout)
                        > move_limit
                        or self.best is None
                        or (
                            parent_mode != "basin"
                            and candidate.distance(self.best.layout)
                            > self.args.improvement_best_radius
                        )
                    )
                )
            ):
                return False
            candidates[candidate.key] = (candidate, parent_mode)
            candidate_family_counts[candidate.relative_key] = (
                candidate_family_counts.get(candidate.relative_key, 0) + 1
            )
            return True

        skew_proposed = 0
        if not discovery and self.best is not None:
            skew_parents = [(self.best, "incumbent")]
            used_parent_families = {self.best.layout.relative_key}
            for record in nearby_population[:3]:
                if record.layout.relative_key in used_parent_families:
                    continue
                skew_parents.append((record, "nearby"))
                used_parent_families.add(record.layout.relative_key)
            for record in basin_population:
                if len(skew_parents) >= self.args.balanced_skew_parents:
                    break
                if record.layout.relative_key in used_parent_families:
                    continue
                skew_parents.append((record, "basin"))
                used_parent_families.add(record.layout.relative_key)

            skew_options = []
            for parent_record, parent_mode in skew_parents:
                if time.perf_counter() - start >= self.args.proposal_seconds:
                    break
                options = self.generator.balanced_skew_candidates(
                    parent_record.layout
                )
                skew_options.append(
                    (
                        parent_record,
                        parent_mode,
                        options[: self.args.balanced_skew_per_parent],
                    )
                )
            for option_number in range(
                self.args.balanced_skew_per_parent
            ):
                for parent_record, parent_mode, options in skew_options:
                    if (
                        len(candidates) >= proposal_target
                        or time.perf_counter() - start
                        >= self.args.proposal_seconds
                    ):
                        break
                    if option_number >= len(options):
                        continue
                    candidate = options[option_number]
                    move_limit = (
                        self.args.basin_move_limit
                        if parent_mode == "basin"
                        else self.args.improvement_move_limit
                    )
                    if admit_candidate(
                        candidate,
                        parent_record,
                        parent_mode,
                        move_limit,
                    ):
                        skew_proposed += 1

        while (
            len(candidates) < proposal_target
            and attempts < maximum_attempts
            and time.perf_counter() - start < self.args.proposal_seconds
        ):
            attempts += 1
            (
                parent_record,
                donor_record,
                parent_mode,
            ) = self._parent_and_donor(
                discovery,
                population=population,
                relative_population=relative_population,
                signature_population=signature_population,
                basin_population=basin_population,
                nearby_population=nearby_population,
            )
            move_limit = (
                self.args.basin_move_limit
                if parent_mode == "basin"
                else self.args.improvement_move_limit
            )
            candidate = self.generator.generate(
                parent_record.layout,
                None if donor_record is None else donor_record.layout,
                discovery=discovery,
                guidance_dual=parent_record.equality_dual,
                basin=parent_mode == "basin",
                move_limit=move_limit,
            )
            admit_candidate(
                candidate,
                parent_record,
                parent_mode,
                move_limit,
            )
        scored = [
            (
                self.cut_pool.upper_bound(candidate),
                self.rng.random(),
                candidate,
                parent_mode,
            )
            for candidate, parent_mode in candidates.values()
        ]
        scored.sort(reverse=True)
        basin_selected = 0
        if discovery:
            novel = [
                item
                for item in scored
                if self.relative_counts.get(item[2].relative_key, 0) == 0
            ]
            reserved = min(
                len(novel),
                max(1, round(count * self.args.discovery_novel_share)),
            )
            selected = novel[:reserved]
            selected_keys = {item[2].key for item in selected}
            selected.extend(
                item
                for item in scored
                if item[2].key not in selected_keys
            )
            scored = selected
        else:
            basin = [
                item
                for item in scored
                if item[3] == "basin"
            ]
            basin_share = max(
                0.0,
                1.0
                - self.args.incumbent_parent_share
                - self.args.nearby_parent_share,
            )
            reserved = min(
                len(basin),
                round(count * basin_share),
            )
            selected = basin[:reserved]
            selected_keys = {item[2].key for item in selected}
            selected.extend(
                item
                for item in scored
                if item[2].key not in selected_keys
            )
            scored = selected
            basin_selected = min(
                reserved,
                len(scored[:count]),
            )
        selected_layouts = [item[2] for item in scored[:count]]
        elapsed = time.perf_counter() - start
        proposal_capped = (
            len(candidates) < proposal_target
            and elapsed >= self.args.proposal_seconds
        )
        if discovery:
            self.discovery_seconds += elapsed
            self.discovery_units += len(selected_layouts)
        else:
            self.local_seconds += elapsed
            self.local_units += len(selected_layouts)
        best_upper = scored[0][0] if scored else math.nan
        print(
            f"{phase}: proposed {len(candidates)}/{proposal_target}, "
            f"selected {len(selected_layouts)}/{count} exact-feasible "
            f"layouts in {elapsed:.2f}s from {attempts} attempts "
            f"best_dual_ub={best_upper:.3f}"
            + (
                f" balanced_skew={skew_proposed}"
                if not discovery
                else ""
            )
            + (
                f" basin_selected={basin_selected}"
                if not discovery
                else ""
            )
            + (" proposal_cap=hit" if proposal_capped else ""),
            flush=True,
        )
        return selected_layouts

    @staticmethod
    def _parent_plane_bound(
        record: ExactRecord,
        layout: FreeCoordinateLayout,
    ) -> float:
        dual = record.equality_dual
        return float(
            record.bound
            + np.sum(dual[record.layout.selected_network_indices])
            - np.sum(dual[layout.selected_network_indices])
        )

    def _ranked_translations(
        self,
        record: ExactRecord,
        limit: int,
    ) -> list[FreeCoordinateLayout]:
        candidates = {
            candidate.key: candidate
            for candidate in self.generator.translation_candidates(
                record.layout
            )
            if candidate.key not in self.seen_layouts
        }
        ranked = sorted(
            candidates.values(),
            key=lambda layout: (
                self.cut_pool.upper_bound(layout),
                self._parent_plane_bound(record, layout),
                self.rng.random(),
            ),
            reverse=True,
        )
        return ranked[:limit]

    def _audit_promising_families(self, records, generation):
        if self.args.translation_audit <= 0 or self.best is None:
            return []
        family_best = {}
        threshold = self.best.bound - self.args.translation_trigger_gap
        for record in records:
            key = record.layout.relative_key
            previous = family_best.get(key)
            if (
                record.bound >= threshold
                and key not in self.audited_relative_keys
                and (
                    previous is None
                    or record.bound > previous.bound
                )
            ):
                family_best[key] = record
        if not family_best:
            return []

        audited = []
        for record in sorted(
            family_best.values(),
            key=lambda item: item.bound,
            reverse=True,
        )[: self.args.translation_audit_families]:
            key = record.layout.relative_key
            self.audited_relative_keys.add(key)
            start = time.perf_counter()
            layouts = self._ranked_translations(
                record,
                self.args.translation_audit,
            )
            self.local_seconds += time.perf_counter() - start
            self.local_units += len(layouts)
            if not layouts:
                continue
            print(
                f"translation audit: {len(layouts)} dual-ranked placements "
                f"for new family at {record.bound:.6f}",
                flush=True,
            )
            audited.extend(
                self.evaluate(layouts, "translation_audit", generation)
            )
        return audited

    def _phase_counts(self):
        batch_size = self.args.batch_size
        target = self.args.discovery_share
        total = self.discovery_seconds + self.local_seconds
        discovery_average = (
            self.discovery_seconds / max(1, self.discovery_units)
            if self.discovery_seconds > 0
            else 1.0
        )
        improvement_average = (
            self.local_seconds / max(1, self.local_units)
            if self.local_seconds > 0
            else discovery_average
        )
        if total == 0:
            discovery_count = round(batch_size * target)
        else:
            denominator = (
                (1.0 - target) * discovery_average
                + target * improvement_average
            )
            numerator = (
                target * total
                - self.discovery_seconds
                + target * batch_size * improvement_average
            )
            discovery_count = round(
                numerator / max(0.01, denominator)
            )
            # Keep a small live discovery channel even after a bootstrap has
            # temporarily pushed its cumulative share above target.
            discovery_count = max(1, discovery_count)
        discovery_count = min(batch_size - 1, discovery_count)
        return discovery_count, batch_size - discovery_count

    def _write_state(self, outcome, generation):
        total = self.local_seconds + self.discovery_seconds
        discovery_share = (
            self.discovery_seconds / total if total > 0 else 0.0
        )
        path = self.folder / "free_coordinate_state.txt"
        temporary = _unique_temporary_path(path)
        with temporary.open("w") as handle:
            handle.write(f"outcome={outcome}\n")
            handle.write(f"random_seed={self.random_seed}\n")
            handle.write(f"generation={generation}\n")
            handle.write(f"evaluations={self.evaluation_number}\n")
            handle.write(
                f"relative_families={len(self.relative_counts)}\n"
            )
            handle.write(f"improvement_seconds={self.local_seconds:.6f}\n")
            handle.write(f"discovery_seconds={self.discovery_seconds:.6f}\n")
            handle.write(
                f"measured_discovery_share={discovery_share:.9f}\n"
            )
            handle.write(
                "best_bound="
                + (
                    "nan"
                    if self.best is None
                    else f"{self.best.bound:.16g}"
                )
                + "\n"
            )
            handle.write(
                "best_path="
                + str(self.folder / "best_free_coordinate.sol")
                + "\n"
            )
        _replace_with_retry(
            temporary,
            path,
            required=False,
        )

    def run(self):
        seed_layouts = [record[0] for record in self.seed_records]
        print(
            f"exactly evaluating {len(seed_layouts)} free-coordinate seeds "
            f"(random_seed={self.random_seed})",
            flush=True,
        )
        self.evaluate(seed_layouts, "seed", 0)
        if self.best is None:
            raise RuntimeError("No seed has a feasible unchanged Stage-B LP.")
        self._load_seen_histories()
        self._write_state("running", 0)

        target_already_reached = (
            np.isfinite(self.args.target)
            and self.best.bound
            >= self.args.target - self.args.improvement_tolerance
        )
        if (
            self.args.translation_bootstrap > 0
            and not target_already_reached
        ):
            bootstrap_start = time.perf_counter()
            bootstrap_parent = self.best
            self.audited_relative_keys.add(
                bootstrap_parent.layout.relative_key
            )
            translated_layouts = self._ranked_translations(
                bootstrap_parent,
                self.args.translation_bootstrap,
            )
            self.discovery_seconds += time.perf_counter() - bootstrap_start
            self.discovery_units += len(translated_layouts)
            print(
                f"bootstrap: exactly evaluating {len(translated_layouts)} "
                "dual-ranked whole-layout translations",
                flush=True,
            )
            self.evaluate(translated_layouts, "translation_bootstrap", 0)
            self._write_state("running", 0)

        generation = 0
        try:
            while (
                self.args.generations == 0
                or generation < self.args.generations
            ):
                if (
                    np.isfinite(self.args.target)
                    and self.best.bound
                    >= self.args.target - self.args.improvement_tolerance
                ):
                    break
                generation += 1
                discovery_count, improvement_count = self._phase_counts()

                # Generate both channels before solving Stage B.  Discovery
                # makes basin jumps; improvement either stays near the
                # incumbent or performs a reserved 3-6-pole climb inside a
                # distinct basin.
                discovery_layouts = self.generate_phase(
                    discovery_count,
                    True,
                )
                improvement_layouts = self.generate_phase(
                    improvement_count,
                    False,
                )
                discovery_records = self.evaluate(
                    discovery_layouts,
                    "discovery",
                    generation,
                )
                improvement_records = self.evaluate(
                    improvement_layouts,
                    "improvement",
                    generation,
                )
                translation_records = self._audit_promising_families(
                    [*discovery_records, *improvement_records],
                    generation,
                )
                total_search = self.discovery_seconds + self.local_seconds
                measured_share = (
                    self.discovery_seconds / total_search
                    if total_search
                    else 0.0
                )
                print(
                    f"generation {generation}: exact discovery="
                    f"{len(discovery_records)} improvement="
                    f"{len(improvement_records)} translations="
                    f"{len(translation_records)} total="
                    f"{self.evaluation_number} best={self.best.bound:.6f} "
                    f"measured discovery={100 * measured_share:.1f}%",
                    flush=True,
                )
                self._write_state("running", generation)
        except KeyboardInterrupt:
            print("Ctrl-C received; saved the current best and state.", flush=True)
            self._write_state("stopped", generation)
            return

        outcome = (
            "target"
            if (
                np.isfinite(self.args.target)
                and self.best.bound
                >= self.args.target - self.args.improvement_tolerance
            )
            else "complete"
        )
        self._write_state(outcome, generation)
        print(
            f"free-coordinate search {outcome}: generations={generation} "
            f"evaluations={self.evaluation_number} "
            f"best={self.best.bound:.6f} output={self.folder}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Free all five substations and ten medium poles, enforce exact "
            "periodic tileability in coordinates, and score with Stage B."
        )
    )
    parser.add_argument(
        "--seed-sol",
        type=Path,
        action="append",
        default=[],
        help="Explicit old Stage-A or Stage-B solution seed; repeatable.",
    )
    parser.add_argument(
        "--ignore-seen-history",
        action="store_true",
        help=(
            "Do not reuse evaluated coordinate keys from a seed run's "
            "free_coordinate_progress.csv."
        ),
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        action="append",
        default=[],
        help="Recursively reuse coordinate progress below this root; repeatable.",
    )
    parser.add_argument("--seed-count", type=int, default=12)
    parser.add_argument("--seed-scan-limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--lp-seconds",
        type=float,
        default=0.0,
        help="Per-LP limit; zero means no time limit (default).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=0,
        help="Zero runs until Ctrl-C or the target (default).",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=math.inf,
        help="Optional stopping bound; omitted runs until Ctrl-C.",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--discovery-share", type=float, default=0.35)
    parser.add_argument("--discovery-min-changes", type=int, default=4)
    parser.add_argument("--discovery-max-changes", type=int, default=9)
    parser.add_argument("--local-destroy", type=int, default=3)
    parser.add_argument("--local-elites", type=int, default=32)
    parser.add_argument("--incumbent-parent-share", type=float, default=0.25)
    parser.add_argument("--nearby-parent-share", type=float, default=0.40)
    parser.add_argument("--nearby-parent-distance", type=int, default=6)
    parser.add_argument("--improvement-move-limit", type=int, default=4)
    parser.add_argument("--improvement-best-radius", type=int, default=6)
    parser.add_argument("--basin-parent-pool", type=int, default=24)
    parser.add_argument("--basin-min-distance", type=int, default=4)
    parser.add_argument("--basin-separation", type=int, default=3)
    parser.add_argument("--basin-move-limit", type=int, default=6)
    parser.add_argument("--basin-snake-probability", type=float, default=0.75)
    parser.add_argument("--balanced-skew-parents", type=int, default=8)
    parser.add_argument("--balanced-skew-per-parent", type=int, default=2)
    parser.add_argument(
        "--improvement-donor-probability",
        type=float,
        default=0.10,
    )
    parser.add_argument("--nearby-donor-distance", type=int, default=5)
    parser.add_argument("--discovery-parent-pool", type=int, default=64)
    parser.add_argument(
        "--discovery-quality-parent-share",
        type=float,
        default=0.55,
    )
    parser.add_argument("--topology-parent-share", type=float, default=0.25)
    parser.add_argument("--discovery-novel-share", type=float, default=0.70)
    parser.add_argument("--relative-family-cap", type=int, default=1)
    parser.add_argument(
        "--improvement-proposal-factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--discovery-proposal-factor",
        type=float,
        default=1.25,
    )
    parser.add_argument(
        "--proposal-seconds",
        type=float,
        default=45.0,
        help=(
            "Maximum construction time per phase; returns fewer proposals "
            "instead of limiting the persistent search or any Stage-B LP."
        ),
    )
    parser.add_argument("--dual-cut-pool", type=int, default=384)
    parser.add_argument("--dual-safety", type=float, default=0.02)
    parser.add_argument("--dual-guidance-weight", type=float, default=35.0)
    parser.add_argument("--coordinate-pool", type=int, default=700)
    parser.add_argument("--seam-pool", type=int, default=240)
    parser.add_argument("--construction-attempts", type=int, default=10)
    parser.add_argument("--construction-elite", type=int, default=18)
    parser.add_argument("--construction-noise", type=float, default=35.0)
    parser.add_argument("--repair-steps", type=int, default=28)
    parser.add_argument("--exact-repair-pool", type=int, default=18)
    parser.add_argument("--repair-temperature", type=float, default=250.0)
    parser.add_argument("--local-radius", type=int, default=5)
    parser.add_argument("--scaffold-probability", type=float, default=0.65)
    parser.add_argument("--translation-probability", type=float, default=0.0)
    parser.add_argument("--translation-radius", type=int, default=8)
    parser.add_argument("--translation-bootstrap", type=int, default=40)
    parser.add_argument("--translation-audit", type=int, default=12)
    parser.add_argument("--translation-audit-families", type=int, default=1)
    parser.add_argument(
        "--translation-trigger-gap",
        type=float,
        default=1.0,
    )
    parser.add_argument("--walk-probability", type=float, default=0.75)
    parser.add_argument("--walk-step-factor", type=int, default=3)
    parser.add_argument(
        "--improvement-snake-probability",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--discovery-snake-probability",
        type=float,
        default=0.70,
    )
    parser.add_argument("--snake-attempts", type=int, default=3)
    parser.add_argument("--snake-min-poles", type=int, default=3)
    parser.add_argument("--snake-max-poles", type=int, default=6)
    parser.add_argument("--shallow-edge-bonus", type=float, default=14.0)
    parser.add_argument("--axial-edge-penalty", type=float, default=6.0)
    parser.add_argument("--target-axial-edges", type=int, default=4)
    parser.add_argument("--target-shallow-edges", type=int, default=4)
    parser.add_argument("--shallow-excess-penalty", type=float, default=10.0)
    parser.add_argument("--moderate-edge-penalty", type=float, default=6.0)
    parser.add_argument("--deep-edge-penalty", type=float, default=16.0)
    parser.add_argument("--degree-two-bonus", type=float, default=8.0)
    parser.add_argument("--leaf-penalty", type=float, default=6.0)
    parser.add_argument("--attempt-factor", type=int, default=4)
    parser.add_argument("--improvement-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed; omitted chooses and records a fresh seed.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-all-solutions", action="store_true")
    parser.set_defaults(
        physical_wire_offsets=True,
        true_periodic_coverage=True,
    )
    parser.add_argument(
        "--physical-wire-offsets",
        dest="physical_wire_offsets",
        action="store_true",
        help="Use physical building centers for wire range (default).",
    )
    parser.add_argument(
        "--legacy-binary-wire-offsets",
        dest="physical_wire_offsets",
        action="store_false",
        help="Audit only: reproduce the historical reversed mixed offset.",
    )
    parser.add_argument(
        "--periodic-only-connectivity",
        action="store_true",
        help=(
            "Allow a cell to connect only through neighboring copies. "
            "Default also requires the zero-shift cell graph connected."
        ),
    )
    parser.add_argument(
        "--true-periodic-coverage",
        dest="true_periodic_coverage",
        action="store_true",
        help="Let electric supply areas continue through cell borders (default).",
    )
    parser.add_argument(
        "--legacy-clipped-coverage",
        dest="true_periodic_coverage",
        action="store_false",
        help="Audit only: clip electric supply at the 50x50 cell border.",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_args(args):
    positive = (
        "seed_count",
        "workers",
        "batch_size",
        "discovery_min_changes",
        "discovery_max_changes",
        "local_destroy",
        "local_elites",
        "nearby_parent_distance",
        "improvement_move_limit",
        "improvement_best_radius",
        "basin_parent_pool",
        "basin_min_distance",
        "basin_separation",
        "basin_move_limit",
        "balanced_skew_parents",
        "balanced_skew_per_parent",
        "nearby_donor_distance",
        "discovery_parent_pool",
        "relative_family_cap",
        "dual_cut_pool",
        "coordinate_pool",
        "seam_pool",
        "construction_attempts",
        "construction_elite",
        "repair_steps",
        "exact_repair_pool",
        "local_radius",
        "translation_radius",
        "translation_audit_families",
        "walk_step_factor",
        "snake_attempts",
        "snake_min_poles",
        "snake_max_poles",
        "attempt_factor",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.seed_scan_limit < 0:
        raise ValueError("--seed-scan-limit cannot be negative.")
    if args.generations < 0:
        raise ValueError("--generations cannot be negative.")
    if args.lp_seconds < 0:
        raise ValueError("--lp-seconds cannot be negative.")
    if (
        not math.isfinite(args.proposal_seconds)
        or args.proposal_seconds <= 0
    ):
        raise ValueError("--proposal-seconds must be finite and positive.")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least two.")
    if (
        not math.isfinite(args.improvement_tolerance)
        or args.improvement_tolerance < 0
    ):
        raise ValueError(
            "--improvement-tolerance must be finite and nonnegative."
        )
    if not 0 < args.discovery_share < 1:
        raise ValueError("--discovery-share must lie strictly between 0 and 1.")
    if not 0 <= args.scaffold_probability <= 1:
        raise ValueError("--scaffold-probability must lie in [0,1].")
    if not 0 <= args.translation_probability <= 1:
        raise ValueError("--translation-probability must lie in [0,1].")
    if not 0 <= args.walk_probability <= 1:
        raise ValueError("--walk-probability must lie in [0,1].")
    probabilities = (
        "incumbent_parent_share",
        "nearby_parent_share",
        "improvement_donor_probability",
        "basin_snake_probability",
        "discovery_quality_parent_share",
        "topology_parent_share",
        "discovery_novel_share",
        "improvement_snake_probability",
        "discovery_snake_probability",
    )
    for name in probabilities:
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(
                f"--{name.replace('_', '-')} must lie in [0,1]."
            )
    if args.incumbent_parent_share + args.nearby_parent_share > 1:
        raise ValueError(
            "Incumbent and nearby parent shares cannot exceed one."
        )
    if (
        args.discovery_quality_parent_share
        + args.topology_parent_share
        > 1
    ):
        raise ValueError(
            "Discovery quality and topology parent shares cannot exceed one."
        )
    if args.translation_bootstrap < 0:
        raise ValueError("--translation-bootstrap cannot be negative.")
    if args.translation_audit < 0:
        raise ValueError("--translation-audit cannot be negative.")
    nonnegative = (
        "translation_trigger_gap",
        "dual_safety",
        "dual_guidance_weight",
        "shallow_edge_bonus",
        "axial_edge_penalty",
        "target_axial_edges",
        "target_shallow_edges",
        "shallow_excess_penalty",
        "moderate_edge_penalty",
        "deep_edge_penalty",
        "degree_two_bonus",
        "leaf_penalty",
    )
    for name in nonnegative:
        if getattr(args, name) < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} cannot be negative."
            )
    if (
        args.improvement_proposal_factor < 1
        or args.discovery_proposal_factor < 1
    ):
        raise ValueError("Proposal factors must be at least one.")
    if (
        args.snake_min_poles > args.snake_max_poles
        or args.snake_max_poles > EXACT_MEDIUMS
    ):
        raise ValueError("Invalid shallow-snake pole range.")
    if (
        args.basin_min_distance > EXACT_SUBSTATIONS + EXACT_MEDIUMS
        or args.basin_separation > EXACT_SUBSTATIONS + EXACT_MEDIUMS
        or args.basin_move_limit > EXACT_SUBSTATIONS + EXACT_MEDIUMS
    ):
        raise ValueError("Invalid basin distance or move limit.")
    if (
        args.discovery_min_changes > args.discovery_max_changes
        or args.discovery_max_changes
        > EXACT_SUBSTATIONS + EXACT_MEDIUMS
    ):
        raise ValueError("Invalid discovery change range.")


def print_seed(number, layout, path, result, oracle):
    print(
        f"seed {number}: path={path} colors={oracle.color_counts(layout)} "
        f"edges={result.certificate.edge_count} "
        f"windings={result.certificate.windings}",
        flush=True,
    )
    print(f"  substations={layout.substations}", flush=True)
    print(f"  medium_poles={layout.medium_poles}", flush=True)


def main():
    args = parse_args()
    validate_args(args)
    build_start = time.perf_counter()
    oracle = FreePeriodicOracle(
        physical_wire_offsets=args.physical_wire_offsets,
        periodic_only_connectivity=args.periodic_only_connectivity,
        true_periodic_coverage=args.true_periodic_coverage,
    )
    seed_records = discover_seed_layouts(
        oracle,
        args.seed_sol,
        args.seed_count,
        args.seed_scan_limit,
    )
    mode = (
        "physical"
        if args.physical_wire_offsets
        else "binary-compatible"
    )
    coverage_mode = (
        "periodic" if args.true_periodic_coverage else "legacy-clipped"
    )
    print(
        f"free-coordinate 5+10 oracle ready in "
        f"{time.perf_counter() - build_start:.2f}s: "
        f"30 integer coordinates, wire_mode={mode}, "
        f"coverage_mode={coverage_mode}, "
        f"valid_seeds={len(seed_records)}",
        flush=True,
    )
    if not seed_records:
        raise ValueError(
            "No valid 5+10 seed was found. Supply one with --seed-sol."
        )
    for number, (layout, path, result) in enumerate(seed_records):
        print_seed(number, layout, path, result, oracle)
        values = layout_to_binary_solution(layout)
        validate_binary_roundtrip(values, oracle)

    if args.validate_only:
        print(
            "validation complete: every seed roundtrips to the dense "
            "5,006-variable binary formulation",
            flush=True,
        )
        return

    search = FreeCoordinateSearch(args, oracle, seed_records)
    try:
        search.run()
    finally:
        search.close()


if __name__ == "__main__":
    main()
