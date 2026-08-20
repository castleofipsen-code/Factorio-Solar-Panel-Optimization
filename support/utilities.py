from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import parameters as parameters


_SOLUTION_LINE = re.compile(
    r"^\s*x\[(\d+)\]\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


@dataclass(frozen=True)
class NetworkLayout:
    """Substation and medium-pole coordinates read from a saved solution."""

    substations: tuple[tuple[int, int], ...]
    medium_poles: tuple[tuple[int, int], ...]


def read_solution_layout(path: str | Path) -> NetworkLayout:
    """Read network coordinates from a Stage-A or Stage-B ``.sol`` file.

    Stage-A free-coordinate files store substations and medium poles in the
    first two grid-sized blocks. Stage-B files use the third and fifth blocks
    because solar panels and accumulators are included as well.
    """
    path = Path(path)
    values: dict[int, float] = {}
    with path.open(errors="replace") as handle:
        for line in handle:
            match = _SOLUTION_LINE.match(line)
            if match:
                values[int(match.group(1))] = float(match.group(2))

    if not values:
        raise ValueError(f"{path} contains no indexed x values.")

    grid = parameters.GRID_SIZE
    grid_area = grid * grid
    network_size = 2 * grid_area
    if max(values) >= network_size + 6:
        substation_indices = (
            index - 2 * grid_area
            for index, value in values.items()
            if 2 * grid_area <= index < 3 * grid_area and value > 0.5
        )
        medium_indices = (
            index - 4 * grid_area
            for index, value in values.items()
            if 4 * grid_area <= index < 5 * grid_area and value > 0.5
        )
    else:
        substation_indices = (
            index
            for index, value in values.items()
            if 0 <= index < grid_area and value > 0.5
        )
        medium_indices = (
            index - grid_area
            for index, value in values.items()
            if grid_area <= index < network_size and value > 0.5
        )

    to_coordinate = lambda index: divmod(index, grid)
    return NetworkLayout(
        substations=tuple(sorted(map(to_coordinate, substation_indices))),
        medium_poles=tuple(sorted(map(to_coordinate, medium_indices))),
    )

def matrix_to_coordinates(arrays):

    if isinstance(arrays, np.ndarray):
        coords = np.argwhere(arrays >= .5)
        return coords

    coords_list = []
    for arr in arrays:
        coords = np.argwhere(arr >= .5)
        coords_list.append(coords)

    return coords_list


def vector_to_matrices(x):

    d = parameters.GRID_SIZE
    dd = d**2

    solar_matrix = x[:dd].reshape(d, d)
    accumulator_matrix = x[dd:2*dd].reshape(d, d)
    substation_matrix = x[2*dd:3*dd].reshape(d, d)
    roboport_matrix = x[3*dd:4*dd].reshape(d, d)

    if x.shape[0] > 4*dd:
        pole_matrix = x[4*dd:5*dd].reshape(d, d)
        return [solar_matrix, accumulator_matrix, substation_matrix, roboport_matrix, pole_matrix]

    return [solar_matrix, accumulator_matrix, substation_matrix, roboport_matrix] 


def state_vector_to_coordinates(x):

    matrices = vector_to_matrices(x)

    return matrix_to_coordinates(vector_to_matrices(x))

def vector_to_coordinates(x):

    k_values = np.where(x >= 0.5)[0]

    i_values = k_values // parameters.GRID_SIZE
    j_values = k_values % parameters.GRID_SIZE

    return np.array([i_values,j_values]).T

def block_indices(k, n_left, n_right, size=parameters.GRID_SIZE, consider_wrapping = True):
    # k -> (i, j)
    i = k // size
    j = k % size

    indices = []

    for di in range(-n_left, n_right + 1):
        for dj in range(-n_left, n_right + 1):
            if consider_wrapping:
                ii = (i + di) % size
                jj = (j + dj) % size
                kk = ii * size + jj
                indices.append(kk)
            else:
                ii = (i + di) 
                jj = (j + dj) 

                if 0 <= ii < size and 0 <= jj < size:
                    kk = ii * size + jj
                    indices.append(kk)

    return indices

def circle_indices(k, radius, size=parameters.GRID_SIZE, consider_wrapping=True):

    i = k // size
    j = k % size

    indices = []

    for di in range(-radius-1, radius + 2):
        for dj in range(-radius-1, radius + 2):
            if di**2 + dj**2 <= radius**2:
                if consider_wrapping:
                    ii = (i + di) % size
                    jj = (j + dj) % size
                    kk = ii * size + jj
                    indices.append(kk)
                else:
                    ii = i + di
                    jj = j + dj

                    if 0 <= ii < size and 0 <= jj < size:
                        kk = ii * size + jj
                        indices.append(kk)

    return indices

def circle_indices_outside(k, radius, size=parameters.GRID_SIZE, axis=0):

    i = k // size
    j = k % size

    indices = []

    for di in range(-radius-1, radius + 2):
        for dj in range(-radius-1, radius + 2):
            if di**2 + dj**2 <= radius**2:

                ii = i + di
                jj = j + dj

                if axis == 0:
                    if (ii < 0 or ii >= size) and (0 <= jj < size):
                        kk = (ii % size) * size + jj
                        indices.append(kk)

                elif axis == 1:
                    if (0 <= ii < size) and (jj < 0 or jj >= size):
                        kk = ii * size + (jj % size)
                        indices.append(kk)

    return indices


def wrapped_block_counts(k_values, n, m, grid=parameters.GRID_SIZE):

    i = k_values // grid
    j = k_values % grid

    all_indices = []

    for di in range(-n, m + 1):
        for dj in range(-n, m + 1):
            ii = (i + di) % grid
            jj = (j + dj) % grid
            kk = ii * grid + jj
            all_indices.append(kk)

    all_indices = np.concatenate(all_indices)

    return np.bincount(all_indices, minlength=grid**2)

def electric_coverage(k_values, k_electric, n, m, grid=parameters.GRID_SIZE):

    k_electric = set(k_electric) 

    i = k_values // grid
    j = k_values % grid

    covered = np.zeros(grid**2, dtype=int)

    for idx, (ii0, jj0) in enumerate(zip(i, j)):

        is_covered = 0

        for di in range(-n, m + 1):
            for dj in range(-n, m + 1):
                ii = (ii0 + di) % grid
                jj = (jj0 + dj) % grid
                kk = ii * grid + jj

                if kk in k_electric:
                    is_covered = 1
                    break
            if is_covered:
                break

        covered[k_values[idx]] = is_covered

    return covered



def prune_electric_network(k_values, distance=18, grid=parameters.GRID_SIZE):

    if len(k_values) == 0:
        return k_values

    cluster = [k_values[0]]
    unconnected = list(k_values[1:])

    while True:
        newly_connected = []
        still_unconnected = []

        for k in unconnected:
            i = k // grid
            j = k % grid

            connects = False

            for kc in cluster:
                ic = kc // grid
                jc = kc % grid

                if wrapped_distance(i, j, ic, jc, grid) <= distance:
                    connects = True
                    break

            if connects:
                newly_connected.append(k)
            else:
                still_unconnected.append(k)

        if len(newly_connected) == 0:
            break

        cluster.extend(newly_connected)
        unconnected = still_unconnected

    return np.array(cluster)

def wrapped_distance(i1, j1, i2, j2, grid=parameters.GRID_SIZE):
    di = abs(i1 - i2)
    dj = abs(j1 - j2)

    di = min(di, grid - di)
    dj = min(dj, grid - dj)

    return np.sqrt(di**2 + dj**2)

def index_grid(i, j=None, grid=parameters.GRID_SIZE):

    if j is None:

        return (i // grid, i % grid)
    
    return i*grid + j
    
def substation_rectangles_for_object(i, j, l, object_size):
    """
    Returns valid substation rectangles for an object anchored at (i, j).

    object_size:
        3 for solar panel
        2 for accumulator

    Each returned tuple is:
        (row_start, row_end, col_start, col_end)
    inclusive coordinates.
    """

    back = 9
    front = object_size + 7
    # object_size=3 -> front=10
    # object_size=2 -> front=9

    r0 = (i - back) % l
    r1 = (i + front) % l

    c0 = (j - back) % l
    c1 = (j + front) % l

    row_parts = (
        [(r0, r1)]
        if r0 <= r1
        else [(r0, l - 1), (0, r1)]
    )

    col_parts = (
        [(c0, c1)]
        if c0 <= c1
        else [(c0, l - 1), (0, c1)]
    )

    return [
        (row_start, row_end, col_start, col_end)
        for row_start, row_end in row_parts
        for col_start, col_end in col_parts
    ]

def add_prefix_rect_terms(E, col, r1, r2, c1, c2, coef, dd):
    """
    Adds coef * rectangle_sum(r1:r2, c1:c2) to constraint column col.

    rectangle_sum =
        P[r2,c2]
      - P[r1-1,c2]
      - P[r2,c1-1]
      + P[r1-1,c1-1]
    """

    # + P[r2, c2]
    E[5*dd + index_grid(r2, c2), col] += coef

    # - P[r1-1, c2]
    if r1 > 0:
        E[5*dd + index_grid(r1 - 1, c2), col] -= coef

    # - P[r2, c1-1]
    if c1 > 0:
        E[5*dd + index_grid(r2, c1 - 1), col] -= coef

    # + P[r1-1, c1-1]
    if r1 > 0 and c1 > 0:
        E[5*dd + index_grid(r1 - 1, c1 - 1), col] += coef

def validate_warm_start(warm_start, A_sparse, b_lb, b_ub, lb, ub, integrality, tol=1e-7):

    finite_ub = np.isfinite(ub)

    # Bounds check
    bounds_ok = np.all(warm_start >= lb - tol) and np.all(warm_start[finite_ub] <= ub[finite_ub] + tol)

    # Integrality check
    integer_ok = np.allclose(
        warm_start[integrality == 1],
        np.round(warm_start[integrality == 1]),
        atol=tol
    )

    # Constraint check
    lhs = A_sparse @ warm_start

    constraints_ok = np.all(lhs >= b_lb - tol) and np.all(lhs <= b_ub + tol)

    print("bounds_ok:", bounds_ok)
    print("integer_ok:", integer_ok)
    print("constraints_ok:", constraints_ok)

    if not constraints_ok:

        bad_low = np.where(lhs < b_lb - tol)[0]
        bad_high = np.where(lhs > b_ub + tol)[0]

        print("violated lower constraints:", len(bad_low))
        print("violated upper constraints:", len(bad_high))

        if len(bad_low) > 0:
            r = bad_low[0]

            print("first lower violation:")
            print("row:", r)
            print("lhs:", lhs[r])
            print("lb:", b_lb[r])

        if len(bad_high) > 0:
            r = bad_high[0]

            print("first upper violation:")
            print("row:", r)
            print("lhs:", lhs[r])
            print("ub:", b_ub[r])

    return

def load_solution(path, size=None, missing_value=0.0):
    """Load indexed ``x[...]`` values from a solver solution file.

    Dense and sparse ``.sol`` files are both accepted. ``size`` is useful for
    partial warm starts; entries omitted from the file receive missing_value.
    """
    indexed_values = {}
    with Path(path).open(errors="replace") as handle:
        for line in handle:
            match = _SOLUTION_LINE.match(line)
            if match:
                indexed_values[int(match.group(1))] = float(match.group(2))

    if not indexed_values:
        raise ValueError(f"{path} contains no indexed x values.")

    inferred_size = max(indexed_values) + 1
    if size is None:
        size = inferred_size
    if size < inferred_size:
        raise ValueError(
            f"{path} contains x[{inferred_size - 1}], which does not fit "
            f"the requested vector size {size}."
        )

    solution = np.full(size, missing_value, dtype=float)
    for index, value in indexed_values.items():
        solution[index] = value
    return solution


def add_root_variables(x_old, grid):
    """
    Convert old 5*dd+1 solution vector into new 6*dd+1 vector
    with root variables added.

    Input:
        x_old : length 5*dd + 1
            [solar | accum | substations | roboports | MPs | z]

    Output:
        x_new : length 6*dd + 1
            [solar | accum | substations | roboports | MPs | roots | z]

    Root choice:
        lowest-index selected substation
    """

    dd = grid**2

    x_old = np.asarray(x_old).copy()

    x_new = np.zeros(6*dd + 1)

    # copy old variables except z
    x_new[:5*dd] = x_old[:5*dd]

    # copy z
    x_new[-1] = x_old[-1]

    # choose root = lowest-index selected substation
    sub_slice = x_old[2*dd:3*dd]
    sub_indices = np.flatnonzero(sub_slice > 0.5)

    if len(sub_indices) == 0:
        raise ValueError("No substations found.")

    root_idx = sub_indices[0]

    # set root variable
    x_new[5*dd + root_idx] = 1

    return x_new


def circle_indices_mixed_offsets(k, radius, size=parameters.GRID_SIZE, source_offset=(0.0, 0.0), target_offset=(0.0, 0.0), consider_wrapping=True):
    """
    Return target root indices whose offset-position is within radius
    of source root index k plus source_offset.

    Factorio offsets:
        substation:          (1.0, 1.0)
        medium-electric-pole: (0.5, 0.5)

    Uses <= radius.
    """

    i = k // size
    j = k % size

    sx = i + source_offset[0]
    sy = j + source_offset[1]

    indices = []

    max_offset_gap = max(
        abs(source_offset[0] - target_offset[0]),
        abs(source_offset[1] - target_offset[1]),
    )

    max_delta = int(np.ceil(radius + max_offset_gap)) + 1

    for di in range(-max_delta, max_delta + 1):
        for dj in range(-max_delta, max_delta + 1):

            ii_raw = i + di
            jj_raw = j + dj

            if consider_wrapping:
                ii = ii_raw % size
                jj = jj_raw % size

                tx = ii_raw + target_offset[0]
                ty = jj_raw + target_offset[1]

                dx = tx - sx
                dy = ty - sy

                if dx*dx + dy*dy <= radius*radius:
                    indices.append(ii * size + jj)

            else:
                if 0 <= ii_raw < size and 0 <= jj_raw < size:

                    tx = ii_raw + target_offset[0]
                    ty = jj_raw + target_offset[1]

                    dx = tx - sx
                    dy = ty - sy

                    if dx*dx + dy*dy <= radius*radius:
                        indices.append(ii_raw * size + jj_raw)

    return indices


def print_medium_diagnostics(sol, grid):
    dd = grid ** 2
    med = np.rint(sol[4*dd:5*dd]).astype(int)

    coords = []
    color_counts = np.zeros(4, dtype=int)
    mod3_counts = np.zeros((3, 3), dtype=int)
    mod6_counts = np.zeros((6, 6), dtype=int)

    for k in np.flatnonzero(med):
        i = k // grid
        j = k % grid
        color = 2 * (i % 2) + (j % 2)

        coords.append((i, j))
        color_counts[color] += 1
        mod3_counts[i % 3, j % 3] += 1
        mod6_counts[i % 6, j % 6] += 1

    print("medium coords:", coords)
    print("color counts:", tuple(color_counts))
    print("mod3 counts:")
    print(mod3_counts)
    print("mod6 nonzero:")
    for i in range(6):
        for j in range(6):
            if mod6_counts[i, j]:
                print((i, j), int(mod6_counts[i, j]))
