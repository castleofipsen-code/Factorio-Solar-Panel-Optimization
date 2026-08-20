import numpy as np
import scipy.sparse as sp
import parameters as parameters
from support import utilities as util

MEDIUM_COLOR_PATTERNS_2 = [
    (0, 1, 1, 0),
    (1, 0, 0, 1),
]

MEDIUM_COLOR_PATTERNS_5 = [
    (0, 1, 1, 3),
    (0, 1, 4, 0),
    (0, 4, 1, 0),
    (1, 0, 0, 4),
    (1, 0, 3, 1),
    (1, 3, 0, 1),
    (3, 1, 1, 0),
    (4, 0, 0, 1),
]

MEDIUM_COLOR_PATTERNS_6 = [
    (0, 0, 0, 6),
    (0, 0, 3, 3),
    (0, 0, 6, 0),
    (0, 3, 0, 3),
    (0, 3, 3, 0),
    (0, 6, 0, 0),
    (1, 2, 2, 1),
    (2, 1, 1, 2),
    (3, 0, 0, 3),
    (3, 0, 3, 0),
    (3, 3, 0, 0),
    (6, 0, 0, 0),
]

MEDIUM_COLOR_PATTERNS_9 = [
    (0, 0, 0, 9),
    (0, 0, 3, 6),
    (0, 0, 6, 3),
    (0, 0, 9, 0),
    (0, 3, 0, 6),
    (0, 3, 3, 3),
    (0, 3, 6, 0),
    (0, 6, 0, 3),
    (0, 6, 3, 0),
    (0, 9, 0, 0),
    (1, 2, 2, 4),
    (1, 2, 5, 1),
    (1, 5, 2, 1),
    (2, 1, 1, 5),
    (2, 1, 4, 2),
    (2, 4, 1, 2),
    (3, 0, 0, 6),
    (3, 0, 3, 3),
    (3, 0, 6, 0),
    (3, 3, 0, 3),
    (3, 3, 3, 0),
    (3, 6, 0, 0),
    (4, 2, 2, 1),
    (5, 1, 1, 2),
    (6, 0, 0, 3),
    (6, 0, 3, 0),
    (6, 3, 0, 0),
    (9, 0, 0, 0),
]

MEDIUM_COLOR_PATTERNS_10 = [
    (0, 2, 2, 6),
    (0, 2, 5, 3),
    (0, 2, 8, 0),
    (0, 5, 2, 3),
    (0, 5, 5, 0),
    (0, 8, 2, 0),
    (1, 1, 1, 7),
    (1, 1, 4, 4),
    (1, 1, 7, 1),
    (1, 4, 1, 4),
    (1, 4, 4, 1),
    (1, 7, 1, 1),
    (2, 0, 0, 8),
    (2, 0, 3, 5),
    (2, 0, 6, 2),
    (2, 3, 0, 5),
    (2, 3, 3, 2),
    (2, 6, 0, 2),
    (3, 2, 2, 3),
    (3, 2, 5, 0),
    (3, 5, 2, 0),
    (4, 1, 1, 4),
    (4, 1, 4, 1),
    (4, 4, 1, 1),
    (5, 0, 0, 5),
    (5, 0, 3, 2),
    (5, 3, 0, 2),
    (6, 2, 2, 0),
    (7, 1, 1, 1),
    (8, 0, 0, 2),
]

MEDIUM_COLOR_PATTERNS_14 = [
    (m0, m1, m2, 14 - m0 - m1 - m2)
    for m0 in range(15)
    for m1 in range(15 - m0)
    for m2 in range(15 - m0 - m1)
    if (m0 - (14 - m0 - m1 - m2)) % 3 == 0
    and (m1 - m2) % 3 == 0
]

def construct_network_floor_coverage_system_tileable_free_root_color_full(
    grid,
    max_medium_poles=np.inf,
    medium_weight=1.0,
    fixed_root_substation=None,
    use_corner_substations=False,
    use_side_substations=False,
    use_t_substations=False,
    balance_medium_colors=False,
    solar_total=199,
    medium_color_pattern=None,
    medium_color_completion_total=None,
    exact_medium_color_completion_total=None,
    use_medium_pole_obstacle_spacing=False,
    medium_pole_obstacle_radius=4,
    use_substation_bridge_tileability=False,
    substation_bridge_step=1,
    use_central_4x4_restriction=True,
):
    """
    Stage-A network planner.

    Variables:
        0:dd             substations
        dd:2*dd          medium electric poles
        2*dd:...          compact root-candidate block only when root is free

    Objective:
        minimize Ns + medium_weight * Nm

    Constraints:
        - either a fixed root substation, or exactly one freely selected root
        - a free root is the lowest-index selected substation
        - every possible 5x5 fictitious building placement must be covered
        - optional central 4x4 full electric coverage
        - central 4x4 physical clearance
        - lower-rank connectivity
        - number of medium poles <= max_medium_poles
        - optional fixed three-corner substations
        - optional solar-panel + medium-pole color-completion feasibility with exactly solar_total panels
        - optional exact medium-pole color pattern, e.g. (0, 1, 1, 3)
        - optional partial medium-pole pattern completable to a valid color pattern
        - optional medium-pole count congruence implied by solar_total
        - optional packing-style non-overlap between substation 2x2 footprints and inflated medium-pole footprints
        - optional exact tileability via selectable substation-substation bridge pairs

    Factorio wire-center offsets:
        substation:           (1.0, 1.0)
        medium-electric-pole: (0.5, 0.5)
    """

    d = grid
    dd = d**2

    if fixed_root_substation is not None:
        if (
            isinstance(fixed_root_substation, (bool, np.bool_))
            or not isinstance(fixed_root_substation, (int, np.integer))
            or not 0 <= fixed_root_substation < dd
        ):
            raise ValueError("fixed_root_substation must be a valid grid index or None.")
        fixed_root_substation = int(fixed_root_substation)

    if isinstance(solar_total, (bool, np.bool_)) or not isinstance(solar_total, (int, np.integer)) or solar_total < 0:
        raise ValueError("solar_total must be a nonnegative integer.")

    solar_total = int(solar_total)

    sub_offset = (1.0, 1.0)
    med_offset = (0.5, 0.5)

    if fixed_root_substation is None:
        # A free root must be no larger than every substation that is fixed on.
        # Restricting the candidate block this way is exact and avoids thousands
        # of root binaries that can never be selected.
        fixed_root_upper_bounds = []
        if use_corner_substations:
            fixed_root_upper_bounds.append(8 * grid + 8)
        if use_side_substations:
            fixed_root_upper_bounds.append(8 * grid + (grid // 2 - 1))
        if use_t_substations:
            fixed_root_upper_bounds.append(8 * grid + 8)

        root_candidate_max = (
            min(fixed_root_upper_bounds) if fixed_root_upper_bounds else dd - 1
        )
        root_selector_count = root_candidate_max + 1
        root_selector_offset = 2 * dd
    else:
        root_candidate_max = -1
        root_selector_count = 0
        root_selector_offset = None

    base_size = 2 * dd + root_selector_count
    size = base_size

    if balance_medium_colors:
        color_offset = size

        a_var = color_offset + 0
        b_var = color_offset + 1
        c_var = color_offset + 2
        d_var = color_offset + 3
        X_var = color_offset + 4
        q_var = color_offset + 5

        size += 6

    completion_patterns = None
    completion_selector_offset = None

    if medium_color_completion_total is not None:
        if isinstance(medium_color_completion_total, (bool, np.bool_)):
            raise ValueError("medium_color_completion_total must be one of 2, 5, 6, 9, 10, or 14.")

        medium_color_completion_total = int(medium_color_completion_total)
        patterns_by_total = {
            2: MEDIUM_COLOR_PATTERNS_2,
            5: MEDIUM_COLOR_PATTERNS_5,
            6: MEDIUM_COLOR_PATTERNS_6,
            9: MEDIUM_COLOR_PATTERNS_9,
            10: MEDIUM_COLOR_PATTERNS_10,
            14: MEDIUM_COLOR_PATTERNS_14,
        }

        if medium_color_completion_total not in patterns_by_total:
            raise ValueError("medium_color_completion_total must be one of 2, 5, 6, 9, 10, or 14.")

        completion_patterns = patterns_by_total[medium_color_completion_total]
        completion_selector_offset = size
        size += len(completion_patterns)

    color_mod_03_var = None
    color_mod_12_var = None

    if exact_medium_color_completion_total is not None:
        if (
            isinstance(exact_medium_color_completion_total, (bool, np.bool_))
            or not isinstance(exact_medium_color_completion_total, (int, np.integer))
            or exact_medium_color_completion_total < 0
        ):
            raise ValueError("exact_medium_color_completion_total must be a nonnegative integer or None.")

        exact_medium_color_completion_total = int(exact_medium_color_completion_total)
        color_mod_03_var = size
        color_mod_12_var = size + 1
        size += 2


    if use_substation_bridge_tileability:
        bridge_coords = list(range(0, grid, substation_bridge_step))
        bridge_pair_count_per_axis = len(bridge_coords)
        bridge_selector_offset = size
        size += 2 * bridge_pair_count_per_axis

    rows = []
    b_lb = []
    b_ub = []
    fixed_substation_indices = []

    def node_var(kind, idx):
        if kind == "sub":
            return idx
        if kind == "med":
            return dd + idx
        raise ValueError("Unknown node kind")

    if fixed_root_substation is None:
        ## Exactly one root, and root_i <= sub_i

        row = np.zeros(size)
        row[root_selector_offset:root_selector_offset + root_selector_count] = 1
        rows.append(row)
        b_lb.append(1)
        b_ub.append(1)

        for i in range(root_selector_count):
            row = np.zeros(size)
            row[root_selector_offset + i] = 1
            row[i] = -1
            rows.append(row)
            b_lb.append(-np.inf)
            b_ub.append(0)

        # No explicit lowest-index prefix constraints are needed. With exactly
        # one root, every other selected node requiring a lower-rank parent,
        # any selected node below the root would create an impossible finite
        # descending chain. Connectivity therefore enforces this implicitly.

    ## Optional fixed three-corner substations

    if use_corner_substations:
        corner_lo = 8
        corner_hi = grid - 10

        corner_substations = [
            corner_lo * grid + corner_lo,
            corner_lo * grid + corner_hi,
            corner_hi * grid + corner_lo,
        ]

        fixed_substation_indices.extend(corner_substations)

    ## Optional 4-sides substations

    if use_side_substations:
        side_offset = 8
        side_mid = grid // 2 - 1

        side_substations = [
            side_offset * grid + side_mid,                 # top side
            (grid - 10) * grid + side_mid,                 # bottom side
            side_mid * grid + side_offset,                 # left side
            side_mid * grid + (grid - 10),                 # right side
        ]

        fixed_substation_indices.extend(side_substations)

    if use_t_substations:
        side_lo = 8
        side_hi = grid - 10
        side_mid = grid // 2 - 1

        t_substations = [
            side_lo * grid + side_lo,    # 1: top-left
            side_lo * grid + side_mid,   # 2: top-middle
            side_lo * grid + side_hi,    # 3: top-right
            side_hi * grid + side_mid,   # 4: bottom-middle, tileable stem
        ]

        fixed_substation_indices.extend(t_substations)

    ## Optional exact tileability via selectable substation-substation bridge pairs
    ##
    ## This is exact, but much smaller than all possible bridge pairs. It forces
    ## one selected substation-substation bridge across each wrapped axis. The
    ## bridge location can move along the seam. For grid=50, low=8 and high=40
    ## give wrapped axis distance 18, exactly the substation wire range. With
    ## substation_bridge_step=1 this adds 2*grid binary selectors and 4*grid + 2
    ## constraints, i.e. 100 variables for a 50x50 tile.

    if use_substation_bridge_tileability:
        bridge_low = 8
        bridge_high = grid - 10
        wrapped_gap = bridge_low + (grid - bridge_high)

        if wrapped_gap > 18:
            raise ValueError("Substation bridge anchors are too far apart for range 18.")

        for axis in [0, 1]:
            axis_offset = bridge_selector_offset + axis * bridge_pair_count_per_axis

            # At least one exact substation-substation bridge pair on this axis.
            row = np.zeros(size)
            row[axis_offset:axis_offset + bridge_pair_count_per_axis] = 1
            rows.append(row)
            b_lb.append(1)
            b_ub.append(np.inf)

            for p, coord in enumerate(bridge_coords):
                selector = axis_offset + p

                if axis == 0:
                    idx_a = bridge_low * grid + coord
                    idx_b = bridge_high * grid + coord
                else:
                    idx_a = coord * grid + bridge_low
                    idx_b = coord * grid + bridge_high

                # selector <= sub_idx_a
                row = np.zeros(size)
                row[selector] = 1
                row[idx_a] = -1
                rows.append(row)
                b_lb.append(-np.inf)
                b_ub.append(0)

                # selector <= sub_idx_b
                row = np.zeros(size)
                row[selector] = 1
                row[idx_b] = -1
                rows.append(row)
                b_lb.append(-np.inf)
                b_ub.append(0)

    ## Medium pole count limiter

    if np.isfinite(max_medium_poles):
        row = np.zeros(size)
        row[dd:2*dd] = 1
        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(max_medium_poles)

    ## Optional exact medium-pole color pattern
    ##
    ## medium_color_pattern = (m0, m1, m2, m3) forces the selected medium
    ## poles to have exactly that checkerboard color distribution, where
    ## color = 2 * (i % 2) + (j % 2).
    ##
    ## This can be used to test one of the feasible M=5 or M=9 patterns
    ## directly instead of letting the color-completion equations choose one.

    if medium_color_pattern is not None:
        medium_color_pattern = tuple(int(v) for v in medium_color_pattern)

        if len(medium_color_pattern) != 4:
            raise ValueError("medium_color_pattern must be a length-4 tuple/list: (m0, m1, m2, m3).")

        if any(v < 0 for v in medium_color_pattern):
            raise ValueError("medium_color_pattern entries must be nonnegative integers.")

        for color, target_count in enumerate(medium_color_pattern):
            row = np.zeros(size)

            for k in range(dd):
                i = k // grid
                j = k % grid

                if 2 * (i % 2) + (j % 2) == color:
                    row[dd + k] = 1

            rows.append(row)
            b_lb.append(target_count)
            b_ub.append(target_count)

    ## Optional partial medium-pole color pattern
    ##
    ## Choose one valid target pattern of the requested total and require the
    ## Stage-A medium-pole counts to fit inside it componentwise. Any unused
    ## target slots are intentionally left for Stage B. For example, actual
    ## counts (1, 1, 2, 3) fit inside the valid M=10 target (3, 2, 2, 3).

    if completion_patterns is not None:
        selector_vars = np.arange(
            completion_selector_offset,
            completion_selector_offset + len(completion_patterns),
            dtype=int,
        )

        row = np.zeros(size)
        row[selector_vars] = 1
        rows.append(row)
        b_lb.append(1)
        b_ub.append(1)

        for color in range(4):
            row = np.zeros(size)

            for k in range(dd):
                i = k // grid
                j = k % grid

                if 2 * (i % 2) + (j % 2) == color:
                    row[dd + k] = 1

            for selector, pattern in zip(selector_vars, completion_patterns):
                row[selector] = -pattern[color]

            rows.append(row)
            b_lb.append(-np.inf)
            b_ub.append(0)

    ## Compact exact color completion
    ##
    ## With an exact medium-pole total, the valid color patterns are exactly
    ## characterized by m0 - m3 and m1 - m2 both being divisible by 3. This
    ## replaces the pattern-selector disjunction with two small integers.

    if exact_medium_color_completion_total is not None:
        color_indices = [[] for _ in range(4)]
        for k in range(dd):
            i = k // grid
            j = k % grid
            color_indices[2 * (i % 2) + (j % 2)].append(k)

        row = np.zeros(size)
        row[dd + np.asarray(color_indices[0], dtype=int)] = 1
        row[dd + np.asarray(color_indices[3], dtype=int)] = -1
        row[color_mod_03_var] = -3
        rows.append(row)
        b_lb.append(0)
        b_ub.append(0)

        row = np.zeros(size)
        row[dd + np.asarray(color_indices[1], dtype=int)] = 1
        row[dd + np.asarray(color_indices[2], dtype=int)] = -1
        row[color_mod_12_var] = -3
        rows.append(row)
        b_lb.append(0)
        b_ub.append(0)

    ## Optional packing-style obstacle spacing
    ##
    ## Build a placement/colocation matrix exactly like the packing objective,
    ## but only for network obstacles:
    ##     - substations occupy their real 2x2 footprint
    ##     - medium poles occupy a virtual centered (2R+1)x(2R+1) footprint
    ##
    ## Then each tile may be claimed by at most one such obstacle footprint.
    ## This prevents inflated medium poles from overlapping each other or
    ## overlapping substation footprints, without pairwise constraints.

    if use_medium_pole_obstacle_spacing:
        R = medium_pole_obstacle_radius
        obstacle_A = np.zeros((size, dd))

        for k in range(dd):
            obstacle_A[k, util.block_indices(k, 0, 1, grid)] = 1
            obstacle_A[dd + k, util.block_indices(k, R, R, grid)] = 1

        rows.extend(obstacle_A.T)
        b_lb.extend(-np.inf * np.ones(dd))
        b_ub.extend(np.ones(dd))


    ## Solar + medium-pole color completion feasibility
    ##
    ## There must exist nonnegative integers a,b,c,d,X such that:
    ##
    ##   4a + 2b + 2c + 1d + m_0 = X
    ##   2a + 4b + 1c + 2d + m_1 = X
    ##   2a + 1b + 4c + 2d + m_2 = X
    ##   1a + 2b + 2c + 4d + m_3 = X
    ##
    ## where m_q is the number of selected medium poles rooted on color q.
    ##
    ## Substations are ignored because they cover one tile of each color.
    ##
    ## With solar_total panels, total solar color coverage is 9*solar_total.
    ## Equal color totals require 9*solar_total + M = 4X, so M must have
    ## residue (-9*solar_total) mod 4. We add this explicitly to cut off
    ## impossible medium-pole counts.

    if balance_medium_colors:
        color_indices = [[] for _ in range(4)]

        for k in range(dd):
            i = k // grid
            j = k % grid
            color = 2 * (i % 2) + (j % 2)
            color_indices[color].append(k)

        solar_color_matrix = np.array([
            [4, 2, 2, 1],
            [2, 4, 1, 2],
            [2, 1, 4, 2],
            [1, 2, 2, 4],
        ])

        solar_vars = [a_var, b_var, c_var, d_var]

        # Exactly solar_total panels across the four color configurations.
        row = np.zeros(size)
        row[solar_vars] = 1
        rows.append(row)
        b_lb.append(solar_total)
        b_ub.append(solar_total)

        # Total medium-pole count must have the required residue modulo 4.
        # This is implied by the four color equations plus the panel total,
        # but adding it explicitly gives the solver a stronger formulation.
        residue = (-9 * solar_total) % 4
        row = np.zeros(size)
        row[dd:2*dd] = 1
        row[q_var] = -4
        rows.append(row)
        b_lb.append(residue)
        b_ub.append(residue)

        for color in range(4):
            row = np.zeros(size)

            # Solar-panel contribution for this color.
            for t in range(4):
                row[solar_vars[t]] = solar_color_matrix[color, t]

            # Medium-pole contribution for this color.
            row[dd + np.array(color_indices[color], dtype=int)] = 1

            # Equal target X for all colors.
            row[X_var] = -1

            rows.append(row)
            b_lb.append(0)
            b_ub.append(0)

    ## Coverage matrices

    E_sub = np.zeros((dd, dd))
    E_med = np.zeros((dd, dd))
    C5 = np.zeros((dd, dd))

    for i in range(dd):

        block_indices = util.block_indices(i, 8, 9, grid)
        E_sub[i, block_indices] = 1

        block_indices = util.block_indices(i, 3, 3, grid)
        E_med[i, block_indices] = 1

        block_indices = util.block_indices(i, 0, 4, grid)
        C5[i, block_indices] = 1

    M_sub = np.clip(C5 @ E_sub.T, 0, 1)
    M_med = np.clip(C5 @ E_med.T, 0, 1)

    ## Floor coverage constraints

    for k in range(dd):
        row = np.zeros(size)

        sub_cover = np.flatnonzero(M_sub[k])
        med_cover = np.flatnonzero(M_med[k])

        row[sub_cover] = 1
        row[dd + med_cover] = 1

        rows.append(row)
        b_lb.append(1)
        b_ub.append(np.inf)

    ## Roboport 4x4 physical clearance, with optional full tile coverage
    ##
    ## For 50x50, there is one central roboport at root 1173 = (23, 23).
    ## For 100x100, there are two roboports at roots 2323 = (23, 23)
    ## and 7373 = (73, 73), i.e. the centers of the bottom-left and
    ## top-right 50x50 quadrants.

    if grid == 50:
        roboport_roots = [1173]
    elif grid == 100:
        roboport_roots = [2323, 7373]
    else:
        raise ValueError("Roboport clearance restrictions are defined only for grid=50 or grid=100.")

    central_tiles = []
    for root in roboport_roots:
        root_i = root // grid
        root_j = root % grid
        central_tiles.extend([
            i * grid + j
            for i in range(root_i, root_i + 4)
            for j in range(root_j, root_j + 4)
        ])

    central_tiles = sorted(set(central_tiles))

    if use_central_4x4_restriction:

        ## Each roboport tile must have electric coverage.

        for k in central_tiles:
            row = np.zeros(size)

            sub_cover = np.flatnonzero(E_sub[:, k])
            med_cover = np.flatnonzero(E_med[:, k])

            row[sub_cover] = 1
            row[dd + med_cover] = 1

            rows.append(row)
            b_lb.append(1)
            b_ub.append(np.inf)

    ## Roboport 4x4 areas must always be physically unblocked by substations and medium poles.

    row = np.zeros(size)

    for tile in central_tiles:
        for s in range(dd):
            if tile in util.block_indices(s, 0, 1, grid):
                row[s] = 1

        row[dd + tile] = 1

    rows.append(row)
    b_lb.append(-np.inf)
    b_ub.append(0)
    ## Connectivity matrices with mixed offsets

    N_sub_sub = np.zeros((dd, dd))
    N_sub_med = np.zeros((dd, dd))
    N_med_sub = np.zeros((dd, dd))
    N_med_med = np.zeros((dd, dd))

    for i in range(dd):

        N_sub_sub[i, util.circle_indices_mixed_offsets(i, 18, grid, source_offset=sub_offset, target_offset=sub_offset, consider_wrapping=False)] = 1
        N_sub_med[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=sub_offset, target_offset=med_offset, consider_wrapping=False)] = 1
        N_med_sub[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=med_offset, target_offset=sub_offset, consider_wrapping=False)] = 1
        N_med_med[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=med_offset, target_offset=med_offset, consider_wrapping=False)] = 1

    ## Connectivity constraints

    rank = np.arange(dd)
    if fixed_root_substation is not None:
        rank[fixed_root_substation] = -1

    for j in range(dd):
        if fixed_root_substation is not None and j == fixed_root_substation:
            continue

        row = np.zeros(size)
        row[j] = 1
        if root_selector_offset is not None and j <= root_candidate_max:
            row[root_selector_offset + j] = -1

        sub_parents = np.flatnonzero(N_sub_sub[:, j])
        sub_parents = sub_parents[rank[sub_parents] < rank[j]]

        med_parents = np.flatnonzero(N_med_sub[:, j])
        med_parents = med_parents[rank[med_parents] < rank[j]]

        row[sub_parents] -= 1
        row[dd + med_parents] -= 1

        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(0)

    for j in range(dd):
        row = np.zeros(size)
        row[dd + j] = 1

        sub_parents = np.flatnonzero(N_sub_med[:, j])
        sub_parents = sub_parents[rank[sub_parents] < rank[j]]

        med_parents = np.flatnonzero(N_med_med[:, j])
        med_parents = med_parents[rank[med_parents] < rank[j]]

        row[sub_parents] -= 1
        row[dd + med_parents] -= 1

        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(0)

    A_total = np.vstack(rows)
    b_lb = np.array(b_lb)
    b_ub = np.array(b_ub)

    n = size

    c = np.zeros(n)
    c[0:dd] = 1
    c[dd:2*dd] = 0.25 * medium_weight

    lb = np.zeros(n)
    ub = np.ones(n)

    # Existing network variables are binary.
    ub[:base_size] = 1

    if fixed_substation_indices:
        lb[np.unique(fixed_substation_indices)] = 1

    if fixed_root_substation is not None:
        # The compact fixed-root formulation contains no root selectors.
        lb[fixed_root_substation] = 1

    if exact_medium_color_completion_total is not None:
        color_mod_bound = exact_medium_color_completion_total // 3
        lb[color_mod_03_var] = -color_mod_bound
        ub[color_mod_03_var] = color_mod_bound
        lb[color_mod_12_var] = -color_mod_bound
        ub[color_mod_12_var] = color_mod_bound

    # Color-completion witness variables are nonnegative integers.
    # Bounds for exactly solar_total panels plus selected medium-pole coverage.
    if balance_medium_colors:
        ub[a_var] = solar_total
        ub[b_var] = solar_total
        ub[c_var] = solar_total
        ub[d_var] = solar_total
        ub[X_var] = 4 * solar_total + dd
        ub[q_var] = dd

    integrality = np.ones(n)

    return A_total, b_lb, b_ub, n, c, lb, ub, integrality

def construct_network_floor_coverage_system_tileable_free_root_color_mae(
    grid,
    substation_location_scores,
    medium_location_scores,
    interaction_pairs=None,
    interaction_scores=None,
    minimum_supported_pair_products=None,
    exact_substation_count=5,
    exact_medium_poles=10,
    medium_weight=1.0,
    fixed_root_substation=None,
    use_corner_substations=False,
    use_side_substations=False,
    use_t_substations=False,
    balance_medium_colors=False,
    solar_total=198,
    medium_color_pattern=None,
    medium_color_completion_total=None,
    medium_roboport_forbidden_gap=1,
    substation_roboport_forbidden_gap=1,
    use_medium_pole_obstacle_spacing=False,
    medium_pole_obstacle_radius=4,
    use_substation_bridge_tileability=False,
    substation_bridge_step=1,
    use_central_4x4_restriction=True,
):
    """
    Exact-count Stage-A planner maximizing a learned Stage-B location score.

    This is a separate alternative to the damage formulation. It
    keeps the same network, connectivity, color, roboport, and exact-count
    restrictions, but its objective is the prediction

        sum(substation_location_scores[k] * substation[k])
        + sum(medium_location_scores[k] * medium[k])
        + sum(interaction_scores[q] * network[i_q] * network[j_q]).

    The returned model minimizes the negative of that score. Optional pair
    products are represented by continuous McCormick auxiliaries. Since both
    network endpoints are binary, each auxiliary is forced to the exact 0/1
    product in every incumbent without becoming an additional integer
    variable. No damage roots or damage constraints are introduced. The
    caller is responsible for fitting or otherwise supplying the scores.
    """

    for name, value in [
        ("exact_substation_count", exact_substation_count),
        ("exact_medium_poles", exact_medium_poles),
        ("medium_roboport_forbidden_gap", medium_roboport_forbidden_gap),
        ("substation_roboport_forbidden_gap", substation_roboport_forbidden_gap),
    ]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer.")

    exact_substation_count = int(exact_substation_count)
    exact_medium_poles = int(exact_medium_poles)
    medium_roboport_forbidden_gap = int(medium_roboport_forbidden_gap)
    substation_roboport_forbidden_gap = int(substation_roboport_forbidden_gap)

    dd = grid**2
    substation_location_scores = np.asarray(
        substation_location_scores, dtype=float
    ).reshape(-1)
    medium_location_scores = np.asarray(
        medium_location_scores, dtype=float
    ).reshape(-1)

    for name, scores in [
        ("substation_location_scores", substation_location_scores),
        ("medium_location_scores", medium_location_scores),
    ]:
        if scores.shape != (dd,):
            raise ValueError(f"{name} must have exactly grid**2 entries.")
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{name} must contain only finite values.")

    if interaction_pairs is None:
        interaction_pairs = np.empty((0, 2), dtype=int)
    else:
        interaction_pairs = np.asarray(interaction_pairs, dtype=int)
    if interaction_scores is None:
        interaction_scores = np.zeros(len(interaction_pairs))
    else:
        interaction_scores = np.asarray(interaction_scores, dtype=float).reshape(-1)

    if interaction_pairs.ndim != 2 or interaction_pairs.shape[1] != 2:
        raise ValueError("interaction_pairs must have shape (number_of_pairs, 2).")
    if interaction_scores.shape != (len(interaction_pairs),):
        raise ValueError("interaction_scores must have one entry per interaction pair.")
    if not np.all(np.isfinite(interaction_scores)):
        raise ValueError("interaction_scores must contain only finite values.")
    if len(interaction_pairs):
        if np.any(interaction_pairs < 0) or np.any(interaction_pairs >= 2 * dd):
            raise ValueError("interaction pair indices must address the two network blocks.")
        interaction_pairs = np.sort(interaction_pairs, axis=1)
        if np.any(interaction_pairs[:, 0] == interaction_pairs[:, 1]):
            raise ValueError("interaction pair endpoints must be different.")
        if len(np.unique(interaction_pairs, axis=0)) != len(interaction_pairs):
            raise ValueError("interaction_pairs must not contain duplicates.")

    if minimum_supported_pair_products is not None:
        if (
            isinstance(minimum_supported_pair_products, (bool, np.bool_))
            or not isinstance(minimum_supported_pair_products, (int, np.integer))
            or not 0 <= minimum_supported_pair_products <= len(interaction_pairs)
        ):
            raise ValueError(
                "minimum_supported_pair_products must be an integer between "
                "zero and the number of interaction pairs."
            )
        minimum_supported_pair_products = int(minimum_supported_pair_products)

    # With an exact medium total, the two modulo-3 equations are equivalent to
    # the much larger valid-pattern selector disjunction.
    use_compact_color_completion = (
        medium_color_completion_total is not None
        and medium_color_completion_total == exact_medium_poles
    )
    selector_completion_total = (
        None if use_compact_color_completion else medium_color_completion_total
    )
    compact_completion_total = (
        exact_medium_poles if use_compact_color_completion else None
    )

    A, b_lb, b_ub, n, _, lb, ub, integrality = (
        construct_network_floor_coverage_system_tileable_free_root_color_full(
            grid,
            max_medium_poles=np.inf,
            medium_weight=medium_weight,
            fixed_root_substation=fixed_root_substation,
            use_corner_substations=use_corner_substations,
            use_side_substations=use_side_substations,
            use_t_substations=use_t_substations,
            balance_medium_colors=balance_medium_colors,
            solar_total=solar_total,
            medium_color_pattern=medium_color_pattern,
            medium_color_completion_total=selector_completion_total,
            exact_medium_color_completion_total=compact_completion_total,
            use_medium_pole_obstacle_spacing=use_medium_pole_obstacle_spacing,
            medium_pole_obstacle_radius=medium_pole_obstacle_radius,
            use_substation_bridge_tileability=use_substation_bridge_tileability,
            substation_bridge_step=substation_bridge_step,
            use_central_4x4_restriction=use_central_4x4_restriction,
        )
    )

    # Keep exactly the same roboport-gap bounds as the damage constructor.
    if grid == 50:
        roboport_roots = [1173]
    elif grid == 100:
        roboport_roots = [2323, 7373]
    else:
        raise ValueError("Roboport gap restrictions are defined only for grid=50 or grid=100.")

    for root in roboport_roots:
        root_i = root // grid
        root_j = root % grid
        offset = medium_roboport_forbidden_gap + 1

        outer_row_lo = max(0, root_i - offset)
        outer_row_hi = min(grid, root_i + 4 + offset)
        outer_col_lo = max(0, root_j - offset)
        outer_col_hi = min(grid, root_j + 4 + offset)

        inner_row_lo = root_i - offset + 1
        inner_row_hi = root_i + 4 + offset - 1
        inner_col_lo = root_j - offset + 1
        inner_col_hi = root_j + 4 + offset - 1

        for i in range(outer_row_lo, outer_row_hi):
            for j in range(outer_col_lo, outer_col_hi):
                inside_inner_ring = (
                    inner_row_lo <= i < inner_row_hi
                    and inner_col_lo <= j < inner_col_hi
                )
                if not inside_inner_ring:
                    ub[dd + i * grid + j] = 0

        sub_outer_row_lo = max(
            0, root_i - substation_roboport_forbidden_gap - 2
        )
        sub_outer_row_hi = min(
            grid, root_i + 4 + substation_roboport_forbidden_gap + 1
        )
        sub_outer_col_lo = max(
            0, root_j - substation_roboport_forbidden_gap - 2
        )
        sub_outer_col_hi = min(
            grid, root_j + 4 + substation_roboport_forbidden_gap + 1
        )

        sub_inner_row_lo = sub_outer_row_lo + 1
        sub_inner_row_hi = sub_outer_row_hi - 1
        sub_inner_col_lo = sub_outer_col_lo + 1
        sub_inner_col_hi = sub_outer_col_hi - 1

        for i in range(sub_outer_row_lo, sub_outer_row_hi):
            for j in range(sub_outer_col_lo, sub_outer_col_hi):
                inside_sub_inner_ring = (
                    sub_inner_row_lo <= i < sub_inner_row_hi
                    and sub_inner_col_lo <= j < sub_inner_col_hi
                )
                if not inside_sub_inner_ring:
                    ub[i * grid + j] = 0

    interaction_count = len(interaction_pairs)
    interaction_offset = n
    new_n = n + interaction_count

    base_A = sp.hstack(
        [sp.csr_matrix(A), sp.csr_matrix((len(b_lb), interaction_count))],
        format="csr",
    )

    exact_rows = sp.lil_matrix((2, new_n))
    exact_rows[0, :dd] = 1
    exact_rows[1, dd:2 * dd] = 1

    interaction_rows = sp.lil_matrix((3 * interaction_count, new_n))
    interaction_lb = np.empty(3 * interaction_count)
    interaction_ub = np.empty(3 * interaction_count)
    for pair_number, (left, right) in enumerate(interaction_pairs):
        product = interaction_offset + pair_number
        row = 3 * pair_number

        # product <= left
        interaction_rows[row, product] = 1
        interaction_rows[row, left] = -1
        interaction_lb[row] = -np.inf
        interaction_ub[row] = 0

        # product <= right
        interaction_rows[row + 1, product] = 1
        interaction_rows[row + 1, right] = -1
        interaction_lb[row + 1] = -np.inf
        interaction_ub[row + 1] = 0

        # product >= left + right - 1
        interaction_rows[row + 2, product] = 1
        interaction_rows[row + 2, left] = -1
        interaction_rows[row + 2, right] = -1
        interaction_lb[row + 2] = -1
        interaction_ub[row + 2] = np.inf

    if minimum_supported_pair_products is None:
        support_rows = sp.csr_matrix((0, new_n))
        support_lb = np.empty(0)
        support_ub = np.empty(0)
    else:
        support_rows = sp.lil_matrix((1, new_n))
        support_rows[0, interaction_offset:new_n] = 1
        support_rows = support_rows.tocsr()
        support_lb = np.asarray([minimum_supported_pair_products])
        support_ub = np.asarray([np.inf])

    A_total = sp.vstack(
        [
            base_A,
            exact_rows.tocsr(),
            interaction_rows.tocsr(),
            support_rows,
        ],
        format="csr",
    )
    b_lb = np.concatenate(
        [
            b_lb,
            [exact_substation_count, exact_medium_poles],
            interaction_lb,
            support_lb,
        ]
    )
    b_ub = np.concatenate(
        [
            b_ub,
            [exact_substation_count, exact_medium_poles],
            interaction_ub,
            support_ub,
        ]
    )

    c = np.zeros(new_n)
    c[:dd] = -substation_location_scores
    c[dd:2 * dd] = -medium_location_scores
    c[interaction_offset:] = -interaction_scores

    lb = np.concatenate([lb, np.zeros(interaction_count)])
    ub = np.concatenate([ub, np.ones(interaction_count)])
    integrality = np.concatenate([integrality, np.zeros(interaction_count)])

    return A_total, b_lb, b_ub, new_n, c, lb, ub, integrality

def construct_network_floor_coverage_system_tileable_free_root_color_min_damage(
    grid,
    exact_substation_count=5,
    exact_medium_poles=9,
    solar_damage_weight=1.0,
    accumulator_damage_weight=0.5,
    medium_weight=1.0,
    fixed_root_substation=None,
    use_corner_substations=False,
    use_side_substations=False,
    use_t_substations=False,
    balance_medium_colors=False,
    solar_total=199,
    medium_color_pattern=None,
    medium_color_completion_total=None,
    medium_roboport_forbidden_gap=1,
    substation_roboport_forbidden_gap=1,
    use_medium_pole_obstacle_spacing=False,
    medium_pole_obstacle_radius=4,
    use_substation_bridge_tileability=False,
    substation_bridge_step=1,
    use_central_4x4_restriction=True,
):
    """
    Exact-count Stage-A network planner minimizing damage to Stage B.

    This keeps the full network formulation from
    construct_network_floor_coverage_system_tileable_free_root_color_full,
    then adds:

        - exactly exact_substation_count selected substations
        - exactly exact_medium_poles selected medium poles
        - one damage variable for every periodic 3x3 solar root
        - one damage variable for every periodic 2x2 accumulator root
        - optional forbidden medium-pole gap around the fixed roboport
        - optional forbidden substation-footprint gap around the fixed roboport

    A building root counts as damaged only when:

        - its footprint does not overlap a selected substation, medium pole,
          or the fixed central roboport, and
        - none of its tiles receives electric coverage from the selected
          network.

    Thus roots already made unusable by a physical network obstacle are not
    penalized a second time. The objective is the weighted number of damaged
    solar and accumulator roots. Exact network counts make the original
    substation/medium-pole objective constant, so it is replaced completely.

    Damage variables are continuous in [0,1]. Because the network variables
    are integral and damage has a positive objective coefficient, they still
    take their exact required 0/1 values in every integer network solution.
    This avoids adding 2*grid**2 unnecessary integer variables.

    The returned constraint matrix is CSR sparse. This preserves the standard
    return interface while avoiding a very large dense 2*grid**2-column
    augmentation.
    """

    for name, value in [
        ("exact_substation_count", exact_substation_count),
        ("exact_medium_poles", exact_medium_poles),
        ("medium_roboport_forbidden_gap", medium_roboport_forbidden_gap),
        ("substation_roboport_forbidden_gap", substation_roboport_forbidden_gap),
    ]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")

    for name, value in [
        ("solar_damage_weight", solar_damage_weight),
        ("accumulator_damage_weight", accumulator_damage_weight),
    ]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number.")

    exact_substation_count = int(exact_substation_count)
    exact_medium_poles = int(exact_medium_poles)
    medium_roboport_forbidden_gap = int(medium_roboport_forbidden_gap)
    substation_roboport_forbidden_gap = int(substation_roboport_forbidden_gap)

    # When Stage A already fixes the medium total, color completion no longer
    # needs a selector for every valid pattern. Two modulo-3 equations describe
    # exactly the same set of complete patterns.
    use_compact_color_completion = (
        medium_color_completion_total is not None
        and medium_color_completion_total == exact_medium_poles
    )
    selector_completion_total = (
        None if use_compact_color_completion else medium_color_completion_total
    )
    compact_completion_total = (
        exact_medium_poles if use_compact_color_completion else None
    )

    A, b_lb, b_ub, n, _, lb, ub, integrality = (
        construct_network_floor_coverage_system_tileable_free_root_color_full(
            grid,
            max_medium_poles=np.inf,
            medium_weight=medium_weight,
            fixed_root_substation=fixed_root_substation,
            use_corner_substations=use_corner_substations,
            use_side_substations=use_side_substations,
            use_t_substations=use_t_substations,
            balance_medium_colors=balance_medium_colors,
            solar_total=solar_total,
            medium_color_pattern=medium_color_pattern,
            medium_color_completion_total=selector_completion_total,
            exact_medium_color_completion_total=compact_completion_total,
            use_medium_pole_obstacle_spacing=use_medium_pole_obstacle_spacing,
            medium_pole_obstacle_radius=medium_pole_obstacle_radius,
            use_substation_bridge_tileability=use_substation_bridge_tileability,
            substation_bridge_step=substation_bridge_step,
            use_central_4x4_restriction=use_central_4x4_restriction,
        )
    )

    dd = grid**2
    solar_damage_offset = n
    accumulator_damage_offset = n + dd
    new_n = n + 2 * dd

    def placement_matrix(n_left, n_right, consider_wrapping):
        row_indices = []
        column_indices = []

        for root in range(dd):
            tiles = util.block_indices(
                root,
                n_left,
                n_right,
                grid,
                consider_wrapping=consider_wrapping,
            )
            row_indices.extend([root] * len(tiles))
            column_indices.extend(tiles)

        data = np.ones(len(row_indices))
        return sp.csr_matrix(
            (data, (row_indices, column_indices)),
            shape=(dd, dd),
        )

    # Physical footprints and electric coverage footprints.
    F_sub = placement_matrix(0, 1, consider_wrapping=True)
    E_sub = placement_matrix(8, 9, consider_wrapping=True)
    E_med = placement_matrix(3, 3, consider_wrapping=True)

    # Periodic Stage-B building footprints.
    C_solar = placement_matrix(0, 2, consider_wrapping=True)
    C_accumulator = placement_matrix(0, 1, consider_wrapping=True)

    def binary_product(left, right_transpose):
        result = (left @ right_transpose).tocsr()
        result.data[:] = 1
        result.eliminate_zeros()
        return result

    def damage_eligibility(building_matrix):
        # Electric coverage of at least one tile in the building footprint.
        electric_sub = binary_product(building_matrix, E_sub.T)
        electric_med = binary_product(building_matrix, E_med.T)

        # Physical overlap with the fixed network footprint.
        obstacle_sub = binary_product(building_matrix, F_sub.T)
        obstacle_med = building_matrix.copy()

        eligible_sub = electric_sub + obstacle_sub
        eligible_sub.data[:] = 1

        eligible_med = electric_med + obstacle_med
        eligible_med.data[:] = 1

        return eligible_sub.tocsr(), eligible_med.tocsr()

    solar_sub, solar_med = damage_eligibility(C_solar)
    accumulator_sub, accumulator_med = damage_eligibility(C_accumulator)

    # Roots intersecting the fixed central roboport are already physically
    # unavailable, so their damage lower bound is zero.
    if grid == 50:
        roboport_roots = [1173]
    elif grid == 100:
        roboport_roots = [2323, 7373]
    else:
        raise ValueError("Roboport damage restrictions are defined only for grid=50 or grid=100.")

    roboport_tiles = np.zeros(dd)
    for root in roboport_roots:
        roboport_tiles[util.block_indices(root, 0, 3, grid)] = 1

    # Forbid only the ring that leaves exactly the requested number of empty
    # tiles between a medium pole and the 4x4 roboport. With a gap of 1, the
    # immediately adjacent (touching) ring remains available, while the next
    # ring out is forbidden. The roboport footprint itself is already blocked
    # by the base model's physical-clearance constraint.
    for root in roboport_roots:
        root_i = root // grid
        root_j = root % grid
        offset = medium_roboport_forbidden_gap + 1

        outer_row_lo = max(0, root_i - offset)
        outer_row_hi = min(grid, root_i + 4 + offset)
        outer_col_lo = max(0, root_j - offset)
        outer_col_hi = min(grid, root_j + 4 + offset)

        inner_row_lo = root_i - offset + 1
        inner_row_hi = root_i + 4 + offset - 1
        inner_col_lo = root_j - offset + 1
        inner_col_hi = root_j + 4 + offset - 1

        for i in range(outer_row_lo, outer_row_hi):
            for j in range(outer_col_lo, outer_col_hi):
                inside_inner_ring = (
                    inner_row_lo <= i < inner_row_hi
                    and inner_col_lo <= j < inner_col_hi
                )
                if not inside_inner_ring:
                    ub[dd + i * grid + j] = 0

        # A substation occupies a 2x2 footprint. For a one-tile forbidden gap,
        # roots on the outer perimeter 20..28 around the 23..26 roboport are
        # forbidden, while roots on the inner 21..27 perimeter may touch it.
        sub_outer_row_lo = max(
            0, root_i - substation_roboport_forbidden_gap - 2
        )
        sub_outer_row_hi = min(
            grid, root_i + 4 + substation_roboport_forbidden_gap + 1
        )
        sub_outer_col_lo = max(
            0, root_j - substation_roboport_forbidden_gap - 2
        )
        sub_outer_col_hi = min(
            grid, root_j + 4 + substation_roboport_forbidden_gap + 1
        )

        sub_inner_row_lo = sub_outer_row_lo + 1
        sub_inner_row_hi = sub_outer_row_hi - 1
        sub_inner_col_lo = sub_outer_col_lo + 1
        sub_inner_col_hi = sub_outer_col_hi - 1

        for i in range(sub_outer_row_lo, sub_outer_row_hi):
            for j in range(sub_outer_col_lo, sub_outer_col_hi):
                inside_sub_inner_ring = (
                    sub_inner_row_lo <= i < sub_inner_row_hi
                    and sub_inner_col_lo <= j < sub_inner_col_hi
                )
                if not inside_sub_inner_ring:
                    ub[i * grid + j] = 0

    solar_damage_rhs = np.ones(dd)
    solar_damage_rhs[np.asarray(C_solar @ roboport_tiles).ravel() > 0] = 0

    accumulator_damage_rhs = np.ones(dd)
    accumulator_damage_rhs[np.asarray(C_accumulator @ roboport_tiles).ravel() > 0] = 0

    base_A = sp.csr_matrix(A)
    base_A = sp.hstack(
        [base_A, sp.csr_matrix((base_A.shape[0], 2 * dd))],
        format="csr",
    )

    # Exact network counts.
    exact_rows = sp.lil_matrix((2, new_n))
    exact_rows[0, 0:dd] = 1
    exact_rows[1, dd:2*dd] = 1

    base_tail = sp.csr_matrix((dd, n - 2 * dd))
    identity = sp.eye(dd, format="csr")
    zero = sp.csr_matrix((dd, dd))

    solar_damage_rows = sp.hstack(
        [
            solar_sub,
            solar_med,
            base_tail,
            identity,
            zero,
        ],
        format="csr",
    )

    accumulator_damage_rows = sp.hstack(
        [
            accumulator_sub,
            accumulator_med,
            base_tail,
            zero,
            identity,
        ],
        format="csr",
    )

    A_total = sp.vstack(
        [
            base_A,
            exact_rows.tocsr(),
            solar_damage_rows,
            accumulator_damage_rows,
        ],
        format="csr",
    )

    b_lb = np.concatenate(
        [
            b_lb,
            [exact_substation_count, exact_medium_poles],
            solar_damage_rhs,
            accumulator_damage_rhs,
        ]
    )
    b_ub = np.concatenate(
        [
            b_ub,
            [exact_substation_count, exact_medium_poles],
            np.inf * np.ones(dd),
            np.inf * np.ones(dd),
        ]
    )

    c = np.zeros(new_n)
    c[solar_damage_offset:solar_damage_offset + dd] = solar_damage_weight
    c[accumulator_damage_offset:accumulator_damage_offset + dd] = accumulator_damage_weight

    lb = np.concatenate([lb, np.zeros(2 * dd)])
    ub = np.concatenate([ub, np.ones(2 * dd)])
    integrality = np.concatenate([integrality, np.zeros(2 * dd)])

    return A_total, b_lb, b_ub, new_n, c, lb, ub, integrality

def construct_network_floor_coverage_system_tileable_fixed_corners_color_obstacle(
    grid,
    max_substations=np.inf,
    max_medium_poles=np.inf,
    medium_weight=1.0,
    balance_medium_colors=True,
    solar_total=199,
    use_medium_pole_obstacle_spacing=True,
    medium_pole_obstacle_radius=4,
    use_central_4x4_restriction=False,
):
    """
    Simpler Stage-A network planner with fixed tileable corner substations and no root-selector/prefix variables.

    Variables:
        0:dd             substations
        dd:2*dd          medium electric poles
        2*dd:2*dd+6      optional color witness variables a,b,c,d,X,q

    Objective:
        minimize Ns + 0.25 * medium_weight * Nm

    Constraints:
        - fixed three-corner substations at (8,8), (8,grid-10), (grid-10,8)
        - every possible 5x5 fictitious building placement must be covered
        - optional central 4x4 full electric coverage
        - central 4x4 must be physically unblocked by substations / medium poles
        - fixed-root lower-rank connectivity using the lower-left fixed corner as root
        - optional substation count limit
        - optional medium pole count limit
        - optional solar-panel + medium-pole color completion with exactly solar_total solar panels
        - optional packing-style obstacle non-overlap:
            substation real 2x2 footprint vs. medium pole virtual centered 9x9 footprint when radius=4

    Tileability:
        The three fixed corner substations provide exact substation-substation bridges across both wrapped axes.
    """

    d = grid
    dd = d ** 2

    sub_offset = (1.0, 1.0)
    med_offset = (0.5, 0.5)

    base_size = 2 * dd
    size = base_size

    if balance_medium_colors:
        color_offset = size
        a_var = color_offset + 0
        b_var = color_offset + 1
        c_var = color_offset + 2
        d_var = color_offset + 3
        X_var = color_offset + 4
        q_var = color_offset + 5
        size += 6

    rows = []
    b_lb = []
    b_ub = []

    ## Fixed three-corner substations

    corner_lo = 8
    corner_hi = grid - 10

    corner_substations = [
        corner_lo * grid + corner_lo,
        corner_lo * grid + corner_hi,
        corner_hi * grid + corner_lo,
    ]

    root_substation = corner_substations[0]

    row = np.zeros(size)
    for idx in corner_substations:
        row[idx] = 1
    rows.append(row)
    b_lb.append(3)
    b_ub.append(3)

    ## Substation count limiter

    if np.isfinite(max_substations):
        row = np.zeros(size)
        row[:dd] = 1
        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(max_substations)

    ## Medium pole count limiter

    if np.isfinite(max_medium_poles):
        row = np.zeros(size)
        row[dd:2*dd] = 1
        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(max_medium_poles)

    ## Packing-style obstacle spacing
    ## Same construction style as the packing objective: build an occupancy matrix and append its transpose.

    if use_medium_pole_obstacle_spacing:
        R = medium_pole_obstacle_radius
        obstacle_A = np.zeros((size, dd))

        for k in range(dd):
            obstacle_A[k, util.block_indices(k, 0, 1, grid)] = 1
            obstacle_A[dd + k, util.block_indices(k, R, R, grid)] = 1

        rows.extend(obstacle_A.T)
        b_lb.extend(-np.inf * np.ones(dd))
        b_ub.extend(np.ones(dd))

    ## Solar + medium-pole color completion feasibility

    if balance_medium_colors:
        color_indices = [[] for _ in range(4)]

        for k in range(dd):
            i = k // grid
            j = k % grid
            color = 2 * (i % 2) + (j % 2)
            color_indices[color].append(k)

        solar_color_matrix = np.array([
            [4, 2, 2, 1],
            [2, 4, 1, 2],
            [2, 1, 4, 2],
            [1, 2, 2, 4],
        ])

        solar_vars = [a_var, b_var, c_var, d_var]

        # Exactly solar_total solar panels total across the four color configurations.
        row = np.zeros(size)
        row[solar_vars] = 1
        rows.append(row)
        b_lb.append(solar_total)
        b_ub.append(solar_total)

        # Medium total congruence: 9*solar_total + M = 4X.
        # For solar_total=199, this is M = 4q + 1.
        residue = (-9 * solar_total) % 4
        row = np.zeros(size)
        row[dd:2*dd] = 1
        row[q_var] = -4
        rows.append(row)
        b_lb.append(residue)
        b_ub.append(residue)

        for color in range(4):
            row = np.zeros(size)

            for t in range(4):
                row[solar_vars[t]] = solar_color_matrix[color, t]

            row[dd + np.array(color_indices[color], dtype=int)] = 1
            row[X_var] = -1

            rows.append(row)
            b_lb.append(0)
            b_ub.append(0)

    ## Coverage matrices

    E_sub = np.zeros((dd, dd))
    E_med = np.zeros((dd, dd))
    C5 = np.zeros((dd, dd))

    for i in range(dd):
        E_sub[i, util.block_indices(i, 8, 9, grid)] = 1
        E_med[i, util.block_indices(i, 3, 3, grid)] = 1
        C5[i, util.block_indices(i, 0, 4, grid)] = 1

    M_sub = np.clip(C5 @ E_sub.T, 0, 1)
    M_med = np.clip(C5 @ E_med.T, 0, 1)

    ## Floor coverage constraints

    for k in range(dd):
        row = np.zeros(size)
        row[np.flatnonzero(M_sub[k])] = 1
        row[dd + np.flatnonzero(M_med[k])] = 1
        rows.append(row)
        b_lb.append(1)
        b_ub.append(np.inf)

    ## Optional central 4x4 full tile coverage

    center_lo = grid // 2 - 2
    center_hi = grid // 2 + 2
    central_tiles = [i * grid + j for i in range(center_lo, center_hi) for j in range(center_lo, center_hi)]

    if use_central_4x4_restriction:
        for k in central_tiles:
            row = np.zeros(size)
            row[np.flatnonzero(E_sub[:, k])] = 1
            row[dd + np.flatnonzero(E_med[:, k])] = 1
            rows.append(row)
            b_lb.append(1)
            b_ub.append(np.inf)

    ## Central 4x4 must be physically unblocked by substations and medium poles

    row = np.zeros(size)

    for tile in central_tiles:
        for s in range(dd):
            if tile in util.block_indices(s, 0, 1, grid):
                row[s] = 1
        row[dd + tile] = 1

    rows.append(row)
    b_lb.append(-np.inf)
    b_ub.append(0)

    ## Connectivity matrices with mixed offsets

    N_sub_sub = np.zeros((dd, dd))
    N_sub_med = np.zeros((dd, dd))
    N_med_sub = np.zeros((dd, dd))
    N_med_med = np.zeros((dd, dd))

    for i in range(dd):
        N_sub_sub[i, util.circle_indices_mixed_offsets(i, 18, grid, source_offset=sub_offset, target_offset=sub_offset, consider_wrapping=False)] = 1
        N_sub_med[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=sub_offset, target_offset=med_offset, consider_wrapping=False)] = 1
        N_med_sub[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=med_offset, target_offset=sub_offset, consider_wrapping=False)] = 1
        N_med_med[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=med_offset, target_offset=med_offset, consider_wrapping=False)] = 1

    ## Fixed-root lower-rank connectivity

    rank = np.arange(dd)
    rank[root_substation] = -1

    for j in range(dd):
        if j == root_substation:
            continue

        row = np.zeros(size)
        row[j] = 1

        sub_parents = np.flatnonzero(N_sub_sub[:, j])
        sub_parents = sub_parents[rank[sub_parents] < rank[j]]

        med_parents = np.flatnonzero(N_med_sub[:, j])
        med_parents = med_parents[rank[med_parents] < rank[j]]

        row[sub_parents] -= 1
        row[dd + med_parents] -= 1

        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(0)

    for j in range(dd):
        row = np.zeros(size)
        row[dd + j] = 1

        sub_parents = np.flatnonzero(N_sub_med[:, j])
        sub_parents = sub_parents[rank[sub_parents] < rank[j]]

        med_parents = np.flatnonzero(N_med_med[:, j])
        med_parents = med_parents[rank[med_parents] < rank[j]]

        row[sub_parents] -= 1
        row[dd + med_parents] -= 1

        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(0)

    A_total = np.vstack(rows)
    b_lb = np.array(b_lb)
    b_ub = np.array(b_ub)

    n = size

    c = np.zeros(n)
    c[0:dd] = 1
    c[dd:2*dd] = 0.25 * medium_weight

    lb = np.zeros(n)
    ub = np.ones(n)

    if balance_medium_colors:
        ub[a_var] = solar_total
        ub[b_var] = solar_total
        ub[c_var] = solar_total
        ub[d_var] = solar_total
        ub[X_var] = 9 * solar_total + dd
        ub[q_var] = dd

    integrality = np.ones(n)

    return A_total, b_lb, b_ub, n, c, lb, ub, integrality

def construct_network_floor_coverage_system_tileable_fixed_medium_corners_color_obstacle(
    grid,
    max_medium_poles=np.inf,
    medium_weight=1.0,
    balance_medium_colors=True,
    solar_total=199,
    use_medium_pole_obstacle_spacing=True,
    medium_pole_obstacle_radius=4,
    fixed_medium_poles=None,
    root_medium_pole=None,
):
    """
    Simpler Stage-A network planner with fixed tileable corner medium poles and no root-selector/prefix variables.

    Variables:
        0:dd             substations
        dd:2*dd          medium electric poles
        2*dd:2*dd+6      optional color witness variables a,b,c,d,X,q

    Objective:
        minimize Ns + 0.25 * medium_weight * Nm

    Constraints:
        - fixed tileability medium poles; defaults to the three corner poles
          at (4,4), (4,grid-5), and (grid-5,4)
        - every possible 5x5 fictitious building placement must be covered
        - central 4x4 must have full electric coverage
        - central 4x4 must be physically unblocked by substations / medium poles
        - fixed-root lower-rank connectivity using a selected fixed medium
          pole as root
        - optional medium pole count limit
        - optional solar-panel + medium-pole color completion with exactly solar_total solar panels
        - optional packing-style obstacle non-overlap:
            substation real 2x2 footprint vs. medium pole virtual centered 9x9 footprint when radius=4

    Tileability:
        The default three fixed corner medium poles provide exact
        medium-medium bridges across both wrapped axes. Callers can supply
        another fixed pattern and root with the optional arguments.
    """

    d = grid
    dd = d ** 2

    sub_offset = (1.0, 1.0)
    med_offset = (0.5, 0.5)

    base_size = 2 * dd
    size = base_size

    if balance_medium_colors:
        color_offset = size
        a_var = color_offset + 0
        b_var = color_offset + 1
        c_var = color_offset + 2
        d_var = color_offset + 3
        X_var = color_offset + 4
        q_var = color_offset + 5
        size += 6

    rows = []
    b_lb = []
    b_ub = []

    ## Fixed tileability medium poles
    ##
    ## Use pole roots close to the wrapped borders. For grid=50 these are
    ## (4,4), (4,45), and (45,4). Across a wrapped seam, the medium-medium
    ## center distance is 4 + (grid - (grid - 5)) = 9, so these fixed poles
    ## provide exact tileability bridges across both axes while their centered
    ## 9x9 virtual obstacle footprints do not overlap across the seam.

    if fixed_medium_poles is None:
        pole_lo = 4
        pole_hi = grid - 5
        fixed_medium_poles = [
            pole_lo * grid + pole_lo,
            pole_lo * grid + pole_hi,
            pole_hi * grid + pole_lo,
        ]
    else:
        fixed_medium_poles = [
            int(index) for index in fixed_medium_poles
        ]

    if (
        not fixed_medium_poles
        or len(set(fixed_medium_poles)) != len(fixed_medium_poles)
        or any(index < 0 or index >= dd for index in fixed_medium_poles)
    ):
        raise ValueError("Fixed medium-pole indices must be unique grid tiles.")

    if root_medium_pole is None:
        root_medium_pole = fixed_medium_poles[0]
    root_medium_pole = int(root_medium_pole)
    if root_medium_pole not in fixed_medium_poles:
        raise ValueError("The connectivity root must be a fixed medium pole.")

    row = np.zeros(size)
    for idx in fixed_medium_poles:
        row[dd + idx] = 1
    rows.append(row)
    b_lb.append(len(fixed_medium_poles))
    b_ub.append(len(fixed_medium_poles))

    ## Medium pole count limiter

    if np.isfinite(max_medium_poles):
        row = np.zeros(size)
        row[dd:2*dd] = 1
        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(max_medium_poles)

    ## Packing-style obstacle spacing
    ## Same construction style as the packing objective: build an occupancy matrix and append its transpose.

    if use_medium_pole_obstacle_spacing:
        R = medium_pole_obstacle_radius
        obstacle_A = np.zeros((size, dd))

        for k in range(dd):
            obstacle_A[k, util.block_indices(k, 0, 1, grid)] = 1
            obstacle_A[dd + k, util.block_indices(k, R, R, grid)] = 1

        rows.extend(obstacle_A.T)
        b_lb.extend(-np.inf * np.ones(dd))
        b_ub.extend(np.ones(dd))

    ## Solar + medium-pole color completion feasibility

    if balance_medium_colors:
        color_indices = [[] for _ in range(4)]

        for k in range(dd):
            i = k // grid
            j = k % grid
            color = 2 * (i % 2) + (j % 2)
            color_indices[color].append(k)

        solar_color_matrix = np.array([
            [4, 2, 2, 1],
            [2, 4, 1, 2],
            [2, 1, 4, 2],
            [1, 2, 2, 4],
        ])

        solar_vars = [a_var, b_var, c_var, d_var]

        # Exactly solar_total solar panels total across the four color configurations.
        row = np.zeros(size)
        row[solar_vars] = 1
        rows.append(row)
        b_lb.append(solar_total)
        b_ub.append(solar_total)

        # Medium total congruence: 9*solar_total + M = 4X.
        # For solar_total=199, this is M = 4q + 1.
        residue = (-9 * solar_total) % 4
        row = np.zeros(size)
        row[dd:2*dd] = 1
        row[q_var] = -4
        rows.append(row)
        b_lb.append(residue)
        b_ub.append(residue)

        for color in range(4):
            row = np.zeros(size)

            for t in range(4):
                row[solar_vars[t]] = solar_color_matrix[color, t]

            row[dd + np.array(color_indices[color], dtype=int)] = 1
            row[X_var] = -1

            rows.append(row)
            b_lb.append(0)
            b_ub.append(0)

    ## Coverage matrices

    E_sub = np.zeros((dd, dd))
    E_med = np.zeros((dd, dd))
    C5 = np.zeros((dd, dd))

    for i in range(dd):
        E_sub[i, util.block_indices(i, 8, 9, grid)] = 1
        E_med[i, util.block_indices(i, 3, 3, grid)] = 1
        C5[i, util.block_indices(i, 0, 4, grid)] = 1

    M_sub = np.clip(C5 @ E_sub.T, 0, 1)
    M_med = np.clip(C5 @ E_med.T, 0, 1)

    ## Floor coverage constraints

    for k in range(dd):
        row = np.zeros(size)
        row[np.flatnonzero(M_sub[k])] = 1
        row[dd + np.flatnonzero(M_med[k])] = 1
        rows.append(row)
        b_lb.append(1)
        b_ub.append(np.inf)

    ## Central 4x4 full tile coverage

    center_lo = grid // 2 - 2
    center_hi = grid // 2 + 2
    central_tiles = [i * grid + j for i in range(center_lo, center_hi) for j in range(center_lo, center_hi)]

    for k in central_tiles:
        row = np.zeros(size)
        row[np.flatnonzero(E_sub[:, k])] = 1
        row[dd + np.flatnonzero(E_med[:, k])] = 1
        rows.append(row)
        b_lb.append(1)
        b_ub.append(np.inf)

    ## Central 4x4 must be physically unblocked by substations and medium poles

    row = np.zeros(size)

    for tile in central_tiles:
        for s in range(dd):
            if tile in util.block_indices(s, 0, 1, grid):
                row[s] = 1
        row[dd + tile] = 1

    rows.append(row)
    b_lb.append(-np.inf)
    b_ub.append(0)

    ## Connectivity matrices with mixed offsets

    N_sub_sub = np.zeros((dd, dd))
    N_sub_med = np.zeros((dd, dd))
    N_med_sub = np.zeros((dd, dd))
    N_med_med = np.zeros((dd, dd))

    for i in range(dd):
        N_sub_sub[i, util.circle_indices_mixed_offsets(i, 18, grid, source_offset=sub_offset, target_offset=sub_offset, consider_wrapping=False)] = 1
        N_sub_med[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=sub_offset, target_offset=med_offset, consider_wrapping=False)] = 1
        N_med_sub[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=med_offset, target_offset=sub_offset, consider_wrapping=False)] = 1
        N_med_med[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=med_offset, target_offset=med_offset, consider_wrapping=False)] = 1

    ## Fixed-root lower-rank connectivity

    rank = np.arange(dd)
    rank[root_medium_pole] = -1

    for j in range(dd):
        row = np.zeros(size)
        row[j] = 1

        sub_parents = np.flatnonzero(N_sub_sub[:, j])
        sub_parents = sub_parents[rank[sub_parents] < rank[j]]

        med_parents = np.flatnonzero(N_med_sub[:, j])
        med_parents = med_parents[rank[med_parents] < rank[j]]

        row[sub_parents] -= 1
        row[dd + med_parents] -= 1

        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(0)

    for j in range(dd):
        if j == root_medium_pole:
            continue

        row = np.zeros(size)
        row[dd + j] = 1

        sub_parents = np.flatnonzero(N_sub_med[:, j])
        sub_parents = sub_parents[rank[sub_parents] < rank[j]]

        med_parents = np.flatnonzero(N_med_med[:, j])
        med_parents = med_parents[rank[med_parents] < rank[j]]

        row[sub_parents] -= 1
        row[dd + med_parents] -= 1

        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(0)

    A_total = np.vstack(rows)
    b_lb = np.array(b_lb)
    b_ub = np.array(b_ub)

    n = size

    c = np.zeros(n)
    c[0:dd] = 1
    c[dd:2*dd] = 0.25 * medium_weight

    lb = np.zeros(n)
    ub = np.ones(n)

    if balance_medium_colors:
        ub[a_var] = solar_total
        ub[b_var] = solar_total
        ub[c_var] = solar_total
        ub[d_var] = solar_total
        ub[X_var] = 9 * solar_total + dd
        ub[q_var] = dd

    integrality = np.ones(n)

    return A_total, b_lb, b_ub, n, c, lb, ub, integrality

def construct_network_floor_coverage_system_tileable_medium_corners_5_10(
    grid,
):
    """Exact 5+10 network with three fixed tileability-corner mediums.

    This is deliberately separate from
    ``construct_network_floor_coverage_system_tileable_free_root_color_full``.
    It keeps all five substations free, fixes medium poles at (4, 4),
    (4, grid-5), and (grid-5, 4), and uses the first fixed medium pole as the
    lower-rank connectivity root.
    """
    result = (
        construct_network_floor_coverage_system_tileable_fixed_medium_corners_color_obstacle(
            grid,
            max_medium_poles=10,
            medium_weight=1.0,
            balance_medium_colors=True,
            solar_total=198,
            use_medium_pole_obstacle_spacing=False,
        )
    )
    A, b_lb, b_ub, n, c, lb, ub, integrality = result

    exact_counts = np.zeros((2, n))
    dd = grid**2
    exact_counts[0, :dd] = 1
    exact_counts[1, dd:2*dd] = 1

    A = np.vstack((A, exact_counts))
    b_lb = np.concatenate((b_lb, (5.0, 10.0)))
    b_ub = np.concatenate((b_ub, (5.0, 10.0)))
    return A, b_lb, b_ub, n, c, lb, ub, integrality


def construct_network_floor_coverage_system_tileable_medium_side_cross_5_10(
    grid,
):
    """Exact 5+10 network with four fixed side-center medium poles.

    The poles form an axis-aligned cross.  Each pole root is four tiles in
    from its border, matching the seam geometry of the corner constructor.
    Opposite poles share their other coordinate, so their wrapped
    medium-medium center distance is exactly nine:

        (4, center), (grid-5, center),
        (center, 4), (center, grid-5)

    The top pole is the fixed lower-rank connectivity root.  The established
    three-corner constructor and its defaults remain unchanged.
    """
    pole_lo = 4
    pole_hi = grid - 5
    center = grid // 2
    cross_medium_poles = [
        pole_lo * grid + center,
        pole_hi * grid + center,
        center * grid + pole_lo,
        center * grid + pole_hi,
    ]

    result = (
        construct_network_floor_coverage_system_tileable_fixed_medium_corners_color_obstacle(
            grid,
            max_medium_poles=10,
            medium_weight=1.0,
            balance_medium_colors=True,
            solar_total=198,
            use_medium_pole_obstacle_spacing=False,
            fixed_medium_poles=cross_medium_poles,
            root_medium_pole=cross_medium_poles[0],
        )
    )
    A, b_lb, b_ub, n, c, lb, ub, integrality = result

    exact_counts = np.zeros((2, n))
    dd = grid**2
    exact_counts[0, :dd] = 1
    exact_counts[1, dd:2*dd] = 1

    A = np.vstack((A, exact_counts))
    b_lb = np.concatenate((b_lb, (5.0, 10.0)))
    b_ub = np.concatenate((b_ub, (5.0, 10.0)))
    return A, b_lb, b_ub, n, c, lb, ub, integrality


def construct_network_floor_coverage_system_tileable_simple(
    grid,
    max_medium_poles=np.inf,
    medium_weight=1.0,
):
    """
    Simple Stage-A network planner.

    This is a thin wrapper around the latest free-root network model with all
    optional experimental restrictions disabled. Use this for a clean baseline.
    """

    return construct_network_floor_coverage_system_tileable_free_root_color_full(
        grid,
        max_medium_poles=max_medium_poles,
        medium_weight=medium_weight,
        use_corner_substations=True,
        use_side_substations=False,
        use_t_substations=False,
        balance_medium_colors=False,
        medium_color_pattern=None,
        use_medium_pole_obstacle_spacing=False,
        medium_pole_obstacle_radius=4,
        use_substation_bridge_tileability=False,
        substation_bridge_step=1,
    )

def construct_network_floor_coverage_system_tileable_free_root_color_full_100(
    grid,
    max_medium_poles=np.inf,
    medium_weight=1.0,
    use_corner_substations=False,
    use_side_substations=False,
    use_t_substations=False,
    balance_medium_colors=True,
    medium_color_pattern=None,
    color_witness_bound=None,
    use_medium_pole_obstacle_spacing=False,
    medium_pole_obstacle_radius=4,
    use_substation_bridge_tileability=False,
    substation_bridge_step=1,
):
    """
    Stage-A network planner.

    Variables:
        0:dd             substations
        dd:2*dd          medium electric poles
        2*dd:3*dd        root selector variables

    Objective:
        minimize Ns + medium_weight * Nm

    Constraints:
        - exactly one selected substation is chosen as root
        - root is the lowest-index selected substation
        - every possible 5x5 fictitious building placement must be covered
        - central 4x4 must have full electric coverage
        - central 4x4 must be physically unblocked by substations / medium poles
        - lower-rank connectivity
        - number of medium poles <= max_medium_poles
        - optional fixed three-corner substations
        - optional solar-panel + medium-pole color-completion feasibility with free solar count
        - optional exact medium-pole color pattern, e.g. (0, 1, 1, 3)
        - optional packing-style non-overlap between substation 2x2 footprints and inflated medium-pole footprints
        - optional exact tileability via selectable substation-substation bridge pairs

    Factorio wire-center offsets:
        substation:           (1.0, 1.0)
        medium-electric-pole: (0.5, 0.5)
    """

    d = grid
    dd = d**2

    sub_offset = (1.0, 1.0)
    med_offset = (0.5, 0.5)

    base_size = 3 * dd
    size = base_size

    if balance_medium_colors:
        color_offset = size

        a_var = color_offset + 0
        b_var = color_offset + 1
        c_var = color_offset + 2
        d_var = color_offset + 3
        X_var = color_offset + 4

        size += 5


    if use_substation_bridge_tileability:
        bridge_coords = list(range(0, grid, substation_bridge_step))
        bridge_pair_count_per_axis = len(bridge_coords)
        bridge_selector_offset = size
        size += 2 * bridge_pair_count_per_axis

    rows = []
    b_lb = []
    b_ub = []

    def node_var(kind, idx):
        if kind == "sub":
            return idx
        if kind == "med":
            return dd + idx
        raise ValueError("Unknown node kind")

    ## Exactly one root, and root_i <= sub_i

    row = np.zeros(size)
    row[2*dd:3*dd] = 1
    rows.append(row)
    b_lb.append(1)
    b_ub.append(1)

    for i in range(dd):
        row = np.zeros(size)
        row[2*dd + i] = 1
        row[i] = -1
        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(0)

    ## If root_i = 1, no lower-index substation may be selected.

    for i in range(1, dd):
        row = np.zeros(size)
        row[:i] = 1
        row[2*dd + i] = i
        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(i)

    ## Optional fixed three-corner substations

    if use_corner_substations:
        corner_lo = 8
        corner_hi = grid - 10

        corner_substations = [
            corner_lo * grid + corner_lo,
            corner_lo * grid + corner_hi,
            corner_hi * grid + corner_lo,
        ]

        row = np.zeros(size)
        for idx in corner_substations:
            row[idx] = 1

        rows.append(row)
        b_lb.append(3)
        b_ub.append(3)

    ## Optional 4-sides substations

    if use_side_substations:
        side_offset = 8
        side_mid = grid // 2 - 1

        side_substations = [
            side_offset * grid + side_mid,                 # top side
            (grid - 10) * grid + side_mid,                 # bottom side
            side_mid * grid + side_offset,                 # left side
            side_mid * grid + (grid - 10),                 # right side
        ]

        row = np.zeros(size)
        for idx in side_substations:
            row[idx] = 1

        rows.append(row)
        b_lb.append(4)
        b_ub.append(4)

    if use_t_substations:
        side_lo = 8
        side_hi = grid - 10
        side_mid = grid // 2 - 1

        t_substations = [
            side_lo * grid + side_lo,    # 1: top-left
            side_lo * grid + side_mid,   # 2: top-middle
            side_lo * grid + side_hi,    # 3: top-right
            side_hi * grid + side_mid,   # 4: bottom-middle, tileable stem
        ]

        row = np.zeros(size)
        for idx in t_substations:
            row[idx] = 1

        rows.append(row)
        b_lb.append(4)
        b_ub.append(4)

    ## Optional exact tileability via selectable substation-substation bridge pairs
    ##
    ## This is exact, but much smaller than all possible bridge pairs. It forces
    ## one selected substation-substation bridge across each wrapped axis. The
    ## bridge location can move along the seam. For grid=50, low=8 and high=40
    ## give wrapped axis distance 18, exactly the substation wire range. With
    ## substation_bridge_step=1 this adds 2*grid binary selectors and 4*grid + 2
    ## constraints, i.e. 100 variables for a 50x50 tile.

    if use_substation_bridge_tileability:
        bridge_low = 8
        bridge_high = grid - 10
        wrapped_gap = bridge_low + (grid - bridge_high)

        if wrapped_gap > 18:
            raise ValueError("Substation bridge anchors are too far apart for range 18.")

        for axis in [0, 1]:
            axis_offset = bridge_selector_offset + axis * bridge_pair_count_per_axis

            # At least one exact substation-substation bridge pair on this axis.
            row = np.zeros(size)
            row[axis_offset:axis_offset + bridge_pair_count_per_axis] = 1
            rows.append(row)
            b_lb.append(1)
            b_ub.append(np.inf)

            for p, coord in enumerate(bridge_coords):
                selector = axis_offset + p

                if axis == 0:
                    idx_a = bridge_low * grid + coord
                    idx_b = bridge_high * grid + coord
                else:
                    idx_a = coord * grid + bridge_low
                    idx_b = coord * grid + bridge_high

                # selector <= sub_idx_a
                row = np.zeros(size)
                row[selector] = 1
                row[idx_a] = -1
                rows.append(row)
                b_lb.append(-np.inf)
                b_ub.append(0)

                # selector <= sub_idx_b
                row = np.zeros(size)
                row[selector] = 1
                row[idx_b] = -1
                rows.append(row)
                b_lb.append(-np.inf)
                b_ub.append(0)

    ## Medium pole count limiter

    if np.isfinite(max_medium_poles):
        row = np.zeros(size)
        row[dd:2*dd] = 1
        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(max_medium_poles)

    ## Optional exact medium-pole color pattern
    ##
    ## medium_color_pattern = (m0, m1, m2, m3) forces the selected medium
    ## poles to have exactly that checkerboard color distribution, where
    ## color = 2 * (i % 2) + (j % 2).
    ##
    ## This can be used to test one of the feasible M=5 or M=9 patterns
    ## directly instead of letting the color-completion equations choose one.

    if medium_color_pattern is not None:
        medium_color_pattern = tuple(int(v) for v in medium_color_pattern)

        if len(medium_color_pattern) != 4:
            raise ValueError("medium_color_pattern must be a length-4 tuple/list: (m0, m1, m2, m3).")

        if any(v < 0 for v in medium_color_pattern):
            raise ValueError("medium_color_pattern entries must be nonnegative integers.")

        for color, target_count in enumerate(medium_color_pattern):
            row = np.zeros(size)

            for k in range(dd):
                i = k // grid
                j = k % grid

                if 2 * (i % 2) + (j % 2) == color:
                    row[dd + k] = 1

            rows.append(row)
            b_lb.append(target_count)
            b_ub.append(target_count)

    ## Optional packing-style obstacle spacing
    ##
    ## Build a placement/colocation matrix exactly like the packing objective,
    ## but only for network obstacles:
    ##     - substations occupy their real 2x2 footprint
    ##     - medium poles occupy a virtual centered (2R+1)x(2R+1) footprint
    ##
    ## Then each tile may be claimed by at most one such obstacle footprint.
    ## This prevents inflated medium poles from overlapping each other or
    ## overlapping substation footprints, without pairwise constraints.

    if use_medium_pole_obstacle_spacing:
        R = medium_pole_obstacle_radius
        obstacle_A = np.zeros((size, dd))

        for k in range(dd):
            obstacle_A[k, util.block_indices(k, 0, 1, grid)] = 1
            obstacle_A[dd + k, util.block_indices(k, R, R, grid)] = 1

        rows.extend(obstacle_A.T)
        b_lb.extend(-np.inf * np.ones(dd))
        b_ub.extend(np.ones(dd))


    ## Solar + medium-pole color completion feasibility
    ##
    ## There must exist nonnegative integers a,b,c,d,X such that:
    ##
    ##   4a + 2b + 2c + 1d + m_0 = X
    ##   2a + 4b + 1c + 2d + m_1 = X
    ##   2a + 1b + 4c + 2d + m_2 = X
    ##   1a + 2b + 2c + 4d + m_3 = X
    ##
    ## where m_q is the number of selected medium poles rooted on color q.
    ##
    ## Substations are ignored because they cover one tile of each color.
    ##
    ## There is intentionally no fixed value for a+b+c+d here. This is the
    ## general color-completion test for larger grids, where the required
    ## number of solar panels is not known in advance.

    if balance_medium_colors:
        color_indices = [[] for _ in range(4)]

        for k in range(dd):
            i = k // grid
            j = k % grid
            color = 2 * (i % 2) + (j % 2)
            color_indices[color].append(k)

        solar_color_matrix = np.array([
            [4, 2, 2, 1],
            [2, 4, 1, 2],
            [2, 1, 4, 2],
            [1, 2, 2, 4],
        ])

        solar_vars = [a_var, b_var, c_var, d_var]

        for color in range(4):
            row = np.zeros(size)

            # Solar-panel contribution for this color.
            for t in range(4):
                row[solar_vars[t]] = solar_color_matrix[color, t]

            # Medium-pole contribution for this color.
            row[dd + np.array(color_indices[color], dtype=int)] = 1

            # Equal target X for all colors.
            row[X_var] = -1

            rows.append(row)
            b_lb.append(0)
            b_ub.append(0)

    ## Coverage matrices

    E_sub = np.zeros((dd, dd))
    E_med = np.zeros((dd, dd))
    C5 = np.zeros((dd, dd))

    for i in range(dd):

        block_indices = util.block_indices(i, 8, 9, grid)
        E_sub[i, block_indices] = 1

        block_indices = util.block_indices(i, 3, 3, grid)
        E_med[i, block_indices] = 1

        block_indices = util.block_indices(i, 0, 4, grid)
        C5[i, block_indices] = 1

    M_sub = np.clip(C5 @ E_sub.T, 0, 1)
    M_med = np.clip(C5 @ E_med.T, 0, 1)

    ## Floor coverage constraints

    for k in range(dd):
        row = np.zeros(size)

        sub_cover = np.flatnonzero(M_sub[k])
        med_cover = np.flatnonzero(M_med[k])

        row[sub_cover] = 1
        row[dd + med_cover] = 1

        rows.append(row)
        b_lb.append(1)
        b_ub.append(np.inf)

    ## Central 4x4 full tile coverage

    center_lo = grid // 2 - 2
    center_hi = grid // 2 + 2

    central_tiles = [
        i * grid + j
        for i in range(center_lo, center_hi)
        for j in range(center_lo, center_hi)
    ]

    for k in central_tiles:
        row = np.zeros(size)

        sub_cover = np.flatnonzero(E_sub[:, k])
        med_cover = np.flatnonzero(E_med[:, k])

        row[sub_cover] = 1
        row[dd + med_cover] = 1

        rows.append(row)
        b_lb.append(1)
        b_ub.append(np.inf)

    ## Central 4x4 must be physically unblocked by substations and medium poles

    row = np.zeros(size)

    for tile in central_tiles:
        for s in range(dd):
            if tile in util.block_indices(s, 0, 1, grid):
                row[s] = 1

        row[dd + tile] = 1

    rows.append(row)
    b_lb.append(-np.inf)
    b_ub.append(0)

    ## Connectivity matrices with mixed offsets

    N_sub_sub = np.zeros((dd, dd))
    N_sub_med = np.zeros((dd, dd))
    N_med_sub = np.zeros((dd, dd))
    N_med_med = np.zeros((dd, dd))

    for i in range(dd):

        N_sub_sub[i, util.circle_indices_mixed_offsets(i, 18, grid, source_offset=sub_offset, target_offset=sub_offset, consider_wrapping=False)] = 1
        N_sub_med[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=sub_offset, target_offset=med_offset, consider_wrapping=False)] = 1
        N_med_sub[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=med_offset, target_offset=sub_offset, consider_wrapping=False)] = 1
        N_med_med[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=med_offset, target_offset=med_offset, consider_wrapping=False)] = 1

    ## Connectivity constraints

    for j in range(dd):
        row = np.zeros(size)
        row[j] = 1
        row[2*dd + j] = -1

        sub_parents = np.flatnonzero(N_sub_sub[:, j])
        sub_parents = sub_parents[sub_parents < j]

        med_parents = np.flatnonzero(N_med_sub[:, j])
        med_parents = med_parents[med_parents < j]

        row[sub_parents] -= 1
        row[dd + med_parents] -= 1

        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(0)

    for j in range(dd):
        row = np.zeros(size)
        row[dd + j] = 1

        sub_parents = np.flatnonzero(N_sub_med[:, j])
        sub_parents = sub_parents[sub_parents < j]

        med_parents = np.flatnonzero(N_med_med[:, j])
        med_parents = med_parents[med_parents < j]

        row[sub_parents] -= 1
        row[dd + med_parents] -= 1

        rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(0)

    A_total = np.vstack(rows)
    b_lb = np.array(b_lb)
    b_ub = np.array(b_ub)

    n = size

    c = np.zeros(n)
    c[0:dd] = 1
    c[dd:2*dd] = 0.25 * medium_weight

    lb = np.zeros(n)
    ub = np.ones(n)

    # Existing network variables are binary.
    ub[:3*dd] = 1

    # Color-completion witness variables are nonnegative integers.
    # No fixed solar-panel count is imposed. The bound is deliberately loose;
    # tighten color_witness_bound if you know a better upper bound for a,b,c,d.
    if balance_medium_colors:
        if color_witness_bound is None:
            color_witness_bound = dd

        ub[a_var] = color_witness_bound
        ub[b_var] = color_witness_bound
        ub[c_var] = color_witness_bound
        ub[d_var] = color_witness_bound
        ub[X_var] = 4 * color_witness_bound + dd

    integrality = np.ones(n)

    return A_total, b_lb, b_ub, n, c, lb, ub, integrality
