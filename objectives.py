import numpy as np
import parameters as parameters
from support import utilities as util
import matplotlib.pyplot as plt
import scipy.sparse as sp

def construct_max_building_problem(grid, n_solar):

    """
    Fills the area with as many accumulators as possible while keeping a number of solar panels fixed

    Variables:
        0:dd             solar
        dd:2*dd          accumulators

    Objective:
        minimize N_acc

    Constraints:
        - Solar panel number is fixed
    """


    ## COLOCATION MATRIX

    d = grid
    dd = d**2
    size = 2*dd

    A = np.zeros((size, dd))

    # Solar panels

    for i in range(dd):
        block_indices = util.block_indices(i, 0, 2, grid)
        A[i, block_indices] = 1

    # Accumulators

    for i in range(dd):
        block_indices = util.block_indices(i, 0, 1, grid)
        A[dd+i, block_indices] = 1

    b_ub = np.ones(dd)
    b_lb = -np.inf*np.ones(dd)

    # Area Clearing for Roboports

    if d == 100:
        block_indices = util.block_indices(2323, 0, 3, grid)
        b_ub[block_indices] = 0
        block_indices = util.block_indices(7373, 0, 3, grid)
        b_ub[block_indices] = 0
    else:
        block_indices = util.block_indices(1173, 0, 3, grid)
        b_ub[block_indices] = 0


    ## NUMBER LIMIT CONSTRAINTS

    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[0:1*dd, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [n_solar]))
    b_lb = np.concatenate((b_lb, [n_solar]))

    ## RHS

    n = 2*dd

    c = -np.ones(n)

    lb = np.zeros(n)
    ub = np.ones(n)

    integrality = np.ones(n)

    return A.T, b_lb, b_ub, n, c, lb, ub, integrality

def construct_packing_problem(grid, n_solar, n_accumulator, n_substations, n_roboports, n_medium):

    """
    Fills the area with the selected number of buildings

    Variables:
        0:dd                solar
        dd:2*dd             accumulators
        2*dd:3*dd           substations
        3*dd:4*dd           roboports
        4*dd:5*dd           medium poles / 1x1 empty tiles

    Objective:
        Fulfill all constraints

    Constraints:
        - Number of buildings is fixed
        - Roboport placement is fixed
    """

    area = n_solar*9 + n_accumulator*4 + n_roboports*16 + n_substations*4 + n_medium

    if area != grid**2:
        print("Numbers not adding up to square area")
        return

    ## COLOCATION MATRIX

    d = grid
    dd = d**2
    size = 5*dd

    A = np.zeros((size, dd))

    # Solar panels

    for i in range(dd):
        block_indices = util.block_indices(i, 0, 2, grid)
        A[i, block_indices] = 1

    # Accumulators

    for i in range(dd):
        block_indices = util.block_indices(i, 0, 1, grid)
        A[dd+i, block_indices] = 1

    # Substations

    for i in range(dd):
        block_indices = util.block_indices(i, 0, 1, grid)
        A[2*dd+i, block_indices] = 1

    # Roboports

    for i in range(dd):
        block_indices = util.block_indices(i, 0, 3, grid)
        A[3*dd+i, block_indices] = 1

    # Medium

    for i in range(dd):
        block_indices = util.block_indices(i, 0, 0, grid)
        A[4*dd+i, block_indices] = 1

    b_ub = np.ones(dd)
    b_lb = np.ones(dd)

    ## NUMBER LIMIT CONSTRAINTS

    # Solar limit: (For testing)
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[0:1*dd, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [n_solar]))
    b_lb = np.concatenate((b_lb, [n_solar]))

    # Accumulator limit: (For testing)
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[dd:2*dd, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [n_accumulator]))
    b_lb = np.concatenate((b_lb, [n_accumulator]))

    # Substation limit:
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[2*dd:3*dd, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [n_substations]))
    b_lb = np.concatenate((b_lb, [n_substations]))

    # Roboport limit:
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[3*dd:4*dd, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [n_roboports]))
    b_lb = np.concatenate((b_lb, [n_roboports]))

    # Medium Pole limit:
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[4*dd:5*dd, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [n_medium]))
    b_lb = np.concatenate((b_lb, [n_medium]))

    # Placement Constraints for Robo
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[3*dd:4*dd, 0] = 1
    if grid == 50:
        single_column_constraint[3*dd+1173, 0] = 0
    else:
        single_column_constraint[3*dd+2323, 0] = 0
        single_column_constraint[3*dd+7373, 0] = 0
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [0]))

    ## RHS

    n = 5*dd

    c = -np.ones(n)

    lb = np.zeros(n)
    ub = np.ones(n)

    integrality = np.ones(n)

    return A.T, b_lb, b_ub, n, c, lb, ub, integrality

def construct_prefix_coverage_system(grid):

    """
    EDIT: This might be kind of broken by some refactoring elsewhere. I never got anything too great with this formulation either.
    Electric coverage of all buildings ensured via prefix variables (number of substations under in subsquare 0-i, 0-j)
    This generates considerably less non-zeros in the system compared to the matrix coverage variant, but in turn increases variable count

    Variables:
        0:dd                solar
        dd:2*dd             accumulators
        2*dd:3*dd           substations
        3*dd:4*dd           roboports
        4*dd:5*dd           medium poles / 1x1 empty tiles
        5*dd:6*dd           prefix coverage variables

    Objective:
        maximize power

    Constraints: (there are more, but these are the relevant ones to the actual solar panel setup problem)
        - Max number of substations and roboport count fixed
        - Roboport placement is fixed
        - All buildings covered by electric network
    """

    roboport_count = 2
    substation_max = 9

    if grid == 50:
        roboport_count -= 1
        substation_max = 36

    d = grid
    dd = d**2
    size = 6*dd + 1

    ## Basic Placement matrix and Bounds

    A = np.zeros((size, dd))

    # Solar panels
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 2, grid)
        A[i, block_indices] = 1

    # Accumulators
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 1, grid)
        A[dd+i, block_indices] = 1

    # Substations
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 1, grid)
        A[2*dd+i, block_indices] = 1

    # Roboports
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 3, grid)
        A[3*dd+i, block_indices] = 1

    # Empty
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 0, grid)
        A[4*dd+i, block_indices] = 1

    b_ub = np.ones(dd)
    b_lb = np.ones(dd)

    ## Roboport Count is fixed by Area size already

    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[3*dd:4*dd, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [roboport_count]))
    b_lb = np.concatenate((b_lb, [roboport_count]))

    ## Roboport Placement is also fixed

    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[3*dd:4*dd, 0] = 1
    if roboport_count == 1:
        single_column_constraint[3*dd+1173, 0] = 0
    else:
        single_column_constraint[3*dd+2323, 0] = 0
        single_column_constraint[3*dd+7373, 0] = 0
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [0]))

    ## Power Constraints

    # Power constraint 1
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[0:dd, 0] = -parameters.SOLAR_PANEL_POWER * parameters.ETA_S
    single_column_constraint[-1, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [-np.inf]))

    # Power constraint 2
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[dd:2*dd, 0] = -parameters.ACCUMULATOR_CHARGE / parameters.DAY_DURATION / parameters.C_ON
    single_column_constraint[3*dd:4*dd, 0] = -parameters.ACCUMULATOR_CHARGE / parameters.DAY_DURATION / parameters.C_ON * 4
    single_column_constraint[-1, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [-np.inf]))

    ## Coverage Constraint Block

    # Electric coverage auxiliary prefix variables
    E = np.zeros((size, dd))

    for k in range(dd):
        i, j = util.index_grid(k)

        E[5*dd + k, k] = 1

        if i > 0:
            E[5*dd + util.index_grid(i-1, j), k] = -1
            if j > 0:
                E[5*dd + util.index_grid(i-1, j-1), k] = 1

        if j > 0:
            E[5*dd + util.index_grid(i, j-1), k] = -1

        E[2*dd + k, k] = -1

    e_ub = np.zeros(dd)
    e_lb = np.zeros(dd)

    # Coverage constraints
    C = np.zeros((size, 2*dd))

    for k in range(dd):
        i, j = util.index_grid(k)

        # Solar coverage: solar_k - rect_sum <= 0
        col = k
        C[k, col] = 1

        rects = util.substation_rectangles_for_object(i, j, grid, object_size=3)

        for r1, r2, c1, c2 in rects:
            util.add_prefix_rect_terms(C,col,r1, r2, c1, c2, coef=-1,dd=dd)

        # Accumulator coverage: acc_k - rect_sum <= 0
        col = dd + k
        C[dd + k, col] = 1

        rects = util.substation_rectangles_for_object(i, j, grid, object_size=2)

        for r1, r2, c1, c2 in rects:
            util.add_prefix_rect_terms(C,col,r1, r2, c1, c2,coef=-1,dd=dd)

    c_ub = np.zeros(2*dd)
    c_lb = -np.inf * np.ones(2*dd)

    # Combine all constraints
    A = np.hstack([A, E, C])
    b_ub = np.concatenate([b_ub, e_ub, c_ub])
    b_lb = np.concatenate([b_lb, e_lb, c_lb])

    ## RHS

    n = 6*dd + 1

    c = np.zeros(n)
    c[-1] = -1

    lb = np.zeros(n)
    ub = np.ones(n)

    ub[5*dd:6*dd] = substation_max
    ub[-1] = np.inf

    integrality = np.ones(n)
    integrality[-1] = 0   

    return A.T, b_lb, b_ub, n, c, lb, ub, integrality

def construct_matrix_coverage_system(grid):

    """
    Electric coverage ensured via coverage matrix. 

    Variables:
        0:dd                solar
        dd:2*dd             accumulators
        2*dd:3*dd           substations
        3*dd:4*dd           roboports
        4*dd:5*dd           medium poles / 1x1 empty tiles

    Objective:
        maximize power

    Constraints: 
        - Max number of substations and roboport count fixed
        - Roboport placement is fixed
        - All buildings covered by electric network
    """

    roboport_count = 2

    if grid == 50:
        roboport_count -= 1

    d = grid
    dd = d**2
    size = 5*dd + 1

    ## Basic Placement matrix and Bounds

    A = np.zeros((size, dd))

    # Solar panels
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 2, grid)
        A[i, block_indices] = 1

    # Accumulators
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 1, grid)
        A[dd+i, block_indices] = 1

    # Substations
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 1, grid)
        A[2*dd+i, block_indices] = 1

    # Roboports
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 3, grid)
        A[3*dd+i, block_indices] = 1

    # Empty
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 0, grid)
        A[4*dd+i, block_indices] = 1

    b_ub = np.ones(dd)
    b_lb = np.ones(dd)

    ## Roboport Count is fixed by Area size already

    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[3*dd:4*dd, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [roboport_count]))
    b_lb = np.concatenate((b_lb, [roboport_count]))

    ## Roboport Placement is also fixed

    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[3*dd:4*dd, 0] = 1
    if roboport_count == 1:
        single_column_constraint[3*dd+1173, 0] = 0
    else:
        single_column_constraint[3*dd+2323, 0] = 0
        single_column_constraint[3*dd+7373, 0] = 0
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [0]))

    ## Power Constraints

    # Power constraint 1
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[0:dd, 0] = -parameters.SOLAR_PANEL_POWER * parameters.ETA_S
    single_column_constraint[-1, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [-np.inf]))

    # Power constraint 2
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[dd:2*dd, 0] = -parameters.ACCUMULATOR_CHARGE / parameters.DAY_DURATION / parameters.C_ON
    single_column_constraint[3*dd:4*dd, 0] = -parameters.ACCUMULATOR_CHARGE / parameters.DAY_DURATION / parameters.C_ON * 4
    single_column_constraint[-1, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [-np.inf]))

    ## Coverage Constraint Block

    E = np.zeros((dd, dd))
    Cs = np.zeros((dd, dd))
    Cc = np.zeros((dd, dd))

    # Substation Network

    for i in range(dd):

        #Substations
        block_indices = util.block_indices(i, 8, 9, grid)
        E[i, block_indices] = 1

        #Solar Panels
        block_indices = util.block_indices(i, 0, 2, grid)
        Cs[i, block_indices] = 1

        #Accumulators
        block_indices = util.block_indices(i, 0, 1, grid)
        Cc[i, block_indices] = 1

    M1 = Cs @ E.T
    M1 = np.clip(M1, 0, 1)

    M2 = Cc @ E.T
    M2 = np.clip(M2, 0, 1)

    I = np.eye(dd)
    Z = np.zeros((dd, dd))

    M = np.block([
    [I, Z, -M1, Z, Z],
    [Z, I, -M2, Z, Z],
    ])

    zero_col = np.zeros((2*dd, 1))
    M = np.hstack([M, zero_col])

    m_ub = np.zeros(2*dd)
    m_lb = -np.inf*np.ones(2*dd)

    A_total = np.vstack([A.T, M])

    b_lb = np.hstack([b_lb, m_lb])
    b_ub = np.hstack([b_ub, m_ub])

    ## RHS

    n = 5*dd + 1

    c = np.zeros(n)
    c[-1] = -1

    lb = np.zeros(n)
    ub = np.ones(n)

    ub[-1] = np.inf

    integrality = np.ones(n)
    integrality[-1] = 0   

    return A_total, b_lb, b_ub, n, c, lb, ub, integrality

def construct_restrictive_solver(
    grid,
    roboport_substitution_factor=4,
    *,
    min_substations=None,
    max_substations=None,
    min_medium_poles=None,
    max_medium_poles=None,
):
    """
    Chooses the packing and the electric network simultaneously:
        0:dd                solar panels
        dd:2*dd             accumulators
        2*dd:3*dd           substations
        3*dd:4*dd           roboports
        4*dd:5*dd           medium poles / 1x1 blockers
        5*dd                power variable z

    Objective:
        maximize z, represented as minimize -z.

    Connectivity:
        Fixed-root lower-rank connectivity with the root fixed at the
        lower-left corner substation. Connectivity uses Factorio wire-center
        offsets:
            sub -> sub: range 18, (1.0, 1.0) to (1.0, 1.0)
            sub -> med: range  9, (1.0, 1.0) to (0.5, 0.5)
            med -> sub: range  9, (0.5, 0.5) to (1.0, 1.0)
            med -> med: range  9, (0.5, 0.5) to (0.5, 0.5)

    Tileability:
        Three corner substations are fixed so tileability is ensured automatically

    Roboports:
        Fixed to the original central placement(s), with at least one tile of
        every 4x4 roboport covered by a substation or medium pole. The full
        4x4 area is not required to be covered.

    Optional network-size limits:
        min_substations <= total substations <= max_substations
        min_medium_poles <= total medium poles <= max_medium_poles

        A limit left as None is not imposed. The three fixed corner substations
        mean that every feasible solution contains at least three substations
        even when min_substations is None or less than three.
    """

    roboport_count = 2
    if grid == 50:
        roboport_count -= 1

    d = grid
    dd = d**2
    size = 5*dd + 1

    count_limits = (
        ("substations", min_substations, max_substations, 3),
        ("medium poles", min_medium_poles, max_medium_poles, 0),
    )
    for name, minimum, maximum, required_minimum in count_limits:
        for label, value in (("minimum", minimum), ("maximum", maximum)):
            if value is None:
                continue
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"The {label} number of {name} must be an integer.")
            if not 0 <= value <= dd:
                raise ValueError(
                    f"The {label} number of {name} must be between 0 and {dd}."
                )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(
                f"The minimum number of {name} cannot exceed the maximum."
            )
        if maximum is not None and maximum < required_minimum:
            raise ValueError(
                f"At least {required_minimum} {name} are fixed or required."
            )

    sub_offset = (1.0, 1.0)
    med_offset = (0.5, 0.5)

    ## Basic placement matrix and bounds

    A = np.zeros((size, dd))

    # Solar panels, 3x3.
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 2, grid)
        A[i, block_indices] = 1

    # Accumulators, 2x2.
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 1, grid)
        A[dd + i, block_indices] = 1

    # Substations, 2x2.
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 1, grid)
        A[2*dd + i, block_indices] = 1

    # Roboports, 4x4.
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 3, grid)
        A[3*dd + i, block_indices] = 1

    # Medium electric poles, 1x1.
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 0, grid)
        A[4*dd + i, block_indices] = 1

    # Each tile can be occupied by at most one building; empty tiles allowed.
    b_ub = np.ones(dd)
    b_lb = np.zeros(dd)

    ## Roboport count

    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[3*dd:4*dd, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [roboport_count]))
    b_lb = np.concatenate((b_lb, [roboport_count]))

    ## Fixed roboport placement

    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[3*dd:4*dd, 0] = 1
    if roboport_count == 1:
        roboport_roots = (1173,)
    else:
        roboport_roots = (2323, 7373)
    for root in roboport_roots:
        single_column_constraint[3*dd + root, 0] = 0
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [0]))

    ## Fixed corner substations for tileability

    corner_lo = 8
    corner_hi = grid - 10

    corner_substations = [
        corner_lo * grid + corner_lo,
        corner_lo * grid + corner_hi,
        corner_hi * grid + corner_lo,
    ]

    root_substation = corner_substations[0]

    single_column_constraint = np.zeros((size, 1))
    for idx in corner_substations:
        single_column_constraint[2*dd + idx, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [3]))
    b_lb = np.concatenate((b_lb, [3]))

    ## Optional substation and medium-pole count limits

    for start, stop, minimum, maximum in (
        (2*dd, 3*dd, min_substations, max_substations),
        (4*dd, 5*dd, min_medium_poles, max_medium_poles),
    ):
        if minimum is None and maximum is None:
            continue
        single_column_constraint = np.zeros((size, 1))
        single_column_constraint[start:stop, 0] = 1
        A = np.hstack((A, single_column_constraint))
        b_lb = np.concatenate((
            b_lb,
            [-np.inf if minimum is None else minimum],
        ))
        b_ub = np.concatenate((
            b_ub,
            [np.inf if maximum is None else maximum],
        ))

    ## Power constraints

    # z <= solar-supported power.
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[0:dd, 0] = -parameters.SOLAR_PANEL_POWER * parameters.ETA_S
    single_column_constraint[-1, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [-np.inf]))

    # z <= accumulator-supported power.
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[dd:2*dd, 0] = -parameters.ACCUMULATOR_CHARGE / parameters.DAY_DURATION / parameters.C_ON
    single_column_constraint[3*dd:4*dd, 0] = -parameters.ACCUMULATOR_CHARGE / parameters.DAY_DURATION / parameters.C_ON * roboport_substitution_factor
    single_column_constraint[-1, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [-np.inf]))

    ## Electric coverage matrices

    E_sub = np.zeros((dd, dd))
    E_med = np.zeros((dd, dd))
    Cs = np.zeros((dd, dd))
    Cc = np.zeros((dd, dd))

    for i in range(dd):
        # Substation electric coverage: 18x18, continued across cell borders.
        block_indices = util.block_indices(i, 8, 9, grid)
        E_sub[i, block_indices] = 1

        # Medium pole electric coverage: 7x7, continued across cell borders.
        block_indices = util.block_indices(i, 3, 3, grid)
        E_med[i, block_indices] = 1

        # Solar panels: 3x3.
        block_indices = util.block_indices(i, 0, 2, grid)
        Cs[i, block_indices] = 1

        # Accumulators: 2x2.
        block_indices = util.block_indices(i, 0, 1, grid)
        Cc[i, block_indices] = 1

    ## At least one tile of each fixed 4x4 roboport must have electricity.

    for root in roboport_roots:
        root_row, root_column = divmod(root, grid)
        roboport_tiles = [
            row * grid + column
            for row in range(root_row, root_row + 4)
            for column in range(root_column, root_column + 4)
        ]
        substation_providers = np.flatnonzero(
            np.any(E_sub[:, roboport_tiles] > 0.5, axis=1)
        )
        medium_providers = np.flatnonzero(
            np.any(E_med[:, roboport_tiles] > 0.5, axis=1)
        )

        single_column_constraint = np.zeros((size, 1))
        single_column_constraint[
            2*dd + substation_providers,
            0,
        ] = 1
        single_column_constraint[
            4*dd + medium_providers,
            0,
        ] = 1
        A = np.hstack((A, single_column_constraint))
        b_lb = np.concatenate((b_lb, [1]))
        b_ub = np.concatenate((b_ub, [np.inf]))

    M1 = np.clip(Cs @ E_sub.T, 0, 1)
    M3 = np.clip(Cs @ E_med.T, 0, 1)
    M2 = np.clip(Cc @ E_sub.T, 0, 1)
    M4 = np.clip(Cc @ E_med.T, 0, 1)

    ## Connectivity matrices with physical mixed offsets.  Rows are source
    ## roots and columns are target roots, so N_*[:, j] gives possible parents
    ## for target j directly; transposing reverses the half-tile mixed offset.

    N_sub_sub = np.zeros((dd, dd))
    N_sub_med = np.zeros((dd, dd))
    N_med_sub = np.zeros((dd, dd))
    N_med_med = np.zeros((dd, dd))

    for i in range(dd):
        N_sub_sub[i, util.circle_indices_mixed_offsets(i, 18, grid, source_offset=sub_offset, target_offset=sub_offset, consider_wrapping=False)] = 1
        N_sub_med[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=sub_offset, target_offset=med_offset, consider_wrapping=False)] = 1
        N_med_sub[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=med_offset, target_offset=sub_offset, consider_wrapping=False)] = 1
        N_med_med[i, util.circle_indices_mixed_offsets(i, 9, grid, source_offset=med_offset, target_offset=med_offset, consider_wrapping=False)] = 1

    ## Fixed-root lower-rank connectivity constraints

    conn_rows = []
    conn_lb = []
    conn_ub = []

    # The fixed root is first; everything else follows natural index order.
    rank = np.arange(dd)
    rank[root_substation] = -1

    # sub_j <= lower-rank connected parents, except fixed root.
    for j in range(dd):
        if j == root_substation:
            continue

        row = np.zeros(size)
        row[2*dd + j] = 1

        sub_parents = np.flatnonzero(N_sub_sub[:, j])
        sub_parents = sub_parents[rank[sub_parents] < rank[j]]

        med_parents = np.flatnonzero(N_med_sub[:, j])
        med_parents = med_parents[rank[med_parents] < rank[j]]

        row[2*dd + sub_parents] -= 1
        row[4*dd + med_parents] -= 1

        conn_rows.append(row)
        conn_lb.append(-np.inf)
        conn_ub.append(0)

    # med_j <= lower-rank connected parents.
    for j in range(dd):
        row = np.zeros(size)
        row[4*dd + j] = 1

        sub_parents = np.flatnonzero(N_sub_med[:, j])
        sub_parents = sub_parents[rank[sub_parents] < rank[j]]

        med_parents = np.flatnonzero(N_med_med[:, j])
        med_parents = med_parents[rank[med_parents] < rank[j]]

        row[2*dd + sub_parents] -= 1
        row[4*dd + med_parents] -= 1

        conn_rows.append(row)
        conn_lb.append(-np.inf)
        conn_ub.append(0)

    C_conn = np.vstack(conn_rows)
    c_conn_lb = np.array(conn_lb)
    c_conn_ub = np.array(conn_ub)

    ## Coverage constraints for powered buildings

    I = np.eye(dd)
    Z = np.zeros((dd, dd))

    M = np.block([
        [I, Z, -M1, Z, -M3],
        [Z, I, -M2, Z, -M4],
    ])

    zero_col = np.zeros((2*dd, 1))
    M = np.hstack([M, zero_col])

    m_ub = np.zeros(2*dd)
    m_lb = -np.inf * np.ones(2*dd)

    A_total = np.vstack([A.T, M, C_conn])

    b_lb = np.hstack([b_lb, m_lb, c_conn_lb])
    b_ub = np.hstack([b_ub, m_ub, c_conn_ub])

    ## Return model arrays

    n = size

    c = np.zeros(n)
    c[-1] = -1

    lb = np.zeros(n)
    ub = np.ones(n)
    ub[-1] = np.inf

    integrality = np.ones(n)
    integrality[-1] = 0

    return A_total, b_lb, b_ub, n, c, lb, ub, integrality

def construct_matrix_coverage_connectivity_ensured_variable_root_color_target(
    grid,
    roboport_substitution_factor=4,
    min_power=8316.0,
    allowed_medium_totals=(2, 5, 6, 9, 10, 14),
):
    """
    A+B variant of construct_restrictive_solver.

    It keeps the original simultaneous packing, power, electric-coverage, fixed
    roboport, and three-corner tileability formulation, but replaces its fixed
    root connectivity with a compact variable-root formulation and adds:

        - z >= min_power (8316 by default)
        - exactly one valid complete medium-pole color pattern

    Substation and medium-pole totals remain free. Roboport gap restrictions
    are intentionally omitted because the exact packing objective prices their
    consequences directly.

    Variables retain the original first 5*dd+1 positions, followed by compact
    root-selector binaries and valid-color-pattern selector binaries.
    """

    if not np.isfinite(min_power) or min_power < 0:
        raise ValueError("min_power must be a finite nonnegative number.")

    from coverage_objectives import (
        MEDIUM_COLOR_PATTERNS_2,
        MEDIUM_COLOR_PATTERNS_5,
        MEDIUM_COLOR_PATTERNS_6,
        MEDIUM_COLOR_PATTERNS_9,
        MEDIUM_COLOR_PATTERNS_10,
        MEDIUM_COLOR_PATTERNS_14,
    )

    patterns_by_total = {
        2: MEDIUM_COLOR_PATTERNS_2,
        5: MEDIUM_COLOR_PATTERNS_5,
        6: MEDIUM_COLOR_PATTERNS_6,
        9: MEDIUM_COLOR_PATTERNS_9,
        10: MEDIUM_COLOR_PATTERNS_10,
        14: MEDIUM_COLOR_PATTERNS_14,
    }

    allowed_medium_totals = tuple(int(total) for total in allowed_medium_totals)
    unsupported_totals = sorted(set(allowed_medium_totals) - set(patterns_by_total))
    if unsupported_totals:
        raise ValueError(
            "No valid color-pattern table for medium totals "
            f"{unsupported_totals}."
        )
    if not allowed_medium_totals:
        raise ValueError("allowed_medium_totals must contain at least one total.")

    color_patterns = [
        tuple(pattern)
        for total in dict.fromkeys(allowed_medium_totals)
        for pattern in patterns_by_total[total]
    ]

    original = construct_restrictive_solver(
        grid,
        roboport_substitution_factor=roboport_substitution_factor,
    )
    A_original, b_lb_original, b_ub_original, original_n, c_original, lb_original, ub_original, integrality_original = original

    dd = grid**2
    z_index = 5 * dd

    # The original tail consists of (dd-1) substation connectivity rows and
    # dd medium-pole connectivity rows. Preserve every earlier A+B row and
    # replace only this fixed-root tail.
    original_connectivity_rows = 2 * dd - 1
    retained_rows = A_original.shape[0] - original_connectivity_rows
    A_retained = sp.csr_matrix(A_original[:retained_rows])
    b_lb_retained = np.asarray(b_lb_original[:retained_rows])
    b_ub_retained = np.asarray(b_ub_original[:retained_rows])

    # Since the lowest-index fixed corner is selected, no root above it can be
    # the lowest selected substation. Restricting the selector block is exact.
    corner_lo = 8
    root_candidate_max = corner_lo * grid + corner_lo
    root_selector_count = root_candidate_max + 1
    root_selector_offset = original_n
    color_selector_offset = root_selector_offset + root_selector_count
    color_selector_count = len(color_patterns)
    new_n = color_selector_offset + color_selector_count

    A_retained = sp.hstack(
        [
            A_retained,
            sp.csr_matrix((A_retained.shape[0], new_n - original_n)),
        ],
        format="csr",
    )

    root_exact_row = 0
    root_link_offset = 1
    sub_connectivity_offset = root_link_offset + root_selector_count
    med_connectivity_offset = sub_connectivity_offset + dd
    color_exact_row = med_connectivity_offset + dd
    color_count_offset = color_exact_row + 1
    extra_row_count = color_count_offset + 4

    extra = sp.lil_matrix((extra_row_count, new_n))
    extra_lb = -np.inf * np.ones(extra_row_count)
    extra_ub = np.zeros(extra_row_count)

    # Exactly one selected root, and a root selector implies a substation.
    extra[root_exact_row, root_selector_offset:color_selector_offset] = 1
    extra_lb[root_exact_row] = 1
    extra_ub[root_exact_row] = 1

    for index in range(root_selector_count):
        row = root_link_offset + index
        extra[row, root_selector_offset + index] = 1
        extra[row, 2 * dd + index] = -1

    sub_offset = (1.0, 1.0)
    med_offset = (0.5, 0.5)

    # Lower-rank connectivity with a freely selected substation root.
    for target in range(dd):
        row = sub_connectivity_offset + target
        extra[row, 2 * dd + target] = 1
        if target <= root_candidate_max:
            extra[row, root_selector_offset + target] = -1

        sub_parents = np.asarray(
            util.circle_indices_mixed_offsets(
                target,
                18,
                grid,
                source_offset=sub_offset,
                target_offset=sub_offset,
                consider_wrapping=False,
            ),
            dtype=int,
        )
        sub_parents = sub_parents[sub_parents < target]

        med_parents = np.asarray(
            util.circle_indices_mixed_offsets(
                target,
                9,
                grid,
                source_offset=sub_offset,
                target_offset=med_offset,
                consider_wrapping=False,
            ),
            dtype=int,
        )
        med_parents = med_parents[med_parents < target]

        extra[row, 2 * dd + sub_parents] = -1
        extra[row, 4 * dd + med_parents] = -1

    for target in range(dd):
        row = med_connectivity_offset + target
        extra[row, 4 * dd + target] = 1

        sub_parents = np.asarray(
            util.circle_indices_mixed_offsets(
                target,
                9,
                grid,
                source_offset=med_offset,
                target_offset=sub_offset,
                consider_wrapping=False,
            ),
            dtype=int,
        )
        sub_parents = sub_parents[sub_parents < target]

        med_parents = np.asarray(
            util.circle_indices_mixed_offsets(
                target,
                9,
                grid,
                source_offset=med_offset,
                target_offset=med_offset,
                consider_wrapping=False,
            ),
            dtype=int,
        )
        med_parents = med_parents[med_parents < target]

        extra[row, 2 * dd + sub_parents] = -1
        extra[row, 4 * dd + med_parents] = -1

    # Pick one complete valid color pattern. This simultaneously restricts the
    # medium total to one of allowed_medium_totals while leaving the optimizer
    # free to choose which total and which pattern is best.
    extra[
        color_exact_row,
        color_selector_offset:color_selector_offset + color_selector_count,
    ] = 1
    extra_lb[color_exact_row] = 1
    extra_ub[color_exact_row] = 1

    color_indices = [[] for _ in range(4)]
    for index in range(dd):
        row_index = index // grid
        column_index = index % grid
        color_indices[2 * (row_index % 2) + (column_index % 2)].append(index)

    for color in range(4):
        row = color_count_offset + color
        extra[row, 4 * dd + np.asarray(color_indices[color], dtype=int)] = 1
        for selector, pattern in enumerate(color_patterns):
            extra[row, color_selector_offset + selector] = -pattern[color]
        extra_lb[row] = 0
        extra_ub[row] = 0

    A_total = sp.vstack([A_retained, extra.tocsr()], format="csr")
    b_lb = np.concatenate([b_lb_retained, extra_lb])
    b_ub = np.concatenate([b_ub_retained, extra_ub])

    c = np.concatenate([c_original, np.zeros(new_n - original_n)])
    lb = np.concatenate([lb_original, np.zeros(new_n - original_n)])
    ub = np.concatenate([ub_original, np.ones(new_n - original_n)])
    integrality = np.concatenate(
        [integrality_original, np.ones(new_n - original_n)]
    )
    lb[z_index] = min_power

    return A_total, b_lb, b_ub, new_n, c, lb, ub, integrality


def construct_matrix_coverage_connectivity_ensured_fixed_root_color_target(
    grid,
    roboport_substitution_factor=4,
    min_power=8316.0,
):
    """
    Compact constrained A+B variant.

    This reuses the original restrictive constructor unchanged, including its
    fixed root at the lowest-index corner substation. It adds only:

        - z fixed to min_power, making this a pure feasibility model
        - m0-m3 and m1-m2 divisible by three

    There are no root auxiliaries and no color-pattern selector binaries. The
    only added variables are two general-integer modulo witnesses. The combined
    model remains free to choose its own substation and medium-pole totals.
    """

    if not np.isfinite(min_power) or min_power < 0:
        raise ValueError("min_power must be a finite nonnegative number.")

    original = construct_restrictive_solver(
        grid,
        roboport_substitution_factor=roboport_substitution_factor,
    )
    A_original, b_lb_original, b_ub_original, original_n, _, lb_original, ub_original, integrality_original = original

    dd = grid**2
    z_index = 5 * dd
    color_mod_03_var = original_n
    color_mod_12_var = original_n + 1
    new_n = original_n + 2

    roboport_count = 1 if grid == 50 else 2
    solar_unit_power = parameters.SOLAR_PANEL_POWER * parameters.ETA_S
    accumulator_unit_power = (
        parameters.ACCUMULATOR_CHARGE
        / parameters.DAY_DURATION
        / parameters.C_ON
    )
    minimum_solar = int(np.ceil(min_power / solar_unit_power - 1e-9))
    minimum_accumulators = max(
        0,
        int(np.ceil(
            min_power / accumulator_unit_power
            - roboport_substitution_factor * roboport_count
            - 1e-9
        )),
    )
    maximum_network_area = (
        dd
        - 9 * minimum_solar
        - 4 * minimum_accumulators
        - 16 * roboport_count
    )

    A_base = sp.hstack(
        [
            sp.csr_matrix(A_original),
            sp.csr_matrix((A_original.shape[0], new_n - original_n)),
        ],
        format="csr",
    )

    # Two color equalities plus three redundant target/area inequalities. The
    # latter expose consequences of the power and packing blocks directly,
    # instead of making the LP derive them from thousands of rows.
    strengthening_rows = sp.lil_matrix((5, new_n))

    color_indices = [[] for _ in range(4)]
    for index in range(dd):
        row_index = index // grid
        column_index = index % grid
        color_indices[2 * (row_index % 2) + (column_index % 2)].append(index)

    strengthening_rows[0, 4 * dd + np.asarray(color_indices[0], dtype=int)] = 1
    strengthening_rows[0, 4 * dd + np.asarray(color_indices[3], dtype=int)] = -1
    strengthening_rows[0, color_mod_03_var] = -3

    strengthening_rows[1, 4 * dd + np.asarray(color_indices[1], dtype=int)] = 1
    strengthening_rows[1, 4 * dd + np.asarray(color_indices[2], dtype=int)] = -1
    strengthening_rows[1, color_mod_12_var] = -3

    strengthening_rows[2, 0:dd] = 1
    strengthening_rows[3, dd:2 * dd] = 1
    strengthening_rows[4, 2 * dd:3 * dd] = 4
    strengthening_rows[4, 4 * dd:5 * dd] = 1

    A_total = sp.vstack([A_base, strengthening_rows.tocsr()], format="csr")
    b_lb = np.concatenate([
        b_lb_original,
        np.array([0.0, 0.0, minimum_solar, minimum_accumulators, -np.inf]),
    ])
    b_ub = np.concatenate([
        b_ub_original,
        np.array([0.0, 0.0, np.inf, np.inf, maximum_network_area]),
    ])

    # Any layout capable of more than min_power can set z to min_power. Use a
    # light network-size objective to give the LP relaxation and branching a
    # useful direction. If the introductory solver stops at its first
    # feasible layout, this does not trigger a costly proof of minimum
    # network size after feasibility.
    c = np.zeros(new_n)
    c[2 * dd:3 * dd] = 1.0
    c[4 * dd:5 * dd] = 0.25
    lb = np.concatenate([lb_original, np.zeros(new_n - original_n)])
    ub = np.concatenate([ub_original, np.ones(new_n - original_n)])
    integrality = np.concatenate(
        [integrality_original, np.ones(new_n - original_n)]
    )

    # Tight witness bounds from the area that can remain after the minimum
    # solar panels, accumulators, fixed roboports, and three corner substations
    # needed at the requested power target. For 50x50 and target 8316 with no
    # roboport substitution, at most 18 medium poles can fit, so |q| <= 6.
    maximum_mediums_by_area = max(
        0,
        maximum_network_area
        - 4 * 3,
    )
    color_mod_bound = maximum_mediums_by_area // 3
    lb[color_mod_03_var] = -color_mod_bound
    ub[color_mod_03_var] = color_mod_bound
    lb[color_mod_12_var] = -color_mod_bound
    ub[color_mod_12_var] = color_mod_bound
    lb[z_index] = min_power
    ub[z_index] = min_power

    return A_total, b_lb, b_ub, new_n, c, lb, ub, integrality


def construct_matrix_coverage_fixed_network(
    grid,
    substation_slice,
    medium_pole_slice,
    min_power=np.inf,
    roboport_substitution_factor=4,
    periodic_electric_coverage=True,
):

    """
    Chooses the packing and the electric network simultaneously:
        0:dd                solar panels
        dd:2*dd             accumulators
        2*dd:3*dd           substations
        3*dd:4*dd           roboports
        4*dd:5*dd           medium poles / 1x1 blockers
        5*dd                power variable z

    Objective:
        maximize z, represented as minimize -z.

    Constraints:
        - Substations and medium poles are completely fixed to the values in substation_slice and medium_pole_slice
        - Roboport placement and count
        - At fixed network, the maximum power is trivial, the packing is the problem to solve, so min_power is introduced so that 
        if the bound falls instantly below that, the configuration can be discarded right away.
    """


    roboport_count = 2

    if grid == 50:
        roboport_count -= 1

    d = grid
    dd = d**2
    size = 5*dd + 1

    substation_slice = np.asarray(substation_slice).reshape(dd)
    medium_pole_slice = np.asarray(medium_pole_slice).reshape(dd)

    ## Basic Placement matrix and Bounds

    A = np.zeros((size, dd))

    # Solar panels
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 2, grid)
        A[i, block_indices] = 1

    # Accumulators
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 1, grid)
        A[dd+i, block_indices] = 1

    # Substations
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 1, grid)
        A[2*dd+i, block_indices] = 1

    # Roboports
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 3, grid)
        A[3*dd+i, block_indices] = 1

    # Medium Electric Poles
    for i in range(dd):
        block_indices = util.block_indices(i, 0, 0, grid)
        A[4*dd+i, block_indices] = 1

    # Empty spaces allowed.
    b_ub = np.ones(dd)
    b_lb = np.zeros(dd)

    ## Roboport Count

    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[3*dd:4*dd, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [roboport_count]))
    b_lb = np.concatenate((b_lb, [roboport_count]))

    ## Roboport Placement

    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[3*dd:4*dd, 0] = 1

    if roboport_count == 1:
        single_column_constraint[3*dd+1173, 0] = 0
    else:
        single_column_constraint[3*dd+2323, 0] = 0
        single_column_constraint[3*dd+7373, 0] = 0

    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [0]))

    ## Power Constraints

    # z <= solar power
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[0:dd, 0] = -parameters.SOLAR_PANEL_POWER * parameters.ETA_S
    single_column_constraint[-1, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [-np.inf]))

    # z <= accumulator-supported power
    single_column_constraint = np.zeros((size, 1))
    single_column_constraint[dd:2*dd, 0] = -parameters.ACCUMULATOR_CHARGE / parameters.DAY_DURATION / parameters.C_ON
    single_column_constraint[3*dd:4*dd, 0] = -parameters.ACCUMULATOR_CHARGE / parameters.DAY_DURATION / parameters.C_ON * roboport_substitution_factor
    single_column_constraint[-1, 0] = 1
    A = np.hstack((A, single_column_constraint))
    b_ub = np.concatenate((b_ub, [0]))
    b_lb = np.concatenate((b_lb, [-np.inf]))

    ## Coverage Constraint Block

    E_sub = np.zeros((dd, dd))
    E_med = np.zeros((dd, dd))
    Cs = np.zeros((dd, dd))
    Cc = np.zeros((dd, dd))

    for i in range(dd):

        # Substations
        block_indices = util.block_indices(
            i,
            8,
            9,
            grid,
            consider_wrapping=periodic_electric_coverage,
        )
        E_sub[i, block_indices] = 1

        # Medium Poles
        block_indices = util.block_indices(
            i,
            3,
            3,
            grid,
            consider_wrapping=periodic_electric_coverage,
        )
        E_med[i, block_indices] = 1

        # Solar Panels
        block_indices = util.block_indices(i, 0, 2, grid)
        Cs[i, block_indices] = 1

        # Accumulators
        block_indices = util.block_indices(i, 0, 1, grid)
        Cc[i, block_indices] = 1

    M1 = Cs @ E_sub.T
    M1 = np.clip(M1, 0, 1)

    M3 = Cs @ E_med.T
    M3 = np.clip(M3, 0, 1)

    M2 = Cc @ E_sub.T
    M2 = np.clip(M2, 0, 1)

    M4 = Cc @ E_med.T
    M4 = np.clip(M4, 0, 1)

    ## Final matrix building

    I = np.eye(dd)
    Z = np.zeros((dd, dd))

    M = np.block([
        [I, Z, -M1, Z, -M3],
        [Z, I, -M2, Z, -M4],
    ])

    zero_col = np.zeros((2*dd, 1))
    M = np.hstack([M, zero_col])

    m_ub = np.zeros(2*dd)
    m_lb = -np.inf*np.ones(2*dd)

    A_total = np.vstack([A.T, M])

    b_lb = np.hstack([b_lb, m_lb])
    b_ub = np.hstack([b_ub, m_ub])

    ## RHS

    n = 5*dd + 1

    c = np.zeros(n)
    c[-1] = -1

    lb = np.zeros(n)
    ub = np.ones(n)

    # Fix substations and medium poles to the supplied network.
    lb[2*dd:3*dd] = substation_slice
    ub[2*dd:3*dd] = substation_slice

    lb[4*dd:5*dd] = medium_pole_slice
    ub[4*dd:5*dd] = medium_pole_slice

    ub[-1] = np.inf
    lb[-1] = min_power
    
    integrality = np.ones(n)
    integrality[-1] = 0

    return A_total, b_lb, b_ub, n, c, lb, ub, integrality
