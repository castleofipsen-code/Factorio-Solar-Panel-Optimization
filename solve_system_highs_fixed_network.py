from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import scipy.sparse as sp

import objectives
import parameters
from support import panel_plot, utilities

grid = 50 

# Solver settings. Change these values directly for a different run.
# THREAD COUNT: this is the maximum number of CPU threads HiGHS may use.

thread_count = 1
time_limit = 100 #seconds
minimum_power = 0
network_path = "support/sample_network.txt"
roboport_substitution_factor = 0

# Load the fixed electrical network and construct only the packing stage.

parameters.GRID_SIZE = grid
grid_area = grid**2
network = utilities.load_solution(network_path, size=2 * grid_area)

(
    constraint_matrix,
    constraint_lower_bounds,
    constraint_upper_bounds,
    variable_count,
    objective,
    variable_lower_bounds,
    variable_upper_bounds,
    integrality,
) = objectives.construct_matrix_coverage_fixed_network(
    grid,
    network[:grid_area],
    network[grid_area:2 * grid_area],
    min_power=minimum_power,
    roboport_substitution_factor=roboport_substitution_factor, #
)

constraint_matrix = sp.csr_matrix(constraint_matrix)


# Give the existing arrays directly to HiGHS through scipy.optimize.milp.

constraints = LinearConstraint(
    constraint_matrix,
    constraint_lower_bounds,
    constraint_upper_bounds,
)
bounds = Bounds(variable_lower_bounds, variable_upper_bounds)

result = milp(
    c=objective,
    integrality=integrality,
    bounds=bounds,
    constraints=constraints,
    options={
        "disp": True,
        "threads": thread_count,
        "time_limit": time_limit,
    },
)

print(result.message)


# Save and plot the best solution returned by HiGHS.

if result.x is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = Path("results") / f"fixed_network_highs_{timestamp}"
    result_folder.mkdir(parents=True, exist_ok=True)

    solution = np.asarray(result.x)
    solution[integrality == 1] = np.rint(solution[integrality == 1])

    solution_path = result_folder / "best.sol"
    with solution_path.open("w") as file:
        file.write(f"# Objective value = {result.fun:.16g}\n")
        for index, value in enumerate(solution):
            file.write(f"x[{index}] {value:.16g}\n")

    component_coordinates = utilities.state_vector_to_coordinates(
        solution.astype(int)
    )
    panel_plot.plot_solar_array_periodic(grid, *component_coordinates)

    plt.title(f"Sustained power: {solution[-1]:.6f} kW")
    plt.savefig(result_folder / "best.png", dpi=200, bbox_inches="tight")
    plt.show()

    print("power:", solution[-1])
    print("solution:", solution_path.resolve())
