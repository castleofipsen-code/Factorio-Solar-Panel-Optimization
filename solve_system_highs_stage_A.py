"""Generate an electrical network for solve_system_highs_fixed_network.py."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import scipy.sparse as sp

import coverage_objectives
import parameters
from support import panel_plot


# Solver settings. Change these values directly for a different run.
# THREAD COUNT: this is the maximum number of CPU threads HiGHS may use.

grid = 50
thread_count = 20
time_limit = 300  # Seconds. Use None to run until HiGHS finishes.

# Stage-A network preferences.

maximum_substations = 10
maximum_medium_poles = 10
medium_pole_weight = 1.0


# Construct the Stage-A electrical-network model.

parameters.GRID_SIZE = grid

(
    constraint_matrix,
    constraint_lower_bounds,
    constraint_upper_bounds,
    variable_count,
    objective,
    variable_lower_bounds,
    variable_upper_bounds,
    integrality,
) = coverage_objectives.construct_network_floor_coverage_system_tileable_fixed_corners_color_obstacle(
    grid,
    max_substations=maximum_substations,
    max_medium_poles=maximum_medium_poles,
    medium_weight=medium_pole_weight,
    balance_medium_colors=True,  
    use_medium_pole_obstacle_spacing=False,
    use_central_4x4_restriction=False,
)

constraint_matrix = sp.csr_matrix(constraint_matrix)
constraints = LinearConstraint(
    constraint_matrix,
    constraint_lower_bounds,
    constraint_upper_bounds,
)
bounds = Bounds(variable_lower_bounds, variable_upper_bounds)


# Give the existing arrays directly to HiGHS through scipy.optimize.milp.

solver_options = {
    "disp": True,
    "threads": thread_count,
}
if time_limit is not None:
    solver_options["time_limit"] = time_limit

result = milp(
    c=objective,
    integrality=integrality,
    bounds=bounds,
    constraints=constraints,
    options=solver_options,
)

print(result.message)


# Save the network in the exact two-block format expected by Stage B.

if result.x is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = Path("results") / f"stage_A_highs_{timestamp}"
    result_folder.mkdir(parents=True, exist_ok=True)

    grid_area = grid**2
    network_size = 2 * grid_area
    network = np.rint(result.x[:network_size]).astype(int)

    substation_count = int(network[:grid_area].sum())
    medium_pole_count = int(network[grid_area:].sum())
    network_cost = float(objective @ result.x)

    network_path = result_folder / "best_network.sol"
    with network_path.open("w") as file:
        file.write(f"# Objective value = {network_cost:.16g}\n")
        file.write(f"# Substations = {substation_count}\n")
        file.write(f"# Medium poles = {medium_pole_count}\n")
        for index, value in enumerate(network):
            file.write(f"x[{index}] {value}\n")

    substation_coordinates = np.argwhere(
        network[:grid_area].reshape(grid, grid) == 1
    )
    medium_pole_coordinates = np.argwhere(
        network[grid_area:].reshape(grid, grid) == 1
    )
    empty_coordinates = np.empty((0, 2), dtype=int)

    panel_plot.plot_solar_array_periodic(
        grid,
        empty_coordinates,
        empty_coordinates,
        substation_coordinates,
        empty_coordinates,
        medium_pole_coordinates,
        plot_electric=True,
    )
    plt.title(
        f"Stage A network: {substation_count} substations, "
        f"{medium_pole_count} medium poles"
    )
    plt.savefig(result_folder / "best_network.png", dpi=200, bbox_inches="tight")
    plt.show()

    print("substations:", substation_count)
    print("medium poles:", medium_pole_count)
    print("Stage-B network path:", network_path.resolve())
else:
    print("HiGHS did not return a network to save.")
