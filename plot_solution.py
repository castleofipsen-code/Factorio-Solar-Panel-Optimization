import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import parameters
from support import blueprints, panel_plot, utilities


# Select the solution to plot.

grid = 50
solution_path = Path(__file__).resolve().parent / "support" / "sample_solution.sol"

# Alternatively, paste a Factorio blueprint string here. When this is set,
# solution_path is ignored.
blueprint_string = None


# Load the solution and recover the five types of buildings.

parameters.GRID_SIZE = grid

if blueprint_string:
    component_coordinates = blueprints.factorio_blueprint_to_coordinates(
        blueprint_string
    )
else:
    solution = utilities.load_solution(solution_path)
    solution = np.rint(solution).astype(int)
    component_coordinates = utilities.state_vector_to_coordinates(solution)

(
    solar_panels,
    accumulators,
    substations,
    roboports,
    medium_poles,
) = component_coordinates



# Generate the Factorio blueprint string.

if blueprint_string:
    blueprint = blueprint_string.strip()
else:
    blueprint = blueprints.coordinates_to_factorio_blueprint(
        component_coordinates,
        label=Path(solution_path).stem,
    )

print("\nBlueprint:")
print(blueprint)


# Print a short summary.

print("Solar panels:", len(solar_panels))
print("Accumulators:", len(accumulators))
print("Substations:", len(substations))
print("Roboports:", len(roboports))
print("Medium poles:", len(medium_poles))



# Plot the complete periodic array.

panel_plot.plot_solar_array_periodic(
    grid,
    solar_panels,
    accumulators,
    substations,
    roboports,
    medium_poles,
    plot_electric=False,
)

power = min(
    len(solar_panels) * parameters.SOLAR_PANEL_POWER * parameters.ETA_S,
    (len(accumulators) + 4 * len(roboports))
    * parameters.ACCUMULATOR_CHARGE
    / parameters.DAY_DURATION
    / parameters.C_ON,
)

plt.title(f"Sustained Power: {power:g} kW")
plt.show()
