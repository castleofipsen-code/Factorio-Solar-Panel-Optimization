from pathlib import Path
import sys

import numpy as np

import parameters
from support import blueprints, utilities


solution_path = "render/solution_8358.sol"
grid = 50

parameters.GRID_SIZE = grid
solution = utilities.load_solution(solution_path)

coordinates = utilities.state_vector_to_coordinates(
    np.rint(solution).astype(int)
)

blueprint = blueprints.coordinates_to_factorio_blueprint(
    coordinates,
    label=solution_path,
)

print(blueprint)
