"""Color-completion helper used by the free-coordinate network oracle."""

import numpy as np


GRID = 50
DD = GRID * GRID
NETWORK_SIZE = 2 * DD


def color_witness(network):
    """Return the four panel-color counts for a valid 198-panel tiling."""
    mediums = np.flatnonzero(network[DD:NETWORK_SIZE] > 0.5)
    if len(mediums) != 10:
        raise ValueError("A network must contain exactly ten medium poles.")

    counts = np.bincount(
        2 * ((mediums // GRID) % 2) + ((mediums % GRID) % 2),
        minlength=4,
    )
    solar_total = 198
    equal_total = (9 * solar_total + 10) // 4
    matrix = np.asarray(
        (
            (4, 2, 2, 1),
            (2, 4, 1, 2),
            (2, 1, 4, 2),
            (1, 2, 2, 4),
        ),
        dtype=float,
    )
    panels = np.linalg.solve(matrix, equal_total - counts)
    rounded = np.rint(panels)
    if (
        np.max(np.abs(panels - rounded)) > 1e-7
        or np.any(rounded < 0)
        or int(np.sum(rounded)) != solar_total
    ):
        raise ValueError("Medium colors have no 198-panel completion.")
    return np.concatenate((rounded, (float(equal_total), 2.0)))
