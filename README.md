# Factorio Solar Panel Optimization

Hi! This repository contains the optimization code discussed in the video. It is
intended as a practical, tutorial-style introduction to the models, rather than
as a polished software package.

If you are running the project for the first time, start with the fixed-network
solver. It produces useful results quickly and is the easiest model to follow.

## Getting started

Install the Python dependencies:

```bash
python -m pip install numpy scipy matplotlib
```

Then run the fixed-network example:

```bash
python solve_system_highs_fixed_network.py
```

The solver writes its result to the local `results/` folder. To plot a saved
solution, set `solution_path` near the top of [`plot_solution.py`](plot_solution.py)
and run:

```bash
python plot_solution.py
```

Most settings—including thread count, time limit, building limits, and input
paths—are deliberately kept near the top of each solver script so they are easy
to find and edit.

## Solvers

### [`solve_system_highs_fixed_network.py`](solve_system_highs_fixed_network.py)

The recommended starting point. This is a **Stage B** solver: it takes a fixed
electrical network and optimizes the solar-panel and accumulator packing around
it. A good example network is included in
[`support/sample_network.txt`](support/sample_network.txt).

The example is limited to 100 seconds and should begin producing results
quickly.

### [`solve_system_highs_stage_A.py`](solve_system_highs_stage_A.py)

A simple **Stage A** solver for generating electrical networks. The default run
is limited to 300 seconds. Its score is the number of network tiles divided by
four, so it can be read approximately as an equivalent substation count.

As configured, it should find networks scoring roughly 8 to 8.5. These can be
used as inputs to the fixed-network Stage B solver and should make an 8232 kW
array reasonably accessible.

### [`solve_system_highs.py`](solve_system_highs.py)

The complete **Stage A+B** model, including electrical connectivity. It requires
no network input, but the full mixed-integer problem is extremely demanding.
With HiGHS, an incumbent may take days to appear and a complete run may take
much longer.

### [`staged_solver_highs.py`](staged_solver_highs.py)

The experimental staged search used to find the 8316 kW permanent-roboport
setup shown in the video. It combines network discovery, packing bounds, and
penalty-guided search. It is currently specialized for fixed building counts
and uses the included files in [`staged_seeds/`](staged_seeds/) as starting
points, but it can be adapted to related layouts.

## Other useful files

| File | Purpose |
| --- | --- |
| [`parameters.py`](parameters.py) | Main problem and building constants. You normally do not need to change these. |
| [`plot_solution.py`](plot_solution.py) | Plots either a saved `.sol` file or a pasted Factorio blueprint string. |
| [`print_blueprint.py`](print_blueprint.py) | Converts a saved solution into a Factorio blueprint string and prints it. |
| [`objectives.py`](objectives.py) | Model constructors for Stage B and combined Stage A+B optimization. |
| [`coverage_objectives.py`](coverage_objectives.py) | Model constructors for Stage A connectivity and tileability. |
| [`support/`](support/) | Plotting, blueprint, solution-loading, and sample-data helpers. |

The constructor modules are the most technical part of the project. It is worth
becoming familiar with the smaller solver scripts before modifying them.

## License

This project is available under the [MIT License](LICENSE).
