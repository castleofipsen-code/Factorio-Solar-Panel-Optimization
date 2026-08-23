# Factorio Solar Panel Optimization

Hi!

As I mentioned in the video, this code is not the cleanest you will ever see,
but you can probably figure out how it works. Just in case, I'm leaving this as
a quick guideline on how to run your first optimization.

## Useful files

- [`parameters.py`](parameters.py) holds most of the main problem constants. You
  probably do not want to touch this too much.
- [`plot_solution.py`](plot_solution.py) can take either a path string or a
  blueprint string and plot the panel just as you see in the video.
- [`print_blueprint.py`](print_blueprint.py) just prints out a blueprint from a
  path. I left that just so you can see how the blueprint parser works.
- [`solar_vs_substation.py`](solar_vs_substation.py) plots the optimality-proof
  charts, showing either the required electric-network efficacy or the number
  of empty tiles.

## Main solvers

### [`solve_system_highs_fixed_network.py`](solve_system_highs_fixed_network.py)

This is what you want to start with. It takes in a fixed electric network, and
I left one pretty good one for you to use. The optimization is limited to just
100 seconds, and you should see results right away, which you can plot with
`plot_solution.py`. This is what I call a Stage B solver: it takes in a network
and solves the packing around it.

### [`solve_system_highs_stage_A.py`](solve_system_highs_stage_A.py)

This is a Stage A solver, so you can generate your own networks. It is currently
set to run for only 300 seconds, just so you can see how it works. The score is
the number of tiles used by the network divided by 4, so it can be interpreted
as the number of substations. If you run it as is, you should be able to get an
8 to 8.5 result, which you can then plug into the previous fixed-network solver
to get an easy 8232 kW solution.

### [`solve_system_highs.py`](solve_system_highs.py)

This is the real-deal Stage A+B solver with the full problem, including
connectivity, so it takes forever to run but will give you solutions without an
electric-network input. When I say forever, I mean forever. On a single core,
using HiGHS, you might be waiting for days until you see an incumbent, or weeks
for more progress. You probably want to find a parallel-capable solver to run
this well, but if you have time, you can theoretically do it with HiGHS.

### [`staged_solver_highs.py`](staged_solver_highs.py)

This is the staged solver I used to find the 8316 permanent-roboport setup in
the video. It is currently made for that specific purpose, so it takes advantage
of fixed building counts and also uses seeds I generated with the other two
solvers. You can make modifications to suit your needs and solve other similar
problems, but its not as easy as changing variables, this is not made fully genetic, but specifically intended at solving the recalcitrant 8316 problem.

## Model constructors

- [`objectives.py`](objectives.py) contains most of my constructors for Stage B
  or Stage A+B. Do not touch this unless you are familiar with everything else.
- [`coverage_objectives.py`](coverage_objectives.py) contains constructors for
  Stage A (connectivity and tileability). Again, do not touch this until you are
  familiar with the formulation.

## License

This project is available under the [MIT License](LICENSE).
