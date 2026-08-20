Hi!

As I mentioned in the video, this code is not the most clean you will ever see, but you can probably figure out how it works. Just in case, I'm leaving
this as a quick guideline on how to run your first optimization.

 /parameters.py                             Holds most of the main problem constants. You probably do not want to touch this too much.
 /plot_solution.py                          Can take either a path string or a blueprint string and plot the panel just as you see in the video.
 /print_blueprint.py                        Just prints out a blueprint from a path. I left that just so you can see how the blueprint parser works.
 

MAIN SOLVERS

 /solve_system_highs_fixed_network.py       This is what you want to start with, it takes in a fixed electric network. I left one pretty good for you to use.
                                            The optimization is limited to just 100 seconds and you should see results right away, which you can plot with
                                            plot_solution.py. This is what I call a Stage B solver, takes in a network, and solves the packing around it.

 /solve_system_highs_stage_A.py             This is a Stage A solver, so you can generate your own networks. Currently set to run for 300 seconds only just so 
                                            you see how it works. Score is number of tiles used by the network over 4, so it can be interpreted as number of 
                                            substations. If you run it as is you should be able to get a 8 to 8.5 result which you can then plug in 
                                            into the previous fixed network solver to get an easy 8232 kW solution.

 /solve_system_highs.py                     This is a real deal Stage A+B solver with the full problem, including connectivity, so this takes forever to run
                                            but will give you solutions without an electric network input. When I say forever, I mean forever, on single core
                                            and using HiGHS you might be waiting for days until you see an incumbents or weeks. You probably want to find
                                            a parallel capable solver like "gurobi" and license to run this well, but if you have time, you can theoretically do
                                            it with HiGHS. 

 /staged_solver_highs.py                    This one is the staged solver made mostly by AI which I use to find the 8316 permanent roboport setup in the video.
                                            Its currently made for that specific purpose, so it takes advantages of fixed building counts and also uses seeds 
                                            I generated with the other two solvers. You can make modifications to suit your needs and solve other similar problems.                   




 /objectives.py                             This contains most of my constructors for the STAGE B or STAGE A+B. Do not touch this unless you are familiar 
                                            with everything else.
 /coverage_objectives.py                    Contains constructots for STAGE A (Connectivity + Tileability). Again, do not touch until you are familiar with 
                                            the formulation.

