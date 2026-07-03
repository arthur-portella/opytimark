import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 1. Define the WFG2 test problem (e.g., 2 objectives, 3 position parameters, 3 distance parameters)
problem = get_problem("wfg2", n_var=6, n_obj=2)

# 2. Configure NSGA-II with 100 individuals and 250 generations
algorithm = NSGA2(pop_size=100)

# 3. Run the optimization
res = minimize(problem,
               algorithm,
               ('n_gen', 250),
               seed=1,
               verbose=False)

# 4. Visualize and save the graphic
plot = Scatter()
plot.add(res.F, facecolor="none", edgecolor="red", label="Non-dominated Solutions")
plot.show()
