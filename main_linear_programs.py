import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import experiments

import power_alm.functions as fun
import power_alm.inner_solver as optim
import power_alm.solver as solver
import power_alm.problems as problems
import power_alm.alm as alm

import linear_problem_generator as gen

"""
This experiment considers LPs with linear INEQUALITY constraints.
"""

def generate_problem(m, n, cond = 1250, seed = 141234):
    params = gen.Parameters()
    np.random.seed(seed)
    params.seed = seed
    params.condition_number = cond
    params.norm_A = 350
    (A, b, c, opt_x, opt_y, opt_s) = gen.generate_lo_problem_with_opt(n, m, params)

    return (b, -A.T, c, -opt_y, opt_x)

class LPs(experiments.Experiment):
    def __init__(self, name, config, optimizer_configs, num_runs):
        super().__init__(name, config["n"], config, optimizer_configs, num_runs)

    def get_filename(self):
        return (self.name + "_config_"
            + self.config["name"]
            + "_num_runs_" + str(self.num_runs)
            + "_seed_" + str(self.config["seed"])
            + "_m_" + str(self.config["m"])
            + "_n_" + str(self.config["n"]))
    
    def initialize_composite_problem(self):
        m = config["m"]; n = config["n"]

        cond = 1000
        seed = np.random.randint(1, 1000000)

        #########################################
        ### Generate LP (inequality constrained)
        #########################################
        (c, G, h, opt_x, opt_y) = generate_problem(m, n, cond, seed)

        # No lower and upper bounds
        lb = -np.inf; lb=np.array((lb,) * n)
        ub = +np.inf; ub=np.array((ub,) * n)

        # ALM parameters
        penalty = 100.
        x_init = np.random.randn(n)
        ye_init = np.zeros(m)
        yi_init = np.zeros(m)

        ye_init = None
        
        problem = problems.ConvexQuadraticProblem(np.zeros((n, n)), c, None, None, G, h, lb, ub)
        alm_problem = alm.ConvexQuadraticAugmentedLagrangian(problem, 2., penalty, ye_init, yi_init, norm2=True)
        self.composite_problem = optim.CompositeOptimizationProblem(x_init, alm_problem, fun.Zero())

        self.opt_obj = problem.eval_objective(opt_x)
        self.opt_const = 0.

    def compute_optimum(self, x_init):
        pass

    def eval_objective(self, x):
        return self.composite_problem.diffable._problem.eval_objective(x)

    def eval_constraint_violation(self, x):
        return np.linalg.norm(np.maximum(self.composite_problem.diffable.eval_ineq_res(x), 0.))
    
    def eval_power_stepsize(self, x):
        return np.power(self.composite_problem.diffable.penalty, self.composite_problem.diffable._q - 1.) * np.power(np.linalg.norm(self.composite_problem.diffable.yi - self.composite_problem.diffable.yi_old), 2 - self.composite_problem.diffable._q)

name = "lp"
num_runs = 20
configs = [
    {
        "name": "lp200x100",
        "m": 200,
        "n": 100,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": False,
    },
    {
        "name": "lp400x200",
        "m": 400,
        "n": 200,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": False,
    },
    {
        "name": "lp600x300",
        "m": 600,
        "n": 300,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": False,
    },
    {
        "name": "lp300x100",
        "m": 300,
        "n": 100,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": False,
    },
    {
        "name": "lp600x200",
        "m": 600,
        "n": 200,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": False,
    },
    {
        "name": "lp400x100",
        "m": 400,
        "n": 100,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": False,
    },
    {
        "name": "lp500x100",
        "m": 500,
        "n": 100,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": False,
    },
    {
        "name": "lp600x100",
        "m": 600,
        "n": 100,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": False,
    },
]

# Clear csv
with open("results/" + name + ".csv", "w") as empty_csv:
    pass

for config in configs:
    optimizer_configs = [
        {
            "marker": "^",
            "linestyle": "solid",
            "color": "black",
            "name": "LAN (p = 2, lamb = 1000)",
            "label": "LAN (${p = 2, \lambda = 1000}$)",
            "class": optim.BFGS,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=1500,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=1000,
            )
        },
        {
            "marker": "^",
            "linestyle": "dashed",
            "color": "black",
            "name": "LAN (p = 2, lamb = 10000)",
            "label": "LAN (${p = 2, \lambda = 10000}$)",
            "class": optim.BFGS,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=1500,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=10000.,
            )
        },
        {
            "marker": "^",
            "linestyle": "dashdot",
            "color": "black",
            "name": "LAN (p = 2, adap lamb = 100)",
            "label": "LAN (${p = 2, \lambda = 100}$ adaptive)",
            "class": optim.BFGS,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=1500,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=100,
                adaptive_penalty=True,
                adaptive_penalty_tightness = 0.001,
            )
        },
        {
            "marker": "^",
            "linestyle": "dotted",
            "color": "black",
            "name": "LAN (p = 2, adap lamb = 1000)",
            "label": "LAN (${p = 2, \lambda = 1000}$ adaptive)",
            "class": optim.BFGS,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=1500,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=1000,
                adaptive_penalty=True,
                adaptive_penalty_tightness = 0.001,
            )
        },
        {
            "marker": "*",
            "linestyle": "solid",
            "color": "blue",
            "name": "LAN (p = 1.9, lamb = 100)",
            "label": "LAN (${p = 1.9, \lambda = 100}$)",
            "class": optim.BFGS,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=1500,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 1.9,
                penalty=100,
            )
        },
        {
            "marker": "*",
            "linestyle": "solid",
            "color": "blue",
            "name": "LAN (p = 1.9, lamb = 1000)",
            "label": "LAN (${p = 1.9, \lambda = 1000}$)",
            "class": optim.BFGS,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=1500,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 1.9,
                penalty=1000,
            )
        },
        {
            "marker": "o",
            "linestyle": "solid",
            "color": "purple",
            "name": "LAN (p = 1.8, lamb = 100)",
            "label": "LAN (${p = 1.8, \lambda = 100}$)",
            "class": optim.BFGS,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=1500,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 1.8,
                penalty=100,
            )
        },
        {
            "marker": "o",
            "linestyle": "solid",
            "color": "purple",
            "name": "LAN (p = 1.8, lamb = 1000)",
            "label": "LAN (${p = 1.8, \lambda = 1000}$)",
            "class": optim.BFGS,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=1500,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 1.8,
                penalty=1000,
            )
        },
    ]

    experiment = LPs(name, config, optimizer_configs, num_runs)
    experiment.run(overwrite_file=False)

    if num_runs > 1:
        m = config["m"]; n = config["n"]
        print(f"\nm = {m}, n = {n}")

        experiment.compute_grid(MEDIANS_ONLY=True, P95S_ONLY=False, SAVE_CSV=True, filename=name)

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("$m=" + str(config["m"]) + "$, " +
                 "$n=" + str(config["n"]) + "$, ", fontsize=16)
    ax.grid(True)
    experiment.plot(markevery=config["markevery"], PLOT_TOTAL_NIT=True, run = 0)

    filename = experiment.get_filename()
    suffix = ".pdf"
    plt.savefig(experiments.RESULTS + filename + suffix, bbox_inches='tight')
    plt.show(block=False)

    # fig = plt.figure()
    # experiment.plot_powerstepsizes(markevery=config["markevery"], PLOT_TOTAL_NIT=True, run = 0)
    # plt.show(block=True)