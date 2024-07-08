import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import experiments

import power_alm.functions as fun
import power_alm.inner_solver as optim
import power_alm.solver as solver
import power_alm.problems as problems
import power_alm.alm as alm

class L1Regression(experiments.Experiment):
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

        A = 10 * np.random.rand(m, n) - 5
        b = np.random.randn(m)
        # b = np.zeros(m)
        theta = 100.

        # ALM parameters
        penalty = 1.
        x_init = np.random.randn(n)
        y_init = np.random.randn(m)
        
        problem = problems.L1RegressionProblem(A, b, theta = theta)
        alm_problem = alm.L1RegressionAlmWrappedProblem(problem, 2., penalty, y_init)
        self.composite_problem = optim.CompositeOptimizationProblem(x_init, alm_problem, fun.Zero())

    def compute_optimum(self, x_init):
        config = self.config

        np.random.seed(config["seed"])

        m = config["m"]; n = config["n"]
        A = 10 * np.random.rand(m, n) - 5
        b = np.random.randn(m)
        # b = np.zeros(m)
        theta = 100.

        # ALM parameters
        penalty = 100.
        x_init = np.random.randn(n)
        y_init = np.random.randn(m)
        
        problem = problems.L1RegressionProblem(A, b, theta = theta)
        alm_problem = alm.L1RegressionAlmWrappedProblem(problem, 2., penalty, y_init)
        composite_problem = optim.CompositeOptimizationProblem(x_init, alm_problem, fun.Zero())

        optimizer = solver.Solver(
            composite_problem, 
            solver.SolverParameters(optim.Parameters(maxit=800, tol=1e-10, epsilon=1e-12, Wolfe=True), maxit=10, norm2=True),
            inner_solver=optim.LBFGS,
            callback= lambda k, nit, x : print(k - 1, nit, self.eval_objective_(x, composite_problem.diffable.y), self.eval_constraint_violation_(x, composite_problem.diffable.y), np.linalg.norm(composite_problem.diffable.eval_gradient(x)))
        )

        optimizer.run()
        self.opt_obj = self.eval_objective_(optimizer.x, composite_problem.diffable.y)
        self.opt_const = 0.

    """
    Primal cost
    """
    def eval_objective_(self, x, y):
        # return self.composite_problem.diffable._problem.eval_objective(x) + np.dot(self.composite_problem.diffable.eval_res(x), y)
        return self.composite_problem.diffable._problem.eval_objective(x) + np.linalg.norm(self.composite_problem.diffable.eval_res(x), 1)

    def eval_objective(self, x):
        return self.eval_objective_(x, self.composite_problem.diffable.y)

    """
    Primal dual gap
    """
    def eval_constraint_violation_(self, x, y):
        return self.eval_objective_(x, y) + np.dot(y, self.composite_problem.diffable._problem.b) + 1. / (2 * self.composite_problem.diffable._problem.theta) * np.power(np.linalg.norm(- self.composite_problem.diffable._problem.A.T @ y), 2)
    
    def eval_constraint_violation(self, x):
        return self.eval_objective(x) + np.dot(self.composite_problem.diffable.y, self.composite_problem.diffable._problem.b) + 1. / (2 * self.composite_problem.diffable._problem.theta) * np.power(np.linalg.norm(- self.composite_problem.diffable._problem.A.T @ self.composite_problem.diffable.y), 2)
    
    def eval_power_stepsize(self, x):
        if self.config["norm2"]:
            return np.power(self.composite_problem.diffable.penalty, self.composite_problem.diffable._q - 1.) * np.power(np.linalg.norm(self.composite_problem.diffable.y - self.composite_problem.diffable.y_old), 2 - self.composite_problem.diffable._q)

name = "l1_regression"
num_runs = 1
configs = [
    {
        "name": "l1_regression_normp_120x145",
        "m": 120,
        "n": 145,
        "markevery": 2,
        "plotevery": 20,
        "seed": 120,
        "maxit": 35,
        "tol": 1e-7,
        "init_proc": "np.zeros",
        "norm2": False,
    },
    {
        "name": "l1_regression_norm2_120x145",
        "m": 120,
        "n": 145,
        "markevery": 2,
        "plotevery": 20,
        "seed": 120,
        "maxit": 35,
        "tol": 1e-7,
        "init_proc": "np.zeros",
        "norm2": True,
    },
]

for config in configs:
    optimizer_configs = [
        {
            "marker": "^",
            "linestyle": "solid",
            "color": "black",
            "name": "p = 2, lamb = 1",
            "label": "${q = 1, \lambda = 1}$",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=100,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                penalty = 1.,
                norm2=config["norm2"],
            )
        },
        {
            "marker": "^",
            "linestyle": "dashdot",
            "color": "black",
            "name": "p = 2, adap lamb",
            "label": "${q = 1, \lambda_0 = 1}$",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=100,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                penalty = 1.,
                norm2=config["norm2"],
                adaptive_penalty=True,
            )
        },
        {
            "marker": "^",
            "linestyle": "dashed",
            "color": "black",
            "name": "p = 2, lamb = 10",
            "label": "${q = 1, \lambda = 10}$",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=250,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                penalty = 10.,
                norm2=config["norm2"],
            )
        },
        {
            "marker": "*",
            "linestyle": "solid",
            "color": "blue",
            "name": "p = 1.9, lamb = 1",
            "label": "${q = 0.9, \lambda = 1}$",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=100,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                p = 1.9,
                penalty = 1.,
                norm2=config["norm2"],
            )
        },
        {
            "marker": "o",
            "linestyle": "solid",
            "color": "purple",
            "name": "p = 1.8, lamb = 1",
            "label": "${q = 0.8, \lambda = 1}$",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=100,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                p = 1.8,
                penalty = 1.,
                norm2=config["norm2"],
            )
        },
        {
            "marker": "x",
            "linestyle": "solid",
            "color": "darkgreen",
            "name": "p = 1.7, lamb = 1",
            "label": "${q = 0.7, \lambda = 1}$",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=100,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                penalty = 1.,
                p = 1.7,
                norm2=config["norm2"],
            )
        },
    ]

    experiment = L1Regression(name, config, optimizer_configs, num_runs)
    experiment.run(overwrite_file=False)
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rcParams.update({'legend.fontsize': 18})
    fig, ax = plt.subplots(figsize=(6, 5))
    # ax.set_xticks([]); ax.set_yticks([])
    plt.xticks(fontsize=10); plt.yticks(fontsize=10)
    # fig.suptitle("$m=" + str(config["m"]) + "$, " +
                #  "$n=" + str(config["n"]) + "$, ", fontsize=16)
    ax.grid(True)
    experiment.plot(markevery=config["markevery"], SINGLE_PLOT=True, xlabels = [""], ylabels=[""])

    filename = experiment.get_filename()
    suffix = ".pdf"
    plt.savefig(experiments.RESULTS + filename + suffix, bbox_inches='tight')
    plt.show(block=False)

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rcParams.update({'legend.fontsize': 18})
    fig, ax = plt.subplots(figsize=(6, 5))
    # ax.set_xticks([]); ax.set_yticks([])
    plt.xticks(fontsize=10); plt.yticks(fontsize=10) 
    ax.grid(True)
    experiment.plot_powerstepsizes(markevery=config["markevery"], PLOT_TOTAL_NIT=True, xlabels = [""], ylabels=[""])
    plt.show(block=True)