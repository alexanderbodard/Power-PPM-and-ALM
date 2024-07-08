import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils.experiments as experiments

import power_alm.functions as fun
import power_alm.inner_solver as optim
import power_alm.solver as solver
import power_alm.problems as problems
import power_alm.alm as alm

class ConvexQuadratics(experiments.Experiment):
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

        ### Define quadratic objective
        u = np.maximum(np.random.randn(n) + 5., 0.)
        k = np.random.randint(n / 4., n / 2.)
        indices = np.array(np.random.choice([*range(0, n)], k))
        u[indices] = 0.
        U = np.random.randn(n, n)
        V, _ = np.linalg.qr(U)
        Q = V @ np.diag(u) @ V.T
        Q = 0.5 * (Q + Q.T)
        q = np.random.randn(n)

        ### Define linear equalities
        A = np.random.randn(m, n)
        x0 = np.random.randn(n)
        x0 = x0 / np.max(x0) * 1.3
        b = np.dot(A, x0)

        ### Define linear inequalities
        G = np.random.randn(m, n)
        h = G @ x0 + np.random.randn(m)

        # Lower and upper bounds
        lb = -np.inf; lb=np.array((lb,) * n)
        ub = +np.inf; ub=np.array((ub,) * n)

        # ALM parameters
        penalty = 100.
        x_init = np.random.randn(n)
        ye_init = np.zeros(m)
        yi_init = np.zeros(m)

        A = None; b = None
        ye_init = None
        
        problem = problems.ConvexQuadraticProblem(Q, q, A, b, G, h, lb, ub)
        alm_problem = alm.ConvexQuadraticAugmentedLagrangian(problem, 2., penalty, ye_init, yi_init, norm2=True)
        self.composite_problem = optim.CompositeOptimizationProblem(x_init, alm_problem, fun.Zero())

    def compute_optimum(self, x_init):
        config = self.config
        m = config["m"]; n = config["n"]

        ### Define quadratic objective
        Q = self.composite_problem.diffable._problem._fun._Q
        q = self.composite_problem.diffable._problem._fun._q

        ### Define linear inequalities
        G = self.composite_problem.diffable._problem._ci._A
        h = self.composite_problem.diffable._problem._ci._b

        # Use qpalm to compute solution
        import qpalm
        qpalm_data = qpalm.Data(n, m)
        qpalm_data.Q = Q
        qpalm_data.q = q
        qpalm_data.A = G
        qpalm_data.bmin = np.array((-np.inf,) * m)
        qpalm_data.bmax = h

        # Configure the solver 
        settings = qpalm.Settings()
        settings.eps_abs = 1e-12
        settings.eps_rel = 1e-10

        # Solve the problem
        qpalm_solver = qpalm.Solver(qpalm_data, settings)
        qpalm_solver.solve()
        assert qpalm_solver.info.status_val == qpalm.Info.SOLVED
        print(qpalm_solver.info.iter_out)
        print(qpalm_solver.info.iter)
        x_qpsolve = qpalm_solver.solution.x

        self.opt_obj = self.eval_objective(x_qpsolve)
        self.opt_const = 0.

        # Ensure that some constraints are active at the solution to avoid trivial problem realizations
        temp = G @ x_qpsolve - h
        assert sum(temp < -1e-10) < len(temp)

    def eval_objective(self, x):
        return self.composite_problem.diffable._problem.eval_objective(x)

    def eval_constraint_violation(self, x):
        return np.linalg.norm(np.maximum(self.composite_problem.diffable.eval_ineq_res(x), 0.))
     
    def eval_power_stepsize(self, x):
        return np.power(self.composite_problem.diffable.penalty, self.composite_problem.diffable._q - 1.) * np.power(np.linalg.norm(self.composite_problem.diffable.yi - self.composite_problem.diffable.yi_old), 2 - self.composite_problem.diffable._q)

name = "convex_quadratic"
num_runs = 1
configs = [
    {
        "name": "quadratic75x50",
        "m": 800,
        "n": 400,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 100,
        "tol": 1e-6,
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
            "name": "p = 2, lamb = 10",
            "label": "${q = 1, \lambda = 10}$",
            "class": optim.LBFGS_SCIPY,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=250,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=10,
            )
        },
        {
            "marker": "^",
            "linestyle": "dashdot",
            "color": "black",
            "name": "p = 2, adap lamb",
            "label": "${q = 1, \lambda_0 = 10}$",
            "class": optim.LBFGS_SCIPY,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=250,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=10,
                adaptive_penalty=True,
            )
        },
        {
            "marker": "^",
            "linestyle": "dashed",
            "color": "black",
            "name": "p = 2, lambda = 100",
            "label": "${q = 1, \lambda = 100}$",
            "class": optim.LBFGS_SCIPY,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=250,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=100,
            )
        },
        {
            "marker": "*",
            "linestyle": "solid",
            "color": "blue",
            "name": "p = 1.9",
            "label": "${q = 0.9, \lambda = 10}$",
            "class": optim.LBFGS_SCIPY,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=250,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 1.9,
                penalty=10,
            )
        },
        {
            "marker": "o",
            "linestyle": "solid",
            "color": "purple",
            "name": "p = 1.8",
            "label": "${q = 0.8, \lambda = 10}$",
            "class": optim.LBFGS_SCIPY,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=250,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 1.8,
                penalty=10,
            )
        },
        {
            "marker": "x",
            "linestyle": "solid",
            "color": "darkgreen",
            "name": "p = 1.7",
            "label": "${q = 0.7, \lambda = 10}$",
            "class": optim.LBFGS_SCIPY,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=250,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 1.7,
                penalty=10,
            )
        },
    ]

    experiment = ConvexQuadratics(name, config, optimizer_configs, num_runs)
    experiment.run(overwrite_file=False)

    if num_runs > 1:
        experiment.compute_grid()

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    # matplotlib.rcParams.update({'font.size': 20})
    # matplotlib.rcParams.update({'legend.fontsize': 18})
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xticks([]); ax.set_yticks([])
    plt.xticks(fontsize=10); plt.yticks(fontsize=10)
    # fig.suptitle("$m=" + str(config["m"]) + "$, " +
    #              "$n=" + str(config["n"]) + "$, ", fontsize=16)
    ax.grid(True)
    experiment.plot(markevery=config["markevery"], PLOT_TOTAL_NIT=True, SINGLE_PLOT=False, run = 0)

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