import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import experiments

import power_alm.functions as fun
import power_alm.inner_solver as optim
import power_alm.solver as solver
import power_alm.problems as problems
import power_alm.alm as alm

"""
This experiment considers QPs with linear EQUALITY constraints and BOUND constraints on the variables.
The former are imposed using ALM, the latter directly by the inner solver
"""

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
        b = np.random.rand(m) * 2 - 1 # Uniformly random over [-1, 1].

        ### Define linear inequalities
        G = None
        h = None

        # Lower and upper bounds
        lb_ = -.8; lb=np.array((lb_,) * n)
        ub_ = +.8; ub=np.array((ub_,) * n)

        # ALM parameters
        penalty = 100.
        x_init = np.random.randn(n)
        ye_init = np.zeros(m)
        yi_init = np.zeros(m)

        yi_init = None
        
        problem = problems.ConvexQuadraticProblem(Q, q, A, b, G, h, lb, ub)
        alm_problem = alm.ConvexQuadraticAugmentedLagrangian(problem, 2., penalty, ye_init, yi_init, norm2=True)
        self.composite_problem = optim.CompositeOptimizationProblem(x_init, alm_problem, fun.IndicatorBox(l = lb_, u =  ub_))

    def compute_optimum(self, x_init):
        config = self.config
        m = config["m"]; n = config["n"]

        ### Define quadratic objective
        Q = self.composite_problem.diffable._problem._fun._Q
        q = self.composite_problem.diffable._problem._fun._q

        ### Define linear inequalities
        A = self.composite_problem.diffable._problem._ce._A
        b = self.composite_problem.diffable._problem._ce._b

        ### Bounds
        lb = self.composite_problem.diffable._problem._lb
        ub = self.composite_problem.diffable._problem._ub

        # Use qpalm to compute solution
        import qpalm
        qpalm_data = qpalm.Data(n, m + n)
        qpalm_data.Q = Q
        qpalm_data.q = q
        qpalm_data.A = np.row_stack((A, np.identity(n)))
        qpalm_data.bmin = np.concatenate((
            b,
            lb,
        ))
        qpalm_data.bmax = np.concatenate((
            b,
            ub,
        ))

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
        print(x_qpsolve)
        temp_ub = x_qpsolve - ub[0]; temp_lb = lb[0] - x_qpsolve
        assert sum(temp_ub < -1e-10) < len(temp_ub) or sum(temp_lb < -1e-10) < len(temp_lb)

    def eval_objective(self, x):
        return self.composite_problem.diffable._problem.eval_objective(x)

    def eval_constraint_violation(self, x):
        return np.linalg.norm(np.maximum(self.composite_problem.diffable.eval_eq_res(x), 0.))
    
    def eval_power_stepsize(self, x):
        return self.composite_problem.diffable.penalty * np.power(np.linalg.norm(self.composite_problem.diffable.eval_eq_res(x)), self.composite_problem.diffable._p - 2.)

name = "convex_quadratic_v2"
num_runs = 20
configs = [
    {
        "name": "quadratic200x400",
        "m": 200,
        "n": 400,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": True,
    },
    {
        "name": "quadratic250x500",
        "m": 250,
        "n": 500,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": True,
    },
    {
        "name": "quadratic300x600",
        "m": 300,
        "n": 600,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": True,
    },
    {
        "name": "quadratic350x700",
        "m": 350,
        "n": 700,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": True,
    },
    {
        "name": "quadratic400x800",
        "m": 400,
        "n": 800,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": True,
    },    
    {
        "name": "quadratic450x900",
        "m": 450,
        "n": 900,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": True,
    },
    {
        "name": "quadratic150x450",
        "m": 150,
        "n": 450,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": True,
    },
    {
        "name": "quadratic200x600",
        "m": 200,
        "n": 600,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": True,
    },
    {
        "name": "quadratic250x750",
        "m": 250,
        "n": 750,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": True,
    },
    {
        "name": "quadratic300x900",
        "m": 300,
        "n": 900,
        "markevery": 2,
        "plotevery": 20,
        "seed": 1,
        "maxit": 150,
        "tol": 1e-6,
        "init_proc": "np.zeros",
        "norm2": True,
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
            "name": "LAN (p = 2, lamb = 0.1)",
            "label": "LAN (${p = 2, \lambda = 0.1}$)",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=5000,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=0.1,
            )
        },
        {
            "marker": "^",
            "linestyle": "solid",
            "color": "black",
            "name": "LAN (p = 2, lamb = 1)",
            "label": "LAN (${p = 2, \lambda = 1}$)",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=5000,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=1.,
            )
        },
        {
            "marker": "^",
            "linestyle": "solid",
            "color": "black",
            "name": "LAN (p = 2, lamb = 10)",
            "label": "LAN (${p = 2, \lambda = 10}$)",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=5000,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=10.,
            )
        },
        {
            "marker": "^",
            "linestyle": "dotted",
            "color": "black",
            "name": "LAN (p = 2, adap lamb = 0.01)",
            "label": "LAN (${p = 2, \lambda = 0.01}$ adaptive)",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=5000,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=0.01,
                adaptive_penalty=True,
                adaptive_penalty_tightness = 0.001,
            )
        },
        {
            "marker": "^",
            "linestyle": "dotted",
            "color": "black",
            "name": "LAN (p = 2, adap lamb = 0.1)",
            "label": "LAN (${p = 2, \lambda = 0.1}$ adaptive)",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=5000,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=0.1,
                adaptive_penalty=True,
                adaptive_penalty_tightness = 0.1,
            )
        },
        {
            "marker": "^",
            "linestyle": "dashdot",
            "color": "black",
            "name": "LAN (p = 2, adap lamb = 1)",
            "label": "LAN (${p = 2, \lambda = 1}$ adaptive)",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=5000,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 2.,
                penalty=1,
                adaptive_penalty=True,
                adaptive_penalty_tightness = 0.1,
            )
        },
        {
            "marker": "*",
            "linestyle": "solid",
            "color": "blue",
            "name": "LAN (p = 1.9, lamb = 0.1)",
            "label": "LAN (${p = 1.9, \lambda = 0.1}$)",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=400,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 1.9,
                penalty=0.1,
            )
        },
        {
            "marker": "o",
            "linestyle": "solid",
            "color": "purple",
            "name": "LAN (p = 1.8, lamb = 0.1)",
            "label": "LAN (${p = 1.8, \lambda = 0.1}$)",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=400,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 1.8,
                penalty=0.1,
            )
        },
        {
            "marker": "x",
            "linestyle": "solid",
            "color": "darkgreen",
            "name": "LAN (p = 1.7, lamb = 0.1)",
            "label": "LAN (${p = 1.7, \lambda = 0.1}$)",
            "class": optim.UniversalFastPGMLan,
            "parameters": solver.SolverParameters(
                optim.Parameters(
                    maxit=750,
                    tol=1e-10,
                ),
                maxit=config["maxit"],
                tol=config["tol"],
                norm2=config["norm2"],
                p = 1.7,
                penalty=0.1,
            )
        },
    ]

    experiment = ConvexQuadratics(name, config, optimizer_configs, num_runs)
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