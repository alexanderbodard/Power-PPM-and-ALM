import sys
import os
relative_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(relative_path + "/../../src")

import numpy as np
import random

import Problems.linear_problem_generator as gen
from Problems.problems import *
from alm import InnerSolver, PowerALM

def run(m, n, seed = None):
    if seed is not None:
        np.random.seed(seed)
    
    u = np.maximum(np.random.randn(n) + 5., 0.)
    k = np.random.randint(n / 4., n / 2.)
    indices = np.array(random.sample([*range(0, n)], k=k))
    u[indices] = 0.

    ### Define quadratic objective
    U = np.random.randn(n, n)
    V, R = np.linalg.qr(U)

    Q = V @ np.diag(u) @ V.T
    
    q = np.random.randn(n)

    f = QuadFunc(
        Q,
        q
    )

    ### Define linear equalities
    A = np.random.randn(m, n)
    b = np.random.rand(m) * 2 - 1 # Uniformly random over [-1, 1].

    ### Define linear inequalities
    G = None
    h = None

    ### Reference solution
    lb = -.8
    ub = .8

    import qpalm

    # Data
    data = qpalm.Data(n, m + n)
    data.Q = f.Q
    data.q = f.q
    data.A = np.row_stack((A, np.identity(n)))
    data.bmin = np.concatenate((
        b,
        np.array((lb,) * n),
    ))
    data.bmax = np.concatenate((
        b,
        np.array((ub,) * n),
    ))

    # Configure the solver 
    settings = qpalm.Settings()
    settings.eps_abs = 1e-12
    settings.eps_rel = 1e-10

    # Solve the problem
    qpalm_solver = qpalm.Solver(data, settings)
    qpalm_solver.solve()

    print(qpalm_solver.info.iter_out)
    print(qpalm_solver.info.iter)

    x_qpsolve = qpalm_solver.solution.x

    ### Define problem
    problem = Problem(
        f.eval,
        f.grad,
        A,
        b,
        G,
        h,
        is_quad_obj=True,
        lb = lb,
        ub = ub
    )

    # Initialize algorithm with random x0
    x0 = np.random.rand(n)

    lbfgsb_iterations = []

    ###################################
    ### Power ALM 2-norm p = 1.5
    ###################################
    p = 1.5
    q = p / (p - 1.)
    maxit = 150
    alm_power = PowerALM(
        problem, 
        p,
        m,
        n,
        maxit = maxit,
        sigma = .05,
        tol = 1e-6,
        inner_solver=InnerSolver.LBFGS_SCIPY,
        logging = False,
        hess_f = f.hess,
        norm2 = True
    )

    x_alm = alm_power.solve(np.copy(x0), f_ref = f.eval(x_qpsolve), inner_tol=lambda i: 1e1 / np.power(i, q))

    print(np.abs(f.eval(x_alm) - f.eval(x_qpsolve)))
    print(np.linalg.norm(x_alm - x_qpsolve))

    lbfgsb_iterations.append(alm_power.total_nit)

    ###################################
    ### Power ALM 2-norm p = 4 / 3
    ###################################
    p = 4. / 3.
    q = p / (p - 1.)
    maxit = 150
    alm_power = PowerALM(
        problem, 
        p,
        m,
        n,
        maxit = maxit,
        sigma = 0.05,
        tol = 1e-6,
        inner_solver=InnerSolver.LBFGS_SCIPY,
        logging = False,
        hess_f = f.hess,
        norm2 = True
    )

    x_alm = alm_power.solve(np.copy(x0), f_ref = f.eval(x_qpsolve), inner_tol=lambda i: 1e2 / np.power(i, q))

    print(np.abs(f.eval(x_alm) - f.eval(x_qpsolve)))
    print(np.linalg.norm(x_alm - x_qpsolve))

    lbfgsb_iterations.append(alm_power.total_nit)

    ###################################
    ### Classical ALM tuned Sigma
    ###################################
    p = 2
    q = p / (p - 1.)
    maxit = 150
    alm_classical = PowerALM(
        problem, 
        p,
        m,
        n,
        maxit = maxit,
        sigma = .1,
        tol = 1e-6,
        inner_solver=InnerSolver.LBFGS_SCIPY,
        logging = False,
        hess_f = f.hess,
        should_update_sigma=False
    )

    x_alm = alm_classical.solve(np.copy(x0), f_ref = f.eval(x_qpsolve), inner_tol = lambda i: 1e-2 / np.power(i, q))

    print(np.abs(f.eval(x_alm) - f.eval(x_qpsolve)))
    print(np.linalg.norm(x_alm - x_qpsolve))

    lbfgsb_iterations.append(alm_classical.total_nit)

    ###################################
    ### Classical ALM Adaptive Sigma
    ###################################
    p = 2
    q = p / (p - 1.)
    maxit = 150
    alm_classical = PowerALM(
        problem, 
        p,
        m,
        n,
        maxit = maxit,
        sigma = .1,
        tol = 1e-6,
        inner_solver=InnerSolver.LBFGS_SCIPY,
        logging = False,
        hess_f = f.hess,
        should_update_sigma=True
    )

    x_alm = alm_classical.solve(np.copy(x0), f_ref = f.eval(x_qpsolve), inner_tol = lambda i: 1e-2 / np.power(i, q))

    print(np.abs(f.eval(x_alm) - f.eval(x_qpsolve)))
    print(np.linalg.norm(x_alm - x_qpsolve))

    lbfgsb_iterations.append(alm_classical.total_nit)

    return np.array(lbfgsb_iterations)

N = 20
n_solvers = 4

seed = 1
np.random.seed(seed)
mn_list = [(100, 200), (200, 400), (300, 600), (400, 800), (500, 1000), (600, 1200), (100, 300), (200, 600), (300, 900), (400, 1200)]
mn_iters = np.zeros((len(mn_list), 2 * n_solvers))

for mn_i, mn in enumerate(mn_list):
    m, n = mn
    lbfgsb_iters = np.zeros((N, n_solvers))
    for i in range(N):
        lbfgsb_iters[i, :] = run(m, n)
    mn_iters[mn_i, 0] = np.median(lbfgsb_iters, axis=0)[0]
    mn_iters[mn_i, 1] = np.percentile(lbfgsb_iters, 95, axis=0)[0]
    mn_iters[mn_i, 2] = np.median(lbfgsb_iters, axis=0)[1]
    mn_iters[mn_i, 3] = np.percentile(lbfgsb_iters, 95, axis=0)[1]
    mn_iters[mn_i, 4] = np.median(lbfgsb_iters, axis=0)[2]
    mn_iters[mn_i, 5] = np.percentile(lbfgsb_iters, 95, axis=0)[2]
    mn_iters[mn_i, 6] = np.median(lbfgsb_iters, axis=0)[3]
    mn_iters[mn_i, 7] = np.percentile(lbfgsb_iters, 95, axis=0)[3]

import sys
import os
relative_path = os.path.dirname(os.path.realpath(__file__))
log_path = f'{relative_path}/results/{N}_{seed}'
from datetime import datetime
log_path = datetime.now().strftime(f'{log_path}_%H_%M_%S_%d_%m_%Y')
os.makedirs(log_path)
log_path = log_path
np.savetxt(log_path + '/lbfgs_iters.csv', mn_iters)

print(mn_iters)