import sys
import os
relative_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(relative_path + "/../../src")

import numpy as np
import random

import Problems.linear_problem_generator as gen
from Problems.problems import *
from alm import InnerSolver, PowerALM

def generate_problem(m, n, cond = 1250, seed = 141234):
    params = gen.Parameters()
    np.random.seed(seed)
    params.seed = seed
    params.condition_number = cond
    params.norm_A = 350
    (A, b, c, opt_x, opt_y, opt_s) = gen.generate_lo_problem_with_opt(n, m, params)

    return (b, -A.T, c, -opt_y, opt_x)

def run(m, n, cond = 100, seed = None):
    if seed is not None:
        np.random.seed(seed)
    
    #########################################
    ### Generate LP (inequality constrained)
    #########################################
    (c, A, b, opt_x, opt_y) = generate_problem(m, n, cond, seed)
    f = QuadFunc(
        np.zeros((n, n)),
        c
    )
    x_qpsolve = opt_x
    problem = Problem(
        f.eval,
        f.grad,
        None,
        None,
        A,
        b
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
        sigma = 10,
        tol = 1e-6,
        inner_solver=InnerSolver.BFGS_SCIPY,
        logging = False,
        hess_f = f.hess,
        norm2 = False
    )

    x_alm = alm_power.solve(np.copy(x0), f_ref = f.eval(x_qpsolve), inner_tol=lambda i: 1e-3 / np.power(i, q))

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
        sigma = 1.,
        tol = 1e-6,
        inner_solver=InnerSolver.BFGS_SCIPY,
        logging = False,
        hess_f = f.hess,
        norm2 = False
    )

    x_alm = alm_power.solve(np.copy(x0), f_ref = f.eval(x_qpsolve), inner_tol=lambda i: 1e-1 / np.power(i, q))

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
        sigma = 1000,
        tol = 1e-6,
        inner_solver=InnerSolver.BFGS_SCIPY,
        logging = False,
        hess_f = f.hess,
        should_update_sigma=False
    )

    x_alm = alm_classical.solve(np.copy(x0), f_ref = f.eval(x_qpsolve), inner_tol = lambda i: 1e-3 / np.power(i, q))

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
        sigma = 250,
        tol = 1e-6,
        inner_solver=InnerSolver.BFGS_SCIPY,
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
mn_list = [(200, 100), (400, 200), (600, 300), (300, 100), (600, 200), (400, 100), (500, 100), (600, 100),]
mn_iters = np.zeros((len(mn_list), 2 * n_solvers))

for mn_i, mn in enumerate(mn_list):
    m, n = mn
    lbfgsb_iters = np.zeros((N, n_solvers))
    for i in range(N):
        lbfgsb_iters[i, :] = run(m, n, cond=100)
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
np.savetxt(log_path + '/lp_bfgs_iters.csv', mn_iters)

print(mn_iters)