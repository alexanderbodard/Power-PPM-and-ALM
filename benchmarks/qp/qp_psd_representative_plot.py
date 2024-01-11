import sys
import os
relative_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(relative_path + "/../../src")

import numpy as np
import random

import Problems.linear_problem_generator as gen
from Problems.problems import *
from alm import InnerSolver, PowerALM

results_dir = relative_path + '/results/'

def make_plots(logs, plot_path, block_final = False):
    should_block = False

    from matplotlib import pyplot as plt
    outer_iters = {}
    total_nits = {}
    grad_al = {}
    f_error = {}
    const_viol = {}
    ps_per_inner_solver = {}
    sigmas_per_inner_solver = {}
    inner_solvers = []
    ps = []
    sigmas = []

    for outer_iter_log in logs:
        # print(outer_iter_log)
        # print(logs)
        inner_solver = outer_iter_log['inner_solver']
        p = outer_iter_log['p']
        sigma = outer_iter_log['sigma']

        if inner_solver not in inner_solvers:
            inner_solvers.append(inner_solver)
            outer_iters[inner_solver] = []
            total_nits[inner_solver] = []
            grad_al[inner_solver] = []
            f_error[inner_solver] = []
            const_viol[inner_solver] = []
        if p not in ps:
            ps.append(p)
        if sigma not in sigmas:
            sigmas.append(sigma)
        
        ps_per_inner_solver[inner_solver] = p
        sigmas_per_inner_solver[inner_solver] = sigma

        outer_iters[inner_solver].append(outer_iter_log['outer_iter'])
        total_nits[inner_solver].append(outer_iter_log['total_nit'])
        grad_al[inner_solver].append(outer_iter_log['grad_al'])
        f_error[inner_solver].append(outer_iter_log['f - f_ref'])
        const_viol[inner_solver].append(outer_iter_log['const_viol'])

    colormap = plt.cm.nipy_spectral
    N = len(inner_solvers)
    linestyles = ["-", "--", ":", "-."]
    markers = ['o', 'x', '+', '*']

    # plt.figure()
    # for i, inner_solver in enumerate(inner_solvers):
    #     p_index = ps.index(ps_per_inner_solver[inner_solver])
    #     sigma_index = sigmas.index(sigmas_per_inner_solver[inner_solver])
    #     plt.plot(outer_iters[inner_solver], total_nits[inner_solver], marker = markers[sigma_index], linestyle=linestyles[p_index], label=inner_solver, color = colormap(1. * i / N))
    # plt.xlabel('# outer iterations')
    # plt.ylabel('total # of iterations')
    # plt.legend()
    # plt.savefig(plot_path + "/total_outer.pdf")
    # plt.show(block=should_block)

    # plt.figure()
    # for i, inner_solver in enumerate(inner_solvers):
    #     p_index = ps.index(ps_per_inner_solver[inner_solver])
    #     sigma_index = sigmas.index(sigmas_per_inner_solver[inner_solver])
    #     plt.semilogy(outer_iters[inner_solver], f_error[inner_solver], marker = markers[sigma_index], linestyle=linestyles[p_index], label=inner_solver, color = colormap(1. * i / N))
    # plt.xlabel('# outer iterations')
    # plt.ylabel(r'$\vert f - f^* \vert$')
    # plt.legend()
    # plt.savefig(plot_path + "/ferror_outer.pdf")
    # plt.show(block=should_block)

    # plt.figure()
    # for i, inner_solver in enumerate(inner_solvers):
    #     p_index = ps.index(ps_per_inner_solver[inner_solver])
    #     sigma_index = sigmas.index(sigmas_per_inner_solver[inner_solver])
    #     plt.semilogy(outer_iters[inner_solver], const_viol[inner_solver], marker = markers[sigma_index], linestyle=linestyles[p_index], label=inner_solver, color = colormap(1. * i / N))
    # plt.xlabel('# outer iterations')
    # plt.ylabel('Constraint violation')
    # plt.legend()
    # plt.savefig(plot_path + "/constviol_outer.pdf")
    # plt.show(block=should_block)

    # plt.figure()
    # for i, inner_solver in enumerate(inner_solvers):
    #     p_index = ps.index(ps_per_inner_solver[inner_solver])
    #     sigma_index = sigmas.index(sigmas_per_inner_solver[inner_solver])
    #     plt.semilogy(total_nits[inner_solver], grad_al[inner_solver], marker = markers[sigma_index], linestyle=linestyles[p_index], label=inner_solver, color = colormap(1. * i / N))
    # plt.xlabel('total # of iterations')
    # plt.ylabel(r'$\nabla L(x, y, \eta)$')
    # plt.legend()
    # plt.savefig(plot_path + "/gradal_total.pdf")
    # plt.show(block=should_block)

    plt.figure()
    for i, inner_solver in enumerate(inner_solvers):
        p_index = ps.index(ps_per_inner_solver[inner_solver])
        sigma_index = sigmas.index(sigmas_per_inner_solver[inner_solver])
        plt.semilogy(total_nits[inner_solver], f_error[inner_solver], marker = markers[sigma_index], linestyle=linestyles[p_index], label=inner_solver, color = colormap(1. * i / N))
    plt.xlabel('# LBFGS inner iterations')
    plt.ylabel(r'$\vert f - f^* \vert$')
    plt.legend()
    plt.savefig(plot_path + "/ferror_total.pdf")
    plt.show(block=should_block)

    plt.figure()
    for i, inner_solver in enumerate(inner_solvers):
        p_index = ps.index(ps_per_inner_solver[inner_solver])
        sigma_index = sigmas.index(sigmas_per_inner_solver[inner_solver])
        plt.semilogy(total_nits[inner_solver], const_viol[inner_solver], marker = markers[sigma_index], linestyle=linestyles[p_index], label=inner_solver, color = colormap(1. * i / N))
    plt.xlabel('# LBFGS inner iterations')
    plt.ylabel('Constraint violation')
    plt.legend()
    plt.savefig(plot_path + "/constviol_total.pdf")
    plt.show(block=block_final)

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

    logs = []

    if seed is not None:
        np.random.seed(seed)


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
        logging = True,
        hess_f = f.hess,
        norm2 = True
    )

    x_alm = alm_power.solve(np.copy(x0), f_ref = f.eval(x_qpsolve), inner_tol=lambda i: 1e1 / np.power(i, q))

    print(np.abs(f.eval(x_alm) - f.eval(x_qpsolve)))
    print(np.linalg.norm(x_alm - x_qpsolve))

    logs.extend(*[alm_power.logs])

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
        logging = True,
        hess_f = f.hess,
        norm2 = True
    )

    x_alm = alm_power.solve(np.copy(x0), f_ref = f.eval(x_qpsolve), inner_tol=lambda i: 1e2 / np.power(i, q))

    print(np.abs(f.eval(x_alm) - f.eval(x_qpsolve)))
    print(np.linalg.norm(x_alm - x_qpsolve))

    # logs.extend(*[alm_power.logs])

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
        logging = True,
        hess_f = f.hess,
        should_update_sigma=False
    )

    x_alm = alm_classical.solve(np.copy(x0), f_ref = f.eval(x_qpsolve), inner_tol = lambda i: 1e-2 / np.power(i, q))

    print(np.abs(f.eval(x_alm) - f.eval(x_qpsolve)))
    print(np.linalg.norm(x_alm - x_qpsolve))

    logs.extend(*[alm_classical.logs])

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
        logging = True,
        hess_f = f.hess,
        should_update_sigma=True
    )

    x_alm = alm_classical.solve(np.copy(x0), f_ref = f.eval(x_qpsolve), inner_tol = lambda i: 1e-2 / np.power(i, q))

    print(np.abs(f.eval(x_alm) - f.eval(x_qpsolve)))
    print(np.linalg.norm(x_alm - x_qpsolve))

    logs.extend(*[alm_classical.logs])

    ###################################
    ### Dump the logs
    ###################################
    print('Dumping logs...')
    log_path = f'{relative_path}/results/{m}_{n}_{seed}'

    import os
    from datetime import datetime
    log_path = datetime.now().strftime(f'{log_path}_%H_%M_%S_%d_%m_%Y')
    os.makedirs(log_path)
    log_path = log_path

    file = open(log_path + '/data.log', 'wb')

    import pickle
    pickle.dump(
        logs, 
        file
    )

    file.close()

    make_plots(logs, log_path, block_final=True)

seed = 1
m = 400
n = 800
run(m, n, seed=seed)