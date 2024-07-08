import numpy as np

from power_alm.inner_solver import *
from power_alm.alm import *

class SolverParameters:
    def __init__(self, inner_solver_params, maxit = 500, tol = 1e-5, p = None, norm2 = None, penalty = None, c_tol = 1e-3, adaptive_penalty = False, adaptive_penalty_tightness = 0.1):
        self.maxit = maxit
        self.tol = tol
        self.p = p
        self.inner_solver_params = inner_solver_params
        self.norm2 = norm2
        self.penalty = penalty
        self.c_tol = c_tol
        self.adaptive_penalty = adaptive_penalty
        self.adaptive_penalty_tightness = adaptive_penalty_tightness

class Solver:
    def __init__(self, composite_alm_problem, params, callback=None, inner_solver = UniversalFastPGMLan):

        self.params = params
        self.composite_alm_problem = composite_alm_problem
        self.inner_solver = inner_solver
        self.callback = callback

        self.x_init = np.copy(composite_alm_problem.x_init)

        # Initialize x
        self.x = np.zeros_like(composite_alm_problem.x_init)
        self.x[:] = composite_alm_problem.x_init[:]

        # Inner tolerances
        self.min_inner_tol = self.params.inner_solver_params.tol
        self.c_tol = self.params.c_tol

        # Initialize multipliers
        self.composite_alm_problem.diffable.initialize_multipliers()

        # Use the p provided by the parameters
        if self.params.p is not None:
            self.composite_alm_problem.diffable.update_power(self.params.p)

        # Update the norm2 provided by the parameters
        if self.params.norm2 is not None:
            self.composite_alm_problem.diffable._norm2 = self.params.norm2

        # Update penalty provided by the parameters
        if self.params.penalty is not None:
            self.composite_alm_problem.diffable.penalty = self.params.penalty
        else:
            self.params.penalty = self.composite_alm_problem.diffable.penalty
        if self.params.adaptive_penalty:
            self.adaptive_penalty_e = 0.1 * np.linalg.norm(self.composite_alm_problem.diffable.adaptive_penalty_eval_res(self.x_init))

    def run(self):
        x = self.x

        nit = 0

        self.callback(0., nit, x)

        for k in range(1, self.params.maxit + 1):
            # self.callback(k, nit, x)

            # Primal update
            self.params.inner_solver_params.tol = np.maximum(self.c_tol / np.power(k, self.composite_alm_problem.diffable._q), self.min_inner_tol)
            self.composite_alm_problem.x_init[:] = x[:]
            optimizer = self.inner_solver(self.params.inner_solver_params, self.composite_alm_problem)
            nit += optimizer.run()
            x[:] = optimizer.x[:]

            # Dual update  
            self.composite_alm_problem.diffable.update_multipliers(x)

            # Update penalty adaptively here
            if self.params.adaptive_penalty:
                r = np.linalg.norm(self.composite_alm_problem.diffable.adaptive_penalty_eval_res(x))
                if r > self.adaptive_penalty_e:
                    self.composite_alm_problem.diffable.penalty = np.minimum(10. * self.params.penalty, 2. * self.composite_alm_problem.diffable.penalty)
                else:
                    self.adaptive_penalty_e *= self.params.adaptive_penalty_tightness * r

            # if res <= self.params.tol:
            #     break

            if self.callback(k, nit, x):
                break
        
        self.post_process()

    def post_process(self):
        self.composite_alm_problem.x_init[:] = self.x_init[:]