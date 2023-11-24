import numpy as np
import scipy
from enum import Enum

InnerSolver = Enum('InnerSolver', [
    'LBFGS_SCIPY',
    'BFGS_SCIPY',
])

class PowerALM:
    def __init__(
        self,
        problem,
        p,
        m,
        n,
        maxit = 150,
        sigma = 100.,
        tol = 1e-8,
        inner_solver = InnerSolver.LBFGS_SCIPY,
        logging = False,
        log_path = None,
        hess_f = None,
        norm2 = False,
        should_update_sigma = False,
    ):
        # Problem
        self.problem = problem

        # Norms to use
        self.p = p
        self.q = p / (p - 1.)
        self.norm2 = norm2

        # ALM settings
        self.maxit = maxit
        self.sigma = sigma
        self.should_update_sigma = should_update_sigma

        # Primal and dual vector sizes
        self.m = m
        self.n = n

        # Tolerance
        self.tol = tol

        # Inner solver
        self.inner_solver = inner_solver

        # Logging
        self.logging = logging
        self.logs = []
        self.log_path = log_path
        if log_path != None:
            import os
            from datetime import datetime
            log_path = datetime.now().strftime(f'{log_path}_%H_%M_%S_%d_%m_%Y')
            os.makedirs(log_path)
            self.log_path = log_path

        self.hess_f = hess_f
        self.total_nit = 0

    """
    \varphi(x) = 1 / p * \sum_{i}^n \vert x_i \vert^p
    """
    def primal_norm_pow(self, x):
        if self.norm2:
            return np.power(np.sqrt(np.sum(np.square(x))), self.p) / (self.p)
        
        return np.sum(np.power(np.abs(x), self.p)) / self.p
    
    def primal_grad_norm_pow(self, x):
        if self.norm2:
            if np.sqrt(np.sum(np.square(x))) == 0.:
                    return np.zeros_like(x)
            return np.power(np.sqrt(np.sum(np.square(x))), self.p-2) * x
        
        return np.sign(x) * np.power(np.abs(x), self.p - 1)
    
    def primal_hess_norm_pow(self, x):
        if self.norm2:
            raise RuntimeError("Not implemented")
        
        xbar = np.array([np.power(np.abs(xx), self.p - 2) if xx != 0. else 1. for xx in x])
        return (self.p - 1) * np.diag(xbar)

    def dual_norm_pow(self, x):
        if self.norm2:
            return np.power(np.sqrt(np.sum(np.square(x))), self.q) / (self.q)
        return np.sum(np.power(np.abs(x), self.q)) / self.q
    
    def dual_grad_norm_pow(self, x):
        if self.norm2:
            if np.sqrt(np.sum(np.square(x))) == 0.:
                    return np.zeros_like(x)
            return np.power(np.sqrt(np.sum(np.square(x))), self.q-2) * x
        return np.sign(x) * np.power(np.abs(x), self.q - 1)
    
    def dual_hess_norm_pow(self, x):
        if self.norm2:
            raise RuntimeError("Not implemented")
        
        xbar = np.array([np.power(np.abs(xx), self.q - 2) if xx != 0. else 1e8 for xx in x])
        return (self.q - 1) * np.diag(xbar)
    
    def p_norm_pow(self, x, p, W):
        return np.sum(W * np.power(np.abs(x), p)) / p
    
    def p_grad_norm_pow(self, x, p, W):
        return np.sign(x) * W * np.power(np.abs(x), p - 1)
    
    def p_hess_norm_pow(self, x, p, W):
        xbar = np.array([np.power(np.abs(xx), p - 2) if xx != 0. else 1e8 for xx in x])
        return (p - 1) * np.diag(W * xbar) 

    def al(self, x, y, eta):
        v = np.maximum(0., eta + self.primal_grad_norm_pow(self.problem.ineq_res(x)) * self.sigma)

        return self.problem.fun(x) \
            + np.dot(y, self.problem.eq_res(x)) \
            + self.primal_norm_pow(self.problem.eq_res(x)) * self.sigma\
            + np.dot(v, self.problem.ineq_res(x)) \
            - np.power(1. / self.sigma, self.q-1) * self.dual_norm_pow(v - eta)
        
    def grad_al(self, x, y, eta):
        v = np.maximum(0., eta + self.primal_grad_norm_pow(self.problem.ineq_res(x)) * self.sigma)

        if not (self.problem.A is None) and not (self.problem.G is None):
            # Equality and inequality constraints
            return self.problem.grad_fun(x)\
                + self.problem.A.T @ y \
                + self.problem.A.T @ self.primal_grad_norm_pow(self.problem.eq_res(x)) * self.sigma\
                + self.problem.G.T @ v
        elif not(self.problem.A is None):
            # Only equality constraints
            return self.problem.grad_fun(x)\
                + self.problem.A.T @ y \
                + self.problem.A.T @ self.primal_grad_norm_pow(self.problem.eq_res(x)) * self.sigma
        elif not(self.problem.G is None):            
            # Only inequality constraints
            return self.problem.grad_fun(x)\
                + self.problem.G.T @ v


    def hess_al(self, x, y, eta):
        if not (self.problem.A is None) and not (self.problem.G is None):
            # Equality and inequality constraints
            constr_selec = eta + self.primal_grad_norm_pow(self.problem.ineq_res(x)) * self.sigma
            Lmod = np.diag(self.primal_hess_norm_pow(self.problem.ineq_res(x)))
            Lmod = np.array([Lmod[i] if l > 0. else 0. for i, l in enumerate(constr_selec)])
            return self.hess_f(x) + \
                self.problem.A.T @ self.primal_hess_norm_pow(self.problem.eq_res(x)) @ self.problem.A * self.sigma +\
                self.problem.G.T @ np.diag(Lmod) @ self.problem.G * self.sigma
        elif not(self.problem.A is None):
            # Only equality constraints
            return self.hess_f(x) + \
                self.problem.A.T @ self.primal_hess_norm_pow(self.problem.eq_res(x)) @ self.problem.A * self.sigma
        elif not(self.problem.G is None):            
            # Only inequality constraints
            constr_selec = eta + self.primal_grad_norm_pow(self.problem.ineq_res(x)) * self.sigma
            Lmod = np.diag(self.primal_hess_norm_pow(self.problem.ineq_res(x)))
            Lmod = np.array([Lmod[i] if l > 0. else 0. for i, l in enumerate(constr_selec)])
            return self.hess_f(x) + \
                self.problem.G.T @ np.diag(Lmod) @ self.problem.G * self.sigma

    def update_sigma(self, x, ek):
        if self.problem.A is not None and np.linalg.norm(np.maximum(0., self.problem.eq_res(x))) < ek:
            ek *= .5
            self.sigma *= 2.
        elif self.problem.G is not None and np.linalg.norm(np.maximum(0., self.problem.ineq_res(x))) < ek:
            ek *= .5
            self.sigma *= 2.
            
        return ek

    def solve(self, x0, f_ref=None, verbose=True, inner_tol = None):
        x = np.copy(x0)
        y = 0.0 if self.problem.A is None else np.zeros(self.m)
        eta = 0.0 if self.problem.G is None else np.zeros(self.m)
        ek = 1.

        if self.logging:
            self.logs.append({
                'outer_iter': 0,
                'total_nit': 0,
                'grad_al': np.linalg.norm(self.grad_al(x, y, eta)),
                'f - f_ref': np.abs(self.problem.fun(x) - f_ref),
                'inner_solver': f'{self.inner_solver.name} (p = {self.p:.2f}, sigma = {"Adaptive" if self.should_update_sigma else self.sigma})',
                'p': self.p,
                'sigma': self.sigma,
                'eq_const': self.problem.A is not None,
                'ineq_const': self.problem.G is not None,
                'const_viol': np.linalg.norm(self.problem.eq_res(x)) + np.linalg.norm(np.maximum(0., self.problem.ineq_res(x)))
            })

        total_nit = 0
        self.total_nit = total_nit
        for i in range(self.maxit):
            if self.inner_solver == InnerSolver.LBFGS_SCIPY:
                lbfgs_scipy_tol = 1e-50
                gtol = lbfgs_scipy_tol if inner_tol is None else inner_tol(i+1)
                options = {
                    "ftol": lbfgs_scipy_tol,
                    "gtol": gtol,#100 / np.power(i+1, self.q),
                    "maxiter": 500, 
                    "eps": 1e-16,
                    # "maxcor": 30,
                }
                res = scipy.optimize.minimize(
                    lambda x: self.al(x, y, eta),
                    x, 
                    method='L-BFGS-B',
                    bounds=np.array(((self.problem.lb, self.problem.ub),)*self.n),
                    jac=lambda x: self.grad_al(x, y, eta),
                    options=options
                )
                x, nit = res['x'], res['nit']
            elif self.inner_solver == InnerSolver.BFGS_SCIPY:
                if (not np.isneginf(self.problem.lb)) or (not np.isposinf(self.problem.ub)):
                    raise RuntimeError(f'Attempting to use inner solver {self.inner_solver}, with lower / upper bounds.')
                
                # bfgs_scipy_tol = 100. / np.power(i+1, self.q - 1)
                bfgs_scipy_tol = lbfgs_scipy_tol if inner_tol is None else inner_tol(i+1)
                options = {
                    "gtol": bfgs_scipy_tol,
                    "maxiter": 500, 
                    "eps": 1e-16
                }
                res = scipy.optimize.minimize(
                    lambda x: self.al(x, y, eta),
                    x, 
                    method='BFGS',
                    jac=lambda x: self.grad_al(x, y, eta),
                    options=options
                )
                x, nit = res['x'], res['nit']
            else:
                raise ValueError(
                    f"inner_solver = {self.inner_solver} is not supported."
                )

            y = y + self.primal_grad_norm_pow(self.problem.eq_res(x)) * self.sigma
            eta = np.maximum(0., eta + self.primal_grad_norm_pow(self.problem.ineq_res(x)) * self.sigma)

            if self.should_update_sigma:
                ek = self.update_sigma(x, ek)

            total_nit += nit
            self.total_nit = total_nit

            if self.logging:
                self.logs.append({
                    'outer_iter': i + 1,
                    'total_nit': total_nit,
                    'grad_al': np.linalg.norm(self.grad_al(x, y, eta)),
                    'f - f_ref': np.abs(self.problem.fun(x) - f_ref),
                    'inner_solver': f'{self.inner_solver.name} (p = {self.p:.2f}, sigma = {"Adaptive" if self.should_update_sigma else self.sigma})',
                    'p': self.p,
                    'sigma': 'Adaptive' if self.should_update_sigma else self.sigma,
                    'eq_const': self.problem.A is not None,
                    'ineq_const': self.problem.G is not None,
                    'const_viol': np.linalg.norm(self.problem.eq_res(x)) + np.linalg.norm(np.maximum(0., self.problem.ineq_res(x)))
                })
            if verbose:
                if f_ref is not None:
                    print(i + 1, total_nit, np.linalg.norm(self.grad_al(x, y, eta)), np.linalg.norm(np.maximum(0., self.problem.eq_res(x))), np.linalg.norm(self.problem.fun(x) - f_ref))
                else :
                    print(i + 1, total_nit, np.linalg.norm(self.grad_al(x, y, eta)), np.linalg.norm(np.maximum(0., self.problem.eq_res(x))))

            if f_ref is not None and np.linalg.norm(self.problem.fun(x) - f_ref) < self.tol and np.linalg.norm(np.maximum(0., self.problem.eq_res(x))) < self.tol:
                self.post_solve()
                return x
        
        self.post_solve()
        return x
    
    def post_solve(self):
        if self.logging and self.log_path:
            file = open(self.log_path + '/data.log', 'wb')
            
            import pickle
            pickle.dump(self.logs, file)

            file.close()