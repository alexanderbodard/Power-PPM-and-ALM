from abc import ABC, abstractmethod
import numpy as np

import power_alm.functions as func
import power_alm.constraints as const

"""
    \minimize_{x} fun(x)
    s.t.    ce(x) = 0
    and     ci(x) <= 0
    and     lb <= x <= ub
"""
class ConstrainedProblem:
    def __init__(self, fun, ce = None, ci = None, lb = -np.inf, ub = np.inf):
        self._fun = fun
        self._ce = ce
        self._ci = ci
        self._lb = lb
        self._ub = ub

    def eval_objective(self, x):
        return self._fun.eval(x)
    
    def eval_gradient_objective(self, x):
        return self._fun.eval_gradient(x)
    
    def has_eq_constraints(self):
        return self._ce is not None
    
    def has_ineq_constraints(self):
        return self._ci is not None

    def eval_eq_res(self, x):
        assert self.has_eq_constraints()
        return self._ce.eval_res(x)
        
    def eval_ineq_res(self, x):
        assert self.has_ineq_constraints()
        return self._ci.eval_res(x)
    
    def eval_gradient_eq_res(self, x):
        assert self.has_eq_constraints()
        return self._ce.eval_gradient_res(x)
    
    def eval_gradient_ineq_res(self, x):
        assert self.has_ineq_constraints()
        return self._ci.eval_gradient_res(x)

    def get_lb(self):
        return self._lb
    
    def get_ub(self):
        return self._ub
    
"""
    \minimize_{x} 0.5 x.T Q x + q.T x
    s.t.    A x = b
    and     Gx <= h
    and     lb <= x <= ub
"""
class ConvexQuadraticProblem(ConstrainedProblem):
    def __init__(self, Q, q, A=None, b = None, G = None, h = None, lb=-np.inf, ub=np.inf):
        assert (A is None and b is None) or (A is not None and b is not None)
        assert (G is None and h is None) or (G is not None and h is not None)

        objective = func.Quadratic(Q, q)
        
        if (A is None and b is None):
            ce = None
        else:
            ce = const.Affine(A, b)

        if (G is None and h is None):
            ci = None
        else:
            ci = const.Affine(G, h)

        super().__init__(objective, ce, ci, lb, ub)

"""
    \minimize_{x} 0.5 * theta * || x ||_2^2 + || A x - b ||_1
"""
class L1RegressionProblem:
    def __init__(self, A, b, theta = 1.):
        self._fun = func.Quadratic(np.eye(A.shape[1]), np.zeros(A.shape[1]), weight=theta)
        self.A = A
        self.b = b
        self.theta = theta

    def eval_objective(self, x):
        return self._fun.eval(x)
    
    def eval_gradient_objective(self, x):
        return self._fun.eval_gradient(x)