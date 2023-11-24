from Problems.utils.functions import *

class Problem():
    """
    \minimize_{x} fun(x)
    s.t.    A x = b
    and     Gx <= h

    grad_fun is the gradient of fun
    """
    def __init__(self, fun, grad_fun, A, b, G, h, is_quad_obj = False, is_p_norm_obj = False, f = None, lb = -np.inf, ub = +np.inf):
        self.fun = fun
        self.grad_fun = grad_fun
        self.A = A
        self.b = b
        self.G = G
        self.h = h
        self.is_quad_obj = is_quad_obj
        self.is_p_norm_obj = is_p_norm_obj
        self.f = f
        self.lb = lb
        self.ub = ub

    """
    Returns the residual of the equality constraint:
    return A x - b
    """
    def eq_res(self, x):
        if not (self.A is None) and not (self.b is None):
            return np.dot(self.A, x) - self.b
        else:
            return 0.
    
    """
    Returns the residual of the inequality constraint:
    return G x - h
    """
    def ineq_res(self, x):
        if not (self.G is None) and not (self.h is None):
            return np.dot(self.G, x) - self.h
        else:
            return 0.
    