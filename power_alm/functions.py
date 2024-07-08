from abc import ABC, abstractmethod
import numpy as np

##
# Global variables
##
counting_enabled = True

##
# Interfaces
##
class Function(ABC):
    def __init__(self, weight = 1.):
        assert weight >= 0

        self._weight = weight

    def eval(self, x):
        return self._weight * self.eval_unweighted(x)

    @abstractmethod
    def eval_unweighted(self, x):
        pass

class Proxable(Function):
    def eval_prox(self, x, step_size, power = 2, norm = 2):
        return self.eval_prox_unweighted(x, self._weight * step_size, power, norm)

    @abstractmethod
    def eval_prox_unweighted(self, x, step_size, power = 2, norm = 2):
        pass

class Diffable(Function):
    def eval_gradient(self, x):
        return self._weight * self.eval_gradient_unweighted(x)

    @abstractmethod
    def eval_gradient_unweighted(self, x):
        pass

class Constraint(Function):
    pass

##
# Implementations
##
class Quadratic(Diffable):
    def __init__(self, Q, q, weight=1):
        super().__init__(weight)

        self._Q = Q
        self._q = q

    def eval_unweighted(self, x):
        return 0.5 * x.T @ self._Q @ x + x.T @ self._q
    
    def eval_gradient_unweighted(self, x):
        return self._Q @ x + self._q
    
class Zero(Proxable):
    def eval_unweighted(self, x):
        return 0.

    def eval_prox_unweighted(self, x, step_size, power = 2, norm = 2):
        return x
    
class IndicatorBox(Proxable):
    def __init__(self, l = -1., u = 1., weight = 1.):
        self._l = l
        self._u = u
        super().__init__(weight = weight)

    def eval_unweighted(self, x):
        if np.max(x) <= self._u and np.min(x) >= self._l:
            return 0
        return np.Inf

    def eval_prox_unweighted(self, x, _, power = 2, norm = 2):
        return np.maximum(self._l, np.minimum(self._u, x))