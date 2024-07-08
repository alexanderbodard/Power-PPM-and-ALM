from abc import ABC, abstractmethod
import numpy as np

##
# Interfaces
##
class Constraint(ABC):
    def __init__(self, weight = 1.):
        assert weight >= 0

        self._weight = weight

    def eval_res(self, x):
        return self._weight * self.eval_res_unweighted(x)

    @abstractmethod
    def eval_res_unweighted(self, x):
        pass

    def eval_gradient_res(self, x):
        return self._weight * self.eval_gradient_res_unweighted(x)
    
    @abstractmethod
    def eval_gradient_res_unweighted(self, x):
        pass

##
# Implementations
##
class Affine(Constraint):
    def __init__(self, A, b = 0, weight=1):
        super().__init__(weight)

        self._A = A
        self._b = b

    def eval_res_unweighted(self, x):
        return np.dot(self._A, x) - self._b
    
    def eval_gradient_res_unweighted(self, x):
        return self._A
