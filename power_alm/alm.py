from abc import ABC, abstractmethod
import numpy as np


###
# Interfaces
###
class AugmentedLagrangian(ABC):
    def __init__(self, problem, p, penalty, norm2 = True):
        self._problem = problem
        self._q = p / (p - 1.)
        self._p = p
        self.penalty = penalty
        self._norm2 = norm2

        self.t = 1.

    def _p_norm_pow(self, x):
        if self._norm2:
            return np.power(np.sqrt(np.sum(np.square(x))), self._p) / (self._p)
        return np.sum(np.power(np.abs(x), self._p)) / self._p
    
    def _p_grad_norm_pow(self, x):
        if self._norm2:
            if np.sqrt(np.sum(np.square(x))) == 0.:
                    return np.zeros_like(x)
            return np.power(np.sqrt(np.sum(np.square(x))), self._p-2) * x
        
        return np.sign(x) * np.power(np.abs(x), self._p - 1)
    
    def _q_norm_pow(self, x):
        if self._norm2:
            return np.power(np.sqrt(np.sum(np.square(x))), self._q) / (self._q)
        return np.sum(np.power(np.abs(x), self._q)) / self._q

    def update_power(self, p):
        self._q = p / (p - 1.)
        self._p = p

    @abstractmethod
    def initialize_multipliers(self):
        pass

    @abstractmethod
    def update_multipliers(self, x):
        pass

    @abstractmethod
    def adaptive_penalty_eval_res(self, x):
        pass

###
# Implementations
###
class ConvexQuadraticAugmentedLagrangian(AugmentedLagrangian):
    def __init__(self, problem, p, penalty, ye_init, yi_init, norm2=True):
        self.ye_init = np.copy(ye_init)
        self.yi_init = np.copy(yi_init)

        self.ye = np.copy(ye_init)
        self.yi = np.copy(yi_init)

        self.ye_old = np.copy(self.ye)
        self.yi_old = np.copy(self.yi)

        super().__init__(problem, p, penalty, norm2)

    def eval(self, x):
        ye = self.ye; yi = self.yi
        res = self._problem.eval_objective(x)
        
        if self._problem.has_eq_constraints():
            res += np.dot(ye, self._problem.eval_eq_res(x)) \
                + self._p_norm_pow(self._problem.eval_eq_res(x)) * self.penalty

        if self._problem.has_ineq_constraints():
            # v = np.maximum(0., yi + self._p_grad_norm_pow(self._problem.eval_ineq_res(x)) * self.penalty)
            _, v = self.get_updated_multipliers(x, ye, yi)
            res += np.dot(v, self._problem.eval_ineq_res(x)) \
                - np.power(1. / self.penalty, self._q-1) * self._q_norm_pow(v - yi)
        
        return res

    def eval_gradient(self, x):
        ye = self.ye; yi = self.yi
        res = self._problem.eval_gradient_objective(x)

        if self._problem.has_eq_constraints():
            res += self._problem.eval_gradient_eq_res(x).T @ ye \
                + self._problem.eval_gradient_eq_res(x).T @ self._p_grad_norm_pow(self._problem.eval_eq_res(x)) * self.penalty
            
        if self._problem.has_ineq_constraints():
            # v = np.maximum(0., yi + self._p_grad_norm_pow(self._problem.eval_ineq_res(x)) * self.penalty)
            _, v = self.get_updated_multipliers(x, ye, yi)
            res += self._problem.eval_gradient_ineq_res(x).T @ v

        return res
    
    ###############################
    
    def has_eq_constraints(self):
        return self._problem.has_eq_constraints()
    
    def has_ineq_constraints(self):
        return self._problem.has_ineq_constraints()

    def eval_eq_res(self, x):
        return self._problem.eval_eq_res(x)
        
    def eval_ineq_res(self, x):
        return self._problem.eval_ineq_res(x)
    
    def eval_gradient_eq_res(self, x):
        return self._problem.eval_gradient_eq_res(x)
    
    def eval_gradient_ineq_res(self, x):
        return self._problem.eval_gradient_ineq_res(x)

    def get_lb(self):
        return self._problem.get_lb()
    
    def get_ub(self):
        return self._problem.get_ub()

    def set_ye(self, ye):
        self.ye[:] = ye[:]

    def set_yi(self, yi):
        self.yi[:] = yi[:]

    def initialize_multipliers(self):
        if self.has_eq_constraints():
            self.ye[:] = self.ye_init[:]
        if self.has_ineq_constraints():
            self.yi[:] = self.yi_init[:]

    def get_updated_multipliers(self, x, ye, yi):
        if self.has_eq_constraints():
            ye += self._p_grad_norm_pow(self.eval_eq_res(x)) * self.penalty
        if self.has_ineq_constraints():
            # yi = np.maximum(0., yi + self._p_grad_norm_pow(self.eval_ineq_res(x)) * self.penalty)

            if self._norm2: # 2-norm
                prox_term = lambda t : np.maximum(yi + 1. / t * (self.eval_ineq_res(x)), 0.)

                def residual(t):
                    return t - np.power(np.linalg.norm(prox_term(t) - yi), self._q - 2) * np.power(self.penalty, 1. - self._q)

                # Determine upper bound
                t_high = self.t
                while True:
                    if residual(t_high) > 0:
                        break
                    t_high = t_high * 2

                # Determine lower bound
                t_low = self.t
                while True:
                    if residual(t_low) < 0:
                        break
                    t_low = t_low / 2

                # Bisection method
                for k in range(100):
                    self.t = t_low + (t_high - t_low) / 2
                    if residual(self.t) < 0:
                        t_low = self.t
                    elif residual(self.t) > 0:
                        t_high = self.t
                    else:
                        break

                yi = prox_term(self.t)
            else: # p-norm
                yi = np.maximum(0., yi + self._p_grad_norm_pow(self.eval_ineq_res(x)) * self.penalty)

        return ye, yi

    def update_multipliers(self, x):
        if self.has_eq_constraints():
            self.ye_old[:] = self.ye[:]
        if self.has_ineq_constraints():
            self.yi_old[:] = self.yi[:]
        self.ye, self.yi = self.get_updated_multipliers(x, self.ye, self.yi)

    def adaptive_penalty_eval_res(self, x):
        if self.has_ineq_constraints():
            return self.eval_ineq_res(x)
        else:
            return self.eval_eq_res(x)

"""
\min_x \max_{y \in [-1, 1]} 0.5 * theta * (x - c).T @ (x - c) + < y, A x >
"""
class L1RegressionAlmWrappedProblem(AugmentedLagrangian):
    def __init__(self, problem, p, penalty, y_init, norm2=True):
        self.y_init = np.copy(y_init)
        self.y = np.copy(y_init)
        self.y_old = np.copy(self.y)

        super().__init__(problem, p, penalty, norm2)

    def eval(self, x):
        v = self.get_updated_multipliers(x, self.y)

        return self._problem.eval_objective(x) + np.dot(v, self.eval_res(x)) \
            - np.power(1. / self.penalty, self._q-1) * self._q_norm_pow(v - self.y)

    def eval_gradient(self, x):
        v = self.get_updated_multipliers(x, self.y)

        return self._problem.eval_gradient_objective(x) + self._problem.A.T @ v

    def initialize_multipliers(self):
        self.y[:] = self.y_init[:]

    def get_updated_multipliers(self, x, y):
        if self._norm2: # 2-norm
            prox_term = lambda t : np.minimum(np.maximum(y + 1. / t * (self.eval_res(x)), -1.), 1.)

            def residual(t):
                return t - np.power(np.linalg.norm(prox_term(t) - y), self._q - 2) * np.power(self.penalty, 1. - self._q)

            # Determine upper bound
            t_high = self.t
            while True:
                if residual(t_high) > 0:
                    break
                t_high = t_high * 2

            # Determine lower bound
            t_low = self.t
            while True:
                if residual(t_low) < 0:
                    break
                t_low = t_low / 2

            # Bisection method
            for k in range(100):
                self.t = t_low + (t_high - t_low) / 2
                if residual(self.t) < 0:
                    t_low = self.t
                elif residual(self.t) > 0:
                    t_high = self.t
                else:
                    break

            return prox_term(self.t)
        else: # p-norm
            return np.minimum(np.maximum(self.y + self._p_grad_norm_pow(self.eval_res(x)) * self.penalty, -1.), 1.)

    def update_multipliers(self, x):
        self.y_old[:] = self.y[:]
        self.y[:] = self.get_updated_multipliers(x, self.y)

    def eval_res(self, x):
        return self._problem.A @ x - self._problem.b
    
    def adaptive_penalty_eval_res(self, x):
        return self.eval_res(x)