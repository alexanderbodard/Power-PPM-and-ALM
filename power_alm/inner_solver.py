from abc import ABC, abstractmethod
import numpy as np

import scipy


class OptimizationProblem(ABC):
    def __init__(self, x_init):
        self.x_init = x_init

    @abstractmethod
    def eval_objective(self, x):
        pass


class CompositeOptimizationProblem(OptimizationProblem):
    def __init__(self, x_init, diffable, proxable):
        super().__init__(x_init)
        self.diffable = diffable
        self.proxable = proxable

    def eval_objective(self, x):
        return self.diffable.eval(x) + self.proxable.eval(x)

class DiffableOptimizationProblem(OptimizationProblem):
    def __init__(self, x_init, diffable):
        super().__init__(x_init)
        self.diffable = diffable

    def eval_objective(self, x):
        return self.diffable.eval(x)

class Parameters:
    def __init__(self, pi = 1., epsilon = 1e-12, gamma_init = 1., maxit = 500, tol = 1e-5, initialization_procedure = 1
                 ,Gamma_init = 1., alpha = 0., Wolfe = True, mem = 200, sigma = 1e-4, eta = 0.9):
        self.maxit = maxit
        self.tol = tol
        self.gamma_init = gamma_init
        self.epsilon = epsilon
        self.pi = pi
        self.initialization_procedure = initialization_procedure
        self.Gamma_init = Gamma_init
        self.alpha = alpha

        # Quasi-Newton stuff
        self.mem = mem
        self.Wolfe = Wolfe
        self.sigma = sigma
        self.eta = eta




class Optimizer(ABC):
    def __init__(self, params, problem, callback = None):
        self.params = params
        self.problem = problem
        self.callback = callback

    @abstractmethod
    def run(self):
        pass

class LineSearchDescentMethodBaseClass(Optimizer):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)
        self.iter = 0
        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

    @abstractmethod
    def get_descent_direct(self):
        pass

    @abstractmethod
    def post_update(self, x):
        pass

    def run(self):
        sigma = self.params.sigma
        eta = self.params.eta

        nit = 0

        cum_num_backtracks = 0
        gamma = self.params.gamma_init
        for k in range(self.params.maxit):
            nit += 1
            fx, self.grad, d = self.get_descent_direct()

            res = np.linalg.norm(self.grad)

            # if self.callback(k, cum_num_backtracks, gamma, self.x, res):
            #     break

            if res <= self.params.tol:
                break

            direc_deriv = np.dot(self.grad, d)

            gamma = self.params.gamma_init
            gamma_low = 0
            gamma_high = np.inf
            while True:
                cum_num_backtracks += 1

                fx_plus = self.problem.diffable.eval(self.x + gamma * d)
                grad_plus = self.problem.diffable.eval_gradient(self.x + gamma * d)

                if fx_plus > fx + sigma * gamma * direc_deriv + self.params.epsilon:
                    gamma_high = gamma
                    gamma = 0.5 * (gamma_low + gamma_high)
                elif self.params.Wolfe and np.dot(grad_plus, d) < eta * np.dot(self.grad, d) - self.params.epsilon:
                    gamma_low = gamma
                    if gamma_high == np.inf:
                        gamma = 2 * gamma_low
                    else:
                        gamma = 0.5 * (gamma_low + gamma_high)
                else:
                    break

            x = self.x + gamma * d

            self.post_update(x)
            self.iter = self.iter + 1
            self.x[:] = x[:]

        return nit

class LineSearchGradientDescent(LineSearchDescentMethodBaseClass):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

    def get_descent_direct(self):
        fx = self.problem.diffable.eval(self.x)
        grad = self.problem.diffable.eval_gradient(self.x)
        d = -grad
        return fx, grad, d

    def post_update(self, x):
        pass


class BFGS(LineSearchDescentMethodBaseClass):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.H = np.identity(self.x.shape[0])

    def get_descent_direct(self):
        fx = self.problem.diffable.eval(self.x)
        grad = self.problem.diffable.eval_gradient(self.x)
        d = -np.dot(self.H, grad)
        return fx, grad, d

    def post_update(self, x):
        s = x - self.x
        y = self.problem.diffable.eval_gradient(x) - self.grad

        rho = 1 / np.dot(y, s)

        V = np.identity(self.x.shape[0]) - rho * np.outer(y, s)
        self.H = np.dot(np.dot(V.T, self.H), V) + rho * np.outer(s, s)


class LBFGS(LineSearchDescentMethodBaseClass):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.S = []
        self.Y = []

    def get_descent_direct(self):
        fx = self.problem.diffable.eval(self.x)
        grad = self.problem.diffable.eval_gradient(self.x)

        q = grad

        alpha = [0.] * len(self.S)
        rho = [0.] * len(self.S)
        for j in reversed(range(len(self.S))):
            rho[j] = 1 / np.dot(self.Y[j], self.S[j])
            alpha[j] = rho[j] * np.dot(self.S[j], q)

            q = q - alpha[j] * self.Y[j]

        if self.iter > len(self.S):
            H = (np.dot(self.S[-1], self.Y[-1]) / np.dot(self.Y[-1], self.Y[-1])) * np.identity(self.x.shape[0])
        else:
            H = np.identity(self.x.shape[0])

        d = np.dot(H, q)
        for j in range(len(self.S)):
            beta = rho[j] * np.dot(self.Y[j], d)
            d = d + (alpha[j] - beta) * self.S[j]

        return fx, grad, -d

    def post_update(self, x):
        if len(self.S) >= self.params.mem:
            self.S.pop(0)
            self.Y.pop(0)

        self.S.append(x - self.x)
        self.Y.append(self.problem.diffable.eval_gradient(x) - self.grad)

##
# Composite optimizers
##
class FISTA(Optimizer):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

    def run(self):
        gamma = 1 / self.problem.diffable.get_Lip_gradient()

        y = np.copy(self.x)
        t = 1.
        nit = 0
        for k in range(self.params.maxit):
            nit += 1
            grad = self.problem.diffable.eval_gradient(y)
            res = np.linalg.norm(grad)

            # if self.callback(k, 0, gamma, self.x, res):
            #     break

            if res <= self.params.tol:
                break

            x = self.problem.proxable.eval_prox(y - gamma * grad, gamma)
            t_plus = (1 + np.sqrt(1 + 4 * t * t)) / 2

            y = x + ((t - 1) / t_plus) * (x - self.x)
            t = t_plus
            self.x[:] = x[:]

        return nit

class BacktrackingFISTA(Optimizer):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback=None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

    def run(self):
        gamma = self.params.gamma_init

        y = np.copy(self.x)
        t = 1.
        nit = 0
        cum_num_backtracks = 0
        for k in range(self.params.maxit):
            nit += 1
            grad = self.problem.diffable.eval_gradient(y)
            res = np.linalg.norm(grad)

            # if self.callback(k, cum_num_backtracks, gamma, self.x, res):
            #     break

            if res <= self.params.tol:
                break

            val = self.problem.eval_objective(y)
            while True:
                cum_num_backtracks += 1

                z = self.problem.proxable.eval_prox(y - gamma * grad, gamma)
                if (self.problem.eval_objective(z) <=
                        val + np.dot(grad, z - y) + 0.5 / gamma * np.dot(z - y, z - y) + self.params.epsilon):
                    break

                gamma = self.params.alpha * gamma


            t_plus = (1 + np.sqrt(1 + 4 * t * t)) / 2

            y = z + ((t - 1) / t_plus) * (z - self.x)
            t = t_plus
            self.x[:] = z[:]
        
        return nit

class NesterovAcceleratedGradient(Optimizer):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback=None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

    def run(self):
        nit = 0
        v = np.copy(self.x)
        gamma = self.params.gamma_init
        cum_num_backtracks = 0
        res = np.Inf
        for k in range(self.params.maxit):
            nit += 1
            # if self.callback(k, cum_num_backtracks, gamma, self.x, res):
            #     break

            if res <= self.params.tol:
                break

            theta = 2 / (k + 1)
            y = (1 - theta) * self.x + theta * v
            fy = self.problem.eval_objective(y)
            grad = self.problem.diffable.eval_gradient(y)
            res = np.linalg.norm(grad)

            x = y - gamma * grad
            while self.problem.eval_objective(x) > fy + np.dot(grad, x - y) + (0.5 / gamma) * np.dot(x - y, x - y) + self.params.epsilon:
                gamma = 0.5 * gamma
                x = y - gamma * grad

            v = self.x + (1 / theta) * (x - self.x)

            self.x[:] = x[:]

        return nit

class ProximalGradientDescent(Optimizer):
    evals_per_iteration = 2
    evals_per_linesearch = 0

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]


    def run(self):
        gamma = 1.99 / self.problem.diffable.get_Lip_gradient()

        nit = 0
        for k in range(self.params.maxit):
            nit += 1
            grad = self.problem.diffable.eval_gradient(self.x)
            res = np.linalg.norm(grad)

            # if self.callback(k, 0, gamma, self.x, res):
            #     break

            if res <= self.params.tol:
                break

            self.x = self.problem.proxable.eval_prox(self.x - gamma * grad, gamma)
        
        return nit

class LineSearchProximalGradientDescent(Optimizer):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]


    def run(self):
        cum_num_backtracks = 0

        gamma_min = 1.99 / self.problem.diffable.get_Lip_gradient()

        if self.params.gamma_init > 0:
            gamma = self.params.gamma_init
        else:
            gamma = self.params.gamma_min

        nit = 0
        for k in range(self.params.maxit):
            nit += 1
            grad = self.problem.diffable.eval_gradient(self.x)
            res = np.linalg.norm(grad)

            # if self.callback(k, cum_num_backtracks, gamma, self.x, res):
            #     break

            if res <= self.params.tol:
                break

            value = self.problem.diffable.eval(self.x)

            while True:
                gamma = np.maximum(gamma_min, gamma)
                x_new = self.problem.proxable.eval_prox(self.x - gamma * grad, gamma)
                if gamma == gamma_min or self.params.alpha <= 0.:
                    break

                cum_num_backtracks += 1

                if self.problem.diffable.eval(x_new) <= value + np.dot(grad, x_new - self.x) + (0.5 / gamma) * np.dot(x_new - self.x, x_new - self.x) + self.params.epsilon:
                    break

                gamma = gamma * self.params.alpha

            if self.params.alpha > 0.:
                gamma = gamma / self.params.alpha
            self.x = x_new
    
        return nit


class NesterovUniversalProximalGradientMethod(Optimizer):
    evals_per_iteration = 1
    evals_per_linesearch = 1


    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]


    def run(self):
        gamma = self.params.gamma_init

        nit = 0
        cum_num_backtracks = 0
        res = np.Inf
        for k in range(self.params.maxit):
            nit += 1
            # if self.callback(k, cum_num_backtracks, gamma, self.x, res):
            #     break

            grad = self.problem.diffable.eval_gradient(self.x)
            value = self.problem.diffable.eval(self.x)

            while True:
                cum_num_backtracks = cum_num_backtracks + 1

                x = self.problem.proxable.eval_prox(self.x - gamma * grad, gamma)

                upper_bound = value + np.dot(grad, x - self.x) + 0.5 / gamma * np.dot(x - self.x, x - self.x) + self.params.epsilon / 2
                if self.problem.diffable.eval(x) <= upper_bound:
                    break

                gamma = gamma * 0.5

            res = np.linalg.norm(x - self.x, 2) / gamma
            gamma = gamma * 2


            self.x[:] = x[:]

            if res <= self.params.tol:
                break
        
        return nit


class NesterovUniversalFastProximalGradientMethod(Optimizer):
    evals_per_iteration = 3
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

        self.y = np.zeros(self.problem.x_init.shape)
        self.y[:] = self.x[:]

        self.A = 0


    def run(self):
        gamma = self.params.gamma_init

        v = np.zeros(self.problem.x_init.shape)
        v[:] = self.x[:]

        phi = np.zeros(self.problem.x_init.shape)
        theta = 0

        res = np.Inf
        nit = 0
        cum_num_backtracks = 0
        for k in range(self.params.maxit):
            nit += 1
            if not self.callback is None and self.callback(k, cum_num_backtracks, gamma, self.x, res):
                break

            while True:
                cum_num_backtracks = cum_num_backtracks + 1

                a = (gamma + np.sqrt(gamma ** 2 + 4 * gamma * self.A)) / 2
                A = self.A + a
                tau = a / A

                x = tau * v + (1 - tau) * self.y

                grad = self.problem.diffable.eval_gradient(x)
                value = self.problem.diffable.eval(x)
                x_hat = self.problem.proxable.eval_prox(v - a * grad, a)

                y = tau * x_hat + (1 - tau) * self.y

                upper_bound = (value + np.dot(grad, y - x)
                               + (0.5 / gamma) * np.dot(y - x, y - x)
                               + 0.5 * self.params.epsilon * tau)

                if self.problem.diffable.eval(y) <= upper_bound:
                    break

                gamma = gamma * 0.5

            res = np.linalg.norm(x - self.x, 2) / gamma

            gamma = gamma * 2

            self.y[:] = y[:]
            self.A = A

            phi = phi + a * grad
            theta = theta + a

            v = self.problem.proxable.eval_prox(self.problem.x_init - phi, theta)

            self.x[:] = x[:]

            if np.linalg.norm(grad) <= self.params.tol:
                break

        return nit


class AdaptiveProximalGradientMethod(Optimizer):
    evals_per_iteration = 2
    evals_per_linesearch = 0

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

        self.grad = problem.diffable.eval_gradient(self.x)

        self.gamma = self.params.gamma_init


    def run(self):
        if self.params.initialization_procedure == 0:
            x = self.problem.proxable.eval_prox(self.x - self.gamma * self.grad, self.gamma)
            gamma = self.gamma
        else:
            x_new = self.problem.proxable.eval_prox(self.x - self.params.gamma_init * self.grad, self.params.gamma_init)
            grad_x_new = self.problem.diffable.eval_gradient(x_new)
            L = np.linalg.norm(self.grad - grad_x_new) / np.linalg.norm(self.x - x_new)

            if self.params.pi - 2 * L < 0:
                self.gamma = self.params.gamma_init
            else:
                self.gamma = self.params.gamma_init * (self.params.pi * 2 * L) / (self.params.pi - 2 * L)
            gamma = self.params.gamma_init
            x = np.copy(x_new)
        res = np.Inf
        nit = 0
        for k in range(self.params.maxit):
            nit += 1
            # if self.callback(k, 0, gamma, self.x, res / gamma):
            #     break

            grad = self.problem.diffable.eval_gradient(x)
            res = np.linalg.norm(x - self.x)

            if res / gamma <= self.params.tol:
                break

            ell = np.dot(grad - self.grad, x - self.x) / res ** 2
            L = np.linalg.norm(grad - self.grad) / res

            rho = gamma / self.gamma
            alpha = np.sqrt(1 / self.params.pi + rho)
            delta = gamma ** 2 * L ** 2 - (2 - self.params.pi) * gamma * ell + 1 - self.params.pi

            if delta <= 0.:
                beta = np.Inf
            else:
                beta = 1 / np.sqrt(2 * delta)

            self.gamma = gamma

            gamma = gamma * np.minimum(alpha, beta)

            self.x[:] = x[:]
            self.grad[:] = grad[:]

            x = self.problem.proxable.eval_prox(self.x - gamma * self.grad, gamma)

        return nit


class UniversalFastPGMLan(Optimizer):
    evals_per_iteration = 2
    evals_per_linesearch = 0

    def __init__(self, params, problem, callback=None):
        super().__init__(params, problem, callback)

        # Initialize x
        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]
        self.grad = self.problem.diffable.eval_gradient(self.x)

    def compute_lipschitz_estimate(self, x, x_old, grad_x, grad_x_old, tau):
        inner_approx = self.problem.diffable.eval(x_old) - \
                       self.problem.diffable.eval(x) - np.inner(grad_x, x_old - x)
        L = pow(np.linalg.norm(grad_x - grad_x_old), 2) / (2 * inner_approx + self.params.epsilon / tau)
        return L

    def run(self):
        x = np.copy(self.x)
        y = np.copy(x)
        #tau = 0
        beta = 1 - np.sqrt(3) / 2
        #tau_old = 0
        self.grad = self.problem.diffable.eval_gradient(self.x)
        grad = np.copy(self.grad)
        L = 0
        res = 0
        gamma = self.params.gamma_init

        # Initial iterates are computed outside the main loop to avoid multiple if statements
        #  k = 1  #
        # self.callback(0, 0, gamma, x, res)
        x_new = np.copy(x)
        grad_new = np.copy(grad)
        for i in range(1, 20):  # Terminate the ill-defined line-search after 10 iterates!
            x_new = self.problem.proxable.eval_prox(x - gamma * grad, gamma)
            grad_new = self.problem.diffable.eval_gradient(x_new)
            L = (np.sqrt(pow(np.linalg.norm(x_new - x), 2)
                         * pow(np.linalg.norm(grad_new - self.grad), 2) + pow(self.params.epsilon / 4,
                                                                              2)) - self.params.epsilon / 4) \
                / pow(np.linalg.norm(x_new - x), 2)
            if beta / (4 * (1 - beta) * L) <= gamma <= 1 / (3 * L):
                break
            gamma = gamma * 0.5
        # print(i)
        self.grad[:] = grad[:]
        grad[:] = grad_new[:]
        self.x[:] = x[:]
        x[:] = x_new[:]

        #  k = 2  #
        res = np.linalg.norm(x - self.x, 2) / gamma
        # self.callback(1, 0, gamma, x, res)
        gamma = beta / (2 * L)
        self.x[:] = x[:]
        z = self.problem.proxable.eval_prox(y - gamma * grad, gamma)
        y = (1 - beta) * y + beta * z
        x = (z + 2 * x) / 3
        self.grad = np.copy(grad)
        grad = self.problem.diffable.eval_gradient(x)
        L = self.compute_lipschitz_estimate(x, self.x, grad, self.grad, 2)

        nit = 2

        tau_old = 0
        tau = 2
        #  Recursion starts after k = 3  #
        for k in range(3, self.params.maxit + 1):
            nit += 1
            # if self.callback(k - 1, 0, gamma, x, res):
            #     break

            # Store the old values of tau
            tau_prev = tau_old
            tau_old = tau

            # Compute tau and beta
            tau = tau_old + self.params.alpha / 2 + 2 * gamma * L * (1 - self.params.alpha) / (
                                  beta * tau_old)

            # Compute gamma:
            gamma = min(np.abs(beta * tau_old / (4 * L)), ((tau_prev + 1) / tau_old) * gamma)
            # Main part of the algorithm
            self.x[:] = x[:]
            z = self.problem.proxable.eval_prox(y - gamma * grad, gamma)
            y = (1 - beta) * y + beta * z
            x = (z + tau * x) / (1 + tau)
            self.grad[:] = grad[:]
            grad = self.problem.diffable.eval_gradient(x)
            res = np.linalg.norm(x - self.x, 2) / gamma

            # Compute estimates
            L = self.compute_lipschitz_estimate(x, self.x, grad, self.grad, tau)

            if res <= self.params.tol:
                break

            if np.linalg.norm(grad) <= self.params.tol:
                break
        return nit

class LBFGS_SCIPY(Optimizer):
    def __init__(self, params, problem, callback=None):
        super().__init__(params, problem, callback)

        # Initialize x
        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

    def run(self):
        lbfgs_scipy_tol = 1e-50
        gtol = lbfgs_scipy_tol if self.params.tol is None else self.params.tol
        options = {
            "ftol": lbfgs_scipy_tol,
            "gtol": gtol,
            "maxiter": self.params.maxit, 
            "eps": 1e-16,
        }
        res = scipy.optimize.minimize(
            self.problem.eval_objective,
            self.x, 
            method='L-BFGS-B',
            bounds=np.array(list(zip(self.problem.diffable.get_lb(), self.problem.diffable.get_ub()))),
            jac=self.problem.diffable.eval_gradient,
            options=options
        )

        self.x[:] = res['x'][:]

        return res['nit']