import numpy as np

class Func:
    pass

class QuadFunc(Func):
    def __init__(self, Q, q):
        self.Q = Q
        self.q = q

    def eval(self, x):
        return 0.5 * x.T @ self.Q @ x + x.T @ self.q
    
    def grad(self, x):
        return self.Q @ x + self.q
    
    def hess(self, x):
        return self.Q