import numpy as np
import scipy
from scipy.sparse import diags, bmat
from scipy.special import expit
from functools import partial


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class Barrier_Lasso_Oracle(BaseSmoothOracle):
    def __init__(self, A, b, regcoef, t):
        self.A = A
        self.b = b
        self.n = A.shape[1]
        self.regcoef = regcoef
        self.t = t

    def func(self, x_u):
        x, u = x_u[:self.n], x_u[self.n:]
        function = 1 / 2 * np.sum((self.A @ x - self.b) ** 2) + self.regcoef * u.sum()
        barrier = (np.log(u + x) + np.log(u - x)).sum()
        return self.t * function - barrier

    def grad(self, x_u):
        x, u = x_u[:self.n], x_u[self.n:]
        grad_x = self.t * self.A.T @ (self.A @ x - self.b) + 1 / (u - x) - 1 / (u + x)
        grad_u = self.t * self.regcoef * np.ones_like(u) - 1 / (u - x) - 1 / (u + x)
        grad = np.hstack((grad_x, grad_u))
        return grad

    def hess(self, x_u):
        x, u = x_u[:self.n], x_u[self.n:]
        hess_xx = self.t * self.A.T @ self.A + diags(
            1 / (x + u) ** 2 + 1 / (-x + u) ** 2, format='csc'
        )
        hess_uu = diags(
            1 / (x + u) ** 2 + 1 / (-x + u) ** 2, format='csc'
        )
        hess_xu = hess_ux = diags(
            1 / (x + u) ** 2 - 1 / (-x + u) ** 2, format='csc'
        )
        hess = bmat([[hess_xx, hess_ux], [hess_xu, hess_uu]], dtype=float, format='csc')
        return hess

    def lasso_duality_gap(self, x):
        """
        Estimates f(x) - f* via duality gap for
            f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        """
        Ax_b = self.A @ x - self.b
        ATAx_b = self.A.T @ Ax_b
        mu = min(1, self.regcoef / (max(np.abs(ATAx_b)))) * Ax_b
        estimate = 1 / 2 * Ax_b @ Ax_b + self.regcoef * np.sum(np.abs(x)) + 1 / 2 * mu @ mu + self.b @ mu
        return estimate
