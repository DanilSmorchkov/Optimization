import numpy as np
import scipy
from scipy.special import expit
from scipy.sparse import diags


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


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        m = self.b.shape[0]
        return 1 / m * np.ones(m) @  np.logaddexp(0, -self.b * self.matvec_Ax(x)) + self.regcoef / 2 * (x @ x)

    def grad(self, x):
        m = self.b.shape[0]
        return - 1 / m * self.matvec_ATx(expit(-self.b * self.matvec_Ax(x)).T * self.b) + self.regcoef * x

    def hess(self, x):
        m = self.b.shape[0]
        return 1 / m * self.matmat_ATsA(diags(expit(-self.b * self.matvec_Ax(x)) * (1 - expit(-self.b * self.matvec_Ax(x))))) + self.regcoef * np.identity(x.shape[0])


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A @ x
    matvec_ATx = lambda x: A.T @ x

    def matmat_ATsA(s):
        return A.T @ s @ A

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    result = np.apply_along_axis(func1d=func, arr=x + eps * np.eye(x.shape[0]), axis=1)
    result = (result - func(x)) / eps
    return result


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """

    arr1 = np.tile(eps * np.eye(x.shape[0]), reps=(1, 1, x.shape[0])).reshape(x.shape[0], x.shape[0], x.shape[0])
    arr2 = np.tile(eps * np.eye(x.shape[0]), reps=(1, x.shape[0], 1)).reshape(x.shape[0], x.shape[0], x.shape[0])
    arr = arr1 + arr2 + x.reshape(1, 1, x.shape[0])
    func1 = np.apply_along_axis(func1d=func, arr=arr, axis=2)
    arr3 = (x + eps * np.eye(x.shape[0]))
    func2 = np.apply_along_axis(func1d=func, arr=arr3, axis=1)
    arr4 = (x.reshape(x.shape[0], 1) + eps * np.eye(x.shape[0]))
    func3 = np.apply_along_axis(func1d=func, arr=arr4, axis=0).reshape(x.shape[0], 1)
    result = (func1 - func2 - func3 + func(x)) / (eps ** 2)

    # Я нашел другую формулу для вычисления приближения гессиана, в некоторых случаях она дает более точный ответ
    # arr1 = np.tile(eps * np.eye(x.shape[0]), reps=(1, 1, x.shape[0])).reshape(x.shape[0], x.shape[0], x.shape[0])
    # arr2 = np.tile(eps * np.eye(x.shape[0]), reps=(1, x.shape[0], 1)).reshape(x.shape[0], x.shape[0], x.shape[0])
    # arr = arr1 + arr2 + x.reshape(1, 1, x.shape[0])
    # func1 = np.apply_along_axis(func1d=func, arr=arr, axis=2)
    # arr3 = arr1 - arr2 + x.reshape(1, 1, x.shape[0])
    # func2 = np.apply_along_axis(func1d=func, arr=arr3, axis=2)
    # arr4 = -arr1 + arr2 + x.reshape(1, 1, x.shape[0])
    # func3 = np.apply_along_axis(func1d=func, arr=arr4, axis=2)
    # arr5 = -arr1 - arr2 + x.reshape(1, 1, x.shape[0])
    # func4 = np.apply_along_axis(func1d=func, arr=arr5, axis=2)
    # result = (func1 - func2 - func3 + func4) / (4 * eps ** 2)
    return result
