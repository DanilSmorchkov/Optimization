from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from scipy.sparse.linalg import spsolve
from time import time
from oracles import Barrier_Lasso_Oracle
import datetime


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased, and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    trace : bool
        If True, the progress information is appended into history dictionary
        during training. Otherwise, None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    t = t_0
    x, u = x_0, u_0
    x_u = np.hstack((x, u))
    oracle = Barrier_Lasso_Oracle(A, b, reg_coef, t)
    if trace:
        history['time'].append(0.0)
        history['func'].append(0.5 * np.sum((A @ x - b) ** 2) + reg_coef * np.sum(u))
        history['duality_gap'].append(oracle.lasso_duality_gap(x))
        if x.shape[0] <= 2:
            history['x'].append(x)

    start = time()
    for outer_i in range(max_iter):

        oracle = Barrier_Lasso_Oracle(A, b, reg_coef, t) if outer_i != 0 else oracle
        grad_0 = oracle.grad(x_u)
        for inner_i in range(max_iter_inner):
            x_u = np.hstack((x, u))
            grad_xu = oracle.grad(x_u)
            hess_xu = oracle.hess(x_u)
            d = solve(hess_xu.todense(), -grad_xu)
            id_pos = np.identity(x.size, dtype=float)
            id_neg = -np.identity(x.size, dtype=float)
            q = np.block(
                [[id_pos, id_neg],
                 [id_neg, id_neg]]
            )
            q_prod_d = q @ d
            mask = q_prod_d > 0
            if np.sum(mask) != 0:
                alpha_max = np.min((-q @ x_u)[mask] / q_prod_d[mask])
                prev_alpha = min(1, 0.99 * alpha_max)
                alpha = Armijo_alpha(oracle, x_u, d, prev_alpha, c1)
            else:
                alpha = Armijo_alpha(oracle, x_u, d, c1=c1)
            x_u += alpha * d
            if np.any(np.isinf(x_u)) or np.any(np.isnan(x_u)):
                return x, 'computational_error', history
            x, u = x_u[:x.size], x_u[x.size:]

            if grad_xu @ grad_xu <= tolerance_inner * grad_0 @ grad_0:
                print(f"Newton coveraged on {outer_i} iteration")
                break
        else:
            print(f"Newton didn't coverage on {outer_i} iteration")

        gap = oracle.lasso_duality_gap(x)
        if trace:
            history['time'].append(time() - start)
            history['func'].append(0.5 * np.sum((A @ x - b) ** 2) + reg_coef * np.sum(u))
            history['duality_gap'].append(oracle.lasso_duality_gap(x))
            if x.shape[0] <= 2:
                history['x'].append(x)

        if gap <= tolerance:
            return (x, u), 'success', history
        else:
            t *= gamma

    return (x, u), 'iterations_exceeded', history


def Armijo_alpha(oracle, xu, d, prev_alpha=1, c1=1e-4):
    alpha = prev_alpha
    while oracle.func(xu + alpha * d) > oracle.func(xu) + c1 * alpha * oracle.grad(xu) @ d:
        alpha /= 2
    return alpha


