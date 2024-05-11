import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool
from datetime import datetime
from copy import deepcopy


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    max_iter = max_iter if max_iter is not None else x_0.size
    history = defaultdict(list) if trace else None
    b_norm_sqr = np.dot(b, b)
    x_k = np.copy(x_0)
    g_k_prev = matvec(x_k) - b
    d_k = -g_k_prev
    grad_dot_prev = np.dot(g_k_prev, g_k_prev)
    if trace:
        history['time'].append(0.0)
        history['residual_norm'].append(np.sqrt(grad_dot_prev))
        if x_k.shape[0] <= 2:
            history['x'].append(x_k)

    start = datetime.now()
    for _ in range(max_iter):
        matvec_A_d_k = matvec(d_k)
        alpha_k = grad_dot_prev / (np.dot(d_k, matvec_A_d_k))
        x_k = x_k + alpha_k * d_k
        g_k_curr = g_k_prev + alpha_k * matvec_A_d_k
        grad_dot_curr = np.dot(g_k_curr, g_k_curr)

        if trace:
            history['time'].append((datetime.now() - start).total_seconds())
            history['residual_norm'].append(np.sqrt(grad_dot_curr))
            if x_k.shape[0] <= 2:
                history['x'].append(x_k)

        if grad_dot_curr <= tolerance * b_norm_sqr:
            break
        else:
            beta = grad_dot_curr / grad_dot_prev
            d_k = -g_k_curr + beta * d_k
            g_k_prev = g_k_curr
            grad_dot_prev = grad_dot_curr

    if grad_dot_curr > tolerance * b_norm_sqr:
        return x_k, 'iterations_exceeded', history
    return x_k, 'success', history


def lbfgs_multiply(v, H_deque: deque, gamma_0, max_mem, curr_mem):
    if curr_mem == max_mem:
        return gamma_0 * v
    s, y = H_deque.pop()
    H_deque.appendleft((s, y))
    s_dot_v = s @ v
    y_dot_s = y @ s
    v_new = v - s_dot_v / y_dot_s * y
    z = lbfgs_multiply(v_new, H_deque, gamma_0, max_mem, curr_mem+1)
    return z + (s_dot_v - y @ z) / y_dot_s * s


def lbfgs_directional(grad_k, H_deque: deque):
    s, y = H_deque[-1]
    gamma_0 = (y @ s) / (y @ y)
    max_mem = len(H_deque)
    curr_mem = 0
    return lbfgs_multiply(-grad_k, H_deque, gamma_0, max_mem, curr_mem)


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    H_deque = deque(maxlen=memory_size)
    x_k = np.copy(x_0)
    grad_k = grad_0 = oracle.grad(x_k)

    if trace:
        history['time'].append(0.0)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.linalg.norm(grad_k))
        if x_k.shape[0] <= 2:
            history['x'].append(x_k)
    start = datetime.now()
    for i in range(max_iter):
        if grad_k @ grad_k <= tolerance * grad_0 @ grad_0:
            break
        if i == 0 or memory_size == 0:
            d_k = -grad_k
        else:
            d_k = lbfgs_directional(grad_k, H_deque)
        alpha_k = line_search_tool.line_search(oracle=oracle,
                                               x_k=x_k, d_k=d_k,
                                               previous_alpha=1)
        x_k_new = (x_k + alpha_k * d_k).copy()
        grad_k_new = oracle.grad(x_k_new)
        if memory_size != 0:
            H_deque.append((x_k_new - x_k, grad_k_new - grad_k))
        x_k = x_k_new
        grad_k = grad_k_new

        if trace:
            history['time'].append((datetime.now() - start).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(grad_k))
            if x_k.shape[0] <= 2:
                history['x'].append(x_k)

    if grad_k @ grad_k > tolerance * grad_0 @ grad_0:
        return x_k, 'iterations_exceeded', history

    # Use line_search_tool.line_search() for adaptive step size.
    return x_k, 'success', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    g_k = oracle.grad(x_k)
    d_k = - g_k
    g_k_sqr = g_0_sqr = g_k.dot(g_k)

    if trace:
        history['time'].append(0.0)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.sqrt(g_k_sqr))
        if x_k.shape[0] <= 2:
            history['x'].append(x_k)
    start = datetime.now()
    for _ in range(max_iter):
        if g_k_sqr <= tolerance * g_0_sqr:
            break
        eps_k = min(1 / 2, np.power(g_k_sqr, 0.25)) * np.sqrt(g_k_sqr)
        func_hess_v = oracle.call_hess_vec(x_k)
        while True:
            d_k, _, _ = conjugate_gradients(func_hess_v, -g_k, d_k, eps_k)
            if np.dot(g_k, d_k) < 0:
                break
            else:
                eps_k *= 0.1

        alpha_k = line_search_tool.line_search(oracle=oracle,
                                               x_k=x_k, d_k=d_k,
                                               previous_alpha=1)
        x_k = (x_k + alpha_k * d_k).copy()
        g_k = oracle.grad(x_k)
        g_k_sqr = g_k.dot(g_k)

        if trace:
            history['time'].append((datetime.now() - start).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(g_k_sqr))
            if x_k.shape[0] <= 2:
                history['x'].append(x_k)

    if g_k_sqr > tolerance * g_0_sqr:
        return x_k, 'iterations exceeded', history
    # Use line_search_tool.line_search() for adaptive step size.
    return x_k, 'success', history
