import math
import numpy as np
import scipy
from scipy.special import logsumexp, softmax
from datetime import datetime
from collections import defaultdict
import warnings

"""
The BaseSmoothOracle and OracleCallsCounter classes are originally found in
Nikita Doikov's super-newton repository: https://github.com/doikov/super-newton
which has the following MIT License:

MIT License

Copyright (c) 2022 Nikita

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

leapssn is a modification of the original Doikov's super_newton algorithm.

"""
class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of the function at point x.
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

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)

    def third_vec_vec(self, x, v):
        """
        Computes tensor-vector-vector product with the third derivative tensor
        D^3 f(x)[v, v].
        """
        raise NotImplementedError('Third derivative oracle is not implemented.')

class OracleCallsCounter(BaseSmoothOracle):
    """
    Wrapper to count oracle calls.
    """

    def __init__(self, oracle):
        self.oracle = oracle
        self.func_calls = 0
        self.grad_calls = 0
        self.hess_calls = 0
        self.hess_vec_calls = 0
        self.third_vec_vec_calls = 0
        self.func_at_hess_eval = []

    def func(self, x):
        self.func_calls += 1
        return self.oracle.func(x)

    def grad(self, x):
        self.grad_calls += 1
        return self.oracle.grad(x)

    def hess(self, x):
        self.hess_calls += 1
        self.func_at_hess_eval.append(self.oracle.func(x))
        return self.oracle.hess(x)

    def hess_vec(self, x, v):
        self.hess_vec_calls += 1
        return self.oracle.hess_vec(x, v)

    def third_vec_vec(self, x, v):
        self.third_vec_vec_calls += 1
        return self.oracle.third_vec_vec(x, v)

class LeapSSN(BaseSmoothOracle):
    def leapssn(self, x_0, n_iters=1000, Lambda_0=1.0, alpha=0.5, beta=0.25,
                    adaptive_search=True, adaptive_search_max_iter = 100_000, trace=True, B=None, Binv=None, eps=1e-8,
                    f_star=None, grad_tol=None, warnings=False, accept_tol=1e-8):
        """
        Run our algorithm
        for 'n_iters' iterations, minimizing smooth function.

        'oracle' is an instance of BaseSmoothOracle representing the objective.
        """
        oracle = OracleCallsCounter(self)
        start_timestamp = datetime.now()

        # Initialization of the dual norm
        if B is None:
            B = np.eye(x_0.shape[0])
            # dual_norm_sqr = lambda x: np.linalg.norm(x, 2) ** 2
            dual_norm_sqr = lambda x: x.dot(x)
        else:
            if Binv is None and B is None:
                Binv = np.eye(x_0.shape[0])
            elif Binv is None:
                warnings.warning("Numerically computing an inverse matrix. " \
                "This is numerically unstable and slow. Consider passing in the Riesz map directly.")
                Binv = np.linalg.inv(B)
            dual_norm_sqr = lambda x: Binv.dot(x).dot(x)

        # Initialization of the method
        x_k = np.copy(x_0)
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        print("Residual norm: %s"%(np.linalg.norm(g_k)))
        g_k_norm = np.sqrt(dual_norm_sqr(g_k))
        Lambda_k = Lambda_0

        history = defaultdict(list) if trace else None
        matrix_inverses = 0
        status = ""

        # Main loop
        no_backtracks = 0
        for k in range(n_iters + 1):

            if trace:
                history['func'].append(f_k)
                history['grad_norm'].append(g_k_norm)
                history['Lambda_k'].append(Lambda_k)
                history['grad_calls'].append(oracle.grad_calls)
                history['matrix_inverses'].append(matrix_inverses)
                history['hess_calls'].append(oracle.hess_calls)
                history['time'].append(
                    (datetime.now() - start_timestamp).total_seconds())

            if (f_star is not None and f_k - f_star < eps) or \
                    (grad_tol is not None and g_k_norm < grad_tol):
                status = "success, %d iters" % k
                break

            if k == n_iters:
                status = "iterations_exceeded"
                break

            Hess_k = oracle.hess(x_k)
            f_current = oracle.func(x_k)

            history["func_at_hess_eval"].append(oracle.func(x_k))

            for i in range(adaptive_search_max_iter + 1):
                if i == adaptive_search_max_iter:
                    if warnings:
                        print(('W: adaptive_iterations_exceeded, k = %d' % k),
                            flush=True)
                    break

                lambda_k = Lambda_k
                print(lambda_k)
                try:
                    # Compute the regularized Newton step

                    delta_x = scipy.linalg.cho_solve(scipy.linalg.cho_factor(
                        Hess_k + lambda_k * B, lower=False), -g_k)
                    # delta_x = np.linalg.solve(Hess_k + lambda_k * B, -g_k)
                    matrix_inverses += 1

                except (np.linalg.LinAlgError, ValueError) as e:
                    if warnings:
                        print('W: linalg_error', flush=True)
                f_new = oracle.func(x_k + delta_x)
                history["func_at_lssolve"].append(f_new)

                g_new = oracle.grad(x_k + delta_x)
                g_new_norm_sqr = dual_norm_sqr(g_new)

                if not adaptive_search:
                    break

                # Check condition for Lambda_k
                c1 = g_new.dot(-delta_x)
                c2 = alpha * g_new_norm_sqr / lambda_k
                c3 = f_current - f_new
                c4 = beta * lambda_k * delta_x.dot((B.dot(delta_x)))
                print("c1=%s, c2=%s, c3=%s, c4=%s"%(c1,c2,c3,c4))

                if (((c1 >= c2) and (c3 >= c4)) 
                    or (np.abs(c1)<=accept_tol and np.abs(c2)<=accept_tol and np.abs(c3)<=accept_tol and np.abs(c4)<=accept_tol)):
                    print("Update accepted.")
                    Lambda_k = Lambda_k * 0.5
                    break
                Lambda_k = Lambda_k * 2
                no_backtracks += 1

            history["reg_parameter"].append(lambda_k)

            # Update the point
            x_k += delta_x
            f_k = f_new
            g_k = g_new
            print("Residual norm: %s"%(np.linalg.norm(g_k)))
            g_k_norm = np.sqrt(g_new_norm_sqr)
        print("No of backtracks: ", no_backtracks)
        return x_k, status, history