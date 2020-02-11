# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
#
# Author: Mathieu Blondel
# Author: Kyohei Atarashi
# License: BSD

import numpy as np
cimport numpy as np

from libc.math cimport fabs, exp, log, sqrt

from lightning.impl.randomkit.random_fast cimport RandomState
from lightning.impl.dataset_fast cimport ColumnDataset
from .primal_cd_fast cimport LossFunction

DEF LOWER = 1e-2
DEF UPPER = 1e9

def _omp_primal_cd(self,
                   np.ndarray[double, ndim=2, mode='c'] w,
                   np.ndarray[double, ndim=2, mode='c'] b,
                   ColumnDataset X,
                   np.ndarray[int, ndim=1] y,
                   np.ndarray[double, ndim=2, mode='fortran'] Y,
                   int k,
                   np.ndarray[int, ndim=1, mode='c'] active_set,
                   LossFunction loss,
                   selection,
                   int permute,
                   termination,
                   double C,
                   double alpha,
                   int max_iter,
                   int max_steps,
                   double violation_init,
                   RandomState rs,
                   double tol,
                   callback,
                   int n_calls,
                   int verbose,
                   int n_nonzero_coefs):

    # Dataset
    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()
    cdef int n_vectors = w.shape[0]
    cdef double DBL_MAX = np.finfo(np.double).max

    # Counters
    cdef int t, s, i, j, n, m

    # Optimality violations
    cdef double violation_max_old = DBL_MAX
    cdef double violation_max
    cdef double violation
    cdef double violation_sum

    # Convergence
    cdef int check_violation_sum = termination == "violation_sum"
    cdef int check_violation_max = termination == "violation_max"
    cdef int stop = 0
    cdef int has_callback = callback is not None

    # Coordinate selection
    cdef int cyclic = selection == "cyclic"
    cdef int uniform = selection == "uniform"
    if uniform:
        permute = 0

    # Lipschitz constants
    cdef np.ndarray[double, ndim=1, mode='c'] Lcst
    Lcst = np.zeros(n_features, dtype=np.float64)
    if max_steps == 0:
        loss.lipschitz_constant_mt(n_vectors, X, C, <double*>Lcst.data)

    # Vector containers
    cdef double* b_ptr
    cdef double* y_ptr
    cdef double* w_ptr
    cdef np.ndarray[double, ndim=1, mode='c'] g  # Partial gradient
    cdef np.ndarray[double, ndim=1, mode='c'] d  # Block update
    cdef np.ndarray[double, ndim=1, mode='c'] d_old  # Block update (old)
    cdef np.ndarray[double, ndim=1, mode='c'] buf  # Buffer
    cdef double* buf_ptr
    if k == -1:
        # multitask.
        g = np.zeros(n_vectors, dtype=np.float64)
        d = np.zeros(n_vectors, dtype=np.float64)
        d_old = np.zeros(n_vectors, dtype=np.float64)
        b_ptr = <double*>b.data
    else:
        # Binary classification or regression.
        b_ptr = <double*>b.data + k * n_samples
        y_ptr = <double*>Y.data + k * n_samples
        w_ptr = <double*>w.data + k * n_features
        buf = np.zeros(n_samples, dtype=np.float64)
        buf_ptr = <double*>buf.data

    # for omp selection
    cdef int n_selected_features, tmp, j_selected, jj, jj_selected
    n_selected_features = 0
    cdef double derivative_selected, Lp, Lpp, L
    cdef double* data
    cdef int* indices
    cdef int n_nz
    for m in range(n_nonzero_coefs):
        # select feature 
        s = 0
        derivative_selected = 0
        while s < (n_features - n_selected_features):
            jj = n_selected_features+s
            j = active_set[jj]
            X.get_column_ptr(j, &indices, &data, &n_nz)
            loss.derivatives(j, C, indices, data, n_nz, y_ptr, b_ptr, &Lp, 
                             &Lpp, &L)
            if fabs(Lp) > derivative_selected:
                j_selected = j
                jj_selected = jj
                derivative_selected = fabs(Lp)
            s += 1
        tmp = active_set[n_selected_features] 
        active_set[n_selected_features] = j_selected
        active_set[jj_selected] = tmp
        n_selected_features += 1

        # fitting with warm stat
        for t in xrange(max_iter):
            # Permute features (cyclic case only)
            if permute:
                rs.shuffle(active_set[:n_selected_features])

            # Initialize violations.
            violation_max = 0
            violation_sum = 0

            s = 0
            while s < n_selected_features:
                # Select coordinate.
                if cyclic:
                    j = active_set[s]
                elif uniform:
                    j = active_set[rs.randint(n_selected_features - 1)]

                loss.solve_l2(j, C, alpha, w_ptr, X, y_ptr, b_ptr, &violation)

                # Update violations.
                violation_max = max(violation_max, violation)
                violation_sum += violation

                # Callback
                if has_callback and s % n_calls == 0:
                    ret = callback(self)
                    if ret is not None:
                        stop = 1
                        break

                s += 1
            # end while n_selected_features

            if stop:
                break

            # Initialize violations.
            if t == 0 and violation_init == 0:
                if check_violation_sum:
                    violation_init = violation_sum
                elif check_violation_max:
                    violation_init = violation_max

            # Verbose output.
            if verbose >= 1:
                if check_violation_sum:
                    print("iter", t + 1, violation_sum / violation_init,
                        "(%d)" % n_selected_features)
                elif check_violation_max:
                    print("iter", t + 1, violation_max / violation_init,
                        "(%d)" % n_selected_features)

            # Check convergence.
            if (check_violation_sum and
                violation_sum <= tol * violation_init) or \
            (check_violation_max and
                violation_max <= tol * violation_init):
                if verbose >= 1:
                    print("\nConverged at iteration", t)
                    break
            violation_max_old = violation_max

    if k == -1:
        return violation_init, w, b
    else:
        return violation_init, w[k], b[k]


