"""
==========================================
Orthogonal Matching Pursuit with Primal Coordinate Descent Solvers
==========================================

This module provides orthogonal matching pursuit with coordinate descent solvers for 
a variety of loss functions.
"""

# Author: Mathieu Blondel
# Author: Kyohei Atarashi
# License: BSD

import numpy as np

from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals.six.moves import xrange

from .base import BaseClassifier
from .base import BaseRegressor

from .dataset_fast import get_dataset
from .omp_primal_cd_fast import _omp_primal_cd

from .primal_cd_fast import Squared
from .primal_cd_fast import SmoothHinge
from .primal_cd_fast import SquaredHinge
from .primal_cd_fast import ModifiedHuber
from .primal_cd_fast import Log


class _BaseOMP(object):

    def _get_loss(self):
        params = {"max_steps": self._get_max_steps(),
                  "sigma": self.sigma,
                  "beta": self.beta,
                  "verbose": self.verbose}

        losses = {
            "squared": Squared(verbose=self.verbose),
            "smooth_hinge": SmoothHinge(**params),
            "squared_hinge": SquaredHinge(**params),
            "modified_huber": ModifiedHuber(**params),
            "log": Log(**params),
        }

        return losses[self.loss]

    def _get_max_steps(self):
        if self.max_steps == "auto":
            if self.loss == "log":
                max_steps = 0
            else:
                max_steps = 30
        else:
            max_steps = self.max_steps
        return max_steps

    def _init_errors(self, Y):
        n_samples, n_vectors = Y.shape
        if self.loss == "squared":
            self.errors_ = -Y.T
        else:
            self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)


class OMPClassifier(_BaseOMP, BaseClassifier):
    """Estimator for learning linear classifiers by orthognal matching pursuit 
    with coordinate descent.

    The objective functions considered take the form

    minimize F(W) = C * L(W) + alpha * R(W),

    where L(W) is a loss term and R(W) is a penalty term.

    Parameters
    ----------
    n_nonzero_coefs: int
        Desired number of non-zero entries in the solution. If None (default),
        this value is set to 10% of n_features.

    loss : str, 'squared_hinge', 'log', 'modified_huber', 'squared'
        The loss function to be used.

    C : float
        Weight of the loss term.

    alpha : float
        Weight of the penalty term.

    max_iter : int
        Maximum number of iterations to perform.

    tol : float
        Tolerance of the stopping criterion.

    termination : str, 'violation_sum', 'violation_max'
        Stopping criterion to use.

    max_steps : int or "auto"
        Maximum number of steps to use during the line search. Use max_steps=0
        to use a constant step size instead of the line search. Use
        max_steps="auto" to let CDClassifier choose the best value.

    sigma : float
        Constant used in the line search sufficient decrease condition.

    beta : float
        Multiplicative constant used in the backtracking line search.

    callback : callable
        Callback function.

    selection : str, 'cyclic', 'uniform'
        Strategy to use for selecting coordinates.

    permute : bool
        Whether to permute coordinates or not before cycling (only when
        selection='cyclic').
    
    n_calls : int
        Frequency with which `callback` must be called.

    random_state : RandomState or int
        The seed of the pseudo random number generator to use.

    verbose : int
        Verbosity level.

    n_jobs : int
        Number of CPU's to be used. By default use one CPU.
        If set to -1, use all CPU's

    References
    ----------
    Block Coordinate Descent Algorithms for Large-scale Sparse Multiclass
    Classification.  Mathieu Blondel, Kazuhiro Seki, and Kuniaki Uehara.
    Machine Learning, May 2013.
    """

    def __init__(self, n_nonzero_coefs=None,
                 loss="squared_hinge",
                 C=1.0, alpha=1.0, max_iter=50, tol=1e-3,
                 termination="violation_sum",
                 max_steps="auto", sigma=0.01, beta=0.5,
                 selection="cyclic", permute=True,
                 callback=None, n_calls=100,
                 random_state=None, verbose=0, n_jobs=1):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.C = C
        self.alpha = alpha
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.termination = termination
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.selection = selection
        self.permute = permute
        self.callback = callback
        self.n_calls = n_calls
        self.random_state = random_state
        self.verbose = verbose
        self.coef_ = None
        self.violation_init_ = {}
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit model according to X and y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : classifier
            Returns self.
        """
        rs = self._get_random_state()

        # Create dataset
        ds = get_dataset(X, order="fortran")
        n_samples = ds.get_n_samples()
        n_features = ds.get_n_features()

        # Create label transformers
        # neg_label = 0 if self.penalty == "nn" else -1
        y, n_classes, n_vectors = self._set_label_transformers(y,
                                                               -1,
                                                               neg_label=-1)
        Y = np.asfortranarray(self.label_binarizer_.transform(y),
                              dtype=np.float64)

        # Initialize coefficients
        self.C_init = self.C
        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        self._init_errors(Y)

        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)
        indices = np.arange(n_features, dtype=np.int32)

        max_steps = self._get_max_steps()
        n_pos = np.zeros(n_vectors)
        vinit = self.C / self.C_init * np.ones_like(n_pos)
        for k in xrange(n_vectors):
            n_pos[k] = np.sum(Y[:, k] == 1)
            vinit[k] *= self.violation_init_.get(k, 0)
        n_neg = n_samples - n_pos
        tol = self.tol * np.maximum(np.minimum(n_pos, n_neg), 1) / n_samples
        
        n_nonzero_coefs = self.n_nonzero_coefs
        if not isinstance(n_nonzero_coefs, int) or n_nonzero_coefs < 1:
            raise ValueError("n_nonzero_coefs must be positive integer.")
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1*n_features)
        
        jobs = (delayed(_omp_primal_cd)(self, self.coef_, self.errors_,
                                        ds, y, Y, k, indices, self._get_loss(),
                                        self.selection, self.permute, 
                                        self.termination, self.C, self.alpha,
                                        self.max_iter, max_steps,
                                        vinit[k], rs, tol[k], self.callback, 
                                        self.n_calls, self.verbose,
                                        n_nonzero_coefs)
                for k in xrange(n_vectors))
        model = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(jobs)
        viol, coefs, errors = zip(*model)
        self.coef_ = np.asarray(coefs)
        self.errors_ = np.asarray(errors)

        for k in range(n_vectors):
            if k not in self.violation_init_:
                self.violation_init_[k] = viol[k]

        return self


class OMPRegressor(_BaseOMP, BaseRegressor):
    """Estimator for learning linear regressors by orthognal matching pursuit 
    with coordinate descent.

    The objective functions considered take the form

    minimize F(W) = C * L(W) + alpha * R(W),

    where L(W) is a loss term and R(W) is a penalty term.

    Parameters
    ----------
    loss : str, 'squared'
        The loss function to be used.

    For other parameters, see `OMPClassifier`.
    """

    def __init__(self, n_nonzero_coefs=None, C=1.0, alpha=1.0,
                 loss="squared", max_iter=50, tol=1e-3, 
                 termination="violation_sum",
                 max_steps=30, sigma=0.01, beta=0.5,
                 selection="cyclic", permute=True,
                 callback=None, n_calls=100,
                 random_state=None, verbose=0, n_jobs=1):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.C = C
        self.alpha = alpha
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.termination = termination
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.selection = selection
        self.permute = permute
        self.callback = callback
        self.n_calls = n_calls
        self.random_state = random_state
        self.verbose = verbose
        self.coef_ = None
        self.violation_init_ = {}
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit model according to X and y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values.

        Returns
        -------
        self : regressor
            Returns self.
        """
        rs = self._get_random_state()

        # Create dataset
        ds = get_dataset(X, order="fortran")
        n_features = ds.get_n_features()

        self.outputs_2d_ = len(y.shape) == 2
        if self.outputs_2d_:
            Y = y
        else:
            Y = y.reshape(-1, 1)
        Y = np.asfortranarray(Y, dtype=np.float64)
        y = np.empty(0, dtype=np.int32)
        n_vectors = Y.shape[1]

        # Initialize coefficients
        self.C_init = self.C
        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        self._init_errors(Y)

        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)
        indices = np.arange(n_features, dtype=np.int32)

        vinit = np.asarray([self.violation_init_.get(k, 0)
                for k in xrange(n_vectors)]) * self.C / self.C_init
        n_nonzero_coefs = self.n_nonzero_coefs
        if not isinstance(n_nonzero_coefs, int) or n_nonzero_coefs < 1:
            raise ValueError("n_nonzero_coefs must be positive integer.")
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1*n_features)
    
        jobs = (delayed(_omp_primal_cd)(self, self.coef_, self.errors_,
                                        ds, y, Y, k, indices, self._get_loss(),
                                        self.selection, self.permute,
                                        self.termination, self.C, self.alpha,
                                        self.max_iter, self.max_steps, vinit[k],
                                        rs, self.tol, self.callback, self.n_calls,
                                        self.verbose, n_nonzero_coefs)
                for k in xrange(n_vectors))

        model = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(jobs)
        viol, self.coef_, self.error_ = zip(*model)
        self.coef_ = np.asarray(self.coef_)
        self.error_ = np.asarray(self.error_)

        return self
