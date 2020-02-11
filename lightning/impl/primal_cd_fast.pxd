cimport numpy as np
from lightning.impl.dataset_fast cimport ColumnDataset


cdef class LossFunction:

    cdef int max_steps
    cdef double sigma
    cdef double beta
    cdef int verbose

    cdef void solve_l2(self,
                       int j,
                       double C,
                       double alpha,
                       double *w,
                       ColumnDataset X,
                       double *y,
                       double *b,
                       double *violation)
 
    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L)

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *y,
                     double *b,
                     double *L_new)
    
    cdef void recompute(self,
                        ColumnDataset X,
                        double* y,
                        double* w,
                        double* b)

    cdef void _lipschitz_constant(self,
                                  ColumnDataset X,
                                  double scale,
                                  double* out)


    cdef int solve_l1(self,
                        int j,
                        double C,
                        double alpha,
                        double *w,
                        int n_samples,
                        ColumnDataset X,
                        double *y,
                        double *b,
                        double Lcst,
                        double violation_old,
                        double *violation,
                        int shrinking)

    cdef int solve_l1l2(self,
                        int j,
                        double C,
                        double alpha,
                        np.ndarray[double, ndim=2, mode='c'] w,
                        int n_vectors,
                        ColumnDataset X,
                        int* y,
                        np.ndarray[double, ndim=2, mode='fortran'] Y,
                        int multiclass,
                        np.ndarray[double, ndim=2, mode='c'] b,
                        double Lcst,
                        double *g,
                        double *d,
                        double *d_old,
                        double* Z,
                        double violation_old,
                        double *violation,
                        int shrinking)

    cdef void derivatives_mc(self,
                             int j,
                             double C,
                             int n_samples,
                             int n_vectors,
                             int* indices,
                             double *data,
                             int n_nz,
                             int* y,
                             double* b,
                             double* g,
                             double* Z,
                             double* L,
                             double* Lpp_max)

    cdef void update_mc(self,
                        double C,
                        int n_samples,
                        int n_vectors,
                        int* indices,
                        double *data,
                        int n_nz,
                        int* y,
                        double *b,
                        double *d,
                        double *d_old,
                        double* Z,
                        double* L_new)

    cdef void recompute_mc(self,
                           int n_vectors,
                           ColumnDataset X,
                           int* y,
                           np.ndarray[double, ndim=2, mode='c'] w,
                           np.ndarray[double, ndim=2, mode='c'] b)

    cdef void lipschitz_constant_mt(self,
                                    int n_vectors,
                                    ColumnDataset X,
                                    double C,
                                    double* out)

    cdef void lipschitz_constant_mc(self,
                                    int n_vectors,
                                    ColumnDataset X,
                                    double C,
                                    double* out)

