import numpy as np
cimport numpy as np
from libc cimport string
from python_ref cimport Py_INCREF, Py_DECREF

np.import_array()

cdef extern from "tron_helper.h":
    ctypedef void (*tron_cb)(double *, void *, double *, int)
    cdef cppclass func_callback:
        func_callback(void *, tron_cb,
            void *, tron_cb,
            void *, tron_cb, int nr_variable)
        double fun(double *)

cdef extern from "tron.h":
    cdef cppclass TRON:
        TRON(func_callback *, double, int)
        void tron(double *)

cdef public void c_func(double *w, void *f_py, double *b, int nr_variable):
    cdef np.ndarray[np.float64_t, ndim=1] x0_np
    x0_np = np.empty(nr_variable, dtype=np.float64)
    string.memcpy(<void *> x0_np.data, <void *> w, nr_variable * sizeof(double))
    b[0] = (<object> f_py)(x0_np)


cdef public void c_grad(double *w, void *f_py, double *b, int nr_variable):
    cdef np.ndarray[np.float64_t, ndim=1] g_np
    cdef np.ndarray[np.float64_t, ndim=1] x0_np
    x0_np = np.zeros(nr_variable, dtype=np.float64)
    string.memcpy(<void *> x0_np.data, <void *> w, nr_variable * sizeof(double))
    out = (<object> f_py)(x0_np)
    g_np = np.asarray(out)
    assert g_np.size == nr_variable
    string.memcpy(<void *> b, <void *> g_np.data, nr_variable * sizeof(double))


cdef void c_hess(double *w, void *f_py, double *b, int nr_variable):
    cdef np.ndarray[np.float64_t, ndim=1] Hs_np
    cdef np.ndarray[np.float64_t, ndim=1] x0_np
    x0_np = np.empty(nr_variable, dtype=np.float64)
    string.memcpy(<void *> x0_np.data, <void *> w, nr_variable * sizeof(double))
    out = (<object> f_py)(x0_np)
    Hs_np = np.asarray(out).ravel('C')
    assert Hs_np.size == nr_variable * nr_variable
    string.memcpy(<void *> b, <void *> Hs_np.data, 
        nr_variable * nr_variable * sizeof(double))

def minimize(f, grad, hess, x0, args=(), max_iter=1000, tol=1e-6):
    """
    f : callable
        f(w, *args)
    grad: callable
    hess: callable
    """

    cdef np.ndarray[np.float64_t, ndim=1] x0_np
    x0_np = np.asarray(x0, dtype=np.float64)
    cdef int nr_variable = x0.size
    cdef double c_tol = tol
    cdef int c_max_iter = max_iter

    def py_func(w):
        return f(w, *args)

    def py_grad(w):
        return grad(w, *args)

    def py_hess(w):
        return hess(w, *args)

    cdef func_callback * fc = new func_callback(
        <void *> py_func, c_func, 
        <void *> py_grad, c_grad,
        <void *> py_hess, c_hess, nr_variable)

    cdef TRON *solver = new TRON(fc, c_tol, c_max_iter)
    print('const')
    solver.tron(<double *> x0_np.data)
    print('out')

#    del fc
#    del solver
    # import sys
    # print sys.getrefcount(py_func)
    # print sys.getrefcount(py_grad)

    return x0_np
