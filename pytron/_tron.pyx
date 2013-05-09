from cython cimport view
import numpy as np
cimport numpy as np
from libc cimport string
#from python_ref cimport Py_INCREF, Py_DECREF

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

cdef void c_func(double *w, void *f_py, double *b, int nr_variable):
    # 1. convert w to numpy array
    # 2. call f_py
    # 3. get the result
    # 4. free temp objects
    cdef view.array w0 = view.array(shape=(nr_variable,), itemsize=sizeof(double),
        mode='c', format='d', allocate_buffer=False)
    w0.data = <char *> w
    b[0] = (<object> f_py)(w0)
    w0 = None
    del w0


cdef void c_grad(double *w, void *f_py, double *b, int nr_variable):
    cdef view.array b0 = view.array(shape=(nr_variable,), itemsize=sizeof(double),
        mode='c', format='d', allocate_buffer=False)
    b0.data = <char *> b
    cdef view.array w0 = view.array(shape=(nr_variable,), itemsize=sizeof(double),
        mode='c', format='d', allocate_buffer=False)
    w0.data = <char *> w
#    cdef view.array w0 = <double[:nr_variable]> w
    out = (<object> f_py)(w0)
    b0[...] = out
    del b0, w0, out


cdef void c_hess(double *w, void *f_py, double *b, int nr_variable):
    cdef view.array b0 = view.array(shape=(nr_variable * nr_variable,), 
        itemsize=sizeof(double), format='d',
        mode='c', allocate_buffer=False)
    b0.data = <char *> b
    cdef view.array w0 = view.array(shape=(nr_variable,), 
        itemsize=sizeof(double), format='d',
        mode='c', allocate_buffer=False)
    w0.data = <char *> w
    out = (<object> f_py)(w0)
    b0[:] = out
    del b0, w0, out

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
        w0 = np.asarray(w)
        return f(w0, *args)

    def py_grad(w):
        w0 = np.asarray(w)        
        out = grad(w0, *args)
        return out.astype(np.float64).ravel()

    def py_hess(w):
        w0 = np.asarray(w)
        out = hess(w0, *args)
        return out.astype(np.float64).ravel()

    cdef func_callback * fc = new func_callback(
        <void *> py_func, c_func, 
        <void *> py_grad, c_grad,
        <void *> py_hess, c_hess, nr_variable)

    cdef TRON *solver = new TRON(fc, c_tol, c_max_iter)
    solver.tron(<double *> x0_np.data)

#    del fc
#    del solver
    import sys
    print sys.getrefcount(py_func)
    # print sys.getrefcount(py_grad)

    return x0_np
