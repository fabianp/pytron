from cython cimport view
import numpy as np
cimport numpy as np
from libc cimport string
from cpython cimport Py_INCREF, Py_XDECREF, PyObject

cdef extern from "tron_helper.h":
    ctypedef void (*func_cb)(double *, void *, double *, int, void *)
    ctypedef void (*grad_cb)(double *, void *, void **, double *, int, void *)
    ctypedef void (*hess_cb)(double *, void *, double *, int, void *)
    cdef cppclass func_callback:
        func_callback(double *, void *, func_cb,
            void *, grad_cb, hess_cb, int nr_variable, void *)


cdef extern from "tron.h":
    cdef cppclass TRON:
        TRON(func_callback *, double, int)
        void tron(double *)


cdef void c_func(double *w, void *f_py, double *b, int nr_variable,
                 void *py_args):
    cdef view.array w0 = view.array(shape=(nr_variable,), itemsize=sizeof(double),
        mode='c', format='d', allocate_buffer=False)
    w0.data = <char *> w
    out = (<object> f_py)(np.asarray(w0), *(<object> py_args))
    b[0] = out
    # TODO: error check


cdef void c_grad(double *w, void *grad_hess_py, void **hess_py,
                 double *b, int nr_variable, void *py_args):
    cdef view.array b0 = view.array(shape=(nr_variable,), itemsize=sizeof(double),
        mode='c', format='d', allocate_buffer=False)
    b0.data = <char *> b
    cdef view.array w0 = view.array(shape=(nr_variable,), itemsize=sizeof(double),
        mode='c', format='d', allocate_buffer=False)
    w0.data = <char *> w
    out = (<object> grad_hess_py)(np.asarray(w0), *(<object> py_args))
    #Py_XDECREF(<PyObject *> hess_py[0]) # liberate previous one
    grad, hess = out[0], out[1]
    Py_INCREF(hess) # segfault otherwise
    b0[:] = grad[:]
    hess_py[0] = <void *> hess


cdef void c_hess(double *s, void *hess_py, double *b, int nr_variable,
                 void *py_args):
    cdef view.array b0 = view.array(shape=(nr_variable,),
        itemsize=sizeof(double), format='d',
        mode='c', allocate_buffer=False)
    cdef view.array s0 = view.array(shape=(nr_variable,),
        itemsize=sizeof(double), format='d',
        mode='c', allocate_buffer=False)
    s0.data = <char *> s
    b0.data = <char *> b
    out = (<object> hess_py)(np.asarray(s0))
    out = np.asarray(out, dtype=np.float)
    b0[:] = out[:]


def minimize(func, grad_hess, x0, args=(), max_iter=1000, tol=1e-6):
    """minimize func using Trust Region Newton algorithm

    Parameters
    ----------
    func : callable
        func(w, *args) is the evaluation of the function at w, It
        should return a float.
    grad: callable
        grad(w, *args) is the gradient of func at w, it
        should return a numpy array of size x0.size
    hess: callable
        hess(w, s, *args) returns the dot product H.dot(s), where
        H is the Hessian matrix at w. It must return a numpy array
        of size x0.size
    tol: float
        stopping criterion. XXX TODO. what is the stopping criterion ?

    Returns
    -------
    w : array
    """

    cdef np.ndarray[np.float64_t, ndim=1] x0_np
    x0_np = np.asarray(x0, dtype=np.float64)
    cdef int nr_variable = x0.size
    cdef double c_tol = tol
    cdef int c_max_iter = max_iter
    cur_w = None

    cdef func_callback * fc = new func_callback(
        <double *> x0_np.data,
        <void *> func, c_func,
        <void *> grad_hess, c_grad,
        c_hess, nr_variable, <void *> args)

    cdef TRON *solver = new TRON(fc, c_tol, c_max_iter)
    solver.tron(<double *> x0_np.data)

    del fc
    del solver

    return x0_np
