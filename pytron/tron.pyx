from cython cimport view
import numpy as np
from scipy import optimize
cimport numpy as np
from libc cimport string
from cpython cimport Py_INCREF, Py_XDECREF, PyObject

cdef extern from "tron_helper.h":
    ctypedef double (*func_cb)(double *, void *, int, void *)
    ctypedef int (*grad_cb)(double *, void *, void **, double *, int, void *)
    ctypedef int (*hess_cb)(double *, void *, double *, int, void *)
    cdef cppclass func_callback:
        func_callback(double *, void *, func_cb,
            void *, grad_cb, hess_cb, void *, func_cb, int nr_variable, void *)


cdef extern from "tron.h":
    cdef cppclass TRON:
        TRON(func_callback *, double, double, int)
        void tron(double *, double *, int)
        int n_iter
        double gnorm
        double fun


cdef double c_func(double *w, void *f_py, int nr_variable, void *py_args):
    cdef view.array w0 = view.array(shape=(nr_variable,), itemsize=sizeof(double),
        mode='c', format='d', allocate_buffer=False)
    cdef double b
    w0.data = <char *> w
    # TODO: exception return value
    out = (<object> f_py)(np.asarray(w0), *(<object> py_args))
    b = out
    return b


cdef int c_grad(double *w, void *grad_hess_py, void **hess_py,
                 double *b, int nr_variable, void *py_args):
    cdef view.array b0 = view.array(shape=(nr_variable,), itemsize=sizeof(double),
        mode='c', format='d', allocate_buffer=False)
    b0.data = <char *> b
    cdef view.array w0 = view.array(shape=(nr_variable,), itemsize=sizeof(double),
        mode='c', format='d', allocate_buffer=False)
    w0.data = <char *> w
    try:
        out = (<object> grad_hess_py)(np.asarray(w0), *(<object> py_args))
    except:
        return -1
    #Py_XDECREF(<PyObject *> hess_py[0]) # liberate previous one
    grad, hess = out[0], out[1]
    Py_INCREF(hess) # segfault otherwise
    b0[:] = grad[:]
    hess_py[0] = <void *> hess
    return 0


cdef int c_hess(double *s, void *hess_py, double *b, int nr_variable,
                 void *py_args):
    cdef view.array b0 = view.array(shape=(nr_variable,),
        itemsize=sizeof(double), format='d',
        mode='c', allocate_buffer=False)
    cdef view.array s0 = view.array(shape=(nr_variable,),
        itemsize=sizeof(double), format='d',
        mode='c', allocate_buffer=False)
    s0.data = <char *> s
    b0.data = <char *> b
    try:
        out = (<object> hess_py)(np.asarray(s0))
    except:
        return -1
    out = np.asarray(out, dtype=np.float)
    b0[:] = out[:]
    return 0

cdef double c_callback(double *w, void *py_callback, int nr_variable,
                     void *py_args):

    cdef view.array w0 = view.array(shape=(nr_variable,), itemsize=sizeof(double),
        mode='c', format='d', allocate_buffer=False)
    w0.data = <char *> w
    out = (<object> py_callback)(np.asarray(w0))
    return 0.


def minimize(func, grad_hess, x0, args=(), max_iter=500, tol=1e-6, gtol=1.,
             callback=None, verbose=False):
    """minimize func using Trust Region Newton algorithm

    Parameters
    ----------
    func : callable
        func(w, *args) is the evaluation of the function at w, It
        should return a float.
    grad_hess: callable
        TODO
    x0 : array
        starting point for iteration.
    gtol: float
        Gradient infinity norm must be less than gtol
        before succesful termination.

    Returns
    -------
    res : scipy.optimize.Result
        The optimization result represented as a scipy.optimize.Result object.
        Important attributes are: ``x`` the solution array, ``success`` a
        boolean flag indicating if the optimizer exited successfully,
        ``nit`` an integer for the number of iterations performed
    """

    cdef np.ndarray[np.float64_t, ndim=1] x0_np
    cdef np.ndarray[np.float64_t, ndim=1] grad
    cdef int nr_variable = x0.size
    cdef double c_gtol = gtol
    cdef double c_tol = tol
    cdef int c_max_iter = max_iter
    cdef void *py_callback
    cur_w = None
    x0_np = np.asarray(x0, dtype=np.float64)
    grad = np.empty(x0_np.size, dtype=np.float64)
    if callback is None:
        py_callback = NULL
    else:
        py_callback = <void *> callback

    cdef func_callback * fc = new func_callback(
        <double *> x0_np.data,
        <void *> func, c_func,
        <void *> grad_hess, c_grad,
        c_hess, py_callback, c_callback, nr_variable, <void *> args)

    cdef TRON *solver = new TRON(fc, c_tol, c_gtol, c_max_iter)
    solver.tron(<double *> x0_np.data, <double *>grad.data, <int> verbose)
    success = solver.gnorm < gtol
    result = optimize.Result(
        x=x0_np, success=success, nit=solver.n_iter, gnorm=solver.gnorm,
        fun=solver.fun, jac=grad, message='TODO')

    del fc
    del solver

    return result
