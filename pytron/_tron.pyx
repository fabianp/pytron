import numpy as np
cimport numpy as np

from libc cimport string


cdef extern from "tron_helper.h":
    ctypedef void (*f_cb)(double *, void *, double *, int)
    ctypedef void (*grad_cb)(double *, double *)
    ctypedef void (*hess_cb)(double *, double *)
    cdef cppclass func_callback:
        func_callback(void *f_py, f_cb c_func, grad_cb c_grad, 
            hess_cb c_hess, int nr_variable)
        double fun(double *)

cdef extern from "tron.h":
    # cdef cppclass function:
    #     double fun(double *w)
    #     void grad(double *w, double *g)
    #     void Hv(double *s, double *Hs)
    #     int get_nr_variable()

    cdef cppclass TRON:
        TRON(func_callback *, double, int)
        tron(double *)


cdef void c_func(double *w, void *f_py, double *out, int nr_variable):
  w_np = np.empty(nr_variable, dtype=np.float64)
  string.memcpy(<void *> w_np.data, w, nr_variable * sizeof(double))
  out[0] = <double> (<object> f_py)(w_np)

cdef void c_grad(double *a, double *b):
    return

cdef void c_hess(double *a, double *b):
    return

def minimize(f, grad, hess, x0, *args):
    """
    f : callable
        f(w, *args)
    grad: callable
    hess: callable
    """

    x0 = np.asarray(x0)
    cdef int nr_variable = x0.size

    cdef np.ndarray[np.float64_t, ndim=1] x0_np
    cdef np.ndarray[np.float64_t, ndim=1] w_np
    #cdef np.ndarray[np.float64_t, ndim=1] g_np
    #cdef np.ndarray[np.float64_t, ndim=1] s_np
    cdef np.ndarray[np.float64_t, ndim=1] Hs_np

    x0_np = np.asarray(x0)
    w_np = np.empty(x0.size, dtype=np.float64)
    Hs_np = np.empty(x0.size * x0.size, dtype=np.float64)

    cdef void *f_py
    fpy = lambda x: f(x, *args)
    cdef func_callback * ff = new func_callback(f_py, c_func, c_grad,
         c_hess, nr_variable)

    ff.fun(<double *> w_np.data)


    # cdef void c_grad(double *w, double *g):
    #   # memcpy could be avoided
    #   string.memcpy(w_np.data, w, x0.size * sizeof(double))
    #   g_np = grad(w_np, *args)
    #   string.memcpy(g, g_np.data, x0.size * sizeof(double))

    # cdef void c_hess(double *s, double *Hs):
    #   s_np = np.empty(x0.size, dtype=np.float64)
    #   string.memcpy(s_np.data, s, x0.size * sizeof(double))
    #   hess_np = hess(s_np, *args)
    #   string.memcpy(Hs, Hs_np, Hs_np.size * sizeof(double))


