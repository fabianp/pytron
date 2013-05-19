A Trust-Region Newton Method in Python
======================================

.. DANGER::
    This is alpha quality software and still quite rough on the edges.
    Specifically the error management is still lacking (which means that
    if something goes wrong in the optimization you won't see an error
    message but just get garbage). These things are being worked out but
    we're not quite there yet.

.. image:: http://fa.bianp.net/blog/static/images/2013/comparison_logistic_corr_10.png

The main function is pytron.minimize::

    def minimize(func, grad_hess, x0, args=(), max_iter=1000, tol=1e-6):

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



Stopping criterion
------------------

It stops whenever ||grad(x)|| < eps or the maximum number of iterations is
attained.

TODO: add tol

Examples
--------

Code
----
This software uses the `TRON optimization software
<http://www.mcs.anl.gov/~more/tron/>`_  (files src/tron.{h,cpp}) that was
taken from LIBLINEAR 1.93 (BSD licensed).

The modifications with respect to the orginal code are:

    * Do not initialize values to zero, allow arbitrary initializations

    * Modify stopping criterion to comply with scipy.optimize API. Stop
      whenever gradient is smaller than a given quantity, specified in the
      gtol argument

    * Return the gradient from TRON::tron (pass by reference)

    * Add `tol` option to TRON

    * Rename `eps` to `gtol`.

TODO
----
    * return status from TRON::TRON
    * callback argument


References
----------
If you use the software please consider citing some of the references below.

The method is described in the paper "Newton's Method for Large
Bound-Constrained Optimization Problems", Chih-Jen Lin and Jorge J. Moré
(http://epubs.siam.org/doi/abs/10.1137/S1052623498345075)

It is also discussed in the contex of Logistic Regression in the paper "Trust
Region Newton Method for Logistic Regression", Chih-Jen Lin, Ruby C. Weng,
S. Sathiya Keerthi (http://dl.acm.org/citation.cfm?id=1390703)

The website http://www.mcs.anl.gov/~more/tron/ contains reference to this
implementation, although the links to the software seem to be currently
broken (May 2013).


License
-------
This code is licensed under the terms of the BSD license. See file COPYING
for more details.


Acknowledgement
---------------
The source code for the
