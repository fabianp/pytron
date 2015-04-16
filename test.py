import numpy as np
from scipy import optimize
from example_logistic import loss, grad_hess
from sklearn import datasets, cross_validation
from nose import tools


def test_grad_logistic():
    X, y = datasets.make_classification()
    y[y==0] = -1
    y = y.astype(np.float)

    f = lambda x: loss(x, X, y, 1.)
    f_grad = lambda x: grad_hess(x, X, y, 1.)[0]

    small = optimize.check_grad(f, f_grad, np.random.randn(X.shape[1]))
    tools.assert_less(small, 1.)

