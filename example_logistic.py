from __future__ import print_function
import numpy as np
from pytron import minimize

def phi(t):
    # helper function
    return 1. / (1 + np.exp(-t))


def loss(w, X, y, alpha):
    # loss function to be optimized, it's the logistic loss
    z = X.dot(w)
    yz = y * z
    idx = yz > 0
    out = np.zeros_like(yz)
    out[idx] = np.log(1 + np.exp(-yz[idx]))
    out[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
    out = out.sum() + .5 * alpha * w.dot(w)
    return out


def grad_hess(w, X, y, alpha):
    # gradient of the logistic loss
    z = X.dot(w)
    z = phi(y * z)
    z0 = (z - 1) * y
    grad = X.T.dot(z0) + alpha * w
    def Hs(s):
        d = z * (1 - z)
        wa = d * X.dot(s)
        return X.T.dot(wa) + alpha * s

    return grad, Hs


def hessian(s, w, X, y, alpha):
    # H.dot(s), where H is the hessian at w of the logistic
    # loss
    z = X.dot(w)
    z = phi(y * z)
    d = z * (1 - z)
    wa = d * X.dot(s)
    Hs = X.T.dot(wa)
    out = Hs + alpha * s
    return  out


# set the data
n_samples, n_features = 10, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(X.dot(5 * np.random.randn(n_features)))
alpha = 1.
x0 = np.zeros(n_features)

# call the solver
sol = minimize(loss, grad_hess, x0, args=(X, y, alpha),
    max_iter=5, tol=.1)

from sklearn import linear_model
clf = linear_model.LogisticRegression(C=1./alpha, fit_intercept=False)
clf.fit(X, y)

print()
print('Solution using TRON:         %s' % sol)
print('Solution using scikit-learn: %s' % sol)