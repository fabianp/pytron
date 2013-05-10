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
    res = np.log(1 + np.exp(-z[idx])).sum() + (-yz[~idx] + np.log(1 + np.exp(yz[~idx]))).sum()
    return res + .5 * alpha * w.dot(w)

def gradient(w, X, y, alpha):
    # gradient of the logistic loss
    tmp = X * y[:, None]
    return - tmp.T.dot(1 - phi(tmp.dot(w))) + alpha * w

def hessian(s, w, X, y, alpha):
    # returns Hessian.dot(w)
    tmp0 = X * y[:, None]
    tmp1 = tmp0.dot(w)
    d = phi(tmp1) * (1 - phi(tmp1))
    return X.T.dot(d * X.dot(s)) + alpha * s


# set the data
n_samples, n_features = 1000, 10000
X = np.random.randn(n_samples, n_features)
y = np.sign(X.dot(5 * np.random.randn(n_features)))
alpha = .1
x0 = np.zeros(n_features)

# call the solver
sol = minimize(loss, gradient, hessian, x0, args=(X, y, alpha), 
    max_iter=5000, tol=1e-3)

from sklearn import linear_model
clf = linear_model.LogisticRegression(C=1./alpha, fit_intercept=False)
clf.fit(X, y)

print()
print('Solution using TRON:         %s' % sol)
print('Solution using scikit-learn: %s' % sol)