import numpy as np
from scipy import linalg
from example_logistic import loss, grad_hess


n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(X.dot(5 * np.random.randn(n_features)))
alpha = 1.
x0 = np.zeros(n_features)


def test1():
	import numdifftools as nd
	H1 = nd.Hessian(lambda x: loss(x, X, y, 1.))(X[0])

	def hessian(w, X, y, alpha):
	    # Hessian-times s
	    grad, hess = grad_hess(w, X, y, alpha)
	    return hess

	H2 = [hessian(X[0], X, y, 1.)(np.eye(X.shape[1])[i])  \
		for i in range(X.shape[1])]
	assert linalg.norm(H2 - H1) < 1e-3


if __name__ == '__main__':
	test1()