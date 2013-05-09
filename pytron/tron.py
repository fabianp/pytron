from _tron import minimize


if __name__ == '__main__':
	import numpy as np

	def func(w):
		return .5 * ((w - 1) * (w - 1)).sum()

	def grad(w):
		return w - 1

	def hess(w):
		return np.eye(w.size).dot(w)

	from _tron import minimize
	x0 = np.ones(10)
	out = minimize(func, grad, hess, x0, max_iter=10)
	print('Solution %s' % out)