import numpy as np


def func(b):
	return np.sum(b)

from _tron import minimize
x0 = np.array([0, 1, 2], dtype=np.float)
minimize(func, None, None, x0, 1, 2)
