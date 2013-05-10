__version__ = '0.1-git'

try:
	from _tron import minimize
except ImportError:
	print('Cannot import, must compile first')