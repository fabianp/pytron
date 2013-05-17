
from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np
from glob import glob

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.2
Programming Language :: Python :: 3.3
Topic :: Software Development
Operating System :: POSIX
Operating System :: Unix

"""

sources =['pytron/tron.pyx', 'pytron/src/tron.cpp', 'pytron/src/tron_helper.cpp'] + \
    glob('pytron/src/blas/*.c')


setup(
    name='pytron',
    description='Python bindings for TRON optimizer',
    long_description=open('README.rst').read(),
    version='0.1',
    author='Fabian Pedregosa',
    author_email='f@fabianp.net',
    url='http://pypi.python.org/pypi/pytron',
    packages=['pytron'],
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    license='Simplified BSD',
    requires=['numpy', 'scipy'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension('pytron.tron',
        sources=sources,
        language='c++', include_dirs=[np.get_include(), 'pytron/src/'])],

)
