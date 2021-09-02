from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "pandas",
    "torch"
]

tests_require = [
    "pytest",
    "os"
]

setup(
    name='robertastree',
    packages=find_packages(include=['robertastree']),
    version='0.1.0',
    description='Implementation of RobertasTree model',
    author='Lorenzo Squadrani',
    license='None',
    install_requires=install_requires,
    tests_require=tests_require,
    classifiers=['Development Status :: 1 - Beta',
                 'Intended Audience :: Developers',
                 'Topic :: Software Development :: Libraries',
                 'Programming Language :: Python :: 3.7'],
)
