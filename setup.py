from setuptools import find_packages, setup

setup(
    name='robertastree',
    packages=find_packages(include = ['robertastree']),
    version='0.1.0',
    description='Implementation of RobertasTree model',
    author='Lorenzo Squadrani',
    license='None',
    classifiers=[   'Development Status :: 1 - Beta',
                    'Intended Audience :: Developers',
                    'Topic :: Software Development :: Libraries',
                    'Programming Language :: Python :: 3.7'      ],
    setup_requires = ['transformers']
)