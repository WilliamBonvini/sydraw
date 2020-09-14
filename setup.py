from setuptools import find_packages, setup
setup(
    name='datafact',
    packages=find_packages(include=['datafact']),
    version='0.1.18',
    description='My first Python library',
    author='William Bonvini',
    license='MIT',
    install_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
