from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='markov_epidemic',
    packages=['markov_epidemic'],
    version='1.0.2',
    author="Patrick Saux",
    author_email="patrick.jr.saux@gmail.com",
    description="Library for stochastic simulation and study of epidemics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sauxpa/markov_epidemic",
    install_requires=['numpy', 'networkx'],
    python_requires='>=3.6',
)
