PEPTIDE-QSPR
====

This is a Python package written to do both motif modeling and QSPR modeling of peptides,
using only their sequences as input data. The QSPR model uses calculated chemical descriptors
(e.g. number of hydrogen bond donors, net charge) and uses
[PyMC3](https://github.com/pymc-devs/pymc3) training to fit a Gaussian mixture model with
user-specified kernel number to an input dataset. The motif model uses a house-written
motif-capturing model trained using per-coordinate infinite-horizon stochastic gradient descent
with L1 regularization. See the [white paper](https://arxiv.org/abs/1804.06327) for more
mathematical detail.

gaussmix
----
The QSPR training code. Written for Python3 with [PyMC3](https://github.com/pymc-devs/pymc3).

gibbs
----
The motif model training code. Written in C++ as a Python extension, so Python is used to interface.

qspr_plots
----
Contains definitions used in plot-generating throughout the package. Used extensively in scripts.

resources
----
This is where the data needed to set up and train the models is located, as well as the pre-trained
distributions from the best-cases exhibited in the [white paper](https://arxiv.org/abs/1804.06327).

scripts
----
Contains a variety of scripts used throughout the project, including those used to produce the
figures in the [white paper](https://arxiv.org/abs/1804.06327). 