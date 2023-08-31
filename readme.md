# Approximate Stein Classes for Truncated Density Estimation

This python repo is intended to reproduce the results from the [Approximate Stein Classes for Truncated Density Estimation](https://proceedings.mlr.press/v202/williams23b/williams23b.pdf) paper. Within it you will find examples of the TKSD method for basic scenarios, and scripts for reproducing plots found in the paper.


## Installation and Requirements

First, you must clone this environment to your working directory, using
```
git clone https://github.com/dannyjameswilliams/tksd.git 
cd tksd
```

This work was created using Python 3.8.13. We have included a `requirements.txt` file, made from [pipreqs](https://pypi.org/project/pipreqs/) on this workspace. We recommend if you are going to use this repo, to make a virtual environment. For example, with [Anaconda](https://www.anaconda.com/), you can do the following:

```bash
conda create -y --name tksd python=3.8.13
conda activate tksd
conda install -y --file requirements.txt 
pip install git+https://github.com/wittawatj/kernel-gof.git
```

This will create a `conda` virtual environment called `tksd`, and install all the packages necessary. We also give credit to [the kgof](https://github.com/wittawatj/kernel-gof) package, from the [A Linear-Time Kernel Goodness-of-Fit Test](https://proceedings.neurips.cc/paper/2017/file/979d472a84804b9f647bc185a877a8b5-Paper.pdf) paper, for fast computation of Gram matrices.

Additionally, you could use `pip` to install it to any environment, not just a virtual one

## Reproducing Results

The following lines will reproduce the plots presented in the paper.

### Main Text

**Figure 1 (top)**
```
python results/usa_boundary.py
```


**Figure 1 (bottom)**
```
python results/usa_benchmark.py
```


**Figure 2**

In this example, you can modify the argument to the script to be either the $\ell_2$ ball, or the $\ell_1$ ball. For example,
```
python results/dimension_benchmark.py 2
```
will run the experiment for the l2 ball, and
```
python results/dimension_benchmark.py 1
```
will run it for the $\ell_1$ ball. Note that running for the $\ell_1$ ball will take a while due to the computational complexity of calculating projections on the $\ell_1$ ball for the distance functions in _TruncSM_ and bdKSD. 

### Appendix

**Figure 3**
```
python appendix/g_convergence.py
```

**Figure 4**
```
python appendix/decreasing_epsilon.py
```

**Figure 5**
```
python appendix/consistency.py
```

**Figure 6**
```
python appendix/mixture.py
```

**Figure 7**
```
python appendix/regression.py
```

**Figure 8 and 9**
```
python appendix/boundary_distribution.py
```

**Figure 10**
```
python appendix/percentage_truncation.py
```


