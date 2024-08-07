# The Power Proximal Point and Augmented Lagrangian Method

This repository contains the code to reproduce the numerical experiments in

> Oikonomidis A. Konstantinos, Bodard Alexander, Laude Emanuel, Patrinos Panagiotis. [Global Convergence Analysis of the Power Proximal Point and Augmented Lagrangian Method](https://arxiv.org/abs/2312.12205).

## Installation

We recommend to create a new Python virtual environment by running

`python3 -m venv .venv`

Install the required dependencies as follows

`pip install -r requirements.txt`

## Benchmarks

### Linear Programming

Run the script [main_linear_programs.py](benchmarks/main_linear_programs.py) to reproduce the experiment.

### Quadratic Programming

Run the script [main_convex_quadratics_v2.py](benchmarks/main_convex_quadratics_v2.py) to reproduce the Table of the experiment.

Run the script [main_convex_quadratics.py](benchmarks/main_convex_quadratics.py) to reproduce the Plots of the experiment.

### $\ell_2$-regularized $\ell_1$-regression problem

Run the script [main_l1_regression.py](benchmarks/main_l1_regression.py) to reproduce the Plots of the experiment.