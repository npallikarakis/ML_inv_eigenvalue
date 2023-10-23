# ML_inv_eigenvalue
Scripts and data for the paper "Application of machine learning regression models to inverse eigenvalue problems", https://arxiv.org/abs/2212.04279 

We study the numerical solution of inverse eigenvalue problems from a machine learning perspective. Two different problems are considered: the inverse Sturm-Liouville eigenvalue problem for symmetric potentials and the inverse transmission eigenvalue problem for spherically symmetric refractive indices. Firstly, we solve the corresponding direct problems to produce the required eigenvalues datasets in order to train the machine learning algorithms. Next, we consider several examples of inverse problems and compare the performance of each model to predict the unknown potentials and refractive indices respectively, from a given small set of the lowest eigenvalues. The supervised regression models we use are k-Nearest Neighbours, Random Forests and Multi-Layer Perceptron. Our experiments show that these machine learning methods, under appropriate tuning on their parameters, can numerically solve the examined inverse eigenvalue problems.

1. INVERSE STURM-LIOUVILLE EIGENVALUE PROBLEM: 
We solve the inverse Sturm-Liouville eigenvalue problem with Neumann boundary conditions, for family of symmetric potentials q(x)=(1 − exp(b(x−1/2))^2, 0<x<1. We use kNN, Random Forests and Multilayer Perceptron. 
Files kNN_SL.py, RandomForest_SL.py and MultiMLP_SL.py include the corresponding python code. 
The direct problems are solved using the matlab package MATSLISE.

2. INVERSE TRANSMISSION EIGENVALUE PROBLEM: 
We solve the inverse transmission eigenvalue problem for piecewise constant refractive indices in discs. We use kNN, Random Forests and Multilayer Perceptron. 
Files kNN_ITEP_2L.py, RF_ITEP_2L.py and MLP_ITEP_2L.py include the corresponding python code. 
The direct problems are solved using a spectral-galerkin method, developed in https://iopscience.iop.org/article/10.1088/0266-5611/29/10/104010.

The raw data of the direct problems are available upon request. 
