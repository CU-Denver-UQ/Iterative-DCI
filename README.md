# A repo for Iterative-DCI (iDCI) and Copula-Transformed iDCI (CT+iDCI)

This repo contains examples demonstrating an iterative approach to data-consistent inversion (DCI).

The examples coincide with two papers: 

- Iterative Data-Consistent Inversion with Multiple Push-Forward Constraints:
  - [arXiv version](https://arxiv.org/abs/2603.00033)
  - [Published version in International Journal of Uncertainty Quantification, Vol. 16, Issue 3, 2026, pp. 27-60](https://www.dl.begellhouse.com/journals/52034eb04b657aea,430a9ab664b7b491,22309ac560989c09.html)
- Copula Transformations for Data-Consistent Inversion:
  - arXiv version (posting soon)

## Examples for the iDCI-paper

### Example 1: Linear QoI

Executing either the Jupyter notebook or the Python script in the `Example-1-Linear` directory will generate all the data as well as the figures shown in the example of Section 6.1 of a recently completed manuscript on iterative DCI.

### Example 2: A Higher-Dimensional PDE-based Example

This example requires generating data associated with a single-phase incompressible flow model with a 100-dimensional parameter space defined by a truncated Karhunen-Lo\`{e}ve expansion of the log transformation of the permeability field. Instructions on generating the data with MrHyDE are given below. After these data are obtained, executing either the Jupyter notebook or the Python script in the `Example-2-PDE` directory will generate all the figures shown in Example 6.2 of a recently completely manuscript on iterative DCI.

#### Instructions to Reproduce PDE-based Example:
Due to the large size of the data-files associated with the PDE-based example, they are omitted in this repository.
Instead, we provide these instructions for reproducing them.
The results are generated using MrHyDE, which is an open source C++ package for solving multiscale/multiphysics systems.
See: https://github.com/sandialabs/MrHyDE

MrHyDE is built on top of Trilinos and requires a few third-party libraries:
CMake, openmpi, zlib, netcdf, pnetcdf, Boost, HDF5 (probably mpi version), libx11, ninja
We recommend building on a Mac and using Homebrew to install all of these.  
The numerical results were generated on a Mac laptop.

Next, download Trilinos and MrHyDE.
Downloading Trilinos
Trilinos is an open source collection of packages stored on GitHub. The repository can be cloned using:

git clone https://github.com/trilinos/Trilinos.git

Downloading MrHyDE
MrHyDE is an open source software framework currently stored in a repository on Sandia’s external github site. The repository can be cloned using:

git clone git@github.com:sandialabs/MrHyDE.git

Next, you need to build Trilinos, then MrHyDE.
These instructions are quite lengthy, but detailed instructions are here:e 
https://github.com/sandialabs/MrHyDE/wiki/Getting-Started

Once you have Trilinos and MrHyDE built, link to MrHyDE (or put it on your path) and regenerate the predicted data:
./mrhyde input.yaml

Next, regenerate the data-generating data:
./mrhyde input-datagen.yaml

From here, you have everything you need to run either the Jupyter notebook or the Python script and recreate the results in the paper.

## Examples for the CT+iDCI-paper

### Example 1: Dependence transformation in Linear Inverse Problems

Executing either the Jupyter notebook `CT+iDCI_paper_example_1_Gaussians.ipynb` or `CT+iDCI_paper_example_1_Betas.ipynb` will create the numerical results associated with Section 6.1 in the paper that utilize either Gaussian or Beta distributions for the data-generating distributions.

### Example 2: Adaptive Reference-Measure Refinement

Executing the Jupyter notebook `CT+iDCI_paper_example_2.ipynb` will recreate the results associated with Section 6.2 that involves adaptive sampling of an evolving reference measure.

### Example 3: Enriching Feasible Sets with Asynchronous Experiments

Executing the Jupyter notebook `CT+iDCI_paper_example_3.ipynb` will create the numerical results associated with Section 6.3 in the paper. Datasets are loaded from the `example_3_data` directory. The [`LUQ`](https://github.com/CU-Denver-UQ/LUQ/) package needs to be installed to execute this notebook.