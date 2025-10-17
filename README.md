# Iterative-DCI
Iterative Approach to Data Consistent Inversion

Instructions to Reproduce PDE-based Example:
The data-files associated with the PDE-based example are very large so we don't include them in this repository.
Instead, we provide these instructions for reproducing them yourself.
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

From here, you have everything you need to run the python scripts and recreate the results in the paper.
