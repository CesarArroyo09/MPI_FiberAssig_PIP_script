# MPI Fiber Assignment PIP scripts
This repository contains the scripts to compute the bitwise vectors and IIP weights for DESI's fiber assignment.

The intent of this repository is to contain:
- The Python script which runs DESI's fiber assignation in parallel and compute both the bitwise vectors.
- Documentation on how to set up a Conda environment in Cori NERSC which can run a modified version of `fiberassign-1.0.0` and can use `mpi4py`.
- The job script to run the MPI_FiberAssig_PIP script on Cori NERSC.
- Documentation on how the files should be prepared to be usable (don't pay attention to this for now).
