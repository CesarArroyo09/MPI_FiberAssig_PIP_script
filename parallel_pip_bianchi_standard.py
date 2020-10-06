#!/usr/bin/env python
# This looks for the Python executable in $PATH environment variable.

# --------------------------------------------------------------------------------------------------
# IMPORTING LIBRARIES
#
# The different libraries are imported here. We will be using:
# - mpi4py
# - numpy
# - fiberassign/1.0.0
# - os
# --------------------------------------------------------------------------------------------------

from mpi4py import MPI
import numpy as np
from numpy.random import (RandomState, random)
from fiberassign.hardware import load_hardware
from fiberassign.tiles import load_tiles
from fiberassign.targets import (TARGET_TYPE_SCIENCE,
                                 TARGET_TYPE_SKY, TARGET_TYPE_STANDARD,
                                 Targets, TargetsAvailable,
                                 TargetTree, FibersAvailable,
                                 load_target_file)
from fiberassign.assign import Assignment

import os


# --------------------------------------------------------------------------------------------------
# READING ARGUMENTS FROM THE JOB
#
# This script is meant to be used in a job. Here I define the arguments to be read from the job and
# the variables which are going to contain those parameters which are basically the paths to the
# required files to run DESI's fiber assignment.
# --------------------------------------------------------------------------------------------------

# Define arguments and the help information to be displayed.
import argparse

def parseOptions(comm):
    parser = argparse.ArgumentParser()

    parser.add_argument("--mtl", help="Path to the MTL file", required=True)
    
    parser.add_argument("--stdstar1", help="Path to the first standard stars file", required=True)
    
    parser.add_argument("--stdstar2", help="Path to the second standard stars file")
    
    parser.add_argument("--sky", help="Path to the sky fiber locations", required=True)
    
    parser.add_argument("--tiles", help="Path to the tiles file")
    
    parser.add_argument("--outdir", help="Path to the output directory", required=True)
    
    parser.add_argument("--label", help="Labels for the files to be saved", required=True)
    
    #parser.add_argument("--rank-group", help="Random seeds are in the range [62*rank_group,"
    #                    "62*(rank_group + 1) - 1", required=True, type=int)

    args = None
    try:
        if comm.Get_rank() == 0:
            args = parser.parse_args()
    finally:
        args = comm.bcast(args, root=0)

    if args is None:
        exit(0)
    return args

# Variable for world communicator, number of MPI tasks and rank of the present MPI process.
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

args = parseOptions(comm)

# Paths for the necessary files to run fiber assignment.
mtlfile = args.mtl
stdstar1 = args.stdstar1
stdstar2 = args.stdstar2
sky = args.sky
tilesfile = args.tiles
outdir = args.outdir
label = args.label
#rank_group = args.rank_group

#  python --cpu-bind=verbose parallel_pip_bianchi_general_testing.py --mtl /global/project/projectdirs/desi/users/arroyoc/pip_scripts/parallel_pip_bianchi/mtlz-patch-thesis-version.fits --stdstar-dark /global/project/projectdirs/desi/users/jguy/mocks/darksky-v1.0.1-v3/targets-0/standards-dark.fits --stdstar-bright /global/project/projectdirs/desi/users/jguy/mocks/darksky-v1.0.1-v3/targets-0/standards-bright.fits --sky /global/project/projectdirs/desi/users/jguy/mocks/darksky-v1.0.1-v3/targets-0/sky.fits --tiles /global/project/projectdirs/desi/users/arroyoc/pip_scripts/parallel_pip_bianchi/test-tiles.fits --rank-group 0


# --------------------------------------------------------------------------------------------------
# READING AND LOADING THE RELEVANT DATA (FIRST PART)
#
# In this part of the code the main objects to be used are defined and the data to make the
# assignation process is loaded.
# --------------------------------------------------------------------------------------------------

# Read hardware properties.
hw = load_hardware()

# Read the tiles to be used and define the number of tiles. Path obtained from command line 
# argument.
if (tilesfile is None):
    nominal_tiles = load_tiles()
else:
    nominal_tiles = load_tiles(tiles_file=tilesfile)

ntiles = len(nominal_tiles.id)

# Container for the Mersenne Twister pseudo-random number generator. Since every MPI rank is seeded 
# with a different number, size number of different subpriorities are generated.
#seed = 62*rank_group + rank
seed = rank
random_generator = RandomState(seed=seed)

# Load Target data.
tgs = Targets()

# First load the MTL file and compute the target IDs.
load_target_file(tgs, mtlfile, random_generator=random_generator)

# --------------------------------------------------------------------------------------------------
# TARGET IDs IN PARALLEL
#
# Target IDs for each tracer in parallel.
# The total number of targets is split almost evenly between all processes. Each MPI task takes a
# portion of the total targets and extract the corresponding target IDs for each type of tracer.
# --------------------------------------------------------------------------------------------------

# Total number of targets.
ntargets = len(tgs.ids())

# Dictionary with DESI bitmask values for LRGs, ELGs and QSOs.
desi_bitmask = {'lrg':65537, 'elg':131074, 'qso':262148}

# Targets that each MPI task will process (except for the last one).
targets_per_process = ntargets // size

# Extraction of target IDs for each tracer of each process.
if rank == size - 1:
    initial_index = rank * targets_per_process
    lrg_targets_ids_local = np.array([tid for tid in tgs.ids()[initial_index:] if \
                                      (tgs.get(tid).desi_target == desi_bitmask['lrg'])],
                                     dtype=np.int64)
    elg_targets_ids_local = np.array([tid for tid in tgs.ids()[initial_index:] if \
                                      (tgs.get(tid).desi_target == desi_bitmask['elg'])],
                                     dtype=np.int64)
else:
    initial_index = rank * targets_per_process
    final_index = (rank + 1) * targets_per_process
    lrg_targets_ids_local = np.array([tid for tid in tgs.ids()[initial_index:final_index] if \
                                      (tgs.get(tid).desi_target == desi_bitmask['lrg'])],
                                     dtype=np.int64)
    elg_targets_ids_local = np.array([tid for tid in tgs.ids()[initial_index:final_index] if \
                                      (tgs.get(tid).desi_target == desi_bitmask['elg'])],
                                     dtype=np.int64)

# Number of targets per tracer each process obtained.
nlrg_local = len(lrg_targets_ids_local)
nelg_local = len(elg_targets_ids_local)

# Create vector containing the number of targets per tracer each process obtained.
nlrg_local = comm.allgather(nlrg_local)
nelg_local = comm.allgather(nelg_local)

# Synchronize all processes.
comm.Barrier()

# Displacement array. Needed for use in Allgatherv.
displsarray_lrg = [0]
displsarray_elg = [0]

# Fill the displacement arrays.
for i in range(1, size):
    displsarray_lrg.append(displsarray_lrg[i-1] + nlrg_local[i-1])
    displsarray_elg.append(displsarray_elg[i-1] + nelg_local[i-1])

# Total number of targets for each type of tracer.
total_lrg_targets = displsarray_lrg[-1] + nlrg_local[-1]
total_elg_targets = displsarray_elg[-1] + nelg_local[-1]

# Set up the size of each vector to be gathered and the displacements.
nlrg_local = tuple(nlrg_local)
nelg_local = tuple(nelg_local)

displsarray_lrg = tuple(displsarray_lrg)
displsarray_elg = tuple(displsarray_elg)

# Buffer to save the target IDs for each tracer type.
lrg_targets_ids = np.empty([total_lrg_targets], dtype=np.int64)
elg_targets_ids = np.empty([total_elg_targets], dtype=np.int64)

# Every process contains the vector of all of the target IDs for each type of tracer.
# Synchronize.
comm.Allgatherv(lrg_targets_ids_local,
                [lrg_targets_ids, nlrg_local, displsarray_lrg, MPI.LONG_LONG])
comm.Allgatherv(elg_targets_ids_local,
                [elg_targets_ids, nelg_local, displsarray_elg, MPI.LONG_LONG])
comm.Barrier()

if rank == 0:
    lrg_targets_ids_filename = os.path.join(outdir, 'lrg_targets_ids_{}.npy'.format(label))
    elg_targets_ids_filename = os.path.join(outdir, 'elg_targets_ids_{}.npy'.format(label))
    
    if(not(os.path.exists(outdir))):
        os.mkdir(outdir)
    
    if(not(os.path.exists(lrg_targets_ids_filename))):
        np.save(lrg_targets_ids_filename, lrg_targets_ids)
    
    if(not(os.path.exists(elg_targets_ids_filename))):
        np.save(elg_targets_ids_filename, elg_targets_ids)


# --------------------------------------------------------------------------------------------------
# READING AND LOADING THE RELEVANT DATA (SECOND PART)
#
# In this part of the code the main objects to be used are defined and the data to make the
# assignation process is loaded.
# --------------------------------------------------------------------------------------------------

# Target files paths for sky fibers and standard stars. Obtained from command line arguments.
target_files = [sky, stdstar1, stdstar2]

for tgfile in target_files:
    if(not(tgfile is None)):
        load_target_file(tgs, tgfile, random_generator=random_generator)
    
# fiberassign.targets.Targets is a class representing a list of targets. This is always created 
# first and then the targets are appended to this object. This is done with the foor loop which 
# actually appends the standard stars, sky fibers and mtl to one instance.


# --------------------------------------------------------------------------------------------------
# CREATING OBJECTS FOR FIBER ASSIGNMENT PROCESS
#
# Tree objects are computed as well as targets available to each fiber and fibers available to each
# target. Object carrying the assignation is also created.
# --------------------------------------------------------------------------------------------------

# Create a hierarchical triangle mesh lookup of the targets positions
tree = TargetTree(tgs)

# Compute the targets available to each fiber for each tile.
tgsavail = TargetsAvailable(hw, tgs, nominal_tiles, tree)

# Compute the fibers on all tiles available for each target
favail = FibersAvailable(tgsavail)

# Create assignment object
asgn = Assignment(tgs, tgsavail, favail)


# --------------------------------------------------------------------------------------------------
# FIBER ASSIGNMENT PROCESS
#
# Fiber assignment process is carried out here. Results are stored in the Assignment type object.
# --------------------------------------------------------------------------------------------------

# First-pass assignment of science targets
asgn.assign_unused(TARGET_TYPE_SCIENCE)

# Redistribute science targets across available petals
asgn.redistribute_science()

# Assign standards, 10 per petal
asgn.assign_unused(TARGET_TYPE_STANDARD, 10)
asgn.assign_force(TARGET_TYPE_STANDARD, 10)

# Assign sky, up to 40 per petal
asgn.assign_unused(TARGET_TYPE_SKY, 40)
asgn.assign_force(TARGET_TYPE_SKY, 40)

# If there are any unassigned fibers, try to place them somewhere.
asgn.assign_unused(TARGET_TYPE_SCIENCE)
asgn.assign_unused(TARGET_TYPE_SKY)


# --------------------------------------------------------------------------------------------------
# EXTRACTING THE ASSIGNED TARGET IDS
#
# In this part the assigned target IDs are extracted from the relevant objects. Each realization is
# saved in .npy format.
# --------------------------------------------------------------------------------------------------

# List to save the assigned target IDs.
assigned_target_ids = []
    
##  Loop over assigned tiles and update our bit array for this realization. 
for i, tile in enumerate(nominal_tiles.id):

    ##  Assigned target data. 
    tdata           = asgn.tile_fiber_target(tile)
       
    ##  We are updating only assigned targets, line 206 of
    ## https://github.com/desihub/fiberassign/blob/master/py/fiberassign/assign.py
    #L214
    # Find the assigned target ids
    tgids           = [y for x, y in tdata.items()]
    
    # Append to the whole list
    assigned_target_ids = assigned_target_ids + tgids

# Convert to numpy array.
assigned_target_ids = np.array(assigned_target_ids, dtype = np.int64)
assigned_target_ids = np.unique(assigned_target_ids)

# Save the assigned target IDs.
#asgn_dir = ('/global/cscratch1/sd/caac_a/pip_new_version_test/output_for_crossco/'
#            'assignation_files/')
#assignation_file = asgn_dir + 'assignation_{}.npy'.format(seed)
#np.save(assignation_file, assigned_target_ids)


# --------------------------------------------------------------------------------------------------
# ENCODING THE RESULTS
#
# In this part, the assignation process is encoded in Bianchi's code format.
# --------------------------------------------------------------------------------------------------

## Create result array for each of the tracers
## If another dtype of less bits than int32 is used, inconsistent results are drawn.
lrg_result = np.isin(lrg_targets_ids, assigned_target_ids).astype(dtype=np.int32)
elg_result = np.isin(elg_targets_ids, assigned_target_ids).astype(dtype=np.int32)
#qso_result = np.isin(qso_targets_ids, assigned_target_ids).astype(dtype=np.int32)
    
## This initializes the other communicator.
## This communicator just split the processes in groups
## of processes_per_integer processes.
## The idea is to gather the _result arrays and pack the bits
## in 32 bits integers just by converting a binary number
## to its decimal representation.
processes_per_integer = 31
color = rank // processes_per_integer
comm_4bytes = MPI.Comm.Split(comm, color, rank)
rank_4bytes = comm_4bytes.Get_rank()
size_4bytes = comm_4bytes.Get_size()

# Creating buffers for results to be saved.
if(rank_4bytes == 0):
    # If another dtype of less bits than int32 is used, inconsistent results are drawn.
    lrg_bitw8 = np.empty([size_4bytes, total_lrg_targets], dtype=np.int32)
    elg_bitw8 = np.empty([size_4bytes, total_elg_targets], dtype=np.int32)
    #qso_bitw8 = np.empty([size_4bytes, total_qso_targets], dtype=np.int32)
else:
    lrg_bitw8 = None
    elg_bitw8 = None
    #qso_bitw8 = None

# Gathering the information in groups of 31 realizations.
comm_4bytes.Gather(lrg_result, lrg_bitw8, root=0)
comm_4bytes.Gather(elg_result, elg_bitw8, root=0)
#comm_4bytes.Gather(qso_result, qso_bitw8, root=0)

# Encoding the information in groups of 31 realizations.
if(rank_4bytes == 0):
    if(size < 31):
        encoding_powers = np.power(2, np.arange(0, size, dtype=np.int32), dtype=np.int32)
    else:
        encoding_powers = np.power(2, np.arange(0, 31, dtype=np.int32), dtype=np.int32)
    lrg_bitw8 = lrg_bitw8.T
    lrg_bitw8 = np.multiply(lrg_bitw8, encoding_powers, dtype=np.int32)
    lrg_bitw8 = np.sum(lrg_bitw8, axis=1, dtype=np.int32)
    
    elg_bitw8 = elg_bitw8.T
    elg_bitw8 = np.multiply(elg_bitw8, encoding_powers, dtype=np.int32)
    elg_bitw8 = np.sum(elg_bitw8, axis=1, dtype=np.int32)
    
    #qso_bitw8 = qso_bitw8.T
    #qso_bitw8 = np.multiply(qso_bitw8, encoding_powers, dtype=np.int32)
    #qso_bitw8 = np.sum(qso_bitw8, axis=1, dtype=np.int32)

# Ranks which contain the encoded 32-bit integers.
boss_ranks = [i*size_4bytes for i in range(size//size_4bytes)]    
n_boss_ranks = len(boss_ranks)

## Here I define another communicator.
## This communicator just group the boss processes which contain
## the packed information.
boss_group = comm.group.Incl(boss_ranks)
boss_comm = comm.Create_group(boss_group)

# Creating buffers to receive the whole information.
if rank in boss_ranks:
    if(rank == 0):
        lrg_result_32bitarray = np.empty([n_boss_ranks, total_lrg_targets], dtype=np.int32)
        elg_result_32bitarray = np.empty([n_boss_ranks, total_elg_targets], dtype=np.int32)
        #qso_result_32bitarray = np.empty([n_boss_ranks, total_qso_targets], dtype=np.int32)
    else:
        lrg_result_32bitarray = None
        elg_result_32bitarray = None
        #qso_result_32bitarray = None

# Gathering the whole encoded information per tracer.
if rank in boss_ranks:
    boss_comm.Gather(lrg_bitw8, lrg_result_32bitarray, root=0)
    boss_comm.Gather(elg_bitw8, elg_result_32bitarray, root=0)
    #boss_comm.Gather(qso_bitw8, qso_result_32bitarray, root=0)

if(rank == 0):
    lrg_result_32bitarray = lrg_result_32bitarray.T
    elg_result_32bitarray = elg_result_32bitarray.T
    #qso_result_32bitarray = qso_result_32bitarray.T
    
    lrg_bitweight_file = os.path.join(outdir, 'pip_weights_lrg_{}.txt'.format(label))
    elg_bitweight_file = os.path.join(outdir, 'pip_weights_elg_{}.txt'.format(label))
    #qso_bitweight_file = os.path.join(outdir, 'pip_weights_qso_{}.txt'.format(label))
    
    np.savetxt(lrg_bitweight_file, lrg_result_32bitarray, fmt='%d')
    np.savetxt(elg_bitweight_file, elg_result_32bitarray, fmt='%d')
    #np.savetxt(qso_bitweight_file, qso_result_32bitarray, fmt='%d')