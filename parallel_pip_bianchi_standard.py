#!/usr/bin/env python
# This looks for the Python executable in $PATH environment variable.

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

# Paths for the necessary files to run fiber assignment.
# Perhaps a better interface for this is needed.
mtlfile = '/global/project/projectdirs/desi/users/arroyoc/pip_scripts/parallel_pip_bianchi/mtlz-patch.fits'
stdstar1 = '/global/project/projectdirs/desi/users/jguy/mocks/darksky-v1.0.1-v2/targets-0/standards-dark.fits'
stdstar2 = '/global/project/projectdirs/desi/users/jguy/mocks/darksky-v1.0.1-v2/targets-0/standards-bright.fits'
sky = '/global/project/projectdirs/desi/users/jguy/mocks/darksky-v1.0.1-v2/targets-0/sky.fits'
tilesfile = '/global/project/projectdirs/desi/users/arroyoc/pip_scripts/parallel_pip_bianchi/test-tiles.fits'

# Read hardware properties.
hw = load_hardware()

# Read the nominal footprint.
nominal_tiles = load_tiles(tiles_file=tilesfile)
ntiles = len(nominal_tiles.id)

# In this case this is only done for our test tiles.

# Target files. Get these from command line arguments, etc.
target_files = [mtlfile, sky, stdstar1, stdstar2]

# Variable for world communicator, number of MPI tasks and rank of the present
# MPI process.
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Container for the Mersenne Twister pseudo-random number generator.
# Since every MPI rank is seeded with a different number, size number of
# different subpriorities are generated.
random_generator = RandomState(seed=rank)

# Load Target data. There may be some contention from all
# the MPI processes reading at once, but this is only done
# one time at the start of the job.
tgs = Targets()
for tgfile in target_files:
    load_target_file(tgs, tgfile, random_generator=random_generator)
    
# fiberassign.targets.Targets is a class representing a list of targets.
# This is always created first and then the targets are appended to this
# object. This is done with the foor loop which actually appends the
# standard stars, sky fibers and mtl to one instance.

# Create a hierarchical triangle mesh lookup of the targets positions
tree = TargetTree(tgs)

# Compute the targets available to each fiber for each tile.
tgsavail = TargetsAvailable(hw, tgs, nominal_tiles, tree)

# Compute the fibers on all tiles available for each target
favail = FibersAvailable(tgsavail)

# All target IDs
targetids = tgs.ids()

# Find target IDs of reachable science DESI targets.
# Here I have all ranks do this.
# Another way would be computing in one rank and then broadcast the result.
# I don't know which one is faster.

# Dictionary with DESI bitmask values for LRGs, ELGs and QSOs.
desi_bitmask = {'lrg':65537, 'elg':131074, 'qso':262148}

# Find the target IDs for each target type.

# Target IDs for LRGs
lrg_targets_ids = np.array([tid for tid in targetids if (tgs.get(tid).desi_target == desi_bitmask['lrg'])],
                            dtype=np.int64)

# Target IDs for ELGs
#elg_targets_ids = np.array([tid for tid in targetids if (tgs.get(tid).desi_target == desi_bitmask['elg'])],
#                            dtype=np.int64)

# Target IDs for QSOs
#qso_targets_ids = np.array([tid for tid in targetids if (tgs.get(tid).desi_target == desi_bitmask['qso'])],
#                            dtype=np.int64)

# Function to flatten a list of lists.
# This ensures that 64 bits integers stay that type.
flatten = lambda l: [item for sublist in l for item in sublist]

# Iterate over tiles to find unique reachable DESI LRG targets.
for i, tile in enumerate(nominal_tiles.id):
    # Obtain data for tile tile.
    _tgsavail_tile  = tgsavail.tile_data(tile)
    
    # Find unique target ids in tile tile.
    tile_targets = np.unique(flatten(list(_tgsavail_tile.values())))
    
    # Mask to obtain only ELGs, LRGs and QSOs target IDs.
    lrg_mask = np.in1d(tile_targets, lrg_targets_ids)
    #elg_mask = np.in1d(tile_targets, elg_targets_ids)
    #qso_mask = np.in1d(tile_targets, qso_targets_ids)
    
    
    # Create a vector of reachable LRG, ELG and QSO targets
    if(i != 0):
        reachable_lrg_targets = np.concatenate([reachable_lrg_targets,
                                                tile_targets[lrg_mask]])
        #reachable_elg_targets = np.concatenate([reachable_elg_targets,
        #                                        tile_targets[elg_mask]])
        #reachable_qso_targets = np.concatenate([reachable_qso_targets,
        #                                        tile_targets[qso_mask]])
    else:
        reachable_lrg_targets = tile_targets[lrg_mask]
        #reachable_elg_targets = tile_targets[elg_mask]
        #reachable_qso_targets = tile_targets[qso_mask]

# Eliminate possible duplicate IDs because of overlapping tiles.
reachable_lrg_targets = np.unique(reachable_lrg_targets)
#reachable_elg_targets = np.unique(reachable_elg_targets)
#reachable_qso_targets = np.unique(reachable_qso_targets)

# Total number of reachable targets
total_lrg_targets = len(reachable_lrg_targets)
#total_elg_targets = len(reachable_elg_targets)
#total_qso_targets = len(reachable_qso_targets)

#if rank == 0:
#    np.save('/global/homes/c/caac_a/arroyoc/pip_scripts/parallel_pip_bianchi/lrg/reachable_lrg.npy',
#           reachable_science_targets)
    
# Create assignment object
asgn = Assignment(tgs, tgsavail, favail)

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

# ************ THIS PART HERE CAN BE IGNORED *******************
# Extract just the unique science IDs of LRGs
#science_assigned_target_ids = np.unique(assigned_target_ids[np.in1d(assigned_target_ids,
#                                                                   reachable_targets)])
    
#asgn_dir = '/global/homes/c/caac_a/arroyoc/parallel_pip_bianchi/assigned_targets/'
#assignation_file = asgn_dir + 'assignation_{}.npy'.format(rank)
#np.save(assignation_file, np.unique(assigned_target_ids))   
# **************************************************************



## Create result array for each of the tracers
lrg_result = np.isin(reachable_lrg_targets, assigned_target_ids).astype(dtype=np.int32)
#elg_result = np.isin(reachable_elg_targets, assigned_target_ids).astype(dtype=np.int32)
#qso_result = np.isin(reachable_qso_targets, assigned_target_ids).astype(dtype=np.int32)
    
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
    
if(rank_4bytes == 0):
    lrg_bitw8 = np.empty([size_4bytes, total_lrg_targets], dtype=np.int32)
    #elg_bitw8 = np.empty([size_4bytes, total_elg_targets], dtype=np.int32)
    #qso_bitw8 = np.empty([size_4bytes, total_qso_targets], dtype=np.int32)
else:
    lrg_bitw8 = None
    #elg_bitw8 = None
    #qso_bitw8 = None
    
comm_4bytes.Gather(lrg_result, lrg_bitw8, root=0)
#comm_4bytes.Gather(elg_result, elg_bitw8, root=0)
#comm_4bytes.Gather(qso_result, qso_bitw8, root=0)

if(rank_4bytes == 0):
    encoding_powers = np.power(2, np.arange(0, 31, dtype=np.int32), dtype=np.int32)
    lrg_bitw8 = lrg_bitw8.T
    lrg_bitw8 = np.multiply(lrg_bitw8, encoding_powers, dtype=np.int32)
    lrg_bitw8 = np.sum(lrg_bitw8, axis=1, dtype=np.int32)
    
    #elg_bitw8 = elg_bitw8.T
    #elg_bitw8 = np.multiply(elg_bitw8, encoding_powers, dtype=np.int32)
    #elg_bitw8 = np.sum(elg_bitw8, axis=1, dtype=np.int32)
    
    #qso_bitw8 = qso_bitw8.T
    #qso_bitw8 = np.multiply(qso_bitw8, encoding_powers, dtype=np.int32)
    #qso_bitw8 = np.sum(qso_bitw8, axis=1, dtype=np.int32)

boss_ranks = [i*size_4bytes for i in range(size//size_4bytes)]    
n_boss_ranks = len(boss_ranks)

## Here I define another communicator.
## This communicator just group the boss processes which contain
## the packed information.
boss_group = comm.group.Incl(boss_ranks)
boss_comm = comm.Create_group(boss_group)

if rank in boss_ranks:
    if(rank == 0):
        lrg_result_32bitarray = np.empty([n_boss_ranks, total_lrg_targets], dtype=np.int32)
        #elg_result_32bitarray = np.empty([n_boss_ranks, total_elg_targets], dtype=np.int32)
        #qso_result_32bitarray = np.empty([n_boss_ranks, total_qso_targets], dtype=np.int32)
    else:
        lrg_result_32bitarray = None
        #elg_result_32bitarray = None
        #qso_result_32bitarray = None

if rank in boss_ranks:
    boss_comm.Gather(lrg_bitw8, lrg_result_32bitarray, root=0)
    #boss_comm.Gather(elg_bitw8, elg_result_32bitarray, root=0)
    #boss_comm.Gather(qso_bitw8, qso_result_32bitarray, root=0)

if(rank == 0):
    lrg_result_32bitarray = lrg_result_32bitarray.T
    #elg_result_32bitarray = elg_result_32bitarray.T
    #qso_result_32bitarray = qso_result_32bitarray.T
    np.savetxt('/global/project/projectdirs/desi/users/arroyoc/pip_scripts/parallel_pip_bianchi/lrg/pip_weights_lrg.txt',
               result_32bitarray, fmt='%d')