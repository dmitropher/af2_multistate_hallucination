#!/software/conda/envs/pyrosetta/bin/python
# Get resid (indexed at zero) of surface residues and print these to a file for use with AF2_MCMC_2state.py

import pyrosetta
from pyrosetta.rosetta.core.select.residue_selector import LayerSelector
pyrosetta.init()
import numpy as np
import sys

def layer_selector_mover(layer, core_cutoff=3.0, surface_cutoff=2.0): # code from Hugh

    """
    Set up a PyRosetta Mover that can be used to select a specific layer using the
    side-chain neighbor algorithm

    Args:
        `layer` (string) : the layer to be selected. This variable can be
            "core", "boundary", or "surface".
        `core_cutoff` (float) : the cutoff used to define the core using
            the side-chain neighbor algorithm. Residues with at least
            this many neighbors are considered to be in the core.
        `surface_cutoff` (float) : the cutoff used to define the surface
            using the side-chain neighbor algorithm. Residues with fewer
            than this many neighbors are considered to be on the surface.

    Returns:
        `select_layer` (PyRosetta mover) : a PyRosetta LayerSelector
            Mover that can be applied be applied to a pose to select the
            layer specified by the input arguments.
    """
    # Initiate the mover, defining layer cutoffs
    select_layer = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
    select_layer.set_use_sc_neighbors(True)
    select_layer.set_cutoffs(core=core_cutoff, surf=surface_cutoff)

    # Select the specific layer given the input
    layer_bools = {
        'core' : [True, False, False],
        'boundary' : [False, True, False],
        'surface' : [False, False, True]
    }

    pick_core, pick_boundary, pick_surface = layer_bools[layer]
    select_layer.set_layers(
        pick_core=pick_core, pick_boundary=pick_boundary,
        pick_surface=pick_surface
    )
    
    return select_layer

surface_selector = layer_selector_mover('surface')

pdbs = sys.argv[1:]

for pdb in pdbs:
    
    pose = pyrosetta.pose_from_pdb(pdb)
    surf_pos = np.array(np.arange(0, len(pose.sequence()))[np.array(surface_selector.apply(pose))])
    surf_pos
    
    with open(pdb.replace('.pdb', '.res'), 'w') as f:
        f.write(pose.sequence() + '\n')
        f.write(' '.join(surf_pos.astype(str)))
