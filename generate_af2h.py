#!/software/conda/envs/pyrosetta/bin/python
# Script to generate .af2h resfiles from pdb structures based on layer selection

import pyrosetta
from pyrosetta.rosetta.core.select.residue_selector import LayerSelector
pyrosetta.init()
import numpy as np
import sys

def layer_selector_mover(layer, core_cutoff=3.9, surface_cutoff=3.0): # code from Hugh

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


for pdb in  sys.argv[1:]:

    pose = pyrosetta.pose_from_pdb(pdb)
    core_selector = layer_selector_mover('core')
    boundary_selector = layer_selector_mover('boundary')
    surface_selector = layer_selector_mover('surface')

    core_pos = np.array(core_selector.apply(pose), dtype=int)
    boundary_pos = np.array(boundary_selector.apply(pose), dtype=int)
    surface_pos = np.array(surface_selector.apply(pose), dtype=int)

    mask = surface_pos + boundary_pos # select both surface and boundary residues

    id2chain = {1:'A', 2:'B'}
    with open(pdb.replace('.pdb','.af2h'), 'w') as f:
        for i, ch in id2chain.items():
            chain_seq = pose.chain_sequence(i)
            f.write('>' + ch + '\n')
            f.write(chain_seq + '\n')
            f.write(','.join(mask[pose.chain_begin(i)-1:pose.chain_end(i)].astype(str)) + '\n')
