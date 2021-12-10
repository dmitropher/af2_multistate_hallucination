# losses module

import numpy as np
import sys; sys.path.append('/projects/ml/alphafold/alphafold_git/')
from alphafold.common import protein
# dssp loss imports
from Bio.PDB.DSSP import DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
# to run tmalign
import subprocess
from scipy import linalg




from scoring import scores_from_loss

from loss_factory import get_creator_from_dicts

from simple_losses import get_loss_dict as get_simple_loss_dict
from rosetta_losses import get_loss_dict as get_rosetta_loss_dict

############################
# LOSS COMPUTATION
############################


def compute_loss(loss_names, oligo, args, loss_weights, score_container=None):
    """
    Compute the loss of a single oligomer.
    losses: list of list of losses and their associated arguments (if any).
    oligo: an Oligomer object.
    args: the whole argument namespace (some specific arguments are required for some specific losses).
    loss_weights: list of weights associated with each loss.
    """
    # intialize scores
    scores = []
    # iterate over all losses
    # args_dict = vars(args)

    main_creator = get_creator_from_dicts(
        get_simple_loss_dict(), get_rosetta_loss_dict()
    )

    for loss_idx, current_loss in enumerate(loss_names):
        args_dict = vars(args)
        loss_type, loss_params = current_loss
        args_dict["loss_params"] = loss_params
        loss_obj = main_creator.get_loss(
            loss_type, oligo_obj=oligo, **args_dict
        )
        print(f"{loss_obj}")
        score = loss_obj.score()

        #TODO: turn into loss object
        elif loss_type == 'aspect_ratio':
            # NOTE:
            # This loss adds a geometric term that forces an aspect ratio of 1 (spherical protomers) to prevent extended structures.
            # At each step, the PDB is generated, and a singular value decomposition is performed on the coordinates of the CA atoms.
            # The ratio of the two largest values is taken as the aspect ratio of the protomer.
            # For oligomers, the aspect ratio is calculated for each protomer independently, and the average returned.

            c = get_coord('CA', oligo) # get CA atoms, returns array [[chain, resid, x, y, z]]
            aspect_ratios = []
            chains = set(c[:,0])
            for ch in chains:
                coords = np.array([a[2:][0] for a in c[c[:,0]==ch]])
                coords -= coords.mean(axis=0) # mean-center the protomer coordinates
                s = linalg.svdvals(coords) # singular values of the coordinates
                aspect_ratios.append(s[1] / s[0])

            score = 1. - np.mean(aspect_ratios) # average aspect ratio across all protomers of an oligomer

        scores.append(score)

        print(f"loss_type: {loss_type}")
        print(f"scores: {scores}")
        if score_container is not None:
            score_container.add_scores(*scores_from_loss(loss_obj))

    # Normalize loss weights vector.
    loss_weights_normalized = np.array(loss_weights) / np.sum(loss_weights)

    # Total loss for this oligomer is the sum of its weighted scores.
    final_score = np.sum(np.array(scores) * loss_weights_normalized)

    # The loss counts positively or negatively to the overall loss depending on whether this oligomer is positively or negatively designed.
    if oligo.positive_design == True:
        loss = float(final_score)
    else:
        loss = float(final_score) + 1

    return loss
