# losses module

import numpy as np

# import sys


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
    args_dict = vars(args)

    main_creator = get_creator_from_dicts(
        get_simple_loss_dict(), get_rosetta_loss_dict()
    )

    for loss_idx, current_loss in enumerate(loss_names):
        loss_type, loss_params = current_loss
        args_dict["loss_params"] = loss_params
        loss_obj = main_creator.get_loss(
            loss_type, oligo_obj=oligo, **args_dict
        )
        print(f"{loss_obj}")
        score = loss_obj.score()

        scores.append(score)
        if score_container is not None:
            score_container.add_scores(*scores_from_loss(loss_obj))

    # Normalize loss weights vector.
    loss_weights_normalized = np.array(loss_weights) / np.sum(loss_weights)

    # Total loss for this oligomer is the average of its weighted scores.
    final_score = np.mean(np.array(scores) * loss_weights_normalized)

    # The loss counts positively or negatively to the overall loss depending on whether this oligomer is positively or negatively designed.
    if oligo.positive_design == True:
        loss = float(final_score)
    else:
        loss = float(final_score) + 1

    return loss
