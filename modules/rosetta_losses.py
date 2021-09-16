import numpy as np

import pyrosetta

from file_io import dummy_pdbfile
from loss_factory import Loss, get_loss_creator


class SAPLoss(Loss):
    """
    Value is the SAP score of the protein oligo when loaded
    """

    def __init__(self, oligo=None, **params):
        self._rosetta_flags_string = (
            params["rosetta_flags_string"]
            if "rosetta_flags_string" in params.keys()
            else None
        )
        pyrosetta.distributed.maybe_init(
            ""
            if self._rosetta_flags_string is None
            else self._rosetta_flags_string
        )
        self.value = self.compute()
        self._information_string(
            f"""This loss computes total sap for the molecule.
        Score rescales it between 0-1, higher is better (less sap)"""
        )

    def compute(self):

        dummy = dummy_pdbfile(self.oligo)
        dummy_path = dummy.name
        pose = pyrosetta.pose_from_file(dummy_path)

        true_selector = (
            pyrosetta.rosetta.core.select.residue_selector.TrueSelector()
        )

        self.value = pyrosetta.rosetta.core.pack.guidance_scoreterms.sap.calculate_sap(
            pose,
            true_selector.clone(),
            true_selector.clone(),
            true_selector.clone(),
        )

        return self.value

    def score(self):
        mid = 30
        max_val = 1
        steep = 0.15
        rescaled = max_val / (1 + np.exp(-1 * steep * (self.value - mid)))
        return 1 - rescaled


global_losses_dict = {"sap_loss": SAPLoss}


def get_loss_dict():
    return global_losses_dict


def get_global_creator():
    global_creator = get_loss_creator(**get_loss_dict())
    return global_creator


def get_loss(loss_name, **loss_params):
    return get_global_creator().get_loss(loss_name, **loss_params)


def get_allowed_rosetta_loss_names():
    """
    Returns the DSSP string values registered with the current creator
    """
    from copy import deepcopy

    return deepcopy(get_global_creator()._creators)
