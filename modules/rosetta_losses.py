import sys

import numpy as np

sys.path.append("/software/pyrosetta3.8/latest/")
import pyrosetta

from file_io import dummy_pdbfile
from loss_factory import Loss, get_loss_creator
from simple_losses import weightedLoss, dualLoss, maxLoss


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


class SAPPlusDual(weightedLoss):
    def __init__(self, oligo_obj=None, **user_kwargs):
        """
        init for this combined loss

        The default args for weights in the super init take into account that dual
        is an equally weighted combined score from two values, hence the 1/3,2/3 instead of "even"
        """
        super().__init__(
            dualLoss(oligo_obj=oligo_obj, loss_name="dual"),
            SAPLoss(oligo_obj=oligo_obj, loss_name="sap_loss", **user_kwargs),
            weights={"sap_loss": (1 / 3), "dual": (2 / 3)},
            even=False,
            invert=False,
            **user_kwargs,
        )
        self._information_string = f"""Three part equally weighted loss, using SAP, pLDDT, and ptm
        0-1, lower is better.
        SAP is weighted according to the SAPLoss object config. The SAPLoss component inherits kwargs from this at time of writing"""


class MaxSAPDual(maxLoss):
    def __init__(self, oligo_obj=None, **user_kwargs):
        """
        init for this combined loss

        The default args for weights in the super init take into account that dual
        is an equally weighted combined score from two values, hence the 1/3,2/3 instead of "even"
        """
        super().__init__(
            dualLoss(oligo_obj=oligo_obj, loss_name="dual"),
            SAPLoss(oligo_obj=oligo_obj, loss_name="sap_loss", **user_kwargs),
            invert=False,
            **user_kwargs,
        )
        self._information_string = f"""Three part loss, using SAP, pLDDT, and ptm: takes the worst from the three
        0-1, lower is better.
        SAP is weighted according to the SAPLoss object config. The SAPLoss component inherits kwargs from this at time of writing"""


global_losses_dict = {
    "sap_loss": SAPLoss,
    "sap_plddt_ptm_equal": SAPPlusDual,
    "sap_plddt_ptm_max": MaxSAPDual,
}


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
