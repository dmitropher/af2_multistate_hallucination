import sys

import numpy as np
from json import load

sys.path.append("/software/pyrosetta3.8/latest/")
import pyrosetta

from file_io import dummy_pdbfile
from loss_factory import Loss, get_loss_creator
from simple_losses import weightedLoss, dualLoss, maxLoss
from transform_decomposition import (
    helical_axis_data,
    helical_axis_to_rise_rotation_radius_axis,
)


class CyclicSymmLoss(Loss):
    """
    Value is the SAP score of the protein oligo when loaded
    """

    def __init__(self, oligo_obj=None, **params):
        super().__init__(oligo_obj=None, **params)
        self.oligo = oligo_obj
        pyrosetta.distributed.maybe_init()
        self._n_repeats = len(self.oligo.subunits)
        self.value = self.compute()
        self._information_string = f"""This loss computes deviation from ideal cyclic.
        Score rescales it between 0-1, lower is better"""

    def compute(self):

        dummy = dummy_pdbfile(self.oligo)
        dummy_path = dummy.name
        pose = pyrosetta.pose_from_file(dummy_path)

        s, C, theta, d2, dstar = helical_axis_data(pose, self._n_repeats)
        self._params_dict = {
            "axis_direction": s,
            "axis_point": C,
            "rotation_about": theta,
            "d2": d2,
            "dstar": dstar,
            "rise": np.dot(d2 / np.linalg.norm(d2), d2 + dstar),
        }

        self.value = [
            self._params_dict["axis_direction"],
            self._params_dict["axis_point"],
            self._params_dict["rotation_about"],
            self._params_dict["d2"],
            self._params_dict["dstar"],
            self._params_dict["rise"],
        ]

        return self.value

    def score(self):
        self.compute()
        d_rotation = np.degrees(
            abs(
                self._params_dict["rotation_about"]
                - (np.pi * 2 / self._n_repeats)
            )
        )

        rescaled_theta = self.logistic_rescale(4, 1, 1.5, val=d_rotation)

        rescaled_rise = self.logistic_rescale(
            2, 1, 2, val=abs(self._params_dict["rise"])
        )

        return (rescaled_rise + rescaled_theta) / 2

    def get_base_values(self):
        name_dict = {self.loss_name: self.score()}
        data_dict = {
            "d2": np.linalg.norm(self._params_dict["d2"]),
            "dstar": np.linalg.norm(self._params_dict["dstar"]),
            "rise": np.linalg.norm(self._params_dict["rise"]),
            "d_rotation": np.degrees(
                abs(
                    self._params_dict["rotation_about"]
                    - (np.pi * 2 / self._n_repeats)
                )
            ),
            "raw_rotation": np.degrees(self._params_dict["rotation_about"]),
        }

        all_dict = {**data_dict, **name_dict}
        return all_dict


class CyclicParamLoss(Loss):
    """
    Value is the SAP score of the protein oligo when loaded
    """

    def __init__(self, oligo_obj=None, **params):
        super().__init__(oligo_obj=None, **params)
        self.oligo = oligo_obj
        pyrosetta.distributed.maybe_init()
        self._n_repeats = len(self.oligo.subunits)
        self.value = self.compute()
        config = params["loss_params"]
        json_path = config[0]
        with open(json_path, "r") as f:
            self._helical_param_reference = load(f)
        self._information_string = f"""This loss computes deviation from ideal cyclic.
        Score rescales it between 0-1, lower is better"""

    def compute(self):

        dummy = dummy_pdbfile(self.oligo)
        dummy_path = dummy.name
        pose = pyrosetta.pose_from_file(dummy_path)

        s, C, theta, d2, dstar = helical_axis_data(pose, self._n_repeats)
        rise, rotation, s, C, radius = helical_axis_to_rise_rotation_radius_axis(
            s, C, theta, d2, dstar
        )
        self._params_dict = {
            "axis_direction": s,
            "axis_point": C,
            "rota": rotation,
            "d2": d2,
            "dstar": dstar,
            "rise": rise,
            "radi": radius,
        }

        self.value = [
            self._params_dict["axis_direction"],
            self._params_dict["axis_point"],
            self._params_dict["rota"],
            self._params_dict["d2"],
            self._params_dict["dstar"],
            self._params_dict["rise"],
            self._params_dict["radi"],
        ]

        return self.value

    def score(self):
        self.compute()
        deltas_dict = {}
        for key in self._helical_param_reference.keys():

            if key in self._params_dict.keys():

                delta_val = abs(
                    self._params_dict[key] - self._helical_param_reference[key]
                )
                deltas_dict[key] = delta_val
        deltas_keys = deltas_dict.keys()
        rescaled_list = []
        if "rota" in deltas_keys:
            rescaled_theta = self.logistic_rescale(
                5, 1, 0.6, val=np.degrees(deltas_dict["rota"])
            )
            rescaled_list.append(rescaled_theta)
        if "rise" in deltas_keys:
            rescaled_rise = self.logistic_rescale(
                2, 1, 1.5, val=deltas_dict["rise"]
            )
            rescaled_list.append(rescaled_rise)
        if "radi" in deltas_keys:
            rescaled_rise = self.logistic_rescale(
                2, 1, 1.5, val=deltas_dict["radi"]
            )
            rescaled_list.append(rescaled_rise)

        return sum(rescaled_list) / len(rescaled_list)

    def get_base_values(self):
        name_dict = {self.loss_name: self.score()}
        data_dict = {
            "rise": self._params_dict["rise"],
            "rota": np.degrees(self._params_dict["rota"]),
            "radi": self._params_dict["radi"],
        }
        target_dict = {}
        for key in (["rise","rota","radi"]):
            val = self._helical_param_reference.get(key)
            if not (val is None):
                new_key = f"target_{key}"
                target_dict[new_key] = val

        all_dict = {**data_dict, **name_dict, **target_dict}
        return all_dict


class CyclicPlusDual(weightedLoss):
    def __init__(
        self, loss_name="cyclic_dual_2t1", oligo_obj=None, **user_kwargs
    ):
        """
        init for this combined loss, dual is downweighted to half of cyclic
        """
        super().__init__(
            dualLoss(oligo_obj=oligo_obj, loss_name="dual"),
            CyclicSymmLoss(
                oligo_obj=oligo_obj, loss_name="cyclic_loss", **user_kwargs
            ),
            weights={"cyclic_loss": (2 / 3), "dual": (1 / 3)},
            even=False,
            invert=False,
            **user_kwargs,
        )
        self._information_string = f"""Three part loss, cyclic, plddt, ptm. Cyclic is 2:1 weighted above the average of the other two"""


class SAPLoss(Loss):
    """
    Value is the SAP score of the protein oligo when loaded
    """

    def __init__(self, oligo_obj=None, **params):
        super().__init__(oligo_obj=None, **params)
        self.oligo = oligo_obj
        # self._rosetta_flags_string = (
        #     params["rosetta_flags_string"]
        #     if "rosetta_flags_string" in params.keys()
        #     else None
        # )
        pyrosetta.distributed.maybe_init()
        #     ""
        #     if self._rosetta_flags_string is None
        #     else self._rosetta_flags_string
        # )
        self.value = self.compute()
        self._information_string = f"""This loss computes total sap for the molecule.
        Score rescales it between 0-1, lower is better (less sap)"""

    def compute(self):

        dummy = dummy_pdbfile(self.oligo)
        dummy_path = dummy.name
        pose = pyrosetta.pose_from_file(dummy_path)

        true_selector = (
            pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
        )

        self.value = pyrosetta.rosetta.core.pack.guidance_scoreterms.sap.calculate_sap(
            pose,
            true_selector.clone(),
            true_selector.clone(),
            true_selector.clone(),
        )

        return self.value

    def score(self):
        mid = 75
        max_val = 1
        steep = 0.08
        rescaled = max_val / (1 + np.exp(-1 * steep * (self.value - mid)))
        return rescaled


class SAPPlusDual(weightedLoss):
    def __init__(
        self, loss_name="sap_plddt_ptm_equal", oligo_obj=None, **user_kwargs
    ):
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
    def __init__(
        self, loss_name="sap_plddt_ptm_max", oligo_obj=None, **user_kwargs
    ):
        """

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
    "cyclic_symm_loss": CyclicSymmLoss,
    "cyclic_dual_2t1": CyclicPlusDual,
    "helical_param_loss": CyclicParamLoss,
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
