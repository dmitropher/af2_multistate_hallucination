# losses module

import numpy as np

# import sys

from loss_factory import Loss, CombinedLoss, get_loss_creator
from loss_functions import (
    pae_sub_mat,
    pae_asym,
    get_separation_std,
    tm_score,
    dssp_wrapper,
    calculate_dssp_fractions,
)

from file_io import dummy_pdbfile

######################################################
# LOSS FACTORY FRAMEWORK
######################################################


def get_allowed_loss_names():
    """
    Returns the DSSP string values registered with the current creator
    """
    from copy import deepcopy

    return deepcopy(get_loss_creator()._creators)


class oligoLoss(Loss):
    """
    Thin wrapper to override the compute with an oligo accessor

    Requires oligo as positional argument
    """

    def __init__(self, oligo_obj=None, **user_kwargs):
        print(f"user_kwargs: {user_kwargs}")
        super().__init__(oligo_obj=oligo_obj, **user_kwargs)
        self.oligo = oligo_obj
        self.value = self.compute()
        self._information_string = f"""This loss object for: {self.loss_name} formats some data from the oligo object.
        Override this info when inheriting.
        """
        self.value = self.compute()

    def compute(self):
        """
        Overrides the default compute, accesses value from the oligo
        """
        if self.oligo is None:
            raise AssertionError(
                "Compute has been run on a without an oligo. This loss object is just a thin wrapper to read values from the Oligo object"
            )
        self.value = np.mean(self.oligo.try_prediction_results[self.loss_name])
        return self.value


class pLDDTLoss(oligoLoss):
    """
    """

    def __init__(self, oligo_obj=None, **user_kwargs):

        super().__init__(oligo_obj=oligo_obj, **user_kwargs)
        self.value = self.compute()
        self._information_string = f"""This loss object for: {self.loss_name} formats some data from the oligo object.
        NOTE:
        Using this loss will optimise plddt (predicted lDDT) for the sequence(s).
        Early benchmarks suggest that this is not appropriate for forcing the emergence of complexes.
        Optimised sequences tend to be folded (or have good secondary structures) without forming inter-chain contacts.
        Outputs mean plddt from oligo from 0 to 1, higher is better"""

    def score(self):
        return 1 - self.compute()


class ptmLoss(oligoLoss):
    """
    """

    def __init__(self, oligo_obj=None, **user_kwargs):

        super().__init__(oligo_obj=oligo_obj, **user_kwargs)
        self.value = self.compute()
        self._information_string = f"""This loss object for: {self.loss_name} formats some data from the oligo object.
        NOTE:
        Using this loss will optimise ptm (predicted TM-score) for the sequence(s).
        Early benchmarks suggest that while it does force the apparition of inter-chain contacts,
        it might be at the expense of intra-chain contacts and therefore folded protomer structures.
        Outputs mean ptm, 0 to 1, higher is better"""

    def score(self):
        return 1 - self.compute()


class paeLoss(oligoLoss):
    """
    """

    def __init__(self, oligo_obj=None, **user_kwargs):

        super().__init__(oligo_obj=oligo_obj, **user_kwargs)
        self.value = self.compute()
        self._information_string = f"""This loss object for: {self.loss_name} formats some data from the oligo object.
        Using this loss will optimise the mean of the pae matrix (predicted alignment error).
        This loss has not been properly benchmarked, but some early results suggest that it might suffer from the same problem as ptm.
        During optimisation, off-digonal contacts (inter-chain) may get optimsed at the expense of the diagonal elements (intra-chain).
        Outputs Mean pae: 0+ ,lower is better"""

    def compute(self):
        norm = np.mean(
            self.oligo.init_prediction_results["predicted_aligned_error"]
        )
        self.value = (
            np.mean(
                self.oligo.try_prediction_results["predicted_aligned_error"]
            )
            / norm
        )
        return self.value


class paeSubMatLoss(Loss):
    """
    """

    def __init__(self, oligo_obj=None, **user_kwargs):

        super().__init__(oligo_obj=oligo_obj, **user_kwargs)
        self.oligo = oligo_obj
        self.value = self.compute()
        self._information_string = f"""This loss object for: {self.loss_name} formats some data from the oligo object.
        This loss optimises the mean of the pae sub-matrices means.
        The value of the loss will be different to pae in the case of hetero-oligomers that have chains of different lenghts, but identical otherwise.
        The mean of the sub matrices' means is different from the overall mean if the sub matrices don't all have the same shape.
        Outputs Mean pae for sub matrices: 0+ ,lower is better"""

    def compute(self,):
        """
        Compute pae_sub_mat loss
        """
        self.value = pae_sub_mat(self.oligo)
        return self.value


class paeAsymLoss(oligoLoss):
    """
    """

    def __init__(self, oligo_obj=None, **user_kwargs):

        super().__init__(oligo_obj=oligo_obj, **user_kwargs)
        self.value = self.compute()
        self._information_string = f"""This loss object for: {self.loss_name} formats some data from the oligo object.
        NOTE:
        Using this loss will optimise the mean of the pae matrix (predicted alignment error).
        This loss has not been properly benchmarked, but some early results suggest that it might suffer from the same problem as ptm.
        During optimisation, off-digonal contacts (inter-chain) may get optimsed at the expense of the diagonal elements (intra-chain).
        Outputs Mean pae: 0+ ,lower is better"""

    def compute(self):
        self.value = pae_asym(self.oligo)
        return self.value


class weightedLoss(CombinedLoss):
    """
    A combined loss which weights the input values (evenly by default)

    inverts the sum by default ( score = 1 - sum(weighted_losses))
    """

    def __init__(
        self, *losses, even=True, invert=True, weights=None, **user_kwargs
    ):
        super().__init__(*losses, **user_kwargs)
        self.even = even
        self.invert = invert
        self.weights = weights
        if not even and weights is None:
            raise ValueError(
                "cannot use a weighted loss without weights if not set to evenly weight"
            )

        elif weights is None:
            n_losses = len(losses)
            self.weights = {loss.loss_name: 1 / n_losses for loss in losses}
        self.value = self.compute()
        self._information_string = f"""This loss object applies a weight to each input loss and sums them.
        Format of weights is a dict: {{loss_name:weight}}.
        included losses:
        {self.get_base_values().keys()}
        returns a weighted sum of the input losses"""

    def compute(self):
        self.value = sum(
            loss.value * self.weights[loss.loss_name] for loss in self.losses
        )
        return self.value

    def score(self):
        return 1 - self.value if self.invert else self.value


class maxLoss(CombinedLoss):
    """
    A combined loss which takes the biggest loss of the group

    inverts the sum by default ( score = 1 - max(*weighted_losses))
    """

    def __init__(self, *losses, invert=True, **user_kwargs):
        super().__init__(*losses, **user_kwargs)
        self.invert = invert

        self.value = self.compute()
        self._information_string = f"""This loss object only keeps the biggest loss.
        Can be used with the invert option to output 1-max(*losses) (default is invert).
        included losses:
        {self.get_base_values().keys()}
        returns the biggest of the input losses"""

    def compute(self):
        self.value = max(
            loss.value * self.weights[loss.loss_name] for loss in self.losses
        )
        return self.value

    def score(self):
        return 1 - self.value if self.invert else self.value


class minLoss(CombinedLoss):
    """
    A combined loss which outputs the smallest loss of the inputs

    inverts the min by default ( score = 1 - min(weighted_losses))
    """

    def __init__(self, *losses, invert=True, **user_kwargs):
        super().__init__(*losses, **user_kwargs)
        self.invert = invert

        self.value = self.compute()
        self._information_string = f"""This loss object only keeps the biggest loss.
        Can be used with the invert option to output 1-min(*losses) (default is invert!).
        included losses:
        {self.get_base_values().keys()}
        returns the smallest of the input losses"""

    def compute(self):
        self.value = min(
            loss.value * self.weights[loss.loss_name] for loss in self.losses
        )
        return self.value

    def score(self):
        return 1 - self.value if self.invert else self.value


class separationLoss(Loss):
    """
    Loss based on separation of units in the oligo (standard deviation)
    """

    def __init__(self, oligo_obj=None, **user_kwargs):

        super().__init__(oligo_obj=oligo_obj, **user_kwargs)
        self.value = self.compute()
        self._information_string = f"""This loss object for: {self.loss_name}.
        A geometric term that forces a cyclic symmetry. A PDB is generated, and the distance between the center of mass of adjacent units computed.
        The standard deviation of these neighbour distances is the loss.
        Score is separation dist std: 0+ ,lower is better"""

    def compute(self):
        self.value = get_separation_std(self.oligo)
        return self.value


class fracDSSPLoss(Loss):
    """
    Loss based on separation of units in the oligo (standard deviation)
    """

    def __init__(self, oligo_obj=None, **user_kwargs):

        super().__init__(oligo_obj=oligo_obj, **user_kwargs)
        config = user_kwargs["loss_params"]
        dssp_reference = {data[0]: float(data[1:]) for data in config}
        self.desired_dssp = dssp_reference
        if sum(self.desired_dssp.keys()) > 1:
            raise AttributeError(
                "Fractions dssp desired can not sum to greater than 1"
            )
        self.value = self.compute()
        self._information_string = f"""This loss object for: {self.loss_name}.
        This loss computes the deviation from the desired fraction dssp
        ranges 0-1 higher is better, near 0 is very very bad"""

    def compute(self):
        dummy = dummy_pdbfile(self.oligo)
        frac_beta, frac_alpha, frac_other = calculate_dssp_fractions(
            dssp_wrapper(dummy)
        )
        dummy.close()
        actual = {"E": frac_beta, "H": frac_alpha, "O": frac_other}
        chosen_fracs = self.desired_dssp.keys()
        self.value = 1 - sum(
            (abs(actual[key] - self.desired_dssp[key]) ** 2)
            / (len(chosen_fracs))
            for key in chosen_fracs
        )
        return self.value


class tmAlignLoss(Loss):
    """
    Loss based on separation of units in the oligo (standard deviation)
    """

    def __init__(self, oligo_obj=None, **user_kwargs):

        super().__init__(oligo_obj=oligo_obj, **user_kwargs)
        self.value = self.compute()
        self.template = user_kwargs["template"]
        self.template_alignment = user_kwargs["template_alignment"]
        self.temp_out = user_kwargs["temp_out"]
        self.value = self.compute()
        self._information_string = f"""This loss object for: {self.loss_name}.
        Computes TMalign to a user_provided tempalate
        Score is separation dist std: 0+ ,lower is better"""

    def compute(self):
        self.value = tm_score(
            self.oligo,
            self.template,
            self.template_alignment,
            temp_out=self.temp_out,
        )
        return self.value


class dualLoss(weightedLoss):
    """
    Loss invoked by the 'dual' option
    """

    def __init__(self, oligo_obj=None, weights=None, **user_kwargs):
        super().__init__(
            pLDDTLoss(oligo_obj=oligo_obj, loss_name="plddt"),
            ptmLoss(oligo_obj=oligo_obj, loss_name="ptm"),
            even=True,
            invert=True,
            **user_kwargs,
        )
        self._information_string = f"""This loss jointly optimises ptm and plddt (equal weights).
        It attemps to combine the best of both worlds -- getting folded structures that are in contact.
        This loss is currently recommended unless cyclic geometries are desired (tends to generate linear oligomers)."""


class dualCyclicLoss(weightedLoss):
    """
    Loss invoked by the 'dual_cyclic' option
    """

    def __init__(self, oligo_obj=None, weights=None, **user_kwargs):
        super().__init__(
            pLDDTLoss(oligo_obj=oligo_obj, loss_name="plddt"),
            ptmLoss(oligo_obj=oligo_obj, loss_name="ptm"),
            separationLoss(oligo_obj=oligo_obj, loss_name="separation"),
            even=False,
            invert=True,
            weights={"plddt": 2.0, "ptm": 2.0, "separation": -1.0},
            **user_kwargs,
        )
        self._information_string = f"""This loss jointly optimises ptm and plddt (equal weights).
        It attemps to combine the best of both worlds -- getting folded structures that are in contact.
        This is the cyclic version that includes a geometric term to minimize std between subunits."""


class dualTMAlignLoss(weightedLoss):
    def __init__(self, oligo_obj=None, weights=None, **user_kwargs):
        super().__init__(
            pLDDTLoss(oligo_obj=oligo_obj, loss_name="plddt"),
            ptmLoss(oligo_obj=oligo_obj, loss_name="ptm"),
            tmAlignLoss(
                oligo_obj=oligo_obj, loss_name="tmAlign", **user_kwargs
            ),
            even=True,
            invert=True,
            **user_kwargs,
        )
        self._information_string = f"""This loss jointly optimises tmalign to template,ptm and plddt (equal weights).
        It attemps to combine the best of both worlds -- getting folded structures that are in contact which match the original model"""


class minDSSPptmlDDT(minLoss):
    def __init__(self, oligo_obj=None, **user_kwargs):
        super().__init__(
            pLDDTLoss(oligo_obj=oligo_obj, loss_name="plddt"),
            ptmLoss(oligo_obj=oligo_obj, loss_name="ptm"),
            fracDSSPLoss(
                oligo_obj=oligo_obj,
                loss_name="dssp",
                loss_params=user_kwargs["loss_params"],
            ),
            invert=True,
            **user_kwargs,
        )
        self._information_string = f"""This loss jointly optimises tmalign to template,ptm and plddt (equal weights).
        It attemps to combine the best of both worlds -- getting folded structures that are in contact which match the original model"""


global_losses_dict = {
    "plddt": pLDDTLoss,
    "ptm": ptmLoss,
    "pae": paeLoss,
    "pae_sub_mat": paeSubMatLoss,
    "pae_asym": paeAsymLoss,
    "dual": dualLoss,
    "separation": separationLoss,
    "min_dssp_ptm_lddt": minDSSPptmlDDT,
}


global_creator = get_loss_creator(**global_losses_dict)


def get_loss(loss_name, **loss_params):
    return global_creator.get_loss(loss_name, **loss_params)


def get_loss_dict():
    return global_losses_dict


############################
# LOSS COMPUTATION
############################


def compute_loss(loss_names, oligo, args, loss_weights):
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
    for loss_idx, current_loss in enumerate(loss_names):
        loss_type, loss_params = current_loss
        args_dict["loss_params"] = loss_params
        score = get_loss(loss_type, oligo_obj=oligo, **args_dict).score()

        scores.append(score)

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
