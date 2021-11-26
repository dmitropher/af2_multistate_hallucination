import numpy as np
from itertools import groupby

from loss_factory import Loss, CombinedLoss, get_loss_creator
from loss_functions import (
    pae_sub_mat,
    pae_asym,
    get_separation_std,
    tm_score,
    dssp_wrapper,
    calculate_dssp_fractions,
    dssp_diff,
)
from file_io import dummy_pdbfile


class oligoLoss(Loss):
    """
    Thin wrapper to override the compute with an oligo accessor

    Requires oligo as positional argument
    """

    def __init__(self, oligo_obj=None, **user_kwargs):
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
    A combined loss which weights the input scores (evenly by default)

    Does not invert the sum by default, but can: ( score = 1 - sum(weighted_losses))
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
            loss.score() * self.weights[loss.loss_name] for loss in self.losses
        )
        return self.value

    def score(self):
        return (1 - self.value) if self.invert else self.value


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
        Can be used with the invert option to output 1-max(*losses) (default is not invert).
        included losses:
        {self.get_base_values().keys()}
        returns the biggest of the input losses"""

    def compute(self):
        self.value = max(loss.score() for loss in self.losses)
        return self.value

    def score(self):
        return (1 - self.value) if self.invert else self.value


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
        Can be used with the invert option to output 1-min(*losses) (default is not invert!).
        included losses:
        {self.get_base_values().keys()}
        returns the smallest of the input losses"""

    def compute(self):
        self.value = min(loss.score() for loss in self.losses)
        return self.value

    def score(self):
        return (1 - self.value) if self.invert else self.value


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
    Loss based on mean deviation from user specified fraction dssp
    """

    def __init__(self, oligo_obj=None, **user_kwargs):

        super().__init__(oligo_obj=oligo_obj, **user_kwargs)
        config = user_kwargs["loss_params"]
        dssp_reference = {data[0]: float(data[1:]) for data in config}
        self.desired_dssp = dssp_reference
        if sum(self.desired_dssp.values()) > 1:
            raise AttributeError(
                "Fractions dssp desired can not sum to greater than 1"
            )
        self.oligo = oligo_obj
        self._delta_dssp = {}
        self.value = self.compute()
        self._information_string = f"""This loss object for: {self.loss_name}.
        This loss computes the deviation from the desired fraction dssp
        Value is the mean deviation from desired fraction dssp.
        get_base_values is overwridden to provide raw dssp deviation for each type
        Score is a logistically reweighted value from 0-1, lower is better"""

    def compute(self):
        dummy = dummy_pdbfile(self.oligo)
        dummy_path = dummy.name
        frac_beta, frac_alpha, frac_other = calculate_dssp_fractions(
            dssp_wrapper(dummy_path)
        )
        dummy.close()
        actual = {"E": frac_beta, "H": frac_alpha, "O": frac_other}
        chosen_fracs = self.desired_dssp.keys()
        self._delta_dssp = {
            ("delta_" + key): (abs(actual[key] - self.desired_dssp[key]))
            for key in chosen_fracs
        }
        self.value = np.mean(list(self._delta_dssp.values()))
        return self.value

    # def logistic_rescale(self,):
    #     mid = 0.25
    #     max_val = 1
    #     steep = 10
    #     return max_val / (1 + np.exp(-1 * steep * (self.value - mid)))

    def get_base_values(self):
        name_dict = {self.loss_name: self.value}
        all_dict = {**self._delta_dssp, **name_dict}
        return all_dict

    # TODO: allow logistical parameters to be rescaled from user_args
    def score(self):
        return self.logistic_rescale(0.25, 1, 10)


class fuzzyFracDSSPLoss(fracDSSPLoss):
    """
    Loss based on mean deviation from user specified fraction dssp
    """

    def __init__(self, oligo_obj=None, **user_kwargs):
        super().__init__(oligo_obj=oligo_obj, **user_kwargs)

        self._information_string = f"""This loss object for: {self.loss_name}.
        This loss computes the deviation from the desired fraction dssp
        Value is the mean deviation from desired fraction dssp.
        The delta values for this version of the loss are overriden when they approach 0
        get_base_values is overwridden to provide raw dssp deviation for each type
        Score is a logistically reweighted value from 0-1, lower is better"""

    def compute(self):
        dummy = dummy_pdbfile(self.oligo)
        dummy_path = dummy.name
        frac_beta, frac_alpha, frac_other = calculate_dssp_fractions(
            dssp_wrapper(dummy_path)
        )
        dummy.close()
        actual = {"E": frac_beta, "H": frac_alpha, "O": frac_other}
        chosen_fracs = self.desired_dssp.keys()
        self._delta_dssp = {
            ("delta_" + key): (abs(actual[key] - self.desired_dssp[key]))
            for key in chosen_fracs
        }
        make_fuzzy = lambda val: val if val > 0.1 else 0
        self._fuzzy_delta_dssp = {
            ("fuzzy_delta_" + key): (
                make_fuzzy(abs(actual[key] - self.desired_dssp[key]))
            )
            for key in chosen_fracs
        }
        self.value = np.mean(list(self._fuzzy_delta_dssp.values()))
        return self.value


class dsspFoldProxyLoss(Loss):
    """
    Loss based on mean deviation from user specified fraction dssp
    """

    def __init__(self, oligo_obj=None, **user_kwargs):

        super().__init__(oligo_obj=oligo_obj, **user_kwargs)
        dssp_target_pattern = user_kwargs["loss_params"]
        self.oligo = oligo_obj
        self._target_dssp_fold_string = dssp_target_pattern
        self.value = self.compute()
        self._information_string = f"""This loss object for: {self.loss_name}.
        This loss computes the deviation from the desired pattern of dssp
        Value is the current dssp pattern. Deviation is computed by BioPython
        dynamic programming global alignment with gaps
        Score is reweighted value from 0-1, lower is better"""

    def compute(self):
        dummy = dummy_pdbfile(self.oligo)
        dummy_path = dummy.name
        dssp_string = "".join(dssp_wrapper(dummy_path))
        dummy.close()
        foldssp = "".join(["".join(x for x, y in groupby(dssp_string))])
        self._current_dssp_fold_string = foldssp
        align, max = dssp_diff(
            self._target_dssp_fold_string, self._current_dssp_fold_string
        )
        self.value = align / max
        return self.value

    def get_base_values(self):
        all_dict = {
            self.loss_name: self.value,
            "target_dssp_pattern": self._target_dssp_fold_string,
            "current_dssp_pattern": self._current_dssp_fold_string,
        }
        return all_dict

    # TODO: allow logistical parameters to be rescaled from user_args
    def score(self):
        return 1 - self.logistic_rescale(0.25, 1, 10)


# TODO change this to a logisically mapped rescale, with some TMAlign midpoint around 2 or something
# (rmsd greater than 2 is worse than linearly bad),less than 2 is better than linearly bad
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
            invert=False,
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
            invert=False,
            weights={"plddt": 2.0, "ptm": 2.0, "separation": 1.0},
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
            invert=False,
            **user_kwargs,
        )
        self._information_string = f"""This loss jointly optimises tmalign to template,ptm and plddt (equal weights).
        It attemps to combine the best of both worlds -- getting folded structures that are in contact which match the original model"""


class maxDSSPptmlDDT(maxLoss):
    def __init__(self, oligo_obj=None, **user_kwargs):
        super().__init__(
            pLDDTLoss(oligo_obj=oligo_obj, loss_name="plddt"),
            ptmLoss(oligo_obj=oligo_obj, loss_name="ptm"),
            fracDSSPLoss(
                oligo_obj=oligo_obj,
                loss_name="dssp",
                loss_params=user_kwargs["loss_params"],
            ),
            invert=False,
            **user_kwargs,
        )
        self._information_string = f"""this loss takes the worst loss of the three: ptm, plddt, frac dssp, ensuring the trajectory only walks on all "ok" paths """


global_losses_dict = {
    "plddt": pLDDTLoss,
    "ptm": ptmLoss,
    "pae": paeLoss,
    "pae_sub_mat": paeSubMatLoss,
    "pae_asym": paeAsymLoss,
    "dual": dualLoss,
    "dual_cyclic": dualCyclicLoss,
    "separation": separationLoss,
    "max_dssp_ptm_lddt": maxDSSPptmlDDT,
    "frac_dssp": fracDSSPLoss,
    "fuzzy_frac_dssp": fuzzyFracDSSPLoss,
}


def get_loss_dict():
    return global_losses_dict


def get_global_creator():
    global_creator = get_loss_creator(**get_loss_dict())
    return global_creator


def get_loss(loss_name, **loss_params):
    return get_global_creator().get_loss(loss_name, **loss_params)


def get_allowed_simple_loss_names():
    """
    Returns the DSSP string values registered with the current creator
    """
    from copy import deepcopy

    return deepcopy(get_global_creator()._creators)
