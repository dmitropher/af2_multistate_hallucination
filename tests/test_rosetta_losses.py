import os

from unittest.mock import Mock, MagicMock
from rosetta_losses import (
    CyclicSymmLoss,
    SAPLoss,
    CyclicPlusDual,
    SAPPlusDual,
    MaxSAPDual,
)
from transform_decomposition import (
    helical_axis_data,
    helical_axis_to_rise_rotation_radius_axis,
)
from file_io import dummy_pdbfile
import pyrosetta

from alphafold.protein import Protein


def mock_oligo_from_pdb(pdb_path):
    test_protein = Protein()
    with open(pdb_path, "r") as f:
        pdb_str = f.read()
        test_protein.from_pdb_string(pdb_str)
    test_oligo = Mock()
    test_oligo.try_unrelaxed_structure = test_protein
    return test_oligo


class TestCyclicSymmLoss:
    def mock_oligo_from_pdb(self, pdb_path):
        test_protein = Protein()
        with open(pdb_path, "r") as f:
            pdb_str = f.read()
            test_protein.from_pdb_string(pdb_str)
        test_oligo = Mock()
        test_oligo.try_unrelaxed_structure = test_protein
        return test_oligo

    def setup(self):
        resources = os.path.dirname(__file__) + "test_resources"
        self.monkey_pdb_path = resources + "/monkey.pdb"
        self.good_c3_pdb_path = resources + "/good_c3.pdb"
        test_oligo = Mock()
        test_oligo.try_unrelaxed_structure = mock_oligo_from_pdb(
            self.monkey_pdb_path
        )
        test_oligo_good_c3 = Mock()
        test_oligo_good_c3.try_unrelaxed_structure = mock_oligo_from_pdb(
            self.good_c3_pdb_path
        )
        self.cyc_loss = CyclicSymmLoss(oligo_obj=test_oligo)
        self.cyc_loss_good_c3 = CyclicSymmLoss(oligo_obj=test_oligo_good_c3)

    def test_compute(self):

        pyrosetta.distributed.maybe_init()

        pose = pyrosetta.pose_from_file(self.test_pdb_path)

        s, C, theta, d2, dstar = helical_axis_data(pose, self._n_repeats)
        rise, rotation, s, C, radius = helical_axis_to_rise_rotation_radius_axis(
            s, C, theta, d2, dstar
        )

        assert self.cyc_loss.value == [
            self.cyc_loss._params_dict["axis_direction"],
            self.cyc_loss._params_dict["axis_point"],
            self.cyc_loss._params_dict["rotation_about"],
            self.cyc_loss._params_dict["d2"],
            self.cyc_loss._params_dict["dstar"],
            self.cyc_loss._params_dict["rise"],
        ]
        assert self.cyc_loss._params_dict["axis_direction"] == s
        assert self.cyc_loss._params_dict["axis_point"] == C
        assert self.cyc_loss._params_dict["rotation_about"] == theta
        assert self.cyc_loss._params_dict["d2"] == d2
        assert self.cyc_loss._params_dict["dstar"] == dstar
        assert self.cyc_loss._params_dict["rise"] == rise

    def test_score(self):
        assert 0 < self.cyc_loss.score() < 1
        assert self.cyc_loss_good_c3 < self.cyc_loss

    def test_get_base_values(self):
        test_keys = self.cyc_loss.get_base_values().keys()
        required_keys = ["d2", "dstar", "rise", "d_rotation", "raw_rotation"]
        for key in required_keys:
            assert key in test_keys


class TestCyclicPlusDual:
    def setup(self):
        resources = os.path.dirname(__file__) + "test_resources"
        self.monkey_pdb_path = resources + "/monkey.pdb"
        self.cyc_dual_2t1 = CyclicPlusDual(
            loss_name="cyclic_plus_dual_2t1",
            oligo=mock_oligo_from_pdb(self.monkey_pdb_path),
        )

    def test_weights(self):
        assert self.cyc_dual_2t1.weights["sap_loss"] == 1 / 3
        assert self.cyc_dual_2t1.weights["dual"] == 2 / 3


class TestSAPLoss:
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


class TestSAPPlusDual:
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


class TestMaxSAPDual:
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
