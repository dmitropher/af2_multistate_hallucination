import numpy as np

from file_io import default_aa_freq
from copy import deepcopy

########################################
# PROTOMERS / OLIGOMER CLASSES
########################################


class Protomers:
    """
    Object for keeping track of protomer sequences during hallucination.
    Contains the mutation method function.
    unique_protomer: set of unique letters, one for each protomer
    lengths: length of each protomer (must be in alphabetic order)
    aa_freq: dictonary containing the frequencies of each aa
    sequences: sequences corresponding to each protomer (in alphabetic order). Optional.
    """

    def __init__(
        self,
        unique_protomers=[],
        lengths=[],
        aa_freq=default_aa_freq(),
        sequences=None,
        position_weights=None,
    ):

        # Initialise sequences.
        self.init_sequences = {}
        self.position_weights = {}
        self.aa_freq = aa_freq
        if sequences is None:
            for z in list(zip(unique_protomers, lengths)):
                self.init_sequences[z[0]] = "".join(
                    np.random.choice(
                        list(self.aa_freq.keys()),
                        size=z[1],
                        p=list(self.aa_freq.values()),
                    )
                )
                self.position_weights[z[0]] = np.ones(z[1]) / z[1]

        else:
            for p, proto in enumerate(unique_protomers):
                if sequences[p] == "":  # empty sequence
                    self.init_sequences[proto] = "".join(
                        np.random.choice(
                            list(self.aa_freq.keys()),
                            size=lengths[p],
                            p=list(self.aa_freq.values()),
                        )
                    )
                    self.position_weights[proto] = (
                        np.ones(lengths[p]) / lengths[p]
                    )

                else:
                    self.init_sequences[proto] = sequences[p]
                    if position_weights is None:
                        self.position_weights[proto] = (
                            np.ones(lengths[p]) / lengths[p]
                        )

                    else:
                        if position_weights[p] == "":
                            self.position_weights[proto] = (
                                np.ones(lengths[p]) / lengths[p]
                            )

                        else:
                            self.position_weights[proto] = np.array(
                                position_weights[p]
                            )

        # Initialise lengths.
        self.lengths = {}
        for proto, seq in self.init_sequences.items():
            self.lengths[proto] = len(seq)

        self.current_sequences = {p: s for p, s in self.init_sequences.items()}
        self.try_sequences = {p: s for p, s in self.init_sequences.items()}

    # Method functions.
    def assign_mutable_positions(self, mutable_positions):
        """Assign dictonary of protomers with arrays of mutable positions."""
        self.mutable_positions = mutable_positions

    def assign_mutations(self, mutated_protomers):
        """Assign mutated sequences to try_sequences."""
        self.try_sequences = mutated_protomers

    def update_mutations(self):
        """Update current sequences to try sequences."""
        self.current_sequences = dict(self.try_sequences)


class Oligomer:
    """
    Object for keeping track of oligomers during hallucination.
    Also keeps track of AlphaFold2 scores and structure.
    Sort of like a Pose object in Rosetta.
    """

    def __init__(self, oligo_string: str, protomers: Protomers):

        self.name = oligo_string
        self.positive_design = (lambda x: True if x == "+" else False)(
            oligo_string[-1]
        )
        self.subunits = oligo_string[:-1]

        # Initialise oligomer sequence (concatanation of protomers).
        self.init_seq = ""
        for unit in self.subunits:
            self.init_seq += protomers.init_sequences[unit]

        # Initialise overall length and protomer lengths.
        self.oligo_L = len(self.init_seq)

        self.chain_Ls = []
        for unit in self.subunits:
            self.chain_Ls.append(len(protomers.init_sequences[unit]))

        self.current_seq = str(self.init_seq)
        self.try_seq = str(self.init_seq)

    def init_prediction(self, af2_prediction):
        """Initalise scores/structure"""
        self.init_prediction_results, self.init_unrelaxed_structure = (
            af2_prediction
        )
        self.current_prediction_results = deepcopy(
            self.init_prediction_results
        )
        self.current_unrelaxed_structure = deepcopy(
            self.init_unrelaxed_structure
        )
        self.try_prediction_results = deepcopy(self.init_prediction_results)
        self.unrelaxed_structure = deepcopy(self.init_unrelaxed_structure)

    def init_loss(self, loss):
        """Initalise loss"""
        self.init_loss = loss
        self.current_loss = float(self.init_loss)
        self.try_loss = float(self.init_loss)

    def assign_oligo(self, protomers):
        """Make try oligomer sequence from protomer sequences"""
        self.try_seq = ""
        for unit in self.subunits:
            self.try_seq += protomers.try_sequences[unit]

    def update_oligo(self):
        """Update current oligomer sequence to try ones."""
        self.current_seq = str(self.try_seq)

    def assign_prediction(self, af2_prediction):
        """Assign try AlphaFold2 prediction (scores and structure)."""
        self.try_prediction_results, self.try_unrelaxed_structure = (
            af2_prediction
        )

    def update_prediction(self):
        """Update current scores/structure to try scores/structure."""
        self.current_unrelaxed_structure = deepcopy(
            self.try_unrelaxed_structure
        )
        self.current_prediction_results = deepcopy(self.try_prediction_results)

    def assign_loss(self, loss):
        """Assign try loss."""
        self.try_loss = float(loss)

    def update_loss(self):
        """Update current loss to try loss."""
        self.current_loss = float(self.try_loss)
