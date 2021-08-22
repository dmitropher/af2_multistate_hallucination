# sequence mutation module
import numpy as np

def mutate_random(n_mutations, proto_object, aa_freq):
    '''Mutate protomer sequences randomly, using frequency-adjusted AA picking.'''

    mutated_proto_sequences = {}
    for proto, seq in proto_object.current_sequences.items():
        mutable_pos = np.random.choice(range(len(seq)), size=n_mutations, replace=False)

        for p in mutable_pos:
            seq = seq[:p] + np.random.choice(list(aa_freq.keys()), p=list(aa_freq.values())) + seq[p+1:]

        mutated_proto_sequences[proto] = seq

    return mutated_proto_sequences


def mutate_plddt(n_mutations, proto_object, oligo_dict, aa_freq):
    '''
    Mutate positions based on lowest plddt, using frequency-adjusted AA picking.
    First/last three positions of each protomers are choice frequency adjusted to avoid picking N/C term every time (they tend to score much lower).
    '''

    mutated_proto_sequences = {}

    unique_protomers = list(proto_object.init_sequences.keys())

    # For each protomer, get the plddt arrays of its different states/oligomers.
    stacked_plddts = {proto:[] for proto in unique_protomers}
    for oligo in oligo_dict.values():

        prev = 0
        ch_ranges = []
        for L in oligo.chain_Ls:
            Lcorr = prev + L
            ch_ranges.append([prev, Lcorr])
            prev = Lcorr

        for n, proto in enumerate(oligo.subunits):
            stacked_plddts[proto].append(oligo.current_prediction_results['plddt'][ch_ranges[n][0]:ch_ranges[n][1]])

    # For each protomer, get mutable positions based on plddt values.
    # Lowest values are taken across all states/oligomer for each protomer.
    # Sequence edges (3 residues) have lower chances of making it in the final selection.
    # This avoids selecting N/C termini over and over again (these tend to score lower).
    mutable_pos = {}
    for proto, plddts in stacked_plddts.items():

        proto_L = proto_object.lengths[proto]

        # Weights associated with each position in the protomer
        weights = np.array([0.25, 0.5, 0.75] + [1] * (proto_L - 6)+ [0.75, 0.5, 0.25])

        # Sub-select lowest 25% quantile of plddt positions
        n_potential = round(proto_L * 0.25)
        consensus_min = np.min(plddts, axis=0)
        potential_sites = np.argsort(consensus_min)[:n_potential]

        # Select mutable sites
        sub_w = weights[potential_sites]
        sub_w = [w/np.sum(sub_w) for w in sub_w]
        sites = np.random.choice(potential_sites, size=n_mutations, replace=False, p=sub_w)

        mutable_pos[proto] = sites

    for proto, seq in proto_object.current_sequences.items():

        for p in mutable_pos[proto]:
            seq = seq[:p] + np.random.choice(list(aa_freq.keys()), p=list(aa_freq.values())) + seq[p+1:]

        mutated_proto_sequences[proto] = seq

    return mutated_proto_sequences


def mutate_resfile(n_mutations, proto_object, aa_freq):
    '''Choice of protomer positions to mutate is based on weights provided in .af2h file'''

    mutated_proto_sequences = {}
    for proto, seq in proto_object.current_sequences.items():
        # Choice of positions is biased by the user-defined position-specific weights.
        mutable_pos = np.random.choice(range(len(seq)), size=n_mutations, replace=False, p=proto_object.position_weights[proto])

        for p in mutable_pos:
            seq = seq[:p] + np.random.choice(list(aa_freq.keys()), p=list(aa_freq.values())) + seq[p+1:]

        mutated_proto_sequences[proto] = seq

    return mutated_proto_sequences
