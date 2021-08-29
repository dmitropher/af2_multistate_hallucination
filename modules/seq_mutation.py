# sequence mutation module
import numpy as np

def extract_stacked_plddts(proto_object, oligo_dict ):
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

    return stacked_plddts


def mutate_random(n_mutations, proto_object, aa_freq):
    '''Mutate protomer sequences randomly, using frequency-adjusted AA picking.'''

    mutated_proto_sequences = {}
    for proto, seq in proto_object.current_sequences.items():
        mutable_pos = np.random.choice(range(len(seq)), size=n_mutations, replace=False)

        for p in mutable_pos:
            seq = seq[:p] + np.random.choice(list(aa_freq.keys()), p=list(aa_freq.values())) + seq[p+1:]

        mutated_proto_sequences[proto] = seq

    return mutated_proto_sequences


def mutate_plddt(n_mutations, proto_object, oligo_dict, aa_freq, update_params ):
    '''
    Mutate positions based on lowest plddt, using frequency-adjusted AA picking.
    First/last three positions of each protomers are choice frequency adjusted to avoid picking N/C term every time (they tend to score much lower).
    '''

    if update_params != None:
        #manually specifiec quantile
        mutate_plddt_quantile = float(update_params)
    else:
        #default worst pLDDT quantile to mutate
        mutate_plddt_quantile = 0.25

    mutated_proto_sequences = {}

    stacked_plddts = extract_stacked_plddts(proto_object, oligo_dict )

    # For each protomer, get mutable positions based on plddt values.
    # Lowest values are taken across all states/oligomer for each protomer.
    # Sequence edges (3 residues) have lower chances of making it in the final selection.
    # This avoids selecting N/C termini over and over again (these tend to score lower).
    mutable_pos = {}
    for proto, plddts in stacked_plddts.items():

        proto_L = proto_object.lengths[proto]

        # Weights associated with each position in the protomer
        # to account for termini systematically scoring worse in pLDDT
        weights = np.array([0.25, 0.5, 0.75] + [1] * (proto_L - 6)+ [0.75, 0.5, 0.25])

        # Sub-select lowest 25% quantile of plddt positions
        n_potential = round(proto_L * mutate_plddt_quantile)
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


def mutate_resfile(n_mutations, proto_object, oligo_dict, aa_freq, update_params ):
    '''Choice of protomer positions to mutate is based on weights provided in .af2h file
    if given as .af2h|0.25 bool selection probabilities as round up or down
    select worst pLDDT scoring quantile, here 0.25 and mutate within this selection at random
    '''

    # if selecting worst residues based on plddt, extract stacked pLDDT array across all
    # protomers in the various oligomers
    if update_params != None :
        # quantile specified
        quantile = float(update_params)
        stacked_plddts = extract_stacked_plddts(proto_object, oligo_dict )


    mutated_proto_sequences = {}
    for proto, seq in proto_object.current_sequences.items():

        if update_params != None :
            # position weights is a np array
            # make position weights 1.0 if non-zero for this protomer
            # np.nonzero returns tuple of arrays along each axis, here choose for axis=0
            allowed_positions = np.nonzero( np.ceil( proto_object.position_weights[proto] ) )[0]
            #extract stacked plddts for this protomer from stacked dict
            plddts = stacked_plddts[proto]
            # Sub-select lowest % quantile in pLDDT of allowed positions
            n_potential = round(len(allowed_positions) * quantile)
            # find absolute worst plddt containing protomer for given positions
            # !!! needs review
            consensus_min = np.min( [ a[allowed_positions] for a in  plddts ], axis=0)
            # sort extract plddts for positions, sort, find worst quantile, extract indices of these
            # from array of allowed positions
            potential_sites = allowed_positions [ np.argsort(consensus_min)[:n_potential] ]
            #randomly select indices to mutate within the worst quantile
            mutable_pos = potential_sites[ np.random.choice(range(len(potential_sites)), size=n_mutations, replace=False) ]


        else:
            position_weights = proto_object.position_weights[proto]
            # Choice of positions is biased by the user-defined position-specific weights.
            mutable_pos = np.random.choice(range(len(seq)), size=n_mutations, replace=False, p=position_weights )

        for p in mutable_pos:
            seq = seq[:p] + np.random.choice(list(aa_freq.keys()), p=list(aa_freq.values())) + seq[p+1:]

        mutated_proto_sequences[proto] = seq

    return mutated_proto_sequences
