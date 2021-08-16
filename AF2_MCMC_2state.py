#!/software/conda/envs/SE3/bin/python

## Script for performing 2-state design using AlphaFold2 MCMC hallucination.
## Basile Wicky <basile.wicky@gmail.com>
## 2021-08-14

## SUMMARY:
## - MCMC trajectories can either be seeded with an input sequence, or started randomly.
## - At each step, position(s) are chosen for mutation, either randomly, according to plDDT, 
##   or based on a user-provided .res file that specifies mutable positions. 
## - The oligomeric state (number of subunits) for each oligomer (state) can be specified. 
## - Specific amino acids can be exluded.
## - MCMC paramters (initial temperature, annealing half-life, steps) can be specified.
## - The loss function can either be:
##   - plddt: plDDT seem to have trouble converging to complex formation.
##   - ptm: pTM tends to 'melt' input structures.
##   - pae: similar to result as ptm?
##   - dual: combination of plddt and ptm losses.
##   - entropy: current implementation unlikely to work.
##   - pae_sub_mat: initially made to enforce symmetry, but probably not working.
##   - cyclic: new trial loss to enforce symmetry. Not sure it is working. Needs to be benchmarked.
##   - dual_cyclic: dual with an added geometric loss term to enforce symmetry.

## MINIMLAL INPUTS:
## - The length of each protomer.
## - The number of subunits in each oligomeric state.

## OUTPUTS:
## - PDB structure for each accepted move of the MCMC trajectory.
## - A file (.out) containing the scores at each step of the MCMC trajectory.

## TODO:
## - A CCE-based loss to enable constrained hallucination based on an input structure? 
##   Or RMSD-based to a motif/structure?
## - Choice of negative/positive design? Currently only does positive 2-state design.
## - Check AA frequencies.
## - Check if normalising pae and pae-derived losses by their init value is an appropriate scaling method?

#######################################
# LIBRARIES 
#######################################

import os,sys; sys.path.append('/projects/ml/alphafold/alphafold_git/') # where AF2 stuff lives
import mock
import random
import numpy as np
import argparse
import copy
import pickle
from timeit import default_timer as timer
from typing import Dict

# AF2-specific libraries
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from jax.lib import xla_bridge
from jax.nn import softmax
print(xla_bridge.get_backend().platform)


#####################################
# PARSE INPUTS
#####################################

parser = argparse.ArgumentParser(
        description='2-state hallucination using AF2 (MCMC)'
        )
parser.add_argument(
        '--random',
        help='use this flag if you want randomised sequence(s) as seeds. Overwritten by user-provided sequences if provided. Default is False.',
        action='store_true',
        default=False
        )
parser.add_argument(
        '--single',
        help='use this flag if you want to do 1-state design (ignores the second oligomer). Default is False.',
        action='store_true',
        default=False
        )
parser.add_argument(
        '--L1',
        help='the length of protomer 1. Must be specified if using the random flag.',
        action='store',
        type=int
        )
parser.add_argument(
        '--L2',
        help='the length of protomer 2. Must be specified if using the random flag and a homo-/hetero-oligomer 2-state design is desired. Optinal otherwise.',
        action='store',
        type=int
        )
parser.add_argument(
        '--seq1',
        help='the input sequence for protomer 1. Corresponds to the homo-oligomer in the homo-/hetero-oligomer design case. Must be provided if random is not specified.',
        action='store',
        type=str
        )
parser.add_argument(
        '--seq2',
        help='the input sequence for protomer 2. Corresponds to the second chain of the hetero-oligomer in the homo-/hetero-oligomer design case. Optional otherwise.',
        action='store',
        )
parser.add_argument(
        '--oligo1',
        help='the number of subunits for the first homo-oligomer. Must be specified.',
        action='store',
        type=int
        )
parser.add_argument(
        '--oligo2',
        help='the number of subunits for the second homo-oligomer. Corresponds to the number of subunits in the hetero-oligomer in the homo-/hetero-oligomer design case. Must be specified in either case.',
        action='store',
        type=int
        )
parser.add_argument(
        '--prefix',
        help='the prefix appended to the output files. Should be specified.',
        default='test',
        action='store',
        type=str
        )
parser.add_argument(
        '--exclude_AA',
        help='amino acids to exclude during hallucination. Must be a continous string (no spaces) in one-letter code. Default removes C.',
        default='C',
        action='store',
        type=str
        )
parser.add_argument(
        '--mutations',
        help='number of mutations at each MCMC step (start-finish, decayed linearly). Should probably be scaled with protomer length. Default is 3-1.',
        default='3-1',
        action='store'
        )
parser.add_argument(
        '--update',
        help='how to update the sequence at each step. Choice of {random, plddt, FILE.res}. FILE.res needs to be a file specifying mutable positions. Default is random.',
        default='random',
        action='store',
        type=str
        )
parser.add_argument(
        '--T_init',
        help='starting temperature for simulated annealing. Temperature is decayed exponentially. Default is 0.01.',
        default=0.01,
        action='store',
        type=float
        )
parser.add_argument(
        '--half_life',
        help='half-life for the temperature decay during simulated annealing. Default is 1000.',
        default=1000,
        action='store',
        type=float
        )
parser.add_argument(
        '--steps',
        help='number for steps for the MCMC trajectory. Default is 5000.',
        default=5000,
        action='store', 
        type=int
        )
parser.add_argument(
        '--model',
        help='AF2 model used during prediction (choice of 1-5). Default is 4. NB uses _ptm models',
        default=4,
        action='store',
        type=int
        )
parser.add_argument(
        '--loss',
        help='the loss function used during optimization. Choice of {plddt, ptm, pae, entropy, dual, dual_cyclic, pae_sub_mat, cyclic}. Default is dual_cyclic.',
        default='dual_cyclic',
        type=str
        )
parser.add_argument(
        '--recycle',
        help='the number of recycles through the network used during structure prediction. Larger numbers increase accuracy but linearly affect runtime. Default is 1.',
        default=1,
        action='store',
        type=int
        )

parser.add_argument(
        '--msa_clusters',
        help='the number of MSA clusters used during feature generation (?). Larger numbers increase accuracy but significantly affect runtime. Default is 1.',
        default=1, 
        action='store',
        type=int
        )


##################################
# INITALISATION
##################################

args = parser.parse_args()
print(args)
# Some sanity-checks.

# Errors.
if (args.random == True and args.L1 is None) or (args.random == False and args.L1 is not None):
    print('ERROR: If random sequence initalisation is desired, both the protomer length (L1) and the --random flag must be provided. System exiting...')
    sys.exit()

elif (args.oligo1 is None or args.oligo2 is None) and args.single == False:
    print('ERROR: The number of subunits for each oligomer must be specified for 2-state design. System exiting...')
    sys.exit()

elif args.oligo1 is None and args.single == True:
    print('ERROR: the number of subunits must be specified. System exiting...')

elif args.oligo2 != 2 and (args.seq2 is not None or args.L2 is not None):
    print('ERROR: 2-state hallucination for homo-/hetero-oligomer where the hetero-oligomer is not a dimer (--oligo2 != 2) has not been implemented yet. System exiting...')
    sys.exit()

# Warnings.
elif args.random == True and args.seq1 is not None:
    print('WARNING: Both a user-defined sequence and the --random flag were provided. Are you sure of what you are doing? The simulation will continue assuming you wanted to use the provided sequence as seed.')

elif (args.seq1 is not None and args.L1 is not None) or (args.seq2 is not None and args.L2 is not None):
    print('WARNING: Both a sequence and a length were provided. Are you sure of what you are doing?')

# Notes.
if args.seq2 is None and args.L2 is None and args.single == False:
    print('NOTE: Hallucination will produce one sequence that can exist in two different homo-oligomeric states (quasi-symmetric design).')
    sim_type = 'quasi-sym'

elif args.single == True:
    print('NOTE: Hallucination will produce one sequence and optimise it for a single oligomer state (1-state design).')
    sim_type = 'single'

else:
    print('NOTE: Hallucination will produce two sequences, one existing in both a homo-oligomeric and hetero-oligomeric state, and the second one existing as a hetero-oligomer.')
    sim_type = 'hetero-sym'

# Start.
print(f'Predictions will be performed with model_{args.model}_ptm, with recyling set to {args.recycle}, and {args.msa_clusters} MSA cluster(s)')
print(f'The choice of position to mutate at each step will be based on {args.update}')
print(f'The loss function was set to {args.loss}')

# From my rough calculation on 16k random sequences from uniref50 -- should be double-checked.
AA_freq = {'A': 0.08792778710242188,
         'C': 0.01490447165931344,
         'D': 0.05376829211614807,
         'E': 0.06221732055447876,
         'F': 0.0387452994166819,
         'G': 0.06967025329309677,
         'H': 0.0220976574048796,
         'I': 0.05310343411361993,
         'K': 0.050663741170247516,
         'L': 0.09526978211127052,
         'M': 0.02104293453672198,
         'N': 0.04018028904075636,
         'P': 0.051666128157006476,
         'Q': 0.03820000002411093,
         'R': 0.061578750547546295,
         'S': 0.07520039163719089,
         'T': 0.05700516530640848,
         'V': 0.06437948487920657,
         'W': 0.013588957652402187,
         'Y': 0.02837870159741062
 }

for aa in args.exclude_AA:
    del AA_freq[aa]

sum_freq = np.sum(list(AA_freq.values()))
adj_freq = [f/sum_freq for f in list(AA_freq.values())] # re-compute frequencies to sum to 1

print(f'Allowed AA: {len(AA_freq.keys())} [{" ".join([aa for aa in list(AA_freq.keys())])}]')
print(f'Excluded AA: {len(args.exclude_AA)} [{" ".join([aa for aa in args.exclude_AA])}]')

# Initialise sequences, etc.
if sim_type == 'quasi-sym':
    oligomers = ['oligo1', 'oligo2']

elif sim_type == 'hetero-sym':
    oligomers = ['homo', 'hetero']

elif sim_type == 'single':
    oligomers = ['oligo']

prefix = args.prefix

os.makedirs(f'{prefix}_models', exist_ok=True)

with open(f'{prefix}_models/{prefix}.out', 'w') as f:
    print_str = 'step accepted temperature mutations loss plddt_mean ptm_mean '
    for oligo in oligomers:
        print_str += f'sequence_{oligo} plddt_{oligo} ptm_{oligo} '
        
    print_str += '\n'
    f.write(print_str)

if args.seq1 is not None:
    init_proto1 = args.seq1

else:
    init_proto1 = ''.join(np.random.choice(list(AA_freq.keys()), size=args.L1, p=adj_freq))


if args.seq2 is not None:
    init_proto2 = args.seq2

else:

    if args.L2 is not None:
        init_proto2 = ''.join(np.random.choice(list(AA_freq.keys()), size=args.L2, p=adj_freq))

    else:
        init_proto2 = None


Ls = {}
if sim_type == 'quasi-sym':
    proto1_L = len(init_proto1)
    Ls['oligo1'] = [proto1_L] * args.oligo1
    Ls['oligo2'] = [proto1_L] * args.oligo2
    print(f'Protomer 1 seed sequence ({proto1_L} AA): {init_proto1}')

elif sim_type == 'hetero-sym':
    proto1_L = len(init_proto1)
    proto2_L = len(init_proto2)
    Ls['homo'] = [proto1_L] * args.oligo1
    Ls['hetero'] = [proto1_L, proto2_L]
    print(f'Protomer 1 seed sequence ({proto1_L} AA): {init_proto1}')
    print(f'Protomer 2 seed sequence ({proto2_L} AA): {init_proto2}')

elif sim_type == 'single':
    proto1_L = len(init_proto1)
    Ls['oligo'] = [proto1_L] * args.oligo1
    print(f'Protomer 1 seed sequence ({proto1_L} AA): {init_proto1}')

print(f'Number of subunits in oligomer 1: {args.oligo1}')
print(f'Number of subunits in oligomer 2: {args.oligo2}')


# Simulated annealing parameters.
Ti = args.T_init
half_life = args.half_life
sf = np.exp(np.log(0.5) / half_life)
steps = args.steps

print(f'Simulated annealing will be performed over {steps} steps with a starting temperature of {Ti} and a half-life for the temperature decay of {half_life} steps.')

Mi, Mf = args.mutations.split('-')
M = np.linspace(int(Mi), int(Mf), steps)
print(f'The mutation rate at each step will go from {Mi} to {Mf} over {steps} steps (linear stepped decay)')


##########################
# SETUP MODELS
##########################

model_runners = {}
model_num = f'model_{args.model}_ptm' # _ptm series of models necessary for computing pTM and PAE.
model_config = config.model_config(model_num)
model_config.model.num_recycle = args.recycle # AF2 default is 3. Linear gain in time?
model_config.data.common.num_recycle = args.recycle # AF2 defalut is 3. Linear gain in time?
model_config.data.common.max_extra_msa = args.msa_clusters  # AF2 default is 5120. Turning off is 8x faster?
model_config.data.eval.max_msa_clusters = args.msa_clusters # AF2 default is 512. Turning off is 8x faster?
model_config.data.eval.num_ensemble = 1
model_params = data.get_model_haiku_params(model_name=model_num, data_dir="/projects/ml/alphafold")
model_runner = model.RunModel(model_config, model_params)

# Make separate model runners for each oligomer to avoid re-compilation and save time.
for oligo in oligomers:
    model_runners[oligo] = model_runner


############################
# FUNCTIONS
############################

def mk_mock_template(query_sequence):
    """Generate mock template features from the input sequence."""
    output_templates_sequence = []
    output_confidence_scores = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []

    for _ in query_sequence:
        templates_all_atom_positions.append(np.zeros((templates.residue_constants.atom_type_num, 3)))
        templates_all_atom_masks.append(np.zeros(templates.residue_constants.atom_type_num))
        output_templates_sequence.append('-')
        output_confidence_scores.append(-1)

    output_templates_sequence = ''.join(output_templates_sequence)
    templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence,
                                                                    templates.residue_constants.HHBLITS_AA_TO_ID)

    template_features = {'template_all_atom_positions': np.array(templates_all_atom_positions)[None], 
                         'template_all_atom_masks': np.array(templates_all_atom_masks)[None], 
                         'template_sequence': [f'none'.encode()],'template_aatype': np.array(templates_aatype)[None],
                         'template_confidence_scores': np.array(output_confidence_scores)[None],
                         'template_domain_names': [f'none'.encode()],
                         'template_release_date': [f'none'.encode()]
                        }

    return template_features


def predict_structure(data_pipeline: pipeline.DataPipeline,
                    model_runners: Dict[str, model.RunModel],
                    Ls: list,
                    random_seed: int,
                    oligo: str):
    """Predicts structure for a given sequence using AlphaFold2."""

    # Get features.
    feature_dict = data_pipeline.process()

    # Add big enough number to residue index to indicate chain breaks.
    idx_res = feature_dict['residue_index']
    L_prev = 0
    # Ls: number of residues in each chain.
    for L_i in Ls[:-1]:
        idx_res[L_prev+L_i:] += 200
        L_prev += L_i
    feature_dict['residue_index'] = idx_res

    # Run the appropriate model.
    processed_feature_dict = model_runners[oligo].process_features(feature_dict, random_seed=random_seed)
    prediction_results = model_runners[oligo].predict(processed_feature_dict)
    unrelaxed_protein = protein.from_prediction(processed_feature_dict, prediction_results)

    return prediction_results, unrelaxed_protein


def single_pass(Ls, query_sequence, oligo):
    """Make a single pass through the appropriate network."""

    # Mock pipeline.
    data_pipeline_mock = mock.Mock()
    data_pipeline_mock.process.return_value = {
        **pipeline.make_sequence_features(sequence=query_sequence,
                                          description="none",
                                          num_res=len(query_sequence)),
        **pipeline.make_msa_features(msas=[[query_sequence]],
                                     deletion_matrices=[[[0]*len(query_sequence)]]),
        **mk_mock_template(query_sequence)
    }

    start = timer()
    prediction_results, unrelaxed_protein = predict_structure(data_pipeline=data_pipeline_mock,
                               model_runners=model_runners,
                               Ls=Ls, 
                               random_seed=0,
                               oligo=oligo)

    end = timer()
    print(f'{oligo} prediction took {(end - start):.2f} s')

    return prediction_results, unrelaxed_protein


####################################
# MCMC WITH SIMULATED ANNEALING
####################################

current_sequence = {}
old_sequence = {}
try_sequence = {}
if sim_type == 'quasi-sym':
    current_sequence[oligomers[0]] = init_proto1 * args.oligo1
    current_sequence[oligomers[1]] = init_proto1 * args.oligo2

elif sim_type == 'hetero-sym':
    current_sequence[oligomers[0]] = init_proto1 * args.oligo1
    current_sequence[oligomers[1]] = init_proto1 + init_proto2

elif sim_type == 'single':
    current_sequence[oligomers[0]] = init_proto1 * args.oligo1

current_unrelaxed_protein = {}
old_unrelaxed_protein = {}
try_unrelaxed_protein = {}

current_plddts = {}
old_plddts = {}
try_plddts = {}

current_ptms = {}
old_ptms = {}
try_ptms = {}

current_pae = {}
old_pae = {}
try_pae = {}

current_dist = {}
old_dist = {}
try_dist = {}

for oligo in oligomers:
    current_unrelaxed_protein[oligo] = None
    current_plddts[oligo] = np.random.normal(loc=30, scale=5, size=len(current_sequence[oligo]))
    current_ptms[oligo] = 0.0
    current_pae[oligo] = np.random.normal(loc=0.5, scale=0.1, size=(len(current_sequence[oligo]), len(current_sequence[oligo])))
    current_dist[oligo] =  np.random.normal(loc=3, scale=0.5, size=(len(current_sequence[oligo]), len(current_sequence[oligo]), 64)) # distogram is 3D

current_loss = 1e42
norm = 1 # for pae, and pae-derived losses normalisation -- re-assigned after first prediction

for i in range(steps):

    T = Ti * sf**i # update temperature

    # Assign current as old.
    for oligo in oligomers:
        old_sequence[oligo] = str(current_sequence[oligo])
        old_unrelaxed_protein[oligo] = copy.deepcopy(current_unrelaxed_protein[oligo])
        old_plddts[oligo] = current_plddts[oligo].copy()
        old_ptms[oligo] = float(current_ptms[oligo])
        old_pae[oligo] = current_pae.copy()
        old_dist[oligo] = current_dist.copy()


    old_loss = float(current_loss)
    accepted = False

    # Mutate sequences.
    n_mutations = round(M[i]) # current mutation rate

    if sim_type == 'quasi-sym' or sim_type == 'single':
        try_proto1 = str(old_sequence[oligomers[0]][:proto1_L])
        try_proto = [try_proto1]

    elif sim_type == 'hetero-sym':
        try_proto1 = str(old_sequence[oligomers[0]][:proto1_L])
        try_proto2 = str(old_sequence[oligomers[1]][proto1_L:])
        try_proto = [try_proto1, try_proto2]

    for j, proto in enumerate(try_proto):
        
        if args.update == 'random': # works
            mutable_pos = np.random.choice(range(len(proto)), size=n_mutations, replace=False)
            
        elif args.update == 'plddt': # not sure it works
            print(f'{args.update} update implementation is doubtful at present. System exiting...')
            sys.exit()
            
            if j == 0:
                reshaped_plddts = old_plddts[oligomers[j]].reshape((len(Ls[oligomers[j]]), -1)) # make plddts array of shape (n_oligo, seq_L)

            elif j == 1:
            
                if sim_type == 'quasi-sym':
                    reshaped_plddts = old_plddts[oligomers[j]].reshape((len(Ls[oligomers[j]]), -1)) # make plddts array of shape (n_oligo, seq_L)

                elif sim_type == 'hetero-sym':
                    reshaped_plddts = old_plddts[oligomers[j]][Ls[oligomers[j]][0]:]
            
            sites = np.argpartition(reshaped_plddts, n_mutations)[:n_mutations]
            adj_sites = []

            for ch, seq_L in enumerate(Ls[oligomers[j]]):
                corr_sites = sites - (ch * seq_L)
                adj_sites.append(np.where(np.logical_and(corr_sites >= 0, corr_sites < seq_L), corr_sites, np.zeros(len(sites))))

            mutable_pos = np.sum(adj_sites, axis=0, dtype=int)

            print(oligomers[j])
            print(reshaped_plddts)
            print(mutable_pos)
             
        elif '.res' in args.update: # not implemented yet
            print(f'{args.update} update implementation not done yet. System exiting...')
            sys.exit()
            
            mutable_pos = np.array(open(args.update, 'r').readlines()[(j * 2) + 1].split(), dtype=int)
            

        for p in mutable_pos:
            proto = proto[:p] + np.random.choice(list(AA_freq.keys()), p=adj_freq) + proto[p+1:]

        try_proto[j] = str(proto)


    if sim_type == 'quasi-sym':
        try_sequence[oligomers[0]] = try_proto[0] * args.oligo1
        try_sequence[oligomers[1]] = try_proto[0] * args.oligo2

    elif sim_type == 'hetero-sym':
        try_sequence[oligomers[0]] = try_proto[0] * args.oligo1
        try_sequence[oligomers[1]] = try_proto[0] + try_proto[1]

    elif sim_type == 'single':
        try_sequence[oligomers[0]] = try_proto[0] * args.oligo1

    if i == 0: # do a first pass through the network before mutating anything -- baseline
        for oligo in oligomers:
            try_sequence[oligo] = str(old_sequence[oligo])

    
    # Run predictions for both oligomers.
    try_prediction_results = {}
    try_unrelaxed_protein = {}
    for oligo in oligomers:
        try_prediction_results[oligo], try_unrelaxed_protein[oligo] = single_pass(Ls[oligo], try_sequence[oligo], oligo)
        try_plddts[oligo] = try_prediction_results[oligo]['plddt']
        try_ptms[oligo] = try_prediction_results[oligo]['ptm']
        try_pae[oligo] = try_prediction_results[oligo]['predicted_aligned_error']
        try_dist[oligo] = softmax(try_prediction_results[oligo]['distogram']['logits'], -1) # convert logit to probs
        # ^Distogram from AF2 represents pairwise Cb-Cb distances, and is outputted as logits.

    # Compute the loss.
    if args.loss == 'plddt':
        # NOTE:
        # Using this loss will optimise plddt (predicted lDDT) for the sequence(s).
        # Early benchmarks suggest that this is not appropriate for forcing the emergence of complexes.
        # Optimised sequences tend to be folded (or have good secondary structures) without forming inter-chain contacts.
        try_loss = 1 - (np.mean([np.mean(try_plddts[oligo]) for oligo in oligomers]) / 100)

    elif args.loss == 'ptm':
        # NOTE:
        # Using this loss will optimise ptm (predicted TM-score) for the sequence(s).
        # Early benchmarks suggest that while it does force the apparition of inter-chain contacts,
        # it usually is at the expense of intra-chain contacts. Resulting structures tend to look more like entangled spagettis. 
        try_loss = 1 - np.mean([try_ptms[oligo] for oligo in oligomers])

    elif args.loss == 'pae':
        # NOTE:
        # Using this loss will optimise the mean of the pae matrix (predicted alignment error). 
        # This loss has not been properly benchmarked, but some early results suggest that it might suffer from the same problem as ptm.
        # During optimisation, off-digonal contacts (inter-chain) may get optimsed at the expense of the diagonal elements (intra-chain).
        try_loss = np.mean([np.mean(try_pae[oligo]) for oligo in oligomers]) / norm

        if i == 0:
            norm = float(try_loss)


    elif args.loss == 'entropy': 
        # CAUTION:
        # This loss is unlikely to yield anything useful at present.
        # i,j pairs that are far away from each other, or for which AF2 is unsure, have max prob in the last bin of their respective distograms.
        # This will generate an artifically low entropy for these positions.
        # Need to find a work around this issue before using this loss.
        print('Entropy is currently improperly implemented. System exiting...')
        sys.exit()

        entropies = [-np.sum((np.array(probs) * np.log(np.array(probs))), axis=-1) for probs in list(try_dist.values())]
        try_loss = np.mean([np.mean(entropy) for entropy in entropies])

    elif args.loss == 'dual':
        # NOTE:
        # This loss jointly optimises ptm and plddt (equal weights).
        # It attemps to combine the best of both worlds -- getting folded structures that are in contact.
        # This loss is currently recommended.
        try_loss = 1 - (np.mean([try_ptms[oligo] for oligo in oligomers]) / 2) - (np.mean([np.mean(try_plddts[oligo]) for oligo in oligomers]) / 200)


    elif args.loss == 'pae_sub_mat':
        # NOTE:
        # This loss optimises the mean of the pae sub-matrices means.
        # The value of the loss will be different to pae in the hetero-oligomeric case.
        # The mean of the sub matrices' means is different from the overall mean if the sub matrices don't all have the same shape. 
        sub_mat = []
        for oligo in oligomers:
            prev1, prev2 = 0, 0
            
            for L1 in Ls[oligo]:
                Lcorr1 = prev1 + L1

                for L2 in Ls[oligo]:
                    Lcorr2 = prev2 + L2
                    sub_mat.append(try_pae[oligo][prev1:Lcorr1, prev2:Lcorr2])
                    prev2 = Lcorr2

                prev2 = 0
                prev1 = Lcorr1

        try_loss = np.mean([np.mean(sub_m) for sub_m in sub_mat]) / norm

        if i == 0:
            norm = float(try_loss)


    elif args.loss == 'cyclic':
        # NOTE:
        # This loss has different weights associated to the means of the different PAE sub matrices.
        # The idea is enforcing loss optimisation for adjacent units to force cyclisation.
        # Off-diagonal elements (+/-1 from the diagaonl, and opposite corners) have higher weights.
        # The weight correction is scaled with the shape of the matrix of sub matrices.
        # By default is scales so that the re-weighted terms count as much as the rest (irrespective of the size of the matrix of sub matrices)

        contribution = 0.5 # if set to one, the re-weighting is done such that diagonal/corner elements count as much as the rest.
        means = []
        for oligo in oligomers:
            sub_mat_means = []
            prev1, prev2 = 0, 0

            for L1 in Ls[oligo]:
                Lcorr1 = prev1 + L1

                for L2 in Ls[oligo]:
                    Lcorr2 = prev2 + L2
                    sub_mat_means.append(np.mean(try_pae[oligo][prev1:Lcorr1, prev2:Lcorr2])) # mean of the sub matrix
                    prev2 = Lcorr2

                prev2 = 0
                prev1 = Lcorr1

            L = np.sum(Ls[oligo])
            w_corr = contribution * (L**2) / (2 * L) # correction scales with the size of the matrix of sub matrices and the desired contribution.
            
            # Weight matrix
            W = np.ones((len(Ls[oligo]), len(Ls[oligo])))
            W[0, -1] = 1 * w_corr
            W[-1, 0] = 1 * w_corr
            W[np.where(np.eye(*W.shape, k=-1) == 1)] = 1 * w_corr
            W[np.where(np.eye(*W.shape, k=1) == 1)] = 1 * w_corr

            means.append(np.mean( W * np.reshape(sub_mat_means, (len(Ls[oligo]), len(Ls[oligo])))))

        try_loss = np.mean(means) / norm

        if i == 0:
            norm = float(try_loss)

    elif args.loss == 'dual_cyclic':
        # NOTE:
        # This loss is based on dual, but adds a geometric term that forces a cyclic symmetry
        # At each step the PDB is generated, and the distance between the center of mass of adjacent units computed.
        # The standard deviation of these neighbour distances is added to the loss.

        separation_std = []
        for oligo in oligomers:
            coord = []
            
            with open(f'{prefix}_models/tmp.pdb', 'w') as f: # write temporary PDB file
                f.write(protein.to_pdb(try_unrelaxed_protein[oligo]))
 
            with open(f'{prefix}_models/tmp.pdb', 'r') as f:
                lines = f.readlines()
            
                for l in lines: # parse PDB and extract CA coordinates
                    s = l.split()
                    
                    if s[0] == 'ATOM' and s[2] == 'CA':
            
                        if len(s[4]) > 1: # residue idx and chain id are no longer space-separated at high id values
                            coord.append([s[4][0], int(s[4][1:]), np.array(s[5:8], dtype=float)])
            
                        else:
                            coord.append([s[4], int(s[5]), np.array(s[6:9], dtype=float)])
            
            c = np.array(coord, dtype=object)

            # Find chain breaks.
            ch_breaks = np.where(np.diff(c[:, 1])>1)[0]
            ch_ends = np.append(ch_breaks, len(c)-1)
            ch_starts = np.insert(ch_ends[:-1], 0, 0)

            chain_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            for k, start_finish in enumerate(list(zip(ch_starts, ch_ends))):
                c[start_finish[0] + 1:start_finish[1]+1, 0] = chain_list[k] # re-assign chains based on chain breaks

            # Compute center of mass (CA) of each chain.
            chains = set(c[:,0])
            center_of_mass = {ch:float for ch in chains}
            for ch in chains:
                center_of_mass[ch] = np.mean(c[c[:,0]==ch][:,2:], axis=0)[0]

            # Compare distances between adjacent chains, including first-last.
            chain_order = sorted(center_of_mass.keys())
            next_chain = np.roll(chain_order, -1)

            proto_dist = []
            for k, ch in enumerate(chain_order):
                proto_dist.append(np.linalg.norm(center_of_mass[next_chain[k]]-center_of_mass[ch])) # compute separation distances.

            separation_std.append(np.std(np.array(proto_dist))) # the standard deviation of the distance separation
        
        try_loss = 1 - (np.mean([try_ptms[oligo] for oligo in oligomers]) / 3) - (np.mean([np.mean(try_plddts[oligo]) for oligo in oligomers]) / 300) + (np.mean(separation_std) / 30)


    delta = try_loss - old_loss # all losses defined such optimising requires minimising.

    # If the new solution is better, accept it.
    if delta < 0:
        accepted = True
        print(f'Step {i}: change accepted\n>>LOSS {old_loss} --> {try_loss}')
        
        for oligo in oligomers:
            print(f'>>{oligo}-plddt {np.mean(old_plddts[oligo])} --> {np.mean(try_plddts[oligo])}')
            print(f'>>{oligo}-ptm {old_ptms[oligo]} --> {try_ptms[oligo]}')
            current_sequence[oligo] = str(try_sequence[oligo])
            current_unrelaxed_protein[oligo] = copy.deepcopy(try_unrelaxed_protein[oligo])
            current_plddts[oligo] = try_plddts[oligo].copy()
            current_ptms[oligo] = float(try_ptms[oligo])
            current_pae[oligo] = try_pae[oligo].copy()
            current_loss = float(try_loss)

    # If the new solution is not better, accept it with a probability of e^(-cost/temp).
    else:

        if np.random.uniform(0, 1) < np.exp( -delta / T):
            accepted = True
            print(f'Step {i}: change accepted despite not improving the loss\n>>LOSS {old_loss} --> {try_loss}')
            
            for oligo in oligomers:
                print(f'>>{oligo}-plddt {np.mean(old_plddts[oligo])} --> {np.mean(try_plddts[oligo])}')
                print(f'>>{oligo}-ptm {old_ptms[oligo]} --> {try_ptms[oligo]}')
                current_sequence[oligo] = str(try_sequence[oligo])
                current_unrelaxed_protein[oligo] = copy.deepcopy(try_unrelaxed_protein[oligo])
                current_plddts[oligo] = try_plddts[oligo].copy()
                current_ptms[oligo] = float(try_ptms[oligo])
                current_pae[oligo] = try_pae[oligo].copy()
                current_loss = float(try_loss)

        else:
            accepted = False
            print(f'Step {i}: change rejected\n>>LOSS {old_loss} !-> {try_loss}')

    # Save PDB is move was accepted.
    if accepted == True:

        for oligo in oligomers:
            
            np.save(f'{prefix}_models/{prefix}_{oligo}_step_{str(i).zfill(4)}.npy', current_pae[oligo]) # save PAE matrix for each accepted step
            
            with open(f'{prefix}_models/{prefix}_{oligo}_step_{str(i).zfill(4)}.pdb', 'w') as f:
                f.write(protein.to_pdb(current_unrelaxed_protein[oligo]))
                f.write(f'plddt_array {",".join(current_plddts[oligo].astype(str))}\n')
                f.write(f'plddt {np.mean(current_plddts[oligo])}\n')
                f.write(f'ptm {current_ptms[oligo]}\n')
                f.write(f'loss {current_loss}\n')

    # Save scores for the step.
    score_string = f'{i} '
    score_string += f'{accepted} '
    score_string += f'{T} '
    score_string += f'{n_mutations} '
    score_string += f'{current_loss} '
    score_string += f'{np.mean([np.mean(v) for v in current_plddts.values()])} '
    score_string += f'{np.mean([v for v in current_ptms.values()])} '

    for oligo in oligomers:
        score_string += f'{current_sequence[oligo]} '
        score_string += f'{np.mean(current_plddts[oligo])} '
        score_string += f'{current_ptms[oligo]} '
    
    with open(f'{prefix}_models/{prefix}.out', 'a') as f:
        f.write(score_string + '\n')


print('Done')
