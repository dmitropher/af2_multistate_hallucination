#!/software/conda/envs/SE3/bin/python

## Script for performing multistate-state design using AlphaFold2 MCMC hallucination.
## Basile Wicky <basile.wicky@gmail.com>
## Started: 2021-08-11
## Re-factored: 2021-08-20

## SUMMARY:
## - Designs (hallucinations) are performed by MCMC searches in sequence space
##   and optimizing (user-defined) losses composed of AlphaFold2 metrics and geometric constraints.
## - Oligomers with arbitrary number of subunits to be designed.
## - Multistate design (either positive or negative) can be specified.
## - MCMC trajectories can either be seeded with input sequence(s), or started randomly using AA frequency adjustment.
## - At each step, position(s) are chosen for mutation based on different options (see modules/seq_mutation.py for details).
## - The oligomeric state (number of subunits) for each oligomer (state) can be specified.
## - Specific amino acids can be exluded.
## - MCMC paramters (initial temperature, annealing half-life, steps, tolerance) can be specified.
## - Currently implemented loss functions are (see modules/losses.py for details):
##   - plddt: plDDT seem to have trouble converging to complex formation.
##   - ptm: pTM tends to 'melt' input structures.
##   - pae: similar to result as ptm?
##   - dual: combination of plddt and ptm losses with equal weights.
##   - entropy: current implementation unlikely to work.
##   - pae_sub_mat: initially made to enforce symmetry, but probably not working.
##   - cyclic: new trial loss to enforce symmetry based on pae sub-matrix sampling. Not sure it is working. Needs to be benchmarked.
##   - dual_cyclic: dual with an added geometric loss term to enforce symmetry. Seems to work well.

## MINIMLAL INPUTS:
## - The number and type of subunits for each oligomer, also indicating whether it is a positive or negative design task.
## - The length of each protomer or one seed sequence per protomer.

## - EXAMPLES:
## - ./AF2_hallucination_multistate.py --oligo AAAA+,AB+ --L 50,50
##   ^this command will perform 2-state positive design, concomently optimising for a homo-tetramer and a hetero-dimer.
## - ./AF2_hallucination_multistate.py --oligo ABC+, --L 40,50,60
##   ^this command will perform single-state design of a hetero-trimer with protomers of different lengths.
## - ./AF2_hallucination_multistate.py --oligo AB+,AA-,BB- --L 50,50
##   ^this command will perform multi-state design concomently optimising for the heterodimer and disfavouring the two homo-dimers.

## OUTPUTS:
## - PDB structures for each accepted move of the MCMC trajectory.
## - A file (.out) containing the scores at each step of the MCMC trajectory.

## TODO:
## - A CCE-based loss to enable constrained hallucination based on an input structure?
## - RMSD-based or TMscore-based loss to a motif/structure?
## - Check AA frequencies.
## - Check if normalising pae and pae-derived losses by their init value is an appropriate scaling method?

#######################################
# LIBRARIES
#######################################

import os, sys
import numpy as np
import copy

sys.path.append('modules/') # import modules
from arg_parser import *
from seq_mutation import *
from af2_net import *
from losses import *


########################################
# PROTOMERS / OLIGOMER CLASSES
########################################

class Protomers:
    '''
    Object for keeping track of protomer sequences during hallucination.
    Contains the mutation method function.
    unique_protomer: set of unique letters, one for each protomer
    lengths: length of each protomer (must be in alphabetic order)
    aa_freq: dictonary containing the frequencies of each aa
    sequences: sequences corresponding to each protomer (in alphabetic order). Optional.
    '''

    def __init__(self, unique_protomers=[], lengths=[], aa_freq={}, sequences=None, position_weights=None):

        # Initialise sequences.
        self.init_sequences = {}
        self.position_weights = {}
        if sequences is None:
            for z in list(zip(unique_protomers, lengths)):
                self.init_sequences[z[0]] = ''.join(np.random.choice(list(AA_freq.keys()), size=z[1], p=list(AA_freq.values())))
                self.position_weights[z[0]] = np.ones(z[1]) / z[1]

        else:
            for p, proto in enumerate(unique_protomers):
                if sequences[p] == '': # empty sequence
                    self.init_sequences[proto] = ''.join(np.random.choice(list(AA_freq.keys()), size=lengths[p], p=list(AA_freq.values())))
                    self.position_weights[proto] = np.ones(lengths[p]) / lengths[p]

                else:
                    self.init_sequences[proto] = sequences[p]
                    if position_weights is None:
                        self.position_weights[proto] = np.ones(lengths[p]) / lengths[p]

                    else:
                        if position_weights[p] == '':
                            self.position_weights[proto] = np.ones(lengths[p]) / lengths[p]

                        else:
                            self.position_weights[proto] = position_weights[p]

        # Initialise lengths.
        self.lengths = {}
        for proto, seq in self.init_sequences.items():
            self.lengths[proto] = len(seq)

        self.current_sequences = {p:s for p, s in self.init_sequences.items()}
        self.try_sequences = {p:s for p, s in self.init_sequences.items()}

    # Method functions
    def assign_mutations(self, mutated_protomers):
        '''Assign mutated sequences to try_sequences.'''
        self.try_sequences = mutated_protomers

    def update_mutations(self):
        '''Update current sequences to try sequences.'''
        self.current_sequences = copy.deepcopy(self.try_sequences)


class Oligomer:
    '''
    Object for keeping track of oligomers during hallucination.
    Also keeps track of AlphaFold2 scores and structure.
    Sort of like a Pose object in Rosetta.
    '''

    def __init__(self, oligo_string:str, protomers:Protomers):

        self.name = oligo_string
        self.positive_design = (lambda x: True if x=='+' else False)(oligo_string[-1])
        self.subunits = oligo_string[:-1]

        # Initialise oligomer sequence (concatanation of protomers).
        self.init_seq = ''
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
        '''Initalise scores/structure'''
        self.init_prediction_results, self.init_unrelaxed_structure = af2_prediction
        self.current_prediction_results = copy.deepcopy(self.init_prediction_results)
        self.current_unrelaxed_structure = copy.deepcopy(self.init_unrelaxed_structure)
        self.try_prediction_results = copy.deepcopy(self.init_prediction_results)
        self.try_unrelaxed_structure = copy.deepcopy(self.init_unrelaxed_structure)

    def init_loss(self, loss):
        '''Initalise loss'''
        self.init_loss = loss
        self.current_loss = float(self.init_loss)
        self.try_loss = float(self.init_loss)

    def assign_oligo(self, protomers):
        '''Make try oligomer sequence from protomer sequences'''
        self.try_seq = ''
        for unit in self.subunits:
            self.try_seq += protomers.try_sequences[unit]

    def update_oligo(self):
        '''Update current oligomer sequence to try ones.'''
        self.current_seq = str(self.try_seq)

    def assign_prediction(self, af2_prediction):
        '''Assign try AlphaFold2 prediction (scores and structure).'''
        self.try_prediction_results, self.try_unrelaxed_structure = af2_prediction

    def update_prediction(self):
        '''Update current scores/structure to try scores/structure.'''
        self.current_unrelaxed_structure = copy.deepcopy(self.try_unrelaxed_structure)
        self.current_prediction_results = copy.deepcopy(self.try_prediction_results)

    def assign_loss(self, loss):
        '''Assign try loss.'''
        self.try_loss = float(loss)

    def update_loss(self):
        '''Update current loss to try loss.'''
        self.current_loss = float(self.try_loss)


##################################
# INITALISATION
##################################

args = get_args(); print(args)

os.makedirs(f'{args.out}_models', exist_ok=True) # where all the outputs will go.

# Notes.
print(f'Git commit: {args.commit}')
print(f'The following oligomers will be designed:')
for oligo in args.oligo.split(','):
    print(f'>> {oligo[:-1]} ({(lambda x: "positive" if x=="+" else "negative")(oligo[-1])} design)')
print(f'Simulated annealing will be performed over {args.steps} steps with a starting temperature of {args.T_init} and a half-life for the temperature decay of {args.half_life} steps.')
print(f'The mutation rate at each step will go from {args.mutations.split("-")[0]} to {args.mutations.split("-")[1]} over {args.steps} steps (stepped linear decay).')
if args.tolerance is not None:
    print(f'A tolerance setting of {args.tolerance} was set, which might terminate the MCMC trajectory early.')
print(f'The choice of position to mutate at each step will be based on {args.update}.')
print(f'Predictions will be performed with AlphaFold2 model_{args.model}_ptm, with recyling set to {args.recycles}, and {args.msa_clusters} MSA cluster(s).')
print(f'The loss function used during optimisation was set to: {args.loss}.')

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
         'Y': 0.02837870159741062}

for aa in args.exclude_AA:
    del AA_freq[aa]

# Re-compute frequencies to sum to 1.
sum_freq = np.sum(list(AA_freq.values()))
adj_freq = [f/sum_freq for f in list(AA_freq.values())]
AA_freq = dict(zip(AA_freq, adj_freq))

print(f'Allowed amino acids: {len(AA_freq.keys())} [{" ".join([aa for aa in list(AA_freq.keys())])}]')
print(f'Excluded amino acids: {len(args.exclude_AA)} [{" ".join([aa for aa in args.exclude_AA])}]')

# Initialise Protomer object (one for the whole simulation).
if args.proto_sequences is None:
    protomers = Protomers(unique_protomers=args.unique_protomers, lengths=args.proto_Ls, aa_freq=AA_freq)

else:
    protomers = Protomers(unique_protomers=args.unique_protomers, lengths=args.proto_Ls, aa_freq=AA_freq, sequences=args.proto_sequences, position_weights=args.position_weights)

for proto, seq in protomers.init_sequences.items():
    print(f'Protomer {proto} init sequence: {seq}')
    print(f'Protomer {proto} position-specific weights: {protomers.position_weights[proto]}')

# Initialise Oligomer objects (one for each specified oligomer).
oligomers = {}
for o in args.oligo.split(','):
    oligomers[o] = Oligomer(o, protomers)

# Setup AlphaFold2 models.
model_runners = setup_models(args.oligo.split(','), model_id=args.model, recycles=args.recycles, msa_clusters=args.msa_clusters)

# Start score file.
with open(f'{args.out}_models/{args.out}.out', 'w') as f:
    print_str = f'# {args}\n'
    print_str += 'step accepted temperature mutations loss plddt ptm pae '
    for oligo in oligomers.keys():
        print_str += f'sequence_{oligo} loss_{oligo} plddt_{oligo} ptm_{oligo} pae_{oligo}'
    print_str += '\n'
    f.write(print_str)


####################################
# MCMC WITH SIMULATED ANNEALING
####################################

Mi, Mf = args.mutations.split('-')
M = np.linspace(int(Mi), int(Mf), args.steps) # stepped linear decay of the mutation rate

current_loss = np.inf
rolling_window = []
for i in range(args.steps):

    if i > 100 and np.std(rolling_window) < args.tolerance:
        print(f'The change in loss over the last 100 steps has fallen under the tolerance threshold ({args.tolerance}). Terminating the simulation...')
        sys.exit()

    # Update a few things.
    T = args.T_init * (np.exp(np.log(0.5) / args.half_life) ** i) # update temperature
    n_mutations = round(M[i]) # update mutation rate
    accepted = False # reset
    try_loss = 0.0
    if i == 0: # do a first pass through the network before mutating anything -- baseline
        for name, oligo in oligomers.items():
            af2_prediction = predict_structure(oligo, model_runners[name], random_seed=np.random.randint(10)) # run AlphaFold2 prediction
            oligo.init_prediction(af2_prediction) # assign
            loss = compute_loss(args.loss, oligo) # calculate the loss
            oligo.init_loss(loss) # assign
            try_loss += loss # increment global loss

    else: # mutate protomer sequences and generate updated oligomer sequences

        if args.update == 'random':
            protomers.assign_mutations(mutate_random(n_mutations, protomers, AA_freq)) # mutate protomers

        elif args.update == 'plddt':
            protomers.assign_mutations(mutate_plddt(n_mutations, protomers, oligomers, AA_freq)) # mutate protomers

        elif '.af2h' in args.update:
            protomers.assign_mutations(mutate_resfile(n_mutations, protomers, AA_freq)) # mutate protomers

        for name, oligo in oligomers.items():
            oligo.assign_oligo(protomers) # make new oligomers from mutated protomer sequences
            oligo.assign_prediction(predict_structure(oligo, model_runners[name], random_seed=np.random.randint(10))) # run AlphaFold2 prediction
            loss = compute_loss(args.loss, oligo) # calculate the loss for that oligomer
            oligo.assign_loss(loss) # assign the loss to the object (for tracking)
            try_loss += loss # increment the globabl loss

    try_loss /= len(oligomers) # take the mean of the individual oligomer losses (forces scaling between 0 and 1)

    delta = try_loss - current_loss # all losses must be defined such that optimising equates to minimising.

    # If the new solution is better, accept it.
    if delta < 0:
        accepted = True

        print(f'Step {i}: change accepted\n>>LOSS {current_loss} --> {try_loss}')

        current_loss = float(try_loss) # accept loss change
        protomers.update_mutations() # accept sequence changes

        for name, oligo in oligomers.items():
            print(f' >{name} loss {oligo.current_loss} --> {oligo.try_loss}')
            print(f' >{name} plddt {np.mean(oligo.current_prediction_results["plddt"])} --> {np.mean(oligo.try_prediction_results["plddt"])}')
            print(f' >{name} ptm {oligo.current_prediction_results["ptm"]} --> {oligo.try_prediction_results["ptm"]}')
            print(f' >{name} pae {np.mean(oligo.current_prediction_results["predicted_aligned_error"])} --> {np.mean(oligo.try_prediction_results["predicted_aligned_error"])}')
            print('-' * 70)
            oligo.update_oligo() # accept sequence changes
            oligo.update_prediction() # accept score/structure changes
            oligo.update_loss() # accept loss change

    # If the new solution is not better, accept it with a probability of e^(-cost/temp).
    else:

        if np.random.uniform(0, 1) < np.exp( -delta / T):
            accepted = True

            print(f'Step {i}: change accepted despite not improving the loss\n>>LOSS {current_loss} --> {try_loss}')

            current_loss = float(try_loss)
            protomers.update_mutations() # accept sequence changes

            for name, oligo in oligomers.items():
                print(f' >{name} loss {oligo.current_loss} --> {oligo.try_loss}')
                print(f' >{name} plddt {np.mean(oligo.current_prediction_results["plddt"])} --> {np.mean(oligo.try_prediction_results["plddt"])}')
                print(f' >{name} ptm {oligo.current_prediction_results["ptm"]} --> {oligo.try_prediction_results["ptm"]}')
                print(f' >{name} pae {np.mean(oligo.current_prediction_results["predicted_aligned_error"])} --> {np.mean(oligo.try_prediction_results["predicted_aligned_error"])}')
                print('-' * 70)
                oligo.update_oligo() # accept sequence changes
                oligo.update_prediction() # accept score/structure changes
                oligo.update_loss() # accept loss change


        else:
            accepted = False
            print(f'Step {i}: change rejected\n>>LOSS {current_loss} !-> {try_loss}')
            print('-' * 70)

    # Save PDB is move was accepted.
    if accepted == True:

        for name, oligo in oligomers.items():

            with open(f'{args.out}_models/{args.out}_{oligo.name}_step_{str(i).zfill(4)}.pdb', 'w') as f:
                f.write(protein.to_pdb(oligo.current_unrelaxed_structure))
                f.write(f'plddt_array {",".join(oligo.current_prediction_results["plddt"].astype(str))}\n')
                f.write(f'plddt {np.mean(oligo.current_prediction_results["plddt"])}\n')
                f.write(f'ptm {oligo.current_prediction_results["ptm"]}\n')
                f.write(f'pae {np.mean(oligo.current_prediction_results["predicted_aligned_error"])}\n')
                f.write(f'loss {oligo.current_loss}\n')

            # Optionally save the PAE matrix
            if args.output_pae == True:
                np.save(f'{args.out}_models/{args.out}_{oligo}_step_{str(i).zfill(4)}.npy', oligo.current_prediction_results['predicted_aligned_error'])


    # Save scores for the step (even if rejected).
    score_string = f'{i} '
    score_string += f'{accepted} '
    score_string += f'{T} '
    score_string += f'{n_mutations} '
    score_string += f'{try_loss} '
    score_string += f'{np.mean([np.mean(r.try_prediction_results["plddt"]) for r in oligomers.values()])} '
    score_string += f'{np.mean([r.try_prediction_results["ptm"] for r in oligomers.values()])} '
    score_string += f'{np.mean([np.mean(r.try_prediction_results["predicted_aligned_error"]) for r in oligomers.values()])} '

    for name, oligo in oligomers.items():
        score_string += f'{oligo.try_seq} '
        score_string += f'{oligo.try_loss} '
        score_string += f'{np.mean(oligo.try_prediction_results["plddt"])} '
        score_string += f'{oligo.try_prediction_results["ptm"]} '
        score_string += f'{np.mean(oligo.try_prediction_results["predicted_aligned_error"])} '

    with open(f'{args.out}_models/{args.out}.out', 'a') as f:
        f.write(score_string + '\n')

    rolling_window.append(current_loss)
    rolling_window = rolling_window[-100:]

print('Done')
