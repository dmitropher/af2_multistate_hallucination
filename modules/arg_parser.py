# arg parser module
import argparse
import sys
import numpy as np
import subprocess

def get_args():
    ''' Parse input arguments'''

    parser = argparse.ArgumentParser(
            description='AlphaFold2 hallucination using a Markov chain Monte Carlo protocol. Monomers, homo-oligomers and hetero-oligomers can be hallucinated. Both positive and negative multistate hallucinations are possible.'
            )

    # General arguments.
    parser.add_argument(
            '--oligo',
            help='oligomer(s) definitions (comma-separated strings, no space). Numbers and types of subunits (protomers), and design type (positive or negative) specifying each oligomer.\
             Protomers are defined by unique letters, and strings indicate oligomeric compositions. The last character of each oligomer has to be [+] or [-] to indicate positive or negative design of that oligomer (e.g. AAAA+,AB+).\
             The number of unique protomers must match --L / --seq. Must be specified.',
            action='store',
            type=str,
            required=True
            )

    parser.add_argument(
            '--L',
            help='lengths of each protomer (comma-separated, no space). Must be specified if not using --seq.',
            action='store',
            type=str
            )

    parser.add_argument(
            '--seq',
            help='seed sequence for each protomer (comma-separated, no space). Optional.',
            action='store',
            type=str
            )

    parser.add_argument(
            '--out',
            help='the prefix appended to the output files. Must be specified.',
            action='store',
            type=str,
            required=True
            )

    # Hallucinations arguments.
    parser.add_argument(
            '--exclude_AA',
            default='C',
            action='store',
            type=str,
            help='amino acids to exclude during hallucination. Must be a continous string (no spaces) in one-letter code format (default: %(default)s).'
            )

    parser.add_argument(
            '--mutations',
            default='3-1',
            action='store',
            help='number of mutations at each MCMC step (start-finish, stepped linear decay). Should probably be scaled with protomer length (default: %(default)s).'

            )

    parser.add_argument(
            '--update',
            default='random',
            action='store',
            type=str,
            help='how to update the sequence at each step. Choose from [random, plddt, FILE.res]. FILE.res needs to be a file specifying mutable positions (default: %(default)s).'
            )

    parser.add_argument(
            '--loss',
            default='dual',
            type=str,
            help='the loss function used during optimization. Choose from [plddt, ptm, pae, pae_sub_mat, pae_asym, entropy, dual, dual_cyclic] (default: %(default)s).'
            )

    # MCMC arguments.
    parser.add_argument(
            '--T_init',
            default=0.01,
            action='store',
            type=float,
            help='starting temperature for simulated annealing. Temperature is decayed exponentially (default: %(default)s).'
            )

    parser.add_argument(
            '--half_life',
            default=1000,
            action='store',
            type=float,
            help='half-life for the temperature decay during simulated annealing (default: %(default)s).'
            )

    parser.add_argument(
            '--steps',
            default=5000,
            action='store',
            type=int,
            help='number for steps for the MCMC trajectory (default: %(default)s).'
            )

    parser.add_argument(
            '--tolerance',
            default=None,
            action='store',
            help='the tolerance on the loss sliding window for terminating the MCMC trajectory early (default: %(default)s).'
            )

    # AlphaFold2 arguments.
    parser.add_argument(
            '--model',
            default=4,
            action='store',
            type=int,
            help='AF2 model (_ptm) used during prediction. Choose from [1, 2, 3, 4, 5] (default: %(default)s).'
            )

    parser.add_argument(
            '--recycles',
            default=1,
            action='store',
            type=int,
            help='the number of recycles through the network used during structure prediction. Larger numbers increase accuracy but linearly affect runtime (default: %(default)s).'
            )

    parser.add_argument(
            '--msa_clusters',
            default=1,
            action='store',
            type=int,
            help='the number of MSA clusters used during feature generation (?). Larger numbers increase accuracy but significantly affect runtime (default: %(default)s).'
            )

    parser.add_argument(
            '--output_pae',
            default=False,
            action='store_true',
            help='output the PAE (predicted alignment error) matrix for each accepted step of the MCMC trajectory (default: %(default)s).'
            )

    args = parser.parse_args()


    ########################################
    # SANITY CHECKS
    ########################################

    # Errors.
    if args.oligo is None:
        print('ERROR: the definiton for each oligomer must be specified. System exiting...')
        sys.exit()

    if args.L is None and args.seq is None:
        print('ERROR: either seed sequence(s) or length(s) must specified. System exiting...')
        sys.exit()

    if np.any([(lambda x: True if x not in ['+', '-'] else False)(d[-1]) for d in args.oligo.split(',')]):
        print('ERROR: the type of design (positive [+] or negative [-]) must be specified for each oligomer. System existing...')
        sys.exit()

    # Warnings.
    if (args.L is not None) and (args.seq is not None):
        print('WARNING: Both user-defined sequence(s) and length(s) were provided. Are you sure of what you are doing? The simulation will continue assuming you wanted to use the provided sequence(s) as seed(s).')

    # Add some arguments.
    args.commit = subprocess.check_output(f'git --git-dir .git rev-parse HEAD', shell=True).decode().strip() # add git hash of current commit.

    args.unique_protomers = sorted(set(args.oligo.replace(',','').replace('+','-').replace('-','')))

    if args.seq is not None:
        args.proto_sequences = args.seq.split(',')

        if args.L is None:
            args.proto_Ls = [len(seq) for seq in args.proto_sequences]

        else:
            args.proto_Ls = [int(length) if length!='' else 0 for length in args.L.split(',')]

    else:
        args.proto_Ls = np.array(args.L.split(','), dtype=int)

    # Additional Errors.
    if args.seq is None:
        if len(args.unique_protomers) != len(args.proto_Ls):
            print('ERROR: the number of unique protomers and the number of specified lengths must match. System exiting...')
            sys.exit()

    if args.L is None:
        if len(args.unique_protomers) != len(args.proto_Ls):
             print('ERROR: the number of unique protomers and the number of specified sequences must match. System exiting...')
             sys.exit()

    return args
