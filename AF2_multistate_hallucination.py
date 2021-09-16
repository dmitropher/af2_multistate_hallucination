#!/software/conda/envs/SE3/bin/python

## Script for performing multistate-state design using AlphaFold2 MCMC hallucination.
## <bwicky@uw.edu> and <lmilles@uw.edu>
## Started: 2021-08-11
## Re-factored: 2021-08-20

#######################################
# LIBRARIES
#######################################

import os, sys
import json

import numpy as np


script_dir = os.path.dirname(os.path.realpath(__file__))

# TODO: fix path issues
sys.path.append(script_dir + "/modules/")  # import modules
sys.path.append(script_dir)  # import util
from arg_parser import get_args

from mutations import mutate, select_positions
from af2_net import predict_structure, setup_models
from losses import compute_loss
from protein import Protomers, Oligomer
from scoring import ScoreContainer  # , Score

from file_io import default_aa_freq

from af2_multistate_util import (
    write_to_score_file,
    oligo_to_pdb_file,
    add_default_scores,
    add_default_oligo_scores,
)


##################################
# INITALISATION
##################################

args = get_args()
print("#", args)

sep = " "

os.makedirs(
    f"{args.out}_models", exist_ok=True
)  # where all the outputs will go.
out_dir = f"{args.out}_models/"
out_basename = f"{os.path.splitext(os.path.basename(args.out))[0]}"
# Notes.
print(f"> Git commit: {args.commit}")
if args.single_chain == True:
    print(
        "> Design(s) will be generated as sequence-symmetric repeat proteins, NOT oligomers."
    )
    print("> The following repeat proteins will be designed:")
else:
    print(f"> The following oligomers will be designed:")
for i, oligo in enumerate(args.oligo.strip(",").split(",")):
    print(
        f' >> {oligo[:-1]} ({(lambda x: "positive" if x=="+" else "negative")(oligo[-1])} design), contributing {args.oligo_weights[i]} to the global loss'
    )
print(
    f"> Simulated annealing will be performed over {args.steps} steps with a starting temperature of {args.T_init} and a half-life for the temperature decay of {args.half_life} steps."
)
print(
    f'> The mutation rate at each step will go from {args.mutation_rate.split("-")[0]} to {args.mutation_rate.split("-")[1]} over {args.steps} steps (stepped linear decay).'
)
if args.tolerance is not None:
    print(
        f"> A tolerance setting of {args.tolerance} was set, which might terminate the MCMC trajectory early."
    )
print(
    f"> The choice of position to mutate at each step will be based on {args.select_positions}, with parameter(s): {args.select_position_params}."
)
print(
    f"> At each step, selected positions will be mutated based on {args.mutation_method}."
)
print(
    f"> Predictions will be performed with AlphaFold2 model_{args.model}_ptm, with recyling set to {args.recycles}, and {args.msa_clusters} MSA cluster(s)."
)
print(
    f"> The loss function used during optimisation was set to: {args.loss}, with respective weights: {args.loss_weights}."
)

AA_freq = default_aa_freq()

for aa in args.exclude_AA:
    del AA_freq[aa]

# Re-compute frequencies to sum to 1.
sum_freq = np.sum(list(AA_freq.values()))
adj_freq = [f / sum_freq for f in list(AA_freq.values())]
AA_freq = dict(zip(AA_freq, adj_freq))

print(
    f'> Allowed amino acids: {len(AA_freq.keys())} [{" ".join([aa for aa in list(AA_freq.keys())])}]'
)
print(
    f'> Excluded amino acids: {len(args.exclude_AA)} [{" ".join([aa for aa in args.exclude_AA])}]'
)

# Initialise Protomer object (one for the whole simulation).
if args.proto_sequences is None:
    protomers = Protomers(
        unique_protomers=args.unique_protomers,
        lengths=args.proto_Ls,
        aa_freq=AA_freq,
    )

else:
    protomers = Protomers(
        unique_protomers=args.unique_protomers,
        lengths=args.proto_Ls,
        aa_freq=AA_freq,
        sequences=args.proto_sequences,
        position_weights=args.position_weights,
    )

for proto, seq in protomers.init_sequences.items():
    print(f" >> Protomer {proto} init sequence: {seq}")
    print(
        f" >> Protomer {proto} position-specific weights: {protomers.position_weights[proto]}"
    )

# Initialise Oligomer objects (one for each specified oligomer).
oligomers = {}
for o in args.oligo.split(","):
    oligomers[o] = Oligomer(o, protomers)

# Setup AlphaFold2 models.
model_runners = setup_models(
    args.oligo.split(","),
    model_id=args.model,
    recycles=args.recycles,
    msa_clusters=args.msa_clusters,
)

# Start score file.

# with open(
#     f"{args.out}_models/{os.path.splitext(os.path.basename(args.out))[0]}.out",
#     "w",
# ) as f:
#     print_str = f"# {args}\n"
#     print_str += "step accepted temperature mutations loss plddt ptm pae"
#     for oligo in oligomers.keys():
#         print_str += f" sequence_{oligo} loss_{oligo} plddt_{oligo} ptm_{oligo} pae_{oligo}"
#     print_str += "\n"
#     f.write(print_str)

# Save args as a json so you can load them with a wrapper easily
# TODO: enhancement, maybe add --args-file ?
with open(f"{args.out}_models/run_args.json", "w") as f:
    json.dump(vars(args), f)

####################################
# MCMC WITH SIMULATED ANNEALING
####################################

# prepare score_container for the run, load it with losses and default scores

Mi, Mf = args.mutation_rate.split("-")
M = np.linspace(
    int(Mi), int(Mf), args.steps
)  # stepped linear decay of the mutation rate

# named scores for the score file to report
default_scores = ["plddt", "ptm", "predicted_aligned_error"]
initial_score_container = ScoreContainer()
current_loss = np.inf
rolling_window = []
rolling_window_width = 100
oligo_weights_normalized = np.array(args.oligo_weights) / np.sum(
    args.oligo_weights
)
print("-" * 100)
print("Starting...")
for name, oligo in oligomers.items():
    af2_prediction = predict_structure(
        oligo,
        args.single_chain,
        model_runners[name],
        random_seed=np.random.randint(42),
    )  # run AlphaFold2 prediction

    oligo.init_prediction(af2_prediction)  # assign

    # Assign the init prediction as the current prediction

    oligo.assign_prediction(
        oligo.init_prediction, oligo.init_unrelaxed_structure
    )  # run AlphaFold2 prediction
    # this func implicitely edits the score container, adding arbitrary scores based on losses used
    loss = compute_loss(
        args.loss,
        oligo,
        args,
        args.loss_weights,
        score_container=initial_score_container,
    )  # calculate the loss
    oligo.init_loss(loss)  # assign
    # try_losses.append(loss)  # increment global loss

    # Pull default scores for this oligo name
    add_default_oligo_scores(initial_score_container, name, oligo)
    oligo_to_pdb_file(oligo, 0, out_dir, out_basename, args)
add_default_scores(
    initial_score_container,
    0,
    True,
    args.T_init,
    oligomers,
    oligo_weights_normalized,
    args,
)
key_list = list(initial_score_container.get_keys())

# header_string = ""
# for key in key_list:
#     header_string += f"{key}{sep}"
# header_string += "\n"
# # write the score file header
# with open(
#     f"{args.out}_models/{os.path.splitext(os.path.basename(args.out))[0]}.out",
#     "w",
# ) as f:
#     f.write(header_string)
write_to_score_file(
    out_dir, out_basename, initial_score_container, key_list=None, sep=" "
)

for i in range(1, args.steps):

    # make a score container for the run
    score_container = ScoreContainer()
    if (
        args.tolerance is not None and i > rolling_window_width
    ):  # check if change in loss falls under the tolerance threshold for terminating the simulation.

        if np.std(rolling_window[-rolling_window_width:]) < args.tolerance:
            print(
                f"The change in loss over the last 100 steps has fallen under the tolerance threshold ({args.tolerance}). Terminating the simulation..."
            )
            sys.exit()

    # Update a few things.
    T = args.T_init * (
        np.exp(np.log(0.5) / args.half_life) ** i
    )  # update temperature
    n_mutations = round(M[i])  # update mutation rate
    accepted = False  # reset
    try_losses = []

    # Mutate protomer sequences and generate updated oligomer sequences
    protomers.assign_mutable_positions(
        select_positions(
            n_mutations,
            protomers,
            oligomers,
            args.select_positions,
            args.select_position_params,
        )
    )  # define mutable positions for each protomer
    protomers.assign_mutations(
        mutate(args.mutation_method, protomers, AA_freq)
    )  # mutate those positions

    for name, oligo in oligomers.items():
        oligo.assign_oligo(
            protomers
        )  # make new oligomers from mutated protomer sequences
        oligo.assign_prediction(
            predict_structure(
                oligo,
                args.single_chain,
                model_runners[name],
                random_seed=np.random.randint(42),
            )
        )  # run AlphaFold2 prediction
        loss = compute_loss(
            args.loss,
            oligo,
            args,
            args.loss_weights,
            score_container=score_container,
        )  # calculate the loss for that oligomer
        oligo.assign_loss(loss)  # assign the loss to the object (for tracking)
        try_losses.append(loss)  # increment the global loss
        add_default_oligo_scores(score_container, name, oligo)

    # Global loss is the weighted average of the individual oligomer losses.
    try_loss = np.mean(np.array(try_losses) * oligo_weights_normalized)

    delta = (
        try_loss - current_loss
    )  # all losses must be defined such that optimising equates to minimising.

    # If the new solution is better, accept it.
    if delta < 0:
        accepted = True

        print(
            f"Step {i:05d}: change accepted >> LOSS {current_loss:2.3f} --> {try_loss:2.3f}"
        )

        current_loss = float(try_loss)  # accept loss change
        protomers.update_mutations()  # accept sequence changes

        for name, oligo in oligomers.items():
            print(
                f" > {name} loss  {oligo.current_loss:2.3f} --> {oligo.try_loss:2.3f}"
            )
            print(
                f' > {name} plddt {np.mean(oligo.current_prediction_results["plddt"]):2.3f} --> {np.mean(oligo.try_prediction_results["plddt"]):2.3f}'
            )
            print(
                f' > {name} ptm   {oligo.current_prediction_results["ptm"]:2.3f} --> {oligo.try_prediction_results["ptm"]:2.3f}'
            )
            print(
                f' > {name} pae   {np.mean(oligo.current_prediction_results["predicted_aligned_error"]):2.3f} --> {np.mean(oligo.try_prediction_results["predicted_aligned_error"]):2.3f}'
            )
            oligo.update_oligo()  # accept sequence changes
            oligo.update_prediction()  # accept score/structure changes
            oligo.update_loss()  # accept loss change

        print("=" * 70)

    # If the new solution is not better, accept it with a probability of e^(-cost/temp).
    else:

        if np.random.uniform(0, 1) < np.exp(-delta / T):
            accepted = True

            print(
                f"Step {i:05d}: change accepted despite not improving the loss >> LOSS {current_loss:2.3f} --> {try_loss:2.3f}"
            )

            current_loss = float(try_loss)
            protomers.update_mutations()  # accept sequence changes

            for name, oligo in oligomers.items():
                print(
                    f" > {name} loss  {oligo.current_loss:2.3f} --> {oligo.try_loss:2.3f}"
                )
                print(
                    f' > {name} plddt {np.mean(oligo.current_prediction_results["plddt"]):2.3f} --> {np.mean(oligo.try_prediction_results["plddt"]):2.3f}'
                )
                print(
                    f' > {name} ptm   {oligo.current_prediction_results["ptm"]:2.3f} --> {oligo.try_prediction_results["ptm"]:2.3f}'
                )
                print(
                    f' > {name} pae   {np.mean(oligo.current_prediction_results["predicted_aligned_error"]):2.3f} --> {np.mean(oligo.try_prediction_results["predicted_aligned_error"]):2.3f}'
                )
                oligo.update_oligo()  # accept sequence changes
                oligo.update_prediction()  # accept score/structure changes
                oligo.update_loss()  # accept loss change

            print("=" * 70)

        else:
            accepted = False
            print(
                f"Step {i:05d}: change rejected >> LOSS {current_loss:2.3f} !-> {try_loss:2.3f}"
            )
            print("-" * 70)

            if np.random.uniform(0, 1) < np.exp(-delta / T):
                accepted = True

<<<<<<< HEAD
                print(
                    f"Step {i:05d}: change accepted despite not improving the loss >> LOSS {current_loss:2.3f} --> {try_loss:2.3f}"
                )

                current_loss = float(try_loss)
                protomers.update_mutations()  # accept sequence changes

                for name, oligo in oligomers.items():
                    print(
                        f" > {name} loss  {oligo.current_loss:2.3f} --> {oligo.try_loss:2.3f}"
                    )
                    print(
                        f' > {name} plddt {np.mean(oligo.current_prediction_results["plddt"]):2.3f} --> {np.mean(oligo.try_prediction_results["plddt"]):2.3f}'
                    )
                    print(
                        f' > {name} ptm   {oligo.current_prediction_results["ptm"]:2.3f} --> {oligo.try_prediction_results["ptm"]:2.3f}'
                    )
                    print(
                        f' > {name} pae   {np.mean(oligo.current_prediction_results["predicted_aligned_error"]):2.3f} --> {np.mean(oligo.try_prediction_results["predicted_aligned_error"]):2.3f}'
                    )
                    oligo.update_oligo()  # accept sequence changes
                    oligo.update_prediction()  # accept score/structure changes
                    oligo.update_loss()  # accept loss change

                print("=" * 70)

            else:
                accepted = False
                print(
                    f"Step {i:05d}: change rejected >> LOSS {current_loss:2.3f} !-> {try_loss:2.3f}"
                )
                print("-" * 70)

        sys.stdout.flush()

        # Save PDB if move was accepted.
        if accepted == True:

            for name, oligo in oligomers.items():

                with open(
                    f"{args.out}_models/{os.path.splitext(os.path.basename(args.out))[0]}_{oligo.name}_step_{str(i).zfill(5)}.pdb",
                    "w",
                ) as f:
                    # write pdb
                    if args.amber_relax == 0:
                        pdb_lines = protein.to_pdb(
                            oligo.current_unrelaxed_structure
                        ).split("\n")
                    elif args.amber_relax == 1:
                        pdb_lines = amber_relax(
                            oligo.current_unrelaxed_structure
                        ).split("\n")

                    # Identify chain breaks and re-assign chains correctly before generating PDB file.
                    split_lines = [l.split() for l in pdb_lines if "ATOM" in l]
                    split_lines = np.array(
                        [
                            l[:4] + [l[4][0]] + [l[4][1:]] + l[5:]
                            if len(l) < 12
                            else l
                            for l in split_lines
                        ]
                    )  # chain and resid no longer space-separated at high resid.
                    splits = (
                        np.argwhere(
                            np.diff(split_lines.T[5].astype(int)) > 1
                        ).flatten()
                        + 1
                    )  # identify idx of chain breaks based on resid jump.
                    splits = np.append(splits, len(split_lines))

                    chains = [
                        "A",
                        "B",
                        "C",
                        "D",
                        "E",
                        "F",
                        "G",
                        "H",
                        "I",
                        "J",
                        "K",
                        "L",
                        "M",
                        "N",
                        "O",
                        "P",
                        "Q",
                        "R",
                        "S",
                        "T",
                        "U",
                        "V",
                        "W",
                        "X",
                        "Y",
                        "Z",
                        "AA",
                        "BB",
                        "CC",
                        "DD",
                        "EE",
                        "FF",
                        "GG",
                        "HH",
                        "II",
                        "JJ",
                        "KK",
                        "LL",
                        "MM",
                        "NN",
                        "OO",
                        "PP",
                        "QQ",
                        "RR",
                        "SS",
                        "TT",
                        "UU",
                        "VV",
                        "WW",
                        "XX",
                        "YY",
                        "ZZ",
                        "AAA",
                        "BBB",
                        "CCC",
                        "DDD",
                        "EEE",
                        "FFF",
                        "GGG",
                        "HHH",
                        "III",
                        "JJJ",
                        "KKK",
                        "LLL",
                        "MMM",
                        "NNN",
                        "OOO",
                        "PPP",
                        "QQQ",
                        "RRR",
                        "SSS",
                        "TTT",
                        "UUU",
                        "VVV",
                        "WWW",
                        "XXX",
                        "YYY",
                        "ZZZ",
                    ]
                    chain_str = ""

                    prev = 0
                    for ch, resid in enumerate(
                        splits
                    ):  # make new chain string
                        length = resid - prev
                        chain_str += chains[ch] * length
                        prev = resid
                    atom_lines = [l for l in pdb_lines if "ATOM" in l]
                    new_lines = [
                        l[:21] + chain_str[k] + l[22:]
                        for k, l in enumerate(atom_lines)
                        if "ATOM" in l
                    ]  # generate chain-corrected PDB lines.

                    # write PDB file and append scores at the end of it.
                    f.write("MODEL     1\n")
                    f.write("\n".join(new_lines))
                    f.write("\nENDMDL\nEND\n")
                    f.write(
                        f'plddt_array {",".join(oligo.current_prediction_results["plddt"].astype(str))}\n'
                    )
                    f.write(
                        f'plddt {np.mean(oligo.current_prediction_results["plddt"])}\n'
                    )
                    f.write(f'ptm {oligo.current_prediction_results["ptm"]}\n')
                    f.write(
                        f'pae {np.mean(oligo.current_prediction_results["predicted_aligned_error"])}\n'
                    )
                    f.write(f"loss {oligo.current_loss}\n")
                    f.write(f"# {str(args)}\n")

                # Optionally save the PAE matrix
                if args.output_pae == True:
                    np.save(
                        f"{args.out}_models/{os.path.splitext(os.path.basename(args.out))[0]}_{oligo.name}_step_{str(i).zfill(5)}.npy",
                        oligo.current_prediction_results[
                            "predicted_aligned_error"
                        ],
                    )

        # Save scores for the step (even if rejected).
        # step accepted temperature mutations loss plddt ptm pae '
        score_string = f"{i:05d} "
        score_string += f"{accepted} "
        score_string += f"{T} "
        score_string += f"{n_mutations} "
        score_string += f"{try_loss} "
        score_string += f'{np.mean([np.mean(r.try_prediction_results["plddt"]) for r in oligomers.values()])} '
        score_string += f'{np.mean([r.try_prediction_results["ptm"] for r in oligomers.values()])} '
        score_string += f'{np.mean([np.mean(r.try_prediction_results["predicted_aligned_error"]) for r in oligomers.values()])} '

=======
        for name, oligo in oligomers.items():

            oligo_to_pdb_file(oligo, i, out_dir, out_basename, args)

            # Optionally save the PAE matrix
            if args.output_pae == True:
                np.save(
                    f"{out_dir}/{out_basename}_{oligo.name}_step_{i:05d}.npy",
                    oligo.current_prediction_results[
                        "predicted_aligned_error"
                    ],
                )
    add_default_scores(score_container, i, accepted, T, oligomers)
    write_to_score_file(
        out_dir, out_basename, score_container, key_list=None, sep=" "
    )
>>>>>>> code written, no debugging yet
    rolling_window.append(current_loss)

print("Done")
