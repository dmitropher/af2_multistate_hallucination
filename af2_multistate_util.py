import os.path

import numpy as np

from af2_net import amber_relax, protein
from scoring import Score


def seq_with_breaks_from_oligo(oligo):
    """
    Returns a sequence with '/' as breaks and the correct number of repeats
    """
    breaked_seq = ""
    Lprev = 0
    for L in oligo.chain_Ls:
        Lcorr = Lprev + L
        breaked_seq += oligo.try_seq[Lprev:Lcorr] + "/"
        Lprev = Lcorr
    return breaked_seq


def add_default_oligo_scores(score_container, name, oligo):
    """
    Adds "default" scores to the container from oligo and user args
    """
    breaked_seq = seq_with_breaks_from_oligo(oligo)
    score_container.add_score(
        Score(name=f"sequence_{name}", value=breaked_seq)
    )
    score_container.add_score(Score(name=f"loss_{name}", value=oligo.try_loss))
    score_container.add_score(
        Score(
            name=f"plddt_{name}",
            value=np.mean(oligo.try_prediction_results["plddt"]),
        )
    )
    score_container.add_score(
        Score(name=f"ptm_{name}", value=oligo.try_prediction_results["ptm"])
    )
    score_container.add_score(
        Score(
            name=f"pae_{name}",
            value=np.mean(
                oligo.try_prediction_results["predicted_aligned_error"]
            ),
        )
    )


def add_default_scores(
    score_container,
    step,
    accepted,
    T,
    mutations,
    oligomers,
    oligo_weights_normalized,
    user_args,
):
    """
    Adds default scores which are not single oligo-type specific
    """
    score_container.add_score(Score(name=f"step", value=f"{step:05d}"))
    score_container.add_score(Score(name=f"accepted", value=f"{accepted}"))
    score_container.add_score(Score(name=f"temperature", value=f"{T}"))
    score_container.add_score(Score(name=f"mutations", value=f"{mutations}"))
    score_container.add_score(
        Score(
            name=f"loss",
            value=np.mean(
                np.array([oligo.try_loss for oligo in oligomers.values()])
                * oligo_weights_normalized
            ),
        )
    )
    score_container.add_score(
        Score(
            name=f"plddt",
            value=np.mean(
                [
                    np.mean(oligo.try_prediction_results["plddt"])
                    for oligo in oligomers.values()
                ]
            ),
        )
    )
    score_container.add_score(
        Score(
            name=f"ptm",
            value=np.mean(
                [
                    oligo.try_prediction_results["ptm"]
                    for oligo in oligomers.values()
                ]
            ),
        )
    )
    score_container.add_score(
        Score(
            name=f"pae",
            value=np.mean(
                [
                    np.mean(
                        oligo.try_prediction_results["predicted_aligned_error"]
                    )
                    for oligo in oligomers.values()
                ]
            ),
        )
    )


def oligo_to_pdb_file(
    oligo, step, out_dir, out_basename, user_args, score_container=None
):
    """
    Outputs the given "oligo" as a pdb in the dir given by out_path

    step is the suffix (step generated) after the name
    """
    with open(
        f"{out_dir}/{out_basename}_{oligo.name}_step_{step:05d}.pdb", "w"
    ) as f:
        # write pdb
        if user_args.amber_relax == 0:
            pdb_lines = protein.to_pdb(
                oligo.current_unrelaxed_structure
            ).split("\n")
        elif user_args.amber_relax == 1:
            pdb_lines = amber_relax(oligo.current_unrelaxed_structure).split(
                "\n"
            )

        # Identify chain breaks and re-assign chains correctly before generating PDB file.
        split_lines = [l.split() for l in pdb_lines if "ATOM" in l]
        split_lines = np.array(
            [
                l[:4] + [l[4][0]] + [l[4][1:]] + l[5:] if len(l) < 12 else l
                for l in split_lines
            ]
        )  # chain and resid no longer space-separated at high resid.
        splits = (
            np.argwhere(np.diff(split_lines.T[5].astype(int)) > 1).flatten()
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
        for ch, resid in enumerate(splits):  # make new chain string
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
        if score_container is None:
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
        else:
            for key in score_container.get_keys():
                f.write(f"{key} {score_container.get_score(key)}\n")
        f.write(f"# {str(user_args)}\n")


def write_to_score_file(
    out_dir, out_basename, score_container, key_list=None, sep=" "
):
    """
    Writes the scores to a score file.

    creates the file with header line if it doesn't exist. Default sep is space
    """
    score_path = f"{out_dir}/{out_basename}.out"
    if not os.path.isfile(score_path):
        with open(score_path, "w+") as f:
            if key_list:
                for key in key_list:
                    f.write(f"{key}{sep}")
            else:
                for key in score_container.get_keys():
                    f.write(f"{key}{sep}")
            f.write("\n")
    with open(score_path, "a") as f:
        if key_list:
            for key in key_list:
                f.write(f"{score_container.get_score(key)}{sep}")
        else:
            for key in score_container.get_keys():
                f.write(f"{score_container.get_score(key)}{sep}")
        f.write("\n")
