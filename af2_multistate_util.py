import os.path
import sys

import numpy as np

from af2_net import amber_relax, protein
from scoring import Score

sys.path.append("/software/pyrosetta3.8/latest/")
import pyrosetta

import itertools, string


def chain_generator():
    for i in itertools.count(1):
        for p in itertools.product(string.ascii_uppercase, repeat=i):
            yield "".join(p)


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

        chain_lists = []
        prev = 0
        chain_iter = chain_generator()
        for ch, resid in enumerate(splits):  # make new chain string
            length = resid - prev
            chain_name = next(chain_iter)
            chain_lists.extend([chain_name] * length)
            prev = resid
        atom_lines = [l for l in pdb_lines if "ATOM" in l]
        new_lines = [
            l[:21] + chain_lists[k] + l[22:]
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


def oligo_to_silent(
    oligo, tag, step, silent_out_path, user_args, score_container=None
):
    """
    Outputs the given "oligo" as a pdb in the dir given by out_path

    step is the suffix (step generated) after the name
    """

    pyrosetta.distributed.maybe_init()
    if user_args.amber_relax == 0:
        pdb_lines = protein.to_pdb(oligo.current_unrelaxed_structure).split(
            "\n"
        )
    elif user_args.amber_relax == 1:
        pdb_lines = amber_relax(oligo.current_unrelaxed_structure).split("\n")

    # Identify chain breaks and re-assign chains correctly before generating PDB file.
    split_lines = [l.split() for l in pdb_lines if "ATOM" in l]
    split_lines = np.array(
        [
            l[:4] + [l[4][0]] + [l[4][1:]] + l[5:] if len(l) < 12 else l
            for l in split_lines
        ]
    )  # chain and resid no longer space-separated at high resid.
    splits = (
        np.argwhere(np.diff(split_lines.T[5].astype(int)) > 1).flatten() + 1
    )  # identify idx of chain breaks based on resid jump.
    splits = np.append(splits, len(split_lines))
    # chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chain_iter = chain_generator()
    chain_lists = []
    prev = 0
    for ch, resid in enumerate(splits):  # make new chain string
        length = resid - prev
        chain_name = next(chain_iter)
        chain_lists.extend([chain_name] * length)
        prev = resid
    atom_lines = [l for l in pdb_lines if "ATOM" in l]
    new_lines = [
        l[:21] + chain_lists[k] + l[22:]
        for k, l in enumerate(atom_lines)
        if "ATOM" in l
    ]  # generate chain-corrected PDB lines.
    pdb_string = "\n".join(new_lines)
    out_pose = pyrosetta.rosetta.core.pose.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(
        out_pose, pdb_string
    )
    silent_name = silent_out_path
    sfd_out = pyrosetta.rosetta.core.io.silent.SilentFileData(
        silent_name,
        False,
        False,
        "binary",
        pyrosetta.rosetta.core.io.silent.SilentFileOptions(),
    )
    struct = sfd_out.create_SilentStructOP()
    name_no_suffix = f"{tag}_{oligo.name}_step_{step:05d}"
    struct.fill_struct(out_pose, name_no_suffix)
    sfd_out.add_structure(struct)
    sfd_out.write_all(silent_name, False)


def oligo_to_file(
    oligo, tag, step, out_basename, user_args, score_container=None
):
    if user_args.silent:
        oligo_to_silent(
            oligo,
            tag,
            step,
            out_basename + ".silent",
            user_args,
            score_container,
        )
    else:
        dirname = os.path.dirname(out_basename)
        pdb_basename = os.path.basename(out_basename)
        oligo_to_pdb_file(
            oligo, step, dirname, pdb_basename, user_args, score_container=None
        )


def can_be_float(s):
    """
    Hehe
    """
    try:
        float(s)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def is_int(s):
    """
    Hehe
    """
    try:
        return int(s) == float(s)
    except ValueError:
        return False
    except TypeError:
        return False


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
        if key_list is None:
            key_list = score_container.get_keys()
        for key in key_list:
            score = score_container.get_score(key)
            if can_be_float(score) and not (is_int(score)):
                float_score = float(score)
                f.write(f"{float_score:7.3f}{sep}")
            else:
                f.write(f"{score}{sep}")
        f.write("\n")
