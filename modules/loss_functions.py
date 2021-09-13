import sys

import numpy as np

from copy import deepcopy

# dssp loss imports
from Bio.PDB.DSSP import DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

sys.path.append("/projects/ml/alphafold/alphafold_git/")
from alphafold.common import protein

import subprocess

from file_io import default_dssp_dict


######################################################
# COMMON FUNCTIONS USED BY DIFFERENT LOSSES.
######################################################


def get_coord(atom_type, oligo_object):
    """
    General function to get the coordinates of an atom type in a pdb. For geometric-based losses.
    Returns an array [chain, resid, x, y, z]
    """
    coordinates = []
    pdb_lines = protein.to_pdb(oligo_object.try_unrelaxed_structure).split(
        "\n"
    )
    for l in pdb_lines:  # parse PDB lines and extract atom coordinates
        if "ATOM" in l and atom_type in l:
            s = l.split()
            if (
                len(s[4]) > 1
            ):  # residue idx and chain id are no longer space-separated at high id values
                coordinates.append(
                    [s[4][0], int(s[4][1:]), np.array(s[5:8], dtype=float)]
                )
            else:
                coordinates.append(
                    [s[4], int(s[5]), np.array(s[6:9], dtype=float)]
                )

    coord = np.array(coordinates, dtype=object)

    # Find chain breaks.
    ch_breaks = np.where(np.diff(coord[:, 1]) > 1)[0]
    ch_ends = np.append(ch_breaks, len(coord) - 1)
    ch_starts = np.insert(ch_ends[:-1], 0, 0)

    chain_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for k, start_finish in enumerate(list(zip(ch_starts, ch_ends))):
        coord[start_finish[0] + 1 : start_finish[1] + 1, 0] = chain_list[
            k
        ]  # re-assign chains based on chain breaks

    return coord


def dssp_wrapper(pdbfile):
    """Compute DSSP string on structure."""

    dssp_dict = dssp_dict_from_pdb_file(pdbfile, DSSP=default_dssp_dict())[0]

    dssp_list = []
    for key in dssp_dict.keys():
        dssp_list.append(dssp_dict[key][2])

    return dssp_list


def calculate_dssp_fractions(dssp_list):
    """Compute DSSP fraction based on a DSSP list."""

    N_residues = len(dssp_list)
    fraction_beta = float(dssp_list.count("E")) / float(N_residues)
    fraction_helix = float(dssp_list.count("H")) / float(N_residues)
    fraction_other = float(1.0 - fraction_beta - fraction_helix)
    # print(dssp_list, fraction_beta, fraction_helix, fraction_other)

    return fraction_beta, fraction_helix, fraction_other


def tmalign_wrapper(template, temp_pdbfile, force_alignment=None):
    if force_alignment == None:
        p = subprocess.Popen(
            f'/home/lmilles/lm_bin/TMalign {template} {temp_pdbfile} | grep -E "RMSD|TM-score=" ',
            stdout=subprocess.PIPE,
            shell=True,
        )
    else:
        p = subprocess.Popen(
            f'/home/lmilles/lm_bin/TMalign {template} {temp_pdbfile} -I {force_alignment} | grep -E "RMSD|TM-score=" ',
            stdout=subprocess.PIPE,
            shell=True,
        )
    output, __ = p.communicate()
    tm_rmsd = float(str(output)[:-3].split("RMSD=")[-1].split(",")[0])
    tm_score = float(str(output)[:-3].split("TM-score=")[-1].split("(if")[0])

    return tm_rmsd, tm_score


def pae_sub_mat(oligo=None):
    """
    """
    sub_mat_init = []
    sub_mat = []
    prev1, prev2 = 0, 0
    for L1 in oligo.chain_Ls:
        Lcorr1 = prev1 + L1

        for L2 in oligo.chain_Ls:
            Lcorr2 = prev2 + L2
            sub_mat_init.append(
                oligo.init_prediction_results["predicted_aligned_error"][
                    prev1:Lcorr1, prev2:Lcorr2
                ]
            )  # means of the initial sub-matrices
            sub_mat.append(
                oligo.try_prediction_results["predicted_aligned_error"][
                    prev1:Lcorr1, prev2:Lcorr2
                ]
            )  # means of the tried move sub-matrices
            prev2 = Lcorr2

        prev2 = 0
        prev1 = Lcorr1

    norm = np.mean([np.mean(sub_m) for sub_m in sub_mat_init])
    return np.mean([np.mean(sub_m) for sub_m in sub_mat]) / norm


def pae_asym(oligo):
    """Returns asymetrically weighted predicted alignment for oligomer generation
    This loss has different weights associated to the means of the different PAE sub matrices (asymmetric weighting).
    The idea is enforcing loss optimisation for adjacent units to force cyclisation.
    Off-diagonal elements (+/-1 from the diagaonl, and opposite corners) have higher weights.
    The weight correction is scaled with the shape of the matrix of sub matrices.
    By default is scales so that the re-weighted terms count as much as the rest (irrespective of the size of the matrix of sub matrices)"""

    contribution = (
        1
    )  # if set to one, the re-weighting is done such that diagonal/corner elements count as much as the rest.

    sub_mat_means_init = []
    sub_mat_means = []
    prev1, prev2 = 0, 0
    for L1 in oligo.chain_Ls:
        Lcorr1 = prev1 + L1

        for L2 in oligo.chain_Ls:
            Lcorr2 = prev2 + L2
            sub_mat_means_init.append(
                np.mean(
                    oligo.init_prediction_results["predicted_aligned_error"][
                        prev1:Lcorr1, prev2:Lcorr2
                    ]
                )
            )  # means of the initial sub-matrices
            sub_mat_means.append(
                np.mean(
                    oligo.try_prediction_results["predicted_aligned_error"][
                        prev1:Lcorr1, prev2:Lcorr2
                    ]
                )
            )  # means of the tried move sub-matrices
            prev2 = Lcorr2

        prev2 = 0
        prev1 = Lcorr1

    w_corr = (
        contribution * (oligo.oligo_L ** 2) / (2.0 * oligo.oligo_L)
    )  # correction scales with the size of the matrix of sub matrices and the desired contribution.

    # Weight matrix
    W = np.ones((len(oligo.subunits), len(oligo.subunits)))
    W[0, -1] = 1 * w_corr
    W[-1, 0] = 1 * w_corr
    W[np.where(np.eye(*W.shape, k=-1) == 1)] = 1 * w_corr
    W[np.where(np.eye(*W.shape, k=1) == 1)] = 1 * w_corr

    norm = np.mean(
        W
        * np.reshape(
            sub_mat_means_init, (len(oligo.subunits), len(oligo.subunits))
        )
    )
    return (
        np.mean(
            W
            * np.reshape(
                sub_mat_means, (len(oligo.subunits), len(oligo.subunits))
            )
        )
        / norm
    )


def get_separation_std(oligo):
    c = get_coord("CA", oligo=oligo)  # get CA atoms

    # Compute center of mass (CA) of each chain.
    chains = set(c[:, 0])
    center_of_mass = {ch: float for ch in chains}
    for ch in chains:
        center_of_mass[ch] = np.mean(c[c[:, 0] == ch][:, 2:], axis=0)[0]

    # Compare distances between adjacent chains, including first-last.
    chain_order = sorted(center_of_mass.keys())
    next_chain = np.roll(chain_order, -1)

    proto_dist = []
    for k, ch in enumerate(chain_order):
        proto_dist.append(
            np.linalg.norm(center_of_mass[next_chain[k]] - center_of_mass[ch])
        )  # compute separation distances.

    return np.std(proto_dist)


def tm_score(oligo, template, template_alignment, temp_out=""):

    temp_pdbfile = f"{temp_out}_models/tmp.pdb"
    with open(temp_pdbfile, "w") as f:
        f.write(protein.to_pdb(oligo.try_unrelaxed_structure))

    # force_alignment = None

    tm_rmsd, tm_score = tmalign_wrapper(
        template, temp_pdbfile, template_alignment
    )
    print("   tm_RMSD, tmscore ", tm_rmsd, tm_score)
    return tm_score
