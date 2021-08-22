# losses module
import numpy as np
import sys; sys.path.append('/projects/ml/alphafold/alphafold_git/')
from alphafold.common import protein

def get_coord(atom_type, oligo_object):
    '''
    General function to get the coordinates of an atom type in a pdb. For geometric-based losses.
    Returns an array [chain, resid, x, y, z]
    '''
    coordinates = []
    pdb_lines = protein.to_pdb(oligo_object.try_unrelaxed_structure).split('\n')
    for l in pdb_lines: # parse PDB lines and extract atom coordinates
        if 'ATOM' in l and atom_type in l:
            s = l.split()
            if len(s[4]) > 1: # residue idx and chain id are no longer space-separated at high id values
                coordinates.append([s[4][0], int(s[4][1:]), np.array(s[5:8], dtype=float)])
            else:
                coordinates.append([s[4], int(s[5]), np.array(s[6:9], dtype=float)])

    coord = np.array(coordinates, dtype=object)

    # Find chain breaks.
    ch_breaks = np.where(np.diff(coord[:, 1]) > 1)[0]
    ch_ends = np.append(ch_breaks, len(coord) - 1)
    ch_starts = np.insert(ch_ends[:-1], 0, 0)

    chain_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for k, start_finish in enumerate(list(zip(ch_starts, ch_ends))):
        coord[start_finish[0] + 1 : start_finish[1]+1 , 0] = chain_list[k] # re-assign chains based on chain breaks

    return coord

def compute_loss(loss_type, oligo):
    '''
    Compute the loss of a single oligomer.
    loss_type: string defining which loss to use.
    oligo: an Oligomer object
    '''

    if loss_type == 'plddt':
        # NOTE:
        # Using this loss will optimise plddt (predicted lDDT) for the sequence(s).
        # Early benchmarks suggest that this is not appropriate for forcing the emergence of complexes.
        # Optimised sequences tend to be folded (or have good secondary structures) without forming inter-chain contacts.
        score = 1 - (np.mean(oligo.try_prediction_results['plddt']) / 100)


    elif loss_type == 'ptm':
        # NOTE:
        # Using this loss will optimise ptm (predicted TM-score) for the sequence(s).
        # Early benchmarks suggest that while it does force the apparition of inter-chain contacts,
        # it might be at the expense of intra-chain contacts and therefore folded protomer structures.
        score = 1 - np.mean(oligo.try_prediction_results['ptm'])


    elif loss_type == 'pae':
        # NOTE:
        # Using this loss will optimise the mean of the pae matrix (predicted alignment error).
        # This loss has not been properly benchmarked, but some early results suggest that it might suffer from the same problem as ptm.
        # During optimisation, off-digonal contacts (inter-chain) may get optimsed at the expense of the diagonal elements (intra-chain).

        norm = np.mean(oligo.init_prediction_results['predicted_aligned_error'])
        score = np.mean(oligo.try_prediction_results['predicted_aligned_error']) / norm


    elif loss_type == 'entropy':
        # CAUTION:
        # This loss is unlikely to yield anything useful at present.
        # i,j pairs that are far away from each other, or for which AF2 is unsure, have max prob in the last bin of their respective distograms.
        # This will generate an artifically low entropy for these positions.
        # Need to find a work around this issue before using this loss.
        print('Entropy definition is most likley improper for loss calculation. Use at your own risk...')

        # Distogram from AlphaFold2 represents pairwise Cb-Cb distances, and is outputted as logits.
        probs = softmax(oligo.try_prediction_results['distogram']['logits'], -1) # convert logit to probs

        score = np.mean(-np.sum((np.array(probs) * np.log(np.array(probs))), axis=-1))


    elif loss_type == 'dual':
        # NOTE:
        # This loss jointly optimises ptm and plddt (equal weights).
        # It attemps to combine the best of both worlds -- getting folded structures that are in contact.
        # This loss is currently recommended unless cyclic geometries are desired (tends to generate linear oligomers).
        score = 1 - (np.mean(oligo.try_prediction_results['plddt']) / 200) - (oligo.try_prediction_results['ptm'] / 2)


    elif loss_type == 'pae_sub_mat':
        # NOTE:
        # This loss optimises the mean of the pae sub-matrices means.
        # The value of the loss will be different to pae in the case of hetero-oligomers that have chains of different lenghts, but identical otherwise.
        # The mean of the sub matrices' means is different from the overall mean if the sub matrices don't all have the same shape.

        sub_mat_init = []
        sub_mat = []
        prev1, prev2 = 0, 0
        for L1 in oligo.chain_Ls:
            Lcorr1 = prev1 + L1

            for L2 in oligo.chain_Ls:
                Lcorr2 = prev2 + L2
                sub_mat_init.append(oligo.init_prediction_results['predicted_aligned_error'][prev1:Lcorr1, prev2:Lcorr2]) # means of the initial sub-matrices
                sub_mat.append(oligo.try_prediction_results['predicted_aligned_error'][prev1:Lcorr1, prev2:Lcorr2]) # means of the tried move sub-matrices
                prev2 = Lcorr2

            prev2 = 0
            prev1 = Lcorr1

        norm =  np.mean([np.mean(sub_m) for sub_m in sub_mat_init])
        score = np.mean([np.mean(sub_m) for sub_m in sub_mat]) / norm


    elif loss_type == 'pae_asym':
        # NOTE:
        # This loss has different weights associated to the means of the different PAE sub matrices (asymmetric weighting).
        # The idea is enforcing loss optimisation for adjacent units to force cyclisation.
        # Off-diagonal elements (+/-1 from the diagaonl, and opposite corners) have higher weights.
        # The weight correction is scaled with the shape of the matrix of sub matrices.
        # By default is scales so that the re-weighted terms count as much as the rest (irrespective of the size of the matrix of sub matrices)

        contribution = 0.5 # if set to one, the re-weighting is done such that diagonal/corner elements count as much as the rest.

        sub_mat_means_init = []
        sub_mat_means = []
        prev1, prev2 = 0, 0
        for L1 in oligo.chain_Ls:
            Lcorr1 = prev1 + L1

            for L2 in oligo.chain_Ls:
                Lcorr2 = prev2 + L2
                sub_mat_means_init.append(np.mean(oligo.init_prediction_results['predicted_aligned_error'][prev1:Lcorr1, prev2:Lcorr2])) # means of the initial sub-matrices
                sub_mat_means.append(np.mean(oligo.try_prediction_results['predicted_aligned_error'][prev1:Lcorr1, prev2:Lcorr2])) # means of the tried move sub-matrices
                prev2 = Lcorr2

            prev2 = 0
            prev1 = Lcorr1

        w_corr = contribution * (oligo.oligo_L**2) / (2 * oligo.oligo_L) # correction scales with the size of the matrix of sub matrices and the desired contribution.

        # Weight matrix
        W = np.ones((len(oligo.subunits), len(oligo.subunits)))
        W[0, -1] = 1 * w_corr
        W[-1, 0] = 1 * w_corr
        W[np.where(np.eye(*W.shape, k=-1) == 1)] = 1 * w_corr
        W[np.where(np.eye(*W.shape, k=1) == 1)] = 1 * w_corr

        norm = np.mean( W * np.reshape(sub_mat_means_init, (len(oligo.subunits), len(oligo.subunits))))
        score = np.mean( W * np.reshape(sub_mat_means, (len(oligo.subunits), len(oligo.subunits)))) / norm


    elif loss_type == 'dual_cyclic':
        # NOTE:
        # This loss is based on dual, but adds a geometric term that forces a cyclic symmetry.
        # At each step the PDB is generated, and the distance between the center of mass of adjacent units computed.
        # The standard deviation of these neighbour distances is added to the loss.

        c = get_coord('CA', oligo) # get CA atoms

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

        separation_std = np.std(proto_dist) # the standard deviation of the distance separations.

        # Compute the score, which is an equal weighting between plddt, ptm and the geometric term.
        score = 1 - (np.mean(oligo.try_prediction_results['plddt']) / 300) - (oligo.try_prediction_results['ptm'] / 3)  + separation_std


    # The loss counts positively or negatively to the overall loss depending on whether this oligomer is positively or negatively designed.
    if oligo.positive_design == True:
        loss = float(score)
    else:
        loss = float(score) + 1

    return loss
