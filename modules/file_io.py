import os
import io
import sys

sys.path.append("/projects/ml/alphafold/alphafold_git/")
from alphafold.common import protein


def default_dssp_dict():
    """
    Return the dssp dict file
    """
    hdf5_path = os.path.join(os.path.dirname(__file__), "resources/dssp")
    return hdf5_path


def dummy_pdbfile(oligo):
    """
    returns a StringIO (in lieu of a file) from an "oligo"
    """
    return io.StringIO(protein.to_pdb(oligo.try_unrelaxed_structure))
