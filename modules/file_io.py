import os
import json
import sys
import tempfile

sys.path.append("/projects/ml/alphafold/alphafold_git/")
from alphafold.common import protein


def default_dssp_dict():
    """
    Return the dssp dict file
    """
    dssp_dict_path = os.path.join(os.path.dirname(__file__), "resources/dssp")
    return dssp_dict_path


def default_aa_freq():
    """
    Return the default aa frequency dict
    """
    aa_freq_json_path = os.path.join(
        os.path.dirname(__file__), "resources/default_aa_freq.json"
    )
    with open(aa_freq_json_path, "r") as f:
        aa_freqs_dict = json.load(f)
    return aa_freqs_dict


def dummy_pdbfile(oligo):
    """
    returns a StringIO (in lieu of a file) from an "oligo"
    """
    temp = tempfile.NamedTemporaryFile(mode="w+")
    temp.write(protein.to_pdb(oligo.try_unrelaxed_structure))
    temp.flush()
    temp.seek(0)
    return temp
