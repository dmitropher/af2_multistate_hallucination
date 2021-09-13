import os


def default_dssp_dict():
    """
    Return the dssp dict file
    """
    hdf5_path = os.path.join(os.path.dirname(__file__), "resources/dssp")
    return hdf5_path
