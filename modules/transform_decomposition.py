import sys
import numpy as np

from pyrosetta.rosetta.core.kinematics import Stub

import numba

sys.path.append("/mnt/home/dzorine/software/homog")
from homog.quat import rot_to_quat, quat_to_rot


def average_quats_mean(quats, weights=[]):
    """
    """
    # print (quats)
    qavg = np.zeros(4)
    if not weights:
        weights = np.ones(len(quats))
    for i in range(len(quats)):
        weight = weights[i]
        if i > 0 and np.dot(quats[i], quats[0]) < 0:
            weight = -weight
        qavg += weight * quats[i]
    normed_avg = qavg / np.linalg.norm(qavg)
    # print (normed_avg)
    return normed_avg


def rotation_translation_to_homog(rotation, translation):
    """
    Takes a rotation matrix and a translation vector and returns a h xform
    """
    return np.array(
        [
            [rotation.xx, rotation.xy, rotation.xz, translation.x],
            [rotation.yx, rotation.yy, rotation.yz, translation.y],
            [rotation.zx, rotation.zy, rotation.zz, translation.z],
            [0, 0, 0, 1],
        ]
    )


def stub_from_residue(
    residue, center_atom="CA", atom1="N", atom2="CA", atom3="C"
):
    """
    Returns a stub. A wrapper for atom.xyz with the default of the bb atoms.
    """
    return Stub(
        residue.atom(center_atom).xyz(),
        residue.atom(atom1).xyz(),
        residue.atom(atom2).xyz(),
        residue.atom(atom3).xyz(),
    )


def homog_super_transform_from_residues(res1, res2):
    """
    Wrapper for making CA to CA homogenous super transform between residues

    super is defined by the left multiplied matrix that moves an object from
    the local xform1->xform2
    """
    stub_1 = stub_from_residue(res1)
    stub_2 = stub_from_residue(res2)
    hstub1, hstub2 = (
        rotation_translation_to_homog(stub_1.M, stub_1.v),
        rotation_translation_to_homog(stub_2.M, stub_2.v),
    )
    hstub1_inv = np.linalg.inv(hstub1)
    return hstub2 @ hstub1_inv


def repeat_xform_from_pose(pose, n_repeats):
    """
    """
    n_res = pose.size()
    repeat_size = int(n_res / n_repeats)
    # print (f"n_res: {n_res}, repeat_size: {repeat_size}")
    if repeat_size != int(repeat_size):
        raise AttributeError(
            "repeat size must be a whole number fraction of pose size!"
        )
    xforms = []

    quats = []
    # old_com = None
    for i in range(1, pose.size() - repeat_size + 1):
        res_repeat_xform = homog_super_transform_from_residues(
            pose.residue(i), pose.residue(i + repeat_size)
        )
        quat = rot_to_quat(res_repeat_xform)

        xforms.append(res_repeat_xform)
        quats.append(quat)

    qavg = average_quats_mean(quats)
    xforms_arr = np.array(xforms)
    translations = xforms_arr[..., :3, 3]
    # print (xforms_arr)

    new_rot = quat_to_rot(qavg)
    new_translation = np.sum(translations, 0) / translations.shape[0]
    avg_xform = np.zeros((4, 4))
    avg_xform[:3, :3] = new_rot
    avg_xform[:3, 3] = new_translation
    avg_xform[3, :] = np.array([0, 0, 0, 1])
    # print (avg_xform)
    return avg_xform


def helical_axis_data(pose, n_repeats):
    """
    Math here:
    https://www.12000.org/my_notes/screw_axis/index.htm
    """
    xform = repeat_xform_from_pose(pose, n_repeats)

    # here we derive the rodrigues vector for the rotation matrix, then use
    # that plus the displacement term and theta to find the reference point for the rotation axis
    # We return these things
    d = xform[:3, 3]
    R = xform[:3, :3]

    B = (R - np.identity(3)) @ np.linalg.inv((R + np.identity(3)))
    # print(B)
    rod_vector = np.array([B[2, 1], B[0, 2], B[1, 0]]).transpose()
    rod_mag = np.linalg.norm(rod_vector)
    s = rod_vector / rod_mag
    theta = 2 * np.arctan(rod_mag)
    k = rod_mag  # np.tan(theta / 2)
    d2 = -(k * s)
    dstar = d - d2
    # C is the reference point for the screw axis
    C = np.cross(rod_vector, (dstar - np.cross(rod_vector, dstar))) / (
        2 * np.dot(rod_vector, rod_vector)
    )

    return s, C, theta, d2, dstar


def helical_axis_to_rise_rotation_radius_axis(s, C, theta, d2, dstar):
    rise = np.abs(np.dot(d2 / np.linalg.norm(d2), d2 + dstar))
    rod_vector_direction = (-np.linalg.norm(d2)) < 0
    rotation = theta * (-1 if rod_vector_direction else 1)  # + 2 * np.pi
    radius = np.dot(dstar / np.linalg.norm(dstar), d2 + dstar) / (
        2 * np.sin(theta / 2)
    )
    return rise, rotation, s, C, radius


def rise_run_radius_axis_from_pose(pose, n_repeats):
    s, C, theta, d2, dstar = helical_axis_data(pose, n_repeats)
    return helical_axis_to_rise_rotation_radius_axis(s, C, theta, d2, dstar)
