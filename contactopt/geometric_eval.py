# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import hand_object
import os
import util
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation as R
import trimesh
from open3d import io as o3dio
from open3d import geometry as o3dg
from open3d import utility as o3du
from open3d import visualization as o3dv
import matplotlib.pyplot as plt
import torch


def np_apply_tform(points, tform):
    """
    The non-batched numpy version
    :param points: (N, 3)
    :param tform: (4, 4)
    :return:
    """
    points_homo = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    points_out = np.matmul(tform, points_homo.T).T
    return points_out[:, :3]


def get_hand_align_tform(hand_joints):
    """
    Find a 4x4 rigid transform to align the joints of a hand to a 'cardinal rotation'
    :param hand_joints: (21, 3)
    :return: tform: (4, 4)
    """
    center_joint = 0
    x_joint = 2
    y_joint = 17

    trans = hand_joints[center_joint, :]

    x_vec = hand_joints[x_joint, :] - hand_joints[center_joint, :]
    x_vec = x_vec / np.linalg.norm(x_vec)

    y_vec = hand_joints[y_joint, :] - hand_joints[center_joint, :]
    y_vec = np.cross(x_vec, y_vec)
    y_vec = y_vec / np.linalg.norm(y_vec)

    z_vec = np.cross(x_vec, y_vec)
    z_vec = z_vec / np.linalg.norm(z_vec)

    tform = np.eye(4)
    tform[:3, 0] = x_vec
    tform[:3, 1] = y_vec
    tform[:3, 2] = z_vec
    tform[:3, 3] = trans

    return np.linalg.inv(tform)


def calc_procrustes(points1, points2, return_tform=False):
    """ Align the predicted entity in some optimality sense with the ground truth.
    Does NOT align scale
    https://github.com/shreyashampali/ho3d/blob/master/eval.py """

    t1 = points1.mean(0)    # Find centroid
    t2 = points2.mean(0)
    points1_t = points1 - t1   # Zero mean
    points2_t = points2 - t2

    R, s = orthogonal_procrustes(points1_t, points2_t)    # Run procrustes alignment, returns rotation matrix and scale

    points2_t = np.dot(points2_t, R.T)  # Apply tform to second pointcloud
    points2_t = points2_t + t1

    if return_tform:
        return R, t1 - t2
    else:
        return points2_t


def align_by_tform(mtx, tform):
    t2 = mtx.mean(0)
    mtx_t = mtx - t2
    R, t1 = tform
    return np.dot(mtx_t, R.T) + t1 + t2


def get_trans_rot_err(points1, points2):
    """
    Given two pointclouds, find the error in centroid and rotation
    :param points1: numpy (V, 3)
    :param points2: numpy (V, 3)
    :return: translation error (meters), rotation error (degrees)
    """
    tform = calc_procrustes(points1, points2, return_tform=True)
    translation_error = np.linalg.norm(tform[1], 2)
    r = R.from_matrix(tform[0])
    rotation_error = r.magnitude() * 180 / np.pi

    return translation_error, rotation_error


def geometric_eval(ho_test, ho_gt):
    """
    Computes many statistics about ground truth and HO

    Note that official HO-3D metrics are available here, but they only consider the hand, and I think they do too much alignment
    https://github.com/shreyashampali/ho3d/blob/master/eval.py

    :param ho_test: hand-object under test
    :param ho_gt: ground-truth hand-object
    :return: dictionary of stats
    """
    stats = dict()
    stats['unalign_hand_verts'] = util.calc_l2_err(ho_gt.hand_verts, ho_test.hand_verts, axis=1)
    stats['unalign_hand_joints'] = util.calc_l2_err(ho_gt.hand_joints, ho_test.hand_joints, axis=1)
    stats['unalign_obj_verts'] = util.calc_l2_err(ho_gt.obj_verts, ho_test.obj_verts, axis=1)

    root_test = ho_test.hand_joints[0, :]
    root_gt = ho_gt.hand_joints[0, :]

    stats['rootalign_hand_joints'] = util.calc_l2_err(ho_gt.hand_joints - root_gt, ho_test.hand_joints - root_test, axis=1)
    stats['rootalign_obj_verts'] = util.calc_l2_err(ho_gt.obj_verts - root_gt, ho_test.obj_verts - root_test, axis=1)

    obj_cent_gt = ho_gt.obj_verts.mean(0)
    obj_cent_test = ho_test.obj_verts.mean(0)
    stats['objalign_hand_joints'] = util.calc_l2_err(ho_gt.hand_joints - obj_cent_gt, ho_test.hand_joints - obj_cent_test, axis=1)

    hand_joints_align_gt = np_apply_tform(ho_gt.hand_joints, get_hand_align_tform(ho_gt.hand_joints))
    hand_joints_align_test = np_apply_tform(ho_test.hand_joints, get_hand_align_tform(ho_test.hand_joints))
    hand_verts_align_gt = np_apply_tform(ho_gt.hand_verts, get_hand_align_tform(ho_gt.hand_joints))
    hand_verts_align_test = np_apply_tform(ho_test.hand_verts, get_hand_align_tform(ho_test.hand_joints))

    stats['handalign_hand_joints'] = util.calc_l2_err(hand_joints_align_gt, hand_joints_align_test, axis=1)
    stats['handalign_hand_verts'] = util.calc_l2_err(hand_verts_align_gt, hand_verts_align_test, axis=1)

    stats['verts'] = ho_gt.obj_verts.shape[0]

    return stats

