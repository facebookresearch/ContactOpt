# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pickle
from contactopt.hand_object import HandObject
import open3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import random
from contactopt.util import SAMPLE_VERTS_NUM


def process_image_pkl(input_file, output_file):
    """
    Reads pre-generated pkl file containing pose estimates and ground truth poses,
    Generates a dataset pkl file and does preprocessing for the PyTorch dataloader
    :param input_file: path of input pkl
    :param output_file: path of output pkl
    """

    input_pkl = pickle.load(open(input_file, 'rb'))
    random.shuffle(input_pkl)

    all_data = []

    for idx, sample_dict in enumerate(tqdm(input_pkl)):
        ho_gt = HandObject()

        # Apply the extrinsic matrix to the pose axis-angle values
        cam_extr = sample_dict['hand_extr_gt']
        rot_pose = R.from_rotvec(sample_dict['hand_pose_gt'][:3])
        rot_extr = R.from_matrix(cam_extr[:3, :3])
        rot_new = rot_extr * rot_pose
        sample_dict['hand_pose_gt'][:3] = rot_new.as_rotvec()   # Overwrite the original axang rotation with new one

        ho_gt.load_from_image(sample_dict['hand_beta_gt'], sample_dict['hand_pose_gt'], sample_dict['obj_faces'], sample_dict['obj_verts_gt'], hand_verts=sample_dict['hand_verts_gt'])

        ho_gt.calc_dist_contact(hand=True, obj=True)
        num_verts_in_contact = np.sum(ho_gt.hand_contact >= 0.9)

        ho_gt.hand_contact *= 0
        ho_gt.obj_contact *= 0

        obj_verts = sample_dict['obj_verts_gt']

        ho_pred = HandObject()
        ho_pred.load_from_image(sample_dict['hand_beta_pred'], sample_dict['hand_pose_pred'], sample_dict['obj_faces'], obj_verts, hand_verts=sample_dict['hand_verts_pred'])

        new_sample = dict()
        new_sample['ho_aug'] = ho_pred
        new_sample['ho_gt'] = ho_gt
        new_sample['obj_sampled_idx'] = np.random.randint(0, len(ho_gt.obj_verts), SAMPLE_VERTS_NUM)
        new_sample['hand_feats_aug'], new_sample['obj_feats_aug'] = ho_pred.generate_pointnet_features(new_sample['obj_sampled_idx'])
        new_sample['num_verts_in_contact'] = num_verts_in_contact

        all_data.append(new_sample)

        if len(all_data) > 10:
            print('Cutting short!')
            break

    pickle.dump(all_data, open(output_file, 'wb'))


if __name__ == '__main__':
    IN_PKL = 'data/pose_estimates.pkl'
    OUT_PKL = 'data/ho3d_image.pkl'
    process_image_pkl(IN_PKL, OUT_PKL)
