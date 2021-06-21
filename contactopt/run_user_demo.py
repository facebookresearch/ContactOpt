# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import trimesh
import json
import contactopt.util as util
import contactopt.arguments as arguments
from contactopt.hand_object import HandObject
from contactopt.run_contactopt import run_contactopt


def create_demo_dataset():
    obj_mesh = trimesh.load('data/demo_obj.obj')    # Load object mesh
    with open('data/demo_mano.json') as json_file:  # Load mano parameters
        mano_params = json.load(json_file)

    # Initialize the HandObject class with the given mano parameters and object mesh.
    # Note that pose must be represented using the 15-dimensional PCA space
    ho_pred = HandObject()
    ho_pred.load_from_mano_params(hand_beta=mano_params['beta'], hand_pose=mano_params['pose'], hand_trans=mano_params['trans'],
                                  obj_faces=obj_mesh.faces, obj_verts=obj_mesh.vertices)

    # To make the dataloader happy, we need a "ground truth" H/O set.
    # However, since this isn't used for this demo, just copy the ho_pred object.
    ho_gt = HandObject()
    ho_gt.load_from_ho(ho_pred)

    new_sample = dict()
    new_sample['ho_aug'] = ho_pred
    new_sample['ho_gt'] = ho_gt

    # Select the random object vertices which will be sampled
    new_sample['obj_sampled_idx'] = np.random.randint(0, len(ho_gt.obj_verts), util.SAMPLE_VERTS_NUM)

    # Calculate hand and object features. The network uses these for improved performance.
    new_sample['hand_feats_aug'], new_sample['obj_feats_aug'] = ho_pred.generate_pointnet_features(new_sample['obj_sampled_idx'])

    return [new_sample]     # Return a dataset of length 1


if __name__ == '__main__':
    dataset = create_demo_dataset()
    args = arguments.run_contactopt_parse_args()

    defaults = {'lr': 0.01,
                'n_iter': 250,
                'w_cont_hand': 2.5,
                'sharpen_thresh': -1,
                'ncomps': 15,
                'w_cont_asym': 2,
                'w_opt_trans': 0.3,
                'w_opt_rot': 1,
                'w_opt_pose': 1.0,
                'caps_rad': 0.001,
                'cont_method': 0,
                'caps_top': 0.0005,
                'caps_bot': -0.001,
                'w_pen_cost': 320,
                'pen_it': 0,
                'rand_re': 8,
                'rand_re_trans': 0.02,
                'rand_re_rot': 5,
                'w_obj_rot': 0,
                'vis_method': 1}

    for k in defaults.keys():
        if vars(args)[k] is None:
            vars(args)[k] = defaults[k]

    args.test_dataset = dataset
    args.split = 'user'

    run_contactopt(args)
