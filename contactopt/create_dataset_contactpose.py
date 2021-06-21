# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from os import path
import sys
import numpy as np
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from contactopt.hand_object import HandObject
from contactopt.util import *

sys.path.append('../ContactPose')   # Change this path to point to the ContactPose repo
from utilities.dataset import get_object_names, ContactPose

object_cut_list = []
# object_cut_list = ['eyeglasses']


def get_all_contactpose_samples():
    """
    Gets all participants and objects from ContactPose
    Cuts out grasps with two hands or grasps using left hand
    :return: list of (participant_num, intent, object_name, ContactPose_object)
    """
    samples = []

    print('Reading ContactPose dataset')
    for participant_id in tqdm(range(1, 51)):
        for intent in ['handoff', 'use']:
            for object_name in get_object_names(participant_id, intent):
                cp = ContactPose(participant_id, intent, object_name, load_mano=False)
                if cp._valid_hands != [1]:  # If anything else than just the right hand, remove
                    continue

                samples.append((participant_id, intent, object_name, cp))

    print('Valid ContactPose samples:', len(samples))

    return samples


def generate_contactpose_dataset(dataset, output_file, low_p, high_p, num_pert=1, aug_trans=0.02, aug_rot=0.05, aug_pca=0.3):
    """
    Generates a dataset pkl file and does preprocessing for the PyTorch dataloader
    :param dataset: List of ContactPose objects
    :param output_file: path to output pkl file
    :param low_p: Lower split location of the dataset, [0-1)
    :param high_p: Upper split location of the dataset, [0-1)
    :param num_pert: Number of random perturbations which are computed for every true dataset sample
    :param aug_trans: Std deviation of hand translation noise added to the datasets, meters
    :param aug_rot: Std deviation of hand rotation noise, axis-angle radians
    :param aug_pca: Std deviation of hand pose noise, PCA units
    """
    low_split = int(len(dataset) * low_p)
    high_split = int(len(dataset) * high_p)
    dataset = dataset[low_split:high_split]

    if len(object_cut_list) > 0:
        dataset = [s for s in dataset if s[2] not in object_cut_list]
        print('Some objects are being removed', object_cut_list)

    def process_sample(s, idx):
        ho_gt = HandObject()
        ho_gt.load_from_contactpose(s[3])
        sample_list = []
        # print('Processing', idx)

        for i in range(num_pert):
            # Since we're only saving pointers to the data, it's memory efficient
            sample_data = dict()

            ho_aug = HandObject()
            aug_t = np.random.randn(3) * aug_trans
            aug_p = np.concatenate((np.random.randn(3) * aug_rot, np.random.randn(15) * aug_pca)).astype(np.float32)
            ho_aug.load_from_ho(ho_gt, aug_p, aug_t)

            sample_data['ho_gt'] = ho_gt
            sample_data['ho_aug'] = ho_aug
            sample_data['obj_sampled_idx'] = np.random.randint(0, len(ho_gt.obj_verts), SAMPLE_VERTS_NUM)
            sample_data['hand_feats_aug'], sample_data['obj_feats_aug'] = ho_aug.generate_pointnet_features(sample_data['obj_sampled_idx'])

            sample_list.append(sample_data)

        return sample_list

    parallel = True
    if parallel:
        num_cores = multiprocessing.cpu_count()
        print('Running on {} cores'.format(num_cores))
        all_data_2d = Parallel(n_jobs=num_cores)(delayed(process_sample)(s, idx) for idx, s in enumerate(tqdm(dataset)))
        all_data = [item for sublist in all_data_2d for item in sublist] # flatten 2d list
    else:
        all_data = []   # Do non-parallel
        for idx, s in enumerate(tqdm(dataset)):
            all_data.extend(process_sample(s, idx))

    print('Writing pickle file, often slow and freezes computer')
    pickle.dump(all_data, open(output_file, 'wb'))


if __name__ == '__main__':
    train_file = 'data/perturbed_contactpose_train.pkl'
    test_file = 'data/perturbed_contactpose_test.pkl'
    fine_file = 'data/contactpose_test.pkl'

    aug_trans = 0.05
    aug_rot = 0.1
    aug_pca = 0.5

    contactpose_dataset = get_all_contactpose_samples()

    # Generate Perturbed ContactPose
    generate_contactpose_dataset(contactpose_dataset, train_file, 0.0, 0.8, num_pert=16, aug_trans=aug_trans, aug_rot=aug_rot, aug_pca=aug_pca)
    generate_contactpose_dataset(contactpose_dataset, test_file, 0.8, 1.0, num_pert=4, aug_trans=aug_trans, aug_rot=aug_rot, aug_pca=aug_pca)

    # Generate "Small Refinements" dataset for optimizing ground-truth thermal contact
    generate_contactpose_dataset(contactpose_dataset, fine_file, 0.0, 1.0, num_pert=1, aug_trans=0, aug_rot=0, aug_pca=0)
