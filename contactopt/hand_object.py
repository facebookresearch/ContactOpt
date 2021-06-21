# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from os import path as osp
import numpy as np
from open3d import io as o3dio
from open3d import geometry as o3dg
from open3d import utility as o3du
from open3d import visualization as o3dv
import json
import transforms3d.quaternions as txq
import torch
import pytorch3d
from pytorch3d.structures import Meshes
import contactopt.util as util
from manopth.manolayer import ManoLayer
from contactopt.diffcontact import calculate_contact_capsule
import matplotlib.pyplot as plt


def mano_get_faces():
    return util.get_mano_closed_faces()


class HandObject:
    """
    Universal data structure to handle hand, object, and contact data.
    This class has many data elements, not all of them are always populated.

    Has many loader functions to take data from multiple sources.
    """
    closed_faces = util.get_mano_closed_faces()

    def __init__(self):
        self.is_left = None
        self.hand_beta = None
        self.hand_pose = None
        self.hand_mTc = None
        self.hand_contact = None
        self.hand_verts = None
        self.hand_joints = None
        self.obj_verts = None
        self.obj_faces = None
        self.obj_contact = None
        self.path = None

        self.obj_normals = None

    def load_from_verts(self, hand_verts, obj_faces, obj_verts):
        """Load from hand/object vertices alone"""
        self.obj_verts = obj_verts
        self.obj_faces = obj_faces
        self.hand_verts = hand_verts

        self.calc_dist_contact(hand=True, obj=True)

    def load_from_image(self, hand_beta, hand_pose, obj_faces, obj_verts, hand_verts=None):
        """Load from image-based results pkl file. Mano root translation is not known, but hand vertices are"""

        self.hand_beta = hand_beta
        self.hand_pose = hand_pose
        self.hand_mTc = np.eye(4)
        self.obj_verts = obj_verts
        self.obj_faces = obj_faces

        self.run_mano()   # Run mano model forwards
        if hand_verts is not None:
            displ = hand_verts[0, :] - self.hand_verts[0, :]    # Find translation by comparing vertices of aligned hands
            self.hand_mTc[:3, 3] = displ
            self.run_mano()  # Rerun mano model to account for translation

            mean_err = np.linalg.norm(self.hand_verts - hand_verts, 2, 1)
            if mean_err.mean() > 1e-6:  # Check if there's much error in reconstruction
                print('Mean verts error', mean_err.mean())
                print('Mano reconstruction failure')

        # self.calc_dist_contact(hand=True, obj=True)
        self.hand_contact = np.zeros((self.hand_verts.shape[0], 1)) # Set to zero since we don't know the ground truth
        self.obj_contact = np.zeros((self.obj_verts.shape[0], 1))

    def load_from_batch(self, hand_beta, hand_pose, hand_mTc, hand_contact, obj_contact, obj_mesh, idx=0, obj_rot=None):
        """Generate HO object from a torch dataloader batch"""
        obj_verts = obj_mesh.verts_list()[idx]
        if obj_rot is not None:
            obj_verts = util.apply_rot(obj_rot[idx, :, :].unsqueeze(0).detach().cpu(), obj_verts.unsqueeze(0), around_centroid=True).squeeze(0)

        self.hand_beta = hand_beta[idx, :].detach().cpu().numpy()
        self.hand_pose = hand_pose[idx, :].detach().cpu().numpy()
        self.hand_mTc = hand_mTc[idx, :, :].detach().cpu().numpy()
        self.hand_contact = hand_contact[idx, :, :].detach().cpu().numpy()
        self.obj_verts = obj_verts.detach().cpu().numpy()
        self.obj_faces = obj_mesh.faces_list()[idx].detach().cpu().numpy()
        self.obj_contact = obj_contact[idx, :self.obj_verts.shape[0], :].detach().cpu().numpy()    # Since we're using a padded array, need to cut off some

        self.run_mano()

    def load_from_contactpose(self, cp_obj):
        """Load HO object from ContactPose dataset"""
        if not osp.isfile(cp_obj.contactmap_filename):
            raise FileNotFoundError('Could not find {}'.format(cp_obj.contactmap_filename))

        obj_mesh = o3dio.read_triangle_mesh(cp_obj.contactmap_filename)     # Includes object mesh and contact map embedded as vertex colors

        vertex_colors = np.array(obj_mesh.vertex_colors, dtype=np.float32)
        self.obj_contact = np.expand_dims(util.fit_sigmoid(vertex_colors[:, 0]), axis=1)    # Normalize with sigmoid, shape (V, 1)
        self.obj_verts = np.array(obj_mesh.vertices, dtype=np.float32)     # Keep as floats since torch uses floats
        self.obj_faces = np.array(obj_mesh.triangles)

        for idx, mp in enumerate(cp_obj.mano_params):
            if mp is None:
                continue

            self.is_left = idx == 0  # Left then right
            self.hand_beta = np.array(mp['betas'])  # 10 shape PCA parameters
            self.hand_pose = np.array(mp['pose'])  # 18 dim length, first 3 ax-angle, 15 PCA pose

            mTc = mp['hTm']
            # mTc = np.linalg.inv(mTc)  # World to object
            self.hand_mTc = mTc

        if self.is_left:
            raise ValueError('Pipeline currently cant handle left hands')

        self.run_mano()
        self.calc_dist_contact(hand=True, obj=False)

    def load_from_ho(self, ho, aug_pose=None, aug_trans=None):
        """Load from another HandObject obj, potentially with augmentation"""
        self.hand_beta = np.array(ho.hand_beta)
        self.hand_pose = np.array(ho.hand_pose)
        self.hand_mTc = np.array(ho.hand_mTc)
        self.obj_verts = ho.obj_verts
        self.obj_faces = ho.obj_faces
        self.obj_contact = ho.obj_contact

        if aug_pose is not None:
            self.hand_pose += aug_pose
        if aug_trans is not None:
            self.hand_mTc[:3, 3] += aug_trans

        self.run_mano()
        # self.calc_dist_contact(hand=True, obj=False)  # DONT calculate hand contact, since it's not ground truth

    def load_from_mano_params(self, hand_beta, hand_pose, hand_trans, obj_faces, obj_verts):
        """Load from mano parameters and object mesh"""
        self.hand_beta = np.array(hand_beta)
        self.hand_pose = np.array(hand_pose)
        self.hand_mTc = np.eye(4)
        self.hand_mTc[:3, 3] = hand_trans

        self.obj_verts = np.array(obj_verts)
        self.obj_faces = np.array(obj_faces)

        self.run_mano()
        self.hand_contact = np.zeros((self.hand_verts.shape[0], 1))     # Set to zero since we don't know the ground truth
        self.obj_contact = np.zeros((self.obj_verts.shape[0], 1))

    def calc_dist_contact(self, hand=True, obj=False, special_contact=False):
        """Set hand and object contact maps based on DiffContact method.
        This is sometimes used when ground truth contact is not known"""
        object_mesh = Meshes(verts=[torch.Tensor(self.obj_verts)], faces=[torch.Tensor(self.obj_faces)])
        hand_mesh = Meshes(verts=torch.Tensor(self.hand_verts).unsqueeze(0), faces=torch.Tensor(self.closed_faces).unsqueeze(0))
        hand_verts = torch.Tensor(self.hand_verts).unsqueeze(0)

        if not special_contact:
            obj_contact, hand_contact = calculate_contact_capsule(hand_verts, hand_mesh.verts_normals_padded(), object_mesh.verts_padded(), object_mesh.verts_normals_padded())
        else:
            # hand_verts_subdivided = util.subdivide_verts(hand_mesh.edges_packed().unsqueeze(0), hand_verts)
            # hand_normals_subdivided = util.subdivide_verts(hand_mesh.edges_packed().unsqueeze(0), hand_mesh.verts_normals_padded())
            hand_verts_subdivided = hand_verts
            hand_normals_subdivided = hand_mesh.verts_normals_padded()

            obj_contact, hand_contact = calculate_contact_capsule(hand_verts_subdivided, hand_normals_subdivided, object_mesh.verts_padded(),
                                                              object_mesh.verts_normals_padded(), caps_rad=0.003)   # needed for paper vis?

        if hand:
            self.hand_contact = hand_contact.squeeze(0).detach().cpu().numpy()
        if obj:
            self.obj_contact = obj_contact.squeeze(0).detach().cpu().numpy()

    def run_mano(self):
        """Runs forward_mano, computing the hand vertices and joints based on pose/beta parameters.
         Handles numpy-pytorch-numpy conversion"""
        if self.hand_pose.shape[0] == 48:   # Special case when we're loading GT honnotate
            mano_model = ManoLayer(mano_root='mano/models', joint_rot_mode="axisang", use_pca=False, center_idx=None, flat_hand_mean=True)
        else:   # Everything else
            mano_model = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=15, side='right', flat_hand_mean=False)

        pose_tensor = torch.Tensor(self.hand_pose).unsqueeze(0)
        beta_tensor = torch.Tensor(self.hand_beta).unsqueeze(0)
        tform_tensor = torch.Tensor(self.hand_mTc).unsqueeze(0)
        mano_verts, mano_joints = util.forward_mano(mano_model, pose_tensor, beta_tensor, [tform_tensor])
        self.hand_verts = mano_verts.squeeze().detach().numpy()
        self.hand_joints = mano_joints.squeeze().detach().numpy()

    def generate_pointnet_features(self, obj_sampled_idx):
        """Calculates per-point features for pointnet. DeepContact uses these features"""
        obj_mesh = Meshes(verts=[torch.Tensor(self.obj_verts)], faces=[torch.Tensor(self.obj_faces)])
        hand_mesh = Meshes(verts=[torch.Tensor(self.hand_verts)], faces=[torch.Tensor(util.get_mano_closed_faces())])

        obj_sampled_verts_tensor = obj_mesh.verts_padded()[:, obj_sampled_idx, :]
        _, _, obj_nearest = pytorch3d.ops.knn_points(obj_sampled_verts_tensor, hand_mesh.verts_padded(), K=1, return_nn=True)  # Calculate on object
        _, _, hand_nearest = pytorch3d.ops.knn_points(hand_mesh.verts_padded(), obj_sampled_verts_tensor, K=1, return_nn=True)  # Calculate on hand

        obj_normals = obj_mesh.verts_normals_padded()
        obj_normals = torch.nn.functional.normalize(obj_normals, dim=2, eps=1e-12)    # Because buggy mistuned value in Pytorch3d, must re-normalize
        norms = torch.sum(obj_normals * obj_normals, dim=2)  # Dot product
        obj_normals[norms < 0.8] = 0.6   # TODO hacky get-around when normal finding fails completely
        self.obj_normals = obj_normals.detach().squeeze().numpy()

        obj_sampled_verts = self.obj_verts[obj_sampled_idx, :]
        obj_sampled_normals = obj_normals[0, obj_sampled_idx, :].detach().numpy()
        hand_normals = hand_mesh.verts_normals_padded()[0, :, :].detach().numpy()

        hand_centroid = np.mean(self.hand_verts, axis=0)
        obj_centroid = np.mean(self.obj_verts, axis=0)

        # Hand features
        hand_one_hot = np.ones((self.hand_verts.shape[0], 1))
        hand_vec_to_closest = hand_nearest.squeeze().numpy() - self.hand_verts
        hand_dist_to_closest = np.expand_dims(np.linalg.norm(hand_vec_to_closest, 2, 1), axis=1)
        hand_dist_along_normal = np.expand_dims(np.sum(hand_vec_to_closest * hand_normals, axis=1), axis=1)
        hand_dist_to_joint = np.expand_dims(self.hand_verts, axis=1) - np.expand_dims(self.hand_joints, axis=0)   # Expand for broadcasting
        hand_dist_to_joint = np.linalg.norm(hand_dist_to_joint, 2, 2)
        hand_dot_to_centroid = np.expand_dims(np.sum((self.hand_verts - obj_centroid) * hand_normals, axis=1), axis=1)

        # Object features
        obj_one_hot = np.zeros((obj_sampled_verts.shape[0], 1))
        obj_vec_to_closest = obj_nearest.squeeze().numpy() - obj_sampled_verts
        obj_dist_to_closest = np.expand_dims(np.linalg.norm(obj_vec_to_closest, 2, 1), axis=1)
        obj_dist_along_normal = np.expand_dims(np.sum(obj_vec_to_closest * obj_sampled_normals, axis=1), axis=1)
        obj_dist_to_joint = np.expand_dims(obj_sampled_verts, axis=1) - np.expand_dims(self.hand_joints, axis=0)   # Expand for broadcasting
        obj_dist_to_joint = np.linalg.norm(obj_dist_to_joint, 2, 2)
        obj_dot_to_centroid = np.expand_dims(np.sum((obj_sampled_verts - hand_centroid) * obj_sampled_normals, axis=1), axis=1)

        # hand_feats = np.concatenate((hand_one_hot, hand_normals, hand_vec_to_closest, hand_dist_to_closest, hand_dist_along_normal, hand_dist_to_joint), axis=1)
        # obj_feats = np.concatenate((obj_one_hot, obj_sampled_normals, obj_vec_to_closest, obj_dist_to_closest, obj_dist_along_normal, obj_dist_to_joint), axis=1)
        hand_feats = np.concatenate((hand_one_hot, hand_dot_to_centroid, hand_dist_to_closest, hand_dist_along_normal, hand_dist_to_joint), axis=1)
        obj_feats = np.concatenate((obj_one_hot, obj_dot_to_centroid, obj_dist_to_closest, obj_dist_along_normal, obj_dist_to_joint), axis=1)

        return hand_feats, obj_feats

    def get_o3d_meshes(self, hand_contact=False, normalize_pos=False):
        """Returns Open3D meshes for visualization
        Draw with: o3dv.draw_geometries([hand_mesh, obj_mesh])"""

        hand_color = np.asarray([224.0, 172.0, 105.0]) / 255
        obj_color = np.asarray([100.0, 100.0, 100.0]) / 255

        obj_centroid = self.obj_verts.mean(0)
        if not normalize_pos:
            obj_centroid *= 0

        hand_mesh = o3dg.TriangleMesh()
        hand_mesh.vertices = o3du.Vector3dVector(self.hand_verts - obj_centroid)
        hand_mesh.triangles = o3du.Vector3iVector(HandObject.closed_faces)
        hand_mesh.compute_vertex_normals()

        if hand_contact and self.hand_contact.mean() != 0:
            util.mesh_set_color(self.hand_contact, hand_mesh)
        else:
            hand_mesh.paint_uniform_color(hand_color)

        obj_mesh = o3dg.TriangleMesh()
        obj_mesh.vertices = o3du.Vector3dVector(self.obj_verts - obj_centroid)
        obj_mesh.triangles = o3du.Vector3iVector(self.obj_faces)
        obj_mesh.compute_vertex_normals()

        if self.obj_contact.mean() != 0:
            util.mesh_set_color(self.obj_contact, obj_mesh)
        else:
            obj_mesh.paint_uniform_color(obj_color)

        return hand_mesh, obj_mesh

    def vis_hand_object(self):
        """Runs Open3D visualizer for the current data"""

        hand_mesh, obj_mesh = self.get_o3d_meshes(hand_contact=True)
        o3dv.draw_geometries([hand_mesh, obj_mesh])
