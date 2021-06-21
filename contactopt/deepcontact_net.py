# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import contactopt.pointnet as pointnet
import torch.nn.functional as F
from pytorch3d import ops, transforms
import contactopt.util as util


class DeepContactNet(nn.Module):
    def __init__(self, normalize_pts=True):
        super(DeepContactNet, self).__init__()
        self.pointnet = pointnet.Net()
        self.normalize_pts = normalize_pts

        pointnet_total_params = sum(p.numel() for p in self.pointnet.parameters() if p.requires_grad)
        print('Backbone params: {}'.format(pointnet_total_params))

    def forward(self, hand_verts, hand_feats, obj_verts, obj_feats):
        device = hand_verts.device
        batch_size = hand_verts.shape[0]
        out = dict()

        if self.normalize_pts:
            tform = self.get_normalizing_tform(hand_verts, obj_verts)
            hand_verts = util.apply_tform(tform, hand_verts)
            obj_verts = util.apply_tform(tform, obj_verts)
            # util.vis_pointcloud(obj_verts, hand_verts)  # View pointnet input

        x, pos, batch = self.verts_to_pointcloud(hand_verts, hand_feats, obj_verts, obj_feats)
        contact_batched = self.pointnet(x, pos, batch)
        contact = contact_batched.view(batch_size, hand_verts.shape[1] + obj_verts.shape[1], 10)

        out['contact_hand'] = contact[:, :hand_verts.shape[1], :]
        out['contact_obj'] = contact[:, hand_verts.shape[1]:, :]

        return out

    @staticmethod
    def get_normalizing_tform(hand_verts, obj_verts, random_rot=True):
        """
        Find a 4x4 rigid transform to normalize the pointcloud. We choose the object center of mass to be the origin,
        the hand center of mass to be along the +X direction, and the rotation around this axis to be random.
        :param hand_verts: (batch, 778, 3)
        :param obj_verts: (batch, 2048, 3)
        :return: tform: (batch, 4, 4)
        """
        with torch.no_grad():
            obj_centroid = torch.mean(obj_verts, dim=1)  # (batch, 3)
            hand_centroid = torch.mean(hand_verts, dim=1)

            x_vec = F.normalize(hand_centroid - obj_centroid, dim=1)  # From object to hand
            if random_rot:
                rand_vec = transforms.random_rotations(hand_verts.shape[0], device=hand_verts.device)   # Generate random rot matrix
                y_vec = F.normalize(torch.cross(x_vec, rand_vec[:, :3, 0]), dim=1)  # Make orthogonal
            else:
                ref_pt = hand_verts[:, 80, :]
                y_vec = F.normalize(torch.cross(x_vec, ref_pt - obj_centroid), dim=1)  # From object to hand ref point

            z_vec = F.normalize(torch.cross(x_vec, y_vec), dim=1)  # Z axis

            tform = ops.eyes(4, hand_verts.shape[0], device=hand_verts.device)
            tform[:, :3, 0] = x_vec
            tform[:, :3, 1] = y_vec
            tform[:, :3, 2] = z_vec
            tform[:, :3, 3] = obj_centroid

            return torch.inverse(tform)

    @staticmethod
    def verts_to_pointcloud(hand_verts, hand_feats, obj_verts, obj_feats):
        """
        Convert hand and object vertices and features from Pytorch3D padded format (batch, vertices, N)
        to Pytorch-Geometric packed format (all_vertices, N)
        """
        batch_size = hand_verts.shape[0]
        device = hand_verts.device

        ptcloud_pos = torch.cat((hand_verts, obj_verts), dim=1)
        ptcloud_x = torch.cat((hand_feats, obj_feats), dim=1)

        _, N, _ = ptcloud_pos.shape  # (batch_size, num_points, 3)
        pos = ptcloud_pos.view(batch_size * N, -1)
        batch = torch.zeros((batch_size, N), device=device, dtype=torch.long)
        for i in range(batch_size):
            batch[i, :] = i
        batch = batch.view(-1)
        x = ptcloud_x.view(-1, hand_feats.shape[2])

        # print('x', x.shape, pos.shape, batch.shape)
        return x, pos, batch
