# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytorch3d.ops
from contactopt.util import *
from pytorch3d.structures import Meshes


def capsule_sdf(mesh_verts, mesh_normals, query_points, query_normals, caps_rad, caps_top, caps_bot, foreach_on_mesh):
    """
    Find the SDF of query points to mesh verts
    Capsule SDF formulation from https://iquilezles.org/www/articles/distfunctions/distfunctions.htm

    :param mesh_verts: (batch, V, 3)
    :param mesh_normals: (batch, V, 3)
    :param query_points: (batch, Q, 3)
    :param caps_rad: scalar, radius of capsules
    :param caps_top: scalar, distance from mesh to top of capsule
    :param caps_bot: scalar, distance from mesh to bottom of capsule
    :param foreach_on_mesh: boolean, foreach point on mesh find closest query (V), or foreach query find closest mesh (Q)
    :return: normalized sdf + 1 (batch, V or Q)
    """
    # TODO implement normal check?
    if foreach_on_mesh:     # Foreach mesh vert, find closest query point
        knn_dists, nearest_idx, nearest_pos = pytorch3d.ops.knn_points(mesh_verts, query_points, K=1, return_nn=True)   # TODO should attract capsule middle?

        capsule_tops = mesh_verts + mesh_normals * caps_top
        capsule_bots = mesh_verts + mesh_normals * caps_bot
        delta_top = nearest_pos[:, :, 0, :] - capsule_tops
        normal_dot = torch.sum(mesh_normals * batched_index_select(query_normals, 1, nearest_idx.squeeze(2)), dim=2)

    else:   # Foreach query vert, find closest mesh point
        knn_dists, nearest_idx, nearest_pos = pytorch3d.ops.knn_points(query_points, mesh_verts, K=1, return_nn=True)   # TODO should attract capsule middle?
        closest_mesh_verts = batched_index_select(mesh_verts, 1, nearest_idx.squeeze(2))    # Shape (batch, V, 3)
        closest_mesh_normals = batched_index_select(mesh_normals, 1, nearest_idx.squeeze(2))    # Shape (batch, V, 3)

        capsule_tops = closest_mesh_verts + closest_mesh_normals * caps_top  # Coordinates of the top focii of the capsules (batch, V, 3)
        capsule_bots = closest_mesh_verts + closest_mesh_normals * caps_bot
        delta_top = query_points - capsule_tops
        normal_dot = torch.sum(query_normals * closest_mesh_normals, dim=2)

    bot_to_top = capsule_bots - capsule_tops  # Vector from capsule bottom to top
    along_axis = torch.sum(delta_top * bot_to_top, dim=2)   # Dot product
    top_to_bot_square = torch.sum(bot_to_top * bot_to_top, dim=2)
    h = torch.clamp(along_axis / top_to_bot_square, 0, 1)   # Could avoid NaNs with offset in division here
    dist_to_axis = torch.norm(delta_top - bot_to_top * h.unsqueeze(2), dim=2)   # Distance to capsule centerline

    return dist_to_axis / caps_rad, normal_dot  # (Normalized SDF)+1 0 on endpoint, 1 on edge of capsule


def sdf_to_contact(sdf, dot_normal, method=0):
    """
    Transform normalized SDF into some contact value
    :param sdf: NORMALIZED SDF, 1 is surface of object
    :param method: select method
    :return: contact (batch, S, 1)
    """
    if method == 0:
        c = 1 / (sdf + 0.0001)   # Exponential dropoff
    elif method == 1:
        c = -sdf + 2    # Linear dropoff
    elif method == 2:
        c = 1 / (sdf + 0.0001)   # Exponential dropoff
        c = torch.pow(c, 2)
    elif method == 3:
        c = torch.sigmoid(-sdf + 2.5)
    elif method == 4:
        c = (-dot_normal/2+0.5) / (sdf + 0.0001)   # Exponential dropoff with sharp normal
    elif method == 5:
        c = 1 / (sdf + 0.0001)   # Proxy for other stuff

    return torch.clamp(c, 0.0, 1.0)


def calculate_contact_capsule(hand_verts, hand_normals, object_verts, object_normals,
                              caps_top=0.0005, caps_bot=-0.0015, caps_rad=0.001, caps_on_hand=False, contact_norm_method=0):
    """
    Calculates contact maps on object and hand.
    :param hand_verts: (batch, V, 3)
    :param hand_normals: (batch, V, 3)
    :param object_verts: (batch, O, 3)
    :param object_normals: (batch, O, 3)
    :param caps_top: ctop, distance to top capsule center
    :param caps_bot: cbot, distance to bottom capsule center
    :param caps_rad: crad, radius of the contact capsule
    :param caps_on_hand: are contact capsules placed on hand or object vertices
    :param contact_norm_method: select a distance-to-contact function
    :return: object contact (batch, O, 1), hand contact (batch, V, 1)
    """
    if caps_on_hand:
        sdf_obj, dot_obj = capsule_sdf(hand_verts, hand_normals, object_verts, object_normals, caps_rad, caps_top, caps_bot, False)
        sdf_hand, dot_hand = capsule_sdf(hand_verts, hand_normals, object_verts, object_normals, caps_rad, caps_top, caps_bot, True)
    else:
        sdf_obj, dot_obj = capsule_sdf(object_verts, object_normals, hand_verts, hand_normals, caps_rad, caps_top, caps_bot, True)
        sdf_hand, dot_hand = capsule_sdf(object_verts, object_normals, hand_verts, hand_normals, caps_rad, caps_top, caps_bot, False)

    obj_contact = sdf_to_contact(sdf_obj, dot_obj, method=contact_norm_method)# * (dot_obj/2+0.5) # TODO dotting contact normal
    hand_contact = sdf_to_contact(sdf_hand, dot_hand, method=contact_norm_method)# * (dot_hand/2+0.5)

    # print('oshape, nshape', obj_contact.shape, (dot_obj/2+0.5).shape)##

    return obj_contact.unsqueeze(2), hand_contact.unsqueeze(2)


def calculate_penetration_cost(hand_verts, hand_normals, object_verts, object_normals, is_thin, contact_norm_method, allowable_pen=0.002):
    """
    Calculates an increasing cost for hands heavily intersecting with objects.
    Foreach hand vertex, find the nearest object point, dot with object normal.
    Include "allowable-pen" buffer margin to account for hand deformation.
    """

    allowable_pen = (torch.zeros_like(is_thin) + allowable_pen) * (1 - is_thin)
    allowable_pen = allowable_pen.unsqueeze(1)

    if contact_norm_method == 5:
        hand_verts_offset = hand_verts + hand_normals * -0.004
    else:
        hand_verts_offset = hand_verts

    knn_dists, nearest_idx, nearest_pos = pytorch3d.ops.knn_points(hand_verts_offset, object_verts, K=1, return_nn=True)   # Foreach hand vert, find closest obj vert

    closest_obj_verts = batched_index_select(object_verts, 1, nearest_idx.squeeze(2))  # Shape (batch, V, 3)
    closest_obj_normals = batched_index_select(object_normals, 1, nearest_idx.squeeze(2))  # Shape (batch, V, 3)

    # print('nearest shape', nearest_pos.shape, closest_obj_verts.shape)
    delta_pos = hand_verts - closest_obj_verts
    dist_along_normal = torch.sum(delta_pos * closest_obj_normals, dim=2)   # Dot product. Negative means backward along normal

    # print('d along normal', dist_along_normal.shape)

    pen_score = torch.nn.functional.relu(-dist_along_normal - allowable_pen)
    # print('pen score', pen_score)

    return pen_score


if __name__ == '__main__':
    # Plot all sdf_to_contact mappings
    import matplotlib.pyplot as plt

    for m in range(4):
        d = torch.linspace(0, 3, 1000)
        c = sdf_to_contact(d, method=m)

        plt.plot(d.numpy(), c.numpy(), label=str(m))

    plt.ylabel('Contact value')
    plt.xlabel('Normalized SDF from center')
    plt.legend()
    plt.show()
