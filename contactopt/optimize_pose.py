# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytorch3d
import time
from contactopt.loader import *
from manopth.manolayer import ManoLayer
from manopth import rodrigues_layer
import contactopt.diffcontact as calculate_contact
import contactopt.util as util
from contactopt.hand_object import HandObject
from contactopt.visualize import show_optimization


def optimize_pose(data, hand_contact_target, obj_contact_target, n_iter=250, lr=0.01, w_cont_hand=2, w_cont_obj=1,
                  save_history=False, ncomps=15, w_cont_asym=2, w_opt_trans=0.3, w_opt_pose=1, w_opt_rot=1,
                  caps_top=0.0005, caps_bot=-0.001, caps_rad=0.001, caps_on_hand=False,
                  contact_norm_method=0, w_pen_cost=600, w_obj_rot=0, pen_it=0):
    """Runs differentiable optimization to align the hand with the target contact map.
    Minimizes the loss between ground truth contact and contact calculated with DiffContact"""
    batch_size = data['hand_pose_aug'].shape[0]
    device = data['hand_pose_aug'].device

    opt_vector = torch.zeros((batch_size, ncomps + 6 + 3), device=device)   # 3 hand rot, 3 hand trans, 3 obj rot
    opt_vector.requires_grad = True

    mano_model = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=ncomps, side='right', flat_hand_mean=False).to(device)

    if data['obj_sampled_idx'].numel() > 1:
        obj_normals_sampled = util.batched_index_select(data['obj_normals_aug'], 1, data['obj_sampled_idx'])
    else:   # If we're optimizing over all verts
        obj_normals_sampled = data['obj_normals_aug']

    optimizer = torch.optim.Adam([opt_vector], lr=lr, amsgrad=True)  # AMSgrad helps
    loss_criterion = torch.nn.L1Loss(reduction='none')  # Benchmarked, L1 performs best vs MSE/SmoothL1
    opt_state = []
    is_thin = mesh_is_thin(data['mesh_aug'].num_verts_per_mesh())
    # print('is thin', is_thin, data['mesh_aug'].num_verts_per_mesh())

    for it in range(n_iter):
        optimizer.zero_grad()

        mano_pose_out = torch.cat([opt_vector[:, 0:3] * w_opt_rot, opt_vector[:, 3:ncomps+3] * w_opt_pose], dim=1)
        mano_pose_out[:, :18] += data['hand_pose_aug']
        tform_out = util.translation_to_tform(opt_vector[:, ncomps+3:ncomps+6] * w_opt_trans)

        hand_verts, hand_joints = util.forward_mano(mano_model, mano_pose_out, data['hand_beta_aug'], [data['hand_mTc_aug'], tform_out])   # 2.2ms

        if contact_norm_method != 0 and not caps_on_hand:
            with torch.no_grad():   # We need to calculate hand normals if using more complicated methods
                mano_mesh = Meshes(verts=hand_verts, faces=mano_model.th_faces.repeat(batch_size, 1, 1))
                hand_normals = mano_mesh.verts_normals_padded()
        else:
            hand_normals = torch.zeros(hand_verts.shape, device=device)

        obj_verts = data['obj_sampled_verts_aug']
        obj_normals = obj_normals_sampled

        obj_rot_mat = rodrigues_layer.batch_rodrigues(opt_vector[:, ncomps+6:])
        obj_rot_mat = obj_rot_mat.view(batch_size, 3, 3)

        if w_obj_rot > 0:
            obj_verts = util.apply_rot(obj_rot_mat, obj_verts, around_centroid=True)
            obj_normals = util.apply_rot(obj_rot_mat, obj_normals)

        contact_obj, contact_hand = calculate_contact.calculate_contact_capsule(hand_verts, hand_normals, obj_verts, obj_normals,
                              caps_top=caps_top, caps_bot=caps_bot, caps_rad=caps_rad, caps_on_hand=caps_on_hand, contact_norm_method=contact_norm_method)

        contact_obj_sub = obj_contact_target - contact_obj
        contact_obj_weighted = contact_obj_sub + torch.nn.functional.relu(contact_obj_sub) * w_cont_asym  # Loss for 'missing' contact higher
        loss_contact_obj = loss_criterion(contact_obj_weighted, torch.zeros_like(contact_obj_weighted)).mean(dim=(1, 2))

        contact_hand_sub = hand_contact_target - contact_hand
        contact_hand_weighted = contact_hand_sub + torch.nn.functional.relu(contact_hand_sub) * w_cont_asym  # Loss for 'missing' contact higher
        loss_contact_hand = loss_criterion(contact_hand_weighted, torch.zeros_like(contact_hand_weighted)).mean(dim=(1, 2))

        loss = loss_contact_obj * w_cont_obj + loss_contact_hand * w_cont_hand

        if w_pen_cost > 0 and it >= pen_it:
            pen_cost = calculate_contact.calculate_penetration_cost(hand_verts, hand_normals, data['obj_sampled_verts_aug'], obj_normals_sampled, is_thin, contact_norm_method)
            loss += pen_cost.mean(dim=1) * w_pen_cost

        out_dict = {'loss': loss.detach().cpu()}
        if save_history:
            out_dict['hand_verts'] = hand_verts.detach().cpu()#.numpy()
            out_dict['hand_joints'] = hand_joints.detach().cpu()#.numpy()
            out_dict['contact_obj'] = contact_obj.detach().cpu()#.numpy()
            out_dict['contact_hand'] = contact_hand.detach().cpu()#.numpy()
            out_dict['obj_rot'] = obj_rot_mat.detach().cpu()#.numpy()
        opt_state.append(out_dict)

        loss.mean().backward()
        optimizer.step()

    tform_full_out = util.aggregate_tforms([data['hand_mTc_aug'], tform_out])
    return mano_pose_out, tform_full_out, obj_rot_mat, opt_state


def show_optimization_video(data, device):
    """Displays video of optimization process of hand converging"""
    data_gpu = util.dict_to_device(data, device)
    contact_obj_pred = util.batched_index_select(data_gpu['obj_contact_gt'], 1, data_gpu['obj_sampled_idx'])

    out_pose, out_tform, obj_rot_mat, opt_state = optimize_pose(data_gpu, data_gpu['hand_contact_gt'], contact_obj_pred, save_history=True)

    show_optimization(data, opt_state, hand_contact_target=data['hand_contact_gt'], obj_contact_target=contact_obj_pred.detach().cpu(), is_video=True, vis_method=1)


if __name__ == '__main__':
    """Show a video optimization from perturbed pose"""
    test_dataset = ContactDBDataset('data/perturbed_contactpose_test.pkl')
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=ContactDBDataset.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx, data in enumerate(dataloader):
        show_optimization_video(data, device)   # do optimization and show video

        if idx >= 10:
            break
