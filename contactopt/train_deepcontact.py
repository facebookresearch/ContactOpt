# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import contactopt.arguments as arguments
from contactopt.deepcontact_net import DeepContactNet
from tqdm import tqdm
import contactopt.util as util
from contactopt.loader import ContactDBDataset


def calc_losses(network_out, contact_obj_gt, contact_hand_gt, sampled_verts_idx):
    losses = dict()

    batch_size = contact_obj_gt.shape[0]
    batch = torch.zeros(sampled_verts_idx.shape, device=device, dtype=torch.long)
    for i in range(batch_size):
        batch[i, :] = i
    batch = batch.view(-1)
    contact_obj_gt = contact_obj_gt[batch, sampled_verts_idx.view(-1), :]   # Select sampled verts
    contact_obj_gt = contact_obj_gt.reshape(batch_size, sampled_verts_idx.shape[1], 1)  # Reshape into network's shape

    class_hand_gt = util.val_to_class(contact_hand_gt).squeeze(2)
    class_obj_gt = util.val_to_class(contact_obj_gt).squeeze(2)
    # print('class obj gt', class_obj_gt.shape, network_out['contact_obj'], class_obj_gt)

    losses['contact_obj'] = criterion(network_out['contact_obj'].permute(0, 2, 1), class_obj_gt)
    losses['contact_hand'] = criterion(network_out['contact_hand'].permute(0, 2, 1), class_hand_gt)

    return losses


def train_epoch(epoch):
    model.train()
    scheduler.step()
    loss_meter = util.AverageMeter('Loss', ':.2f')

    for idx, data in enumerate(tqdm(train_loader)):
        data = util.dict_to_device(data, device)
        batch_size = data['hand_pose_gt'].shape[0]

        optimizer.zero_grad()
        out = model(data['hand_verts_aug'], data['hand_feats_aug'], data['obj_sampled_verts_aug'], data['obj_feats_aug'])
        losses = calc_losses(out, data['obj_contact_gt'], data['hand_contact_gt'], data['obj_sampled_idx'])
        loss = losses['contact_obj'] * args.loss_c_obj + losses['contact_hand'] * args.loss_c_hand

        loss_meter.update(loss.item(), batch_size)   # TODO better loss monitoring
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print('{} / {}'.format(idx, len(train_loader)), loss_meter)

            global_iter = epoch * len(train_loader) + idx
            writer.add_scalar('training/loss_contact_obj', losses['contact_obj'], global_iter)
            writer.add_scalar('training/loss_contact_hand', losses['contact_hand'], global_iter)
            writer.add_scalar('training/lr', scheduler.get_lr(), global_iter)

    print('Train epoch: {}. Avg loss {:.4f} --------------------'.format(epoch, loss_meter.avg))


def test():
    model.eval()

    for idx, data in enumerate(test_loader):
        data = util.dict_to_device(data, device)

        with torch.no_grad():
            out = model(data['hand_verts_aug'], data['hand_feats_aug'], data['obj_sampled_verts_aug'], data['obj_feats_aug'])
            losses = calc_losses(out, data['obj_contact_gt'], data['hand_contact_gt'], data['obj_sampled_idx'])

    global_iter = epoch * len(train_loader)
    writer.add_scalar('testing/loss_contact_obj', losses['contact_obj'], global_iter)
    writer.add_scalar('testing/loss_contact_hand', losses['contact_hand'], global_iter)

    # print('Test epoch: Mean joint err {:.2f} cm --------------------'.format(joint_err_meter.avg))


if __name__ == '__main__':
    util.hack_filedesciptor()
    args = arguments.train_network_parse_args()

    train_dataset = ContactDBDataset(args.train_dataset, train=True)
    test_dataset = ContactDBDataset(args.test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=ContactDBDataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, collate_fn=ContactDBDataset.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepContactNet().to(device)

    if args.checkpoint != '':
        print('Attempting to load checkpoint file:', args.checkpoint)
        pretrained_dict = torch.load(args.checkpoint)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'mano' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    bin_weights = torch.Tensor(np.loadtxt(util.DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)
    # criterion = torch.nn.CrossEntropyLoss(weight=bin_weights)
    criterion = torch.nn.NLLLoss(weight=bin_weights)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)  # TODO automatic?
    writer = SummaryWriter(logdir='runs/' + args.desc)
    writer.add_text('Hyperparams', args.all_str, 0)

    for epoch in range(1, args.epochs):
        train_epoch(epoch)
        test()
        torch.save(model.state_dict(), 'checkpoints/{}.pt'.format(args.desc))
        print('\n')

