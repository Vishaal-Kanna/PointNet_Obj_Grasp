from __future__ import print_function
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import argparse
import os
import random
import math
import torch
import time
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset_PN_from_gt import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import pptk

test = False
device = 'cuda:0'

if test == True:
    no_of_objects = 39
else:
    no_of_objects = 21
no_of_repeats = 50

num_envs = no_of_objects * no_of_repeats

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=no_of_objects, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='PN_gpg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

# # Add custom arguments
# custom_parameters = [
#     {"name": "--controller", "type": str, "default": "osc",
#      "help": "Controller to use for Franka. Options are {ik, osc}"},
#     {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
# ]
# args = gymutil.parse_arguments(
#     description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
#     custom_parameters=custom_parameters,
# )

opt = parser.parse_args()
print(opt)

# opt.manualSeed = random.randint(1, 10000)  # fix seed
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

train_eval_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    data_augmentation=False)
trainevalloader = torch.utils.data.DataLoader(
    train_eval_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    test=True,
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

if test == True:
    classifier.load_state_dict(torch.load(os.getcwd() + '../utils/PN_gpg/gpg_model_10.pth'))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

# print(len(dataset))
# quit()


color_map = np.array([[1, 0, 0], [0, 1, 0]])#, [1, 0, 0], [0, 1, 1], [1, 0, 1]])

if test == True:

    objects_list = []
    gt_grasps = []

    for item in sorted(os.listdir(os.getcwd() + '../dataset/grasping/test_objects')):
        base, _ = os.path.splitext(item)
        objects_list.append(base)
        gt_grasps.append(np.loadtxt(os.getcwd() + '../dataset/grasping/gt_grasps/gt_{}.seg'.format(base)))

    j, data = next(enumerate(testdataloader, 0))
    points, choice, point_set_before_transform = data
    # print(points.shape)
    points = points.transpose(2, 1)
    # point_set_before_transform = point_set_before_transform.transpose(2, 1)
    points = points.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred = torch.sigmoid(pred).detach().cpu().numpy()
    # print(pred.shape)
    # quit()
    # print(point_set_before_transform[7, : ,:].shape)

    preds_all_obj = []
    for id in range(0, 39):
        tp = 0
        fp = 0
        tot_no_grasp_pts = 0
        lst = []
        color = []
        preds_each_obj = []
        for qr in range(1000):
            lst.append([point_set_before_transform[id, qr, 0], point_set_before_transform[id, qr, 1],
                        point_set_before_transform[id, qr, 2]])
            # color.append(pred[id, qr])
            # print(gt_grasps[id].shape)
            # print(choice)
            # quit()
            gt = gt_grasps[id][choice[id][qr]]
            preds_each_obj.append(pred[id, qr])
            if pred[id, qr]>=0.5 and gt == 1:
                tp += 1
            if pred[id, qr]>=0.5 and gt == 0:
                fp += 1
            if gt == 1:
                tot_no_grasp_pts += 1
            color.append(pred[id, qr])
            # print('here')
            # if gt > 0.5:
            #     color.append(1)
            # else:
            #     color.append(0)
        preds_all_obj.append(preds_each_obj)
        if id > 0:
            v.close()
        v = pptk.viewer(lst, color)
        v.color_map(color_map)
        poses = []
        poses.append([0, 0, 0, 0 * np.pi / 2, 0.2* np.pi / 4, 1])
        poses.append([0, 0, 0, 1 * np.pi / 2, 0.2* np.pi / 4, 1])
        poses.append([0, 0, 0, 2 * np.pi / 2, 0.2* np.pi / 4, 1])
        poses.append([0, 0, 0, 3 * np.pi / 2, 0.2* np.pi / 4, 1])
        poses.append([0, 0, 0, 4 * np.pi / 2, 0.2* np.pi / 4, 1])
        v.set(lookat=[-0.0, -0.0, 0], r=0.5, theta=0.1, phi=0.1, show_axis=False)
        try:
            os.makedirs(os.getcwd() + '../utils/PN_op/{}'.format(objects_list[id]))
        except OSError:
            pass
        v.record(os.getcwd() + '../utils/PN_op/{}'.format(objects_list[id]), poses, 2 * np.arange(5), interp='linear')
        # quit()
        if tot_no_grasp_pts == 0 or fp == 0:
            print(objects_list[id], ": No GT Grasp Points")
        else:
            print(objects_list[id], ":" , tp / (tp+fp))
    preds_all_obj = np.array(preds_all_obj)
    np.save(os.getcwd() + '../utils/PN_op_preds.npy', preds_all_obj)
        # time.sleep(10)
    # print("Score : ", score/tot_no_pts)

    quit()

for epoch in range(opt.nepoch):
    scheduler.step()
    # print(dataloader)
    print("Epoch : ", epoch)
    range_ = tqdm(np.arange(100))
    for r, _ in enumerate(range_):
        for i, data in enumerate(dataloader, 0):
            points, coll_check_op, gt_collected, point_set_before_transform = data
            # print(points.shape)
            points = points.transpose(2, 1)
            point_set_before_transform = point_set_before_transform.transpose(2, 1)
            # print(points.shape)
            # quit()
            points, coll_check_op, gt_collected, point_set_before_transform = points.cuda(), coll_check_op.cuda(), gt_collected.cuda(), point_set_before_transform.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            # print(pred.shape)
            # quit()
            pred = pred.view(-1, num_classes)

            # indicies = torch.where(coll_check_op == 1)
            # print(indicies)
            # print(indicies[0].shape)
            # print(indicies[1].shape)
            #
            # print(coll_check_op.shape)

            xs = torch.tensor([], device=device)
            ys = torch.tensor([], device=device)
            zs = torch.tensor([], device=device)

            samples_for_all_env = torch.tensor([], device=device)

            mask = torch.zeros((no_of_objects*1000), device=device)

            gt = torch.zeros((no_of_objects*1000), device=device)

            for q in range(no_of_objects):
                # print()
                indices_gt_true = torch.where(gt_collected[q] == 1)
                indices_gt_false = torch.where(gt_collected[q] == 0)
                samples_false = torch.randint(indices_gt_false[0].shape[0], (int(num_envs / no_of_objects),))
                mask[q*1000+indices_gt_false[0][samples_false]] = 1
                # print(indices[0])
                # print(indices[0].shape)
                # print(indices_gt_true[0].shape[0])
                samples = torch.randint(indices_gt_true[0].shape[0], (int(num_envs / no_of_objects),))
                samples_for_all_env = torch.cat((samples_for_all_env, q*1000+indices_gt_true[0][samples]), -1)
                mask[q*1000+indices_gt_true[0][samples]] = 1
                # print(samples)
                # sampled_idxs = indices[0][samples], -1)
                # print(point_set_before_transform[q, 0, indices[0][samples]])
                # xs = torch.cat((xs, point_set_before_transform[q, 0, indices_gt_true[0][samples]]), -1)
                # ys = torch.cat((ys, point_set_before_transform[q, 1, indices_gt_true[0][samples]]), -1)
                # zs = torch.cat((zs, point_set_before_transform[q, 2, indices_gt_true[0][samples]]), -1)
            # print(samples_for_all_env)
            # quit()

            # gt[samples_for_all_env] = dgt

                # ys = (point_set_before_transform[q, 1, indices[0][samples]].transpose(1, 0)).reshape(-1, )
                # zs = (point_set_before_transform[q, 2, indices[0][samples]].transpose(1, 0)).reshape(-1, )
            #     print(samples_idxs)
            # print(xs)
            #
            #
            # quit()
            # sampled_idxs = torch.randint(1000, (int(num_envs / no_of_objects),))
            # sampled_pred = pred[sampled_idxs]
            # # print(torch.argmax(pred[0].cpu()))
            # coll_check_op = coll_check_op.view(-1, 1)[:, 0] - 1
            #
            # # print(point_set_before_transform.shape)
            #
            # xs = (point_set_before_transform[:, 0, sampled_idxs].transpose(1, 0)).reshape(-1,)
            # ys = (point_set_before_transform[:, 1, sampled_idxs].transpose(1, 0)).reshape(-1,)
            # zs = (point_set_before_transform[:, 2, sampled_idxs].transpose(1, 0)).reshape(-1,)
            # print(point_set_before_transform[29])
            # quit()
            # print(xs)
            # print(ys)
            # print(zs)
            # print(xs)
            # print(xs.shape)
            # xs = (points[:, 0, sampled_idxs].transpose(1, 0)).reshape(-1,)
            # print(xs)
            # print(xs.shape)
            # quit()
            # print(xs[29],ys[29],zs[29])
            # dgt = franka_gripper_pick_simulation(x_d = xs, y_d = ys, z_d = zs)
            # dgt = franka_gripper_pick_simulation(x_d = 0.0, y_d = -0.04, z_d = 0.13)
            # print(dgt)
            # gt[samples_for_all_env.to(torch.int64)] = dgt.to(torch.float)
            # print(torch.sigmoid(pred).shape)
            # print((gt.reshape(-1, 1)).shape)
            loss = 1.0*F.binary_cross_entropy(torch.sigmoid(pred), gt_collected.reshape(-1, 1).float(), weight = mask.reshape(-1,1))

            # print(pred[mask==1], gt.reshape(-1, 1)[mask==1])
            # quit()
            range_.set_postfix(loss = loss.item())
            loss.backward()
            optimizer.step()

    #####Evaluation
    j, data = next(enumerate(trainevalloader, 0))
    points, _, _, point_set_before_transform = data
    # print(points.shape)
    points = points.transpose(2, 1)
    # point_set_before_transform = point_set_before_transform.transpose(2, 1)
    points = points.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred = torch.sigmoid(pred).detach().cpu().numpy()
    # print(pred.shape)
    # quit()
    # print(point_set_before_transform[7, : ,:].shape)
    lst = []
    color = []
    id = random.randint(0, no_of_objects-1)
    for qr in range(1000):
        lst.append([point_set_before_transform[id, qr, 0], point_set_before_transform[id, qr, 1], point_set_before_transform[id, qr, 2]])
        # color.append(pred[id, qr])
        if pred[id, qr] > 0.5:
            color.append(1)
        else:
            color.append(0)
    if epoch>0:
        v.close()
    v = pptk.viewer(lst, color)
    v.color_map(color_map)
    # quit()
    v.set(lookat=[-0.5, -0.5, 0], r=5, theta=0.4, phi=0.707)

    torch.save(classifier.state_dict(), '%s/gpg_model_%d.pth' % (opt.outf, epoch))
