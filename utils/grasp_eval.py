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
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import pptk

test = True

if test == True:
    no_of_objects = 39
else:
    no_of_objects = 21
no_of_repeats = 50

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=no_of_objects, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='gpg', help='output folder')
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

if test == False:

    # acquire gym interface
    gym = gymapi.acquire_gym()

    # set torch device
    device = 'cuda:0'

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = 0
    sim_params.physx.use_gpu = True

    # Set controller parameters
    # IK params
    damping = 0.05

    # OSC params
    kp = 150.
    kd = 2.0 * np.sqrt(kp)
    kp_null = 10.
    kd_null = 2.0 * np.sqrt(kp_null)

    # create sim
    sim = gym.create_sim(0 , 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise Exception("Failed to create sim")

    render = False
    # create viewer
    if render:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise Exception("Failed to create viewer")

    asset_root = "/home/vishaal/Downloads/IsaacGym_Preview_4_Package/isaacgym/assets"

    # create table asset
    table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    # create box asset
    box_size = 0.1
    #asset_options = gymapi.AssetOptions()
    bolt_options = gymapi.AssetOptions()

    bolt_options.vhacd_enabled = True

    _arm_control = None  # Tensor buffer for controlling arm
    _gripper_control = None  # Tensor buffer for controlling gripper
    _pos_control = None  # Position actions
    _effort_control = None

    box_assets = []

    for object_name in sorted(os.listdir('/home/vishaal/Downloads/assets/dataset/grasping/selected_pc')):
        base, _ = os.path.splitext(object_name)
        # print(base)
        asset_root = "/home/vishaal/Downloads/assets"
        box_asset_file = "dataset/grasping/{}/mobility.urdf".format(base)

        box_assets.append(gym.load_asset(sim, asset_root, box_asset_file, bolt_options))

        # tensor([0.4006, 0.4011, 0.3895, 0.4012, 0.4007, 0.4011, 0.4000, 0.4007, 0.3980,
        #         0.4519, 0.4006, 0.4002, 0.4008, 0.4012, 0.4012, 0.4004, 0.4012, 0.4009,
        #         0.4005, 0.3997, 0.4010, 0.4085, 0.4005, 0.4161, 0.4006, 0.3999, 0.4008,
        #         0.4006, 0.4003, 0.4003, 0.4008, 0.3989, 0.4006, 0.4011, 0.4008, 0.4013],
        #        device='cuda:0')

    # load franka asset
    asset_root = "/home/vishaal/omniverse/new_1/Object_manipulation_RL/assets"
    franka_asset_file = "urdf/franka_description/robots/gripper.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = False
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    # asset_options.vhacd_enabled = True
    # asset_options.vhacd_params.resolution = 3000000
    # asset_options.vhacd_params.max_convex_hulls = 1000
    # asset_options.vhacd_params.max_num_vertices_per_ch = 6400
    franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

    # configure franka dofs
    franka_dof_props = gym.get_asset_dof_properties(franka_asset)
    franka_lower_limits = franka_dof_props["lower"]
    franka_upper_limits = franka_dof_props["upper"]
    franka_ranges = franka_upper_limits - franka_lower_limits
    franka_mids = 0.3 * franka_upper_limits + 0.3 * franka_lower_limits

    # use position drive for all dofs
    # grippers
    franka_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"].fill(2000.0)
    franka_dof_props["damping"].fill(40.0)

    # default dof states and position targets
    franka_num_dofs = gym.get_asset_dof_count(franka_asset)
    default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)

    default_dof_pos = franka_upper_limits

    default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos

    # send to torch
    default_dof_pos_tensor = to_torch(default_dof_pos, device=device)
    franka_lower_limits = to_torch(franka_lower_limits, device=device)
    franka_upper_limits = to_torch(franka_upper_limits, device=device)

    # get link index of panda hand, which we will use as end effector
    franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
    franka_hand_index = franka_link_dict["panda_hand"]

    # configure env grid
    num_envs = no_of_objects * no_of_repeats
    num_per_row = int(math.sqrt(num_envs))
    spacing = 1.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    print("Creating %d environments" % num_envs)

    franka_pose = gymapi.Transform()
    franka_pose.p = gymapi.Vec3(0, 0, 2.0)
    franka_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0*np.random.uniform(-math.pi, math.pi))

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

    box_pose = gymapi.Transform()

    envs = []
    box_idxs = []
    hand_idxs = []
    init_pos_list = []
    init_rot_list = []

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    gripper_base_state = torch.zeros(num_envs, 13, device=device)
    obj_base_state = torch.zeros(num_envs, 13, device=device)

    # obj_heights = [0.4006, 0.4012, 0.4007, 0.4000, 0.3980, 0.4006, 0.4008, 0.4009, 0.4012,
    #         0.4004, 0.4005, 0.3997, 0.4010, 0.4005, 0.4161, 0.3999, 0.4008, 0.4006,
    #         0.4003, 0.4003, 0.4006, 0.4011, 0.4008]

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add table
        table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

        # add box
        box_pose.p.x = table_pose.p.x #+ np.random.uniform(-0.2, 0.1)
        box_pose.p.y = table_pose.p.y #+ np.random.uniform(-0.3, 0.3)
        box_pose.p.z = table_dims.z + 0.5 * box_size #obj_heights[i%no_of_objects]
        #        #table_dims.z + 0.5 * box_size

        box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0*np.random.uniform(-math.pi, math.pi))

        obj_base_state[i, 0] = box_pose.p.x
        obj_base_state[i, 1] = box_pose.p.y
        obj_base_state[i, 2] = box_pose.p.z
        obj_base_state[i, 3] = 0.000
        obj_base_state[i, 4] = 0.000
        obj_base_state[i, 5] = 0.000
        obj_base_state[i, 6] = 1.000

        box_handle = gym.create_actor(env, box_assets[i%len(box_assets)], box_pose, "box", i, 0)
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # get global index of box in rigid body state tensor
        box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
        box_idxs.append(box_idx)

        # add franka
        franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)

        # set dof properties
        gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

        # set initial dof states
        gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

        # set initial position targets
        gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

        # get inital hand pose

        hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
        hand_pose = gym.get_rigid_transform(env, hand_handle)
        gripper_base_state[i, 0] = hand_pose.p.x
        gripper_base_state[i, 1] = hand_pose.p.y
        gripper_base_state[i, 2] = hand_pose.p.z
        gripper_base_state[i, 3] = 0.000
        gripper_base_state[i, 4] = 0.000
        gripper_base_state[i, 5] = 0.000
        gripper_base_state[i, 6] = 1.000
        # init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
        # init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

        # get global index of hand in rigid body state tensor
        hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        hand_idxs.append(hand_idx)

    # point camera at middle env
    cam_pos = gymapi.Vec3(4, 3, 2)
    cam_target = gymapi.Vec3(-4, -3, 0)
    middle_env = envs[num_envs // 2 + num_per_row // 2]
    if render:
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    # ==== prepare tensors =====
    # from now on, we will use the tensor API that can run on CPU or GPU
    gym.prepare_sim(sim)


    # get rigid body state tensor
    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(_rb_states)

    # get dof state tensor
    _dof_states = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_states)
    dof_pos = dof_states[:, 0].view(num_envs, 2, 1)
    dof_vel = dof_states[:, 1].view(num_envs, 2, 1)

    _pos_control = torch.zeros((num_envs, 2), dtype=torch.float, device=device)
    _effort_control = torch.zeros_like(_pos_control)

    _gripper_control = _pos_control

    _global_indices = torch.arange(num_envs * 3, dtype=torch.int32, device=device).view(num_envs, -1)

    enable_viewer_sync = True

    # Set action tensors
    pos_action = torch.zeros_like(dof_pos).squeeze(-1)
    effort_action = torch.zeros_like(pos_action)

    time_steps = torch.zeros(num_envs, device=device)

    # xs = np.linspace(0.0, 0.08, num=9)
    # ys = np.linspace(0.0, 0.08, num=9)
    # zs = np.linspace(0.0, 0.08, num=9)

    xs = np.linspace(-0.1, 0.1, num=21)
    ys = np.linspace(-0.1, 0.1, num=21)
    zs = np.linspace(0.0, 0.1, num=11)

    rxs = [0]#np.linspace(3.14, 3.14+3*3.14/2, num=4) #np.linspace(3.14/2, 3*3.14/2, num=5)
    rys = [0]#np.linspace(3.14, 3*3.14, num=5)
    rzs = [0]#np.linspace(-1, 1, num=5) #[3.14, 3.14+3.14/2, 3.14+3.14, 3.14+3*3.14/2, 3.14 + 2*3.14] #np.linspace(3.14, 2*3.14, num=5)

    # rws = np.linspace(-1, 1, num=13)

    # xs = np.linspace(-0.1, 0.1, num=2)
    # ys = np.linspace(-0.1, 0.1, num=2)
    # zs = np.linspace(0.0, 0.1, num=2)

    # success_plot = np.zeros((21,21,11,5,5,8))

    _root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(num_envs, -1, 13)

    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    def franka_gripper_pick_simulation(x_d, y_d, z_d, r_x_d = 0, r_y_d = 0, r_z_d = 0):
        start_iter = 0

        while True:

            gym.refresh_rigid_body_state_tensor(sim)
            gym.refresh_dof_state_tensor(sim)
            gym.refresh_jacobian_tensors(sim)
            gym.refresh_mass_matrix_tensors(sim)

            # refresh tensors
            box_pos = rb_states[box_idxs, :3]
            hand_pos = rb_states[hand_idxs, :3]
            # box_rot = rb_states[box_idxs, 3:7]

            # grasp_loc = torch.zeros_like(box_pos)
            # grasp_loc[:, 0] = box_pos[:, 0] + x_d
            # grasp_loc[:, 1] = box_pos[:, 1] + y_d
            # grasp_loc[:, 2] = box_pos[:, 2] + z_d

            # x_d = 0.00
            # y_d = -0.04
            # z_d = 0.13
            # r_x_d = -0.0015927
            # r_y_d = -0.0000
            # r_z_d = 0.000000
            # r_w_d = -0.9999987

            env_ids = torch.arange(start=0, end=num_envs, device=device, dtype=torch.long)

            multi_env_ids_cubes_int32 = _global_indices[env_ids, 2].flatten()

            if start_iter == 0:
                gripper_state = obj_base_state.clone()
                offset = obj_base_state.clone()
                offset[:, 2] = 1.0
                gripper_state[:, :10] = offset[:, :10]
                # start_iter += 1

                # print(gripper_state)

            quat_angle = gymapi.Quat.from_axis_angle(gymapi.Vec3(r_x_d, r_y_d, r_z_d),
                                                     0.5 * math.pi)  # gymapi.Quat.from_euler_zyx(r_z_d, r_y_d, r_x_d)

            r_x_d1 = -0.0015927
            r_y_d1 = -0.0000
            r_z_d1 = 0.000000
            r_w_d1 = -0.9999987

            if start_iter == 10:
                obj_height = box_pos[:, 2]

            if start_iter == 10:
                gripper_state = obj_base_state.clone()
                offset = 0 * obj_base_state.clone()
                offset[:, 0] = x_d
                offset[:, 1] = y_d
                offset[:, 2] = 0.103 + obj_height + z_d
                # print(offset[29, 0], offset[29, 1], offset[29, 2])
                offset[:, 3] = r_x_d1
                offset[:, 4] = r_y_d1
                offset[:, 5] = r_z_d1
                offset[:, 6] = r_w_d1
                # gripper_state[:7] = obj_base_state[:7] + torch.tensor([x_d, y_d, z_d, quat_angle.x, quat_angle.y, quat_angle.z, quat_angle.w]).cuda()
                gripper_state[:, :2] = obj_base_state[:, :2] + offset[:, :2]
                gripper_state[:, 2] = offset[:, 2]
                gripper_state[:, 3:7] = obj_base_state[:, 3:7] + offset[:, 3:7]

                # print(gripper_state)

                # start_iter += 1
                ####torch.tensor([0.00, -0.05, 0.07, -0.0015927, 0, 0, -0.9999987 , 0, 0, 0.0]).cuda()

            close_gripper = torch.tensor([[False]]*num_envs).cuda()
            grip_acts = torch.where(close_gripper, torch.Tensor([[-0.01, -0.01]] * num_envs).to(device),
                                    torch.Tensor([[0.04, 0.04]] * num_envs).to(device))
            _gripper_control[:, :] = grip_acts
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(_pos_control))
            # print(pos_action)
            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(_effort_control))

            if start_iter >= 15:
                close_gripper = torch.tensor([[True]]*num_envs).cuda()
                grip_acts = torch.where(close_gripper, torch.Tensor([[-0.01, -0.01]] * num_envs).to(device),
                                        torch.Tensor([[0.04, 0.04]] * num_envs).to(device))
                _gripper_control[:, :] = grip_acts
                gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(_pos_control))
                # print(pos_action)
                gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(_effort_control))

            # if start_iter == 16:
            #     time.sleep(10)

            if start_iter >= 16:
                # gripper_state[:10] = gripper_state[:10] + torch.tensor([0.0, 0.00, 0.000, 0, 0, 0, 0, 0.0, 0, 0.1]).cuda()
                rb_forces = torch.zeros((num_envs, 10, 3)).cuda()
                # print(rb_forces.shape)
                # quit()
                rb_forces[:, 2, 2] = 5
                gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(rb_forces), None, gymapi.GLOBAL_SPACE)

            # print(gripper_state)

            if start_iter >= 0 and start_iter < 16:
                _root_state[:, 2, :] = gripper_state.view(num_envs, 13)

                gym.set_actor_root_state_tensor_indexed(
                    sim, gymtorch.unwrap_tensor(_root_state),
                    gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

            if start_iter >= 70:
                multi_env_ids_cubes_int32 = _global_indices[env_ids, 1].flatten()
                _root_state[:, 1, :] = obj_base_state.clone().view(num_envs, 13)

                gym.set_actor_root_state_tensor_indexed(
                    sim, gymtorch.unwrap_tensor(_root_state),
                    gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

                dist = torch.norm(hand_pos - box_pos, dim=-1)

                # print(dist)
                # print(box_pos)
                # quit()

                return torch.logical_and(dist < 0.5, box_pos[:, 2] > 0.6)

                # if dist < 0.5 and box_pos[0, 2] > 0.6:
                #     # success_plot[i, j, k, i1, j1, k1] = 1
                #     print(x_d, ",", y_d, ",", z_d, " --------- ", "Lifted")
                #     return True
                # else:
                #     print(x_d, ",", y_d, ",", z_d, " --------- ", "Not Lifted")
                #     return False

                # break

            if render:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

            # step the physics
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # gym.refresh_rigid_body_state_tensor(sim)
            # gym.refresh_dof_state_tensor(sim)
            # gym.refresh_jacobian_tensors(sim)
            # gym.refresh_mass_matrix_tensors(sim)

            start_iter += 1



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
    classifier.load_state_dict(torch.load("/home/vishaal/omniverse/new_1/pointnet.pytorch/utils/gpg_bak_f/gpg/gpg_model_5.pth"))

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

    for item in sorted(os.listdir('/home/vishaal/Downloads/assets/dataset/grasping/test_objects')):
        base, _ = os.path.splitext(item)
        objects_list.append(base)
        gt_grasps.append(np.loadtxt('/home/vishaal/Downloads/assets/dataset/grasping/gt_grasps/gt_{}.seg'.format(base)))

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
    score = 0
    tot_no_pts = 0
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
            preds_each_obj.append(pred[id, qr]/pred[id].max())
            gt = gt_grasps[id][choice[id][qr]]



            if pred[id, qr]/pred[id].max()>=0.5 and gt == 1:
                tp += 1
            if pred[id, qr]/pred[id].max()>=0.5 and gt == 0:
                fp += 1
            if gt == 1:
                tot_no_grasp_pts += 1
            # gt = gt_grasps[id][choice[id][qr]]
            # if (pred[id, qr]>=0.5 and gt == 1) or (pred[id, qr]<0.5 and gt == 0):
            #     score += 1
            # tot_no_pts += 1
            # # color.append(gt)
            # # print('here')
            color.append(pred[id, qr] / pred[id].max())
            # if pred[id, qr]/pred[id].max() > 0.5:
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
            os.makedirs('/home/vishaal/omniverse/new_1/pointnet.pytorch/utils/gpg_op/{}'.format(objects_list[id]))
        except OSError:
            pass
        v.record('/home/vishaal/omniverse/new_1/pointnet.pytorch/utils/gpg_op/{}'.format(objects_list[id]), poses, 2 * np.arange(5), interp='linear')
        # # quit()
        if tot_no_grasp_pts == 0 or (fp == 0 and tp == 0):
            print(objects_list[id], ": No GT Grasp Points")
        else:
            print(objects_list[id], ":" , tp / (tp+fp))
    preds_all_obj = np.array(preds_all_obj)
    np.save('/home/vishaal/omniverse/new_1/pointnet.pytorch/utils/grasp_sim_preds.npy', preds_all_obj)
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
            points, coll_check_op, point_set_before_transform = data
            # print(points.shape)
            points = points.transpose(2, 1)
            point_set_before_transform = point_set_before_transform.transpose(2, 1)
            # print(points.shape)
            # quit()
            points, coll_check_op, point_set_before_transform = points.cuda(), coll_check_op.cuda(), point_set_before_transform.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            # print(pred.shape)
            # quit()
            pred = torch.sigmoid(pred)
            # pred_max = torch.max(pred, 1).values
            # pred_max = pred_max.repeat_interleave(1000)
            # pred_max = pred_max.reshape(no_of_objects, -1)
            # pred = torch.div(pred, pred_max)
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
                indices_coll_check_true = torch.where(coll_check_op[q] == 1)
                indices_coll_check_false = torch.where(coll_check_op[q] == 0)

                samples_false = torch.randint(indices_coll_check_false[0].shape[0], (int(num_envs / no_of_objects),))
                mask[q*1000+indices_coll_check_false[0][samples_false]] = 1
                # print(indices[0])
                # print(indices[0].shape)
                samples = torch.randint(indices_coll_check_true[0].shape[0], (int(num_envs / no_of_objects),))
                samples_for_all_env = torch.cat((samples_for_all_env, q*1000+indices_coll_check_true[0][samples]), -1)
                mask[q*1000+indices_coll_check_true[0][samples]] = 1
                # print(samples)
                # sampled_idxs = indices[0][samples], -1)
                # print(point_set_before_transform[q, 0, indices[0][samples]])
                xs = torch.cat((xs, point_set_before_transform[q, 0, indices_coll_check_true[0][samples]]), -1)
                ys = torch.cat((ys, point_set_before_transform[q, 1, indices_coll_check_true[0][samples]]), -1)
                zs = torch.cat((zs, point_set_before_transform[q, 2, indices_coll_check_true[0][samples]]), -1)
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
            dgt = franka_gripper_pick_simulation(x_d = xs, y_d = ys, z_d = zs)
            # dgt = franka_gripper_pick_simulation(x_d = 0.0, y_d = -0.04, z_d = 0.13)
            # print(dgt)
            gt[samples_for_all_env.to(torch.int64)] = dgt.to(torch.float)
            # print(torch.sigmoid(pred).shape)
            # print((gt.reshape(-1, 1)).shape)

            loss = 1.0*F.binary_cross_entropy(pred, gt.reshape(-1, 1), weight = mask.reshape(-1,1))

            # print(pred[mask==1], gt.reshape(-1, 1)[mask==1])
            # quit()
            range_.set_postfix(loss = loss.item())
            loss.backward()
            optimizer.step()

    #####Evaluation
    j, data = next(enumerate(trainevalloader, 0))
    points, _, point_set_before_transform = data
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
        if pred[id, qr]/pred[id].max() > 0.5:
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
        # pred_choice = pred.data.max(1)[1]
        # correct = pred_choice.eq(target.data).cpu().sum()

        # print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))

    # if epoch % 10 == 0:
    #     j, data = next(enumerate(testdataloader, 0))
    #     points, target = data
    #     target_plot = target.clone()
    #     points = points.transpose(2, 1)
    #     points, target = points.cuda(), target.cuda()
    #     classifier = classifier.eval()
    #     pred, _, _ = classifier(points)
    #     pred = pred.view(-1, num_classes)
    #     target = target.view(-1, 1)[:, 0] - 1
    #     loss = F.nll_loss(pred, target)
    #     pred_choice = pred.data.max(1)[1]
    #     correct = pred_choice.eq(target.data).cpu().sum()
    #     print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * 2500)))
    #
    #     # print(points.shape)
    #     # quit()
    #     points_set = []
    #     color = []
    #     import pptk
    #
    #     for k in range(1):
    #         for q in range(2500):
    #             points_set.append([points[k, 2, q].cpu(), points[k, 0, q].cpu(), points[k, 1, q].cpu()])
    #             pred_copy = pred.view(-1, 2500, num_classes)
    #             # pred_choice = torch.argmax(pred_copy[k, q].cpu())
    #             print(target_plot)
    #             quit()
    #             pred_choice = torch.argmax(target_plot[k,q])
    #             # print(pred[i,j])
    #             # quit()
    #             color.append(pred_choice + 1)
    #     v = pptk.viewer(points_set, color)
    #     v.color_map(color_map)
    #     # print(color)
    #     # Sets a similar view to the gym viewer in the PPTK viewer
    #     v.set(lookat=[-0.5, -0.5, 0], r=5, theta=0.4, phi=0.707)
    #     print("Point Cloud Complete")

    # if epoch%10==0:
    #
    #     point_set = np.loadtxt('/home/vishaal/omniverse/new_1/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/isaacgym_pc_data/points_test/points_0.pts').astype(np.float32)
    #     # point_set = np.loadtxt('/home/vishaal/omniverse/new_1/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/03001627/points/1a6f615e8b1b5ae4dbbc9440457e303e.pts').astype(np.float32)
    #     # target = np.loadtxt(
    #     #     '/home/vishaal/omniverse/new_1/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/isaacgym_pc_data/points_label/duck1.seg').astype(
    #     #     np.float32)
    #     choice = np.random.choice(point_set.shape[0], 2500)
    #
    #     # resample
    #     point_set = point_set[choice, :]
    #     # target = target[:500]
    #     # point_set_copy = point_set.copy()
    #
    #     point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
    #     dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    #     point_set = point_set / dist  # scale
    #
    #     # print(point_set.shape)
    #     # point_set_ip = point_set[:2500]
    #     point_set_ip = torch.from_numpy(point_set).view(-1, 2500, 3)
    #     point_set_ip = point_set_ip.transpose(2, 1)
    #     point_set_ip = point_set_ip.cuda()
    #
    #     # print(point_set_ip.shape)
    #     classifier1 = classifier.eval()
    #
    #     pred, _, _ = classifier1(point_set_ip)
    #     import pptk
    #
    #     points = []
    #     color = []
    #     for k in range(point_set_ip.shape[0]):
    #         for q in range(point_set_ip.shape[2]):
    #             points.append([point_set_ip[k, 2, q].cpu(), point_set_ip[k, 0, q].cpu(), point_set_ip[k, 1, q].cpu()])
    #             pred_choice = torch.argmax(pred[k,q].cpu()) #target[q]
    #             # print(pred[i,j])
    #             # quit()
    #             color.append(pred_choice+1)
    #     if epoch!=0:
    #         v.close()
    #     v = pptk.viewer(points, color)
    #     v.color_map(color_map)
    #     # print(color)
    #     # Sets a similar view to the gym viewer in the PPTK viewer
    #     v.set(lookat=[-0.5, -0.5, 0], r=5, theta=0.4, phi=0.707)
    #     print("Point Cloud Complete")
    # # print(pred_choice.shape)
    # # print(pred.shape)

    # torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

## benchmark mIOU
# shape_ious = []
# for i,data in tqdm(enumerate(testdataloader, 0)):
#     points, target = data
#     points = points.transpose(2, 1)
#     points, target = points.cuda(), target.cuda()
#     classifier = classifier.eval()
#     pred, _, _ = classifier(points)
#     pred_choice = pred.data.max(2)[1]
#
#     pred_np = pred_choice.cpu().data.numpy()
#     target_np = target.cpu().data.numpy() - 1
#
#     for shape_idx in range(target_np.shape[0]):
#         parts = range(num_classes)#np.unique(target_np[shape_idx])
#         part_ious = []
#         for part in parts:
#             I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#             U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#             if U == 0:
#                 iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
#             else:
#                 iou = I / float(U)
#             part_ious.append(iou)
#         shape_ious.append(np.mean(part_ious))
#
# print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))