r"""
We followed the process of constructing semantic map provided by chaplot.
However, their work doesn't support to build open-vocabulary semantic map.
We improved this by using dynamic feature map.

REFERENCE:
https://github.com/devendrachaplot/Object-Goal-Navigation/tree/master
"""

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import habitat_extensions.pose_utils as pu

from vlnce_baselines.utils.constant import *
from vlnce_baselines.utils.map_utils import *
import vlnce_baselines.utils.depth_utils as du
import vlnce_baselines.utils.visualization as vu
from vlnce_baselines.utils.data_utils import OrderedSet

from vlnce_baselines.map.mapping_skeleton import SkeletonMixin
from vlnce_baselines.map.mapping_overlay import OverlayMixin
from vlnce_baselines.map.mapping_viz import VizMixin

# ------------------------


class Semantic_Mapping(SkeletonMixin, OverlayMixin, VizMixin, nn.Module):
    r"""
    Semantic_Mapping moudle: initialize map and do the projection.
    Projection procedure:
    1. use depth observation to comupte a point cloud
    2. associate predicted semantic categories with each point in the point cloud
    3. project point cloud into 3D space to get voxel representation
    4. summing over height dimension
    """

    """
    Initialize map variables:
    Full map consists of multiple channels containing the following:
    1. Obstacle Map
    2. Explored Area
    3. Current Agent Location
    4. Past Agent Locations
    5,6,7,.. : Semantic Categories
    """
    MAP_CHANNELS = map_channels # map_channels is defined in constant.py

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()
        self.args = args
        self.dropout = 0.5
        self.n_channels = 3
        self.goal = None
        self.curr_loc = None
        self.last_loc = None
        self.vis_classes = []

        self.fov = args.HFOV
        self.min_z = args.MIN_Z # a lager min_z could lost some information on the floor, 2cm is ok
        self.device = args.DEVICE
        self.du_scale = args.DU_SCALE # depth unit
        self.visualize = args.VISUALIZE
        self.screen_w = args.FRAME_WIDTH
        self.screen_h = args.FRAME_HEIGHT
        self.vision_range = args.VISION_RANGE # args.vision_range=100(cm)
        self.resolution = args.MAP_RESOLUTION
        self.print_images = args.PRINT_IMAGES
        self.z_resolution = args.MAP_RESOLUTION
        self.num_environments = args.NUM_ENVIRONMENTS
        self.global_downscaling = args.GLOBAL_DOWNSCALING
        self.cat_pred_threshold = args.CAT_PRED_THRESHOLD
        self.exp_pred_threshold = args.EXP_PRED_THRESHOLD
        self.map_pred_threshold = args.MAP_PRED_THRESHOLD
        self.map_shape = (self.args.MAP_SIZE_CM // self.resolution,
                          self.args.MAP_SIZE_CM // self.resolution)
        self.map_size_cm = args.MAP_SIZE_CM // args.GLOBAL_DOWNSCALING

        if self.visualize or self.print_images:
            self.vis_image = vu.init_vis_image()
            self.rgb_vis = None

        # 72; 3.6m is about the height of one floor
        self.max_height = int(360 / self.z_resolution)

        # -8; we can use negative height to ensure information on the floor is contained
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.AGENT_HEIGHT * 100. # 0.88 * 100 = 88cm
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0] # [250, 0, pi/2]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        # feat's first channel is prepared for obstacles;
        self.feat = torch.ones(
            args.NUM_ENVIRONMENTS, 1, self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

        # Cache for skeleton waypoints extracted from the (local) traversable mask.
        # Key: env id in batch. Value: list[(x_px, y_px)] in local map pixel coordinates.
        self._last_skeleton_waypoints_local = {}

    def reset(self) -> None:
        self.curr_loc = None
        self.last_loc = None
        self.vis_classes = []
        self.feat = torch.ones(
            self.args.NUM_ENVIRONMENTS, 1, self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

        self._last_skeleton_waypoints_local = {}

        if self.visualize or self.print_images:
            self.vis_image = vu.init_vis_image()
            self.rgb_vis = None

    def _dynamic_process(self, num_detected_classes: int) -> None:
        vr = self.vision_range
        self.init_grid = torch.zeros(
            self.args.NUM_ENVIRONMENTS, 1 + num_detected_classes, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)

        if num_detected_classes > (self.feat.shape[1] - 1):
            pad_num = 1 + num_detected_classes - self.feat.shape[1]
            feat_pad = torch.ones(
                self.num_environments,
                pad_num,
                self.screen_h // self.du_scale * self.screen_w // self.du_scale
                ).float().to(self.device)
            self.feat = torch.cat([self.feat, feat_pad], axis=1)

        new_nc = num_detected_classes + self.MAP_CHANNELS
        if new_nc > self.local_map.shape[1]:
            pad_num = new_nc - self.local_map.shape[1]
            local_map_pad = torch.zeros(self.num_environments,
                                        pad_num,
                                        self.local_w,
                                        self.local_h).float().to(self.device)
            full_map_pad = torch.zeros(self.num_environments,
                                       pad_num,
                                       self.full_w,
                                       self.full_h).float().to(self.device)
            self.local_map = torch.cat([self.local_map, local_map_pad], axis=1)
            self.one_step_local_map = torch.cat([self.one_step_local_map, local_map_pad], axis=1)
            self.full_map = torch.cat([self.full_map, full_map_pad], axis=1)
            self.one_step_full_map = torch.cat([self.one_step_full_map, full_map_pad], axis=1)

    def get_full_map(self):
        return self.full_map.cpu().numpy()

    def _prepare(self, nc: int) -> None:
        r"""Create empty full_map, local_map, full_pose, local_pose, origins, local map boundries
        Args:
        nc: num channels
        """

        r"""
        Calculating full and local map sizes
        args.global_downscaling = 2
        full_w = full_h = 480
        local_w, local_h = 240
        """
        self.full_w, self.full_h = self.map_shape
        self.local_w = int(self.full_w / self.global_downscaling)
        self.local_h = int(self.full_h / self.global_downscaling)
        self.visited_vis = np.zeros(self.map_shape)

        r"""
        map_size_cm is the real world word map size(cm), i.e. (2400cm, 2400cm) <=> (24m, 24m)
        each element in the spatial map(full_map) corresponds to a cell of size (5cm, 5cm) in the physical world
        map_resolution = 5 so, the full_map should be (2400 / 5, 2400 / 5) = (480, 480)
        local_map is half of full_map, i.e. (240, 240)
        """
        self.full_map = torch.zeros(self.num_environments,
                                    nc,
                                    self.full_w,
                                    self.full_h).float().to(self.device)
        self.one_step_full_map = torch.zeros(self.num_environments,
                                             nc,
                                             self.full_w,
                                             self.full_h).float().to(self.device)
        self.local_map = torch.zeros(self.num_environments,
                                     nc,
                                     self.local_w,
                                     self.local_h).float().to(self.device)
        self.one_step_local_map = torch.zeros(self.num_environments,
                                              nc,
                                              self.local_w,
                                              self.local_h).float().to(self.device)

        r"""
        pose.shape=(3,): [x, y, orientation]
        full pose: the agent always starts at the center of the map facing east
        """
        self.full_pose = torch.zeros(self.num_environments, 3).float().to(self.device)
        self.local_pose = torch.zeros(self.num_environments, 3).float().to(self.device)
        self.curr_loc = torch.zeros(self.num_environments, 3).float().to(self.device)

        # Origin of local map
        self.origins = np.zeros((self.num_environments, 3))

        # Local Map Boundaries
        self.lmb = np.zeros((self.num_environments, 4)).astype(int)

        # state has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.state = np.zeros((self.num_environments, 7))

    def _get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc # represent agent's position
        local_w, local_h = local_sizes # (240, 240)
        full_w, full_h = full_sizes # (480, 480)

        if self.global_downscaling > 1: # True, since args.global_downscaling = 2
            # calculate local map boundaries in full_map: width: (gx1, gx2); height: (gy1, gy2)
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose(self, num_detected_classes: int):
        r"""
        1. Initialize full_map as all zeros
        2. Initialize agent at the middle of the map
        3. extract the local map from the full map
        """

        nc = num_detected_classes + self.MAP_CHANNELS
        self._prepare(nc)

        self.full_map.fill_(0.)
        self.one_step_full_map.fill_(0.)
        self.full_pose.fill_(0.) # [bs, 3]

        # map_size_cm = 2400
        # full_pos[0]: [x=12m, y=12m, ori=0], agent always start at the center of the map
        self.full_pose[:, :2] = self.args.MAP_SIZE_CM / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        self.state[:, :3] = locs # state: [x,y,z,gx1,gx2,gy1,gy2]
        for e in range(self.num_environments):
            r, c = locs[e, 1], locs[e, 0] # r,c = 12; r is x direction, c is y direction
            loc_r, loc_c = [int(r * 100.0 / self.resolution), # loc_r, loc_c = 12 * 100 / 5 = 240
                            int(c * 100.0 / self.resolution)]

            # current and past agent location: agent takes a (3,3) square in the middle of the (480, 480) map.
            # (3, 3) in spatial map <=> (15cm, 15cm) in physical world
            self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0
            self.one_step_full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0
            safe_radius = int(180.0 / self.resolution)
            Y, X = np.ogrid[:self.full_h, :self.full_w]
            dist_from_center = np.sqrt((Y - loc_r)**2 + (X - loc_c)**2)
            safe_mask = dist_from_center <= safe_radius

            # 强制将出生点周围的"已探索区域"(通道1) 设为 1.0
            # 障碍物(通道0)初始化时本身就是 0.0，所以这块地成了绝对安全的可通行区域
            # 这块安全记录会通过 torch.max 被永久保留，完美撑过 Step 0-12！
            self.full_map[e, 1, safe_mask] = 1.0
            self.one_step_full_map[e, 1, safe_mask] = 1.0
            # lmb: [gx1, gx2, gy1, gy2]
            self.lmb[e] = self._get_local_map_boundaries((loc_r, loc_c),
                                                (self.local_w, self.local_h),
                                                (self.full_w, self.full_h))
            self.state[e, 3:] = self.lmb[e]

            # the origin of the local map is the top-lef corner of local map [6,6,0] meter
            self.origins[e] = [self.lmb[e][2] * self.resolution / 100.0,
                            self.lmb[e][0] * self.resolution / 100.0, 0.]

        for e in range(self.num_environments):
            # extract the local map
            self.local_map[e] = self.full_map[e, :, self.lmb[e, 0] : self.lmb[e, 1], self.lmb[e, 2] : self.lmb[e, 3]]
            self.one_step_local_map[e] = self.one_step_full_map[e, :, self.lmb[e, 0] : self.lmb[e, 1],
                                                                self.lmb[e, 2] : self.lmb[e, 3]]

            # local_pose initialized as (6,6,0) meter
            self.local_pose[e] = self.full_pose[e] - \
                torch.from_numpy(self.origins[e]).to(self.device).float()

            self.curr_loc[e] = self.full_pose[e] - \
                torch.from_numpy(self.origins[e]).to(self.device).float()

    def update_map(self, step: int, detected_classes: OrderedSet, current_episode_id: int) -> None:
        if step == 0:
            self.last_loc = self.state[:, :3]
        else:
            self.last_loc = self.curr_loc

        locs = self.local_pose.cpu().numpy()
        self.state[:, :3] = locs + self.origins
        self.curr_loc = self.state[:, :3]
        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        self.one_step_local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(self.num_environments):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.resolution),
                        int(c * 100.0 / self.resolution)]
            self.local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
            self.one_step_local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.

            self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
                    self.local_map[e]
            self.one_step_full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
                    self.one_step_local_map[e]

            self.full_pose[e] = self.local_pose[e] + \
                    torch.from_numpy(self.origins[e]).to(self.device).float()

        if ((step + 1) % self.args.CENTER_RESET_STEPS) == 0:
            for e in range(self.num_environments):
                self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
                    self.local_map[e]
                self.one_step_full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
                    self.one_step_local_map[e]

                # full_pose is actually global agent position.
                self.full_pose[e] = self.local_pose[e] + \
                    torch.from_numpy(self.origins[e]).to(self.device).float()
                locs = self.full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / self.resolution),
                                int(c * 100.0 / self.resolution)]
                self.lmb[e] = self._get_local_map_boundaries((loc_r, loc_c),
                                                  (self.local_w, self.local_h),
                                                  (self.full_w, self.full_h))
                self.state[e, 3:] = self.lmb[e]
                self.origins[e] = [self.lmb[e][2] * self.resolution / 100.0,
                              self.lmb[e][0] * self.resolution / 100.0, 0.]
                self.local_map[e] = self.full_map[e, :,
                                        self.lmb[e, 0]:self.lmb[e, 1],
                                        self.lmb[e, 2]:self.lmb[e, 3]]
                self.one_step_local_map[e] = self.one_step_full_map[e, :,
                                        self.lmb[e, 0]:self.lmb[e, 1],
                                        self.lmb[e, 2]:self.lmb[e, 3]]
                self.local_pose[e] = self.full_pose[e] - \
                    torch.from_numpy(self.origins[e]).to(self.device).float()

        return (self.full_map.cpu().numpy(),
                self.full_pose.cpu().numpy(),
                self.one_step_full_map.cpu().numpy())

    def forward(self, obs: torch.Tensor, pose_obs: torch.Tensor, step=-1):
        """
        Args:
            obs: (b, c, h, w), b = batch size, c = 3(RGB) + 1(Depth) + num_detected_categories
        """
        # if use CoCo the number of categories is 16(i.e. c=16), but now open-vocabulary;
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :] # depth.shape = (bs, H, W)

        # cut out the needed tensor from presupposed categories dimension
        num_detected_categories = c - 4 # 4=3(RGB) + 1(Depth)
        self._dynamic_process(num_detected_categories)

        # shape: [bs, h, w, 3] 3 is (x, y, z) for each point in (h, w)
        point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)

        agent_view_t = du.transform_camera_view_t(point_cloud_t, self.agent_height, 0, self.device)

        # point cloud in world axis
        # self.shift_loc=[250, 0, pi/2] => heading is always 90(degree), change with turn left
        # shape: [bs, h, w, 3] => (bs, 120, 160, 3)
        agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device)

        max_h = self.max_height # 72
        min_h = self.min_height # -8
        xy_resolution = self.resolution
        z_resolution = self.z_resolution

        # vision_range = 100(cm)
        # in sem_exp.py _preprocess_depth(), all invalid depth values are set as 100
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float() # (bs, x, y, 3) => (bs, 120, 160, 3)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] - vision_range // 2.) / vision_range * 2. # normalize to (-1, 1)
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] - (max_h + min_h) // 2.) / (max_h - min_h) * 2. # normalize
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3]) # [bs, 3, x*y]

        # obs: [b, c, h*w] => [b, 17, 19200], feat is a tensor contains all predicted semantic features
        pool = nn.AvgPool2d(self.du_scale)
        # obs[:, 4, ...] = 0.
        self.min_z = int(25 / z_resolution - min_h) # 25 / 5 - (-8) = 13
        # self.min_z = 2 # use grounded-sam to detect floor
        self.feat[:, 1:, :] = pool(obs[:, 4:, :, :]).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        # self.init_grid: [bs, categories + 1, x=vr, y=vr, z=(max_height - min_height)] => [bs, 17, 100, 100, 80]
        # feat: average of all categories's predicted semantic features, [bs, 17, 19200]
        # XYZ_cm_std: point cloud in physical world, [bs, 3, 19200]
        # splat_feat_nd:
        assert self.init_grid.shape[1] == self.feat.shape[1], "init_grid and feat should have same number of channels!"

        # shape: [bs, num_detected_classes + 1, 100, 100, 80]
        voxels = du.splat_feat_nd(self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)
        max_z = int((self.agent_height + 1) / z_resolution - min_h) # int((88 + 1) / 5 - (-8))= 25

        agent_height_proj = voxels[..., self.min_z:max_z].sum(4) # shape: [bs, num_detected_classes + 1, 100, 100]
        all_height_proj = voxels.sum(4) # shape: [bs, num_detected_classes + 1, 100, 100]

        fp_map_pred = agent_height_proj[:, :1, :, :] # obstacle map
        fp_exp_pred = all_height_proj[:, :1, :, :] # explored map
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = self.local_pose

        agent_view = torch.zeros(bs, self.local_map.shape[1],
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device) # (bs, c, 480, 480) => full_map

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred # obstacle map
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred # explored area

        sem_window = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0
        )
        agent_view[:, 4:, y1:y2, x1:x2] = sem_window

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):
            # pose: (bs, 3) -> x, y, ori(degree)
            # 57.29577951308232 = 180 / pi
            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(self.local_pose, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        # get rotation matrix and translation matrix according to new pose (x, y, theta(degree))
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True) # shape: [bs, c, 240, 240]
        maps2 = torch.cat((self.local_map.unsqueeze(1), translated.unsqueeze(1)), 1)
        one_step_maps2 = torch.cat((self.one_step_local_map.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)
        one_step_map_pred, _ = torch.max(one_step_maps2, 1)
        self.local_map = map_pred
        self.one_step_local_map = one_step_map_pred
        self.local_pose = current_poses

        # return fp_map_pred, map_pred, pose_pred, current_poses
