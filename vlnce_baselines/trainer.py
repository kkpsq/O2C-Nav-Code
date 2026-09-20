import copy
import gzip
import json
import os
from collections import defaultdict
from typing import List, Dict

import numpy as np
from PIL import Image
from fastdtw import fastdtw
from skimage.morphology import binary_closing
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm
import supervision as sv

from habitat import logger
from habitat_extensions.measures import NDTW
from habitat.core.simulator import Observations
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.environments import get_env_class
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry

# Import Habitat visualization utilities
from habitat.utils.visualizations import maps

from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.Policy import FusionMapPolicy
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import get_device
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.utils.constant import base_classes, map_channels
from .utils.depth_utils import get_world_xz_from_pixel
from .utils.visualization import O2CNavVisualizer
from vlnce_baselines.models.sam_point_extractor import SAMPointExtractor
from . import perception
from . import policy as policy_module
from .agent import VLMReasoningAgent

import warnings

warnings.filterwarnings('ignore')

@baseline_registry.register_trainer(name="o2cnav")
class O2CNav(BaseTrainer):
    def __init__(self, config, r2r) -> None:
        super().__init__()
        self.backtrack_steps = 0
        self.r2r = r2r
        self.device = get_device(config.TORCH_GPU_ID)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.config = config
        self.map_args = config.MAP
        self.resolution = config.MAP.MAP_RESOLUTION
        self.width = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        self.height = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
        self.max_step = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.map_shape = (config.MAP.MAP_SIZE_CM // self.resolution,
                          config.MAP.MAP_SIZE_CM // self.resolution)
        
        self.trans = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(
                                             (self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH),
                                             interpolation=Image.NEAREST)
                                         ])

        self.classes = []
        self.current_episode_id = None
        self.current_detections = None
        self.map_channels = map_channels
        self.floor = np.zeros(self.map_shape)
        self.one_step_floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversable = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)


        self.visualize = True
        self.save_dir = getattr(config, 'RGB_SAVE_DIR', './saved_rgb_images')
        self.visualizer = O2CNavVisualizer(None, self.visualize, self.save_dir, self.width, self.height)

        self.visited_targets = []  # List of targets the agent has identified/visited
        self.current_step = 0  # Track current step for navigation decisions

        # Distance thresholds for target management (in map units)
        self.target_reached_threshold = getattr(config, 'TARGET_REACHED_THRESHOLD', 15.0)

        # add yolo-world to detect common items
        self.agent = VLMReasoningAgent(self.visualizer)
        self.distance_threshold = 100.0 / self.resolution
        self.visit_threshold = 2

    def _set_eval_config(self) -> None:
        policy_module.set_eval_config(self)

    def _init_envs(self) -> None:
        # logger.info("start to initialize environments")

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
        )
        logger.info(f"local rank: {self.local_rank}, num of episodes: {self.envs.number_of_episodes}")
        self.detected_classes = OrderedSet()
        # logger.info("initializing environments finished!")

    def _collect_val_traj(self) -> None:
        if not self.r2r:
            role = self.config.TASK_CONFIG.DATASET.ROLES
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        if self.r2r:
            with gzip.open(self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split)) as f:
                gt_data = json.load(f)
        else:
            with gzip.open(self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split, role=role[0])) as f:
                gt_data = json.load(f)

        self.gt_data = gt_data

    def _calculate_metric(self, infos: List):
        curr_eps = self.envs.current_episodes()
        info = infos[0]
        ep_id = curr_eps[0].episode_id
        gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
        pred_path = np.array(info['position']['position'])
        distances = np.array(info['position']['distance'])
        gt_length = distances[0]
        dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
        metric = {}
        metric['steps_taken'] = info['steps_taken']
        metric['distance_to_goal'] = distances[-1]
        metric['success'] = 1. if distances[-1] <= 3. else 0.
        metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
        metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
        metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
        metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
        metric['sdtw'] = metric['ndtw'] * metric['success']
        self.state_eps[ep_id] = metric
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)
        logger.info(f'ep{ep_id}:{self.state_eps[ep_id]}')

    def _initialize_policy(self) -> None:
        policy_module.initialize_policy(self)

    def _concat_obs(self, obs: Observations) -> np.ndarray:
        return perception.concat_obs(obs)

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        state, self.current_point_list = perception.preprocess_state(
            state, self.config, self.map_args, self.trans,
            self.classes, self.sam_extractor, self.current_episode_id,
            self.current_step, self.visualize, self.mapping_module,
            self.detected_classes, self.height, self.width
        )
        return state

    def _get_sem_pred(self, rgb: np.ndarray):
        return perception.get_sem_pred(
            rgb, self.classes, self.sam_extractor, self.current_episode_id,
            self.current_step, self.visualize, self.mapping_module,
            self.detected_classes, self.height, self.width
        )

    def _process_labels(self, labels: List[str]) -> List:
        return perception.process_labels(labels, self.detected_classes)

    def _process_masks(self, masks: np.ndarray, labels: List[str]):
        return perception.process_masks(masks, labels, self.detected_classes, self.height, self.width)

    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        return perception.preprocess_depth(depth, min_depth, max_depth)

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        return perception.preprocess_obs(
            obs, self.config, self.map_args, self.trans,
            self.classes, self.sam_extractor, self.current_episode_id,
            self.current_step, self.visualize, self.mapping_module,
            self.detected_classes, self.height, self.width
        )

    def _batch_obs(self, n_obs: List[Observations]) -> Tensor:
        return perception.batch_obs(
            n_obs, self.device, self.config, self.map_args, self.trans,
            self.classes, self.sam_extractor, self.current_episode_id,
            self.current_step, self.visualize, self.mapping_module,
            self.detected_classes, self.height, self.width
        )

    def _process_classes(self, base_class: List, target_class: List) -> List:
        return perception.process_classes(base_class, target_class)


    def _process_one_step_floor(self, one_step_full_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        return perception.process_one_step_floor(one_step_full_map, self.detected_classes, kernel_size)

    def _process_map(self, step: int, full_map: np.ndarray, kernel_size: int = 3) -> tuple:
        return perception.process_map(step, full_map, self.detected_classes, kernel_size)

    def _maps_initialization(self):
        perception.maps_initialization(self)

    def _look_around(self):
        return perception.look_around(self)


    def reset(self) -> None:
        self.classes = []
        self.current_detections = None
        self.detected_classes = OrderedSet()
        self.floor = np.zeros(self.map_shape)
        self.one_step_floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversable = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)

        # Reset target tracking
        self.visited_targets = []
        self.current_step = 0
        self.backtrack_steps = 0

        self.current_point_list = []

        # 拓扑图
        self.topo_graph = []
        self.distance_threshold = 100.0 / self.resolution
        self.visit_threshold = 3

        self.policy.reset()
        self.mapping_module.reset()
        self.agent.reset()

    def _get_camera_intrinsics(self) -> np.ndarray:
        return perception.get_camera_intrinsics(self.config, self.width, self.height)

    def get_panorama(self, obs: Observations, step: int):
        return perception.get_panorama(self, obs, step)

    def rollout(self):
        """
        Execute a whole episode using bounding box target navigation
        """
        self._maps_initialization()
        look_around_results = self._look_around()
        if look_around_results[1] is None:
            logger.info("Episode finished during look_around. Exiting rollout.")
            if look_around_results[3]:  # infos
                self._calculate_metric(look_around_results[3])
            return

        full_pose, obs, dones, infos = look_around_results

        # logger.info('Sensor pose', obs[0]['sensor_pose'])

        # --- Initialize ---
        action_list = []
        going_to_stop = False
        panorama_got = False
        navigate_or_not = False
        collided = 0
        search_destination = False
        current_pose = full_pose[0] if full_pose is not None else None

        target_map_x, target_map_y = None, None

        max_steps_to_target = 30  # Renavigate after 30 steps
        target_set_step = None  # Record the steps after target set

        # Initial map status
        full_map = self.mapping_module.get_full_map()

        for step in range(12, self.max_step):
            # import sys
            # sys.stdout.flush()
            # logger.info(action_list, panorama_got)
            # =================================================================
            # 1. (ANALYZE STATE for step N)
            #
            # =================================================================
            if dones[0]:
                self._calculate_metric(infos)
                return
            self.visualizer.instruction = self.instruction
            self.visualizer.destination = self.destination
            self.visualizer._action = self._action

            logger.info(f"\nepisode:{self.current_episode_id}, step:{step}")

            # logger.info(f"instr: {self.instruction}")
            # logger.info(f"Targets visited: {len(self.visited_targets)}")

            last_pose = current_pose
            current_pose = full_pose[0]
            self.current_step = step
            self.visualizer.sync(step, self.current_episode_id)

            position = current_pose[:2] * 100 / self.resolution
            agent_map_x, agent_map_y = int(position[0]), int(position[1])
            # logger.info("full pose: ", current_pose)
            # save the clean_vis
            display_rgb = obs[0]['rgb']
            if hasattr(self.mapping_module, 'rgb_vis') and self.mapping_module.rgb_vis is not None:
                display_rgb = cv2.cvtColor(self.mapping_module.rgb_vis, cv2.COLOR_BGR2RGB)
            self.visualizer._save_rgb_frame(obs[0], step, self.visited_targets, self.current_episode_id, (target_map_x, target_map_y), custom_rgb=display_rgb)

            # =================================================================
            # 2. PLAN/DECIDE for step N
            #    Four steps：
            #    Step 1: When there's no target, turn around to get panorama
            #    Step 2: After getting the panorama, navigate_or_backtrack & query_llm to get target
            #    Step 3: Navigate to the target
            #    Step 4: If timeout or arrival, go back to step1
            # =================================================================

            if not action_list:
                # Timeout
                if target_map_x is not None and target_map_y is not None and target_set_step is not None:
                    steps_since_target_set = step - target_set_step
                    if steps_since_target_set >= max_steps_to_target:
                        # logger.info(f"Target timeout: {steps_since_target_set} steps since target set, exceeding {max_steps_to_target} limit.")

                        if len(self.visited_targets) > 0:
                            self.visited_targets.pop()

                        # Step 4: Reset
                        panorama_got = False
                        navigate_or_not = False
                        target_map_x, target_map_y = None, None
                        target_set_step = None
                        # logger.info("Reset navigation state due to timeout.")

                # Check if arrival
                if target_map_x is not None and target_map_y is not None:
                    distance_to_target = np.sqrt((target_map_x - agent_map_x) ** 2 + (target_map_y - agent_map_y) ** 2)
                    # logger.info(f"Agent: ({agent_map_x}, {agent_map_y}), Target: ({target_map_x}, {target_map_y})")
                    # logger.info(f"Distance to target: {distance_to_target:.2f} (threshold: {self.target_reached_threshold})")
                    if distance_to_target < self.target_reached_threshold:
                        # logger.info(f"Target reached! Distance: {distance_to_target:.2f}")

                        # Check arrival image
                        if len(self.visited_targets) > 0:
                            dist_calc = lambda target: np.sqrt(
                                (target['world_coords'][0] - self.visited_targets[-1]['world_coords'][0]) ** 2 + (
                                            target['world_coords'][1] - self.visited_targets[-1]['world_coords'][
                                        1]) ** 2) if 'world_coords' in target else float('inf')
                            for target in self.visited_targets[:-1]:
                                if dist_calc(target) < self.target_reached_threshold:
                                    # logger.info('Removed duplicate waypoint due to proximity. Distance: %f', dist_calc(target))
                                    self.visited_targets.pop()
                                    break

                        # Step 4: Reset status
                        panorama_got = False
                        navigate_or_not = False
                        target_map_x, target_map_y = None, None
                        target_set_step = None
                        # logger.info("Reset navigation state - target reached.")

                # Step 1: No target and haven't got the panorama -> turn around first
                if target_map_x is None and not panorama_got and going_to_stop:
                    # logger.info('Final stop.')
                    action_list.append(0)  # STOP action
                elif target_map_x is None and not navigate_or_not:
                    # logger.info("Step 1: Getting panorama and deciding navigation direction...")
                    current_rgb = obs[0]['rgb'].copy()

                    panorama_frames = self.get_panorama(obs[0], step)
                    if 'episode_finished' in panorama_frames:
                        break

                    # New waypoint record
                    waypoint_id = len(self.visited_targets)
                    self.visited_targets.append({
                        'step': step,
                        'init_image': Image.fromarray(current_rgb) if isinstance(current_rgb,
                                                                                 np.ndarray) else current_rgb,
                        'panorama_frames': panorama_frames,
                        'world_coords': (agent_map_x, agent_map_y)
                    })

                    # Refresh skeleton waypoint cache from the *post-panorama* mapping state,
                    # and overwrite the debug VLM map for this step so it matches the panorama overlays.
                    try:
                        vlm_map = self.mapping_module.create_vlm_map_from_state(
                            self.current_episode_id,
                            0,
                            None,
                            output_size=(1024, 1024),
                            visited_targets=self.visited_targets,
                            display_last=True,
                        ).copy()
                        if hasattr(self, 'visualizer') and self.visualizer.save_dir:
                            ep_folder = os.path.join(self.visualizer.save_dir, str(self.current_episode_id))
                            os.makedirs(ep_folder, exist_ok=True)
                            cv2.imwrite(os.path.join(ep_folder, f"debug_vlm_map_step{step:04d}.png"), vlm_map)
                            step_folder = os.path.join(ep_folder, f"step_{int(step):04d}")
                            os.makedirs(step_folder, exist_ok=True)
                            cv2.imwrite(os.path.join(step_folder, "debug_vlm_map.png"), vlm_map)
                    except Exception as e:
                        logger.info(f"Error refreshing/saving debug VLM map after panorama: {e}")
                        
                    history_waypoints_local = []
                    if self.mapping_module is not None and hasattr(self.mapping_module, 'state'):
                        # mapping_module.state 格式: [start_x, start_y, start_o, gx1, gx2, gy1, gy2]
                        # gx1 和 gy1 是当前局部地图在全局地图上的左上角行列偏移量
                        gx1 = int(self.mapping_module.state[0, 3])
                        gy1 = int(self.mapping_module.state[0, 5])
                        
                        # 遍历历史坐标 (排除刚 append 进去的当前位置 [:-1])
                        for target in self.visited_targets[:-1]:
                            if 'world_coords' in target:
                                map_x, map_y = target['world_coords']
                                # 转换为局部地图像素：X(列) = map_x - gy1; Y(行) = map_y - gx1
                                history_waypoints_local.append((map_x - gy1, map_y - gx1))

                    # Batch overlay skeleton waypoints onto the 4 directional views.
                    # Batch overlay skeleton waypoints onto the 4 directional views.
                    # Store as 'rgb_wp' while keeping raw 'rgb' for SAM/other processing.
                    # Use the refreshed cache (refresh_if_missing=False) to stay consistent with the debug map.
                    try:
                        if self.mapping_module is not None and hasattr(self.mapping_module, "overlay_skeleton_waypoints_on_rgbs"):
                            view_names = ["forward", "left", "behind", "right"]
                            views_rgb = {vn: panorama_frames[i].get('rgb', None) for i, vn in enumerate(view_names) if i < len(panorama_frames)}
                            views_depth = {vn: panorama_frames[i].get('depth', None) for i, vn in enumerate(view_names) if i < len(panorama_frames)}

                            views_rgb_wp, views_wp_pixels = self.mapping_module.overlay_skeleton_waypoints_on_rgbs(
                                views_rgb,
                                id=0,
                                depth_images=views_depth,
                                depth_in_meters=False,
                                depth_min_m=0.1,
                                depth_max_m=5.0,
                                refresh_if_missing=False,
                                draw_indices=True,
                                index_color_bgr=(255, 255, 255),
                                index_scale=0.8,
                                index_thickness=2,
                                return_pixels=True,
                                min_forward_m=0.2,
                                occlusion_margin_m=0.2,
                                occlusion_kernel=1,
                                history_waypoints_local=history_waypoints_local,
                                merge_threshold_px=30,
                            )
                            for i, vn in enumerate(view_names):
                                if i < len(panorama_frames) and isinstance(views_rgb_wp, dict) and vn in views_rgb_wp:
                                    panorama_frames[i]['rgb_wp'] = views_rgb_wp[vn]
                                    if isinstance(views_wp_pixels, dict) and vn in views_wp_pixels:
                                        panorama_frames[i]['wp_pixels'] = views_wp_pixels[vn]
                    except Exception as e:
                        logger.info(f"Error overlaying waypoints on panorama views: {e}")


                    self.visualizer._save_waypoint_panorama_rgb(panorama_frames, waypoint_id, step)

                    current_pos = np.array([agent_map_x, agent_map_y])
                    merged = False
                    loop_warning_msg = ''

                    for node in getattr(self, 'topo_graph', []):
                        dist = np.linalg.norm(current_pos - np.array(node['center']))
                        if dist < self.distance_threshold:
                            node['visits'] += 1
                            merged = True
                            if node['visits'] > self.visit_threshold:
                                loop_warning_msg = (
                                    f"CRITICAL WARNING: You have made decisions in this exact area {node['visits']} times! "
                                    f"You are stuck in a navigation loop. You MUST select a completely new direction, "
                                    f"or if you confirm you are near the destination, approach the position described in the instruction and STOP."
                                )
                                logger.info(f"[Topo-Graph Triggered] Loop detected at {node['center']}! Visits: {node['visits']}")
                            break                        
                    if not merged:
                        if not hasattr(self, 'topo_graph'):
                            self.topo_graph = []
                        self.topo_graph.append({
                            'center': (agent_map_x, agent_map_y),
                            'visits': 1
                        })

                    decision = self.agent.navigate_or_backtrack(
                        instruction=self.instruction,
                        visited_targets=self.visited_targets,
                        episode_id=self.current_episode_id,
                        sam_extractor=self.sam_extractor,
                        classes=self.classes,
                        use_sam_points_for_la=False,
                        include_dino_objects_for_la=True,
                        loop_warning=loop_warning_msg
                    )
                    # logger.info(f"Navigation decision: {decision}")
                    if decision.get('action') == 'STOP':
                        self.visited_targets[-1].update({
                            'progress_analysis': decision.get('progress_analysis', ''),
                            'reasoning': decision.get('reasoning', ''),
                            'direction_decision': 'stop'
                        })
                        going_to_stop = True
                        panorama_got = False
                        continue  # 直接进入下一步去执行停止动作

                    if decision.get('action', 'NAVIGATE') == 'BACKTRACK':
                        target_waypoint_id = decision.get('waypoint', 0)
                        if isinstance(target_waypoint_id, int) and target_waypoint_id < len(self.visited_targets) - 1:
                            target_map_x, target_map_y = self.visited_targets[target_waypoint_id]['world_coords']
                            self.visited_targets.pop()  # remove unfinished waypoint
                            # (f"Backtracking to waypoint {target_waypoint_id} at ({target_map_x}, {target_map_y})")
                            self.visited_targets = self.visited_targets[:target_waypoint_id + 1]
                        else:
                            # logger.info("Invalid waypoint ID for backtrack, continuing with navigation")
                            decision['action'] = 'NAVIGATE'
                        panorama_got = True
                    if decision.get('action', 'NAVIGATE') == 'NAVIGATE':
                        navigate_or_not = True
                        direction = decision.get('direction', 'forward')
                        progress_analysis = decision.get('progress_analysis', '')
                        reasoning = decision.get('reasoning', '')

                        # Save decision
                        self.visited_targets[-1].update({
                            'progress_analysis': progress_analysis,
                            'reasoning': reasoning,
                            'direction_decision': direction,
                            'target_pixel': decision.get('target_pixel')   # <-- record the target pixel for later visualization
                        })

                        # Get corresponding image
                        direction_map = {'forward': 0, 'left': 90, 'behind': 180, 'right': 270}
                        target_angle = direction_map.get(direction, 0)
                        frame_idx = target_angle // 90

                        if frame_idx < len(panorama_frames):
                            dir_rgb = panorama_frames[frame_idx].get('rgb_wp', panorama_frames[frame_idx].get('rgb'))
                            self.visited_targets[-1]['dir_image'] = Image.fromarray(dir_rgb) if isinstance(dir_rgb,
                                                                                                           np.ndarray) else dir_rgb
                            self.visited_targets[-1]['turn_action'] = f"turn {direction}"

                        # Add action according to LA
                        if direction == 'left':
                            action_list.extend([2] * 3)  # left 3*30
                        elif direction == 'right':
                            action_list.extend([3] * 3)  # right 3*30
                        elif direction == 'behind':
                            action_list.extend([2] * 6)  # left 6*30

                        panorama_got = True
                        # logger.info(f"Step 1 completed: Direction decision = {direction}, added turn actions")

                # Step 2: After getting to the right direction, query_llm to get target position
                elif target_map_x is None and panorama_got and not action_list:
                    # logger.info("Step 2: Querying LLM for specific target...")

                    target_pixel = self.visited_targets[-1].get('target_pixel')
                    if target_pixel is not None:
                        coords = target_pixel
                    else:
                        coords = (self.width // 2, int(self.height * 0.75))

                    depth_image = self._preprocess_depth(obs[0]['depth'], 0.1, 5.0) / 100.0
                    # Convert pixel to map position
                    # Note that we don't want the target position untraversible, thus we reduce depth and make the target closer to the agent if so
                    while True:
                        target = get_world_xz_from_pixel(
                            pixel_coords=coords,
                            depth_image=depth_image,
                            full_pose=current_pose,
                            camera_intrinsics=self._get_camera_intrinsics(),
                        )
                        new_target_x = int(target[0] * 100.0 / self.resolution)
                        new_target_y = int(target[1] * 100.0 / self.resolution)
                        new_target_x = max(0, min(new_target_x, self.map_shape[0] - 1))
                        new_target_y = max(0, min(new_target_y, self.map_shape[1] - 1))

                        if self.traversable[new_target_y, new_target_x] == 1 or depth_image.max() < 0.1:
                            target_map_x, target_map_y = new_target_x, new_target_y
                            target_set_step = step
                            # logger.info(f"Target set at map coordinates: ({target_map_x}, {target_map_y}) at step {step}")

                            waypoint = np.array([target_map_y, target_map_x])
                            navigation_action = self.policy._get_action(
                                current_pose, waypoint, full_map[0], self.traversable,
                                self.collision_map, step, self.current_episode_id,
                                self.detected_classes, search_destination
                            )
                            action_list.append(navigation_action)
                            # logger.info(f"Added initial navigation action: {navigation_action}")
                            break
                        depth_image = depth_image - 0.1

                    panorama_got = False  # Reset
                    # logger.info("Step 2 completed: Target acquired from LLM")

                # Step 3: Have target -> navigate
                elif target_map_x is not None and target_map_y is not None and not action_list:
                    # logger.info(f"Step 3: Continuing navigation to target ({target_map_x}, {target_map_y})")
                    waypoint = np.array([target_map_y, target_map_x])
                    navigation_action = self.policy._get_action(
                        current_pose, waypoint, full_map[0], self.traversable,
                        self.collision_map, step, self.current_episode_id,
                        self.detected_classes, search_destination
                    )
                    action_list.append(navigation_action)
                    # logger.info(f"Added navigation action: {navigation_action}")


            if action_list:
                # =================================================================
                # 3. ACT for step N
                #
                # =================================================================
                self._action = action_list[0]
                action_list.pop(0)
                actions = [{"action": self._action}]

                # logger.info(f'Action actually performed: {self._action}')

                outputs = self.envs.step(actions)

                # =================================================================
                # 4. UPDATE for step N+1
                #
                # =================================================================
                obs, _, dones, infos = [list(x) for x in zip(*outputs)]
                # logger.info('Sensor pose', obs[0]['sensor_pose'])

                if not dones[0]:
                    batch_obs = self._batch_obs(obs)
                    poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
                    self.mapping_module(batch_obs, poses, self.current_step)
                    full_map, full_pose, one_step_full_map = \
                        self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
                    self.mapping_module.one_step_full_map.fill_(0.)
                    self.mapping_module.one_step_local_map.fill_(0.)

                    self.traversable, self.floor, self.frontiers = self._process_map(step, full_map[0])
                    self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])

                    last_pose = current_pose
                    current_pose = full_pose[0]
                    if last_pose is not None and current_pose is not None:
                        displacement = calculate_displacement(last_pose, current_pose, self.resolution)
                        if displacement < 0.2 * 100 / self.resolution:
                            collided += 1
                        else:
                            collided = 0
                            replan = False
                        if collided >= 30:
                            fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                                 f"r{self.local_rank}_w{self.world_size}_collision_stuck.txt")
                            with open(fname, "a") as f:
                                f.writelines(
                                    f"id: {str(self.current_episode_id)}; step: {str(step)}; collided: {str(collided)}\n")

                    current_action = self._action
                    if last_pose is not None and current_action is not None and current_action == 1:
                        collision_map = collision_check_fmm(last_pose, current_pose, self.resolution,
                                                            self.mapping_module.map_shape)
                        self.collision_map = np.logical_or(self.collision_map, collision_map)
                    self.traversable[self.collision_map == 1] = 0
                else:
                    self._calculate_metric(infos)
                    return
            else:
                pass
        self._calculate_metric(infos)

    def eval(self):
        self._set_eval_config()
        self._init_envs()
        self._collect_val_traj()
        self._initialize_policy()
        self.agent.reset()


        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))

        self.state_eps = {}
        t1 = time.time()
        for i in tqdm(range(eps_to_eval)):
            self.rollout()
            self.reset()

        self.envs.close()

        logger.info("=== FINAL MODEL USAGE STATISTICS ===")
        final_stats = self.agent.model.print_usage_stats()

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)

        stats_fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                   f"model_usage_stats_{split}_r{self.local_rank}_w{self.world_size}.json")
        with open(stats_fname, "w") as f:
            json.dump(final_stats, f, indent=2)
        logger.info(f"Model usage statistics saved to: {stats_fname}")

        success_eps = [ep_id for ep_id, metrics in self.state_eps.items() if metrics.get('success', 0.) == 1.0]
        failed_eps = [ep_id for ep_id, metrics in self.state_eps.items() if metrics.get('success', 0.) == 0.0]

        summary_data = {
            "total_episodes": len(self.state_eps),
            "success_count": len(success_eps),
            "failed_count": len(failed_eps),
            "success_episodes": success_eps,
            "failed_episodes": failed_eps
        }

        summary_fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                     f"success_fail_eps_{split}_r{self.local_rank}_w{self.world_size}.json")
        with open(summary_fname, "w") as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Success/Fail episode IDs saved to: {summary_fname}")
        
        t2 = time.time()
        logger.info(f"time: {t2 - t1}s")
        logger.info("test time: %d", t2 - t1)


def merge_model_usage_stats(stats_dir, split="val_unseen"):
    import glob
    import json

    pattern = os.path.join(stats_dir, f"model_usage_stats_{split}_r*_w*.json")
    stat_files = glob.glob(pattern)

    if not stat_files:
        print(f"No model usage stat files found in {stats_dir} with pattern {pattern}")
        return

    merged_stats = {
        'la': {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        },
        'total_calls': 0,
        'total_tokens': 0,
        'num_processes': 0,
        'process_stats': []
    }

    for stat_file in stat_files:
        try:
            with open(stat_file, 'r') as f:
                stats = json.load(f)

            merged_stats['la']['calls'] += stats['la']['calls']
            merged_stats['la']['input_tokens'] += stats['la']['input_tokens']
            merged_stats['la']['output_tokens'] += stats['la']['output_tokens']
            merged_stats['la']['total_tokens'] += stats['la']['total_tokens']

            merged_stats['total_calls'] += stats['total_calls']
            merged_stats['total_tokens'] += stats['total_tokens']
            merged_stats['num_processes'] += 1

            process_info = {
                'file': os.path.basename(stat_file),
                'stats': stats
            }
            merged_stats['process_stats'].append(process_info)

            print(f"Loaded stats from: {stat_file}")

        except Exception as e:
            print(f"Error loading {stat_file}: {e}")

    merged_file = os.path.join(stats_dir, f"merged_model_usage_stats_{split}.json")
    with open(merged_file, 'w') as f:
        json.dump(merged_stats, f, indent=2)

    print("=== MERGED MODEL USAGE STATISTICS ===")
    print(f"Number of processes: {merged_stats['num_processes']}")
    print(f"Language Action Model:")
    print(f"  - Total calls: {merged_stats['la']['calls']:,}")
    print(f"  - Total input tokens: {merged_stats['la']['input_tokens']:,}")
    print(f"  - Total output tokens: {merged_stats['la']['output_tokens']:,}")
    print(f"  - Total tokens: {merged_stats['la']['total_tokens']:,}")
    print(f"OVERALL TOTAL:")
    print(f"  - Total calls: {merged_stats['total_calls']:,}")
    print(f"  - Total tokens: {merged_stats['total_tokens']:,}")
    print(f"Merged statistics saved to: {merged_file}")
    print("=====================================")

    return merged_stats
