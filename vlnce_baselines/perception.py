"""Perception module: panorama, depth processing, semantic prediction, mapping."""
from collections import defaultdict
from typing import List

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import binary_closing, disk, remove_small_objects
import torch

from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from vlnce_baselines.utils.map_utils import process_navigable_classes
from vlnce_baselines.utils.constant import map_channels


def concat_obs(obs: Observations) -> np.ndarray:
    rgb = obs['rgb'].astype(np.uint8)
    depth = obs['depth']
    state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)  # (h, w, c)->(c, h, w)

    return state


def preprocess_depth(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    # Preprocesses a depth map by handling missing values, removing outliers, and scaling the depth values.
    # logger.info('max:',depth.max())
    depth = depth[:, :, 0] * 1

    for i in range(depth.shape[1]):
        depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

    mask2 = depth > 0.99  # turn too far pixels to invalid
    depth[mask2] = 0.

    mask1 = depth == 0
    depth[mask1] = 1.0  # then turn all invalid pixels to vision_range(100)
    depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0

    return depth


def process_classes(base_class: List, target_class: List) -> List:
    for item in target_class:
        if item in base_class:
            base_class.remove(item)
    base_class.extend(target_class)

    return base_class


def process_labels(labels: List[str], detected_classes) -> List:
    class_names = []
    for label in labels:
        class_name = " ".join(label.split(' ')[:-1])
        class_names.append(class_name)
        detected_classes.add(class_name)

    return class_names


def process_masks(masks: np.ndarray, labels: List[str], detected_classes,
                  height: int, width: int):
    """Since we are now handling the open-vocabulary semantic mapping problem,
    we need to maintain a mask tensor with dynamic channels. The idea is to combine
    all same class tensors into one tensor, then let the "detected_classes" to
    record all classes without duplication. Finally we can use each class's index
    in the detected_classes to determine as it's channel in the mask tensor.
    The organization of mask is similar to chaplot's Sem_Exp, please refer to this link:
    https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/agents/utils/semantic_prediction.py#L41

    Args:
        masks (np.ndarray): shape:(c,h,w), each instance(even the same class) has one channel
        labels (List[str]): masks' corresponding labels. len(masks) = len(labels)

    Returns:
        final_masks (np.ndarray): each mask will find their channel in detected_classes.
        len(final_masks) = len(detected_classes)
    """
    if masks.shape[0] > 0:  # Check if there are any masks
        same_label_indexs = defaultdict(list)
        for idx, item in enumerate(labels):
            same_label_indexs[item].append(idx)  # dict {class name: [idx]}
        combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
        for i, indexs in enumerate(same_label_indexs.values()):
            combined_mask[i] = np.sum(masks[indexs, ...], axis=0)

        idx = [detected_classes.index(label) for label in same_label_indexs.keys()]

        """
        max_idx = max(idx) + 1, attention: remember to add one becaure index start from 0
        init final masks as [max_idx + 1, h, w]; add not_a_category channel at last
        """
        final_masks = np.zeros((len(detected_classes), *masks.shape[1:]))
        final_masks[idx, ...] = combined_mask
    else:
        final_masks = np.zeros((len(detected_classes), height, width))

    return final_masks


def get_sem_pred(rgb: np.ndarray, classes, sam_extractor, current_episode_id,
                 current_step, visualize, mapping_module, detected_classes,
                 height, width):
    """
    mask.shape=[num_detected_classes, h, w]
    labels looks like: ["kitchen counter 0.69", "floor 0.37"]
    """
    cls2 = classes.copy()
    masks, labels, clean_vis, point_info_list = sam_extractor.process(rgb, cls2)
    if visualize:
        cv2.imwrite(f'saved_rgb_images/{current_episode_id}/step{current_step}_mask.png',
                    clean_vis)
    mapping_module.rgb_vis = clean_vis
    assert len(masks) == len(labels), f"The number of masks not equal to the number of labels!"

    class_names = process_labels(labels, detected_classes)
    masks = process_masks(masks, class_names, detected_classes, height, width)

    return masks.transpose(1, 2, 0), point_info_list


def preprocess_state(state: np.ndarray, config, map_args, trans_fn,
                     classes, sam_extractor, current_episode_id,
                     current_step, visualize, mapping_module,
                     detected_classes, height, width):
    state = state.transpose(1, 2, 0)
    rgb = state[:, :, :3].astype(np.uint8)  # [3, h, w]
    rgb = rgb[:, :, ::-1]  # RGB to BGR
    depth = state[:, :, 3:4]  # [1, h, w]
    min_depth = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
    max_depth = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
    env_frame_width = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH

    sem_seg_pred, current_point_list = get_sem_pred(
        rgb, classes, sam_extractor, current_episode_id,
        current_step, visualize, mapping_module,
        detected_classes, height, width)  # [num_detected_classes, h, w]
    depth = preprocess_depth(depth, min_depth, max_depth)  # [1, h, w]

    """
    ds: Downscaling factor
    args.env_frame_width = 640, args.frame_width = 160
    """
    ds = env_frame_width // map_args.FRAME_WIDTH  # ds = 4
    if ds != 1:
        rgb = np.asarray(trans_fn(rgb.astype(np.uint8)))  # resize
        depth = depth[ds // 2::ds, ds // 2::ds]  # down scaling start from 2, step=4
        sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

    depth = np.expand_dims(depth, axis=2)  # recover depth.shape to (height, width, 1)
    state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)  # (4+num_detected_classes, h, w)

    return state, current_point_list


def preprocess_obs(obs: Observations, config, map_args, trans_fn,
                   classes, sam_extractor, current_episode_id,
                   current_step, visualize, mapping_module,
                   detected_classes, height, width) -> np.ndarray:
    concated_obs = concat_obs(obs)
    state, current_point_list = preprocess_state(
        concated_obs, config, map_args, trans_fn,
        classes, sam_extractor, current_episode_id,
        current_step, visualize, mapping_module,
        detected_classes, height, width)

    return state  # state.shape=(c,h,w)


def batch_obs(n_obs: List[Observations], device, config, map_args, trans_fn,
              classes, sam_extractor, current_episode_id,
              current_step, visualize, mapping_module,
              detected_classes, height, width) -> torch.Tensor:
    n_states = [preprocess_obs(
        obs, config, map_args, trans_fn,
        classes, sam_extractor, current_episode_id,
        current_step, visualize, mapping_module,
        detected_classes, height, width) for obs in n_obs]
    max_channels = max([len(state) for state in n_states])
    batch = np.stack([np.pad(state,
                             [(0, max_channels - state.shape[0]),
                              (0, 0),
                              (0, 0)],
                             mode='constant')
                      for state in n_states], axis=0)

    return torch.from_numpy(batch).to(device)


def process_one_step_floor(one_step_full_map: np.ndarray, detected_classes,
                           kernel_size: int = 3) -> np.ndarray:
    navigable_index = process_navigable_classes(detected_classes)
    not_navigable_index = [i for i in range(len(detected_classes)) if i not in navigable_index]
    # logger.info(f'{navigable_index}, {not_navigable_index}')
    one_step_full_map = remove_small_objects(one_step_full_map.astype(bool), min_size=64)

    obstacles = one_step_full_map[0, ...].astype(bool)
    explored_area = one_step_full_map[1, ...].astype(bool)

    objects = np.sum(one_step_full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)
    navigable = np.logical_or.reduce(one_step_full_map[map_channels:, ...][navigable_index])
    # stairs should remain navigable even if overlapped with objects
    # navigable = np.logical_or(navigable, stairs_mask)
    navigable = np.logical_and(navigable, np.logical_not(objects))

    free_mask = 1 - np.logical_or(obstacles, objects)
    free_mask = np.logical_or(free_mask, navigable)
    # free_mask = np.logical_or(free_mask, stairs_mask)
    floor = explored_area * free_mask
    floor = remove_small_objects(floor, min_size=400).astype(bool)
    floor = binary_closing(floor, footprint=disk(kernel_size))

    return floor


def process_map(step: int, full_map: np.ndarray, detected_classes,
                kernel_size: int = 3) -> tuple:
    navigable_index = process_navigable_classes(detected_classes)
    not_navigable_index = [i for i in range(len(detected_classes)) if i not in navigable_index]
    full_map = remove_small_objects(full_map.astype(bool), min_size=64)

    obstacles = full_map[0, ...].astype(bool)
    explored_area = full_map[1, ...].astype(bool)

    objects = np.sum(full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)

    selem = disk(3)
    obstacles_closed = binary_closing(obstacles, footprint=selem)
    objects_closed = binary_closing(objects, footprint=selem)
    navigable = np.logical_or.reduce(full_map[map_channels:, ...][navigable_index])
    # stairs should remain navigable even if overlapped with objects
    # navigable = np.logical_or(navigable, stairs_mask)
    navigable = np.logical_and(navigable, np.logical_not(objects))
    navigable_closed = binary_closing(navigable, footprint=selem)

    untraversable = np.logical_or(objects_closed, obstacles_closed)
    # ensure stairs override untraversable
    untraversable[navigable_closed == 1] = 0
    # untraversable[stairs_mask == 1] = 0
    untraversable = remove_small_objects(untraversable, min_size=64)
    untraversable = binary_closing(untraversable, footprint=disk(3))
    traversable = np.logical_not(untraversable)

    free_mask = 1 - np.logical_or(obstacles, objects)
    free_mask = np.logical_or(free_mask, navigable)
    # free_mask = np.logical_or(free_mask, stairs_mask)
    floor = explored_area * free_mask
    floor = remove_small_objects(floor, min_size=400).astype(bool)
    floor = binary_closing(floor, footprint=selem)
    traversable = np.logical_or(floor, traversable)

    explored_area = binary_closing(explored_area, footprint=selem)
    contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = np.zeros(full_map.shape[-2:], dtype=np.uint8)
    image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)
    frontiers = np.logical_and(floor, image)
    frontiers = remove_small_objects(frontiers.astype(bool), min_size=64)

    return traversable, floor, frontiers.astype(np.uint8)


def get_camera_intrinsics(config, width: int, height: int) -> np.ndarray:
    """Get camera intrinsics matrix for depth projection"""
    hfov = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
    vfov = 2 * np.arctan(height / width * np.tan(hfov / 2))

    fx = width / (2 * np.tan(np.deg2rad(hfov / 2)))
    fy = height / (2 * np.tan(np.deg2rad(vfov / 2)))
    cx = width / 2
    cy = height / 2

    intrinsics = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
    return intrinsics


def maps_initialization(trainer):
    obs = trainer.envs.reset()  # type(obs): list
    trainer.instruction = obs[0]['instruction']['text']
    trainer.destination = "goal"
    trainer.classes = trainer.base_classes.copy()
    trainer.current_episode_id = trainer.envs.current_episodes()[0].episode_id

    trainer.mapping_module.init_map_and_pose(num_detected_classes=len(trainer.detected_classes))
    batch_obs_t = trainer._batch_obs(obs)
    poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(trainer.device)
    trainer.mapping_module(batch_obs_t, poses, trainer.current_step)
    full_map, full_pose, _ = trainer.mapping_module.update_map(0, trainer.detected_classes, trainer.current_episode_id)
    trainer.mapping_module.one_step_full_map.fill_(0.)
    trainer.mapping_module.one_step_local_map.fill_(0.)

    # Save after map update so RGB and top-down correspond to the same frame.
    trainer.visualizer._save_rgb_frame(obs[0], 0, None, trainer.current_episode_id)


def look_around(trainer):
    full_pose, obs, dones, infos = None, None, None, None
    for step in range(0, 12):
        trainer._action = HabitatSimActions.TURN_LEFT
        actions = []
        for _ in range(trainer.config.NUM_ENVIRONMENTS):
            actions.append({"action": HabitatSimActions.TURN_LEFT})
        outputs = trainer.envs.step(actions)
        obs, _, dones, infos = [list(x) for x in zip(*outputs)]
        trainer.current_step = step
        if dones[0]:
            return full_pose, obs, dones, infos

        batch_obs_t = trainer._batch_obs(obs)
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(trainer.device)
        trainer.mapping_module(batch_obs_t, poses, trainer.current_step)
        full_map, full_pose, one_step_full_map = \
            trainer.mapping_module.update_map(step, trainer.detected_classes, trainer.current_episode_id)
        trainer.mapping_module.one_step_full_map.fill_(0.)
        trainer.mapping_module.one_step_local_map.fill_(0.)
        trainer.traversable, trainer.floor, trainer.frontiers = trainer._process_map(step, full_map[0])
        trainer.one_step_floor = trainer._process_one_step_floor(one_step_full_map[0])

        # Save after map update so RGB and top-down correspond to the same frame.
        trainer.visualizer._save_rgb_frame(obs[0], step, trainer.visited_targets, trainer.current_episode_id)

    return full_pose, obs, dones, infos


def get_panorama(trainer, obs: Observations, step: int):
    """
    Turn around (4 turns x 90°) to get panorama
    """

    panorama_frames = []

    for turn_step in range(1, 4 + 1):
        # Turn 90° left (3 x 30° TURN_LEFT), accumulate pose delta
        accumulated_delta = np.zeros(3)
        for _ in range(3):
            turn_action = [{"action": HabitatSimActions.TURN_LEFT}]
            turn_outputs = trainer.envs.step(turn_action)
            turn_obs, _, turn_dones, turn_infos = [list(x) for x in zip(*turn_outputs)]
            if turn_dones[0]:
                return {'turn_direction': 'episode_done', 'episode_finished': True}
            accumulated_delta += np.array(turn_obs[0]['sensor_pose'])

        panorama_frames.append({
            'rgb': turn_obs[0]['rgb'].copy(),
            'depth': turn_obs[0].get('depth', None).copy() if turn_obs[0].get('depth', None) is not None else None,
            'angle': turn_step * 90 % 360,
            'step': turn_step
        })

        # Update map with the accumulated 90° pose delta
        turn_obs[0]['sensor_pose'] = accumulated_delta
        batch_obs_t = trainer._batch_obs(turn_obs)
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in turn_obs])).float().to(trainer.device)
        trainer.mapping_module(batch_obs_t, poses, trainer.current_step)
        trainer.mapping_module.update_map(step + turn_step, trainer.detected_classes, trainer.current_episode_id)
        trainer.mapping_module.one_step_full_map.fill_(0.)
        trainer.mapping_module.one_step_local_map.fill_(0.)
    panorama_frames = [panorama_frames[-1]] + panorama_frames[:-1]

    return panorama_frames
