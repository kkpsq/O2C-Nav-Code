"""VLM map visualization for Semantic_Mapping."""

import cv2
import numpy as np
from torch import Tensor

from skimage.morphology import disk, binary_closing, binary_erosion

import habitat_extensions.pose_utils as pu
import vlnce_baselines.utils.visualization as vu


class VizMixin:
    """VLM专用地图可视化方法。"""

    def create_vlm_map_from_state(
        self,
        current_episode_id: int,  # 保留以兼容接口, 但在此函数中未使用
        id: int = 0,
        goal: Tensor = None,
        output_size: tuple = (1024, 1024),
        visited_targets: list = None,
        display_last: bool = False
    ) -> np.ndarray:
        """
        根据当前的内部状态(self.local_map, self.state等)，创建一个坐标精确、
        专为VLM设计的、高对比度、信息简化的2D地图图像。

        Args:
            id (int): 要可视化的环境批次索引。
            goal (Tensor, optional): 目标的全局地图坐标。
            output_size (tuple, optional): 输出图像的尺寸。
            visited_targets (list, optional): 历史target位置列表。

        Returns:
            np.ndarray: BGR格式的地图图像。
        """
        if not hasattr(self, 'last_loc') or self.last_loc is None: return self.vis_image

        # 定义高对比度颜色 (BGR格式)
        COLOR_BG = (255, 255, 255)       # 白色 - 未探索
        COLOR_EXPLORED = (220, 220, 220)  # 浅灰色 - 已探索
        COLOR_OBSTACLE = (0, 0, 0)        # 黑色 - 障碍物
        COLOR_PATH = (0, 0, 255)          # 红色 - 轨迹
        COLOR_AGENT = (0, 0, 255)         # 红色 - 智能体
        COLOR_GOAL = (0, 255, 0)          # 亮绿色 - 目标

        # --- 1. 从 self 状态中提取数据 ---
        local_maps = self.local_map.clone()
        obstacle_map = local_maps[id, 0, ...].cpu().numpy()
        explored_map = local_maps[id, 1, ...].cpu().numpy()

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = self.state[id]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.resolution - gx1),
                 int(c * 100.0 / self.resolution - gy1)]
        start = pu.threshold_poses(start, obstacle_map.shape)

        if self.last_loc is not None and len(self.last_loc) > id and self.last_loc[id] is not None:
            last_start_x, last_start_y = self.last_loc[id][0], self.last_loc[id][1]
            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0 / self.resolution - gx1),
                            int(c * 100.0 / self.resolution - gy1)]
            last_start = pu.threshold_poses(last_start, obstacle_map.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = vu.draw_line(last_start, start, self.visited_vis[gx1:gx2, gy1:gy2])

        # 获取轨迹图的局部切片
        trajectory_map_local = self.visited_vis[gx1:gx2, gy1:gy2]

        # --- 2. 创建基础画布并绘制地图结构 ---
        h, w = obstacle_map.shape
        vis_map = np.full((h, w, 3), COLOR_BG, dtype=np.uint8)

        # 加入 not_cat_id 背景判断
        local_maps[:, -1, ...] = 1e-5
        semantic_map = local_maps[id, 4:, ...].argmax(0).cpu().numpy()
        semantic_map += 5
        not_cat_id = local_maps.shape[1]
        not_cat_mask = (semantic_map == not_cat_id)

        obstacle_map_mask = np.rint(obstacle_map) == 1
        explored_map_mask = np.rint(explored_map) == 1

        # 仅当像素为背景(not_cat)时，才依据探索/障碍进行着色
        m_free = np.logical_and(not_cat_mask, explored_map_mask)
        m_obstacle = np.logical_and(not_cat_mask, obstacle_map_mask)

        # 绘制已探索区域和障碍物（带 not_cat 约束）
        vis_map[explored_map_mask] = COLOR_EXPLORED
        vis_map[obstacle_map_mask] = COLOR_OBSTACLE

        # 绘制轨迹
        traversable_raw = np.logical_and(m_free, ~m_obstacle)

        # 闭运算连通掩码（保证走廊骨架连通）
        selem_for_skeleton = disk(7)
        traversable_mask = binary_closing(traversable_raw, footprint=selem_for_skeleton)

        # 安全区域掩码 (Safe Zone)
        selem_safe = disk(2)
        safe_zone = binary_erosion(traversable_raw, footprint=selem_safe)

        try:
            agent_xy = (int(start[1]), int(start[0]))
            valid_waypoints, skeleton_vis = self._refresh_skeleton_waypoints_local(
                id=id,
                traversable_mask=traversable_mask,
                safe_zone=safe_zone,
                agent_xy=agent_xy,
                prefer_band_m=(1.0, 3.0),
                fallback_band_m=(3.0, 5.0),
                skeleton_clip_m=3.0,
                max_skeleton_m=5.0,
                sample_interval_m=0.5,
                merge_eps_m=0.4,
                cache=True,
            )

            if skeleton_vis is not None:
                vis_map[skeleton_vis] = (255, 0, 255)

            for wx, wy in valid_waypoints:
                cv2.circle(vis_map, (int(wx), int(wy)), 3, (0, 255, 255), -1)

        except Exception as e:
            print(f"Skeleton generation skipped/failed: {e}")

        # --- 3. 在翻转前绘制智能体和目标 ---
        goal_local_pos_px = None
        if goal is not None:
            goal_coords = goal.cpu().numpy()
            if len(goal_coords) >= 2:
                goal_map_x, goal_map_y = int(goal_coords[1]), int(goal_coords[0])
                local_goal_x = goal_map_x - gx1
                local_goal_y = goal_map_y - gy1

                if (0 <= local_goal_x < h and 0 <= local_goal_y < w):
                    goal_local_pos_px = (local_goal_y, local_goal_x)

        agent_local_y = int(start_y * 100.0 / self.resolution - gx1)
        agent_local_x = int(start_x * 100.0 / self.resolution - gy1)

        if not vis_map.flags['C_CONTIGUOUS']:
            vis_map = np.ascontiguousarray(vis_map)

        if goal_local_pos_px:
            gx, gy = goal_local_pos_px
            cv2.drawMarker(vis_map, (gx, gy), COLOR_GOAL,
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=15, thickness=3)

        ax, ay = agent_local_x, agent_local_y
        angle_corrected_deg = -start_o
        angle_rad = np.deg2rad(angle_corrected_deg)

        arrow_length = 15
        p_center = (ax, ay)
        p_tip = (int(p_center[0] + arrow_length * np.cos(angle_rad)),
                int(p_center[1] - arrow_length * np.sin(angle_rad)))
        p_left = (int(p_center[0] + (arrow_length/2) * np.cos(angle_rad + np.pi*5/6)),
                int(p_center[1] - (arrow_length/2) * np.sin(angle_rad + np.pi*5/6)))
        p_right = (int(p_center[0] + (arrow_length/2) * np.cos(angle_rad - np.pi*5/6)),
                int(p_center[1] - (arrow_length/2) * np.sin(angle_rad - np.pi*5/6)))

        arrow_points = np.array([[p_tip, p_left, p_right]], dtype=np.int32)
        cv2.fillPoly(vis_map, arrow_points, COLOR_AGENT)
        cv2.circle(vis_map, p_center, 8, (0, 0, 255), -1)

        if visited_targets is not None and len(visited_targets) > 0:
            COLOR_VISITED_TARGET = (255, 0, 0)
            COLOR_TEXT_BG = (255, 255, 255)
            COLOR_TEXT = (0, 0, 0)

            _ = visited_targets

            if not display_last: _ = _[:-1]

            for i, target in enumerate(_):
                world_coords = target.get('world_coords')
                target_name = target.get('description', f'Target{i+1}')

                if world_coords is not None and len(world_coords) >= 2:
                    try:
                        target_world_x, target_world_z = world_coords[1], world_coords[0]

                        target_local_x = int(target_world_x - gx1)
                        target_local_y = int(target_world_z - gy1)

                        if 0 <= target_local_x < w and 0 <= target_local_y < h:
                            cv2.circle(vis_map, (target_local_y, target_local_x), 6, COLOR_VISITED_TARGET, -1)

                            if not hasattr(self, '_target_annotations'):
                                self._target_annotations = []
                            self._target_annotations.append({
                                'id': i + 1,
                                'name': target_name,
                                'local_pos': (target_local_y, target_local_x),
                                'original_size': (h, w)
                            })
                        else:
                            pass
                    except Exception as e:
                        pass
                else:
                    pass

        # --- 4. 最后进行垂直翻转以匹配视觉朝向 ---
        vis_map = np.flipud(vis_map)

        # --- 5. 放大到目标尺寸以保证清晰度 ---
        if output_size != (h, w):
            scale_x = output_size[0] / w
            scale_y = output_size[1] / h
            vis_map = cv2.resize(vis_map, output_size, interpolation=cv2.INTER_NEAREST)
            self._target_annotations = []

        return vis_map
