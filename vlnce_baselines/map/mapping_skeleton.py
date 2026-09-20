"""Skeleton extraction and waypoint sampling for Semantic_Mapping."""

import numpy as np

from skimage.morphology import skeletonize, disk, binary_closing, binary_erosion
from scipy.ndimage import convolve

import habitat_extensions.pose_utils as pu


class SkeletonMixin:
    """骨架提取、航点采样、深度转换方法。"""

    def _get_skeleton_waypoints(self, mask: np.ndarray, min_distance_px: int = 10):
        """
        基于可通行区域生成骨架图，并提取交叉点和端点作为 Waypoints
        mask: 2D numpy array (0为障碍，1为自由空间)
        """
        # 1. 提取骨架 (Skeletonization)
        skeleton = skeletonize(mask > 0)

        # 2. 识别骨架上的关键节点（交叉点和端点）
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])

        # 计算邻居数
        neighbors = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)

        # 提取节点：endpoints(端点), junctions(交叉点)
        endpoints = (skeleton & (neighbors == 1))
        junctions = (skeleton & (neighbors > 2))

        keypoints_mask = endpoints | junctions
        y_coords, x_coords = np.where(keypoints_mask)

        waypoints = []
        # 3. 简单的非极大值抑制（NMS），防止节点过于密集
        for x, y in zip(x_coords, y_coords):
            if not waypoints:
                waypoints.append((x, y))
            else:
                dists = [np.sqrt((x - wx)**2 + (y - wy)**2) for wx, wy in waypoints]
                if min(dists) >= min_distance_px:
                    waypoints.append((x, y))

        return waypoints, skeleton

    def _merge_close_waypoints_px(self, points_xy: list, merge_eps_px: int) -> list:
        """合并距离很近的航点（像素坐标系，(x,y)）。"""
        if not points_xy:
            return []

        clusters = []  # each: [sum_x, sum_y, count]
        for x, y in points_xy:
            merged = False
            for c in clusters:
                cx = c[0] / c[2]
                cy = c[1] / c[2]
                if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= merge_eps_px * merge_eps_px:
                    c[0] += x
                    c[1] += y
                    c[2] += 1
                    merged = True
                    break
            if not merged:
                clusters.append([float(x), float(y), 1.0])

        merged_points = [(int(round(c[0] / c[2])), int(round(c[1] / c[2]))) for c in clusters]
        return merged_points

    def _sample_waypoints_from_skeleton(
        self,
        skeleton: np.ndarray,
        safe_zone: np.ndarray,
        agent_xy: tuple,
        resolution_cm: float,
        prefer_band_m: tuple = (1.0, 3.0),
        fallback_band_m: tuple = (3.0, 5.0),
        skeleton_clip_m: float = 3.0,
        max_skeleton_m: float = 5.0,
        sample_interval_m: float = 0.5,
        merge_eps_m: float = 0.4,
    ) -> tuple:
        """从骨架图中生成更密集的航点：端点/分叉点 + 直线(骨架)上均匀采样，并做近点合并。

        返回:
            (waypoints_xy, skeleton_clip_used)
            - waypoints_xy: list[(x_px, y_px)]，局部地图像素坐标
            - skeleton_clip_used: bool，是否使用了 skeleton_clip_m 的裁剪（若裁剪后无点会回退）
        """
        if skeleton is None or skeleton.size == 0:
            return [], True

        ax, ay = int(agent_xy[0]), int(agent_xy[1])
        res_m = float(resolution_cm) / 100.0
        interval_px = max(2, int(round(sample_interval_m / res_m)))
        merge_eps_px = max(2, int(round(merge_eps_m / res_m)))

        # --- 1) skeleton clip: 以机器人为中心裁剪到 skeleton_clip_m ---
        H, W = skeleton.shape
        Y, X = np.ogrid[:H, :W]
        dist_px = np.sqrt((X - ax) ** 2 + (Y - ay) ** 2)
        clip_px = int(round(skeleton_clip_m / res_m))
        max_px = int(round(max_skeleton_m / res_m))
        clip_mask = dist_px <= clip_px
        max_mask = dist_px <= max_px

        skeleton_bool = (skeleton > 0)
        skeleton_clip = skeleton_bool & clip_mask
        skeleton_full = skeleton_bool & max_mask

        def collect_points(skeleton_mask: np.ndarray) -> list:
            # --- 端点/分叉点（分叉点用邻居数>2）---
            kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]])
            neighbors = convolve(skeleton_mask.astype(int), kernel, mode='constant', cval=0)
            endpoints = (skeleton_mask & (neighbors == 1))
            junctions = (skeleton_mask & (neighbors > 2))

            key_y, key_x = np.where(endpoints | junctions)
            keypoints = list(zip(key_x.tolist(), key_y.tolist()))

            # --- 直线(骨架)上均匀采样：在所有骨架像素上做简单 NMS ---
            sk_y, sk_x = np.where(skeleton_mask)
            if sk_x.size > 0:
                d = (sk_x - ax) ** 2 + (sk_y - ay) ** 2
                order = np.argsort(d)
                sk_points = list(zip(sk_x[order].tolist(), sk_y[order].tolist()))
            else:
                sk_points = []

            selected = []
            for x, y in keypoints:
                if safe_zone is not None and safe_zone.shape == skeleton.shape:
                    if not bool(safe_zone[y, x]):
                        continue
                selected.append((x, y))

            for x, y in sk_points:
                if safe_zone is not None and safe_zone.shape == skeleton.shape:
                    if not bool(safe_zone[y, x]):
                        continue
                if not selected:
                    selected.append((x, y))
                    continue
                too_close = False
                for sx, sy in selected:
                    if (x - sx) * (x - sx) + (y - sy) * (y - sy) < interval_px * interval_px:
                        too_close = True
                        break
                if not too_close:
                    selected.append((x, y))

            return self._merge_close_waypoints_px(selected, merge_eps_px=merge_eps_px)

        # --- 2) 优先用 3m 裁剪骨架挑 2–3m；若为空，用 5m 骨架挑 3–5m ---
        clipped = np.count_nonzero(skeleton_clip) > 0
        merged_clip = collect_points(skeleton_clip) if clipped else []
        merged_full = None  # lazy

        # --- 3) 距离带筛选：优先 2–3m，若没有则回退 3–5m ---
        def in_band(points, band_m):
            lo, hi = band_m
            lo_px = lo / res_m
            hi_px = hi / res_m
            out = []
            for x, y in points:
                dp = np.sqrt((x - ax) ** 2 + (y - ay) ** 2)
                if lo_px <= dp <= hi_px:
                    out.append((x, y))
            return out

        prefer_pts = in_band(merged_clip, prefer_band_m) if merged_clip else []
        if prefer_pts:
            return prefer_pts, clipped

        if merged_full is None:
            merged_full = collect_points(skeleton_full) if np.count_nonzero(skeleton_full) > 0 else []
        fallback_pts = in_band(merged_full, fallback_band_m) if merged_full else []
        return fallback_pts, clipped

    def _depth_norm_to_meters(
        self,
        depth_image: np.ndarray,
        min_depth_m: float = 0.1,
        max_depth_m: float = 5.0,
    ) -> np.ndarray:
        """将 Habitat 的归一化深度(约 [0,1])转换为米。

        约定：0 可能表示缺失；>0.99 视为过远无效。
        """
        if depth_image is None:
            return None

        depth = np.array(depth_image)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        depth = depth.astype(np.float32, copy=False)

        # Replace zeros with per-column max (Habitat sometimes uses 0 for missing).
        if np.any(depth == 0.0):
            col_max = np.max(depth, axis=0, keepdims=True)
            depth = np.where(depth == 0.0, col_max, depth)

        # Too-far pixels become invalid.
        depth = np.where(depth > 0.99, 0.0, depth)

        # Turn invalid pixels to 1.0 (max range after scaling), matching common preprocess.
        depth = np.where(depth == 0.0, 1.0, depth)

        depth_m = float(min_depth_m) + depth * float(max_depth_m - min_depth_m)
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        return depth_m

    def _extract_skeleton_inputs_from_local_state(self, id: int = 0) -> tuple:
        """从当前 local_map/state 中构造骨架航点所需的 (traversable_mask, safe_zone, agent_xy)。

        返回:
            (traversable_mask, safe_zone, agent_xy)
            - traversable_mask/safe_zone: 2D bool ndarray, shape=(H,W)
            - agent_xy: (x_px, y_px) in local map pixel coordinates
        """
        if self.local_map is None or self.state is None:
            return None, None, None

        local_maps = self.local_map.clone()
        obstacle_map = local_maps[id, 0, ...].cpu().numpy()
        explored_map = local_maps[id, 1, ...].cpu().numpy()

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = self.state[id]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.resolution - gx1), int(c * 100.0 / self.resolution - gy1)]
        start = pu.threshold_poses(start, obstacle_map.shape)
        agent_xy = (int(start[1]), int(start[0]))

        # 与 create_vlm_map_from_state 一致的 not_cat 逻辑
        local_maps[:, -1, ...] = 1e-5
        semantic_map = local_maps[id, 4:, ...].argmax(0).cpu().numpy()
        semantic_map += 5
        not_cat_id = local_maps.shape[1]
        not_cat_mask = (semantic_map == not_cat_id)

        obstacle_map_mask = np.rint(obstacle_map) == 1
        explored_map_mask = np.rint(explored_map) == 1
        m_free = np.logical_and(not_cat_mask, explored_map_mask)
        m_obstacle = np.logical_and(not_cat_mask, obstacle_map_mask)

        traversable_raw = np.logical_and(m_free, ~m_obstacle)
        traversable_mask = binary_closing(traversable_raw, footprint=disk(7))
        safe_zone = binary_erosion(traversable_raw, footprint=disk(2))
        return traversable_mask, safe_zone, agent_xy

    def _refresh_skeleton_waypoints_local(
        self,
        id: int,
        traversable_mask: np.ndarray,
        safe_zone: np.ndarray,
        agent_xy: tuple,
        prefer_band_m: tuple = (2.0, 3.0),
        fallback_band_m: tuple = (3.0, 5.0),
        skeleton_clip_m: float = 3.0,
        max_skeleton_m: float = 5.0,
        sample_interval_m: float = 0.5,
        merge_eps_m: float = 0.4,
        cache: bool = True,
    ) -> tuple:
        """基于 traversable_mask 刷新/生成骨架航点，并写入缓存。

        Returns:
            (valid_waypoints, skeleton_vis_mask)
            - valid_waypoints: list[(x_px,y_px)] in local map pixel coordinates
            - skeleton_vis_mask: bool ndarray or None (用于调试可视化骨架)
        """
        if traversable_mask is None or traversable_mask.size == 0:
            if cache:
                self._last_skeleton_waypoints_local[id] = []
            return [], None

        # 骨架（用于 waypoint 采样）
        _, skeleton = self._get_skeleton_waypoints(traversable_mask, min_distance_px=2)
        skeleton_bool = (skeleton > 0) if skeleton.dtype != np.bool_ else skeleton

        skeleton_vis = skeleton_bool
        if safe_zone is not None and safe_zone.shape == skeleton_bool.shape:
            skeleton_vis = skeleton_vis & safe_zone

        valid_waypoints, _ = self._sample_waypoints_from_skeleton(
            skeleton=skeleton_bool.astype(np.uint8),
            safe_zone=safe_zone,
            agent_xy=agent_xy,
            resolution_cm=float(self.resolution),
            prefer_band_m=prefer_band_m,
            fallback_band_m=fallback_band_m,
            skeleton_clip_m=skeleton_clip_m,
            max_skeleton_m=max_skeleton_m,
            sample_interval_m=sample_interval_m,
            merge_eps_m=merge_eps_m,
        )

        if cache:
            self._last_skeleton_waypoints_local[id] = valid_waypoints
        return valid_waypoints, skeleton_vis
