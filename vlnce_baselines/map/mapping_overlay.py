"""RGB overlay and waypoint projection for Semantic_Mapping."""

import cv2
import numpy as np
import torch


class OverlayMixin:
    """RGB投影、路点绘制、多视图叠加方法。"""

    def overlay_skeleton_waypoints_on_rgb(
        self,
        rgb_image: np.ndarray,
        id: int = 0,
        waypoints_local: list = None,
        refresh_if_missing: bool = True,
        depth_image: np.ndarray = None,
        depth_in_meters: bool = False,
        depth_min_m: float = 0.1,
        depth_max_m: float = 5.0,
        draw_indices: bool = False,
        index_color_bgr: tuple = (0, 255, 0),
        index_scale: float = 0.8,
        index_thickness: int = 2,
        index_offset_px: tuple = (6, -6),
        return_pixels: bool = False,
        history_waypoints_local: list = None,
        merge_threshold_px: int = 15,
        **draw_kwargs,
    ) -> np.ndarray:
        """输入当前视角 RGB，返回"叠加骨架航点"的 RGB（便于主程序直接调用）。

        Args:
            rgb_image: RGB 图像，np.ndarray (H,W,3)。支持 uint8 或 [0,1] float。
            id: batch 环境索引。
            waypoints_local: 可选，直接提供 local-map 航点 (x_px,y_px)；不提供则使用缓存。
            refresh_if_missing: 若缓存为空则从当前 local_map/state 刷新一次。
            depth_image: 可选，同帧深度图(归一化或米)，用于遮挡过滤。
            depth_in_meters: depth_image 是否已经是米。
            depth_min_m/depth_max_m: 若 depth 为归一化时用于缩放到米。
            history_waypoints_local: 历史航点列表，用于合并过滤。
            merge_threshold_px: 合并距离阈值（像素），骨架航点与历史航点距离小于此值时被过滤。
            draw_kwargs: 传给 draw_local_waypoints_on_rgb，例如 color_bgr/radius/thickness/occlusion_*。

        Returns:
            RGB uint8 ndarray。
        """
        if rgb_image is None:
            return None

        rgb = rgb_image
        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.array(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return rgb_image

        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating):
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        if waypoints_local is None:
            waypoints_local = self._last_skeleton_waypoints_local.get(id, [])
            if (not waypoints_local) and refresh_if_missing:
                traversable_mask, safe_zone, agent_xy = self._extract_skeleton_inputs_from_local_state(id=id)
                if traversable_mask is not None and agent_xy is not None:
                    waypoints_local, _ = self._refresh_skeleton_waypoints_local(
                        id=id,
                        traversable_mask=traversable_mask,
                        safe_zone=safe_zone,
                        agent_xy=agent_xy,
                        cache=True,
                    )

        if not waypoints_local and not history_waypoints_local:
            return rgb

        depth_m = None
        if depth_image is not None:
            depth_arr = depth_image
            if torch.is_tensor(depth_arr):
                depth_arr = depth_arr.detach().cpu().numpy()
            depth_arr = np.array(depth_arr)
            if depth_in_meters:
                depth_m = depth_arr.astype(np.float32, copy=False)
                if depth_m.ndim == 3 and depth_m.shape[2] == 1:
                    depth_m = depth_m[:, :, 0]
            else:
                depth_m = self._depth_norm_to_meters(depth_arr, min_depth_m=depth_min_m, max_depth_m=depth_max_m)

        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if depth_m is not None and depth_m.shape[:2] != (rgb_bgr.shape[0], rgb_bgr.shape[1]):
            depth_m = cv2.resize(depth_m, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        H, W = rgb_bgr.shape[:2]
        project_kwargs = {k: v for k, v in draw_kwargs.items()
                          if k in ['min_forward_m', 'max_forward_m', 'camera_yaw_deg',
                                   'occlusion_margin_m', 'occlusion_kernel']}

        history_rgb_pixels = []
        if history_waypoints_local is not None and len(history_waypoints_local) > 0:
            history_rgb_pixels = self.project_local_waypoints_to_rgb_pixels(
                history_waypoints_local,
                id=id,
                image_size=(H, W),
                depth_image_m=depth_m,
                **project_kwargs
            )

        filtered_waypoints_local = waypoints_local
        if waypoints_local and history_rgb_pixels:
            skeleton_rgb_pixels = self.project_local_waypoints_to_rgb_pixels(
                waypoints_local,
                id=id,
                image_size=(H, W),
                depth_image_m=depth_m,
                **project_kwargs
            )
            filtered_indices = []
            for idx, (u, v) in enumerate(skeleton_rgb_pixels):
                is_close_to_history = False
                for hu, hv in history_rgb_pixels:
                    dist = np.sqrt((u - hu) ** 2 + (v - hv) ** 2)
                    if dist < merge_threshold_px:
                        is_close_to_history = True
                        break
                if not is_close_to_history:
                    filtered_indices.append(idx)
            if filtered_indices:
                filtered_waypoints_local = [waypoints_local[i] for i in filtered_indices]
            else:
                filtered_waypoints_local = []

        skeleton_color_bgr = (255, 255, 0)
        history_color_bgr = (0, 0, 255)

        skeleton_draw_kwargs = {k: v for k, v in draw_kwargs.items()
                                if k not in ['color_bgr', 'radius', 'thickness', 'start_index']}

        drawn = self.draw_local_waypoints_on_rgb(
            rgb_bgr,
            filtered_waypoints_local,
            id=id,
            depth_image_m=depth_m,
            color_bgr=skeleton_color_bgr,
            radius=5,
            thickness=-1,
            draw_indices=draw_indices,
            index_color_bgr=index_color_bgr,
            index_scale=index_scale,
            index_thickness=index_thickness,
            index_offset_px=index_offset_px,
            return_pixels=return_pixels,
            **skeleton_draw_kwargs,
        )
        if history_waypoints_local is not None and len(history_waypoints_local) > 0:
            img_to_draw = drawn[0] if return_pixels else drawn
            pixel_map_skeleton = drawn[1] if return_pixels else {}

            skeleton_count = len(pixel_map_skeleton) if return_pixels else len(
                self.project_local_waypoints_to_rgb_pixels(
                    filtered_waypoints_local,
                    id=id,
                    image_size=(H, W),
                    depth_image_m=depth_m,
                    **project_kwargs
                )
            )

            history_draw_kwargs = {k: v for k, v in draw_kwargs.items()
                                   if k not in ['color_bgr', 'radius', 'thickness', 'draw_indices', 'return_pixels', 'start_index']}

            img_to_draw = self.draw_local_waypoints_on_rgb(
                img_to_draw,
                history_waypoints_local,
                id=id,
                depth_image_m=depth_m,
                color_bgr=history_color_bgr,
                radius=6,
                thickness=-1,
                draw_indices=draw_indices,
                index_color_bgr=(255, 255, 255),
                index_scale=index_scale,
                index_thickness=index_thickness,
                index_offset_px=index_offset_px,
                return_pixels=False,
                start_index=skeleton_count,
                **history_draw_kwargs
            )

            if return_pixels:
                drawn = (img_to_draw, pixel_map_skeleton)
            else:
                drawn = img_to_draw

        if return_pixels:
            rgb_bgr_out, pixel_map = drawn
            return cv2.cvtColor(rgb_bgr_out, cv2.COLOR_BGR2RGB), pixel_map

        return cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)

    def overlay_skeleton_waypoints_on_rgbs(
        self,
        rgb_images,
        id: int = 0,
        waypoints_local: list = None,
        refresh_if_missing: bool = True,
        depth_images=None,
        depth_in_meters: bool = False,
        depth_min_m: float = 0.1,
        depth_max_m: float = 5.0,
        camera_yaw_degs=None,
        view_order: tuple = ("forward", "left", "behind", "right"),
        return_pixels: bool = False,
        history_waypoints_local: list = None,
        **draw_kwargs,
    ):
        """批量给多视图 RGB 叠加同一批骨架航点（用于每步把 4 视角一起送给 LA）。

        支持两种输入形式:
        - dict: {"forward": rgb, "left": rgb, ...}
        - list/tuple: [rgb_forward, rgb_left, rgb_behind, rgb_right]（顺序由 view_order 约定）

        Args:
            rgb_images: dict 或 list/tuple。
            depth_images: 可选，dict 或 list/tuple，结构与 rgb_images 对齐。
            camera_yaw_degs: 可选，dict 或 list/tuple。
            view_order: 当输入为 list/tuple 时使用的视角顺序。
            其余参数含义与 overlay_skeleton_waypoints_on_rgb 一致。

        Returns:
            与输入同结构的 RGB (uint8)；dict 输入返回 dict，list 输入返回 list。
        """
        if rgb_images is None:
            return None

        default_yaw = {
            "forward": 0.0,
            "left": -90.0,
            "behind": 180.0,
            "right": 90.0,
            "left_front": -60.0,
            "left_behind": -120.0,
            "right_behind": 120.0,
            "right_front": 60.0,
            "left_30": -30.0,
            "left_60": -60.0,
            "left_90": -90.0,
            "left_120": -120.0,
            "left_150": -150.0,
            "right_30": 30.0,
            "right_60": 60.0,
            "right_90": 90.0,
            "right_120": 120.0,
            "right_150": 150.0,
        }

        # Resolve waypoint cache once, then reuse for all views.
        if waypoints_local is None:
            waypoints_local = self._last_skeleton_waypoints_local.get(id, [])
            if (not waypoints_local) and refresh_if_missing:
                traversable_mask, safe_zone, agent_xy = self._extract_skeleton_inputs_from_local_state(id=id)
                if traversable_mask is not None and agent_xy is not None:
                    waypoints_local, _ = self._refresh_skeleton_waypoints_local(
                        id=id,
                        traversable_mask=traversable_mask,
                        safe_zone=safe_zone,
                        agent_xy=agent_xy,
                        cache=True,
                    )

        if isinstance(rgb_images, dict):
            out = {}
            out_pixels = {} if return_pixels else None
            for k, rgb in rgb_images.items():
                depth_k = depth_images.get(k) if isinstance(depth_images, dict) else None
                if isinstance(camera_yaw_degs, dict):
                    yaw_k = float(camera_yaw_degs.get(k, default_yaw.get(k, 0.0)))
                elif camera_yaw_degs is None:
                    yaw_k = float(default_yaw.get(k, 0.0))
                else:
                    yaw_k = float(default_yaw.get(k, 0.0))

                per_view_kwargs = dict(draw_kwargs)
                per_view_kwargs["camera_yaw_deg"] = yaw_k
                res = self.overlay_skeleton_waypoints_on_rgb(
                    rgb,
                    id=id,
                    waypoints_local=waypoints_local,
                    refresh_if_missing=False,
                    depth_image=depth_k,
                    depth_in_meters=depth_in_meters,
                    depth_min_m=depth_min_m,
                    depth_max_m=depth_max_m,
                    return_pixels=return_pixels,
                    **per_view_kwargs,
                    history_waypoints_local=history_waypoints_local,
                )
                if return_pixels:
                    out[k], out_pixels[k] = res
                else:
                    out[k] = res
            return (out, out_pixels) if return_pixels else out

        # list/tuple path (support arbitrary N)
        rgbs = list(rgb_images)
        n_views = len(rgbs)

        depths = list(depth_images) if isinstance(depth_images, (list, tuple)) else []
        if len(depths) < n_views:
            depths = depths + [None] * (n_views - len(depths))
        else:
            depths = depths[:n_views]

        if isinstance(camera_yaw_degs, (list, tuple)):
            yaws = [float(v) for v in list(camera_yaw_degs)]
            if len(yaws) < n_views:
                yaws = yaws + [0.0] * (n_views - len(yaws))
            else:
                yaws = yaws[:n_views]
        else:
            if view_order is None:
                view_order_use = []
            else:
                view_order_use = list(view_order)
            if len(view_order_use) < n_views:
                view_order_use = view_order_use + ["forward"] * (n_views - len(view_order_use))
            else:
                view_order_use = view_order_use[:n_views]
            yaws = [float(default_yaw.get(name, 0.0)) for name in view_order_use]

        out_list = []
        out_pixels_list = [] if return_pixels else None
        for i in range(n_views):
            rgb = rgbs[i]
            depth_k = depths[i]
            yaw_k = yaws[i]
            per_view_kwargs = dict(draw_kwargs)
            per_view_kwargs["camera_yaw_deg"] = float(yaw_k)
            res = self.overlay_skeleton_waypoints_on_rgb(
                rgb,
                id=id,
                waypoints_local=waypoints_local,
                refresh_if_missing=False,
                depth_image=depth_k,
                depth_in_meters=depth_in_meters,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
                return_pixels=return_pixels,
                **per_view_kwargs,
                history_waypoints_local=history_waypoints_local,
            )
            if return_pixels:
                img_k, pixels_k = res
                out_list.append(img_k)
                out_pixels_list.append(pixels_k)
            else:
                out_list.append(res)
        return (out_list, out_pixels_list) if return_pixels else out_list

    def project_local_waypoints_to_rgb_pixels(
        self,
        waypoints_local: list,
        id: int = 0,
        state: np.ndarray = None,
        image_size: tuple = None,
        min_forward_m: float = 0.2,
        max_forward_m: float = None,
        camera_yaw_deg: float = 0.0,
        depth_image_m: np.ndarray = None,
        occlusion_margin_m: float = 0.2,
        occlusion_kernel: int = 1,
        return_ids: bool = False,
    ) -> list:
        """将"局部地图像素坐标系"的路点投影到当前RGB图像像素坐标系。

        坐标约定（与 `depth_utils.get_point_cloud_from_z_t` 一致）:
        - 相机坐标: X 向右, Y 向前(入屏), Z 向上
        - 深度图的深度值对应相机坐标的 Y

        Args:
            waypoints_local: list[(x_px, y_px)]，来自 local map 的像素坐标(列, 行)。
            id: batch 内环境索引。
            state: 可选，传入 `self.state[id]` 覆盖当前状态。
            image_size: (H, W)。默认使用 args.FRAME_HEIGHT/FRAME_WIDTH。
            min_forward_m: 过滤掉相机后方/过近的点。
            max_forward_m: 可选，过滤过远的点。
            camera_yaw_deg: 相机相对 agent 前向的 yaw 偏置（度）。

        Returns:
            list[(u_px, v_px)]：可落在图像内的像素坐标。
        """
        if waypoints_local is None or len(waypoints_local) == 0:
            return []

        if state is None:
            state = self.state[id]

        if image_size is None:
            H, W = int(self.screen_h), int(self.screen_w)
        else:
            H, W = int(image_size[0]), int(image_size[1])

        # state: [start_x, start_y, start_o, gx1, gx2, gy1, gy2]
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = state
        gx1, gy1 = int(gx1), int(gy1)

        # Map resolution: cm -> meters
        res_m = float(self.resolution) / 100.0

        # Camera intrinsics
        xc = (W - 1.0) / 2.0
        zc = (H - 1.0) / 2.0
        f = (W / 2.0) / np.tan(np.deg2rad(float(self.fov) / 2.0))

        # Camera height (meters)
        cam_h_m = float(self.agent_height) / 100.0

        # Precompute rotation for world->agent
        yaw = np.deg2rad(float(start_o))
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))

        # Camera yaw offset
        cam_yaw = np.deg2rad(float(camera_yaw_deg))
        cos_cam = float(np.cos(cam_yaw))
        sin_cam = float(np.sin(cam_yaw))

        def _sample_depth(depth_m: np.ndarray, u_px: int, v_px: int, k: int) -> float:
            if depth_m is None:
                return float('nan')
            if depth_m.ndim == 3:
                depth_m = depth_m[:, :, 0]
            if depth_m.shape[0] != H or depth_m.shape[1] != W:
                return float('nan')

            k = int(k)
            if k <= 0:
                val = float(depth_m[v_px, u_px])
                return val

            u0 = max(0, u_px - k)
            u1 = min(W - 1, u_px + k)
            v0 = max(0, v_px - k)
            v1 = min(H - 1, v_px + k)
            patch = depth_m[v0:v1 + 1, u0:u1 + 1]
            patch = patch[np.isfinite(patch)]
            if patch.size == 0:
                return float('nan')
            patch = patch[patch > 0.0]
            if patch.size == 0:
                return float('nan')
            return float(np.min(patch))

        rgb_points = []
        for wp_idx, (wx, wy) in enumerate(waypoints_local):
            # local map (x=col, y=row) -> full map indices
            global_row = gx1 + int(wy)
            global_col = gy1 + int(wx)

            # full map indices -> world meters (x from col, y from row)
            world_x = global_col * res_m
            world_y = global_row * res_m

            dx = world_x - float(start_x)
            dy = world_y - float(start_y)

            # world -> agent frame (forward/right)
            forward_agent = dx * cos_yaw + dy * sin_yaw
            right_agent = -dx * sin_yaw + dy * cos_yaw

            # agent frame -> camera-forward frame
            forward = forward_agent * cos_cam - right_agent * sin_cam
            right = forward_agent * sin_cam + right_agent * cos_cam

            if forward <= float(min_forward_m):
                continue
            if max_forward_m is not None and forward >= float(max_forward_m):
                continue

            X = -right
            Y = forward
            Z = -cam_h_m

            u = X * f / Y + xc
            grid_z = Z * f / Y + zc
            v = (H - 1) - grid_z

            ui = int(round(u))
            vi = int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                if depth_image_m is not None:
                    obs_depth_m = _sample_depth(depth_image_m, ui, vi, occlusion_kernel)
                    if np.isfinite(obs_depth_m) and obs_depth_m > 0.0:
                        if obs_depth_m + float(occlusion_margin_m) < float(Y):
                            continue
                if return_ids:
                    rgb_points.append((int(wp_idx) + 1, ui, vi))
                else:
                    rgb_points.append((ui, vi))

        return rgb_points

    def draw_local_waypoints_on_rgb(
        self,
        rgb_bgr: np.ndarray,
        waypoints_local: list,
        id: int = 0,
        color_bgr: tuple = (0, 255, 255),
        radius: int = 4,
        thickness: int = -1,
        draw_indices: bool = False,
        index_color_bgr: tuple = (0, 255, 0),
        index_scale: float = 0.6,
        index_thickness: int = 2,
        index_offset_px: tuple = (6, -6),
        return_pixels: bool = False,
        start_index: int = 0,
        **project_kwargs,
    ) -> np.ndarray:
        """在 RGB(BGR) 图上绘制 local map 路点。

        Args:
            start_index: 编号起始值，用于历史航点接续骨架航点编号。
        """
        if rgb_bgr is None:
            return rgb_bgr

        H, W = rgb_bgr.shape[:2]
        pts = self.project_local_waypoints_to_rgb_pixels(
            waypoints_local,
            id=id,
            image_size=(H, W),
            **project_kwargs,
        )
        out = rgb_bgr.copy()
        pixel_map = {} if return_pixels else None

        font = cv2.FONT_HERSHEY_DUPLEX
        try:
            offset_u = int(index_offset_px[0])
            offset_v = int(index_offset_px[1])
        except Exception:
            offset_u, offset_v = 6, -6

        if draw_indices or return_pixels:
            for pid, (u, v) in enumerate(pts, start=1 + start_index):
                u_i, v_i = int(u), int(v)
                cv2.circle(out, (u_i, v_i), int(radius), color_bgr, int(thickness))
                if draw_indices:
                    text = str(int(pid))
                    tx, ty = u_i + offset_u, v_i + offset_v
                    cv2.putText(out, text, (tx, ty), font, float(index_scale), (0, 0, 0), int(index_thickness) + 2, cv2.LINE_AA)
                    cv2.putText(out, text, (tx, ty), font, float(index_scale), index_color_bgr, int(index_thickness), cv2.LINE_AA)
                if return_pixels:
                    pixel_map[int(pid)] = (u_i, v_i)
        else:
            for (u, v) in pts:
                cv2.circle(out, (int(u), int(v)), int(radius), color_bgr, int(thickness))

        return (out, pixel_map) if return_pixels else out
