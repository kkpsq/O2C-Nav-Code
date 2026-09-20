import base64
import io
import json
import os
import re

import cv2
import numpy as np
from PIL import Image

from habitat import logger

from .utils.prompts import O2C_PROMPT
from .utils.api import O2CNavAPI
from .utils.visualization import O2CNavVisualizer


class VLMReasoningAgent:
    def __init__(self, visualizer: O2CNavVisualizer):
        la_api_key = os.getenv('LA_API_KEY', None)
        la_base_url = os.getenv('LA_BASE_URL', None)
        la_model_name = os.getenv("LA_MODEL_NAME", 'gpt-4o-2024-11-20')
        self.model = O2CNavAPI(
            la_api_key=la_api_key,
            la_base_url=la_base_url,
            la_model_name=la_model_name,
        )
        self.model.eval()
        self.visualizer = visualizer


    def img_to_base64(self, img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64

    def navigate_or_backtrack(
        self,
        instruction,
        visited_targets,
        episode_id="",
        sam_extractor=None,
        classes=None,
        use_sam_points_for_la: bool = False,
        include_dino_objects_for_la: bool = True,
        dino_topk: int = 8,
        loop_warning: str = ""
    ):
        """
        Use Language Action Model to analyze instruction + history + 4-dir images，decide where to nav or backtrack
        return LA：navigate to [left, right, forward, behind] / backtrack to <waypoint_id>
        """

        panorama_images = visited_targets[-1]['panorama_frames'] if visited_targets else []

        # History：[init image] -> "turn xxx" -> [dir image] -> "go to xxx" -> [arrival image] -> ...
        history_content = []
        history_list = visited_targets[:-1]
        keep_recent_n = 5
        recent_targets = history_list[-keep_recent_n:]
        # 计算起始索引，确保输入给 LA 的 Waypoint 编号与实际保存的一致
        start_idx = max(0, len(history_list) - keep_recent_n)

        for j, target in enumerate(recent_targets):
            i = start_idx + j  # 还原正确的历史航点全局 ID

            if 'init_image' in target:
                history_content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                history_content.append({"type": "text", "text": f"Waypoint {i}: Initial view"})

            if 'turn_action' in target:
                history_content.append({"type": "text", "text": f"Action: {target['turn_action']}"})

            if 'dir_image' in target:
                history_content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                history_content.append({"type": "text", "text": f"After turn view"})

            if 'description' in target:
                history_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})

        # if len(history_list) > 0:
        #     last_target = history_list[-1] # 获取最近一次完成的航点记录
        #     if 'progress_analysis' in last_target and last_target['progress_analysis']:
        #         history_content.append({
        #             "type": "text",
        #             "text": f"Latest Progress Analysis: {last_target['progress_analysis']}"
        #         })

        # Current 4 views:
        current_views = []
        view_definitions = [
            {'angle': 0, 'name': 'forward', 'label': 'Current FORWARD view'},
            {'angle': 90, 'name': 'left', 'label': 'View after turning LEFT'},
            {'angle': 180, 'name': 'behind', 'label': 'View after turning BEHIND'},
            {'angle': 270, 'name': 'right', 'label': 'View after turning RIGHT'}
        ]

        # view -> {point_id: (x_px, y_px)}
        point_centers = {'forward': {}, 'left': {}, 'behind': {}, 'right': {}}

        for view in view_definitions:
            angle = view['angle']
            frame_idx = angle // 90
            if frame_idx < len(panorama_images):
                rgb_raw = panorama_images[frame_idx].get('rgb', None)
                rgb_prompt = panorama_images[frame_idx].get('rgb_wp', rgb_raw)
                clean_vis = None

                # Prefer skeleton-waypoint pixel map if provided by mapping module.
                wp_pixels = panorama_images[frame_idx].get('wp_pixels', None)
                if isinstance(wp_pixels, dict):
                    try:
                        point_centers[view['name']] = {int(k): tuple(v) for k, v in wp_pixels.items()}
                    except Exception:
                        point_centers[view['name']] = {}

                if isinstance(rgb_raw, np.ndarray):
                    if rgb_raw.dtype != np.uint8:
                        rgb_raw = (rgb_raw * 255).astype(np.uint8)

                    # Add a GroundingDINO object list to the text prompt (no point/box drawing).
                    if include_dino_objects_for_la and sam_extractor is not None and classes is not None:
                        try:
                            bgr_image = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                            obj_pairs = sam_extractor.detect_objects(bgr_image, classes, topk=dino_topk)
                            if obj_pairs:
                                obj_str = ", ".join([f"{n}({c:.2f})" for n, c in obj_pairs])
                                view['label'] = f"{view['label']}\n[Detected Objects in this view: {obj_str}]"
                        except Exception:
                            pass
                    # Keep GroundingDINO/SAM framework, but do not use its points for LA by default.
                    if use_sam_points_for_la and sam_extractor is not None and classes is not None:
                        bgr_images = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                        masks, labels, clean_vis, point_info_list = sam_extractor.process(bgr_images, classes)
                        # record text and location
                        vlm_labels_list = []
                        for pt in point_info_list:
                            pid = pt['id']
                            px, py = pt['coordinates']
                            classes_list = pt['class_names']
                            point_centers[view['name']][int(pid)] = (px, py)
                            # tips text
                            vlm_labels_list.append(f"{pid}: {', '.join(classes_list)}")

                        # For LA prompt, prefer waypoint-overlaid RGB if available.
                        if isinstance(rgb_prompt, np.ndarray):
                            if rgb_prompt.dtype != np.uint8:
                                rgb_prompt = (rgb_prompt * 255).astype(np.uint8)
                            img = Image.fromarray(rgb_prompt)
                        else:
                            img = Image.fromarray(rgb_raw)

                        # add items list to view
                        if vlm_labels_list:
                            detected_str = ",".join(vlm_labels_list)
                            view['label'] = f"{view['label']}\n[Detected Points in this view: {detected_str}]"
                    else:
                        if isinstance(rgb_prompt, np.ndarray):
                            if rgb_prompt.dtype != np.uint8:
                                rgb_prompt = (rgb_prompt * 255).astype(np.uint8)
                            img = Image.fromarray(rgb_prompt)
                        else:
                            img = Image.fromarray(rgb_raw) if isinstance(rgb_raw, np.ndarray) else rgb_prompt
                else:
                    img = rgb_prompt

                # Always save the same image used for LA prompt (prefer rgb_wp/rgb_prompt).
                if hasattr(self, 'visualizer') and self.visualizer.save_dir:
                    step_val = visited_targets[-1].get('step', 'unknown')
                    if isinstance(step_val, (int, np.integer)):
                        step_num = f"{int(step_val):04d}"
                    else:
                        step_num = str(step_val)
                    save_folder = os.path.join(self.visualizer.save_dir, str(episode_id), f"step_{step_num}")
                    os.makedirs(save_folder, exist_ok=True)
                    save_path = os.path.join(save_folder, f"sam_view_{view['name']}.png")
                    try:
                        to_save = rgb_prompt
                        if not isinstance(to_save, np.ndarray):
                            to_save = np.array(to_save)
                        if to_save.dtype != np.uint8:
                            if np.issubdtype(to_save.dtype, np.floating):
                                to_save = (np.clip(to_save, 0.0, 1.0) * 255.0).astype(np.uint8)
                            else:
                                to_save = np.clip(to_save, 0, 255).astype(np.uint8)
                        if to_save.ndim == 3 and to_save.shape[2] == 3:
                            to_save_bgr = cv2.cvtColor(to_save, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(save_path, to_save_bgr)
                        else:
                            cv2.imwrite(save_path, to_save)
                    except Exception:
                        # Do not fall back to any point/box visualizations.
                        pass
                current_views.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
                current_views.append({"type": "text", "text": view['label']})
        # 告知每个方向的航点数量
        point_summary_text = "### Waypoint Availability Summary ###\n"
        direction_counts = {}
        for view_name in ['forward', 'left', 'right', 'behind']:
            count = len(point_centers.get(view_name, {}))
            direction_counts[view_name] = count
            status = "Has Waypoints" if count > 0 else "No Waypoints"
            point_summary_text += f"- {view_name.capitalize()}: {count} waypoints -> {status}\n"

        # Backtrack check
        num_waypoints = len([t for t in visited_targets[:-1] if 'description' in t])
        should_consider_backtrack = 1

        # Prompt construction based on whether backtrack or not
        content = [{"type": "text", "text": f"Navigation Task: \"{instruction}\"\n\nNavigation History:"}]
        content.extend(history_content)
        content.append({"type": "text", "text": "\nCurrent 4-directional views:"})
        content.extend(current_views)
        content.append({"type": "text", "text": "\n" + point_summary_text})

        if loop_warning:
            content.append({"type": "text", "text": f"\n{loop_warning}\n"})

        if visited_targets and len(visited_targets) <= 2:
            early_warning = "CRITICAL WARNING: This is the very beginning of the navigation. You MUST NOT output 'stop' as your action. You must explore the environment first."
            content.append({"type": "text", "text": f"\n{early_warning}\n"})

        prompt = O2C_PROMPT.format(
            width=self.visualizer.width,
            height=self.visualizer.height,
        )

        logger.info(prompt)

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=20480,
            temperature=0.7,
            use_la=True
        )

        logger.info('LA-response:')
        logger.info(f"{output_text}")
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)

        # Json parse
        while not json_match:
            output_text = self.model.generate(
                messages=messages,
                max_new_tokens=20480,
                temperature=0.7,
                use_la=True
            )
            logger.info('Retried.')
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)

        if json_match:
            try:
                response_data = json.loads(json_match.group())
            except:
                response_data = {}
            action = response_data.get('action', 'navigate to forward') or ''
            progress_analysis = response_data.get('progress_analysis', '')
            reasoning = response_data.get('reasoning', '')

            # 添加target_id
            target_id = response_data.get('target_id', None)
            navigable_bbox = response_data.get('navigable_bbox', None)
            # get the pixel coordinate of target_id if exist
            target_pixel = None

            action = action.lower()
            if 'stop' in action:
                return {
                    'action': 'STOP',
                    'direction': 'none',  # 停止不需要方向
                    'progress_analysis': progress_analysis,
                    'reasoning': reasoning,
                    'target_id': None,
                    'target_pixel': None
                }

            if action.startswith('backtrack to'):
                waypoint_id = action.split('backtrack to ')[-1].strip()
                if waypoint_id.startswith('waypoint'):
                    waypoint_id = waypoint_id.split('waypoint')[-1].strip()
                # logger.info('Waypoint:%s', waypoint_id)
                try:
                    waypoint_id = int(waypoint_id)
                    return {
                        'action': 'BACKTRACK',
                        'waypoint': waypoint_id,
                        'progress_analysis': progress_analysis,
                        'reasoning': reasoning,
                        'target_id': None,
                        'target_pixel': None
                    }
                except:
                    pass



            if 'forward' in action:
                direction = 'forward'
            elif 'left' in action:
                direction = 'left'
            elif 'right' in action:
                direction = 'right'
            elif 'behind' in action:
                direction = 'behind'
            else:
                direction = 'forward'

            target_pixel = None
            if target_id is not None:
                try:
                    target_pixel = point_centers.get(direction, {}).get(int(target_id), None)
                except:
                    pass

            if target_pixel is None and navigable_bbox is not None and len(navigable_bbox) == 4:
                try:
                    xmin, ymin, xmax, ymax = navigable_bbox
                    center_x = (xmin + xmax) / 2.0
                    bottom_y = ymax
                    center_x = max(0, min(center_x, self.visualizer.width - 1))
                    bottom_y = max(0, min(bottom_y, self.visualizer.height - 1))
                    target_pixel = (int(center_x), int(bottom_y))
                except Exception as e:
                    logger.info(f"Failed to parse navigable_bbox: {e}")
                    pass

            if target_pixel is None:
                # 兜底
                fallback_x = self.visualizer.width // 2
                fallback_y = int(self.visualizer.height * 0.8)

                # 确保兜底坐标在安全范围内
                fallback_x = max(0, min(fallback_x, self.visualizer.width - 1))
                fallback_y = max(0, min(fallback_y, self.visualizer.height - 1))

                target_pixel = (int(fallback_x), int(fallback_y))

            return {
                'action': 'NAVIGATE',
                'direction': direction,
                'progress_analysis': progress_analysis,
                'reasoning': reasoning,
                'target_id': target_id,
                'target_pixel': target_pixel
            }

        return {
            'action': 'NAVIGATE',
            'direction': 'forward',
            'progress_analysis': 'Unable to analyze due to parsing error',
            'reasoning': 'Fallback to forward navigation',
            'target_id': None,
            'target_pixel': None
        }


    def reset(self):
        # self.model.reset_stats()
        pass
