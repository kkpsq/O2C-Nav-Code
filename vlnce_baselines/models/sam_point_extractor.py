import cv2
import numpy as np

class SAMPointExtractor:
    """
    一个用于提取 SAM (Segment Anything Model) 掩码坐标、合并相近点并进行可视化的工具类。
    """
    
    def __init__(self, segment_module, default_merge_threshold=30):
        """
        初始化提取器。
        
        :param segment_module: 已经实例化的 SAM 分割模型对象 (例如你的 GroundedSAM 实例)
        :param default_merge_threshold: 默认的点合并距离阈值 (像素)
        """
        self.segment_module = segment_module
        self.default_merge_threshold = default_merge_threshold

    def process(self, bgr_image, classes, merge_threshold=None):
        """
        处理图像：执行分割、提取代表点、合并相近点、并在图上标出序号。
        
        :param bgr_image: 输入的 BGR 格式图像 (numpy array)
        :param classes: 需要检测的类别列表
        :param merge_threshold: 覆盖默认合并阈值，如果不传则使用默认值
        :return: masks (张量), labels (列表), clean_vis (画好点的图像), point_info_list (点信息字典列表)
        """
        # 确定使用的阈值
        threshold = merge_threshold if merge_threshold is not None else self.default_merge_threshold
        
        # 1. 调用分割模块
        masks, labels, _, current_detections = self.segment_module.segment(bgr_image, classes=classes)
        
        clean_vis = bgr_image.copy()
        raw_points = []

        # 2. 提取原始点
        if len(masks) > 0:
            for i, mask in enumerate(masks):
                class_name = labels[i].rsplit(' ', 1)[0]
                point_x, point_y = 0, 0

                if class_name == 'floor':
                    y_coords, x_coords = np.where(mask > 0)
                    if len(y_coords) > 0:
                        y_mean = np.mean(y_coords)
                        y_min = np.min(y_coords)
                        point_y = int((y_mean + y_min) / 2)
                        point_x = int(np.mean(x_coords))
                    else:
                        continue
                else:
                    box = current_detections.xyxy[i]
                    x1, y1, x2, y2 = box
                    point_x = int((x1 + x2) / 2)
                    point_y = int(y2)
                
                raw_points.append({'class_name': class_name, 'coords': (point_x, point_y)})

        # 3. 根据距离阈值合并相近点
        merged_points = []
        for raw_pt in raw_points:
            rx, ry = raw_pt['coords']
            r_cls = raw_pt['class_name']
            
            is_merged = False
            for m_pt in merged_points:
                mx, my = m_pt['coordinates']
                # 计算两点之间的欧氏距离
                if np.sqrt((rx - mx)**2 + (ry - my)**2) < threshold:
                    # 如果距离在阈值内，将类别合并到一起，并更新中心坐标
                    if r_cls not in m_pt['class_names']:
                        m_pt['class_names'].append(r_cls)
                    m_pt['coordinates'] = (int((rx + mx) / 2), int((ry + my) / 2))
                    is_merged = True
                    break
            
            if not is_merged:
                merged_points.append({'class_names': [r_cls], 'coordinates': (rx, ry)})

        # 4. 画点、标序号、生成最终列表
        point_info_list = []
        for i, m_pt in enumerate(merged_points):
            point_id = i + 1
            px, py = m_pt['coordinates']
            classes_list = m_pt['class_names']
            
            point_info_list.append({
                'id': point_id,
                'class_names': classes_list,
                'coordinates': (px, py)
            })

            # BGR 格式下的红色点 (0, 0, 255)，绿色字 (0, 255, 0)
            cv2.circle(clean_vis, (px, py), radius=6, color=(0, 0, 255), thickness=-1)
            cv2.putText(clean_vis, str(point_id), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return masks, labels, clean_vis, point_info_list


    def detect_objects(self, bgr_image, classes, topk=None):
        """Return a compact per-image object list (GroundingDINO-only when available).

        This intentionally does NOT draw any points/boxes. It is meant for
        building a textual list for prompting.

        Returns:
            List[Tuple[str, float]]: sorted by confidence desc, unique by class.
        """
        # Prefer DINO-only path to avoid SAM + any visualization.
        if hasattr(self.segment_module, "detect"):
            labels, _ = self.segment_module.detect(bgr_image, classes=classes)
        else:
            # Fallback: use segment() and ignore masks/annotated image.
            _, labels, _, _ = self.segment_module.segment(bgr_image, classes=classes)

        best_by_class = {}
        for item in labels or []:
            try:
                name, conf_str = item.rsplit(" ", 1)
                conf = float(conf_str)
            except Exception:
                name, conf = str(item), 0.0
            prev = best_by_class.get(name)
            if prev is None or conf > prev:
                best_by_class[name] = conf

        pairs = sorted(best_by_class.items(), key=lambda x: x[1], reverse=True)
        if isinstance(topk, int) and topk > 0:
            pairs = pairs[:topk]
        return pairs