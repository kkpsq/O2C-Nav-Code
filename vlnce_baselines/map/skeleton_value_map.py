import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import rotate
from skimage.morphology import skeletonize
from scipy.ndimage import convolve

import networkx as nx
import copy

def prune_skeleton(skeleton_img, num_iter=15):
    """
    对骨架图进行剪枝，去除短小的毛刺分支，保留主干道。
    :param skeleton_img: 骨架图 (0 和 255 的单通道图像)
    :param num_iter: 剪枝迭代次数。数字越大，砍掉的分支越长。
    """
    # 将 0/255 的图像转换为 0/1 的矩阵
    skel = (skeleton_img > 0).astype(np.uint8)
    
    # 构造 8 邻域卷积核。中心点权重设为 10，周围 8 个点设为 1。
    # 核心逻辑：如果一个点是线段的“端点”，它本身是1(10)，且周围只有一个邻居(1)，总和就是 11。
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    for _ in range(num_iter):
        # 计算每个像素的邻域情况
        neighbor_count = cv2.filter2D(skel, -1, kernel)
        
        # 卷积结果为 11 的像素，就是末端死胡同的端点 (Endpoint)
        endpoints = (neighbor_count == 11)
        
        # 将端点删除 (置为 0)
        skel[endpoints] = 0
        
    return skel * 255

def points_to_occ_map_centered(xz_points, voxel_size=0.05, map_size=(400, 400)):

    H, W = map_size
    image = np.zeros((H, W), dtype=np.uint8)

    center = np.array([W // 2, H // 2])

    pixel_coords = (xz_points / voxel_size).astype(int) + center

    mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H)
    )
    pixel_coords = pixel_coords[mask]

    for x, z in pixel_coords:
        image[z, x] = 255 

    return image

def polar_sample(num_angles=60, num_radii=20, max_radius=3.0):

    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    radii = np.linspace(0, max_radius, num_radii + 1)[1:] 
    points = []
    for r in radii:
        for theta in angles:
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            points.append([x, z])
    return np.array(points)

def fill_points_to_shape(image, min_area=30):
    kernel = np.ones((7, 7), np.uint8) 
    dilated = cv2.dilate(image, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    filtered = np.zeros_like(image)
    for i in range(1, num_labels): 
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(image)

    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    return filled

def smooth_edges(image, blur_kernel=(75,75), binary_thresh=127):

    blurred = cv2.GaussianBlur(image, blur_kernel, 0)
    
    _, smoothed = cv2.threshold(blurred, binary_thresh, 255, cv2.THRESH_BINARY)
    
    return smoothed

def get_max_region(img):
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        largest_component = np.zeros_like(binary)
    else:
        largest_contour = max(contours, key=cv2.contourArea)

        largest_component = np.zeros_like(binary)
        cv2.drawContours(largest_component, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    return largest_component

def find_endpoints_and_branchpoints(binary_image):
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)

    binary = (binary_image == 255).astype(np.uint8)

    neighbor_count = convolve(binary, kernel, mode='constant')

    ys, xs = np.where(binary == 1)

    endpoints = []
    branchpoints = []

    for y, x in zip(ys, xs):
        n = neighbor_count[y, x]
        if n == 1:
            endpoints.append((x, y)) 
        # elif n >= 3:
        #     branchpoints.append((x, y))

    return endpoints, branchpoints

def merge_close_points(points, eps=5.0):
    if len(points) == 0:
        return points

    merged_points = []
    visited = set()

    for i, (y1, x1) in enumerate(points):
        if i in visited:
            continue

        close_points = []
        for j, (y2, x2) in enumerate(points):
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist < eps:
                close_points.append([y2, x2])
                visited.add(j)

        if close_points:
            avg_y = int(np.mean([p[0] for p in close_points]))
            avg_x = int(np.mean([p[1] for p in close_points]))
            merged_points.append([avg_y, avg_x])

    return np.array(merged_points)

def pixel_to_world_coords(pixel_coords, voxel_size=0.05, map_size=(400, 400)):

    H, W = map_size
    center = np.array([W // 2, H // 2])
    offset = pixel_coords - center
    world_coords = offset * voxel_size 
    return world_coords

def compute_relative_positions(points, self_pos):

    self_x, self_z = self_pos[0], self_pos[2] if len(self_pos) == 3 else self_pos[1]
    
    relative_coords = points - np.array([self_x, self_z])
    delta_x, delta_z = relative_coords[:, 0], relative_coords[:, 1]
    
    distances = np.sqrt(delta_x**2 + delta_z**2)
    
    angles_rad = np.arctan2(delta_x, -delta_z) 
    angles_rad = np.where(angles_rad < 0, angles_rad + 2 * np.pi, angles_rad) 
    
    angles_deg = np.degrees(angles_rad)

    angles_rad = 2 * np.pi - angles_rad
    angles_deg = 360 - angles_deg
    
    return {
        "relative_coords": relative_coords,
        "distances": distances,
        "angles_rad": angles_rad,
        "angles_deg": angles_deg
    }

def filter_close_points(points, min_dist=1.0):
    selected = []
    for p in points:
        if np.linalg.norm(p - np.array([0,0])) > min_dist:
            selected.append(p)
    return np.array(selected)



def filter_skeleton_within_radius(skeleton, origin_pixel, max_distance_m=1.5, voxel_size=0.05):

    max_dist_pixel = int(max_distance_m / voxel_size)
    H, W = skeleton.shape
    Y, X = np.ogrid[:H, :W]

    cx, cy = origin_pixel
    dist_sq = (X - cx)**2 + (Y - cy)**2
    mask = dist_sq <= max_dist_pixel**2

    skeleton_filtered = np.zeros_like(skeleton)
    skeleton_filtered[mask] = skeleton[mask]
    skeleton_filtered[skeleton_filtered != 255] = 0  
    return skeleton_filtered

def get_structure_wp_from_2d_map(free_space_map, position=[0,0,0], voxel_size=0.05, clamp_dist=(1,2)):
    """
    基于 2D 可通行区域地图提取导航路点
    
    Args:
        free_space_map: numpy array (H, W), uint8 格式。0 代表障碍，255 代表可通行。
        position: 自车当前的物理坐标 [x, y, theta]
        voxel_size: 地图的分辨率 (米/像素)
    """
    map_size = free_space_map.shape 

    # 1. 图像平滑与提取主连通域 (去除噪点，保留主安全区)
    # 注意：这里的 blur_kernel 根据你的实际地图大小可能需要微调
    filled_contour = smooth_edges(free_space_map, blur_kernel=(75, 75)) 
    filled_contour = get_max_region(filled_contour)
    navi_area = copy.deepcopy(filled_contour)

    # 2. 提取骨架 (中心线)
    skeleton = skeletonize(filled_contour // 255).astype(np.uint8) * 255

    # 3. 过滤超出半径的骨架
    center_y, center_x = int(map_size[0]/2), int(map_size[1]/2)
    skeleton_filter = filter_skeleton_within_radius(skeleton, (center_x, center_y), clamp_dist[1], voxel_size)
    
    if skeleton_filter.sum() == 0:
        skeleton_filter = skeleton
    filled_contour[skeleton == 255] = 128

    # 4. 提取端点与分支点
    end_points, branch_points = find_endpoints_and_branchpoints(skeleton_filter)
    merged_wp = merge_close_points(end_points + branch_points, eps=10.0)

    # 5. 坐标系转换 (像素 -> 世界坐标)
    wp_world = pixel_to_world_coords(merged_wp, voxel_size=voxel_size, map_size=map_size)
    wp_world = filter_close_points(wp_world, 1)

    # 6. 计算相对极坐标 (距离和角度)
    wp = compute_relative_positions(wp_world, position)

    # 可视化调试输出 (按需开启)
    # for point in merged_wp:
    #     cv2.circle(filled_contour, (point[0], point[1]), radius=10, color=128, thickness=-1)
    #     cv2.circle(filled_contour, (center_x, center_y), radius=10, color=64, thickness=-1)
    # Image.fromarray(filled_contour).save("debug_ego_map.png")

    return wp['angles_rad'], wp['distances'], navi_area

import cv2
import numpy as np

def find_endpoints_and_branchpoints(skeleton_img):
    # 确保是 0 和 1 的二值图
    skel = (skeleton_img > 0).astype(np.uint8)
    
    # 提取所有骨架像素的 3x3 邻域
    # 我们用一个巧妙的方法：给周围 8 个点编号，看 0->1 的跳变次数
    # P9 P2 P3
    # P8 P1 P4
    # P7 P6 P5
    
    # 结果容器
    endpoints = []
    branchpoints = []

    y_coords, x_coords = np.where(skel == 1)
    
    for x, y in zip(x_coords, y_coords):
        if x <= 0 or y <= 0 or x >= skel.shape[1]-1 or y >= skel.shape[0]-1:
            continue

        # 提取 8 个邻居 (顺时针方向)
        p2 = skel[y-1, x]
        p3 = skel[y-1, x+1]
        p4 = skel[y, x+1]
        p5 = skel[y+1, x+1]
        p6 = skel[y+1, x]
        p7 = skel[y+1, x-1]
        p8 = skel[y, x-1]
        p9 = skel[y-1, x-1]
        
        neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
        
        # 计算跳变次数 (从 0 变到 1 的次数)
        transitions = 0
        for i in range(len(neighbors)):
            if neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1:
                transitions += 1
        
        # 1. 只有 1 个跳变点：端点 (End Point)
        if transitions == 1:
            endpoints.append([int(x), int(y)])
        
        # 2. 有 3 个或更多跳变点：真正的分支点 (Branch Point)
        # 这个逻辑能完美过滤斜线，因为斜线上的跳转次数永远是 2
        elif transitions >= 3:
            branchpoints.append([int(x), int(y)])
            
    return endpoints, branchpoints

def merge_close_points(points, eps=15.0):
    """
    合并距离小于 eps 的点，计算其几何中心。
    
    :param points: 点列表 [[x1, y1], [x2, y2], ...]
    :param eps: 合并阈值（像素距离）
    :return: 合并后的点列表
    """
    if not points:
        return []
        
    merged = []
    # 标记哪些点已经被合并过了
    used = [False] * len(points)
    
    pts = np.array(points)
    
    for i in range(len(pts)):
        if used[i]:
            continue
            
        # 计算当前点到所有其他点的距离
        dist = np.linalg.norm(pts - pts[i], axis=1)
        
        # 找到所有距离小于阈值的点
        close_indices = np.where(dist < eps)[0]
        
        # 提取这些近点，计算它们的平均坐标（中心点）
        close_pts = pts[close_indices]
        center = np.mean(close_pts, axis=0)
        merged.append([int(center[0]), int(center[1])])
        
        # 将这些点标记为已使用
        for idx in close_indices:
            used[idx] = True
            
    return merged