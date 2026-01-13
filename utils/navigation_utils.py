"""
导航工具模块
实现距离估算、空间关系分析、避障路径规划等功能
"""
import numpy as np
import cv2


def estimate_distance(box, frame_height, frame_width, camera_height=1.5, fov_horizontal=60):
    """
    根据物体在画面中的大小和位置，估算它与用户的实际距离
    
    Args:
        box: 检测框，格式为 (x1, y1, x2, y2) 或包含 xyxy 属性的对象
        frame_height: 画面高度（像素）
        frame_width: 画面宽度（像素）
        camera_height: 相机高度（米），默认1.5米（假设是手持设备）
        fov_horizontal: 水平视场角（度），默认60度
    
    Returns:
        float: 估算的距离（米）
    """
    # 提取边界框坐标
    if hasattr(box, 'xyxy'):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    else:
        x1, y1, x2, y2 = box
    
    # 计算边界框中心点和底部位置
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2
    box_bottom_y = y2  # 物体底部在画面中的位置
    
    # 计算边界框高度（像素）
    box_height = y2 - y1
    
    # 方法1：基于物体在画面中的垂直位置估算距离
    # 假设物体底部接触地面，距离越远，物体底部在画面中的位置越高
    # 使用透视投影模型：距离 = (camera_height * frame_height) / (box_bottom_y - frame_height/2)
    if box_bottom_y > frame_height * 0.3:  # 物体在画面下方（较近）
        # 使用底部位置估算
        normalized_bottom = (box_bottom_y - frame_height * 0.5) / (frame_height * 0.5)
        # 经验公式：距离与归一化位置成反比
        distance = camera_height / (0.1 + abs(normalized_bottom) * 0.5)
    else:
        # 物体在画面上方（较远），使用中心点估算
        normalized_center = (box_center_y - frame_height * 0.5) / (frame_height * 0.5)
        distance = camera_height / (0.05 + abs(normalized_center) * 0.3)
    
    # 方法2：基于物体大小估算距离（作为补充）
    # 假设标准物体高度（如人1.7米，汽车1.5米，障碍物1米）
    # 这里使用一个通用的估算：物体高度与距离成反比
    if box_height > 0:
        # 归一化高度（相对于画面高度）
        normalized_height = box_height / frame_height
        # 假设标准物体在画面中占10%高度时距离为5米
        size_based_distance = 5.0 / (normalized_height * 10)
        # 综合两种方法（加权平均）
        distance = 0.7 * distance + 0.3 * size_based_distance
    
    # 限制距离范围（1-50米）
    distance = max(1.0, min(50.0, distance))
    
    return round(distance, 1)


def check_spatial_relationship(obstacle_box, blind_road_boxes=None, sidewalk_boxes=None):
    """
    分析障碍物与盲道/人行道的空间关系
    
    Args:
        obstacle_box: 障碍物检测框
        blind_road_boxes: 盲道检测框列表（可选）
        sidewalk_boxes: 人行道检测框列表（可选）
    
    Returns:
        dict: {
            'on_blind_road': bool,  # 是否在盲道上
            'on_sidewalk': bool,    # 是否在人行道上
            'overlap_ratio': float  # 重叠比例（0-1）
        }
    """
    # 提取障碍物边界框
    if hasattr(obstacle_box, 'xyxy'):
        obs_x1, obs_y1, obs_x2, obs_y2 = obstacle_box.xyxy[0].cpu().numpy().astype(int)
    else:
        obs_x1, obs_y1, obs_x2, obs_y2 = obstacle_box
    
    obs_area = (obs_x2 - obs_x1) * (obs_y2 - obs_y1)
    
    result = {
        'on_blind_road': False,
        'on_sidewalk': False,
        'overlap_ratio': 0.0
    }
    
    # 检查与盲道的重叠
    if blind_road_boxes:
        max_blind_overlap = 0.0
        for blind_box in blind_road_boxes:
            if hasattr(blind_box, 'xyxy'):
                br_x1, br_y1, br_x2, br_y2 = blind_box.xyxy[0].cpu().numpy().astype(int)
            else:
                br_x1, br_y1, br_x2, br_y2 = blind_box
            
            # 计算交集
            inter_x1 = max(obs_x1, br_x1)
            inter_y1 = max(obs_y1, br_y1)
            inter_x2 = min(obs_x2, br_x2)
            inter_y2 = min(obs_y2, br_y2)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                overlap_ratio = inter_area / obs_area if obs_area > 0 else 0
                max_blind_overlap = max(max_blind_overlap, overlap_ratio)
        
        # 如果重叠比例超过30%，认为在盲道上
        if max_blind_overlap > 0.3:
            result['on_blind_road'] = True
            result['overlap_ratio'] = max_blind_overlap
    
    # 检查与人行道的重叠
    if sidewalk_boxes:
        max_sidewalk_overlap = 0.0
        for sidewalk_box in sidewalk_boxes:
            if hasattr(sidewalk_box, 'xyxy'):
                sw_x1, sw_y1, sw_x2, sw_y2 = sidewalk_box.xyxy[0].cpu().numpy().astype(int)
            else:
                sw_x1, sw_y1, sw_x2, sw_y2 = sidewalk_box
            
            # 计算交集
            inter_x1 = max(obs_x1, sw_x1)
            inter_y1 = max(obs_y1, sw_y1)
            inter_x2 = min(obs_x2, sw_x2)
            inter_y2 = min(obs_y2, sw_y2)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                overlap_ratio = inter_area / obs_area if obs_area > 0 else 0
                max_sidewalk_overlap = max(max_sidewalk_overlap, overlap_ratio)
        
        # 如果重叠比例超过30%，认为在人行道上
        if max_sidewalk_overlap > 0.3:
            result['on_sidewalk'] = True
            if result['overlap_ratio'] < max_sidewalk_overlap:
                result['overlap_ratio'] = max_sidewalk_overlap
    
    return result


def find_avoidance_path(obstacle_box, blind_road_boxes, sidewalk_boxes, frame_width, frame_height):
    """
    在人行道区域内寻找避障路径
    
    Args:
        obstacle_box: 障碍物检测框
        blind_road_boxes: 盲道检测框列表
        sidewalk_boxes: 人行道检测框列表
        frame_width: 画面宽度
        frame_height: 画面高度
    
    Returns:
        dict: {
            'direction': str,  # 'left', 'right', 'straight'
            'distance': float,  # 避让距离（米）
            'feasible': bool   # 是否可行
        }
    """
    # 提取障碍物位置
    if hasattr(obstacle_box, 'xyxy'):
        obs_x1, obs_y1, obs_x2, obs_y2 = obstacle_box.xyxy[0].cpu().numpy().astype(int)
    else:
        obs_x1, obs_y1, obs_x2, obs_y2 = obstacle_box
    
    obs_center_x = (obs_x1 + obs_x2) / 2
    obs_width = obs_x2 - obs_x1
    
    # 计算盲道中心位置
    blind_center_x = None
    if blind_road_boxes:
        blind_xs = []
        for blind_box in blind_road_boxes:
            if hasattr(blind_box, 'xyxy'):
                br_x1, _, br_x2, _ = blind_box.xyxy[0].cpu().numpy().astype(int)
            else:
                br_x1, _, br_x2, _ = blind_box
            blind_xs.extend([br_x1, br_x2])
        if blind_xs:
            blind_center_x = (min(blind_xs) + max(blind_xs)) / 2
    
    # 计算人行道可用区域
    sidewalk_left = 0
    sidewalk_right = frame_width
    if sidewalk_boxes:
        sidewalk_xs = []
        for sidewalk_box in sidewalk_boxes:
            if hasattr(sidewalk_box, 'xyxy'):
                sw_x1, _, sw_x2, _ = sidewalk_box.xyxy[0].cpu().numpy().astype(int)
            else:
                sw_x1, _, sw_x2, _ = sidewalk_box
            sidewalk_xs.extend([sw_x1, sw_x2])
        if sidewalk_xs:
            sidewalk_left = min(sidewalk_xs)
            sidewalk_right = max(sidewalk_xs)
    
    # 计算左右两侧的可用空间
    left_space = obs_x1 - sidewalk_left
    right_space = sidewalk_right - obs_x2
    
    # 计算避让距离（基于障碍物宽度，转换为实际距离）
    avoidance_distance = estimate_distance(obstacle_box, frame_height, frame_width) * 0.3  # 避让距离约为检测距离的30%
    avoidance_distance = max(1.0, min(5.0, avoidance_distance))  # 限制在1-5米
    
    # 决定避让方向
    result = {
        'direction': 'straight',
        'distance': avoidance_distance,
        'feasible': False
    }
    
    # 如果左侧空间更大，建议向左避让
    if left_space > right_space and left_space > obs_width * 1.5:
        result['direction'] = 'left'
        result['feasible'] = True
    # 如果右侧空间更大，建议向右避让
    elif right_space > left_space and right_space > obs_width * 1.5:
        result['direction'] = 'right'
        result['feasible'] = True
    # 如果两侧空间都不足，尝试直行（可能需要减速）
    elif left_space > obs_width * 0.5 and right_space > obs_width * 0.5:
        result['direction'] = 'straight'
        result['feasible'] = True
        result['distance'] = avoidance_distance * 0.5  # 直行时距离更短
    
    return result


def get_position_description(box, frame_width):
    """
    获取物体在画面中的位置描述（左前方、右前方等）
    
    Args:
        box: 检测框
        frame_width: 画面宽度
    
    Returns:
        str: 位置描述（'左前方', '右前方', '前方'）
    """
    if hasattr(box, 'xyxy'):
        x1, _, x2, _ = box.xyxy[0].cpu().numpy().astype(int)
    else:
        x1, _, x2, _ = box
    
    box_center_x = (x1 + x2) / 2
    
    if box_center_x < frame_width / 3:
        return '左前方'
    elif box_center_x > 2 * frame_width / 3:
        return '右前方'
    else:
        return '前方'
