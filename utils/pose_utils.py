"""
MediaPipe 姿态分析工具模块
功能：
1. 行人姿态分析 - 判断行人是否朝你走来/背对你/侧身
2. 姿态异常检测 - 辅助暴力行为检测（跌倒、打架姿态、挥拳等）
"""
import sys
import os

# ========== 解决 MediaPipe 中文路径问题 ==========
# 将 mediapipe 库安装到英文路径，然后添加到 Python 搜索路径
MEDIAPIPE_LIB_DIR = "D:/mediapipe_lib"
if os.path.exists(MEDIAPIPE_LIB_DIR) and MEDIAPIPE_LIB_DIR not in sys.path:
    sys.path.insert(0, MEDIAPIPE_LIB_DIR)
    print(f"[MediaPipe] 添加库路径: {MEDIAPIPE_LIB_DIR}")

# MediaPipe 模型缓存到英文路径
MEDIAPIPE_CACHE_DIR = "D:/mediapipe_models"
os.environ['MEDIAPIPE_RESOURCE_DIR'] = MEDIAPIPE_CACHE_DIR
os.environ['TFHUB_CACHE_DIR'] = MEDIAPIPE_CACHE_DIR
os.environ['XDG_CACHE_HOME'] = MEDIAPIPE_CACHE_DIR

# 创建缓存目录
if not os.path.exists(MEDIAPIPE_CACHE_DIR):
    try:
        os.makedirs(MEDIAPIPE_CACHE_DIR)
        print(f"[MediaPipe] 创建缓存目录: {MEDIAPIPE_CACHE_DIR}")
    except Exception as e:
        print(f"[MediaPipe] 创建缓存目录失败: {e}")

import cv2
import numpy as np
import time
import urllib.request
from PIL import Image, ImageDraw, ImageFont

# 中文字体路径（Windows 系统自带字体）
CHINESE_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
CHINESE_FONT = None

def get_chinese_font(size=20):
    """获取中文字体"""
    global CHINESE_FONT
    try:
        if CHINESE_FONT is None or CHINESE_FONT.size != size:
            CHINESE_FONT = ImageFont.truetype(CHINESE_FONT_PATH, size)
        return CHINESE_FONT
    except:
        # 如果加载失败，返回默认字体
        return ImageFont.load_default()


def put_chinese_text(img, text, position, color=(0, 255, 0), font_size=20):
    """
    在图像上绘制中文文字
    
    Args:
        img: OpenCV 图像 (BGR)
        text: 要绘制的文字
        position: (x, y) 位置
        color: BGR 颜色
        font_size: 字体大小
    
    Returns:
        img: 绘制后的图像
    """
    # 转换为 PIL 图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 获取字体
    font = get_chinese_font(font_size)
    
    # BGR 转 RGB
    rgb_color = (color[2], color[1], color[0])
    
    # 绘制文字
    draw.text(position, text, font=font, fill=rgb_color)
    
    # 转回 OpenCV 格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# MediaPipe 延迟导入（避免未安装时报错）
mp = None
mp_pose = None
mp_drawing = None
pose_detector = None

# 新版 API 变量
USE_NEW_API = False
PoseLandmarker = None
mp_tasks = None

# 模型文件路径（新版 API 需要）
POSE_MODEL_PATH = os.path.join(MEDIAPIPE_CACHE_DIR, "pose_landmarker_lite.task")
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


def download_model_if_needed():
    """下载姿态检测模型文件（新版 API 需要）"""
    if os.path.exists(POSE_MODEL_PATH):
        return True
    
    print(f"[MediaPipe] 正在下载模型文件到 {POSE_MODEL_PATH}...")
    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("[MediaPipe] ✓ 模型下载完成")
        return True
    except Exception as e:
        print(f"[MediaPipe] ❌ 模型下载失败: {e}")
        return False


def init_mediapipe():
    """初始化 MediaPipe 姿态检测器（兼容新旧版本 API）"""
    global mp, mp_pose, mp_drawing, pose_detector, USE_NEW_API, PoseLandmarker, mp_tasks
    
    if pose_detector is not None:
        return True
    
    try:
        os.environ['MEDIAPIPE_RESOURCE_DIR'] = MEDIAPIPE_CACHE_DIR
        import mediapipe
        mp = mediapipe
        
        # 尝试旧版 API (mediapipe < 0.10.10)
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            
            pose_detector = mp_pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                enable_segmentation=False
            )
            USE_NEW_API = False
            print("[MediaPipe] ✓ 姿态检测器初始化成功 (旧版 API)")
            return True
        
        # 使用新版 Tasks API (mediapipe >= 0.10.10)
        print("[MediaPipe] 检测到新版 MediaPipe，使用 Tasks API...")
        
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        mp_tasks = mp_python
        PoseLandmarker = vision.PoseLandmarker
        
        # 下载模型文件
        if not download_model_if_needed():
            return False
        
        # 创建姿态检测器
        base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=3,  # 最多检测3个人
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        pose_detector = PoseLandmarker.create_from_options(options)
        USE_NEW_API = True
        
        # 新版没有内置绘图，我们自己实现
        mp_drawing = None
        mp_pose = None
        
        print("[MediaPipe] ✓ 姿态检测器初始化成功 (新版 Tasks API)")
        print(f"[MediaPipe] 模型路径: {POSE_MODEL_PATH}")
        return True
        
    except ImportError as e:
        print(f"[MediaPipe] ❌ 未安装 mediapipe: {e}")
        return False
    except Exception as e:
        print(f"[MediaPipe] ❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_pose_landmarks(frame):
    """
    获取图像中人体的33个关键点（兼容新旧 API）
    
    Args:
        frame: BGR 格式的图像帧
    
    Returns:
        results: 检测结果对象
        None: 如果检测失败或无人
    """
    if pose_detector is None:
        if not init_mediapipe():
            return None
    
    # MediaPipe 需要 RGB 格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if USE_NEW_API:
        # 新版 API
        import mediapipe as mp_new
        mp_image = mp_new.Image(image_format=mp_new.ImageFormat.SRGB, data=rgb_frame)
        results = pose_detector.detect(mp_image)
        # 包装成兼容格式
        return NewAPIResultWrapper(results)
    else:
        # 旧版 API
        results = pose_detector.process(rgb_frame)
        return results


class NewAPIResultWrapper:
    """包装新版 API 结果，使其与旧版兼容"""
    def __init__(self, results):
        self._results = results
        self.pose_landmarks = None
        
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            # 取第一个检测到的人
            self.pose_landmarks = LandmarkListWrapper(results.pose_landmarks[0])
    
    @property
    def pose_world_landmarks(self):
        if self._results.pose_world_landmarks and len(self._results.pose_world_landmarks) > 0:
            return LandmarkListWrapper(self._results.pose_world_landmarks[0])
        return None


class LandmarkListWrapper:
    """包装关键点列表"""
    def __init__(self, landmarks):
        self.landmark = [LandmarkWrapper(lm) for lm in landmarks]


class LandmarkWrapper:
    """包装单个关键点，添加 visibility 属性"""
    def __init__(self, lm):
        self.x = lm.x
        self.y = lm.y
        self.z = lm.z
        self.visibility = getattr(lm, 'visibility', 0.9)  # 新版可能没有 visibility


def analyze_person_orientation(landmarks, image_width, image_height):
    """
    分析行人的朝向（面向你/背对你/侧身）
    
    原理：通过比较左右肩膀的 z 坐标（深度）判断朝向
    - 如果左肩 z > 右肩 z：人面向右侧
    - 如果左肩 z < 右肩 z：人面向左侧
    - 通过鼻子和肩膀中心的位置关系判断正面/背面
    
    Args:
        landmarks: MediaPipe 的 pose_landmarks.landmark
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        dict: {
            'orientation': 'facing_you' | 'facing_away' | 'side_left' | 'side_right',
            'confidence': float (0-1),
            'approaching': bool,  # 是否正在接近
            'description': str    # 中文描述
        }
    """
    if landmarks is None:
        return None
    
    # 获取关键点
    # 索引参考: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_EYE = 2
    RIGHT_EYE = 5
    
    try:
        nose = landmarks[NOSE]
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]
        
        # 计算肩膀中心
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2
        
        # 计算肩膀宽度（用于判断侧身程度）
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        
        # 计算眼睛宽度（用于判断正面/背面）
        eye_width = abs(left_eye.x - right_eye.x)
        
        # 判断逻辑
        orientation = 'unknown'
        confidence = 0.5
        approaching = False
        description = '未知'
        
        # 1. 判断侧身程度（肩膀宽度小于阈值说明侧身）
        is_side = shoulder_width < 0.15
        
        # 2. 判断正面/背面（通过鼻子位置和可见性）
        nose_visible = nose.visibility > 0.5
        eyes_visible = left_eye.visibility > 0.3 and right_eye.visibility > 0.3
        
        if is_side:
            # 侧身判断
            if left_shoulder.z < right_shoulder.z:
                orientation = 'side_right'
                description = '侧身向右'
            else:
                orientation = 'side_left'
                description = '侧身向左'
            confidence = 0.7 + (0.15 - shoulder_width) * 2  # 越窄置信度越高
        elif nose_visible and eyes_visible:
            # 正面判断
            # 鼻子在肩膀中心前方（z值更小）说明面向你
            if nose.z < shoulder_center_z - 0.05:
                orientation = 'facing_you'
                description = '面向你'
                confidence = min(0.9, 0.6 + nose.visibility * 0.3)
                approaching = True  # 面向你的人可能正在接近
            else:
                orientation = 'facing_away'
                description = '背对你'
                confidence = 0.7
        else:
            # 背面判断（鼻子不可见）
            orientation = 'facing_away'
            description = '背对你'
            confidence = 0.8
        
        return {
            'orientation': orientation,
            'confidence': confidence,
            'approaching': approaching,
            'description': description,
            'shoulder_width': shoulder_width,
            'nose_visible': nose_visible
        }
    except Exception as e:
        print(f"[姿态分析] 朝向分析失败: {e}")
        return None


def analyze_pose_abnormality(landmarks, image_width, image_height):
    """
    分析姿态异常（简化版：仅检测防御姿态）
    
    检测规则：
    - 防御姿态：双手护头
    
    注：为减少误报和杂乱播报，已禁用 fall/attack/fight 检测
    
    Args:
        landmarks: MediaPipe 的 pose_landmarks.landmark
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        dict: {
            'is_abnormal': bool,
            'abnormality_type': str,  # 'defense' | 'normal'
            'confidence': float,
            'description': str,
            'severity': int (1-3)  # 严重程度
        }
    """
    if landmarks is None:
        return None
    
    # 关键点索引
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    try:
        nose = landmarks[NOSE]
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        left_wrist = landmarks[LEFT_WRIST]
        right_wrist = landmarks[RIGHT_WRIST]
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        
        # 计算关键高度（y值，越小越高）
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        nose_y = nose.y
        left_wrist_y = left_wrist.y
        right_wrist_y = right_wrist.y
        
        # 身体垂直角度（肩膀到臀部的角度）
        body_angle = abs(shoulder_y - hip_y)
        
        abnormality_type = 'normal'
        confidence = 0.5
        description = '正常'
        severity = 0
        
        # === 仅检测防御姿态 ===
        
        # 防御姿态：双手在头部附近（双手护头）
        if (abs(left_wrist_y - nose_y) < 0.15 and abs(right_wrist_y - nose_y) < 0.15 and
              abs(left_wrist.x - nose.x) < 0.2 and abs(right_wrist.x - nose.x) < 0.2):
            abnormality_type = 'defense'
            description = '防御姿态'
            confidence = 0.7
            severity = 1
        
        return {
            'is_abnormal': abnormality_type != 'normal',
            'abnormality_type': abnormality_type,
            'confidence': confidence,
            'description': description,
            'severity': severity,
            'body_angle': body_angle
        }
    except Exception as e:
        print(f"[姿态分析] 异常检测失败: {e}")
        return None


# 骨架连接定义（用于手动绘制）
POSE_CONNECTIONS = [
    # 面部
    (0, 1), (1, 2), (2, 3), (3, 7),  # 左眼
    (0, 4), (4, 5), (5, 6), (6, 8),  # 右眼
    (9, 10),  # 嘴巴
    # 躯干
    (11, 12),  # 肩膀
    (11, 23), (12, 24),  # 肩到臀
    (23, 24),  # 臀部
    # 左臂
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),  # 左手
    # 右臂
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),  # 右手
    # 左腿
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # 右腿
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def draw_skeleton_manual(frame, landmarks, width, height, color=(0, 255, 0), thickness=2):
    """
    手动绘制骨架（用于新版 API 或旧版绘图不可用时）
    
    Args:
        frame: 图像帧
        landmarks: 关键点列表
        width: 图像宽度
        height: 图像高度
        color: 绘制颜色
        thickness: 线条粗细
    """
    # 绘制关键点
    for i, lm in enumerate(landmarks):
        x = int(lm.x * width)
        y = int(lm.y * height)
        visibility = getattr(lm, 'visibility', 0.9)
        
        if visibility > 0.5:
            cv2.circle(frame, (x, y), 4, color, -1)
    
    # 绘制连接线
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            start_vis = getattr(start_lm, 'visibility', 0.9)
            end_vis = getattr(end_lm, 'visibility', 0.9)
            
            if start_vis > 0.5 and end_vis > 0.5:
                start_point = (int(start_lm.x * width), int(start_lm.y * height))
                end_point = (int(end_lm.x * width), int(end_lm.y * height))
                cv2.line(frame, start_point, end_point, color, thickness)


def analyze_frame(frame):
    """
    综合分析单帧图像中的人体姿态
    
    Args:
        frame: BGR 格式的图像帧
    
    Returns:
        dict: {
            'detected': bool,           # 是否检测到人
            'pose_results': MediaPipe结果,
            'orientation': dict,        # 朝向分析结果
            'abnormality': dict,        # 异常检测结果
            'alert_level': int,         # 警戒等级 0-3
            'alert_message': str,       # 警示消息
            'latency_ms': float         # 处理延迟
        }
    """
    start_time = time.time()
    
    result = {
        'detected': False,
        'pose_results': None,
        'orientation': None,
        'abnormality': None,
        'alert_level': 0,
        'alert_message': '',
        'latency_ms': 0
    }
    
    # 获取姿态检测结果
    pose_results = get_pose_landmarks(frame)
    if pose_results is None or pose_results.pose_landmarks is None:
        result['latency_ms'] = (time.time() - start_time) * 1000
        return result
    
    result['detected'] = True
    result['pose_results'] = pose_results
    
    h, w = frame.shape[:2]
    landmarks = pose_results.pose_landmarks.landmark
    
    # 朝向分析
    orientation = analyze_person_orientation(landmarks, w, h)
    result['orientation'] = orientation
    
    # 异常检测
    abnormality = analyze_pose_abnormality(landmarks, w, h)
    result['abnormality'] = abnormality
    
    # 综合警戒等级评估
    alert_level = 0
    alert_messages = []
    
    # 异常姿态警戒
    if abnormality and abnormality['is_abnormal']:
        alert_level = max(alert_level, abnormality['severity'])
        alert_messages.append(abnormality['description'])
    
    # 面向接近警戒
    if orientation and orientation['approaching']:
        if alert_level < 1:
            alert_level = 1
        alert_messages.append(f"行人{orientation['description']}")
    
    result['alert_level'] = alert_level
    result['alert_message'] = '，'.join(alert_messages) if alert_messages else ''
    result['latency_ms'] = (time.time() - start_time) * 1000
    
    return result


def draw_pose_analysis(frame, analysis_result, draw_skeleton=True):
    """
    在图像上绘制姿态分析结果
    
    Args:
        frame: 要绘制的图像帧
        analysis_result: analyze_frame 的返回结果
        draw_skeleton: 是否绘制骨架
    
    Returns:
        frame: 绘制后的图像
    """
    if not analysis_result['detected']:
        return frame
    
    output = frame.copy()
    h, w = output.shape[:2]
    
    # 绘制骨架
    if draw_skeleton and analysis_result['pose_results']:
        pose_results = analysis_result['pose_results']
        
        if not USE_NEW_API and mp_drawing and mp_pose:
            # 旧版 API 使用内置绘图
            mp_drawing.draw_landmarks(
                output,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                )
            )
        elif pose_results.pose_landmarks:
            # 新版 API 或旧版绘图不可用时，手动绘制
            landmarks = pose_results.pose_landmarks.landmark
            draw_skeleton_manual(output, landmarks, w, h)
    
    # 绘制分析结果文字（使用中文）
    y_offset = 30
    
    # 朝向信息
    if analysis_result['orientation']:
        ori = analysis_result['orientation']
        color = (0, 255, 255) if ori['approaching'] else (0, 255, 0)
        # 中文显示朝向
        approaching_text = "正在接近" if ori['approaching'] else ""
        text = f"行人: {ori['description']} {approaching_text}"
        output = put_chinese_text(output, text, (10, y_offset), color, font_size=22)
        y_offset += 30
    
    # 异常信息
    if analysis_result['abnormality']:
        abn = analysis_result['abnormality']
        if abn['is_abnormal']:
            color = (0, 0, 255) if abn['severity'] >= 2 else (0, 165, 255)
            text = f"⚠ 警告: {abn['description']}"
            output = put_chinese_text(output, text, (10, y_offset), color, font_size=22)
            y_offset += 30
    
    # 警戒等级
    if analysis_result['alert_level'] > 0:
        level_colors = {1: (0, 255, 255), 2: (0, 165, 255), 3: (0, 0, 255)}
        level_names = {1: "低", 2: "中", 3: "高"}
        color = level_colors.get(analysis_result['alert_level'], (0, 0, 255))
        level_name = level_names.get(analysis_result['alert_level'], "未知")
        text = f"警戒等级: {level_name}"
        output = put_chinese_text(output, text, (10, y_offset), color, font_size=22)
    
    return output


# ============== 语音提示生成 ==============

def get_pose_voice_prompt(analysis_result):
    """
    根据姿态分析结果生成语音提示
    
    Args:
        analysis_result: analyze_frame 的返回结果
    
    Returns:
        str: 语音提示文本，如果无需提示则返回空字符串
    """
    if not analysis_result['detected']:
        return ''
    
    prompts = []
    
    # 异常姿态提示（仅防御姿态）
    abnormality = analysis_result.get('abnormality')
    if abnormality and abnormality['is_abnormal']:
        abnorm_type = abnormality['abnormality_type']
        
        # 只播报防御姿态，其他类型不播报（减少杂乱）
        if abnorm_type == 'defense':
            prompts.append('前方可能有冲突')
    
    # 朝向提示（只提示面向你的情况）
    orientation = analysis_result.get('orientation')
    if orientation and orientation['approaching'] and not prompts:
        # 只有没有异常时才提示朝向
        prompts.append('有行人朝你走来')
    
    return '，'.join(prompts)


# ============== 测试函数 ==============

def test_pose_detection():
    """测试姿态检测功能"""
    print("=" * 50)
    print("MediaPipe 姿态检测测试")
    print("=" * 50)
    
    if not init_mediapipe():
        print("初始化失败，请先安装 mediapipe: pip install mediapipe")
        return False
    
    # 尝试打开摄像头测试
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头，跳过实时测试")
        return True
    
    print("按 'q' 键退出测试...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 分析帧
        result = analyze_frame(frame)
        
        # 绘制结果
        output = draw_pose_analysis(frame, result)
        
        # 显示
        cv2.imshow('Pose Detection Test', output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("测试完成")
    return True


if __name__ == '__main__':
    test_pose_detection()
