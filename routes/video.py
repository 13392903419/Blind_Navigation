"""
视频相关路由 - 视频上传、处理、流式传输
异步架构：读帧线程 → frame_queue → 推理线程 → result_queue → 显示线程
"""
from flask import Blueprint, Response, request, jsonify
import cv2
import numpy as np
import ollama
import threading
import time
import os
import queue
import traceback
from werkzeug.utils import secure_filename

import functools
import torch

# 在加载模型前，强制将 torch.load 的默认参数 weights_only 设置为 False
torch.load = functools.partial(torch.load, weights_only=False)

video_bp = Blueprint('video', __name__)
# 允许加载 Ultralytics 的模型类
torch.serialization.add_safe_globals([
    'ultralytics.nn.tasks.DetectionModel',
    'ultralytics.nn.modules.container.Sequential',
    # 如果报错还提到其他类，继续往这里添加
])

from ultralytics import YOLO

from utils.decorators import login_required
from utils.video_utils import allowed_file, create_error_frame, create_info_frame
from utils.voice_utils import speak, speak_urgent, get_prompt_template, set_speech_complete_callback
from config import MODEL_WEIGHTS, UPLOAD_FOLDER, THRESHOLD_SLOPE, CALL_INTERVAL

# 加载YOLO模型 - 使用相对路径
# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 板块一：主模型使用的模型
# 盲道检测模型（用于主模型）
MAIN_BLIND_ROAD_MODEL_PATH = os.path.join(MODELS_DIR, 'blind_road_best.pt')
main_blind_road_model = None
main_blind_road_model_loaded = False

# 环境感知模型（用于主模型）
MAIN_ENVIRONMENT_MODEL_PATH = os.path.join(MODELS_DIR, 'environment_model.pt')
main_environment_model = None
main_environment_model_loaded = False

# 暴力行为检测模型（用于主模型）
MAIN_VIOLENCE_MODEL_PATH = os.path.join(MODELS_DIR, 'violence_model.pt')
main_violence_model = None
main_violence_model_loaded = False

# 板块二：独立子模型
# 模型一：盲道检测模型
BLIND_ROAD_MODEL_PATH = os.path.join(MODELS_DIR, 'blind_road_best.pt')
blind_road_model = None
blind_road_model_loaded = False

# 模型二：环境感知模型
ENVIRONMENT_MODEL_PATH = os.path.join(MODELS_DIR, 'environment_model.pt')
environment_model = None
environment_model_loaded = False

# 模型三：暴力行为检测模型
VIOLENCE_MODEL_PATH = os.path.join(MODELS_DIR, 'violence_model.pt')
violence_model = None
violence_model_loaded = False

# 加载板块一主模型的三个模型
try:
    if os.path.exists(MAIN_BLIND_ROAD_MODEL_PATH):
        main_blind_road_model = YOLO(MAIN_BLIND_ROAD_MODEL_PATH)
        main_blind_road_model_loaded = True
        print(f"[模型加载] ✓ 成功加载主模型-盲道检测: {MAIN_BLIND_ROAD_MODEL_PATH}")
    else:
        print(f"[模型加载] ⚠ 主模型-盲道检测模型不存在: {MAIN_BLIND_ROAD_MODEL_PATH}")
except Exception as e:
    print(f"[模型加载] ❌ 加载主模型-盲道检测失败: {e}")
    main_blind_road_model = None
    main_blind_road_model_loaded = False

try:
    if os.path.exists(MAIN_ENVIRONMENT_MODEL_PATH):
        main_environment_model = YOLO(MAIN_ENVIRONMENT_MODEL_PATH)
        main_environment_model_loaded = True
        print(f"[模型加载] ✓ 成功加载主模型-环境感知: {MAIN_ENVIRONMENT_MODEL_PATH}")
    else:
        print(f"[模型加载] ⚠ 主模型-环境感知模型不存在: {MAIN_ENVIRONMENT_MODEL_PATH}")
except Exception as e:
    print(f"[模型加载] ❌ 加载主模型-环境感知失败: {e}")
    main_environment_model = None
    main_environment_model_loaded = False

try:
    if os.path.exists(MAIN_VIOLENCE_MODEL_PATH):
        main_violence_model = YOLO(MAIN_VIOLENCE_MODEL_PATH)
        main_violence_model_loaded = True
        print(f"[模型加载] ✓ 成功加载主模型-暴力行为检测: {MAIN_VIOLENCE_MODEL_PATH}")
    else:
        print(f"[模型加载] ⚠ 主模型-暴力行为检测模型不存在: {MAIN_VIOLENCE_MODEL_PATH}")
except Exception as e:
    print(f"[模型加载] ❌ 加载主模型-暴力行为检测失败: {e}")
    main_violence_model = None
    main_violence_model_loaded = False

if main_blind_road_model_loaded and main_environment_model_loaded and main_violence_model_loaded:
    print(f"[模型加载] ✓ 板块一主模型（级联推理）已启用（盲道检测 + 环境感知 + 暴力行为检测）")

# 加载板块二独立子模型
try:
    if os.path.exists(BLIND_ROAD_MODEL_PATH):
        blind_road_model = YOLO(BLIND_ROAD_MODEL_PATH)
        blind_road_model_loaded = True
        print(f"[模型加载] ✓ 成功加载板块二-模型一（盲道检测）: {BLIND_ROAD_MODEL_PATH}")
    else:
        print(f"[模型加载] ⚠ 板块二-模型一（盲道检测）不存在: {BLIND_ROAD_MODEL_PATH}")
except Exception as e:
    print(f"[模型加载] ❌ 加载板块二-模型一（盲道检测）失败: {e}")
    blind_road_model = None
    blind_road_model_loaded = False

try:
    if os.path.exists(ENVIRONMENT_MODEL_PATH):
        environment_model = YOLO(ENVIRONMENT_MODEL_PATH)
        environment_model_loaded = True
        print(f"[模型加载] ✓ 成功加载板块二-模型二（环境感知）: {ENVIRONMENT_MODEL_PATH}")
    else:
        print(f"[模型加载] ⚠ 板块二-模型二（环境感知）不存在: {ENVIRONMENT_MODEL_PATH}")
except Exception as e:
    print(f"[模型加载] ❌ 加载板块二-模型二（环境感知）失败: {e}")
    environment_model = None
    environment_model_loaded = False

try:
    if os.path.exists(VIOLENCE_MODEL_PATH):
        violence_model = YOLO(VIOLENCE_MODEL_PATH)
        violence_model_loaded = True
        print(f"[模型加载] ✓ 成功加载板块二-模型三（暴力行为检测）: {VIOLENCE_MODEL_PATH}")
    else:
        print(f"[模型加载] ⚠ 板块二-模型三（暴力行为检测）不存在: {VIOLENCE_MODEL_PATH}")
except Exception as e:
    print(f"[模型加载] ❌ 加载板块二-模型三（暴力行为检测）失败: {e}")
    violence_model = None
    violence_model_loaded = False

# 兼容性：保留单模型模式的model变量
model = main_blind_road_model if main_blind_road_model_loaded else blind_road_model
# 加载YOLO模型并优化性能

# 检测GPU并启用加速
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    model.to(device)
    print("[YOLO] 使用设备: CUDA (GPU), FP32推理")
    # 注意：FP16在某些GPU/环境下fuse时会出现dtype错误，因此默认使用FP32
    # 如需启用FP16，请在推理参数中使用 half=True，而不是在初始化时转换
else:
    # CPU模式，确保使用FP32
    model.model.float()
    print("[YOLO] 使用设备: CPU, FP32推理")

# 禁用代理环境变量，避免代理干扰本地 Ollama 连接
# 保存原有代理设置
_original_http_proxy = os.environ.get('HTTP_PROXY', None)
_original_https_proxy = os.environ.get('HTTPS_PROXY', None)
_original_http_proxy_lower = os.environ.get('http_proxy', None)
_original_https_proxy_lower = os.environ.get('https_proxy', None)

# 临时移除代理设置（仅对 Ollama 连接）
if 'HTTP_PROXY' in os.environ:
    del os.environ['HTTP_PROXY']
if 'HTTPS_PROXY' in os.environ:
    del os.environ['HTTPS_PROXY']
if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']

# 创建 Ollama 客户端（此时已无代理干扰）
ollama_client = ollama.Client(host='http://localhost:11434')

# 恢复原有代理设置（不影响其他HTTP请求，如DeepSeek API）
if _original_http_proxy is not None:
    os.environ['HTTP_PROXY'] = _original_http_proxy
if _original_https_proxy is not None:
    os.environ['HTTPS_PROXY'] = _original_https_proxy
if _original_http_proxy_lower is not None:
    os.environ['http_proxy'] = _original_http_proxy_lower
if _original_https_proxy_lower is not None:
    os.environ['https_proxy'] = _original_https_proxy_lower

# 全局变量
current_video_path = None
video_active = False
last_call_time = 0
current_speech_text = ""

# === 音画同步状态 ===
speech_sync_state = {
    'is_speaking': False,       # 是否正在播报
    'trigger_frame_id': 0,      # 触发语音的帧ID
    'speech_text': '',          # 当前语音内容
    'speech_start_time': 0,     # 语音开始时间
    'speech_duration': 2.0,     # 预估语音持续时间（秒）
    'freeze_frame': None,       # 冻结帧（用于前端暂停时显示）
}
speech_sync_lock = threading.Lock()
# 注意：语音完成回调在 clear_speech_sync_state 定义之后注册（见下方）

# 语音播报优化 - 混合架构（静态优先 + 大模型缓存）
SPEECH_COOLDOWN = 2.0  # 普通语音播报冷却时间（秒）
FIGHT_COOLDOWN = 1.0   # fight紧急播报冷却时间（秒）- 更短以保证及时响应
last_speech_time = 0  # 最后一次语音播报的时间
last_fight_urgent_time = 0  # 最后一次fight紧急播报的时间
last_direction = None  # 最后一次播报的方向（用于去重）
last_obstacle_type = None  # 最后一次播报的障碍物类型（用于去重）
# 音画对齐基准延迟（秒），动态自适应时作为下限
ALIGN_BASE_DELAY = 0.4
voice_latency_avg = 0.4  # 语音生成/入队平均耗时估计（秒）
last_speech_ready_time = 0  # 最近一次语音预计“就绪”时间，用于对齐帧

# === 混合语音策略配置 ===
USE_LLM_FOR_COMPLEX = True   # 是否对复杂场景启用大模型
LLM_TIMEOUT = 0.8            # 大模型超时时间（秒）
LLM_CACHE_SIZE = 100         # 缓存大小
LLM_MIN_OBSTACLES = 2        # 触发大模型的最小障碍物数量

import random
import concurrent.futures
from collections import OrderedDict
random.seed()

# 静态提示词库（高频/紧急场景，零延迟）
PROMPT_LIBRARY = {
    # 方向提示（最高优先级，必须快）
    'direction': {
        'left': '左转',
        'right': '右转',
        'straight': '直行',
        'change': '盲道方向变化'
    },
    # 障碍物提示（对应环境感知模型的5种类别）
    # 0:Automobile 1:Obstacle 2:Person 3:Road 4:Sidewalk
    'obstacles': {
        'automobile': '注意车辆',
        'obstacle': '有障碍物',
        'person': '有行人',
        'road': '前方是道路',
        'sidewalk': '前方是人行道',
        'default': '有障碍'
    },
    # 紧急提示（最高优先级，立即播报）
    'urgent': {
        'danger': '停！危险',
        'collision': '小心碰撞',
        'fight': '注意安全，远离危险'
    },
    # 位置修饰词
    'position': {
        'left': '左侧',
        'right': '右侧',
        'front': '前方'
    },
    # 鼓励词（低优先级）
    'encouragement': [
        '很好，继续',
        '走得不错',
        '加油',
        '状态不错'
    ]
}

# LLM 生成结果缓存（LRU策略）
class LRUCache(OrderedDict):
    def __init__(self, maxsize=100):
        super().__init__()
        self.maxsize = maxsize
    
    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default
    
    def set(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)

llm_cache = LRUCache(maxsize=LLM_CACHE_SIZE)

# 预生成的常见场景缓存（启动时填充）
PREGENERATED_PROMPTS = {
    ('left', 'person'): '左转，注意避让前方行人',
    ('right', 'person'): '右转，注意避让前方行人',
    ('straight', 'person'): '直行，前方有行人请注意',
    ('left', 'stairs'): '左转后有台阶，请小心',
    ('right', 'stairs'): '右转后有台阶，请小心',
    ('straight', 'stairs'): '直行，前方有台阶',
    ('left', 'vehicle'): '左转，注意左侧车辆',
    ('right', 'vehicle'): '右转，注意右侧车辆',
    ('straight', 'construction'): '直行，前方有施工请绕行',
}

# 性能统计变量 - 为每个模型独立存储
main_model_stats = {
    'fps': 0,
    'latency': 0,
    'confidence': 0,
    'last_update': 0
}

blind_road_model_stats = {
    'fps': 0,
    'latency': 0,
    'confidence': 0,
    'last_update': 0
}

environment_model_stats = {
    'fps': 0,
    'latency': 0,
    'confidence': 0,
    'last_update': 0
}

violence_model_stats = {
    'fps': 0,
    'latency': 0,
    'confidence': 0,
    'last_update': 0
}

# 兼容性：保留原有的model_stats
model_stats = main_model_stats
# 新增：tag->stats字典映射，便于异步推理线程写入
stats_map = {
    'blind_road': blind_road_model_stats,
    'environment': environment_model_stats,
    'violence': violence_model_stats
}

def update_stats(tag, fps, latency_ms, confidence):
    """写入各模型实时性能数据"""
    d = stats_map[tag]
    d['fps'] = int(fps)
    d['latency'] = int(latency_ms)
    d['confidence'] = int(confidence)
    d['last_update'] = time.time()

frame_times = []  # 存储最近的帧时间戳
max_frame_history = 30  # 保留最近30帧的数据

# ==================== 异步架构：线程和队列 =============# 线程间通信队列
frame_queue = queue.Queue(maxsize=20)
# 新增三个模型专用队列（5帧缓冲即可）
blind_queue = queue.Queue(maxsize=5)
env_queue = queue.Queue(maxsize=5)
vio_queue = queue.Queue(maxsize=5)    # 读帧线程 → 推理线程
# ---------- Combiner 额外队列 ----------
from collections import defaultdict

combined_queue = queue.Queue(maxsize=30)   # 三模型结果汇聚→合成线程
combine_buffer = defaultdict(dict)         # 临时缓存同一帧三模型结果
# ---------------------------------------
result_queue = queue.Queue(maxsize=3)   # 推理线程 → 显示线程

# 三个子模块的显示队列（推理线程直接产出已绘制帧，供端点消费）
blind_display_queue = queue.Queue(maxsize=3)
env_display_queue   = queue.Queue(maxsize=3)
vio_display_queue   = queue.Queue(maxsize=3)

# 线程控制字典
thread_control = {
    'running': False,           # 主开关：控制所有线程运行
    'reader_alive': False,      # 读帧线程存活标志
    'inference_alive': False,   # 推理线程存活标志
    'error': None,              # 错误信息
    'lock': threading.Lock(),   # 全局锁（用于线程安全操作）
    'video_cap': None          # 视频捕获对象（在读帧线程中使用）
}

# 线程对象引用
reader_thread = None
inference_thread = None

# 性能监控指标
performance_metrics = {
    'frame_queue_size': 0,      # 帧队列积压数量
    'result_queue_size': 0,     # 结果队列积压数量
    'reader_fps': 0,            # 读帧速度
    'inference_fps': 0,         # 推理速度
    'frame_drop_count': 0,      # 丢帧统计
    'last_update': 0
}

# 受控抽帧策略：每 N 帧推理一次，其余帧复用上次结果
INFER_EVERY_N = 2  # 可调：2 表示每隔一帧推理一次
# 记录各模型最近一次推理的输出，供跳帧时复用
last_results = {
    'blind_road': None,
    'environment': None,
    'violence': None,
}

# 转向提示问题
right_turn_question = "请用亲切且简短的话语告知要往右拐，因为盲道是往右拐的"
left_turn_question = "请用亲切且简短的话语告知要往左拐，因为盲道是往左拐的"

# 视频处理优化参数
MAX_FRAME_SIZE = 640  # 最长边缩放到640像素

# ==================== 辅助函数 ====================

def clear_queue(q):
    """清空队列"""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            # 如果系统仍在运行则继续等待，避免生成器提前退出
            if thread_control.get('running', False):
                continue
            else:
                time.sleep(0.5)
                continue
            break

def resize_frame(frame, max_size=MAX_FRAME_SIZE):
    """等比例缩放帧，保持最长边不超过max_size"""
    h, w = frame.shape[:2]
    if max(h, w) <= max_size:
        return frame, 1.0  # 无需缩放
    
    # 计算缩放比例
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 使用INTER_AREA进行下采样（质量更好）
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


# ============== 通用推理线程函数 (Patch B) ==============

def push_inference(in_q, model, tag, color):
    """从专用队列取帧→（受控抽帧）推理→写入合成队列→更新stats"""
    fps_buf = []
    while thread_control['running']:
        try:
            item = in_q.get(timeout=1.0)
        except queue.Empty:
            if not thread_control['reader_alive']:
                break
            continue

        frame = item['frame']
        fid = item.get('fid', 0)

        # 受控抽帧：只有当满足条件时才进行推理，否则复用最近结果
        do_infer = (fid % INFER_EVERY_N == 0) or (last_results.get(tag) is None)
        t0 = time.time() if do_infer else None
        results = model.predict(frame, verbose=False) if do_infer else None
        if tag == 'violence' and results:
            print('[DEBUG-push]', 'cls:', [int(b.cls[0]) for b in results[0].boxes], 'names:', getattr(results[0], 'names', {}))
        latency = (time.time() - t0) * 1000 if do_infer else 0

        # 计算/复用结果
        out_boxes = None
        out_names = {}
        confs = []
        if results:
            out_boxes = results[0].boxes
            out_names = getattr(results[0], 'names', {})
            for box in out_boxes:
                confs.append(float(box.conf[0]))
            # 记录最近结果供跳帧复用
            last_results[tag] = {
                'boxes': out_boxes,
                'names': out_names,
            }
        else:
            # 复用上次推理结果
            lr = last_results.get(tag)
            if lr:
                out_boxes = lr.get('boxes', None)
                out_names = lr.get('names', {})

        # 为子模块显示队列生成已绘制帧
        annotated = frame.copy()
        if out_boxes is not None:
            if tag == 'blind_road':
                for box in out_boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"blind:{conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            elif tag == 'environment':
                for box in out_boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = out_names.get(cls, 'obj')
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(annotated, f"{label}:{conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            elif tag == 'violence':
                for box in out_boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name_map = out_names or {}
                    cls_name = (name_map.get(cls_id, '') or '').lower()
                    is_fight = (cls_name == 'fight') if cls_name else (cls_id == 1)
                    color_v = (0, 255, 255) if is_fight else (0, 128, 255)
                    label = f"{'FIGHT' if is_fight else 'NOFIGHT'}:{conf:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color_v, 3)
                    cv2.putText(annotated, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_v, 2)

        # FPS 计算
        fps_buf.append(time.time())
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps = len(fps_buf)/(fps_buf[-1]-fps_buf[0]) if len(fps_buf)>=2 else 0

        update_stats(tag, fps, latency, np.mean(confs)*100 if confs else 0)
        # ---------- 将结果发送给 combiner_worker ----------
        try:
            combined_queue.put_nowait({
                'fid': fid,
                'frame': frame,               # 原图副本
                'tag': tag,                   # blind_road / environment / violence
                'boxes': out_boxes,
                'names': out_names
            })
        except queue.Full:
            pass

        # ---------- 推送到对应显示队列（供子模块端点消费） ----------
        try:
            pkt = {'frame': annotated}
            if tag == 'blind_road':
                blind_display_queue.put_nowait(pkt)
            elif tag == 'environment':
                env_display_queue.put_nowait(pkt)
            elif tag == 'violence':
                vio_display_queue.put_nowait(pkt)
        except queue.Full:
            pass

# =============== 语音播报辅助函数 ===============
# =============== 混合语音生成函数 ===============

def get_static_prompt(direction, obstacle_type=None, position=None):
    """
    获取静态提示词（零延迟）
    Args:
        direction: 方向 'left', 'right', 'straight', 'change'
        obstacle_type: 障碍物类型（可选）
        position: 障碍物位置（可选）
    Returns:
        str: 静态提示文本
    """
    parts = []
    
    # 方向提示
    dir_text = PROMPT_LIBRARY['direction'].get(direction, '')
    if dir_text:
        parts.append(dir_text)
    
    # 障碍物提示
    if obstacle_type:
        obs_text = PROMPT_LIBRARY['obstacles'].get(obstacle_type, PROMPT_LIBRARY['obstacles']['default'])
        if position:
            pos_text = PROMPT_LIBRARY['position'].get(position, '')
            if pos_text:
                obs_text = f"{pos_text}{obs_text}"
        parts.append(obs_text)
    
    return '，'.join(parts) if parts else '请注意盲道'


def get_urgent_prompt(urgent_type):
    """
    获取紧急提示词（最高优先级，立即返回）
    Args:
        urgent_type: 紧急类型 'danger', 'collision', 'fight'
    Returns:
        str: 紧急提示文本
    """
    return PROMPT_LIBRARY['urgent'].get(urgent_type, '注意安全')


def get_encouragement_prompt():
    """
    返回随机鼓励词（30%概率）
    Returns:
        str: 鼓励文本或空字符串
    """
    if random.random() < 0.3:
        return random.choice(PROMPT_LIBRARY['encouragement'])
    return ""


def call_llm_for_prompt(direction, obstacles, position, user_settings):
    """
    调用大模型生成自然语言提示（用于复杂场景）
    Args:
        direction: 方向
        obstacles: 障碍物列表
        position: 位置信息
        user_settings: 用户设置
    Returns:
        str: 大模型生成的提示文本
    """
    try:
        # 构建简洁的提示
        context = f"方向:{direction}"
        if obstacles:
            context += f"，障碍物:{','.join(obstacles)}"
        if position:
            context += f"，位置:{position}"
        
        prompt = f"""你是盲人导航助手，用一句简短的话（不超过15字）提示用户：
{context}
要求：直接说提示内容，不要解释，语气温和。"""
        
        response = ollama_client.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        result = response.get('message', {}).get('content', '').strip()
        # 限制长度，避免过长
        if result and len(result) <= 30:
            return result
        return None
    except Exception as e:
        print(f"[LLM] 调用失败: {e}")
        return None


def async_generate_and_cache(cache_key, direction, obstacles, position, user_settings):
    """
    异步调用大模型并缓存结果（后台线程）
    """
    try:
        result = call_llm_for_prompt(direction, obstacles, position, user_settings)
        if result:
            llm_cache.set(cache_key, result)
            print(f"[LLM缓存] 已缓存: {cache_key} -> {result}")
    except Exception as e:
        print(f"[LLM缓存] 异步生成失败: {e}")


def get_smart_prompt(direction, obstacles=None, position=None, urgency='normal', user_settings=None):
    """
    智能混合语音生成（核心函数）
    策略：紧急→静态 | 简单→静态 | 复杂→缓存/大模型/降级静态
    
    Args:
        direction: 方向 'left', 'right', 'straight', 'change'
        obstacles: 障碍物列表，如 ['person', 'stairs']
        position: 位置 'left', 'right', 'front'
        urgency: 紧急程度 'urgent', 'normal', 'low'
        user_settings: 用户设置
    Returns:
        str: 最终的语音提示文本
    """
    obstacles = obstacles or []
    
    # === 1. 紧急情况：立即返回静态提示 ===
    if urgency == 'urgent' or direction in ['danger', 'stop', 'fight']:
        return get_urgent_prompt(direction)
    
    # === 2. 简单场景：直接使用静态提示 ===
    if len(obstacles) <= 1:
        prompt = get_static_prompt(direction, obstacles[0] if obstacles else None, position)
        return prompt
    
    # === 3. 复杂场景：尝试缓存 → 大模型 → 降级静态 ===
    if USE_LLM_FOR_COMPLEX and len(obstacles) >= LLM_MIN_OBSTACLES:
        # 构建缓存键
        cache_key = (direction, tuple(sorted(obstacles[:3])))  # 最多取3个障碍物
        
        # 3.1 检查预生成缓存
        if len(obstacles) == 1:
            pregen_key = (direction, obstacles[0])
            if pregen_key in PREGENERATED_PROMPTS:
                return PREGENERATED_PROMPTS[pregen_key]
        
        # 3.2 检查LLM动态缓存
        cached = llm_cache.get(cache_key)
        if cached:
            print(f"[LLM缓存] 命中: {cache_key}")
            return cached
        
        # 3.3 尝试带超时的大模型调用
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    call_llm_for_prompt, 
                    direction, obstacles, position, user_settings or {}
                )
                result = future.result(timeout=LLM_TIMEOUT)
                if result:
                    # 缓存结果
                    llm_cache.set(cache_key, result)
                    return result
        except concurrent.futures.TimeoutError:
            print(f"[LLM] 超时({LLM_TIMEOUT}s)，降级为静态提示")
            # 异步继续生成并缓存，下次使用
            threading.Thread(
                target=async_generate_and_cache,
                args=(cache_key, direction, obstacles, position, user_settings or {}),
                daemon=True
            ).start()
        except Exception as e:
            print(f"[LLM] 调用异常: {e}")
    
    # === 4. 降级：静态组合 ===
    parts = [PROMPT_LIBRARY['direction'].get(direction, '')]
    for obs in obstacles[:2]:  # 最多播报2个障碍物
        obs_text = PROMPT_LIBRARY['obstacles'].get(obs, '')
        if obs_text:
            parts.append(obs_text)
    
    return '，'.join(filter(None, parts)) or '请注意周围环境'


# 兼容旧接口
def get_blind_road_prompt(direction):
    """兼容旧接口：获取盲道方向提示"""
    return get_static_prompt(direction)

def get_obstacle_prompt(obstacle_type):
    """兼容旧接口：获取障碍物提示"""
    return PROMPT_LIBRARY['obstacles'].get(obstacle_type, PROMPT_LIBRARY['obstacles']['default'])

def should_speak_now(current_time):
    """
    检查是否应该播报（冷却检查）
    Args:
        current_time: 当前时间戳
    Returns:
        bool: 是否应该播报
    """
    global last_speech_time
    if current_time - last_speech_time >= SPEECH_COOLDOWN:
        last_speech_time = current_time
        return True
    return False

def check_direction_change(new_direction):
    """
    检查方向是否变化（去重）
    Args:
        new_direction: 新的方向值
    Returns:
        bool: 方向是否有变化
    """
    global last_direction
    if new_direction != last_direction:
        last_direction = new_direction
        return True
    return False

def check_obstacle_change(obstacle_type):
    """
    检查障碍物类型是否变化（去重）
    Args:
        obstacle_type: 新的障碍物类型
    Returns:
        bool: 障碍物类型是否有变化
    """
    global last_obstacle_type
    if obstacle_type != last_obstacle_type:
        last_obstacle_type = obstacle_type
        return True
    return False

# =============== Combiner 线程 ===============
# 全局变量：跟踪各检测类型上次提示的时间（避免重复提示）
last_fight_speak_time = 0
last_obstacle_speak_time = 0
last_blind_road_speak_time = 0

def set_speech_sync_state(frame_id, frame, speech_text, is_urgent=False):
    """
    设置语音同步状态，用于前端音画同步
    Args:
        frame_id: 触发语音的帧ID
        frame: 触发语音时的帧画面（用于冻结显示）
        speech_text: 语音内容
        is_urgent: 是否紧急播报
    """
    global speech_sync_state
    # 根据文本长度估算语音持续时间（约每字0.3秒）
    text_len = len(speech_text) if speech_text else 0
    duration = max(1.5, min(text_len * 0.25, 5.0))  # 1.5~5秒
    if is_urgent:
        duration = max(1.0, duration * 0.7)  # 紧急播报更快
    
    with speech_sync_lock:
        speech_sync_state['is_speaking'] = True
        speech_sync_state['trigger_frame_id'] = frame_id
        speech_sync_state['speech_text'] = speech_text
        speech_sync_state['speech_start_time'] = time.time()
        speech_sync_state['speech_duration'] = duration
        # 保存冻结帧的JPEG编码
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                speech_sync_state['freeze_frame'] = buffer.tobytes()
    
    print(f"[音画同步] 设置语音状态: frame_id={frame_id}, duration={duration:.1f}s, text='{speech_text}'")

def clear_speech_sync_state():
    """清除语音同步状态（语音播放完成后调用）"""
    global speech_sync_state
    with speech_sync_lock:
        speech_sync_state['is_speaking'] = False
        speech_sync_state['freeze_frame'] = None
    print("[音画同步] 语音播放完成，清除同步状态")

# 注册语音完成回调 - 语音播放完成后自动清除同步状态
set_speech_complete_callback(clear_speech_sync_state)

def combiner_worker():
    """
    收集三模型结果，同帧绘制后写入 result_queue
    并根据检测结果输出相应的语音提示
    """
    global last_fight_speak_time, last_obstacle_speak_time, last_blind_road_speak_time, current_speech_text
    global last_direction, last_obstacle_type, last_speech_time, last_speech_ready_time, voice_latency_avg
    global last_fight_urgent_time  # fight紧急播报专用时间戳

    # 各类型的提示间隔（秒）
    FIGHT_SPEAK_INTERVAL = 2.0
    OBSTACLE_SPEAK_INTERVAL = 2.0
    BLIND_ROAD_SPEAK_INTERVAL = 2.0
    
    # === 危险状态追踪（用于危险消失后重新提示盲道方向）===
    was_fight_detected = False      # 上一帧是否检测到打架
    was_obstacle_detected = False   # 上一帧是否检测到盲道上的障碍物
    current_blind_direction = 'straight'  # 当前盲道方向（用于危险消失后重新提示）
    DANGER_CLEAR_COOLDOWN = 3.0     # 危险消失后的提示冷却时间（秒）- 增加到3秒减少频繁播报
    last_danger_clear_time = 0      # 上次危险消失提示的时间
    
    # === 智能播报状态追踪 ===
    obstacle_alert_triggered = False   # 是否已经对当前障碍物播报过（用于判断是否需要播报"已通过"）
    last_obstacle_alert_time = 0       # 上次障碍物警示播报的时间
    consecutive_obstacle_frames = 0    # 连续检测到盲道上障碍物的帧数
    consecutive_clear_frames = 0       # 连续没有检测到盲道上障碍物的帧数
    MIN_OBSTACLE_FRAMES = 3            # 至少连续N帧检测到障碍物才播报（避免误检）
    MIN_CLEAR_FRAMES = 5               # 至少连续N帧没有障碍物才播报"已通过"（避免闪烁）

    def update_voice_ready(start_ts: float):
        """更新语音耗时估计与预计就绪时间"""
        global voice_latency_avg, last_speech_ready_time
        duration = time.time() - start_ts
        voice_latency_avg = 0.7 * voice_latency_avg + 0.3 * duration
        last_speech_ready_time = time.time() + max(ALIGN_BASE_DELAY, voice_latency_avg)

    while thread_control['running']:
        try:
            res = combined_queue.get(timeout=1.0)
        except queue.Empty:
            # 读帧线程已死且队列空 → 退出
            if not thread_control['reader_alive']:
                break
            continue

        fid = res['fid']
        buf = combine_buffer[fid]
        buf.setdefault('frame', res['frame'])
        buf[res['tag']] = res

        # 三模型结果到齐
        if all(k in buf for k in ('blind_road', 'environment', 'violence')):
            frame = buf['frame']
            current_time = time.time()

            # 盲道绘制
            b_res = buf['blind_road']
            blind_detected = False
            blind_confidences = []
            if b_res['boxes'] is not None and len(b_res['boxes']) > 0:
                for box in b_res['boxes']:
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    blind_confidences.append(conf)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,f"blind:{conf:.2f}",(x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                blind_detected = True

            # 检测到盲道且置信度高于0.5时，触发语音提示
            if blind_detected and blind_confidences and max(blind_confidences) > 0.5:
                # 计算盲道方向：基于检测框中心点拟合斜率
                centers = []
                for box in b_res['boxes']:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                    center_x = (bx1 + bx2) / 2
                    center_y = (by1 + by2) / 2
                    centers.append((center_x, center_y))
                
                # 根据检测框数量和斜率判断方向
                detected_direction = 'straight'  # 默认直行
                if len(centers) >= 2:
                    # 使用线性拟合计算斜率：y坐标作为自变量，x坐标作为因变量
                    ys = np.array([c[1] for c in centers])
                    xs = np.array([c[0] for c in centers])
                    try:
                        slope, _ = np.polyfit(ys, xs, 1)
                        # 斜率阈值判断（使用config中的THRESHOLD_SLOPE）
                        if slope < -THRESHOLD_SLOPE:
                            detected_direction = 'left'
                            print(f"[盲道方向] 检测到左转，斜率: {slope:.3f}")
                        elif slope > THRESHOLD_SLOPE:
                            detected_direction = 'right'
                            print(f"[盲道方向] 检测到右转，斜率: {slope:.3f}")
                        else:
                            print(f"[盲道方向] 直行，斜率: {slope:.3f}")
                    except Exception as e:
                        print(f"[盲道方向] 斜率计算失败: {e}")
                
                # 保存当前盲道方向（用于危险消失后重新提示）
                current_blind_direction = detected_direction
                
                if should_speak_now(current_time):
                    if check_direction_change(detected_direction):
                        try:
                            user_settings = get_user_settings_for_video()
                            speech_content = get_blind_road_prompt(detected_direction)
                            # 30% 概率添加鼓励词
                            encouragement = get_encouragement_prompt()
                            if encouragement:
                                speech_content += "，" + encouragement
                            current_speech_text = speech_content
                            # 设置音画同步状态
                            set_speech_sync_state(fid, frame, speech_content, is_urgent=False)
                            _ts = time.time()
                            speak(speech_content, user_settings)
                            update_voice_ready(_ts)
                            print(f"[语音提示] 盲道检测 - {speech_content}")
                        except Exception as e:
                            print(f"[语音提示] 盲道语音播放错误: {e}")
                    last_blind_road_speak_time = current_time

            # 环境感知绘制
            e_res = buf['environment']
            obstacle_detected_on_path = False  # 仅记录盲道上的障碍物用于语音
            obstacle_positions = []  # 记录障碍物位置信息 (位置, 置信度, 类型)
            # 环境感知模型类别映射: 0:Automobile 1:Obstacle 2:Person 3:Road 4:Sidewalk
            ENV_CLASS_MAP = {
                0: 'automobile',
                1: 'obstacle',
                2: 'person',
                3: 'road',
                4: 'sidewalk'
            }
            # 需要提示的障碍物类别（排除road和sidewalk，这些不是障碍）
            ALERT_CLASSES = {0, 1, 2}  # automobile, obstacle, person

            # 计算盲道水平范围，用于过滤盲道以外的障碍物（如两侧树木）
            blind_lane_range = None
            if blind_detected and b_res['boxes'] is not None and len(b_res['boxes']) > 0:
                xs = []
                for box in b_res['boxes']:
                    bx1, _, bx2, _ = box.xyxy[0].cpu().numpy().astype(int)
                    xs.extend([bx1, bx2])
                if xs:
                    lane_min, lane_max = min(xs), max(xs)
                    lane_width = lane_max - lane_min
                    # 给盲道边缘留一点缓冲，避免过于敏感
                    margin = int(lane_width * 0.1)
                    blind_lane_range = (max(0, lane_min - margin), min(frame.shape[1], lane_max + margin))
            
            if e_res['boxes'] is not None:
                for box in e_res['boxes']:
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    # 计算障碍物在图像中的位置（左、中、右）
                    img_width = frame.shape[1]
                    box_center_x = (x1 + x2) / 2
                    if box_center_x < img_width / 3:
                        position = "左侧"
                    elif box_center_x > 2 * img_width / 3:
                        position = "右侧"
                    else:
                        position = "前方"

                    # 获取类别名称
                    cls_name = ENV_CLASS_MAP.get(cls, 'obstacle')
                    
                    # 只有需要警示的类别且位于盲道范围内才记录
                    on_blind_lane = False
                    if blind_lane_range:
                        on_blind_lane = blind_lane_range[0] <= box_center_x <= blind_lane_range[1]
                    if cls in ALERT_CLASSES and on_blind_lane:
                        obstacle_positions.append((position, conf, cls_name))
                        obstacle_detected_on_path = True

                    # 绘制检测框
                    if blind_detected and cls in ALERT_CLASSES:
                        label = cls_name
                        color = (0,0,255)  # 红色表示障碍
                    else:
                        label = cls_name
                        color = (255,0,0) if cls in ALERT_CLASSES else (0,128,0)  # 蓝色障碍，绿色安全区域
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.putText(frame,f"{label}:{conf:.2f}",(x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            # === 智能障碍物播报逻辑 ===
            # 只有盲道范围内的障碍物才影响行走，盲道外的障碍物（如两侧的树）不触发语音
            
            # 更新连续帧数统计
            if obstacle_detected_on_path and obstacle_positions:
                consecutive_obstacle_frames += 1
                consecutive_clear_frames = 0
            else:
                consecutive_clear_frames += 1
                consecutive_obstacle_frames = 0
            
            # 检测到盲道上的障碍物且置信度高于0.5时，触发语音提示
            # 需要连续至少 MIN_OBSTACLE_FRAMES 帧检测到才播报（避免单帧误检）
            if obstacle_detected_on_path and obstacle_positions and consecutive_obstacle_frames >= MIN_OBSTACLE_FRAMES:
                # 找到置信度最高的障碍物
                best_obstacle = max(obstacle_positions, key=lambda x: x[1])
                if best_obstacle[1] > 0.5:
                    if should_speak_now(current_time):
                        # 获取最高置信度障碍物的类型
                        obstacle_types = [best_obstacle[2]]  # 使用实际检测到的类型
                        position = best_obstacle[0]  # 位置：左侧/右侧/前方
                        # 将中文位置映射为英文key
                        pos_map = {'左侧': 'left', '右侧': 'right', '前方': 'front'}
                        pos_key = pos_map.get(position, 'front')
                        
                        # 检查障碍物类型变化
                        obstacle_key = f"{obstacle_types[0]}_{pos_key}"
                        if check_obstacle_change(obstacle_key):
                            try:
                                user_settings = get_user_settings_for_video()
                                # 使用混合架构生成语音（多障碍物时可触发大模型）
                                speech_content = get_smart_prompt(
                                    direction='straight',
                                    obstacles=obstacle_types,
                                    position=pos_key,
                                    urgency='normal',
                                    user_settings=user_settings
                                )
                                current_speech_text = speech_content
                                # 设置音画同步状态
                                set_speech_sync_state(fid, frame, speech_content, is_urgent=False)
                                _ts = time.time()
                                speak(speech_content, user_settings)
                                update_voice_ready(_ts)
                                print(f"[语音提示] 盲道上障碍物 - {speech_content}")
                                last_obstacle_speak_time = current_time
                                # 标记已对此障碍物播报过警示
                                obstacle_alert_triggered = True
                                last_obstacle_alert_time = current_time
                            except Exception as e:
                                print(f"[语音提示] 障碍物语音播放错误: {e}")

            # 暴力绘制
            v_res = buf['violence']
            fight_detected = False
            fight_confidence = 0
            if v_res['boxes'] is not None:
                for box in v_res['boxes']:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    # 优先使用 names 判断；若 names 缺失/为空则回退到 cls_id 规则
                    name_map = v_res.get('names', {})
                    cls_name = (name_map.get(cls_id, '') or '').lower()
                    if cls_name:
                        is_fight = (cls_name == 'fight')
                    else:
                        # 回退：按 cls_id=1 视为 fight（与测试模块一致时再保留）
                        is_fight = (cls_id == 1)

                    if is_fight:
                        color = (0, 255, 255)
                        label = f"FIGHT:{conf:.2f}"
                        fight_detected = True
                        fight_confidence = max(fight_confidence, conf)
                    else:
                        color = (0, 128, 255)
                        label = f"NOFIGHT:{conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 检测到fight行为且置信度高于0.5时，立即触发紧急语音提示
            # 【最高优先级】使用更短的冷却时间(1秒)，跳过去重检查，清空队列后立即播报
            if fight_detected and fight_confidence > 0.5:
                # 使用专用的fight冷却时间（1秒），比普通播报更短
                if current_time - last_fight_urgent_time >= FIGHT_COOLDOWN:
                    try:
                        user_settings = get_user_settings_for_video()
                        # 使用紧急提示，立即返回静态文本
                        speech_content = get_smart_prompt(
                            direction='fight',
                            obstacles=[],
                            position=None,
                            urgency='urgent',
                            user_settings=user_settings
                        )
                        current_speech_text = speech_content
                        # 设置音画同步状态（紧急模式）
                        set_speech_sync_state(fid, frame, speech_content, is_urgent=True)
                        # 使用紧急播报函数 - 清空队列并立即播放
                        speak_urgent(speech_content, user_settings)
                        # 更新fight专用时间戳
                        last_fight_urgent_time = current_time
                        last_fight_speak_time = current_time
                        print(f"[语音提示-紧急] ⚠️ 暴力行为检测 - {speech_content} (置信度: {fight_confidence:.2f})")
                    except Exception as e:
                        print(f"[语音提示] 暴力行为语音播放错误: {e}")

            # === 危险消失检测：当障碍物或打架消失后，重新提示当前盲道方向 ===
            # 智能判断：只有之前确实播报过警示，且连续多帧没有检测到危险时才播报"已通过"
            danger_cleared = False
            clear_type = None
            
            # 检测打架消失
            if was_fight_detected and not fight_detected:
                danger_cleared = True
                clear_type = 'fight'
                print("[危险消失] 打架行为已消失")
            
            # 检测盲道上的障碍物消失（智能判断）
            # 条件1：之前确实播报过障碍物警示（obstacle_alert_triggered = True）
            # 条件2：连续至少 MIN_CLEAR_FRAMES 帧没有检测到盲道上的障碍物（避免闪烁误报）
            # 条件3：距离上次障碍物警示至少1秒（避免刚播报完障碍物立刻说"已通过"）
            if (not danger_cleared and 
                obstacle_alert_triggered and 
                consecutive_clear_frames >= MIN_CLEAR_FRAMES and
                current_time - last_obstacle_alert_time >= 1.0):
                danger_cleared = True
                clear_type = 'obstacle'
                obstacle_alert_triggered = False  # 重置标志，避免重复播报
                print("[危险消失] 盲道上障碍物已通过（智能判断）")
            
            # 危险消失后，重新播报当前盲道方向
            if danger_cleared and (current_time - last_danger_clear_time >= DANGER_CLEAR_COOLDOWN):
                try:
                    user_settings = get_user_settings_for_video()
                    # 根据消失类型构建提示语
                    if clear_type == 'fight':
                        prefix = "危险解除，"
                    else:
                        prefix = "障碍物已通过，"
                    
                    # 获取当前盲道方向提示
                    direction_prompt = get_blind_road_prompt(current_blind_direction)
                    speech_content = prefix + direction_prompt
                    
                    current_speech_text = speech_content
                    set_speech_sync_state(fid, frame, speech_content, is_urgent=False)
                    _ts = time.time()
                    speak(speech_content, user_settings)
                    update_voice_ready(_ts)
                    last_danger_clear_time = current_time
                    # 重置方向去重，确保下次方向变化能正常触发
                    last_direction = current_blind_direction
                    print(f"[语音提示-危险消失] {speech_content}")
                except Exception as e:
                    print(f"[语音提示] 危险消失提示错误: {e}")
            
            # 更新上一帧的危险状态
            was_fight_detected = fight_detected
            was_obstacle_detected = obstacle_detected_on_path  # 只追踪盲道上的障碍物

            # 送给显示线程（携带帧ID用于音画同步）
            try:
                ready_at = max(current_time, last_speech_ready_time)
                result_queue.put_nowait({'frame': frame, 'frame_id': fid, 'ready_at': ready_at})
            except queue.Full:
                pass
            # 清缓存
            combine_buffer.pop(fid, None)
            # 检查是否需要退出（全部结束且队列清空）
            if (not thread_control.get('running', False) and combined_queue.empty() and not combine_buffer):
                break
# =============================================
# ==================== 线程安全包装器 ====================

def safe_thread_wrapper(thread_func, thread_name):
    """
    线程异常包装器：捕获所有异常，防止线程崩溃影响整体系统
    """
    def wrapper(*args, **kwargs):
        try:
            print(f"[{thread_name}] 线程已启动")
            thread_func(*args, **kwargs)
            print(f"[{thread_name}] 线程正常退出")
        except Exception as e:
            print(f"[{thread_name}] 线程异常: {e}")
            traceback.print_exc()

            # 记录错误并触发停止
            thread_control['error'] = f"{thread_name}: {str(e)}"
            thread_control['running'] = False

            # 向结果队列投放错误标记（让显示线程能感知）
            try:
                error_data = {
                    'error': True,
                    'message': f"{thread_name}线程崩溃: {str(e)}"
                }
                result_queue.put(error_data, block=False)
            except queue.Full:
                pass  # 队列满就算了
    return wrapper


# ==================== 线程生命周期管理 ====================

def start_async_processing(video_path):
    """
    启动异步视频处理管道：读帧线程 + 推理线程
    """
    global reader_thread, inference_thread, combine_buffer
    global last_direction, last_obstacle_type, last_speech_time
    global last_fight_speak_time, last_obstacle_speak_time, last_blind_road_speak_time, current_speech_text

    print(f"[异步管道] 准备启动异步处理: {video_path}")

    with thread_control['lock']:
        # 如果已有线程在运行，先停止
        if thread_control['running']:
            print("[异步管道] 检测到旧线程，先停止...")
            stop_async_processing()

        # 清空所有队列与合并缓冲，避免旧帧残留导致显示卡住
        clear_queue(frame_queue)
        clear_queue(result_queue)
        clear_queue(blind_queue)
        clear_queue(env_queue)
        clear_queue(vio_queue)
        clear_queue(combined_queue)
        clear_queue(blind_display_queue)
        clear_queue(env_display_queue)
        clear_queue(vio_display_queue)
        combine_buffer = defaultdict(dict)
        # 重置跳帧缓存与性能指标
        try:
            last_results['blind_road'] = None
            last_results['environment'] = None
            last_results['violence'] = None
            performance_metrics.update({
                'frame_queue_size': 0,
                'result_queue_size': 0,
                'reader_fps': 0,
                'inference_fps': 0,
                'frame_drop_count': 0,
                'last_update': 0
            })
        except Exception:
            pass

        # 重置语音播报相关的去重/冷却状态，避免跨视频遗留
        last_direction = None
        last_obstacle_type = None
        last_speech_time = 0
        last_speech_ready_time = 0
        voice_latency_avg = ALIGN_BASE_DELAY
        last_fight_speak_time = 0
        last_obstacle_speak_time = 0
        last_blind_road_speak_time = 0
        current_speech_text = ""

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            raise Exception(f"无法打开视频文件: {video_path}")

        thread_control['video_cap'] = cap
        # 计算视频原始FPS用于节流播放速率
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps < 1:
            src_fps = 25  # 默认25fps
        thread_control['frame_delay'] = 1.0 / src_fps
        thread_control['running'] = True
        thread_control['error'] = None
        thread_control['reader_alive'] = False
        thread_control['inference_alive'] = False

        # 创建并启动读帧线程
        reader_thread = threading.Thread(
            target=safe_thread_wrapper(frame_reader_worker, "读帧线程"),
            daemon=True,
            name="FrameReader"
        )

        # 创建并启动三模型推理线程 (Patch B)
        reader_thread.start()
        global inference_thread
        inference_thread = None  # 不再使用单线程推理
        threading.Thread(target=safe_thread_wrapper(
            lambda: push_inference(blind_queue, blind_road_model, 'blind_road', (255,0,0)),
            'BlindWorker'), daemon=True, name='BlindWorker').start()
        threading.Thread(target=safe_thread_wrapper(
            lambda: push_inference(env_queue, environment_model, 'environment', (0,0,255)),
            'EnvWorker'), daemon=True, name='EnvWorker').start()
        threading.Thread(target=safe_thread_wrapper(
            lambda: push_inference(vio_queue, violence_model, 'violence', (0,255,255)),
            'VioWorker'), daemon=True, name='VioWorker').start()
        threading.Thread(target=safe_thread_wrapper(
            combiner_worker, "Combiner"), daemon=True, name="Combiner").start()

        print("[异步管道] ✓ 异步处理管道已启动 (三线程)")


def stop_async_processing():
    """
    优雅停止所有异步线程
    """
    global reader_thread, inference_thread

    print("[异步管道] 正在停止异步处理...")

    # 设置停止标志
    thread_control['running'] = False

    # 等待线程退出（最多3秒）
    if reader_thread and reader_thread.is_alive():
        reader_thread.join(timeout=3)
        if reader_thread.is_alive():
            print("[异步管道] ⚠ 读帧线程未能在3秒内退出")

    if inference_thread and inference_thread.is_alive():
        inference_thread.join(timeout=3)
        if inference_thread.is_alive():
            print("[异步管道] ⚠ 推理线程未能在3秒内退出")

    # 释放视频资源
    if thread_control['video_cap']:
        thread_control['video_cap'].release()
        thread_control['video_cap'] = None

    # 清理所有队列，确保不会有残帧或旧事件造成闪现
    clear_queue(frame_queue)
    clear_queue(result_queue)
    clear_queue(blind_queue)
    clear_queue(env_queue)
    clear_queue(vio_queue)
    clear_queue(combined_queue)
    clear_queue(blind_display_queue)
    clear_queue(env_display_queue)
    clear_queue(vio_display_queue)
    combine_buffer.clear()

    # 重置标志
    thread_control['reader_alive'] = False
    thread_control['inference_alive'] = False
    thread_control['running'] = False

    print("[异步管道] ✓ 异步处理已停止，所有队列已清空")


# ==================== 异步线程工作函数 ====================

def frame_reader_worker():
    """
    读帧线程：从视频文件快速读取原始帧并放入队列
    职责：只负责读取，不做任何处理（轻量级，高吞吐）
    """
    cap = thread_control['video_cap']
    frame_count = 0
    reader_start_time = time.time()

    thread_control['reader_alive'] = True

    while thread_control['running'] and cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print(f"[读帧线程] 视频读取完毕，共读取 {frame_count} 帧")
            break

        frame_count += 1

        # 构造帧数据包
        frame_data = {
            'frame': frame,
            'frame_id': frame_count,
            'timestamp': time.time()
        }

        try:
            # 队列满时阻塞（背压机制：推理跟不上就等待）
            #frame_queue.put(frame_data, block=True, timeout=1.0)
            # ---------- Patch A: 广播帧到三模型队列 ----------
            for q in (blind_queue, env_queue, vio_queue):
                try:
                    q.put_nowait({'frame': frame.copy(),
                                  'fid': frame_count})
                except queue.Full:
                    pass
            # 始终按源 FPS 节流，避免一次性读完整段视频
            time.sleep(thread_control.get('frame_delay', 0.04))
            # 更新读帧FPS
            if frame_count % 30 == 0:  # 每30帧统计一次
                elapsed = time.time() - reader_start_time
                performance_metrics['reader_fps'] = int(frame_count / elapsed)

        except queue.Full:
            performance_metrics['frame_drop_count'] += 1
            # —— 节流：按原 FPS 读取，避免倍速 ——
            time.sleep(thread_control.get('frame_delay', 0.04))  # 默认0.04秒
            print(f"[读帧线程] ⚠ 队列已满，丢弃第 {frame_count} 帧")
            continue

    thread_control['reader_alive'] = False
    print(f"[读帧线程] 退出，总计读取 {frame_count} 帧")


def inference_worker():
    """
    推理线程：YOLO推理 + 方向判断 + 绘制检测框 + 触发语音
    职责：核心计算密集型任务，GPU推理
    """
    global last_call_time, current_speech_text, model_stats, frame_times

    thread_control['inference_alive'] = True
    inference_count = 0
    inference_start_time = time.time()

    while thread_control['running']:
        try:
            # 从队列获取帧数据（超时1秒，避免死锁）
            frame_data = frame_queue.get(timeout=1.0)
        except queue.Empty:
            # 检查读帧线程是否还活着
            if not thread_control['reader_alive'] and frame_queue.empty():
                print("[推理线程] 读帧线程已退出且队列为空，准备退出")
                break
            continue

        frame = frame_data['frame']
        frame_id = frame_data['frame_id']
        frame_timestamp = frame_data['timestamp']

        # 记录推理开始时间
        inference_start = time.time()

        # ========== 1. 帧缩放（降低计算量） ==========
        # 统一缩放到较低分辨率以提速
        resized_frame, scale = resize_frame(frame, 640)

        # ========== 2. YOLO推理 ==========
        results = model(resized_frame, verbose=False)

        centers = []  # 检测框中心点
        confidences = []  # 置信度

        # ========== 3. 解析检测结果 ==========
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # GPU tensor转CPU
                x1, y1, x2, y2 = box.xyxy[0].cpu()
                conf = box.conf[0].cpu()
                cls = int(box.cls[0].cpu())

                # 坐标映射回原始尺寸
                orig_x1, orig_y1 = float(x1) / scale, float(y1) / scale
                orig_x2, orig_y2 = float(x2) / scale, float(y2) / scale
                center_x = (orig_x1 + orig_x2) / 2
                center_y = (orig_y1 + orig_y2) / 2

                centers.append((center_x, center_y))
                confidences.append(float(conf))

                # 在原始帧上绘制检测框
                class_names = model.names
                label = f"{class_names[cls]}: {conf:.2f}"
                cv2.rectangle(frame, (int(orig_x1), int(orig_y1)),
                            (int(orig_x2), int(orig_y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(orig_x1), int(orig_y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ========== 4. 计算性能指标 ==========
        inference_end = time.time()
        frame_latency = (inference_end - inference_start) * 1000  # 毫秒

        # 更新FPS统计
        frame_times.append(inference_end)
        if len(frame_times) > max_frame_history:
            frame_times.pop(0)

        if len(frame_times) >= 2:
            time_span = frame_times[-1] - frame_times[0]
            current_fps = int(len(frame_times) / time_span) if time_span > 0 else 0
        else:
            current_fps = 0

        avg_confidence = int(np.mean(confidences) * 100) if confidences else 0

        # ---------- 追加三模型推理并单独统计 ----------
        def update_stats(tag, latency_ms, conf_list):
            s = stats_map[tag]
            s['latency'] = int(latency_ms)
            # fps 复用 current_fps 近似即可
            s['fps'] = current_fps
            s['confidence'] = int(np.mean(conf_list)*100) if conf_list else 0
            s['last_update'] = time.time()

        # 盲道检测 (blind_road_model)
        if blind_road_model_loaded:
            br_start = time.time();
            br_res = blind_road_model(resized_frame, verbose=False)
            br_conf = []
            for r in br_res:
                if r.boxes is not None:
                    for b in r.boxes:
                        br_conf.append(float(b.conf[0]))
            update_stats('blind_road', (time.time()-br_start)*1000, br_conf)

        # 环境感知
        if environment_model_loaded:
            env_start = time.time();
            env_res = environment_model(resized_frame, verbose=False)
            env_conf = []
            for r in env_res:
                if r.boxes is not None:
                    for b in r.boxes:
                        env_conf.append(float(b.conf[0]))
            update_stats('environment', (time.time()-env_start)*1000, env_conf)

        # 暴力检测
        if violence_model_loaded:
            vio_start = time.time();
            vio_res = violence_model(resized_frame, verbose=False)
            vio_conf = []
            for r in vio_res:
                if r.boxes is not None:
                    for b in r.boxes:
                        vio_conf.append(float(b.conf[0]))
            update_stats('violence', (time.time()-vio_start)*1000, vio_conf)

        # 更新全局统计
        model_stats['fps'] = current_fps
        model_stats['latency'] = int(frame_latency)
        model_stats['confidence'] = avg_confidence
        model_stats['last_update'] = time.time()

        # ========== 5. 方向判断与语音提示 ==========
        current_time = time.time()
        if len(centers) >= 2 and current_time - last_call_time >= CALL_INTERVAL:
            ys = np.array([c[1] for c in centers])
            xs = np.array([c[0] for c in centers])
            slope, intercept = np.polyfit(ys, xs, 1)

            print(f"[盲道检测] 斜率: {slope:.3f}")

            user_settings = get_user_settings_for_video()
            direction_detected = None

            if slope < -THRESHOLD_SLOPE:
                direction_detected = 'left'
                print("[盲道检测] 检测到左转")
            elif slope > THRESHOLD_SLOPE:
                direction_detected = 'right'
                print("[盲道检测] 检测到右转")

            # 触发语音提示（在单独线程中处理，避免阻塞）
            if direction_detected:
                question = left_turn_question if direction_detected == 'left' else right_turn_question

                # 启动语音生成线程（不阻塞推理）
                voice_thread = threading.Thread(
                    target=generate_and_speak,
                    args=(question, user_settings),
                    daemon=True
                )
                voice_thread.start()
                last_call_time = current_time

        # ========== 6. 构造结果数据包并投放 ==========
        result_data = {
            'frame': frame,
            'frame_id': frame_id,
            'fps': current_fps,
            'latency': int(frame_latency),
            'confidence': avg_confidence,
            'timestamp': time.time()
        }

        try:
            result_queue.put(result_data, block=True, timeout=1.0)
            inference_count += 1

            # 更新推理FPS
            if inference_count % 30 == 0:
                elapsed = time.time() - inference_start_time
                performance_metrics['inference_fps'] = int(inference_count / elapsed)

        except queue.Full:
            print(f"[推理线程] ⚠ 结果队列已满，丢弃帧 {frame_id}")

    thread_control['inference_alive'] = False
    print(f"[推理线程] 退出，总计推理 {inference_count} 帧")


def generate_and_speak(question, user_settings):
    """
    语音生成辅助函数：AI生成 + 语音播报（在独立线程中运行）
    """
    global current_speech_text

    answer_content = None
    ollama_success = False

    try:
        response = ollama_client.chat(model="qwen2.5:3b", messages=[
            {"role": "system", "content": get_prompt_template(user_settings)},
            {"role": "user", "content": question}
        ], stream=True)

        answer_content = ""
        for chunk in response:
            content = chunk.get('message', {}).get('content', '')
            if content:
                answer_content += content

        if answer_content.strip():
            ollama_success = True

    except Exception as e:
        print(f"[Ollama] 调用失败: {e}")

    # 降级为默认提示
    if not ollama_success or not answer_content:
        direction = "左" if "左" in question else "右"
        answer_content = f"请注意，盲道向{direction}转了，请往{direction}走。"

    # 设置语音文本并播放
    current_speech_text = answer_content
    speak(answer_content, user_settings)
    print(f"[语音提示] {answer_content}")


def get_user_settings_for_video():
    """
    获取用户设置（从 routes.main 模块）
    注意：这个函数在视频流生成器中调用，无法访问 session
    所以使用全局变量
    """
    from routes.main import user_settings
    return user_settings



def generate_main_frames():
    """兼容旧端点：直接复用异步生成器，保证主模型始终有帧输出"""
    yield from generate_frames()

def generate_frames():
    """
    显示生成器（异步版本）：从result_queue获取处理好的帧并流式传输
    职责：JPEG编码 + HTTP流式传输
    """
    global current_speech_text

    # 如果异步处理未启动，显示等待提示
    if not thread_control['running']:
        current_speech_text = "提示：系统会实时分析盲道方向，当方向发生变化时会自动播报语音提示。"
        while not thread_control['running']:
            wait_frame = create_info_frame("请上传视频文件开始分析")
            ret, buffer = cv2.imencode('.jpg', wait_frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)

    print("[显示线程] 开始从结果队列消费帧...")

    # 从结果队列消费并传输
    try:
        while True:
            try:
                # 从队列获取结果（超时0.5秒）
                result_data = result_queue.get(timeout=0.5)

                # 检测到错误标记
                if 'error' in result_data:
                    print(f"[显示线程] 检测到错误: {result_data.get('message', '未知错误')}")
                    error_frame = create_error_frame(result_data.get('message', '处理错误'))
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    break

                # 正常帧数据
                frame = result_data['frame']
                ready_at = result_data.get('ready_at', 0)
                delay = ready_at - time.time()
                if delay > 0:
                    time.sleep(delay)

                # 更新性能监控
                performance_metrics['frame_queue_size'] = frame_queue.qsize()
                performance_metrics['result_queue_size'] = result_queue.qsize()

                # JPEG编码（质量85，平衡大小与质量）
                ret, buffer = cv2.imencode('.jpg', frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])

                if not ret:
                    print("[显示线程] JPEG编码失败")
                    continue

                # HTTP流式传输
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')

            except queue.Empty:
             # 若还有线程在运行则继续等待，否则退出
                if thread_control['running']:
                    continue
                else:
                    break

    except Exception as e:
        print(f"[显示线程] 异常: {e}")
        traceback.print_exc()
        error_frame = create_error_frame(f"显示错误: {str(e)}")
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        print("[显示线程] 退出")
        # 显示结束画面
        end_frame = create_info_frame("视频已播放完毕，请上传新视频")
        ret, buffer = cv2.imencode('.jpg', end_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ============== 通用显示队列流式输出生成器 ==============
def _stream_from_display_queue(q, wait_text: str):
    """从指定显示队列输出JPEG流。未开始时显示占位提示。"""
    # 未启动时提示
    if not thread_control['running']:
        while not thread_control['running']:
            wait_frame = create_info_frame(wait_text)
            ret, buf = cv2.imencode('.jpg', wait_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.8)

    # 运行时从队列取帧
    while True:
        try:
            item = q.get(timeout=1.0)
            frame = item['frame']
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        except queue.Empty:
            if not thread_control['running']:
                # 结束占位
                end_frame = create_info_frame("视频已播放完毕，请上传新视频")
                ret, buf = cv2.imencode('.jpg', end_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                break
            # 运行中但暂时无帧，继续等待
            continue


# 保留旧版generate_frames作为备份（可选）
def generate_frames_legacy():
    """生成视频帧用于流式传输"""
    global last_call_time, current_speech_text, current_video_path, video_active, model_stats, frame_times


    # 如果视频未激活，显示等待上传提示
    if not video_active or not current_video_path:
        # 设置默认的提示文本
        current_speech_text = "提示：系统会实时分析盲道方向，当方向发生变化时会自动播报语音提示。"
        while not video_active or not current_video_path:
            wait_frame = create_info_frame("请上传视频文件开始分析")
            ret, buffer = cv2.imencode('.jpg', wait_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)

    # 视频已激活，开始处理
    try:
        cap = cv2.VideoCapture(current_video_path)

        if not cap.isOpened():
            print(f"无法打开视频: {current_video_path}")
            # 尝试使用ffmpeg参数打开
            cap = cv2.VideoCapture(current_video_path, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                # 仍然无法打开，显示错误信息
                error_frame = create_error_frame(f"无法打开视频文件: {os.path.basename(current_video_path)}")
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                video_active = False
                current_speech_text = "视频无法打开，请尝试上传其他格式的视频。"
                return

        frame_count = 0

        while cap.isOpened() and video_active:
            ret, frame = cap.read()
            frame_count += 1

            if not ret:
                if frame_count < 10:  # 如果连前10帧都读不出来
                    print(f"无法读取视频帧: {current_video_path}")
                    error_frame = create_error_frame("视频文件损坏或格式不支持")
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    video_active = False
                    current_speech_text = "视频文件损坏或格式不支持，请尝试其他视频。"
                    break

                # 视频正常结束
                end_frame = create_info_frame("视频已播放完毕，请上传新视频")
                ret, buffer = cv2.imencode('.jpg', end_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                video_active = False
                current_speech_text = "视频播放完毕，请上传新视频。"
                break

            # 记录帧开始处理时间
            frame_start_time = time.time()


            # 处理视频帧 - 板块一主模型级联推理逻辑（盲道+环境感知+暴力行为）
            # 第一步：使用主模型的盲道检测模型进行推理
            blind_road_detected = False
            centers = []  # 存储所有检测框的 (center_x, center_y)
            confidences = []  # 存储所有检测框的置信度
            annotated_frame = frame.copy()  # 用于绘制的帧

            if main_blind_road_model_loaded:
                blind_road_results = main_blind_road_model.predict(frame, verbose=False)

                # 检查盲道检测结果
                if blind_road_results and len(blind_road_results) > 0:
                    result = blind_road_results[0]
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        blind_road_detected = True
                        # 提取盲道检测框的中心点和置信度
                        for box in boxes:
                            xyxy = box.xyxy[0]
                            if hasattr(xyxy, 'cpu'):
                                xyxy = xyxy.cpu().numpy()
                            x1, y1, x2, y2 = xyxy
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            centers.append((center_x, center_y))
                            conf = float(box.conf[0])
                            confidences.append(conf)

            # 第二步：根据检测结果决策
            if blind_road_detected and main_blind_road_model_loaded:
                # 如果检测到盲道，先绘制盲道检测结果
                annotated_frame = blind_road_results[0].plot()
                print("[主模型级联推理] 检测到盲道，使用盲道检测模型结果")

                # 同时运行环境感知模型
                if main_environment_model_loaded:
                    try:
                        environment_results = main_environment_model.predict(frame, verbose=False)
                        if environment_results and len(environment_results) > 0:
                            env_result = environment_results[0]
                            env_boxes = env_result.boxes

                            # 在已绘制的盲道帧上叠加绘制环境感知结果。
                            # 若已检测到盲道 -> 全部统一标记为 obstacle；
                            # 否则保持原 5 类标签名。
                            if env_boxes is not None and len(env_boxes) > 0:
                                for box in env_boxes:
                                    xyxy = box.xyxy[0]
                                    if hasattr(xyxy, 'cpu'):
                                        xyxy = xyxy.cpu().numpy()
                                    x1, y1, x2, y2 = xyxy.astype(int)
                                    conf = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    if blind_road_detected:
                                        label = f"obstacle: {conf:.2f}"
                                        color = (0, 0, 255)  # red
                                    else:
                                        name = env_result.names.get(cls, 'obj') if hasattr(env_result, 'names') else 'obj'
                                        label = f"{name}: {conf:.2f}"
                                        color = (255, 0, 0)  # blue
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    except Exception as e:
                        print(f"[主模型级联推理] 环境感知模型推理失败: {e}")
            else:
                # 如果没有检测到盲道，使用环境感知模型进行第二次推理
                if main_environment_model_loaded:
                    try:
                        environment_results = main_environment_model.predict(frame, verbose=False)
                        annotated_frame = environment_results[0].plot()
                        print("[主模型级联推理] 未检测到盲道，使用环境感知模型结果（5类输出）")
                    except Exception as e:
                        print(f"[主模型级联推理] 环境感知模型推理失败: {e}")
                        annotated_frame = frame.copy()
                else:
                    annotated_frame = frame.copy()

            # 第三步：同时运行暴力行为检测模型（无论是否检测到盲道都执行）
            if main_violence_model_loaded:
                try:
                    violence_results = main_violence_model.predict(frame, verbose=False)
                    if violence_results and len(violence_results) > 0:
                        violence_result = violence_results[0]
                        violence_boxes = violence_result.boxes

                        # 在已绘制的帧上叠加绘制暴力行为检测结果
                        if violence_boxes is not None and len(violence_boxes) > 0:
                            for box in violence_boxes:
                                cls = int(box.cls[0])
                                # 只绘制fight类别（暴力行为）
                                if hasattr(violence_result, 'names') and violence_result.names.get(cls, '').lower() == 'fight':
                                    xyxy = box.xyxy[0]
                                    if hasattr(xyxy, 'cpu'):
                                        xyxy = xyxy.cpu().numpy()
                                    x1, y1, x2, y2 = xyxy.astype(int)
                                    conf = float(box.conf[0])

                                    # 使用黄色绘制暴力行为
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                                    label = f"FIGHT: {conf:.2f}"
                                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                except Exception as e:
                    print(f"[主模型级联推理] 暴力行为检测模型推理失败: {e}")

            # 使用绘制后的帧进行后续处理
            frame = annotated_frame
            # 缩放帧以加速推理
            resized_frame, scale = resize_frame(frame)

            # 处理视频帧 - YOLO检测（优化参数）
            results = model(
                resized_frame,
                verbose=False  # 关闭日志输出
            )
            centers = []  # 存储所有检测框的 (center_x, center_y)
            confidences = []  # 存储所有检测框的置信度

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 获取缩放后的坐标 (GPU tensor需要先转CPU再转numpy/float)
                    x1, y1, x2, y2 = box.xyxy[0].cpu()
                    conf = box.conf[0].cpu()
                    cls = int(box.cls[0].cpu())

                    # 将坐标缩放回原始尺寸用于中心点计算
                    orig_x1, orig_y1 = float(x1) / scale, float(y1) / scale
                    orig_x2, orig_y2 = float(x2) / scale, float(y2) / scale
                    center_x = (orig_x1 + orig_x2) / 2
                    center_y = (orig_y1 + orig_y2) / 2
                    centers.append((center_x, center_y))
                    confidences.append(float(conf))

                    # 在原始帧上绘制（使用缩放回的坐标）
                    class_names = model.names
                    label = f"{class_names[cls]}: {conf:.2f}"
                    cv2.rectangle(frame, (int(orig_x1), int(orig_y1)), (int(orig_x2), int(orig_y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(orig_x1), int(orig_y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            # 计算性能指标
            frame_end_time = time.time()
            frame_latency = (frame_end_time - frame_start_time) * 1000  # 转换为毫秒

            # 更新帧时间历史
            frame_times.append(frame_end_time)
            if len(frame_times) > max_frame_history:
                frame_times.pop(0)

            # 计算FPS（基于最近的帧）
            if len(frame_times) >= 2:
                time_span = frame_times[-1] - frame_times[0]
                if time_span > 0:
                    current_fps = len(frame_times) / time_span
                else:
                    current_fps = 0
            else:
                current_fps = 0

            # 计算平均置信度（综合三个模型的置信度）
            all_confidences = confidences.copy()
            if main_environment_model_loaded:
                try:
                    env_results = main_environment_model.predict(frame, verbose=False)
                    if env_results and len(env_results) > 0:
                        env_boxes = env_results[0].boxes
                        if env_boxes is not None:
                            for box in env_boxes:
                                all_confidences.append(float(box.conf[0]))
                except:
                    pass

            if main_violence_model_loaded:
                try:
                    violence_results = main_violence_model.predict(frame, verbose=False)
                    if violence_results and len(violence_results) > 0:
                        violence_boxes = violence_results[0].boxes
                        if violence_boxes is not None:
                            for box in violence_boxes:
                                all_confidences.append(float(box.conf[0]))
                except:
                    pass

            avg_confidence = int(np.mean(all_confidences) * 100) if all_confidences else 0

            # 更新主模型统计数据
            main_model_stats['fps'] = int(current_fps)
            main_model_stats['latency'] = int(frame_latency)
            main_model_stats['confidence'] = avg_confidence
            main_model_stats['last_update'] = time.time()

            current_time = time.time()
            # 盲道方向检测和语音提示（仅在检测到盲道时执行）
            if blind_road_detected and len(centers) >= 2 and current_time - last_call_time >= CALL_INTERVAL:
                ys = np.array([c[1] for c in centers])
                xs = np.array([c[0] for c in centers])
                slope, intercept = np.polyfit(ys, xs, 1)

                print(f"[盲道检测] 斜率: {slope}, 拦截: {intercept}")

                user_settings = get_user_settings_for_video()

                if slope < -THRESHOLD_SLOPE:
                    # 斜率显著为负，提示左转
                    print("[盲道检测] 检测到左转")
                    answer_content = None
                    ollama_success = False

                    try:
                        print(f"[Ollama] 尝试连接 Ollama 服务，模型: qwen2.5:3b")
                        print(f"[Ollama] 使用的提示词模板:\n{get_prompt_template(user_settings)}")
                        print(f"[Ollama] 用户问题: {left_turn_question}")

                        response = ollama_client.chat(model="qwen2.5:3b", messages=[
                            {"role": "system", "content": get_prompt_template(user_settings)},
                            {"role": "user", "content": left_turn_question}
                        ], stream=True)

                        answer_content = ""
                        chunk_count = 0
                        for chunk in response:
                            chunk_count += 1
                            content = chunk.get('message', {}).get('content', '')
                            if content:
                                answer_content += content

                        print(f"[Ollama] ✓ 成功获取 AI 响应，共 {chunk_count} 个块")
                        print(f"[Ollama] AI 生成内容: {answer_content}")

                        if answer_content.strip():
                            ollama_success = True
                        else:
                            print(f"[Ollama] ✗ AI 返回内容为空")

                    except ConnectionError as conn_error:
                        print(f"[Ollama] ✗ 连接错误 - Ollama 服务可能未启动: {conn_error}")
                    except Exception as ollama_error:
                        print(f"[Ollama] ✗ 其他错误: {type(ollama_error).__name__}: {ollama_error}")
                        import traceback
                        traceback.print_exc()

                    # 如果 Ollama 失败或返回空内容，使用默认提示
                    if not ollama_success or not answer_content:
                        print(f"[盲道检测] 使用默认左转提示")
                        answer_content = f"请注意，盲道向左转了，请往左走。"

                    # 设置语音文本并播放
                    current_speech_text = answer_content
                    speak(answer_content, user_settings)
                    last_call_time = current_time
                    print(f"[盲道检测] 已发送左转语音提示到播放队列")

                elif slope > THRESHOLD_SLOPE:
                    # 斜率显著为正，提示右转
                    print("[盲道检测] 检测到右转")
                    answer_content = None
                    ollama_success = False

                    try:
                        print(f"[Ollama] 尝试连接 Ollama 服务，模型: qwen2.5:3b")
                        print(f"[Ollama] 使用的提示词模板:\n{get_prompt_template(user_settings)}")
                        print(f"[Ollama] 用户问题: {right_turn_question}")

                        response = ollama_client.chat(model="qwen2.5:3b", messages=[
                            {"role": "system", "content": get_prompt_template(user_settings)},
                            {"role": "user", "content": right_turn_question}
                        ], stream=True)

                        answer_content = ""
                        chunk_count = 0
                        for chunk in response:
                            chunk_count += 1
                            content = chunk.get('message', {}).get('content', '')
                            if content:
                                answer_content += content

                        print(f"[Ollama] ✓ 成功获取 AI 响应，共 {chunk_count} 个块")
                        print(f"[Ollama] AI 生成内容: {answer_content}")

                        if answer_content.strip():
                            ollama_success = True
                        else:
                            print(f"[Ollama] ✗ AI 返回内容为空")

                    except ConnectionError as conn_error:
                        print(f"[Ollama] ✗ 连接错误 - Ollama 服务可能未启动: {conn_error}")
                    except Exception as ollama_error:
                        print(f"[Ollama] ✗ 其他错误: {type(ollama_error).__name__}: {ollama_error}")
                        import traceback
                        traceback.print_exc()

                    # 如果 Ollama 失败或返回空内容，使用默认提示
                    if not ollama_success or not answer_content:
                        print(f"[盲道检测] 使用默认右转提示")
                        answer_content = f"请注意，盲道向右转了，请往右走。"

                    # 设置语音文本并播放
                    current_speech_text = answer_content
                    speak(answer_content, user_settings)
                    last_call_time = current_time
                    print(f"[盲道检测] 已发送右转语音提示到播放队列")

            # 编码并发送绘制后的帧
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    except Exception as e:
        print(f"视频处理错误: {e}")
        import traceback
        traceback.print_exc()
        error_frame = create_error_frame(f"视频处理错误: {str(e)}")
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        video_active = False
        current_speech_text = "视频处理出错，请尝试上传其他视频。"


def generate_blind_road_frames():
    """生成板块二模型一视频帧（仅盲道检测）"""
    global current_video_path, video_active, blind_road_model_stats, frame_times

    if not video_active or not current_video_path or not blind_road_model_loaded:
        while not video_active or not current_video_path:
            wait_frame = create_info_frame("请上传视频文件开始分析" if not current_video_path else "盲道检测模型未加载")
            ret, buffer = cv2.imencode('.jpg', wait_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)
        return

    try:
        cap = cv2.VideoCapture(current_video_path)
        if not cap.isOpened():
            cap = cv2.VideoCapture(current_video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                error_frame = create_error_frame(f"无法打开视频文件")
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                return

        frame_times_local = []
        while cap.isOpened() and video_active:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start_time = time.time()

            # 盲道检测模型推理
            results = blind_road_model.predict(frame, verbose=False)
            if results and len(results) > 0:
                annotated_frame = results[0].plot()

                # 计算置信度
                boxes = results[0].boxes
                confidences = []
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        conf = float(box.conf[0])
                        confidences.append(conf)
                avg_confidence = int(np.mean(confidences) * 100) if confidences else 0
            else:
                annotated_frame = frame.copy()
                avg_confidence = 0

            # 更新统计信息
            frame_end_time = time.time()
            frame_latency = (frame_end_time - frame_start_time) * 1000
            frame_times_local.append(frame_end_time)
            if len(frame_times_local) > max_frame_history:
                frame_times_local.pop(0)

            if len(frame_times_local) >= 2:
                time_span = frame_times_local[-1] - frame_times_local[0]
                current_fps = len(frame_times_local) / time_span if time_span > 0 else 0
            else:
                current_fps = 0

            blind_road_model_stats['fps'] = int(current_fps)
            blind_road_model_stats['latency'] = int(frame_latency)
            blind_road_model_stats['confidence'] = avg_confidence
            blind_road_model_stats['last_update'] = time.time()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()
    except Exception as e:
        print(f"盲道检测模型视频处理错误: {e}")
        error_frame = create_error_frame(f"盲道检测处理错误: {str(e)}")
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_environment_frames():
    """生成板块二模型二视频帧（仅环境感知）"""
    global current_video_path, video_active, environment_model_stats, frame_times

    if not video_active or not current_video_path or not environment_model_loaded:
        while not video_active or not current_video_path:
            wait_frame = create_info_frame("请上传视频文件开始分析" if not current_video_path else "环境感知模型未加载")
            ret, buffer = cv2.imencode('.jpg', wait_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)
        return

    try:
        cap = cv2.VideoCapture(current_video_path)
        if not cap.isOpened():
            cap = cv2.VideoCapture(current_video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                error_frame = create_error_frame(f"无法打开视频文件")
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                return

        frame_times_local = []
        while cap.isOpened() and video_active:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start_time = time.time()

            # 环境感知模型推理
            results = environment_model.predict(frame, verbose=False)
            if results and len(results) > 0:
                annotated_frame = results[0].plot()

                # 计算置信度
                boxes = results[0].boxes
                confidences = []
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        conf = float(box.conf[0])
                        confidences.append(conf)
                avg_confidence = int(np.mean(confidences) * 100) if confidences else 0
            else:
                annotated_frame = frame.copy()
                avg_confidence = 0

            # 更新统计信息
            frame_end_time = time.time()
            frame_latency = (frame_end_time - frame_start_time) * 1000
            frame_times_local.append(frame_end_time)
            if len(frame_times_local) > max_frame_history:
                frame_times_local.pop(0)

            if len(frame_times_local) >= 2:
                time_span = frame_times_local[-1] - frame_times_local[0]
                current_fps = len(frame_times_local) / time_span if time_span > 0 else 0
            else:
                current_fps = 0

            environment_model_stats['fps'] = int(current_fps)
            environment_model_stats['latency'] = int(frame_latency)
            environment_model_stats['confidence'] = avg_confidence
            environment_model_stats['last_update'] = time.time()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()
    except Exception as e:
        print(f"环境感知模型视频处理错误: {e}")
        error_frame = create_error_frame(f"环境感知处理错误: {str(e)}")
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_violence_frames():
    """生成板块二模型三视频帧（仅暴力行为检测）"""
    global current_video_path, video_active, violence_model_stats, frame_times

    if not video_active or not current_video_path or not violence_model_loaded:
        while not video_active or not current_video_path:
            wait_frame = create_info_frame("请上传视频文件开始分析" if not current_video_path else "暴力行为检测模型未加载")
            ret, buffer = cv2.imencode('.jpg', wait_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)
        return

    try:
        cap = cv2.VideoCapture(current_video_path)
        if not cap.isOpened():
            cap = cv2.VideoCapture(current_video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                error_frame = create_error_frame(f"无法打开视频文件")
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                return

        frame_times_local = []
        while cap.isOpened() and video_active:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start_time = time.time()

            # 暴力行为检测模型推理
            results = violence_model.predict(frame, verbose=False)
            if results and len(results) > 0:
                annotated_frame = results[0].plot()

                # 计算置信度
                boxes = results[0].boxes
                confidences = []
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        conf = float(box.conf[0])
                        confidences.append(conf)
                avg_confidence = int(np.mean(confidences) * 100) if confidences else 0
            else:
                annotated_frame = frame.copy()
                avg_confidence = 0

            # 更新统计信息
            frame_end_time = time.time()
            frame_latency = (frame_end_time - frame_start_time) * 1000
            frame_times_local.append(frame_end_time)
            if len(frame_times_local) > max_frame_history:
                frame_times_local.pop(0)

            if len(frame_times_local) >= 2:
                time_span = frame_times_local[-1] - frame_times_local[0]
                current_fps = len(frame_times_local) / time_span if time_span > 0 else 0
            else:
                current_fps = 0

            violence_model_stats['fps'] = int(current_fps)
            violence_model_stats['latency'] = int(frame_latency)
            violence_model_stats['confidence'] = avg_confidence
            violence_model_stats['last_update'] = time.time()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()
    except Exception as e:
        print(f"暴力行为检测模型视频处理错误: {e}")
        error_frame = create_error_frame(f"暴力行为检测处理错误: {str(e)}")
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@video_bp.route('/video_feed')
def video_feed():
    """视频流式传输端点（兼容旧版本，默认使用主模型）"""
    return Response(generate_main_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@video_bp.route('/video_feed/main')
def video_feed_main():
    """主视频流端点：异步处理版本"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@video_bp.route('/video_feed/blind_road')
def video_feed_blind_road():
    """板块二模型一（盲道检测）视频流式传输端点（改为从显示队列流出）"""
    return Response(_stream_from_display_queue(blind_display_queue, "等待测试开始"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@video_bp.route('/video_feed/environment')
def video_feed_environment():
    """板块二模型二（环境感知）视频流式传输端点（改为从显示队列流出）"""
    return Response(_stream_from_display_queue(env_display_queue, "等待测试开始"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@video_bp.route('/video_feed/violence')
def video_feed_violence():
    """板块二模型三（暴力行为检测）视频流式传输端点（改为从显示队列流出）"""
    return Response(_stream_from_display_queue(vio_display_queue, "等待测试开始"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@video_bp.route('/stream_speech_text')
def stream_speech_text():
    """流式传输语音文本"""
    def generate():
        global current_speech_text
        last_sent = ""

        # 设置初始默认消息
        if not current_speech_text:
            current_speech_text = "提示：系统会实时分析盲道方向，当方向发生变化时会自动播报语音提示。"

        while True:
            if current_speech_text != last_sent:
                last_sent = current_speech_text
                yield f"{current_speech_text}\n\n"
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream')


@video_bp.route('/speech_sync_status')
def speech_sync_status():
    """
    获取语音同步状态 - 用于前端音画同步控制
    返回：是否正在播报、触发帧ID、预计剩余时间
    """
    global speech_sync_state
    
    with speech_sync_lock:
        is_speaking = speech_sync_state['is_speaking']
        trigger_frame_id = speech_sync_state['trigger_frame_id']
        speech_text = speech_sync_state['speech_text']
        start_time = speech_sync_state['speech_start_time']
        duration = speech_sync_state['speech_duration']
    
    # 计算剩余时间
    elapsed = time.time() - start_time if start_time > 0 else 0
    remaining = max(0, duration - elapsed)
    
    # 如果已超时，自动清除状态
    if is_speaking and remaining <= 0:
        clear_speech_sync_state()
        is_speaking = False
        remaining = 0
    
    return jsonify({
        'is_speaking': is_speaking,
        'trigger_frame_id': trigger_frame_id,
        'speech_text': speech_text,
        'remaining_time': round(remaining, 2),
        'duration': duration
    })


@video_bp.route('/freeze_frame')
def get_freeze_frame():
    """
    获取冻结帧 - 语音播报时显示的静止画面
    返回：JPEG图像
    """
    global speech_sync_state
    
    with speech_sync_lock:
        freeze_frame = speech_sync_state.get('freeze_frame')
        is_speaking = speech_sync_state.get('is_speaking', False)
    
    if freeze_frame and is_speaking:
        return Response(freeze_frame, mimetype='image/jpeg')
    else:
        # 没有冻结帧时返回空
        return Response(status=204)


@video_bp.route('/start_test', methods=['POST'])
def start_test():
    """激活所有模型的视频处理"""
    global video_active, current_video_path

    if not current_video_path:
        return jsonify({"status": "error", "message": "请先上传视频文件"}), 400

    video_active = True
    print("[API Call] /start_test: 视频处理已激活")
    return jsonify({"status": "success", "message": "视频处理已开始"})


@video_bp.route('/stop_test', methods=['POST'])
def stop_test():
    """停止当前视频处理并彻底清理所有状态，防止旧视频闪现"""
    global video_active, current_video_path
    try:
        # 停止异步处理线程（会清空所有队列）
        stop_async_processing()
    except Exception as e:
        print(f"[API Call] /stop_test 异常: {e}")
    
    # 重置全局状态
    video_active = False
    current_video_path = None
    
    print("[API Call] /stop_test: 视频处理已停止，所有状态已重置")
    return jsonify({"status": "success", "message": "视频处理已停止"})


@video_bp.route('/upload_video', methods=['POST'])
def upload_video():
    """处理视频上传"""
    global current_video_path, video_active

    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "没有上传文件"}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"}), 400

    if not allowed_file(file.filename):
        from config import ALLOWED_EXTENSIONS
        return jsonify(
            {"status": "error", "message": f"不支持的文件类型，允许的类型: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        # 确保旧管道完全停止，避免资源竞争导致新视频不启动
        try:
            stop_async_processing()
        except Exception as e:
            print(f"[上传] 停止旧处理异常: {e}")
        # 创建上传目录（如果不存在）
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # 使用时间戳生成唯一文件名，避免文件名冲突
        timestamp = int(time.time())
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # 检查视频是否可以打开
        test_cap = cv2.VideoCapture(file_path)
        if not test_cap.isOpened():
            test_cap.release()
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"status": "error", "message": "无法打开视频文件，请检查文件格式或尝试其他视频"}), 400

        # 读取几帧确认真的可以读取
        read_success = False
        for _ in range(5):  # 尝试读取前5帧
            ret, _ = test_cap.read()
            if ret:
                read_success = True
                break

        test_cap.release()

        if not read_success:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"status": "error", "message": "视频文件无法正常读取帧，请尝试其他视频"}), 400

        # 如果之前有视频文件，先删除
        if current_video_path and os.path.exists(current_video_path):
            try:
                os.remove(current_video_path)
            except Exception as e:
                print(f"无法删除旧视频文件: {e}")

        current_video_path = file_path
        video_active = True
        print(f"成功上传视频: {file_path}")

        # ========== 启动异步处理管道 ==========
        try:
            start_async_processing(file_path)
            print(f"[上传] ✓ 异步处理管道已启动")
        except Exception as e:
            print(f"[上传] ✗ 启动异步处理失败: {e}")
            traceback.print_exc()
            return jsonify({"status": "error", "message": f"异步处理启动失败: {str(e)}"}), 500

        return jsonify({
            "status": "success",
            "message": "视频上传成功",
            "file_path": file_path
        })
    except Exception as e:
        print(f"视频上传错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"上传失败: {str(e)}"}), 500


@video_bp.route('/get_model_stats', methods=['GET'])
def get_model_stats():
    """获取模型性能统计数据（兼容旧版本，默认返回主模型）"""
    global main_model_stats, video_active

    # 检查视频是否正在运行，如果超过3秒没有更新，认为已停止
    current_time = time.time()
    is_active = video_active and (current_time - main_model_stats.get('last_update', 0)) < 3

    if not is_active:
        # 视频未运行时返回默认值
        return jsonify({
            "status": "success",
            "active": False,
            "fps": 0,
            "latency": 0,
            "confidence": 0
        })

    # 返回实时性能数据
    return jsonify({
        "status": "success",
        "active": True,
        "fps": main_model_stats.get('fps', 0),
        "latency": main_model_stats.get('latency', 0),
        "confidence": main_model_stats.get('confidence', 0)
    })

@video_bp.route('/get_model_stats/main', methods=['GET'])
def get_main_stats():
    """主模型统计直接取盲道检测 stats"""
    d = blind_road_model_stats
    active = (time.time() - d.get('last_update', 0)) < 3
    return jsonify({
        "status": "success",
        "active": active,
        "fps": d.get('fps', 0),
        "latency": d.get('latency', 0),
        "confidence": d.get('confidence', 0)
    })
@video_bp.route('/get_model_stats/main', methods=['GET'])
def get_main_model_stats():
    """获取板块一主模型性能统计数据"""
    global main_model_stats, video_active

    current_time = time.time()
    is_active = video_active and (current_time - main_model_stats.get('last_update', 0)) < 3

    if not is_active:
        return jsonify({
            "status": "success",
            "active": False,
            "fps": 0,
            "latency": 0,
            "confidence": 0
        })

    return jsonify({
        "status": "success",
        "active": True,
        "fps": main_model_stats.get('fps', 0),
        "latency": main_model_stats.get('latency', 0),
        "confidence": main_model_stats.get('confidence', 0)
    })

@video_bp.route('/get_model_stats/blind_road', methods=['GET'])
def get_blind_road_model_stats():
    """获取板块二模型一（盲道检测）性能统计数据"""
    global blind_road_model_stats, video_active

    current_time = time.time()
    is_active = video_active and (current_time - blind_road_model_stats.get('last_update', 0)) < 3

    if not is_active:
        return jsonify({
            "status": "success",
            "active": False,
            "fps": 0,
            "latency": 0,
            "confidence": 0
        })

    return jsonify({
        "status": "success",
        "active": True,
        "fps": blind_road_model_stats.get('fps', 0),
        "latency": blind_road_model_stats.get('latency', 0),
        "confidence": blind_road_model_stats.get('confidence', 0)
    })

@video_bp.route('/get_model_stats/environment', methods=['GET'])
def get_environment_model_stats():
    """获取板块二模型二（环境感知）性能统计数据"""
    global environment_model_stats, video_active

    current_time = time.time()
    is_active = video_active and (current_time - environment_model_stats.get('last_update', 0)) < 3

    if not is_active:
        return jsonify({
            "status": "success",
            "active": False,
            "fps": 0,
            "latency": 0,
            "confidence": 0
        })

    return jsonify({
        "status": "success",
        "active": True,
        "fps": environment_model_stats.get('fps', 0),
        "latency": environment_model_stats.get('latency', 0),
        "confidence": environment_model_stats.get('confidence', 0)
    })

@video_bp.route('/get_model_stats/violence', methods=['GET'])
def get_violence_model_stats():
    """获取板块二模型三（暴力行为检测）性能统计数据"""
    global violence_model_stats, video_active

    current_time = time.time()
    is_active = video_active and (current_time - violence_model_stats.get('last_update', 0)) < 3

    if not is_active:
        return jsonify({
            "status": "success",
            "active": False,
            "fps": 0,
            "latency": 0,
            "confidence": 0
        })

    return jsonify({
        "status": "success",
        "active": True,
        "fps": violence_model_stats.get('fps', 0),
        "latency": violence_model_stats.get('latency', 0),
        "confidence": violence_model_stats.get('confidence', 0)
    })


@video_bp.route('/get_models_status', methods=['GET'])
def get_models_status():
    """获取所有模型的加载状态"""
    global main_blind_road_model_loaded, main_environment_model_loaded, main_violence_model_loaded
    global blind_road_model_loaded, environment_model_loaded, violence_model_loaded
    global MAIN_BLIND_ROAD_MODEL_PATH, MAIN_ENVIRONMENT_MODEL_PATH, MAIN_VIOLENCE_MODEL_PATH
    global BLIND_ROAD_MODEL_PATH, ENVIRONMENT_MODEL_PATH, VIOLENCE_MODEL_PATH

    # 说明：loaded 表示权重已加载；active 表示最近一段时间内确实在处理视频帧。
    # 前端“运行中”应跟 active 同步，而不是跟 loaded 同步。
    now = time.time()
    main_loaded = main_blind_road_model_loaded and main_environment_model_loaded and main_violence_model_loaded
    main_active = bool(video_active) and (now - main_model_stats.get('last_update', 0)) < 3
    blind_road_active = bool(video_active) and (now - blind_road_model_stats.get('last_update', 0)) < 3
    environment_active = bool(video_active) and (now - environment_model_stats.get('last_update', 0)) < 3
    violence_active = bool(video_active) and (now - violence_model_stats.get('last_update', 0)) < 3

    def _status(loaded: bool, active: bool) -> str:
        if not loaded:
            return "pending"
        return "running" if active else "idle"

    return jsonify({
        "status": "success",
        "models": {
            "main": {
                "name": "主模型（级联推理）",
                "loaded": main_loaded,
                "active": main_active,
                "components": {
                    "blind_road": {
                        "loaded": main_blind_road_model_loaded,
                        "path": MAIN_BLIND_ROAD_MODEL_PATH if main_blind_road_model_loaded else None
                    },
                    "environment": {
                        "loaded": main_environment_model_loaded,
                        "path": MAIN_ENVIRONMENT_MODEL_PATH if main_environment_model_loaded else None
                    },
                    "violence": {
                        "loaded": main_violence_model_loaded,
                        "path": MAIN_VIOLENCE_MODEL_PATH if main_violence_model_loaded else None
                    }
                },
                "status": _status(main_loaded, main_active)
            },
            "blind_road": {
                "name": "模型一（盲道检测）",
                "loaded": blind_road_model_loaded,
                "active": blind_road_active,
                "path": BLIND_ROAD_MODEL_PATH if blind_road_model_loaded else None,
                "status": _status(blind_road_model_loaded, blind_road_active)
            },
            "environment": {
                "name": "模型二（环境感知）",
                "loaded": environment_model_loaded,
                "active": environment_active,
                "path": ENVIRONMENT_MODEL_PATH if environment_model_loaded else None,
                "status": _status(environment_model_loaded, environment_active)
            },
            "violence": {
                "name": "模型三（暴力行为检测）",
                "loaded": violence_model_loaded,
                "active": violence_active,
                "path": VIOLENCE_MODEL_PATH if violence_model_loaded else None,
                "status": _status(violence_model_loaded, violence_active)
            }
        }
})
