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
from utils.voice_utils import speak, get_prompt_template
from config import MODEL_WEIGHTS, UPLOAD_FOLDER, THRESHOLD_SLOPE, CALL_INTERVAL

# 加载YOLO模型并优化性能
model = YOLO(MODEL_WEIGHTS)

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

# 性能统计变量
model_stats = {
    'fps': 0,
    'latency': 0,
    'confidence': 0,
    'last_update': 0
}
frame_times = []  # 存储最近的帧时间戳
max_frame_history = 30  # 保留最近30帧的数据

# ==================== 异步架构：线程和队列 ====================
# 线程间通信队列
frame_queue = queue.Queue(maxsize=3)    # 读帧线程 → 推理线程
result_queue = queue.Queue(maxsize=3)   # 推理线程 → 显示线程

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
    global reader_thread, inference_thread
    
    print(f"[异步管道] 准备启动异步处理: {video_path}")
    
    with thread_control['lock']:
        # 如果已有线程在运行，先停止
        if thread_control['running']:
            print("[异步管道] 检测到旧线程，先停止...")
            stop_async_processing()
        
        # 清空队列
        clear_queue(frame_queue)
        clear_queue(result_queue)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            raise Exception(f"无法打开视频文件: {video_path}")
        
        thread_control['video_cap'] = cap
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
        
        # 创建并启动推理线程
        inference_thread = threading.Thread(
            target=safe_thread_wrapper(inference_worker, "推理线程"),
            daemon=True,
            name="InferenceWorker"
        )
        
        reader_thread.start()
        inference_thread.start()
        
        print("[异步管道] ✓ 异步处理管道已启动")


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
    
    # 清理队列
    clear_queue(frame_queue)
    clear_queue(result_queue)
    
    # 重置标志
    thread_control['reader_alive'] = False
    thread_control['inference_alive'] = False
    
    print("[异步管道] ✓ 异步处理已停止")


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
            frame_queue.put(frame_data, block=True, timeout=1.0)
            
            # 更新读帧FPS
            if frame_count % 30 == 0:  # 每30帧统计一次
                elapsed = time.time() - reader_start_time
                performance_metrics['reader_fps'] = int(frame_count / elapsed)
                
        except queue.Full:
            performance_metrics['frame_drop_count'] += 1
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
        resized_frame, scale = resize_frame(frame)
        
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
                # 队列为空，检查线程状态
                if not thread_control['running']:
                    # 异步处理已停止
                    if result_queue.empty():
                        print("[显示线程] 队列为空且异步处理已停止，退出")
                        break
                    # 队列还有数据，继续消费
                    continue
                # 异步处理还在运行，继续等待
                continue
                
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
            
            # 计算平均置信度
            avg_confidence = int(np.mean(confidences) * 100) if confidences else 0
            
            # 更新全局统计数据
            model_stats['fps'] = int(current_fps)
            model_stats['latency'] = int(frame_latency)
            model_stats['confidence'] = avg_confidence
            model_stats['last_update'] = time.time()

            current_time = time.time()
            if len(centers) >= 2 and current_time - last_call_time >= CALL_INTERVAL:
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

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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


@video_bp.route('/video_feed')
def video_feed():
    """视频流式传输端点"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


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
    """获取模型性能统计数据"""
    global model_stats, video_active
    
    # 检查视频是否正在运行，如果超过3秒没有更新，认为已停止
    current_time = time.time()
    is_active = video_active and (current_time - model_stats.get('last_update', 0)) < 3
    
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
        "fps": model_stats.get('fps', 0),
        "latency": model_stats.get('latency', 0),
        "confidence": model_stats.get('confidence', 0)
    })
