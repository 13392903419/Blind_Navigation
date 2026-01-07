"""
视频相关路由 - 视频上传、处理、流式传输
"""
from flask import Blueprint, Response, request, jsonify
import cv2
import numpy as np
import ollama
import threading
import time
import os
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

# 加载YOLO模型
model = YOLO(MODEL_WEIGHTS)

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

# 转向提示问题
right_turn_question = "请用亲切且简短的话语告知要往右拐，因为盲道是往右拐的"
left_turn_question = "请用亲切且简短的话语告知要往左拐，因为盲道是往左拐的"


def get_user_settings_for_video():
    """
    获取用户设置（从 routes.main 模块）
    注意：这个函数在视频流生成器中调用，无法访问 session
    所以使用全局变量
    """
    from routes.main import user_settings
    return user_settings


def generate_frames():
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

            # 处理视频帧 - YOLO检测
            results = model(frame)
            centers = []  # 存储所有检测框的 (center_x, center_y)
            confidences = []  # 存储所有检测框的置信度

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    centers.append((center_x, center_y))
                    confidences.append(float(conf))

                    class_names = model.names
                    label = f"{class_names[cls]}: {conf:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
