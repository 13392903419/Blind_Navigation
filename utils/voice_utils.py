"""
语音相关工具模块
"""
import pyttsx3
import queue
import threading

# 全局变量
voices_cache = None
# 语音播放队列和锁，用于避免多线程冲突
speech_queue = queue.Queue()
speech_thread_started = False
speech_lock = threading.Lock()

# 紧急语音标志 - 用于打断当前播放
urgent_interrupt = threading.Event()
# 当前正在播放的引擎实例（用于紧急打断）
current_engine = None
current_engine_lock = threading.Lock()

# 语音完成回调函数 - 用于通知视频模块清除同步状态
speech_complete_callback = None


def set_speech_complete_callback(callback):
    """
    设置语音播放完成后的回调函数
    用于音画同步：语音结束后通知视频模块解除画面冻结
    
    Args:
        callback: 回调函数，无参数
    """
    global speech_complete_callback
    speech_complete_callback = callback
    print("[语音] 已设置语音完成回调函数")


def get_available_voices():
    """获取系统可用的语音列表"""
    global voices_cache
    if voices_cache is not None:
        return voices_cache

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    available_voices = []

    for voice in voices:
        voice_info = {
            'id': voice.id,
            'name': voice.name,
            'gender': '女声' if 'female' in voice.id.lower() or 'Microsoft Huihui' in voice.name else '男声'
        }
        available_voices.append(voice_info)

    voices_cache = available_voices
    return available_voices


def speech_worker():
    """
    语音播放工作线程，从队列中取出语音任务并播放
    这样可以避免多线程同时调用 pyttsx3 导致的冲突
    """
    print("[语音工作线程] 已启动")
    while True:
        try:
            # 从队列中获取任务
            task = speech_queue.get()
            
            if task is None:  # None 用于停止线程
                print("[语音工作线程] 收到停止信号")
                break
            
            text, user_settings, is_urgent = task
            
            print(f"[语音工作线程] 开始处理: '{text}' (紧急: {is_urgent})")
            _do_speak(text, user_settings, is_urgent)
            speech_queue.task_done()
        except Exception as e:
            print(f"[语音工作线程] 错误: {e}")
            import traceback
            traceback.print_exc()


def _do_speak(text, user_settings, is_urgent=False):
    """
    实际的语音合成和播放函数
    
    Args:
        text: 要播放的文本
        user_settings: 用户设置字典，包含 voice_speed 和 voice_volume
        is_urgent: 是否为紧急播报（紧急播报语速更快）
    """
    global current_engine
    try:
        print(f"[语音] 开始合成语音: '{text}' (紧急: {is_urgent})")
        local_engine = pyttsx3.init()
        
        # 保存当前引擎引用（用于紧急打断）
        with current_engine_lock:
            current_engine = local_engine

        # 获取可用语音列表
        voices = local_engine.getProperty('voices')

        # 优先查找中文语音
        found_chinese_voice = False
        selected_voice = None

        # 首先尝试找中文语音
        for voice in voices:
            voice_name = voice.name.lower()
            # 检查是否包含中文相关关键词
            if "chinese" in voice_name or "huihui" in voice_name or "china" in voice_name or "中文" in voice_name or "zhongwen" in voice_name:
                selected_voice = voice.id
                found_chinese_voice = True
                print(f"[语音] 找到中文语音: {voice.name}")
                break

        # 如果找不到中文语音，使用第一个可用的声音
        if not found_chinese_voice and len(voices) > 0:
            selected_voice = voices[0].id
            print(f"[语音] 未找到中文语音，使用第一个可用语音: {voices[0].name}")

        # 设置选定的声音
        if selected_voice:
            print(f"[语音] 最终使用语音ID: {selected_voice}")
            local_engine.setProperty('voice', selected_voice)
        else:
            print("[语音] 警告: 未找到可用语音")

        # 紧急播报使用更快的语速
        if is_urgent:
            local_engine.setProperty('rate', 280)  # 紧急时更快
            local_engine.setProperty('volume', 1.0)  # 最大音量
            print("[语音] 紧急模式：最快语速+最大音量")
        else:
            # 根据用户设置调整语音速度
            if user_settings["voice_speed"] == "慢":
                local_engine.setProperty('rate', 150)
            elif user_settings["voice_speed"] == "快":
                local_engine.setProperty('rate', 250)
            else:  # 中等
                local_engine.setProperty('rate', 200)

            # 设置音量
            volume_mapping = {
                "低": 0.5,
                "中等": 0.8,
                "高": 1.0
            }
            volume = volume_mapping.get(user_settings["voice_volume"], 0.8)
            local_engine.setProperty('volume', volume)

        # 实际播放语音
        print(f"[语音] 播放文本: {text}")
        local_engine.say(text)

        print("[语音] 开始runAndWait()...")
        local_engine.runAndWait()
        print("[语音] 播放完成")
        
        # 清除引擎引用
        with current_engine_lock:
            current_engine = None
        
        # 调用完成回调 - 通知视频模块清除同步状态
        if speech_complete_callback:
            try:
                speech_complete_callback()
                print("[语音] 已通知同步状态清除")
            except Exception as cb_e:
                print(f"[语音] 回调执行错误: {cb_e}")
        
        return True
    except Exception as e:
        print(f"[语音] 错误: {e}")
        import traceback
        traceback.print_exc()
        with current_engine_lock:
            current_engine = None
        return False


def speak(text, user_settings):
    """
    将语音任务添加到队列中，由工作线程处理
    
    Args:
        text: 要播放的文本
        user_settings: 用户设置字典，包含 voice_speed 和 voice_volume
    """
    global speech_thread_started
    
    # 启动语音工作线程（只启动一次）
    with speech_lock:
        if not speech_thread_started:
            worker_thread = threading.Thread(target=speech_worker, daemon=True)
            worker_thread.start()
            speech_thread_started = True
            print("[语音] 语音工作线程已启动")
    
    # 将任务添加到队列（普通优先级）
    print(f"[语音] 将任务添加到队列: '{text}'")
    speech_queue.put((text, user_settings, False))
    return True


def speak_urgent(text, user_settings):
    """
    紧急语音播报 - 清空队列并立即播报（最高优先级）
    用于危险情况（如检测到暴力行为）
    
    Args:
        text: 要播放的文本
        user_settings: 用户设置字典
    """
    global speech_thread_started
    
    print(f"[语音-紧急] ⚠️ 紧急播报请求: '{text}'")
    
    # 启动语音工作线程（只启动一次）
    with speech_lock:
        if not speech_thread_started:
            worker_thread = threading.Thread(target=speech_worker, daemon=True)
            worker_thread.start()
            speech_thread_started = True
            print("[语音] 语音工作线程已启动")
    
    # 清空当前队列中的所有待播任务
    cleared_count = 0
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
            cleared_count += 1
        except queue.Empty:
            break
    
    if cleared_count > 0:
        print(f"[语音-紧急] 已清空 {cleared_count} 条待播任务")
    
    # 将紧急任务放入队列头部（实际上队列已清空，直接放入即可）
    speech_queue.put((text, user_settings, True))
    print(f"[语音-紧急] 紧急任务已加入队列")
    return True


def get_prompt_template(user_settings):
    """
    获取提示模板（已简化）
    
    注意：Stage 1 优化后，语音播报已改为使用静态提示词库，
    本函数保留用于向后兼容，实际不再调用LLM。
    
    Args:
        user_settings: 用户设置字典
        
    Returns:
        str: 简单的提示文本
    """
    # Stage 1优化：不再调用LLM，直接返回简单提示
    return "正在进行语音导航"

