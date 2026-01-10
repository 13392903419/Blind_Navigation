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
            text, user_settings = speech_queue.get()
            
            if text is None:  # None 用于停止线程
                print("[语音工作线程] 收到停止信号")
                break
            
            print(f"[语音工作线程] 开始处理: '{text}'")
            _do_speak(text, user_settings)
            speech_queue.task_done()
        except Exception as e:
            print(f"[语音工作线程] 错误: {e}")
            import traceback
            traceback.print_exc()


def _do_speak(text, user_settings):
    """
    实际的语音合成和播放函数
    
    Args:
        text: 要播放的文本
        user_settings: 用户设置字典，包含 voice_speed 和 voice_volume
    """
    try:
        print(f"[语音] 开始合成语音: '{text}'")
        local_engine = pyttsx3.init()

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
        return True
    except Exception as e:
        print(f"[语音] 错误: {e}")
        import traceback
        traceback.print_exc()
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
    
    # 将任务添加到队列
    print(f"[语音] 将任务添加到队列: '{text}'")
    speech_queue.put((text, user_settings))
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

