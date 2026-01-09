"""
视频处理工具模块
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from werkzeug.utils import secure_filename
from config import ALLOWED_EXTENSIONS


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_error_frame(message):
    """创建错误信息帧 - 使用PIL支持中文"""
    img = Image.new('RGB', (640, 480), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("simhei.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), message, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((640 - text_width) // 2, (480 - text_height) // 2)

    draw.text(position, message, font=font, fill=(255, 0, 0))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def create_info_frame(message):
    """创建信息提示帧 - 使用PIL支持中文"""
    img = Image.new('RGB', (640, 480), color=(41, 128, 185))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("simhei.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), message, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((640 - text_width) // 2, (480 - text_height) // 2)

    draw.text(position, message, font=font, fill=(255, 255, 255))

    try:
        small_font = ImageFont.truetype("simhei.ttf", 20)
    except IOError:
        small_font = ImageFont.load_default()

    help_text = "支持mp4, avi, mov, mkv, webm格式"
    bbox = draw.textbbox((0, 0), help_text, font=small_font)
    help_width = bbox[2] - bbox[0]
    help_height = bbox[3] - bbox[1]
    help_position = ((640 - help_width) // 2, position[1] + text_height + 20)
    draw.text(help_position, help_text, font=small_font, fill=(200, 200, 200))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

