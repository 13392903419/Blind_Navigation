"""
配置文件示例 - 请复制此文件为 config.py 并填入您的实际配置
"""

# Flask应用配置
SECRET_KEY = '13392903419'  # 请更改为随机字符串，用于session加密

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '@ACC13953201086',  # 填写您的MySQL密码
    'db': 'blind_navigation',
    'charset': 'utf8mb4',
}

# 邮件发送配置
EMAIL_CONFIG = {
    'sender': '2030399660@qq.com',  # 填写您的QQ邮箱
    'password': 'umqbwgwjfheddfja',  # 填写QQ邮箱授权码（不是QQ密码）
    'smtp_server': 'smtp.qq.com',
    'smtp_port': 465
}

# 文件上传配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 限制上传大小为300MB

# YOLO模型配置
MODEL_WEIGHTS = 'yolo/best.pt'  # 使用项目包含的预训练YOLOv8盲道检测模型

# 百度地图MCP配置
# 重要：使用百度地图前，必须在 https://lbsyun.baidu.com/apiconsole/key 中配置 Referer 白名单：
# 
# 白名单格式要求（重要！不要用 http:// 开头）：
#   - 本地开发：*localhost* 或 *127.0.0.1*
#   - 多个域名：*.mysite.com*,*myapp.com*（用英文逗号分隔）
#   - 不限制（谨慎）：* （注意：容易泄露AK给其他网站，线上不建议）
# 
# 常见错误（不要这样做）：
#   ❌ http://localhost:5000  （错误：包含协议和端口）
#   ❌ localhost:5000          （错误：缺少通配符）
#   ✓  *localhost*             （正确）
#   ✓  *127.0.0.1*             （正确）
#
# 配置步骤：
# 1. 访问：https://lbsyun.baidu.com/apiconsole/key
# 2. 选择你的应用和 API Key，点击"编辑"
# 3. 在"应用授权"→"Referer 白名单"中填入上述格式
# 4. 勾选"Web服务"和"浏览器端"权限
# 5. 保存后刷新页面测试
BAIDU_MAP_CONFIG = {
    'api_key': 'JdpmYDQsldEe0886JxFHqOqXFjuXtRd6',  # 填写您的百度地图API密钥
    'base_url': 'https://api.map.baidu.com',
    'web_service_url': 'https://api.map.baidu.com/geocoding/v3/',
    'direction_url': 'https://api.map.baidu.com/direction/v2/',
    'place_search_url': 'https://api.map.baidu.com/place/v2/search'
}

# DeepSeek AI配置
DEEPSEEK_CONFIG = {
    'api_key': 'sk-e35c0b6306cf4a01b5502a73d093e5c2',  # 填写您的DeepSeek API密钥
    'base_url': 'https://api.deepseek.com/chat/completions',
    'model': 'deepseek-chat'
}

# 视频检测配置
THRESHOLD_SLOPE = 0.41  # 盲道方向检测斜率阈值
CALL_INTERVAL = 14  # 语音提示间隔（秒）

# 用户默认设置
DEFAULT_USER_SETTINGS = {
    "gender": "女",  # 性别：男/女/未指定
    "name": "刘宇菲",  # 用户名称
    "age": "青年",  # 年龄段：青年/中年/老年/未指定
    "voice_speed": "中等",  # 语音速度：慢/中等/快
    "voice_volume": "中等",  # 语音音量：低/中等/高
    "user_mode": "盲人端",  # 用户模式：盲人端/家属端
    "encourage": "开"  # 适当时给予鼓励：开/关
}



