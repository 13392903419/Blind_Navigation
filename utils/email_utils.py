"""
邮件和验证码工具模块
"""
import random
import string
import smtplib
import time
import re
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
from config import EMAIL_CONFIG

# 验证码存储
verification_codes = {}  # 格式: {email: {'code': '123456', 'expires': timestamp}}


def generate_verification_code(length=6):
    """生成指定长度的数字验证码"""
    return ''.join(random.choices(string.digits, k=length))


def is_valid_email(email):
    """简单验证邮箱格式"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def send_verification_email(to_email, verification_code):
    """发送验证码邮件"""
    try:
        # 创建HTML邮件内容（关键点：使用HTML格式并添加样式）
        html_content = f"""
        <html>
            <head>
                <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
            </head>
            <body>
                <p style="font-size: 16px; color: #333;">您的验证码是：</p>
                <div style="
                    font-size: 24px;
                    color: #ff4444;
                    font-weight: bold;
                    margin: 10px 0;
                    padding: 12px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    display: inline-block;
                ">{verification_code}</div>
                <p style="font-size: 14px; color: #666; margin-top: 10px;">
                    验证码10分钟内有效，请勿告知他人。如果这不是您本人的操作，请忽略此邮件。
                </p>
            </body>
        </html>
        """

        # 使用MIMEText指定HTML类型
        message = MIMEText(html_content, 'html', 'utf-8')

        # 规范发件人格式
        message['From'] = formataddr(("盲道导航助手", EMAIL_CONFIG['sender']))
        message['To'] = Header(to_email)
        message['Subject'] = Header('【盲道导航助手】验证码', 'utf-8')

        # 建立连接并发送邮件
        server = smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['sender'], [to_email], message.as_string())
        server.quit()

        # 保存验证码，设置10分钟有效期
        verification_codes[to_email] = {
            'code': verification_code,
            'expires': time.time() + 600  # 10分钟后过期
        }

        return True, "验证码已发送"
    except Exception as e:
        print(f"发送邮件失败: {e}")
        return False, f"发送验证码失败: {str(e)}"


def verify_code(email, code):
    """验证邮箱验证码"""
    if email not in verification_codes:
        return False, "验证码不存在或已过期"

    stored_data = verification_codes[email]
    current_time = time.time()

    # 检查验证码是否过期
    if current_time > stored_data['expires']:
        del verification_codes[email]  # 删除过期验证码
        return False, "验证码已过期"

    # 验证码是否匹配
    if stored_data['code'] != code:
        return False, "验证码错误"

    # 验证通过后删除验证码（一次性使用）
    del verification_codes[email]
    return True, "验证成功"

