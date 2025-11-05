"""
装饰器模块
"""
import functools
from flask import session, redirect, url_for


def login_required(f):
    """验证登录的装饰器"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)

    return decorated_function

