#!/usr/bin/env python3
"""
计算器配置文件
用于演示添加新文件并更新到GitHub
"""

# 计算器配置
CALCULATOR_CONFIG = {
    "precision": 6,  # 小数点精度
    "max_history": 10,  # 最大历史记录数
    "welcome_message": "欢迎使用高级计算器！",
    "supported_operations": ["+", "-", "*", "/", "^", "sqrt"],
    "error_messages": {
        "division_by_zero": "错误：除数不能为零",
        "negative_sqrt": "错误：负数无法计算平方根",
        "invalid_operator": "错误：无效的运算符",
        "invalid_input": "错误：输入格式不正确"
    }
}

# 颜色配置（用于终端输出）
COLORS = {
    "reset": "\033[0m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "purple": "\033[95m",
    "cyan": "\033[96m"
}

def get_config(key):
    """获取配置值"""
    return CALCULATOR_CONFIG.get(key)

def get_color(color_name):
    """获取颜色代码"""
    return COLORS.get(color_name, COLORS["reset"])

