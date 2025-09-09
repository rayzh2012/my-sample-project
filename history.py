#!/usr/bin/env python3
"""
计算器历史记录模块
用于演示分支开发功能
"""

from datetime import datetime
from config import get_config

class CalculatorHistory:
    """计算器历史记录类"""
    
    def __init__(self):
        self.history = []
        self.max_history = get_config("max_history") or 10
    
    def add_record(self, operation, operand1, operand2=None, result=None):
        """添加计算记录"""
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": operation,
            "operand1": operand1,
            "operand2": operand2,
            "result": result
        }
        
        self.history.append(record)
        
        # 保持历史记录数量在限制内
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_history(self, count=None):
        """获取历史记录"""
        if count is None:
            return self.history
        return self.history[-count:]
    
    def clear_history(self):
        """清空历史记录"""
        self.history.clear()
    
    def display_history(self):
        """显示历史记录"""
        if not self.history:
            print("暂无历史记录")
            return
        
        print("\n=== 计算历史记录 ===")
        for i, record in enumerate(self.history, 1):
            timestamp = record["timestamp"]
            operation = record["operation"]
            operand1 = record["operand1"]
            operand2 = record["operand2"]
            result = record["result"]
            
            if operand2 is not None:
                print(f"{i}. [{timestamp}] {operand1} {operation} {operand2} = {result}")
            else:
                print(f"{i}. [{timestamp}] {operation}({operand1}) = {result}")
        print("=" * 25)

# 全局历史记录实例
calculator_history = CalculatorHistory()

