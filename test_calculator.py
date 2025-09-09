#!/usr/bin/env python3
"""
计算器测试文件
演示如何添加新文件并推送到GitHub
"""

import unittest
from main import add, subtract, multiply, divide

class TestCalculator(unittest.TestCase):
    """计算器功能测试类"""
    
    def test_add(self):
        """测试加法功能"""
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)
    
    def test_subtract(self):
        """测试减法功能"""
        self.assertEqual(subtract(5, 3), 2)
        self.assertEqual(subtract(1, 1), 0)
        self.assertEqual(subtract(0, 5), -5)
    
    def test_multiply(self):
        """测试乘法功能"""
        self.assertEqual(multiply(3, 4), 12)
        self.assertEqual(multiply(-2, 3), -6)
        self.assertEqual(multiply(0, 5), 0)
    
    def test_divide(self):
        """测试除法功能"""
        self.assertEqual(divide(10, 2), 5)
        self.assertEqual(divide(7, 2), 3.5)
        
        # 测试除零错误
        with self.assertRaises(ValueError):
            divide(5, 0)

if __name__ == "__main__":
    unittest.main()

