#!/usr/bin/env python3
"""
示例Python项目 - 简单的计算器
这是一个演示如何将代码上传到GitHub的示例项目
"""

def add(a, b):
    """加法运算"""
    return a + b

def subtract(a, b):
    """减法运算"""
    return a - b

def multiply(a, b):
    """乘法运算"""
    return a * b

def divide(a, b):
    """除法运算"""
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b

def main():
    """主函数"""
    print("欢迎使用简单计算器！")
    print("支持的操作：+, -, *, /")
    
    while True:
        try:
            num1 = float(input("请输入第一个数字: "))
            operator = input("请输入运算符 (+, -, *, /) 或 'q' 退出: ")
            
            if operator.lower() == 'q':
                print("再见！")
                break
                
            num2 = float(input("请输入第二个数字: "))
            
            if operator == '+':
                result = add(num1, num2)
            elif operator == '-':
                result = subtract(num1, num2)
            elif operator == '*':
                result = multiply(num1, num2)
            elif operator == '/':
                result = divide(num1, num2)
            else:
                print("无效的运算符！")
                continue
                
            print(f"结果: {num1} {operator} {num2} = {result}")
            
        except ValueError as e:
            print(f"错误: {e}")
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    main()

