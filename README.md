# 简单计算器项目

这是一个用Python编写的简单计算器程序，用于演示如何将代码上传到GitHub。

## 功能特性

- 支持基本的四则运算（加、减、乘、除）
- 错误处理和输入验证
- 交互式命令行界面
- 简洁的代码结构
- **新增**: 4个AI研究示例，展示最新AI研究的核心概念

## AI研究示例

本项目包含4个AI研究示例，每个示例都复现了重要的AI研究概念并快速验证结果：

### 1. 注意力机制 (Attention Mechanism)
- 文件: `ai_research_example_1_attention.py`
- 复现: Transformer中的缩放点积注意力
- 论文参考: "Attention Is All You Need" (Vaswani et al., 2017)
- 验证内容:
  - 注意力权重求和为1.0
  - 输出维度正确
  - 注意力权重非负

### 2. 梯度下降优化 (Gradient Descent)
- 文件: `ai_research_example_2_gradient_descent.py`
- 复现: 基本梯度下降算法
- 应用: 所有现代深度学习优化的基础
- 验证内容:
  - 收敛到真实最小值
  - 函数值单调递减
  - 最终迭代中保持稳定

### 3. 神经网络前向传播 (Neural Network Forward Pass)
- 文件: `ai_research_example_3_neural_network.py`
- 复现: 前馈神经网络计算
- 应用: 所有深度学习架构的基础
- 验证内容:
  - 输出和隐藏层维度正确
  - ReLU激活函数工作正常
  - 网络产生非零输出

### 4. Softmax和交叉熵损失 (Softmax & Cross-Entropy)
- 文件: `ai_research_example_4_softmax_loss.py`
- 复现: 分类任务的损失计算
- 应用: 所有神经网络分类任务
- 验证内容:
  - 概率和为1.0
  - 所有概率在[0,1]范围内
  - 完美预测的损失接近0

### 运行AI研究示例

```bash
# 运行单个示例
python ai_research_example_1_attention.py
python ai_research_example_2_gradient_descent.py
python ai_research_example_3_neural_network.py
python ai_research_example_4_softmax_loss.py

# 运行所有测试（包括AI研究示例测试）
python -m unittest test_ai_research_examples.py
```

## 使用方法

### 计算器使用

1. 确保您的系统已安装Python 3.x
2. 安装依赖：
   ```bash
   pip install numpy
   ```
3. 运行计算器程序：
   ```bash
   python main.py
   ```
4. 按照提示输入数字和运算符
5. 输入 'q' 退出程序

## 项目结构

```
my-sample-project/
├── main.py                              # 主计算器程序文件
├── config.py                            # 计算器配置
├── history.py                           # 历史记录模块
├── test_calculator.py                   # 计算器测试
├── ai_research_example_1_attention.py   # AI示例1: 注意力机制
├── ai_research_example_2_gradient_descent.py  # AI示例2: 梯度下降
├── ai_research_example_3_neural_network.py    # AI示例3: 神经网络
├── ai_research_example_4_softmax_loss.py      # AI示例4: Softmax损失
├── test_ai_research_examples.py         # AI示例测试
├── README.md                            # 项目说明文档
└── .gitignore                          # Git忽略文件配置
```

## 贡献

欢迎提交问题和改进建议！

## 许可证

MIT License

