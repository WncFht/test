---
title: fully_connected_network
date: 2024-10-23T10:11:41+0800
modify: 2024-12-06T00:10:21+0800
categories: Computer
dir: Obsidian
share: true
tags:
  - AI
  - Computer
  - Advanced
  - EECS 498-007
---

```python
"""
实现 PyTorch 中的全连接神经网络。
注意: 在每个实现块中不应使用 ".to()" 或 ".cuda()" 方法。
"""
import torch
import random
from a3_helper import svm_loss, softmax_loss
from eecs598 import Solver

def hello_fully_connected_networks():
    """
    示例函数，用于确保环境在 Google Colab 上正确设置。
    """
    print('Hello from fully_connected_networks.py!')

class Linear(object):
    @staticmethod
    def forward(x, w, b):
        """
        计算线性(全连接)层的前向传播。
        
        输入:
        - x: 形状为 (N, d_1, ..., d_k) 的输入数据张量，包含 N 个样本
        - w: 形状为 (D, M) 的权重张量
        - b: 形状为 (M,) 的偏置张量
        
        返回:
        - out: 形状为 (N, M) 的输出
        - cache: 缓存 (x, w, b) 用于反向传播
        """
        out = None
        # 将输入reshape成二维张量并进行矩阵乘法，然后加上偏置
        out = x.view(x.shape[0],-1).mm(w)+b
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        计算线性层的反向传播。
        
        输入:
        - dout: 上游导数，形状为 (N, M)
        - cache: 前向传播时的缓存元组 (x, w, b)
        
        返回:
        - dx: 相对于x的梯度
        - dw: 相对于w的梯度
        - db: 相对于b的梯度
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        
        # 计算各个参数的梯度
        db = dout.sum(dim = 0)  # 偏置的梯度
        dx = dout.mm(w.t()).view(x.shape)  # 输入的梯度
        dw = x.view(x.shape[0],-1).t().mm(dout)  # 权重的梯度
        
        return dx, dw, db

class ReLU(object):
    @staticmethod
    def forward(x):
        """
        计算ReLU激活函数的前向传播。
        
        输入:
        - x: 任意形状的输入张量
        
        返回:
        - out: 与x同形状的输出
        - cache: x (用于反向传播)
        """
        out = x.clone()  # 创建输入的副本
        out[out<0] = 0   # 将小于0的值置为0
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        计算ReLU激活函数的反向传播。
        
        输入:
        - dout: 上游导数
        - cache: 前向传播时的输入x
        
        返回:
        - dx: 相对于x的梯度
        """
        dx, x = None, cache
        dx = dout.clone()
        dx[x<0] = 0  # ReLU反向传播：当输入小于0时梯度为0
        return dx

class Linear_ReLU(object):
    """
    将线性层和ReLU激活函数组合在一起的便捷层
    """
    @staticmethod
    def forward(x, w, b):
        """
        依次执行线性变换和ReLU激活
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        线性层-ReLU组合的反向传播
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db

class TwoLayerNet(object):
    """
    双层全连接神经网络，具有ReLU非线性激活和softmax损失。
    架构为: linear - relu - linear - softmax
    """
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0, dtype=torch.float32, device='cpu'):
        """
        初始化网络参数:
        - input_dim: 输入维度
        - hidden_dim: 隐藏层维度
        - num_classes: 分类数量
        - weight_scale: 权重初始化的标准差
        - reg: L2正则化强度
        """
        self.params = {}
        self.reg = reg
        
        # 初始化第一层权重和偏置
        self.params['W1'] = torch.zeros(input_dim, hidden_dim, dtype=dtype, device=device)
        self.params['W1'] += weight_scale*torch.randn(input_dim, hidden_dim, dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        
        # 初始化第二层权重和偏置
        self.params['W2'] = torch.zeros(hidden_dim, num_classes, dtype=dtype, device=device)
        self.params['W2'] += weight_scale*torch.randn(hidden_dim, num_classes, dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(num_classes, dtype=dtype, device=device)

class TwoLayerNet(object):
    def loss(self, X, y=None):
        """
        计算损失和梯度。
        
        输入:
        - X: 形状为 (N, d_1, ..., d_k) 的输入数据
        - y: 形状为 (N,) 的标签数组
        
        返回:
        - 如果y为None，返回预测分数
        - 如果y不为None，返回 (loss, grads) 元组
        """
        scores = None
        # 前向传播
        out_LR, cache_LR = Linear_ReLU.forward(X, self.params['W1'], self.params['b1'])
        scores, cache_L = Linear.forward(out_LR, self.params['W2'], self.params['b2'])

        if y is None:
            return scores

        loss, grads = 0, {}
        
        # 计算损失和梯度
        loss, dout = softmax_loss(scores, y)
        # 添加L2正则化损失
        loss += ((self.params['W1']*self.params['W1']).sum() + 
                (self.params['W2']*self.params['W2']).sum()) * self.reg
        
        # 反向传播
        dx, dw, db = Linear.backward(dout, cache_L)
        grads['W2'] = dw + 2*self.params['W2']*self.reg
        grads['b2'] = db
        
        dx, dw, db = Linear_ReLU.backward(dx, cache_LR)
        grads['W1'] = dw + 2*self.params['W1']*self.reg
        grads['b1'] = db

        return loss, grads

class FullyConnectedNet(object):
    """
    具有任意隐藏层数的全连接神经网络。
    架构: {linear - relu - [dropout]} x (L - 1) - linear - softmax
    """
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        初始化多层神经网络:
        - hidden_dims: 每个隐藏层的维度列表
        - input_dim: 输入维度
        - num_classes: 分类数量
        - dropout: dropout概率
        - reg: L2正则化强度
        - weight_scale: 权重初始化的标准差
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # 初始化所有层的权重和偏置
        last_dim = input_dim
        for n, hidden_dim in enumerate(hidden_dims):
            i = n+1
            self.params[f'W{i}'] = weight_scale * torch.randn(last_dim, hidden_dim, dtype=dtype, device=device)
            self.params[f'b{i}'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
            last_dim = hidden_dim
            
        # 最后一层（输出层）的初始化
        i += 1
        self.params[f'W{i}'] = weight_scale * torch.randn(last_dim, num_classes, dtype=dtype, device=device)
        self.params[f'b{i}'] = torch.zeros(num_classes, dtype=dtype, device=device)
        
        # dropout参数设置
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

class Dropout(object):
    @staticmethod
    def forward(x, dropout_param):
        """
        实现(反向)dropout的前向传播。
        
        输入:
        - x: 输入数据
        - dropout_param: 包含 mode('train' 或 'test') 和 p(dropout概率) 的字典
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])

        mask = None
        out = None

        if mode == 'train':
            # 训练模式：创建dropout掩码并应用
            mask = torch.rand(x.shape) > p
            out = x.clone()
            out[mask] = 0
        elif mode == 'test':
            # 测试模式：直接返回输入
            out = x

        cache = (dropout_param, mask)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        实现dropout的反向传播。
        
        输入:
        - dout: 上游导数
        - cache: (dropout_param, mask) 从前向传播的缓存
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        if mode == 'train':
            # 训练模式：使用相同的mask来反向传播梯度
            dx = dout.clone()
            dx[mask] = 0
        elif mode == 'test':
            dx = dout
        return dx
```