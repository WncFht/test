---
title: convolutional_network
date: 2024-10-23T11:17:16+0800
modify: 2024-12-06T00:10:20+0800
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
实现PyTorch中的卷积神经网络。
警告：在每个实现块中不应使用 ".to()" 或 ".cuda()" 方法。
"""
import torch
import random
from eecs598 import Solver
from a3_helper import svm_loss, softmax_loss
from fully_connected_networks import *

def hello_convolutional_networks():
    """
    示例函数，用于确保环境在Google Colab上正确设置。
    """
    print('Hello from convolutional_networks.py!')

class Conv(object):
    @staticmethod
    def forward(x, w, b, conv_param):
        """
        卷积层前向传播的朴素实现。
        
        输入:
        - x: 形状为 (N, C, H, W) 的输入数据，其中:
            N: 批量大小
            C: 输入通道数
            H: 图像高度
            W: 图像宽度
        - w: 形状为 (F, C, HH, WW) 的卷积核权重
        - b: 形状为 (F,) 的偏置项
        - conv_param: 包含以下键的字典：
            'stride': 步长
            'pad': 填充大小
            
        返回:
        - out: 输出数据
        - cache: (x, w, b, conv_param)
        """
        out = None
        # 获取参数
        pad = conv_param['pad']
        stride = conv_param['stride']
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        
        # 计算输出维度
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
        
        # 对输入进行填充
        x = torch.nn.functional.pad(x, (pad,pad,pad,pad))
        
        # 初始化输出
        out = torch.zeros((N,F,H_out,W_out), dtype=x.dtype, device=x.device)
        
        # 执行卷积操作
        for n in range(N):          # 遍历每个样本
            for f in range(F):      # 遍历每个卷积核
                for height in range(H_out):    # 遍历输出高度
                    for width in range(W_out): # 遍历输出宽度
                        # 计算卷积
                        out[n,f,height,width] = (x[n,:,height*stride:height*stride+HH,
                                                width*stride:width*stride+WW] * w[f]).sum() + b[f]
        
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        卷积层反向传播的朴素实现。
        
        输入:
        - dout: 上游梯度
        - cache: (x, w, b, conv_param)
        
        返回:
        - dx: 相对于x的梯度
        - dw: 相对于w的梯度
        - db: 相对于b的梯度
        """
        x, w, b, conv_param = cache
        pad = conv_param['pad']
        stride = conv_param['stride']
        N, F, H_dout, W_dout = dout.shape
        F, C, HH, WW = w.shape
        
        # 初始化梯度
        db = torch.zeros_like(b)
        dw = torch.zeros_like(w)
        dx = torch.zeros_like(x)
        
        # 计算梯度
        for n in range(N):
            for f in range(F):
                for height in range(H_dout):
                    for width in range(W_dout):
                        # 计算偏置的梯度
                        db[f] += dout[n,f,height,width]
                        # 计算权重的梯度
                        dw[f] += x[n,:,height*stride:height*stride+HH,
                                width*stride:width*stride+WW] * dout[n,f,height,width]
                        # 计算输入的梯度
                        dx[n,:,height*stride:height*stride+HH,
                           width*stride:width*stride+WW] += w[f] * dout[n,f,height,width]
        
        # 移除填充部分
        dx = dx[:,:,1:-1,1:-1]
        
        return dx, dw, db

class MaxPool(object):
    @staticmethod
    def forward(x, pool_param):
        """
        最大池化层的前向传播实现。
        
        输入:
        - x: 形状为 (N, C, H, W) 的输入数据
        - pool_param: 包含以下键的字典：
            'pool_height': 池化窗口高度
            'pool_width': 池化窗口宽度
            'stride': 步长
            
        返回:
        - out: 输出数据
        - cache: (x, pool_param)
        """
        stride = pool_param['stride']
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']
        N, C, H, W = x.shape
        
        # 计算输出维度
        H_out = int(1 + (H - pool_height) / stride)
        W_out = int(1 + (W - pool_width) / stride)
        
        # 初始化输出
        out = torch.zeros((N,C,H_out,W_out), dtype=x.dtype, device=x.device)
        
        # 执行最大池化
        for n in range(N):
            for height in range(H_out):
                for width in range(W_out):
                    # 获取池化窗口中的最大值
                    val, _ = x[n,:,height*stride:height*stride+pool_height,
                             width*stride:width*stride+pool_width].reshape(C,-1).max(dim=1)
                    out[n,:,height,width] = val
                    
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        最大池化层的反向传播实现。
        
        输入:
        - dout: 上游梯度
        - cache: (x, pool_param)
        
        返回:
        - dx: 相对于输入x的梯度
        """
        x, pool_param = cache
        N, C, H, W = x.shape
        stride = pool_param['stride']
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']
        
        # 计算输出维度
        H_out = int(1 + (H - pool_height) / stride)
        W_out = int(1 + (W - pool_width) / stride)
        
        # 初始化梯度
        dx = torch.zeros_like(x)
        
        # 计算梯度
        for n in range(N):
            for height in range(H_out):
                for width in range(W_out):
                    # 获取当前池化窗口
                    local_x = x[n,:,height*stride:height*stride+pool_height,
                              width*stride:width*stride+pool_width]
                    shape_local_x = local_x.shape
                    reshaped_local_x = local_x.reshape(C,-1)
                    
                    # 找到最大值的位置
                    local_dw = torch.zeros_like(reshaped_local_x)
                    _, indices = reshaped_local_x.max(-1)
                    
                    # 仅在最大值位置传递梯度
                    local_dw[range(C),indices] = dout[n,:,height,width]
                    dx[n,:,height*stride:height*stride+pool_height,
                       width*stride:width*stride+pool_width] = local_dw.reshape(shape_local_x)
        
        return dx

class ThreeLayerConvNet(object):
    """
    三层卷积神经网络，架构为：
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    """
    def __init__(self, input_dims=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=torch.float, device='cpu'):
        """
        初始化三层卷积网络。
        
        参数:
        - input_dims: 输入数据维度 (C, H, W)
        - num_filters: 卷积核数量
        - filter_size: 卷积核大小
        - hidden_dim: 隐藏层的神经元数量
        - num_classes: 分类类别数
        - weight_scale: 权重初始化的标准差
        - reg: L2正则化强度
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # 设置卷积和池化参数
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        C, H, W = input_dims
        
        # 计算各层输出维度
        HH = filter_size
        WW = filter_size
        H_out_conv = int(1 + (H + 2 * conv_param['pad'] - HH) / conv_param['stride'])
        W_out_conv = int(1 + (W + 2 * conv_param['pad'] - WW) / conv_param['stride'])
        H_out = int(1 + (H_out_conv - pool_param['pool_height']) / pool_param['stride'])
        W_out = int(1 + (W_out_conv - pool_param['pool_width']) / pool_param['stride'])
        
        # 初始化第一层（卷积层）参数
        self.params['W1'] = weight_scale * torch.randn(num_filters, C, filter_size, filter_size, 
                                                      dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(num_filters, dtype=dtype, device=device)
        
        # 初始化第二层（全连接层）参数
        self.params['W2'] = weight_scale * torch.randn(num_filters*H_out*W_out, hidden_dim, 
                                                      dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        
        # 初始化第三层（全连接层）参数
        self.params['W3'] = weight_scale * torch.randn(hidden_dim, num_classes, 
                                                      dtype=dtype, device=device)
        self.params['b3'] = torch.zeros(num_classes, dtype=dtype, device=device)

    def loss(self, X, y=None):
        """
        计算损失和梯度。
        
        输入:
        - X: 输入数据
        - y: 标签
        
        返回:
        - 如果y是None，返回预测分数
        - 否则返回 (loss, grads) 元组
        """
        # 设置卷积和池化参数
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        # 前向传播
        out_CRP, cache_CRP = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        out_LR, cache_LR = Linear_ReLU.forward(out_CRP, W2, b2)
        scores, cache_L = Linear.forward(out_LR, W3, b3)
        
        if y is None:
            return scores
            
        # 计算损失和梯度
        loss, grads = 0.0, {}
        
        # 计算softmax损失
        loss, dout = softmax_loss(scores, y)
        
        # 添加L2正则化
        for i in range(1,4):
            loss += (self.params[f'W{i}']*self.params[f'W{i}']).sum()*self.reg
            
        # 反向传播
        # 第三层（全连接层）
        last_dout, dw, db = Linear.backward(dout, cache_L)
        grads['W3'] = dw + 2*self.params['W3']*self.reg
        grads['b3'] = db
        
        # 第二层（全连接层+ReLU）
        last_dout, dw, db = Linear_ReLU.backward(last_dout, cache_LR)
        grads['W2'] = dw + 2*self.params['W2']*self.reg
        grads['b2'] = db
        
        # 第一层（卷积+ReLU+池化）
        last_dout, dw, db = Conv_ReLU_Pool.backward(last_dout, cache_CRP)
        grads['W1'] = dw + 2*self.params['W1']*self.reg
        grads['b1'] = db
        
        return loss, grads

class DeepConvNet(object):
    """
    VGG风格的深度卷积神经网络，具有任意数量的卷积层。
    所有卷积层使用3x3卷积核和padding=1以保持特征图大小。
    所有池化层使用2x2最大池化，步长为2，用于将特征图尺寸减半。

    网络架构:
    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    每个{...}结构是一个"宏层"，包含：
    - 卷积层
    - 可选的批归一化层
    - ReLU激活
    - 可选的池化层
    """
    def __init__(self, input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10, weight_scale=1e-3, reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float, device='cpu'):
        """
        初始化深度卷积网络。

        参数:
        - input_dims: 输入数据的维度 (C, H, W)
        - num_filters: 长度为(L-1)的列表，指定每个宏层的卷积核数量
        - max_pools: 整数列表，指定应该有最大池化的宏层索引
        - batchnorm: 是否在每个宏层中包含批归一化
        - num_classes: 分类数量
        - weight_scale: 随机初始化权重的标准差，或使用"kaiming"初始化
        - reg: L2正则化强度
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        # 初始化网络参数
        filter_size = 3
        conv_param = {'stride': 1, 'pad': 1}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        # 跟踪特征图尺寸变化
        pred_filters, H_out, W_out = input_dims
        HH = filter_size
        WW = filter_size

        # 初始化每一层的参数
        for i, num_filter in enumerate(num_filters):
            # 计算当前层的输出尺寸
            H_out = int(1 + (H_out + 2 * conv_param['pad'] - HH) / conv_param['stride'])
            W_out = int(1 + (W_out + 2 * conv_param['pad'] - WW) / conv_param['stride'])
            
            # 如果使用批归一化，初始化gamma和beta参数
            if self.batchnorm:
                self.params[f'gamma{i}'] = 0.01 * torch.randn(num_filter, device=device, dtype=dtype)
                self.params[f'beta{i}'] = 0.01 * torch.randn(num_filter, device=device, dtype=dtype)
            
            # 如果当前层需要池化，调整输出尺寸
            if i in max_pools:
                H_out = int(1 + (H_out - pool_param['pool_height']) / pool_param['stride'])
                W_out = int(1 + (W_out - pool_param['pool_width']) / pool_param['stride'])
            
            # 初始化卷积层权重
            if weight_scale == 'kaiming':
                self.params[f'W{i}'] = kaiming_initializer(num_filter, pred_filters, 
                                                         K=filter_size, relu=True, 
                                                         device=device, dtype=dtype)
            else:
                self.params[f'W{i}'] = weight_scale * torch.randn(num_filter, pred_filters, 
                                                                 filter_size, filter_size, 
                                                                 dtype=dtype, device=device)
            
            # 初始化偏置
            self.params[f'b{i}'] = torch.zeros(num_filter, dtype=dtype, device=device)
            pred_filters = num_filter

        # 初始化最后一层（全连接层）
        i += 1
        if weight_scale == 'kaiming':
            self.params[f'W{i}'] = kaiming_initializer(num_filter*H_out*W_out, num_classes,
                                                     relu=False, device=device, dtype=dtype)
        else:
            self.params[f'W{i}'] = weight_scale * torch.randn(num_filter*H_out*W_out, 
                                                             num_classes, dtype=dtype, device=device)
        self.params[f'b{i}'] = torch.zeros(num_classes, dtype=dtype, device=device)

# 初始化批归一化参数
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]

    def loss(self, X, y=None):
        """
        评估深度卷积网络的损失和梯度。

        输入:
        - X: 输入数据
        - y: 标签

        返回:
        - 如果y是None，返回预测分数
        - 否则返回 (loss, grads) 元组
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # 设置批归一化层的模式
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        # 设置卷积和池化参数
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # 前向传播
        cache = {}
        out = X
        
        # 通过所有中间层
        for i in range(self.num_layers-1):
            if i in self.max_pools:
                if self.batchnorm:
                    # 使用卷积-批归一化-ReLU-池化层
                    out, cache[f'{i}'] = Conv_BatchNorm_ReLU_Pool.forward(
                        out, self.params[f'W{i}'], self.params[f'b{i}'],
                        self.params[f'gamma{i}'], self.params[f'beta{i}'],
                        conv_param, self.bn_params[i], pool_param)
                else:
                    # 使用卷积-ReLU-池化层
                    out, cache[f'{i}'] = Conv_ReLU_Pool.forward(
                        out, self.params[f'W{i}'], self.params[f'b{i}'],
                        conv_param, pool_param)
            else:
                if self.batchnorm:
                    # 使用卷积-批归一化-ReLU层
                    out, cache[f'{i}'] = Conv_BatchNorm_ReLU.forward(
                        out, self.params[f'W{i}'], self.params[f'b{i}'],
                        self.params[f'gamma{i}'], self.params[f'beta{i}'],
                        conv_param, self.bn_params[i])
                else:
                    # 使用卷积-ReLU层
                    out, cache[f'{i}'] = Conv_ReLU.forward(
                        out, self.params[f'W{i}'], self.params[f'b{i}'],
                        conv_param)

        # 最后一层（全连接层）
        i += 1
        out, cache[f'{i}'] = Linear.forward(out, self.params[f'W{i}'], self.params[f'b{i}'])
        scores = out

        if y is None:
            return scores

        # 计算损失和梯度
        loss, grads = 0, {}
        loss, dout = softmax_loss(scores, y)

        # 添加L2正则化损失
        for i in range(self.num_layers):
            if self.batchnorm and i <= (self.num_layers-2):
                loss += (self.params[f'gamma{i}']**2).sum() * self.reg
            loss += (self.params[f'W{i}']**2).sum() * self.reg

        # 反向传播
        # 最后一层
        last_dout, dw, db = Linear.backward(dout, cache[f'{i}'])
        grads[f'W{i}'] = dw + 2*self.params[f'W{i}']*self.reg
        grads[f'b{i}'] = db

        # 中间层
        for i in range(i-1, -1, -1):
            if i in self.max_pools:
                if self.batchnorm:
                    last_dout, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(
                        last_dout, cache[f'{i}'])
                    grads[f'gamma{i}'] = dgamma + 2*self.params[f'gamma{i}']*self.reg
                    grads[f'beta{i}'] = dbeta
                else:
                    last_dout, dw, db = Conv_ReLU_Pool.backward(last_dout, cache[f'{i}'])
            else:
                if self.batchnorm:
                    last_dout, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(
                        last_dout, cache[f'{i}'])
                    grads[f'gamma{i}'] = dgamma + 2*self.params[f'gamma{i}']*self.reg
                    grads[f'beta{i}'] = dbeta
                else:
                    last_dout, dw, db = Conv_ReLU.backward(last_dout, cache[f'{i}'])
                    
            grads[f'W{i}'] = dw + 2*self.params[f'W{i}']*self.reg
            grads[f'b{i}'] = db

        return loss, grads

def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                       dtype=torch.float32):
    """
    实现Kaiming初始化用于线性层和卷积层。
    
    参数:
    - Din, Dout: 输入和输出维度
    - K: 如果是None，则初始化线性层权重；否则初始化KxK大小的卷积核
    - relu: 如果为True，使用ReLU的增益系数2；否则使用1
    
    返回:
    - weight: 初始化的权重张量
    """
    gain = 2. if relu else 1.
    
    if K is None:
        # 线性层的Kaiming初始化
        weight_scale = gain/Din
        weight = weight_scale * torch.randn(Din, Dout, dtype=dtype, device=device)
    else:
        # 卷积层的Kaiming初始化
        weight_scale = gain/(Din*K*K)
        weight = weight_scale * torch.randn(Din, Dout, K, K, dtype=dtype, device=device)
    
    return weight

def find_overfit_parameters():
    """
    寻找能够在30个epoch内达到100%训练准确率的参数
    """
    weight_scale = 0.1
    learning_rate = 0.001
    return weight_scale, learning_rate

def create_convolutional_solver_instance(data_dict, dtype, device):
    """
    创建一个在CIFAR-10上训练的最佳CNN求解器
    """
    _, learning_rate = find_overfit_parameters()
    input_dims = data_dict['X_train'].shape[1:]
    
    # 创建模型
    model = DeepConvNet(input_dims=input_dims, 
                        num_classes=10,
                        num_filters=[32,16,64],
                        max_pools=[0,1,2],
                        weight_scale='kaiming',
                        reg=1e-5,
                        dtype=dtype,
                        device=device)
    
    # 创建求解器
    solver = Solver(model, data_dict,
                   num_epochs=200, 
                   batch_size=128,
                   update_rule=adam,
                   optim_config={'learning_rate': 3e-3},
                   print_every=20, 
                   device=device)
    
    return solver
```