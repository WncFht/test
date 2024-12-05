---
title: two_layer_net
date: 2024-10-22T22:15:46+0800
modify: 2024-12-06T00:10:24+0800
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
import torch
import random
import statistics
from linear_classifier import sample_batch

def hello_two_layer_net():
    """
    测试函数，用于确认环境设置正确
    """
    print('Hello from two_layer_net.py!')

class TwoLayerNet(object):
    """两层神经网络分类器"""
    
    def __init__(self, input_size, hidden_size, output_size,
                 dtype=torch.float32, device='cuda', std=1e-4):
        """
        初始化神经网络模型
        
        参数:
        - input_size: 输入数据维度 D
        - hidden_size: 隐藏层神经元数量 H
        - output_size: 输出类别数量 C
        - dtype: 权重参数的数据类型
        - device: 计算设备(GPU/CPU)
        - std: 权重初始化的标准差
        """
        # 固定随机种子以确保可重复性
        random.seed(0)
        torch.manual_seed(0)

        # 初始化模型参数
        self.params = {}
        self.params['W1'] = std * torch.randn(input_size, hidden_size, dtype=dtype, device=device)  # 第一层权重
        self.params['b1'] = torch.zeros(hidden_size, dtype=dtype, device=device)                    # 第一层偏置
        self.params['W2'] = std * torch.randn(hidden_size, output_size, dtype=dtype, device=device) # 第二层权重
        self.params['b2'] = torch.zeros(output_size, dtype=dtype, device=device)                    # 第二层偏置

def nn_forward_pass(params, X):
    """
    神经网络的前向传播阶段
    
    架构: 全连接层 -> ReLU激活 -> 全连接层
    
    参数:
    - params: 包含权重和偏置的字典
    - X: 形状为(N, D)的输入数据
    
    返回:
    - scores: 形状为(N, C)的分类得分
    - hidden: 形状为(N, H)的隐藏层输出
    """
    # 解包参数
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    # 计算第一层
    hidden = X.mm(W1) + b1     # 全连接层
    hidden[hidden < 0] = 0     # ReLU激活函数
    
    # 计算第二层
    scores = hidden.mm(W2) + b2  # 全连接层输出scores
    
    return scores, hidden

def nn_forward_backward(params, X, y=None, reg=0.0):
    """
    计算损失和梯度的前向/后向传播
    
    参数:
    - params: 网络参数字典
    - X: 输入数据
    - y: 标签
    - reg: 正则化强度
    
    返回:
    - 如果y为None，返回scores
    - 否则返回(loss, grads)元组
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    # 前向传播
    scores, h1 = nn_forward_pass(params, X)
    if y is None:
        return scores

    # 计算损失
    # 数值稳定性处理：减去最大值
    shifted_logits = scores - scores.max(dim=1, keepdim=True).values
    Z = shifted_logits.exp().sum(dim=1, keepdim=True)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()
    
    # 交叉熵损失 + L2正则化
    loss = (-1.0 / N) * log_probs[torch.arange(N), y].sum()
    loss += reg * torch.sum(W1 * W1) + reg * torch.sum(W2 * W2)

    # 反向传播计算梯度
    grads = {}
    # softmax梯度
    d_scores = probs.clone()
    d_scores[torch.arange(N), y] -= 1
    d_scores /= N

    # 第二层梯度
    grads['b2'] = d_scores.sum(dim=0)
    grads['W2'] = h1.t().mm(d_scores) + 2 * W2 * reg

    # 第一层梯度
    d_h1 = d_scores.mm(W2.T)
    d_h1[h1 == 0] = 0  # ReLU梯度
    grads['b1'] = d_h1.sum(dim=0)
    grads['W1'] = X.t().mm(d_h1) + 2 * W1 * reg

    return loss, grads

def nn_train(params, loss_func, pred_func, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    使用随机梯度下降训练神经网络
    
    参数:
    - params: 网络参数
    - loss_func: 损失函数
    - pred_func: 预测函数
    - X, y: 训练数据和标签
    - X_val, y_val: 验证数据和标签
    - learning_rate: 学习率
    - learning_rate_decay: 学习率衰减因子
    - reg: 正则化强度
    - num_iters: 迭代次数
    - batch_size: 批量大小
    - verbose: 是否打印训练过程
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # 记录训练过程
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        # 采样小批量数据
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)
        
        # 计算损失和梯度
        loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
        loss_history.append(loss.item())

        # 更新参数
        params['W1'] -= grads['W1'] * learning_rate
        params['W2'] -= grads['W2'] * learning_rate
        params['b1'] -= grads['b1'] * learning_rate
        params['b2'] -= grads['b2'] * learning_rate

        # 打印训练过程
        if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss.item()))

        # 每个epoch结束后检查准确率并衰减学习率
        if it % iterations_per_epoch == 0:
            # 计算训练和验证准确率
            y_train_pred = pred_func(params, loss_func, X_batch)
            train_acc = (y_train_pred == y_batch).float().mean().item()
            y_val_pred = pred_func(params, loss_func, X_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # 学习率衰减
            learning_rate *= learning_rate_decay

    return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
    }

def nn_predict(params, loss_func, X):
    """
    使用训练好的权重进行预测
    
    参数:
    - params: 网络参数
    - loss_func: 损失函数
    - X: 输入数据
    
    返回:
    - y_pred: 预测的类别标签
    """
    # 前向传播得到分数
    results, _ = nn_forward_pass(params, X)
    # 选择分数最高的类别
    _, y_pred = results.max(dim=1)
    return y_pred

def nn_get_search_params():
    """
    返回神经网络的候选超参数
    
    返回:
    - learning_rates: 学习率候选值
    - hidden_sizes: 隐藏层大小候选值
    - regularization_strengths: 正则化强度候选值
    - learning_rate_decays: 学习率衰减因子候选值
    """
    # 设置不同的超参数搜索范围
    hidden_sizes = [2, 8, 32, 128]  # 隐藏层大小从小到大
    regularization_strengths = [0, 1e-5, 1e-3, 1e-1]  # 正则化强度从无到强
    learning_rates = [1e-4, 1e-2, 1e0, 1e2]  # 学习率从小到大
    learning_rate_decays = []  # 这里未使用学习率衰减搜索
    
    return learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays

def find_best_net(data_dict, get_param_set_fn):
    """
    通过验证集调优超参数找到最佳的神经网络模型
    
    参数:
    - data_dict: 包含训练和验证数据的字典
    - get_param_set_fn: 获取超参数的函数
    
    返回:
    - best_net: 最佳模型
    - best_stat: 最佳模型的训练统计信息
    - best_val_acc: 最佳验证准确率
    """
    best_net = None
    best_stat = None
    best_val_acc = 0

    # 获取所有超参数组合
    learning_rates, hidden_sizes, regularization_strengths, _ = get_param_set_fn()
    
    # 网格搜索最佳超参数
    for reg in regularization_strengths:
        for lr in learning_rates:
            for hs in hidden_sizes:
                print('train with hidden_size: {}'.format(hs))
                print('train with learning_rate: {}'.format(lr))
                print('train with regularization: {}'.format(reg))
                
                # 创建和训练网络
                net = TwoLayerNet(3 * 32 * 32, hs, 10, 
                                device=data_dict['X_train'].device, 
                                dtype=data_dict['X_train'].dtype)
                
                # 训练模型
                stats = net.train(
                    data_dict['X_train'], data_dict['y_train'],
                    data_dict['X_val'], data_dict['y_val'],
                    num_iters=3000, batch_size=1000,
                    learning_rate=lr, learning_rate_decay=0.95,
                    reg=reg, verbose=False
                )
                
                # 更新最佳模型
                if max(stats['val_acc_history']) > best_val_acc:
                    best_val_acc = max(stats['val_acc_history'])
                    best_net = net
                    best_stat = stats

    return best_net, best_stat, best_val_acc
```

```python
# pytorch  语法

# 1.
# 创建一个 2x3 的张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 在第 0 维 (按列) 求和
sum_dim0 = x.sum(dim=0)  # 输出: tensor([5, 7, 9])

# 在第 1 维 (按行) 求和
sum_dim1 = x.sum(dim=1)  # 输出: tensor([ 6, 15])

# 2.
# 创建一个 2x3 的张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 没有 keepdim，默认行为，减少维度
sum_dim1 = x.sum(dim=1, keepdim=False)  # 输出: tensor([ 6, 15]), 结果是 1D 张量

# 使用 keepdim 保留维度
sum_dim1_keepdim = x.sum(dim=1, keepdim=True)  # 输出: tensor([[ 6], [15]]), 结果是 2D 张量

# 3.
# 生成服从标准正态分布的随机张量，用于初始化权重
torch.randn(input_size, hidden_size, dtype=dtype, device=device)

# 4.
# 固定随机种子，确保结果的可复现性
torch.manual_seed(0)

```