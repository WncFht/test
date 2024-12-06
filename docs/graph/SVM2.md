---
categories: graph
date: 2024-10-29T23:26:48+0800
dir: graph
modify: 2024-12-06T00:13:56+0800
share: true
tags:
  - graph
title: SVM2
---

我来帮你实现向量化的 SVM 损失函数和梯度计算。我会详细解释每一步。

```python
def svm_loss_vectorized(W, X, y, reg):
    """
    结构化 SVM 损失函数的向量化实现。
    
    输入:
    - W: 形状为 (D, C) 的权重矩阵
    - X: 形状为 (N, D) 的输入数据
    - y: 形状为 (N,) 的标签数组
    - reg: 正则化强度
    
    返回:
    - loss: 标量，表示损失值
    - dW: 形状与 W 相同的梯度矩阵
    """
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    #############################################################################
    # 第一部分: 计算损失
    #############################################################################
    
    # 1. 计算所有样本的得分矩阵
    scores = X.dot(W)  # (N, C)
    
    # 2. 获取正确类别的得分
    correct_class_scores = scores[np.arange(num_train), y]  # (N,)
    
    # 3. 计算 margins
    # reshape correct_class_scores 为 (N,1) 以便广播
    margins = scores - correct_class_scores.reshape(-1, 1) + 1  # (N, C)
    
    # 4. 正确类别的 margin 应该是 0
    margins[np.arange(num_train), y] = 0
    
    # 5. 只保留大于 0 的 margin
    margins = np.maximum(0, margins)
    
    # 6. 计算平均损失
    loss = np.sum(margins) / num_train
    
    # 7. 添加正则化项
    loss += reg * np.sum(W * W)
    
    #############################################################################
    # 第二部分: 计算梯度
    #############################################################################
    
    # 1. 将 margins 转换为二进制掩码
    binary_mask = np.zeros_like(margins)
    binary_mask[margins > 0] = 1
    
    # 2. 计算每个样本有多少个类别的 margin > 0
    row_sum = np.sum(binary_mask, axis=1)
    
    # 3. 处理正确类别的梯度
    binary_mask[np.arange(num_train), y] = -row_sum
    
    # 4. 计算最终的梯度
    dW = X.T.dot(binary_mask)  # (D, C)
    
    # 5. 归一化并添加正则化项的梯度
    dW = dW / num_train + 2 * reg * W
    
    return loss, dW
```

让我详细解释每一步的原理：

### 损失计算部分

1. **计算得分矩阵**:

   ```python
   scores = X.dot(W)  # (N, C)
   ```

   - 将输入数据 X (N, D) 与权重矩阵 W (D, C) 相乘
   - 得到每个样本对每个类别的得分，形状为 (N, C)

2. **获取正确类别的得分**:

   ```python
   correct_class_scores = scores[np.arange(num_train), y]  # (N,)
   ```

   - 使用高级索引获取每个样本的正确类别得分

3. **计算 margins**:

   ```python
   margins = scores - correct_class_scores.reshape(-1, 1) + 1
   ```

   - 实现 $s_j - s_{y_i} + \Delta$ 的向量化计算
   - reshape(-1, 1) 使得可以进行广播运算

4. **处理正确类别**:

   ```python
   margins[np.arange(num_train), y] = 0
   ```

   - 正确类别不应该计入损失

5. **应用 hinge loss**:

   ```python
   margins = np.maximum(0, margins)
   ```

   - 实现 max(0, -) 操作

### 梯度计算部分

1. **创建二进制掩码**:

   ```python
   binary_mask = np.zeros_like(margins)
   binary_mask[margins > 0] = 1
   ```

   - 标记哪些地方需要计算梯度

2. **计算每个样本的贡献数**:

   ```python
   row_sum = np.sum(binary_mask, axis=1)
   ```

   - 计算每个样本有多少个违反 margin 的类别

3. **处理正确类别的梯度**:

   ```python
   binary_mask[np.arange(num_train), y] = -row_sum
   ```

   - 正确类别的梯度是所有违反 margin 的类别的和的负值

4. **计算最终梯度**:

   ```python
   dW = X.T.dot(binary_mask) / num_train + 2 * reg * W
   ```

   - 使用矩阵乘法一次性计算所有梯度
   - 添加正则化项的梯度

### 优化点

1. **内存效率**：避免创建不必要的临时数组
2. **计算效率**：使用矩阵运算替代循环
3. **数值稳定性**：使用 reshape 和广播而不是 repeat

这个实现比 naive 版本快得多，因为它充分利用了 NumPy 的向量化操作。