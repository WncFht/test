---
categories: graph
date: 2024-10-29T23:18:38+0800
dir: graph
modify: 2024-12-06T00:13:55+0800
share: true
tags:
  - graph
title: SVM1
---

# SVM 详细推导与实现分析

## 1. SVM 基本理论

### 1.1 二分类 SVM 原理

对于给定的训练数据点 $(x_i, y_i)$，其中 $y_i \in \{-1, 1\}$，SVM 的目标是找到一个超平面：

$$
f(x) = w^T x + b = 0
$$

最大化决策边界的几何间隔，可以归结为以下优化问题：

$$
\begin{aligned}
& \min_{w,b} \frac{1}{2} \|w\|^2 \\
& \text{s.t. } y_i(w^T x_i + b) \geq 1, \quad i=1,\ldots,n
\end{aligned}
$$

### 1.2 多分类 SVM 损失函数

对于多分类问题，我们使用 Hinge Loss：

$$
L_i = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta)
$$

完整的损失函数（包含正则化项）：

$$
L = \frac{1}{N}\sum_{i=1}^N L_i + \lambda \|W\|^2
$$

## 2. 梯度计算

### 2.1 Hinge Loss 梯度

对于单个样本的 Hinge Loss，当 margin > 0 时：

- 对于错误类别 j：

  $$

\frac{\partial L}{\partial w_j} = x_i

$$
- 对于正确类别 yi：
  
$$

\frac{\partial L}{\partial w_{y_i}} = -x_i

$$
### 2.2 正则化项梯度

L2 正则化项的梯度
$$

 \frac{\partial}{\partial W}(\lambda \|W\|^2) = 2\lambda W $$

## 3. 代码实现与分析

### 3.1 核心实现

```python
def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)          # 计算所有类别的得分
        correct_class_score = scores[y[i]]
        
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]      # 错误类别的梯度
                dW[:, y[i]] -= X[i]   # 正确类别的梯度

    # 归一化和正则化
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    
    return loss, dW
```

### 3.2 关键步骤解释

1. **得分计算**：

   ```python
   scores = X[i].dot(W)
   ```

   - X[i] 形状：(D,)
   - W 形状：(D, C)
   - scores 形状：(C,)，表示每个类别的得分

2. **margin 计算**：

   ```python
   margin = scores[j] - correct_class_score + 1
   ```

   - 这实现了 hinge loss 中的 $\max(0, s_j - s_{y_i} + \Delta)$
   - $\Delta = 1$ 是超参数

3. **梯度更新**：

   ```python
   if margin > 0:
       dW[:, j] += X[i]      # 对错误类别
       dW[:, y[i]] -= X[i]   # 对正确类别
   ```

   - 这反映了 hinge loss 的分段性质
   - 只有在 margin > 0 时才需要更新梯度

## 4. 优化策略

### 4.1 向量化实现

可以通过向量化操作提高计算效率：

```python
def svm_loss_vectorized(W, X, y, reg):
    num_train = X.shape[0]
    scores = X.dot(W)
    correct_class_scores = scores[np.arange(num_train), y]
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)
    margins[np.arange(num_train), y] = 0
    
    loss = np.sum(margins) / num_train + reg * np.sum(W * W)
    
    # 梯度计算
    binary = margins
    binary[margins > 0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -row_sum
    dW = X.T.dot(binary) / num_train + 2 * reg * W
    
    return loss, dW
```

### 4.2 性能比较

向量化实现相比循环实现的优势：

1. 计算速度更快
2. 代码更简洁
3. 内存使用更高效

## 5. 实践注意事项

1. **数值稳定性**：
   - 计算 margin 时要注意数值溢出
   - 正则化参数 reg 的选择很重要

2. **超参数调节**：
   - 正则化强度 reg
   - margin 阈值 Delta（代码中设为1）

1. **初始化**：
   - 权重矩阵 W 的初始化方法会影响收敛性