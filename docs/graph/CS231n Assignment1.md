---
categories: graph
date: 2024-11-12T15:47:10+0800
dir: graph
modify: 2024-12-06T00:13:10+0800
share: true
tags:
  - graph
title: CS231n Assignment1
---

```python
from ..layers import *
from ..layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecture should be affine - relu - affine - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        # Forward pass
        a1, cache1 = affine_forward(X, W1, b1)
        h1, cache2 = relu_forward(a1)
        scores, cache3 = affine_forward(h1, W2, b2)

        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        
        # Backward pass
        grads = {}
        dh1, grads['W2'], grads['b2'] = affine_backward(dscores, cache3)
        da1 = relu_backward(dh1, cache2)
        dx, grads['W1'], grads['b1'] = affine_backward(da1, cache1)
        
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1

        return loss, grads
```

```python
from builtins import range
import numpy as np
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    """
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    """
    num_train = X.shape[0]
    scores = X.dot(W)
    correct_class_scores = scores[np.arange(num_train), y]
    margins = np.maximum(0, scores - correct_class_scores.reshape(-1, 1) + 1)
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    binary_mask = np.zeros_like(margins)
    binary_mask[margins > 0] = 1
    row_sum = np.sum(binary_mask, axis=1)
    binary_mask[np.arange(num_train), y] = -row_sum
    dW = X.T.dot(binary_mask) / num_train
    dW += 2 * reg * W

    return loss, dW
```

```python
from builtins import range
import numpy as np
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        loss += -np.log(probs[y[i]])
        for j in range(num_classes):
            dW[:, j] += (probs[j] - (j == y[i])) * X[i]

    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    correct_logprobs = -np.log(probs[np.arange(num_train), y])
    loss = np.sum(correct_logprobs) / num_train
    loss += reg * np.sum(W * W)

    dscores = probs
    dscores[np.arange(num_train), y] -= 1
    dW = X.T.dot(dscores) / num_train
    dW += 2 * reg * W

    return loss, dW
```

```python
from __future__ import print_function
from builtins import range, object
import numpy as np
from ..classifiers.linear_svm import svm_loss_vectorized
from ..classifiers.softmax import softmax_loss_vectorized
from past.builtins import xrange

class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in range(num_iters):
            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch, y_batch = X[indices], y[indices]

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print(f"iteration {it} / {num_iters}: loss {loss}")

        return loss_history

    def predict(self, X):
        scores = X.dot(self.W)
        return np.argmax(scores, axis=1)

    def loss(self, X_batch, y_batch, reg):
        pass

class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
```

## Higher Level Representations: Image Features