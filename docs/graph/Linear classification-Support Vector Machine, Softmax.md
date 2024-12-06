---
categories: graph
date: 2024-10-31T21:02:02+0800
dir: graph
modify: 2024-12-06T00:13:27+0800
share: true
tags:
  - graph
title: Linear classification-Support Vector Machine, Softmax
---

# Linear classification-Support Vector Machine, Softmax

## Linear Classifiaction

$$
L_i = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta)
$$

$$
L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + \Delta) \right] + \lambda \sum_k\sum_l W_{k,l}^2
$$

$$
L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) \hspace{0.5in} \text{or equivalently} \hspace{0.5in} L_i = -f_{y_i} + \log\sum_j e^{f_j}
$$

$$
\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}
= \frac{Ce^{f_{y_i}}}{C\sum_j e^{f_j}}
= \frac{e^{f_{y_i} + \log C}}{\sum_j e^{f_j + \log C}}
$$

![Pasted image 20241031210509.png](../assets/images/Pasted%20image%2020241031210509.png)