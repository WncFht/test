---
title: STLC开始的类型论入门
date: 2024-09-15T14:51:54+0800
modify: 2024-12-06T00:10:26+0800
categories: Computer
dir: Obsidian
share: true
tags:
  - Computer
  - Advanced
  - TCS
---

# STLC开始的类型论入门

## 1 前置知识

### 1.1 类型论，与一个类型论

- STLC
- MLTT
- HoTT  
我们只介绍 STLC

类型论分为 Curry 风格和 Church 风格。

### 1.2 元变量

- 以字符串的形式表示
	- 变量
	- 项
	- 类型
	- 语境
- 自然语言（元语言）
	- A 是取值于 STLC 的类型的元变量
- 某类型 A -> 某个类型，我们用元变量 A 来表示它

### 1.3 符号预定

![QQ_1725517282933.png](https://cdn.jsdelivr.net/gh/WncFht/picture/202409051421634.png)

## 2 $\displaystyle \lambda$ 演算的动机

- 如何构造和使用一个函数
	- $\displaystyle \lambda x.M$ 表示“参数为 $\displaystyle x$，函数体为 $\displaystyle M$ 的函数”
	- 编程语言中的匿名函数来源于 $\displaystyle \lambda$ 演算

## 3 简单类型 $\displaystyle \lambda$ 演算

Simply Typed Lambda Caluculus

### 3.1 相继式演算

- 类型规则 (typing rules)
- 用 sequent calculus 的形式给出  
$\displaystyle \frac{\text{Premise}_1\quad\text{Premise}_2\quad...\quad\text{Premise}_n}{\text{Conclusion}}\text{(Name)}$
- judgement
	- premise
	- conculsion  
如果作为分子的所有条件成立，那么作为分母的结论也成立

### 3.2 类型

$$
\frac{A\in\mathcal{B}}{A\text{ type}}\text{Base}
$$

$$
\frac{A \mathrm{type}\quad B \mathrm{type}}{A\to B \mathrm{type}} \mathrm{Func}
$$

$\displaystyle \to$ 是右结合  
上面两个式子就是

- 如果 $\displaystyle A \in B$ ，那么 A 是 STLC 的类型
- 如果 $\displaystyle A,B$ 是 STLC 的类型，那么 $\displaystyle A  \to B$ 是 STLC 的类型

### 3.3 语境

context

$$
\displaystyle \frac{}{\emptyset \mathrm{~ctx}}
$$

$$
\frac{\Gamma\mathrm{~etx}}{\Gamma,x:A\mathrm{~ctx}}
$$

### 3.4 项

term  
$\displaystyle \Gamma\vdash M:A$ 表示在语境 $\displaystyle \Gamma$ 下 $\displaystyle M$ 是一个合法的项，并且具有类型 $\displaystyle A$

$$
\frac{c\in\mathcal{C}}{\Gamma\vdash c:\text{toType}(c)}\text{Const}
$$

$$
\frac{x:A\in\Gamma}{\Gamma\vdash x:A} \mathrm{Var}
$$

$$
\frac{\Gamma,x:A\vdash M:B}{\Gamma\vdash\lambda(x:A).M:A\to B}\text{Lam}
$$

$$
\frac{\Gamma\vdash M:A\to B\quad\Gamma\vdash N:A}{\Gamma\vdash MN:B}\text{App}
$$

函数调用是左结合  
规则 Lam 可以省略:

$$
\frac{x:A\vdash M:B}{\lambda(x:A).M:A\to B}
$$

![QQ_1725519952556.png](https://cdn.jsdelivr.net/gh/WncFht/picture/202409051505821.png)

$$
\begin{aligned}\mathcal{B}&=\{\top,\bot,\mathrm{Ans}\}\\\mathcal{C}&=\{\mathrm{tt},\mathrm{yes},\mathrm{no}\}\\\mathrm{toType}(c)&=\begin{cases}\top&(c-\mathrm{tt})\\\mathrm{Ans}&(c\in\{\mathrm{yes},\mathrm{no}\})\end{cases}\end{aligned}
$$

### 3.5 替换

#### 3.5.1 出现

occurrence  
if x in M, then the occurrence is called bound occurrence  
and $\displaystyle \lambda x$ is the binding site.  
otherwise it is the free occurenece  
![QQ_1725520393397.png](https://cdn.jsdelivr.net/gh/WncFht/picture/202409051513473.png)  
![QQ_1725520422230.png](https://cdn.jsdelivr.net/gh/WncFht/picture/202409051513141.png)

#### 3.5.2 $\displaystyle \alpha$ conversion

## 4 STLC 的模型

## 5 扩展 STLC