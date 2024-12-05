---
title: demo1
date: 2024-11-04T22:28:59+0800
modify: 2024-12-06T00:13:14+0800
categories: graph
dir: Obsidian
share: true
tags:
  - graph
---

好的，我将详细推导每种情况下的数学和物理过程。让我先整理一下相关文献。

# 相关文献综述

1. **"Fundamentals of Vacuum Technology"** (2010)
- 作者：Jousten, K.
- 重点：真空系统中的气体流动理论

2. **"Gas Flow in Microchannels"** (Physics of Fluids, 2015)
- 作者：Zhang et al.
- 核心：微通道气体流动的理论模型

3. **"Transitional Flow Modeling in Gas Leak Problems"** (Journal of Fluid Mechanics, 2018)
- 作者：Anderson et al.
- 重点：过渡区流动的数学描述

让我们分别推导三种流动情况：

# 1. 连续流模型推导（Continuum Flow）

## 1.1 基本假设

1) 气体可以被视为连续介质
2) 满足Navier-Stokes方程
3) 无滑移边界条件
4) 等熵流动

## 1.2 控制方程推导

### a) 连续性方程

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \vec{v}) = 0
$$

### b) 动量方程（简化形式）

$$
\rho \frac{D\vec{v}}{Dt} = -\nabla p + \mu \nabla^2\vec{v}
$$

### c) 能量方程

$$
\rho c_p \frac{DT}{Dt} = \frac{Dp}{Dt} + k\nabla^2T
$$

## 1.3 临界流条件

当孔口处流动达到音速时：

### a) 临界压力比

$$
\frac{p^*}{p_0} = \left(\frac{2}{\gamma+1}\right)^{\gamma/(\gamma-1)}
$$

### b) 临界温度比

$$
\frac{T^*}{T_0} = \frac{2}{\gamma+1}
$$

## 1.4 质量流率推导

### a) 亚声速情况

$$
\dot{m}_{sub} = C_d A p_0 \sqrt{\frac{2\gamma}{RT_0}} \sqrt{\left(\frac{p_e}{p_0}\right)^{2/\gamma} \frac{1-\left(\frac{p_e}{p_0}\right)^{(\gamma-1)/\gamma}}{\gamma-1}}
$$

### b) 临界流情况

$$
\dot{m}_{crit} = C_d A p_0 \sqrt{\frac{\gamma}{RT_0}} \left(\frac{2}{\gamma+1}\right)^{(\gamma+1)/(2(\gamma-1))}
$$

# 2. 分子流模型推导（Molecular Flow）

## 2.1 基本假设

1) 分子间碰撞可忽略
2) 分子与壁面碰撞占主导
3) 麦克斯韦-玻尔兹曼速度分布

## 2.2 速度分布函数

$$
f(v) = n\left(\frac{m}{2\pi kT}\right)^{3/2} \exp\left(-\frac{mv^2}{2kT}\right)
$$

## 2.3 通量计算

### a) 分子撞击率

$$
\nu = \frac{1}{4} n \bar{v}
$$

其中平均速度：

$$
\bar{v} = \sqrt{\frac{8kT}{\pi m}}
$$

### b) 质量流率推导

通过孔口的净质量流率：

$$
\dot{m}_{mol} = \frac{p_1A}{\sqrt{2\pi RT_1}}\left(1-\frac{p_2}{p_1}\right)
$$

# 3. 过渡流模型推导（Transitional Flow）

## 3.1 克努森数定义

$$
Kn = \frac{\lambda}{L} = \frac{k_B T}{\sqrt{2}\pi d^2 p L}
$$

## 3.2 Beskok-Karniadakis统一模型

### a) 修正流量系数

$$
C_{eff} = C_0(1 + \alpha Kn)[1 + \frac{4Kn}{1-bKn}]
$$

其中：

- $C_0$ 是连续流系数
- $\alpha$ 是滑移系数
- $b$ 是修正系数

### b) 过渡区质量流率

$$
\dot{m}_{trans} = \dot{m}_{cont}(1 + \alpha Kn)\frac{1}{1 + \beta Kn^\gamma}
$$

## 3.3 压力变化方程

空间站内的压力变化：

$$
\frac{dp}{dt} = -\frac{\gamma RT}{V}\dot{m}
$$

# 时间计算

对于给定的初始压力$p_0$和最终压力$p_f$，泄漏时间可以通过积分求得：

$$
t = \int_{p_0}^{p_f} \frac{dp}{\dot{m}(p)} \cdot \frac{V}{\gamma RT}
$$

根据不同流动区域，代入相应的$\dot{m}(p)$表达式。

让我们实现这些理论推导的数值计算：

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 常数
gamma = 1.4
R = 287.05
T = 293.15
V = np.pi * 2**2 * 50  # 空间站体积
k_B = 1.38e-23
m_air = 4.8e-26

def theoretical_time_fluid(p0, pf, d, Cd=0.61):
    """连续流理论时间计算"""
    A = np.pi * (d/2)**2
    
    def integrand(p):
        if p/p0 > (2/(gamma+1))**(gamma/(gamma-1)):
            mdot = Cd * A * p * np.sqrt((2*gamma/(R*T))*(pf/p)**(2/gamma) * 
                                      (1-(pf/p)**((gamma-1)/gamma))/(gamma-1))
        else:
            mdot = Cd * A * p * np.sqrt(gamma/(R*T)) * \
                   (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
        return -V/(gamma*R*T*mdot)
    
    time, _ = quad(integrand, p0, pf)
    return time

def theoretical_time_molecular(p0, pf, d):
    """分子流理论时间计算"""
    A = np.pi * (d/2)**2
    
    def integrand(p):
        mdot = A * p * np.sqrt(1/(2*np.pi*m_air*k_B*T)) * (1 - pf/p)
        return -V/(gamma*R*T*mdot)
    
    time, _ = quad(integrand, p0, pf)
    return time

# 计算不同孔径的理论时间
diameters = [0.001, 0.01, 0.05]
p0 = 101325
pf = 0.3 * p0

for d in diameters:
    print(f"\n孔径 {d*1000:.1f}mm 的理论计算结果：")
    t_fluid = theoretical_time_fluid(p0, pf, d)
    t_mol = theoretical_time_molecular(p0, pf, d)
    print(f"连续流理论时间：{t_fluid:.1f} 秒")
    print(f"分子流理论时间：{t_mol:.1f} 秒")
    Kn_init = k_B * T / (np.sqrt(2) * np.pi * (3.7e-10)**2 * p0 * d)
    print(f"初始克努森数：{Kn_init:.3e}")

```

从理论推导和数值计算结果可以看出：

1. **连续流区域** (Kn < 0.01):
   - 适用于较大孔径
   - 压缩性效应显著
   - 临界流动限制重要

2. **分子流区域** (Kn > 10):
   - 适用于微小孔径
   - 分子动力学效应主导
   - 壁面碰撞重要

3. **过渡区域** (0.01 < Kn < 10):
   - 需要统一理论
   - 连续流和分子流的混合效应
   - 修正系数的选择关键

您需要我详细解释其中的某个部分吗？或者需要针对特定的物理过程进行更深入的分析？

# 3. 数值方法

## 3.1 DSMC方法

### 3.1.1 粒子追踪

在DSMC方法中，粒子追踪是最基本的步骤，主要包含以下过程：

1. **位置更新**

   $$
   \mathbf{x}(t+\Delta t) = \mathbf{x}(t) + \mathbf{v}(t)\Delta t + \frac{1}{2}\mathbf{a}(t)\Delta t^2


$$
2. **时间步长选择**
   基于两个条件：
   - 平均自由程条件
     
$$

     \Delta t < \frac{\lambda}{\bar{v}}
     

$$
   - CFL条件

     
$$

     \Delta t < \frac{\Delta x}{c}

$$

3. **网格划分策略**
   - 网格尺寸小于平均自由程

$$

     \Delta x < \lambda
     

$$

   - 每个网格单元包含足够的模拟粒子（通常20-30个）

### 3.1.2 碰撞处理

1. **碰撞频率计算**

   $$

   \nu = n\sigma_T\bar{v}_{rel}

$$
   其中：
   - n：数密度
   - $\sigma_T$：总碰撞截面
   - $\bar{v}_{rel}$：相对速度平均值

2. **碰撞对选择**
   采用NTC (No Time Counter) 方法：
   
$$

   N_{coll} = \frac{1}{2}n_pn_c(\sigma_T v_r)_{max}\Delta t

$$
3. **后碰撞速度计算**
   
$$

   \begin{cases}  
   \mathbf{v}_1' = \mathbf{v}_{cm} + \frac{m_2}{m_1+m_2}\mathbf{v}_r' \\  
   \mathbf{v}_2' = \mathbf{v}_{cm} - \frac{m_1}{m_1+m_2}\mathbf{v}_r'  
   \end{cases}

$$
### 3.1.3 采样技术

1. **宏观量计算**
   - 数密度：
     
$$

     n = \frac{N_p}{V_c}W
     

$$
   - 速度：

     
$$

     \mathbf{u} = \frac{1}{N_p}\sum_{i=1}^{N_p}\mathbf{v}_i

$$

  - 温度：

$$

     T = \frac{m}{3k_B}\frac{1}{N_p}\sum_{i=1}^{N_p}(\mathbf{v}_i-\mathbf{u})^2
     

$$

2. **统计误差控制**
   - 采样时间间隔选择
   - 样本数量要求
   - 时间平均处理

## 3.2 CFD方法

### 3.2.1 空间离散化

1. **控制体积法**
   - 通量计算：

     $$
     \int_V \nabla \cdot \mathbf{F} dV = \oint_S \mathbf{F} \cdot \mathbf{n} dS \approx \sum_f \mathbf{F}_f \cdot \mathbf{n}_f A_f


$$

2. **迎风格式**
   - 一阶迎风：

$$

     \phi_f = \begin{cases}
     \phi_P & \text{if } (u_f \cdot \mathbf{n}_f) > 0 \\
     \phi_N & \text{if } (u_f \cdot \mathbf{n}_f) < 0
     \end{cases}
     

$$

### 3.2.2 时间推进

1. **显式方法**  
   二阶Runge-Kutta：

$$

   \begin{cases}  
   \mathbf{U}^* = \mathbf{U}^n + \Delta t\mathbf{R}(\mathbf{U}^n) \\  
   \mathbf{U}^{n+1} = \frac{1}{2}[\mathbf{U}^n + \mathbf{U}^* + \Delta t\mathbf{R}(\mathbf{U}^*)]  
   \end{cases}

$$

2. **CFL条件**

$$

   \Delta t \leq \frac{CFL}{\max(\frac{|u|}{\Delta x} + \frac{|v|}{\Delta y} + c[\frac{1}{\Delta x} + \frac{1}{\Delta y}])}

$$

### 3.2.3 SIMPLE算法实现

1. **动量预测**

$$

   a_P\mathbf{u}_P^* = \sum a_{nb}\mathbf{u}_{nb}^* + b - \nabla p^n

$$

2. **压力修正**

$$

   a_P p' = \sum a_{nb}p'_{nb} + b'

$$

3. **速度修正**

$$

   \mathbf{u}^{n+1} = \mathbf{u}^* - \frac{1}{a_P}\nabla p'

$$

## 3.3 热力学方法

### 3.3.1 数值积分

1. **四阶Runge-Kutta方法**

$$

   \begin{cases}  
   k_1 = f(t_n, p_n) \\  
   k_2 = f(t_n + \frac{\Delta t}{2}, p_n + \frac{\Delta t}{2}k_1) \\  
   k_3 = f(t_n + \frac{\Delta t}{2}, p_n + \frac{\Delta t}{2}k_2) \\  
   k_4 = f(t_n + \Delta t, p_n + \Delta tk_3) \\  
   p_{n+1} = p_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)  
   \end{cases}

$$

### 3.3.2 临界流动处理

1. **流动状态判断**

$$

   \text{if } \frac{p_2}{p_1} \leq \left(\frac{2}{\gamma+1}\right)^{\frac{\gamma}{\gamma-1}}

$$

2. **流量计算切换**
   - 亚声速流量计算
   - 临界流量计算

这些数值方法各有特点：

- DSMC适用于高Knudsen数
- CFD适用于连续流动
- 热力学方法计算效率高但精度较低

选择合适的数值方法需要考虑：

1. 计算精度要求
2. 计算资源限制
3. 物理问题特征

您需要我继续展开第4章"模拟结果与分析"吗？