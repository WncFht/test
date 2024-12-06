---
categories: graph
date: 2024-10-20T03:52:28+0800
dir: graph
modify: 2024-12-06T00:14:13+0800
share: true
tags:
  - graph
title: Written Memories_notes
---

切换导航

书面记忆：理解、推导和扩展LSTM  
2016年7月26日，星期二

当我第一次接触到长短期记忆网络（LSTMs）时，我很难忽视它们的复杂性。我不明白它们为什么会被设计成现在这个样子，只知道它们有效。结果发现，LSTMs是可以被理解的，而且尽管它们表面上看起来很复杂，但实际上LSTMs是基于一些非常简单，甚至美丽的关于神经网络的洞察。当我最初学习循环神经网络（RNNs）时，我希望我能拥有这篇文章。

在这篇文章中，我们做几件事情：

我们将定义并描述一般的RNNs，重点关注导致LSTM发展的普通RNNs的局限性。  
我们将描述LSTM架构背后的直觉，这将使我们能够建立并推导出LSTM。在此过程中，我们将推导出GRU。我们还将推导出一个伪LSTM，我们将看到它在原理和性能上都优于标准LSTM。  
然后，我们将扩展这些直觉，以展示它们如何直接导致一些最近和令人兴奋的架构：高速公路和残差网络，以及神经图灵机。  
这是一篇关于理论的文章，而不是实现。关于如何使用Tensorflow实现RNNs，请查看我的文章《Tensorflow中的循环神经网络I》和《Tensorflow中的循环神经网络II》。

内容/快速链接：  
循环神经网络  
RNNs能做什么；选择时间步长  
普通的RNN  
信息形态变化和消失以及爆炸性敏感性  
消失敏感性的数学充分条件  
避免消失梯度的最小权重初始化  
通过时间反向传播和消失敏感性  
处理消失和爆炸性梯度  
书面记忆：LSTM背后的直觉  
使用选择性控制和协调写作  
作为选择性机制的门控  
将门控粘合在一起以推导出原型LSTM  
三个工作模型：标准化原型、GRU和伪LSTM  
推导出LSTM  
带窥孔的LSTM  
基本LSTM和伪LSTM的实证比较  
扩展LSTM  

### 先决条件1  

这篇文章假设读者已经熟悉以下内容：

1. 前馈神经网络  
2. 反向传播  
3. 基础线性代数  
我们将回顾所有其他内容，从一般RNNs开始。

## 循环神经网络  

从一刻到下一刻，我们的大脑作为一个函数运作：它接受来自我们感官（外部）和我们的思想（内部）的输入，并以行动（外部）和新思想（内部）的形式产生输出。我们看到一只熊，然后想到“熊”。我们可以模拟这种行为，使用前馈神经网络：我们可以教会前馈神经网络在显示熊的图像时思考“熊”。

但我们的大脑不是一个一次性函数。它随时间反复运行。我们看到了一只熊，然后想到“熊”，然后想到“跑”。重要的是，将熊的图像转化为“熊”思想的同一个函数也转化了“熊”的思想为“跑”的思想。它是一个循环函数，我们可以用循环神经网络（RNN）来模拟。

RNN是由相同的前馈神经网络组成的，每个时刻或时间步骤都有一个，我们将称之为“RNN单元”。注意，这是比通常给出的RNN定义（“普通”RNN作为LSTM的前身稍后介绍）要宽泛得多的定义。这些单元操作它们自己的输出，允许它们被组合。它们也可以操作外部输入并产生外部输出。这里是单个RNN单元的图表：  
![Pasted image 20241020095758.png](../assets/images/Pasted%20image%2020241020095758.png)  
这里是三个组合RNN单元的图表：  
![Pasted image 20241020095829.png](../assets/images/Pasted%20image%2020241020095829.png)  
你可以将递归输出视为传递给下一个时间步的“状态”。因此，一个RNN单元接受先前的状态和一个（可选的）当前输入，并产生当前状态和一个（可选的）当前输出。

这里是RNN单元的代数描述：

其中：

\[ s_t \] 和 \[ s_{t-1} \] 是我们当前和先前的状态，  
\[ o_t \] 是我们（可能为空的）当前输出，  
\[ x_t \] 是我们（可能为空的）当前输入，和  
\[ f \] 是我们的循环函数。  
我们的大脑在原地运作：当前的神经活动取代了过去的神经活动。我们也可以将RNN看作是在原地运作：因为RNN单元是相同的，它们可以被视为同一个对象，RNN单元的“状态”在每个时间步骤中被覆盖。这里是这种框架的图表：  
![Pasted image 20241020100251.png](../assets/images/Pasted%20image%2020241020100251.png)  
大多数RNN的介绍都从这个“单细胞循环”框架开始，但我认为你会发现顺序框架更直观，特别是当考虑反向传播时。当从单细胞循环框架开始时，RNN被称为“展开”以获得上面的顺序框架。

## RNNs能做什么；选择时间步长  

上面描述的RNN结构非常通用。理论上，它可以做任何事情：如果我们给每个单元内的神经网络至少一个隐藏层，每个单元就成为一个通用函数逼近器。这意味着一个RNN单元可以模拟任何函数，因此，理论上，一个RNN可以完美地模拟我们的大脑。尽管我们知道大脑理论上可以以这种方式建模，但实际设计和训练一个RNN来做到这一点是完全不同的事情。然而，我们正在取得很好的进展。

有了这个大脑的类比，我们所需要做的就是看看我们如何使用RNN来处理一个任务，就是问一个人类会如何处理相同的任务。

以英文到法文翻译为例。一个人阅读一个英文句子（“猫坐在垫子上”），暂停，然后写出法文翻译（“猫坐在垫子上”）。为了用RNN模拟这种行为，我们唯一需要做出的选择（除了设计RNN单元本身，现在我们将其视为一个黑盒子）是决定应该使用的时间步长是什么，这决定了输入和输出的形式，或者RNN如何与外部世界互动。

一个选择是根据内容设置时间步长。也就是说，我们可能使用整个句子作为一个时间步长，在这种情况下，我们的RNN只是一个前馈网络：  
![Pasted image 20241020100516.png](../assets/images/Pasted%20image%2020241020100516.png)  
翻译单个句子时，最终状态无关紧要。然而，如果句子是正在翻译的段落的一部分，那么它可能会很重要，因为它包含了关于之前句子的信息。请注意，上面的初始状态被指示为空白，但在评估单个序列时，将初始状态训练为变量可能是有用的。也许最好的“序列开始”状态表示不一定是空白零状态。

或者，我们可以说每个单词或每个字符是一个时间步长。这里是一个RNN在每个单词的基础上翻译“the cat sat”的图示：  
![Pasted image 20241020100644.png](../assets/images/Pasted%20image%2020241020100644.png)

在第一个时间步长之后，状态包含了“the”的内部表示；在第二个之后，“the cat”；在第三个之后，“the cat sat”。网络在前三个时间步长中没有产生任何输出。当它接收到一个空白输入时，它开始产生输出，此时它知道输入已经结束。当它完成产生输出时，它产生一个空白输出以表示它已经完成。

在实践中，即使像深度LSTMs这样的强大RNN架构也可能在多个任务上表现不佳（这里有两个：阅读，然后翻译）。为了适应这一点，我们可以将网络分成多个RNNs，每个RNN都专门处理一个任务。在这个例子中，我们将使用一个“编码器”网络来读取英文（蓝色）和一个单独的“解码器”网络来读取法文（橙色）：  
![Pasted image 20241020100811.png](../assets/images/Pasted%20image%2020241020100811.png)  
此外，如上图所示，解码器网络正在被输入最后一个真实值（即，在训练期间的目标值，在测试期间网络之前的翻译单词选择）。有关RNN编码器-解码器模型的示例，请参见Cho等人（2014年）。

请注意，拥有两个独立的网络仍然符合单个RNN的定义：我们可以将递归函数定义为一个拆分函数，它接受其其他输入，指定要使用的函数拆分。

时间步长不必基于内容；它可以是一个实际的时间单位。例如，我们可能认为时间步长是一秒钟，并强制执行每秒5个字符的阅读速度。前三个时间步长的输入将是c、at sa和t on。

我们还可以做更有趣的事情：我们可以让RNN决定何时准备好移动到下一个输入，甚至是什么输入。这与人类可能专注于某些单词或短语一段时间以翻译它们或可能回顾源的方式类似。为了做到这一点，我们使用RNN的输出（一个外部动作）来动态确定其下一个输入。例如，我们可能让RNN输出动作，如“再次阅读最后一个输入”，“回溯5个时间步长的输入”等。成功的基于注意力的翻译模型就是在此基础上的：它们在每个时间步长接受整个英文序列，它们的RNN单元决定哪些部分与它们目前正在产生的当前法文单词最相关。

这个英语到法语的翻译示例并没有什么特别之处。无论我们选择哪种人类任务，我们都可以通过选择不同的时间步长来构建不同的RNN模型。我们甚至可以将像手写数字识别这样的任务重新定义为多个时间步长的任务，对于这种任务，一次性函数（单时间步长）是典型的方法。实际上，亲自看看一些MNIST数字，观察你需要比其他数字更长时间地关注它们。前馈神经网络不能表现出这种行为；RNN可以。

## 普通的RNN  

现在我们已经了解了大局，让我们来看一下RNN单元的内部。最基本的RNN单元是一个单层神经网络，其输出用作RNN单元的当前（外部）输出和当前状态：  
![Pasted image 20241020101530.png](../assets/images/Pasted%20image%2020241020101530.png)  
请注意，先前的状态向量与当前状态向量的大小相同。如上所述，这对于RNN单元的组合至关重要。这里是普通RNN单元的代数描述：

$$
s_{t} = \phi (W s_{(t-1)} + U x_{t} + b)
$$

其中：

- \[ \phi \] 是激活函数（例如，sigmoid、tanh、ReLU），  
- \[ s_t \in \mathbb{R}^n \] 是当前状态（和当前输出），  
- \[ s_{t-1} \in \mathbb{R}^n \] 是先前状态，  
- \[ x_t \in \mathbb{R}^m \] 是当前输入，  
- $W \in \mathbb{R}^{n \times n}$ , $U \in \mathbb{R}^{m \times n}$ , 和 $b \in \mathbb{R}^n$ 是权重和偏置，和  
- \[ n \] 和 \[ m \] 是状态和输入大小。  
即使是这种基本的RNN单元也非常强大。尽管它没有满足单一单元内通用函数逼近的标准，但已知一系列组合的普通RNN单元是图灵完备的，因此可以实现任何算法。参见Siegelmann和Sontag（1992）。理论上这很好，但实践中有一个问题：使用反向传播算法训练普通RNNs结果证明是非常困难的，甚至比训练非常深的前馈神经网络更加困难。这种困难是由于信息形态变化和消失和爆炸性敏感性问题，这些问题是由相同的非线性函数重复应用引起的。

信息形态变化和消失和爆炸性敏感性4  
与其考虑大脑，不如将整个世界建模为一个RNN：从每一个时刻到下一个时刻，世界的状态被一个叫做时间的极其复杂的循环函数修改。现在考虑今天发生的一个小变化将如何在一百年后影响世界。可能是像蝴蝶翅膀的扇动最终在世界另一端引起台风。5但也可能我们的今天的行动最终无关紧要。如果爱因斯坦没有发现相对论怎么办？这在1950年代可能会有所不同，但也许那时有人发现了相对论，以至于到了2000年代差异变小，最终到2050年接近零。最后，一个小变化的重要性可能会波动：也许爱因斯坦的发现实际上是由于他的妻子对一只偶然飞过的蝴蝶的评论引起的，以至于蝴蝶在20世纪引发了一个大变化，然后很快就消失了。

在爱因斯坦的例子中，请注意，过去的变化是新信息的引入（相对论），更一般地说，新信息的引入是时间流逝（时间的流动）的直接结果。因此，我们可以将信息本身视为由循环函数变形的变化，其影响消失、爆炸或简单地波动。

这种讨论表明，世界（或RNN）的状态在不断变化，现在对过去的变化可能非常敏感或非常不敏感：效果可以复合或溶解。这些都是问题，它们扩展到RNNs（和前馈神经网络）：

信息形态变化

首先，如果信息不断变化，当我们需要它时，就很难适当地利用过去的信息。信息的最佳可用状态可能在某个时刻已经发生过。除了学习如何在今天利用信息（如果它以原始的可用形式存在），我们还必须学会如何从当前状态解码原始状态，如果那是可能的话。这导致了困难的学习和糟糕的结果。6

在普通RNN中很容易证明信息形态变化的发生。实际上，假设RNN单元能够在没有外部输入的情况下完全保持其先前状态。那么\[ F(x) = \phi(W s_{t-1} + b) \]是关于\[ s_{t-1} \]的恒等函数。但恒等函数是线性的，而\[ F(x) \]是非线性的，所以我们有一个矛盾。因此，RNN单元不可避免地会在一个时间步骤到下一个时间步骤之间变形状态。即使是输出\[ s_t = x_t \]这样的简单任务对于普通RNN来说也是不可能的。

这是一些圈子中所谓的退化问题的根源。参见，例如，He等人（2015）。He等人的作者声称这是“意想不到的”和“违反直觉的”，但我希望这次讨论表明，退化问题，或信息形态变化，实际上是相当自然的（并且在许多情况下是可取的）。我们将在下面看到，尽管信息形态变化不是引入LSTMs的原始动机之一，但LSTM背后的原则碰巧有效地解决了这个问题。事实上，He等人（2015）使用的残差网络的有效性是LSTMs的基本原则的结果。

消失和爆炸性梯度

第二，我们使用反向传播算法训练RNN。但反向传播是基于梯度的算法，消失和爆炸性“敏感性”只是另一种说法，即消失和爆炸性梯度（后者是被接受的术语，但我认为前者更具描述性）。如果梯度爆炸，我们无法训练我们的模型。如果它们消失，我们就很难学习长期依赖关系，因为反向传播对最近的干扰过于敏感。这使得训练变得困难。

我将很快回到通过反向传播训练RNN的困难，但首先我想给出一个简短的数学证明，说明普通RNN容易受到消失梯度的影响，以及我们可以在训练开始时做些什么来帮助避免这一点。

消失敏感性的数学充分条件  
在这一部分，我给出了普通RNN中消失敏感性的充分条件的数学证明。这部分有点数学，你可以安全地跳过证明的细节。它本质上与Pascanu等人（2013）中的类似结果的证明相同，但我认为你会发现这个介绍更容易理解。这里的证明还利用了中值定理，比Pascanu等人更进一步，达到了一个稍微更强的结果，有效地显示了消失的因果关系而不是消失的敏感性。7请注意，关于消失和爆炸梯度的数学分析可以追溯到20世纪90年代初，在Bengio等人（1994）和Hochreiter（1991）（最初是德语，在Hochreiter和Schmidhuber（1997）中总结了相关内容）。

设\[ s_t \]是我们在时间\[ t \]的状态向量，设\[ \Delta v \]是由状态向量在时间\[ t \]的\[ \Delta s_t \]引起的向量\[ v \]的变化。我们的目标是提供一个数学上充分的条件，使得由状态在时间步骤\[ t \]的变化引起的在时间步骤\[ t+k \]的状态变化随着\[ n \to \infty \]消失；即，我们将证明以下条件的充分条件：

\[ \lim_{k \to \infty} \frac{\Delta s_{t+k}}{\Delta s_t} = 0. \]

相比之下，Pascanu等人（2013）证明了以下结果的相同充分条件，这可以很容易地扩展以获得上述结果：

\[ \lim_{k \to \infty} \frac{\partial s_{t+k}}{\partial s_t} = 0. \]

首先，根据我们对普通RNN单元的定义，我们有：

\[ s_{t+1} = \phi(z_t) \text{ where } z_t = W s_t + U x_{t+1} + b. \]

应用多变量中值定理，我们得到存在\[ c \in [z_t, z_t + \Delta z_t] \)使得：

\[ \Delta s_{t+1} = [\phi'(c)] \Delta z_t = [\phi'(c)] \Delta (W s_t). = [\phi'(c)] W \Delta s_t. \]

现在设\[ \|A\| \)表示矩阵2-范数，\[ |v| \)表示欧几里得向量范数，并定义：

\[ \gamma = \sup_{c \in [z_t, z_t + \Delta z_t]} \|[\phi'(c)]\| \]

请注意，对于逻辑sigmoid，\[ \gamma \leq \frac{1}{4} \)，对于tanh，\[ \gamma \leq 1 \)。8

取两边的向量范数，我们得到，其中第一个不等式来自于2-范数的定义（应用了两次），第二个来自于上确界的定义：

\[ |\Delta s_{t+1}| = |[\phi'(c)] W \Delta s_t| \leq \|[\phi'(c)]\| \|W\| |\Delta s_t| \leq \gamma \|W\| |\Delta s_t| = \|\gamma W\| |\Delta s_t|. \]

（1）

通过在\[ k \]个时间步骤中展开这个公式，我们得到\[ |\Delta s_{t+k}| \leq \|\gamma W\|^k |\Delta s_t| \)，使得：

\[ \frac{|\Delta s_{t+k}|}{|\Delta s_t|} \leq \|\gamma W\|^k. \]

因此，如果\[ \|\gamma W\| < 1 \)，我们有\[ \frac{|\Delta s_{t+k}|}{|\Delta s_t| \)随着时间指数级减少，并且我们证明了以下条件的充分条件：

\[ \lim_{k \to \infty} \frac{\Delta s_{t+k}}{\Delta s_t} = 0. \]

当\[ \|\gamma W\| < 1 \)时？\[ \gamma \)对于逻辑sigmoid有界于\[ \frac{1}{4} \)，对于tanh，有界于1。这告诉我们消失梯度的充分条件是\[ \|W\| \)必须小于4或1，分别。

从这一点上，我们立即得到的教训是，如果我们的权重初始化对于\[ W \)太小，我们的RNN可能一开始就无法学习任何东西，由于梯度消失。让我们现在扩展这个分析，以确定一个理想的权重初始化。

避免消失梯度的最小权重初始化  
找到一种不会立即受到这个问题影响的权重初始化是有益的。扩展上述分析，找到使我们尽可能接近等式（1）的权重\[ W \)的初始化，会导致一个不错的结果。

首先，让我们假设\[ \phi = \text{tanh} \)并取\[ \gamma = 1 \)，9但你同样可以假设\[ \phi = \sigma \)并取\[ \gamma = \frac{1}{4} \)以获得不同的结果。

我们的目标是找到一个\[ W \)的初始化，使得：

\[ \|\gamma W\| = 1. \]

我们在等式（1）中尽可能接近等式。  
从第一点，由于我们取\[ \gamma \)为1，我们有\[ \|W\| = 1 \)。从第二点，我们得到我们应该尝试将\[ W \)的所有奇异值设置为1，而不仅仅是最大的。然后，如果\[ W \)的所有奇异值都等于1，这意味着\[ W \)的每个列的范数是1（因为每个列是\[ W e_i \)对于一些基本基向量\[ e_i \)，我们有\[ |W e_i| = |e_i| = 1 \)）。这意味着对于列\[ j \)我们有：

\[ \sum_{i} w_{ij}^2 = 1 \]

在列\[ j \)中有\[ n \)个条目，我们从相同的随机分布中选择每个条目，所以让我们找到一个随机权重\[ w \)的分布，使得：

\[ n E(w^2) = 1 \]

现在让我们假设我们希望在区间\[ [-R, R] \)中均匀地初始化\[ w \)。然后\[ w \)的均值为0，因此，根据定义，\[ E(w^2) \)是它的方差，\[ V(w) \)。在区间\[ [a, b] \)上的均匀分布的方差由\[ \frac{(b-a)^2}{12} \)给出，从中我们得到\[ V(w) = \frac{R^2}{3} \)。将这个代入我们的方程，我们得到：

\[ n \frac{R^2}{3} = 1 \]

所以：

\[ R = \sqrt{\frac{3}{n}} \]

这表明我们应该从区间\[ [-\sqrt{\frac{3}{n}}, \sqrt{\frac{3}{n}}] \)上的均匀分布中初始化我们的权重。

这是一个不错的结果，因为它是方差权重矩阵的Xavier-Glorot初始化，但它是由一个不同的想法驱动的。Xavier-Glorot初始化，由Glorot和Bengio（2010）引入，在实践中已被证明是一种有效的权重初始化处方。更一般地，Xavier-Glorot处方适用于在具有其导数在原点附近为一的激活函数的层中使用的\[ m \times n \)权重矩阵，并且说我们应该根据区间的均匀分布初始化我们的权重：  
\[ [-\sqrt{\frac{6}{m+n}}, \sqrt{\frac{6}{m+n}}] \)。

你可以很容易地修改上述分析，以获得在使用逻辑sigmoid（使用\[ \gamma = \frac{1}{4} \)）和根据不同的随机分布（例如，高斯分布）初始化权重时的初始化处方。

通过时间的反向传播和消失敏感性  
用反向传播训练RNN与用反向传播训练前馈网络非常相似。由于假设你已经熟悉反向传播，只有几点评论：

我们通过时间反向传播错误

对于RNNs，我们需要从当前RNN单元反向传播错误，通过状态，通过时间，回到先前的RNN单元。这允许RNN学习捕获长期时间依赖性。由于模型的参数在RNN单元之间共享（每个RNN单元具有相同的权重和偏置），我们需要分别计算每个时间步骤的梯度，然后将它们加起来。这类似于我们在其他模型中反向传播错误到共享参数的方式，例如卷积网络。

在权重更新频率和准确梯度之间存在权衡

对于所有基于梯度的训练算法，不可避免地存在（1）参数更新频率（向后传递）和（2）准确长期梯度之间的权衡。为了看到这一点，考虑当我们在每个步骤更新梯度，但将错误反向传播超过一个步骤时会发生什么：

在时间\[ t \)，我们使用当前权重\[ W_t \)计算当前输出和当前状态，\[ o_t \)和\[ s_t \)。  
其次，我们使用\[ o_t \)运行向后传递并更新\[ W_t \)到\[ W_{t+1} \)。  
第三，在时间\[ t+1 \)，我们使用\[ W_{t+1} \)和在第一步中计算的\[ s_t \)来计算\[ o_{t+1} \)和\[ s_{t+1} \)。  
最后，我们使用\[ o_{t+1} \)运行向后传递。但\[ o_{t+1} \)是使用\[ s_t \)计算的，这是使用原始\[ W_t \)计算的，这意味着我们为时间步骤\[ t \)的权重计算的梯度是在旧权重\[ W_t \)下评估的，而不是在当前权重\[ W_{t+1} \)下。因此，它们只是梯度的估计，如果它是针对当前权重计算的。随着我们进一步反向传播错误，这种效果将更加复杂。  
我们可以通过减少参数更新（向后传递）的频率来计算更准确的梯度，但这可能会导致我们放弃训练速度（这在训练开始时可能特别有害）。注意这与选择用于小批量梯度下降的小批量大小的权衡相似：批量大小越大，梯度估计越准确，但也更少的梯度更新。

我们还可以选择不将错误反向传播超过我们参数更新频率的步骤，但那时我们没有计算关于权重的完整梯度，这只是硬币的另一面；相同的权衡发生。

这种效应在Williams和Zipser（1995）中讨论，它提供了关于计算基于梯度的训练算法的梯度的选项的极好概述。

消失梯度加上共享参数意味着不平衡的梯度流和对最近干扰的过度敏感

考虑一个前馈神经网络。指数级消失的梯度意味着对早期层的权重所做的改变将比对后期层的权重所做的改变小得多。这很糟糕，即使我们对网络进行指数级更长时间的训练，以至于早期层最终学习。为了看到这一点，考虑在训练期间早期层和后期层学习如何相互通信。早期层最初发送粗略的信号，所以后期层很快就变得非常擅长解释这些粗略的信号。但随后早期层被鼓励学习如何产生更好的粗略符号，而不是产生更复杂的符号。

RNNs的情况更糟，因为与前馈网络不同，早期层和后期层的权重是共享的。这意味着他们不仅可以简单地误解，他们可以直接冲突：对特定权重的梯度可能在早期层是正的，但在后期层是负的，导致总体梯度为负，所以早期层的非学习速度比他们学习的速度更快。用Hochreiter和Schmidhuber（1997）的话说：“通过时间的反向传播对最近的干扰过于敏感。”

因此，截断反向传播是有意义的

限制我们在训练中反向传播错误的步数称为截断反向传播。立即注意到，如果我们要拟合的输入/输出序列无限长，我们必须截断反向传播，否则我们的算法将在向后传递时停止。如果序列是有限的但非常长，我们可能仍然需要由于计算不可行而截断反向传播。

然而，即使我们有一台可以瞬间反向传播错误无限步数的超级计算机，上述第2点告诉我们，由于权重更新导致的梯度变得不准确，我们需要截断我们的反向传播。

最后，消失的梯度为截断我们的反向传播创造了另一个原因。如果我们的梯度消失，那么反向传播许多步的梯度将非常小，对训练几乎没有影响。

请注意，我们不仅选择截断反向传播的频率，还选择更新我们的模型参数的频率。参见我关于截断反向传播风格的帖子，了解两种可能的截断方法的实证比较，或参考Williams和Zipser（1995）中的讨论。

还有所谓的梯度分量的前向传播

知道一些有用的东西（以防你自己想到）是，反向传播并不是我们训练RNN的唯一选择。而不是反向传播错误，我们也可以向前传播梯度分量，允许我们计算每个时间步骤的权重关于错误的梯度。这种替代算法称为“实时递归学习（RTRL）”。完整的RTRL计算成本过高，不切实际，运行时间为\[ O(n^