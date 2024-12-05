---
title: CS路线1
date: 2024-10-12T13:01:56+0800
modify: 2024-12-06T00:17:29+0800
categories: 杂七杂八
dir: Obsidian
share: true
tags:
  - 杂七杂八
  - plan
---

计算机科学是一门理论与工程结合紧密的学科，它所涵盖的内容，可大致分为：

1. 通过“计算”的数学视角理解待解决问题。
2. 理解“计算”如何可靠、高效地运作。
3. 熟悉开发过程本身，熟悉软件工程，熟悉具体子领域的开发生态。

计算机的研究，从左端的人，到右端的机器，是一个光谱。都大抵把握，是不错的。不太区分顺序，SICP（+可以配合某些程序设计课） => 数据结构/算法 => [计算机系统基础](https://www.zhihu.com/search?q=%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%B3%BB%E7%BB%9F%E5%9F%BA%E7%A1%80&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D) 之后，其他课程的顺序凭你喜好即可 。

1. 理论地理解计算本身是什么，如**SICP**（cs61a），学习所谓的递归，函数式（数学式）编程等，可参考南京大学拷贝过来的版本[1](1.md)(#ref_1)。
1. python，c/c++，java，程序设计课上都写过一两千行就可以了。具体的语言不重要的。之后如果用到新的语言，比如js等，更多只是生态上的差异，不必单独特别学习。
3. 了解经典问题背后的数学结构，解决思路（计算方法/算法）。也即**数据结构，**[**算法导论**](https://www.zhihu.com/search?q=%E7%AE%97%E6%B3%95%E5%AF%BC%E8%AE%BA&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D)的内容，cs61b等。这时你大概可以保持练习[leetcode](https://www.zhihu.com/search?q=leetcode&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D)了。
1. **少量地了解计算复杂性、**[**计算理论**](https://www.zhihu.com/search?q=%E8%AE%A1%E7%AE%97%E7%90%86%E8%AE%BA&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D)（什么问题能算，问题之间如何等价、不等价），会有所帮助。[蒋炎岩](https://www.zhihu.com/search?q=%E8%92%8B%E7%82%8E%E5%B2%A9&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D)老师的推荐列表可以看看[2](2.md)(#ref_2)。
5. 了解程序的运行过程。这包括三点：
1. [高层编程语言](https://www.zhihu.com/search?q=%E9%AB%98%E5%B1%82%E7%BC%96%E7%A8%8B%E8%AF%AD%E8%A8%80&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D)，大量复杂的语言特性，是如何准确高效地编译至机器的[二进制语言](https://www.zhihu.com/search?q=%E4%BA%8C%E8%BF%9B%E5%88%B6%E8%AF%AD%E8%A8%80&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D)上**。**这是**编译原理**的内容，以中端优化为主，但全部过程都要有所了解，preprocess/parse/passes/link，重要的工具LLVM等。Crafting the interpreter也推荐看/写。
2. 机器本身，自底向上，从逻辑门开始，如何构建出高性能的计算系统。可参考CSAPP，蒋炎岩老师的《**计算机系统基础**课》[3](3.md)(#ref_3)，更深的看些**计算机系体结构**的内容也好。
3. 人类对机器/设备本身的抽象。[高级编程语言](https://www.zhihu.com/search?q=%E9%AB%98%E7%BA%A7%E7%BC%96%E7%A8%8B%E8%AF%AD%E8%A8%80&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D)不是直接变成code on bear metal，而是被封装为**操作系统**。这部分可以看蒋炎岩老师的操作系统课，B站上有录播，链接同上。
4. 由于机器间必然涉及通信，所以要学习[**计算机网络**](https://www.zhihu.com/search?q=%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%BD%91%E7%BB%9C&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D)。
5. 由于机器需要存储信息，所以要学习**数据库**（至少是使用，少量的理论/细节实现）。
6. 其他好像无关的话题，比如虚拟机，存储，并行并发分布式，图形学，[高性能计算](https://www.zhihu.com/search?q=%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D)，搜索引擎等，学习过程你会慢慢涉及的。
7. 了解计算机较普遍的应用。
1. Web开发，前端/后端等，如vue/[react](https://www.zhihu.com/search?q=react&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D)，spring等，根据兴趣学习，再从某处深入，寻找参与开源/实习的契机。其他**软件开发**等，如游戏开发，归为此类。
2. **AI**。应用一定向更高功能性发展，了解AI理论/实践是必不可少的。当然，会前置一些数学知识，线性代数等。可参考李沐老师的《动手学深度学习》。
3. 同步地了解常见[**设计模式**](https://www.zhihu.com/search?q=%E8%AE%BE%E8%AE%A1%E6%A8%A1%E5%BC%8F&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D)。
9. 了解如何更好的使用电脑，如何更好的和人协作，了解人们需要什么软件。
1. 你自己要会用自己的电脑，成为合格的程序员。可以看[**the missing semester**](https://www.zhihu.com/search?q=the%20missing%20semester&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3054321658%7D)，[https://missing-semester-cn.github.io/](https://link.zhihu.com/?target=https%3A//missing-semester-cn.github.io/)，也务必了解稳定的**安全上网**方式。你也要开始管理自己的工作流/信息流。
2. 非自娱自乐的大型软件的开发管理流程，软件的复杂性，你需要有印象。这是**软件工程**的内容，也有点管理学的感觉。你也需要能够沟通交流，呈现自己的想法，这是重要的软实力。
3. 你需要了解一点产品经理/UX设计者等在做的事情，了解人们如何使用软件，人们需要怎样的软件等。这可能是一点社会学/心理学/**设计学**的内容。

这算是我对计算机各领域的脉络的把握，希望有所帮助。在这之后，你可能希望在某一个领域深入，不论是工程意义上的还是研究意义上的，这时你自己应看得见方向了。更详细的分类，csrankings上[4](4.md)(#ref_4)你可以看到。