---
categories: graph
date: 2024-10-12T13:01:56+0800
dir: graph
modify: 2024-12-06T00:12:54+0800
share: true
tags:
  - graph
title: AI学习路径
---

建议直接看sklearn文档上手，如果是深度学习就看pytorch的tutorial。可以配一本统计学习方法查阅(好像有第三版了)，或者去搜一下动手学深度学习的pytorch版。  
喜欢数学的话，可以看foundation of machine learning。

---

不过认真来说的话，我是经历过从0开始入门的，整个过程找对方法就会比较轻松，找不对方法就会很自闭。

工具的话：推荐先学Python，然后是pytorch（拒绝一切花里胡哨的，官方文档是最新最权威的，建议直接看官方文档）。tensorflow、mxnet、paddle啥的用到再说，没必要急着学。  
理论方面：推荐先看吴恩达在deeplearning.ai的深度学习课程。这个课程比cs230更基础也更适合入门。这门课有五章，前三章我觉得是必看的。后两章最好也看一下，没时间就根据需要看吧。b站链接我放在这里，在评论区你可以找到后四门的链接：

进阶建议看台湾大学李宏毅系列课程，李老师很喜欢讲解深度学习领域前沿的技术，所以我很推荐。李老师主要是下面这几门，前两门我基本刷完了，收获很大。  
1、机器学习基础：讲最简单的机器学习和深度学习方法。内容和吴恩达老师的课有部分重复的，可以跳着看。这门课其实也比较老了，有些东西用不上了，时间紧张的话我更推荐直接看下面这一门。  
[http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML16.html 84](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML16.html)  
2、机器学习进阶：这部分讲深度学习在近几年的发展，内容覆盖GAN、强化学习、模型轻量化、元学习、自动化机器学习、终生机器学习、表示学习、transformer等前沿方向。另外还有TA的补充课，会更加详细地讲解这些技术。  
https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html  
3、machine learning and having it deep structured 这门课讲的是结构化机器学习，我觉得是上面一门的再进阶版，我看了里面的部分内容，感觉比上面一门更深也略偏。时间足够的话建议看。课程链接我放在这里：  
[http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html 14](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)

然后就是具体的方向。如果想做CV我建议先看李飞飞的cs231n，但是这课录像很老了已经落伍了。所以我更推荐密大的eecs498-007，同样是李飞飞的助教讲的但是加了很多新的东西。这里其实要注意，李飞飞老师的课和上面两位老师的风格很不同。上面两位老师会事无巨细的给你讲一些细节。但是李老师的课基本上是讲个大概框架，需要课下自己读论文，否则上完课还是什么都不懂。这两门应该在b站都有搬运，或者YouTube也能找到，我就不贴链接了。

如果想做语音/NLP方向我推荐李宏毅老师的人类语言处理。李老师是做语音方向的，在华人圈都比较有名，课程质量也很高。这门课其实一半在讲语音一半在讲文字。如果对只对其中一个方向感兴趣的话可以跳过另外一部分。  
[http://speech.ee.ntu.edu.tw/~tlkagk/courses_DLHLP20.html 15](http://speech.ee.ntu.edu.tw/~tlkagk/courses_DLHLP20.html)

如果想做强化学习我建议先看李宏毅老师的课入门。  
李老师这部分讲的非常浅显易懂，但是课毕竟是几年前的，一些新的算法没有包括进去。除此之外包括model-based算法也没有提到。所以入门之后我建议看港中文周博磊老师的强化学习纲要。这门课更加数学，更加严谨，和李老师的课是不同的风格，也更加进阶。  
[https://github.com/zhoubolei/introRL 13](https://github.com/zhoubolei/introRL)

如果想做搜推广方向，目前没有很好的网课，我建议看王喆老师的深度学习推荐系统这本书。但是这本书也比较薄，讲的不详细，所以需要速通一遍书然后开始看论文。

最后关于深度学习的工程方向，我建议学习王树森老师的分布式机器学习课程，弄明白分布式模型训练的基本原理，然后在b站开始找些大数据的课程看。这些培训班的课很多我就不推荐了。其他工具比如docker、k8s、mr、spark、flink什么的建议也掌握基本使用方法，进厂搬砖用到的概率比较大。

---

先上油管或者b站搜几个AI科普视频刺激一下兴趣  
然后开刷斯坦福CS230，遇到不懂的就上知乎搜简单解释，这样理论就懂一些了  
看完之后就跟着莫烦python把代码捣鼓一遍，遇到bug就搜Google来debug，这样代码也就入门了  
到这里应该就入门了，后面再要进阶到CV，NLP还是什么别的方向就把以上过程再迭代一遍就行咯

---

学界主流应该是 PyTorch，所以关于 PyTorch 的教程请看官方文档，少去 CSDN [Welcome to PyTorch Tutorials — PyTorch Tutorials 1.10.0+cu102 documentation 23](https://pytorch.org/tutorials/)

除了 PyTorch APIs 之外，还有一件非常重要的事情：**熟悉高级矩阵操作**，无论是 numpy 还是 PyTorch 都是通用的。
能不用 for 循环的地方就不用 for 循环，能用高级索引的地方就用高级索引 （换言之，能向量化就向量化，绝不用 naive Python 实现）
 搞不定这一点不要说自己会 DL
 
---

网络结构经典模型的话

 总结

卷积神经网络这块比较经典的网络可以看看这些 

- AlexNet（这个肯定看过 不过还是列出来一下hh
- VGG Inception DenseNet
- ResNet和ResNeXt  
	下面这几个也是挺经典的
- MobileNet EfficientNet  
	...

然后就是可以看看现代一点的，ConvNeXt这种  

---

Transformer类在cv上就是VIT Swin Transformers这种，开篇就是VIT。

---

然后NLP了解不深就不多嘴了hh.

---

然后还有就是像目标检测的yolo系列，生成对抗的gan家族，近几年比较火的diffusion.

---

此外我觉得比较有意思的是一些自监督和模型压缩方面的，好像有点偏题了hh.

比如上面的MobileNet，EfficientNet就是在模型上进行轻量化设计。

---

还有就是我最近也刚开始学的GNN, 但这块比较偏一点感觉，如果有兴趣再说

鱼鱼说的实践，具体做什么的话感觉还挺难说的？

- 我觉得如果是CV相关的，可以先从cifar100的训练开始，然后就可以在这上面去实现一些模型，快速熟悉几个经典的架构。
- 像目标检测，分类之类的任务应该也有很多开源项目，包括部署到手机或者连接电脑摄像头实时监测的，每玩过的话还挺有意思的。可以找个类似下面这种的试试看  
	[同济子豪兄的个人空间-同济子豪兄个人主页-哔哩哔哩视频](https://space.bilibili.com/1900783/channel/collectiondetail?sid=1631214)

（包括还有什么风格迁移，或者各种比较火的模型/方法的一些玩具项目应该有很多

---

然后还有就是一些更具体的？

- 比如说迁移学习（预训练-微调，包括Lora之类的），应该可以去微调比如一些AI绘图的稍微大一点的模型（比如说用一些自拍去调出自己人脸对应的prompt之类之类），这个应该挺有意思的
- 还想了一些但感觉有的没啥意思，有的价值不大，鱼鱼应该后面有兴趣可以在更细分的方向上找找，到时候再说

---

个人感觉是，pytorch的熟练程度重要性高一点，然后全流程走通后其实大部分dl的项目结构还是比较类似。  
模型方面，基本有里程碑意义的模型都有详细解读，也比较值得看，可以大致了解了解，细分方向的就太多了（另外太细的方向很多东西其实也不那么solid  
NLP感觉需要资源会多一点（特别在现在），加上我了解不是很深，很多东西只是知道有或者了解个大概工作原理  
很多搞dl的人代码能力比较弱，是学习路上一大阻力，但工程能力鱼鱼肯定是不用太担心的（算是一大优势  
具体来说就是，框架了解了解+某个领域经典模型（挑几个试试，挑几个看看），开始先从玩具项目或者能跑通的项目里看看大概全流程是怎么样的，后面就能找一些感觉比较有意思的去做（不管是实现新模型或者找一些细分领域或者做一些实践项目之类的，应该在这时候就会比较容易了

---

https://github.com/apachecn/ai-roadmap/blob/master/ai-union-201904/README.md