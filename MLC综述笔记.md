# 多标签分类问题概况 及 医学影像分类的思考

最近在做眼底图像的多标签分类，读了一下武大的刘威威老师的综述*The Emerging Trends of Multi-Label Learning*[^1]，自己也看了一点医学影像分类和自然图像多标签分类的文章。本文主要总结一下阅读之后对多标签分类（multi-label classification, MLC）问题的理解，以及对于医学影像上的多标签问题的特点的一点思考。

## 综述的概括

为了偷懒这里就不列各个方法的引用了，在概括综述内容的基础上加了一点自己的理解，是跳着读的所以只有部分内容。

### 综述结构
MLC问题的研究重点包含几个方面：
- **Extreme MLC**: XMLC, 就是类别数非常大的MLC场景。随着大数据时代的到来，这个场景的研究意义重大。
  - 大部分工作是基于SLEEC之后做的，主要有基于one-vs-all分类器、树、embedding三种思路。
  - 理论层面需要针对标签稀疏，处理长尾分布问题。
- **MLC with missing/nosiy label**：非全监督学习的MLC版本，主要针对标签的问题进行处理。
  - missing label：预设有的类别无label
  - semi-supervised：传统半监督学习的迁移，部分data有label，部分没有
  - partial multi-label: 部分label不可信，即模糊标签的场景
- **online MLC for stream data**：由于现在web端实时产出大量流数据，针对线上实时场景的MLC被大量关注。
  - 流数据无法预读到内存里获取全局，一般需要实时处理每个时间戳
  - 现有offline MLC model在序列数据上的效果一般
  - online MLC领域目前在实验和理论上暂无特别好的效果(very limited)

### §4 Deep Learning for MLC 

- **BP-MLL**

最早在MLC中使用NN结构的是BP-MLL, 它提出了一种pairwise的loss函数，如下：

$$
E_{i}=\frac{1}{\left|y_{i}^{1}\right|\left|y_{i}^{0}\right|} \sum_{(p, q) \in y_{i}^{1} \times y_{i}^{0}} \exp \left(-\left(F\left(x_{i}\right)^{p}-F\left(x_{i}\right)^{q}\right)\right)
$$

其中$p,q$分别为预测为1和0的类别，使用$e^{-x}$形式惩罚项，使得不同的类别间差值尽可能大，整体是一种rank loss的思路。

在随后的研究中发现，BP-MLL可以使用cross-entropy loss，再加上一点ReLu/Dropout/AdaGrad之类的trick，可以再经典BP-MLL无法应用的大规模文本分类的场景获得新的SOTA性能。

- **C2AE**

经典的Embedding方法只能获取label本身的语意dependency，不可能获取更高维的联系，C2AE（Canonical Correlated AutoEncoder）是第一个基于Embedding的MLC方法，它通过自编码器提取特征，利用DCCA（deep canonical correlation analysis）基于特征提取label间的联系，属于embedding层。

C2AE整体目标函数定义如下：
$$
\min _{F_{x}, F_{e}, F_{d}} \Phi\left(F_{x}, F_{e}\right)+\alpha \Gamma\left(F_{e}, F_{d}\right)
$$
$F_x, F_e, F_d$分别为 特征映射、编码函数、解码函数， $\alpha$是平衡两个惩罚项的权重项。$\Phi, \Gamma$分别为latent空间(feature和encoding之间)和output空间上(encoding和decoding之间)的loss。

借鉴了CCA的思路，C2AE使instance和label的联系尽可能大（最小化差距）
$$
\begin{aligned}
\min _{F_{x}, F_{e}} &\left\|F_{x}(X)-F_{e}(Y)\right\|_{F}^{2} \\
\text { s.t. } & F_{x}(X) F_{x}(X)^{T}=F_{e}(Y) F_{e}(Y)^{T}=I
\end{aligned}
$$

自编码器使用和上文相似的rank loss，使得不同类别的code差别尽可能大。
$$
\begin{array}{l}
\Gamma\left(F_{e}, F_{d}\right)=\sum_{i=1}^{N} E_{i} \\
E_{i}=\frac{1}{\left|y_{i}^{1}\right|\left|y_{i}^{0}\right|} \sum_{(p, q) \in y_{i}^{1} \times y_{i}^{0}} \exp \left(-\left(F_{d}\left(F_{e}\left(x_{i}\right)\right)^{p}-F_{d}\left(F_{e}\left(x_{i}\right)\right)^{q}\right)\right)
\end{array}
$$
后续的DCSPE, DBPC等工作进一步提升了文本分类上的SOTA性能和推理速度。

---
- **patial and weak-supervised MLC**

CVPR 2020中*D. Huynh*的*Interactive multi-label cnn learning with partial labels*和CVPR 2019中*T. Durand*的*Learning a deep convnet for multi-label classification with partial labels*（以下根据坐着名称简称D和T）做了相关研究。

T使用BCE Loss训练有标签部分，然后使用GNN提取标签间联系。实验证明部分标注的大数据集比全标注的小数据集效果要好，进一步证明了partial label的MLC领域的研究意义。

D在T的基础上，使用流形学习的思路，将label和feature的流形平滑度作为BCE Loss函数的cost，再使用半监督的思路，CNN学习和similarity同步进行（我没看这篇文章，听综述的这种描述类似于$\pi$模型或者teacher-student结构）。

- **SOTA的Advanced MLC**

**分类链**：ADIOS把label切分成马尔科夫毯链（Markov Blanket Chain），可以提取label间的关系，然后丢进DNN训练。

**CRNN**：有2篇文章把类别作为序列，使用CRNN或者C-LSTM处理。更进一步对于类别序列的顺序使用attention/RL进行学习，寻找最优顺序。CVPR 2020和AAAI 2020各有一篇此思路的，使用optimal completion distillation+多任务学习/最小alignment的思路，都是尝试动态调整label sequence的order（order-free）。

**graph相关**
- [^2] 建立一个类别间的有向图，然后使用GCN训练。
- SSGRL[^6] 使用embedding进行semantic decoupling, 然后使用GNN学习label+feature构成的-semantic，强化instance和label特征，以学习更高维的label间的联系。
- [^3] 对GCN和CNN的一些layer间添加连接，从而实现label-aware的分类学习
- [^4] 使用GCN获取rich semantic info，再使用non-local attention获取长语意关联。
- [^5] 使用深度森林，一种tree ensemble方式，不依赖回传机制。提出了MLDF(multi-label Deep Forest)，据说可以更好地解决过拟合，在6种指标上取得了SOTA的效果，是lightweight设计的一个探索。

## 医学影像的MLC思考

以前看医学图像分割的文章(DeepIGeoS)，国泰对于医学图像的特殊点概括为：
1. 低对比度，高噪声，存在空腔
2. 患者间scale和feature差异巨大
3. 疾病间的不均匀表征
4. 医生定义不同会造成ground-truth特征不一致

这主要针对与分割而言，因为一般分割任务的CT和MRI图像是高Intensity的灰度图像，感觉在MLC场景中1和2基本都不咋适用。

3在MLC中表现为不同类别的feature的不均匀，例如有的疾病可能可观测症状覆盖很大区域，有的就只是很小的部分会出现可观测的症状，感觉类似于FPN的multi-scale策略对于特征提取会有一些帮助，不过这是一个很general的推测，具体效果需要在具体的场景下多做实验。

4可以联系上MLC中的partial label问题，如果对于疾病的判断是不确定的，例如医生对一个患者得出几种可能病症，此时又没有进一步检查，那么也许可以设计一种方法预测各个label的置信度，哈哈哈感觉这是一个paper的idea了，可惜场景和数据的要求感觉有些苛刻。

另外值得一提的就是类别不平衡，由于一些疾病的病例较少，可能收集到的data里只有个位数的正例，此时这个类别很可能根本学不到啥，目前想法不是很清晰，过几天有时间再专门调研一下这个问题。

最后就是医学图像喜闻乐见的半监督，如果有部分没有标注的数据和一些标注的数据，拿来做半监督对性能也能提升一些，虽然不局限医学图像，但是由于医学标注获取较难，半监督的应用也特别广，大有可为吧可以说。



## 参考文献

[^1]: W. Liu, X. Shen, H. Wang, and I. W. Tsang, “The Emerging Trends of Multi-Label Learning,” arXiv:2011.11197 [cs], Dec. 2020, Accessed: Jan. 08, 2021. [Online]. Available: http://arxiv.org/abs/2011.11197.

[^2]: Z. Chen, X. Wei, P. Wang, and Y. Guo, “Multi-label image recognition with graph convolutional networks,” in CVPR, 2019, pp. 5177–5186.

[^3]: Y. Wang, D. He, F. Li, X. Long, Z. Zhou, J. Ma, and S. Wen, “Multilabel classification with label graph superimposing,” in AAAI, 2020, pp. 12 265–12 272.

[^4]: P. Tang, M. Jiang, B. N. Xia, J. W. Pitera, J. Welser, and N. V. Chawla, “Multi-label patent categorization with non-local attention-based graph convolutional network,” in AAAI, 2020.

[^5]: L. Yang, X. Wu, Y. Jiang, and Z. Zhou, “Multi-label learning with deep forest,” CoRR, vol. abs/1911.06557, 2019.

[^6]: T. Chen, M. Xu, X. Hui, H. Wu, and L. Lin, “Learning semanticspecific graph representation for multi-label image recognition,” in ICCV, 2019, pp. 522–531.
