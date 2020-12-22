# 随手记——几个深度常见metrics

## kappa

### 公式
$$
kappa = \frac {p_o-p_e} {1-p_e}
$$
其中
$$p_o=\frac {分类正确的样本数} {所有样本数} = (混淆矩阵中的)\frac {对角线元素之和} {所有元素之和}$$
$$p_e = \frac {\sum{y_{pred}的第i类样本数 \times y_{true}的第i类样本数}} {(所有样本数)^2} \\
 = (混淆矩阵中的) \frac {\sum {第i行元素和 \times 第i列元素和}} {(所有元素之和)^2} 
$$

### 例子
```
y_true = [2, 0, 1, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
```
根据以上y_true和y_pred构建的混淆矩阵如下
| | 0(y_true) | 1(y_true) | 2(y_true) | 
| :-: | :-: | :-: | :-: |
| **0(y_pred)** | 2 | 0 | 1 |
| **1(y_pred)** | 0 | 0 | 0 |
| **2(y_pred)** | 0 | 2 | 1 |


接着计算
$p_o=(2+1)/6=\frac {1}{2}$，$p_e=(3\times 2+0\times 2+3\times 2)/(6\times 6)=\frac {1}{3}$

最终
$$
kappa = \frac{\frac {1}{2}-\frac {1}{3}}{1-\frac {1}{3}} = 0.25
$$

代码验证
```
from sklearn.metrics import cohen_kappa_score
kappa_value = cohen_kappa_score(y_true, y_pred)
输出结果 0.250000
```
### 设计目的
为了解决类别不平衡问题，kappa依靠$p_e$在类别越不平衡就越大的特点，使得类别不平衡时kappa分数会更低。

## F1-score
回忆基础概念，二分类混淆矩阵中，有这些定义：
- **TP**: true positive， 实际为true，预测为positive
- **TN**: true negative， 实际为true，预测为negative
- **FP**: false positive， 实际为false，预测为positive
- **FN**: false negative， 实际为false，预测为negative

在这些的基础上，定义了三个指标:
- Accuracy: 准确率, $\frac {TP+TN}{TP+TN+FP+FN}$, 预测分类准确的比例
- Precision: 精确率, $\frac {TP}{TP+FP}$, 预测为positive中正确的比例
- Recall: 召回率, $\frac {TP}{TP+TN}$, 实际为true中被预测出来的比例

为了综合Precision和Recall，求取两者的**调和平均数**，**因为无法直接得知TP+FP和TP+TN的数量，直接用加权平均无法确定合理的权重**
$$ F1 = \frac {2*precision*recall}{precision+revall} $$

## Dice 
非常奇妙的是，F1-score也被称作Dice similarity coefficient，也就是说他的含义和医学影像分割中常用的Dice是一毛一样的。

Dice一般定义如下：
$$
\text { DiceCoefficient }=\frac{2|X \cap Y|}{|X|+|Y|}
$$
咱们可以“惊讶”地发现，如果把pred($X$)和ground-truth($Y$)理解为用[0,1]标注的分类(*其实这正是语义分割的原始定义，pixel-wise的分类问题* )，可见Dice确实和F1-score是一样的。

