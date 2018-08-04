# Logistic Regress

[TOC]

## 基础知识

### 熵的定义

熵表示随机变量不确定性的度量. 设X是一个有限个值的离散随机变量,其概率分布为
$$
P(X=x_i) = p_i, i=1,2, \cdots, n
$$
则随机变量X的熵定义为
$$
H(X)=-\sum_{x \in X}P(x)logP(x) = -\sum_{i=1}^np_ilogp_i
$$
熵越大不确定性越大,并且熵的取值范围为
$$
0 \leqslant H(p) \leqslant logn
$$

### 联合熵

设有随机变量（X，Y），其联合概率分布为
$$
P(X=x_i, Y=y_j) = P_{ij}, i=1,2,\cdots,n; j=1,2,\cdots,m
$$
两个随机变量X,Y的联合概率分布,可以形成联合熵
$$
H(X,Y) = -\sum_{x\in X,y \in Y}P(x,y)logP(x,y) =- \sum_{i=1}^n\sum_{j=1}^mP_{ij}logP_{ij}
$$

### 条件熵

条件熵H（Y|X）表示在已知随机变量X的条件下随机变量Y的不确定性，定义为X给定条件下Y的条件概率分布的熵对X的数学期望
$$
\begin{align}
H(Y|X)& = \sum_{x \in X}P(x)H(Y|X=x) = \sum_{i=1}^np(X=x_i)H(Y|X=x_i) \\
 & = \sum_{x \in X}P(x)[-\sum_{y \in Y}P(y|x)logP(y|x)] \\
 &= \sum_{i=1}^nP(X=x_i)[-\sum_{j=1}^mP(Y=y_j|X=x_i)logP(Y=y_j|X=x_i)] \\
 & = -\sum_{x \in X}P(x)\sum_{y \in Y}P(y|x)logP(y|x) \\
\end{align}
$$

根据条件概率公式
$$
P(Y|X) = \frac{ P(X,Y)}{P(X)}
$$
又由于$\sum$如下公式
$$
A = \{a_i \}, i=1,2,\cdots,n \\
B = \{b_j\}, j=1,2, \cdots, m \\
\sum_{a \in A}a \cdot \sum_{b \in B}b = \\
a_1\cdot(b_1+b2+\cdots + b_m) + \\
a_2 \cdot (b_1 + b_2+\cdots + b_m) + \\
\cdots \\
a_n \cdot(b_1 + b_2 + \cdots + b_m) \\
= \\
a_1b_1 + a_1b_2+ \cdots + a_1b_m + \\
a_2b_1 + a_2b_2 + \cdots + a_2b_m + \\
\cdots \\
a_nb_1 + a_nb_2 + \cdots + a_nb_m = \\
\sum_{a \in A,b\in B}a\cdot b
$$
因此,条件熵可以化简为:
$$
\begin{align}
H(Y|X)& = -\sum_{x \in X}P(x)\sum_{y \in Y}P(y|x)logP(y|x) \\
& = -\sum_{x\in X,y \in Y}P(x)P(y|x)P(y|x) \\
& = -\sum_{x \in X, y \in Y}P(x,y)logP(y|x)
\end{align}
$$
条件熵还有另外一个计算方法,可以表示为
$$
\begin{align}
H(Y|X)& = H(X,Y) - H(X) \\
& =  -\sum_{x\in X,y \in Y}P(x,y)logP(x,y) - [-\sum_{x \in X}P(x)logP(x)] \\
& =\sum_{x \in X}P(x)logP(x) -\sum_{x\in X,y \in Y}P(x,y)logP(x,y) \\
& = \sum_{x \in X}[P(x)logP(x) - \sum_{y \in Y}P(x,y)logP(x,y)] \\
& = \sum_{x \in X}[\sum_{y\in Y}P(x,y)logP(x) - \sum_{y \in Y}P(x,y)logP(x,y)] \\
& = \sum_{x \in X}[\sum_{y \in Y}P(x,y)(-log\frac{P(x,y)}{P(x)}]  \\
& = -\sum_{x \in X, y \in Y}P(x,y)logP(y|x)
 \end{align} \\
 其中，\sum_{y \in Y}P(x,y) = P(x)
$$

与上面的结果相同。

### 最大熵原理

学习概率模型时,在所有可能的概率模型中,熵最大的模型是最好的模型.

## 最大熵模型

假设分类模型是一个条件概率分布P（Y|X）， $X\in \chi \subseteq R^n$ 表示输入，$Y \in y$表示输出，这个模型表示的是对于给定的输入X，以条件概率P（Y|X）输出Y。

对于给定训练数据，可以确定联合概率分布P（X，Y）的经验分布和边缘分布P（X）的经验分布：
$$
\tilde{P}(X=x,Y=y) = \frac{v(X=x, Y=y)}{N} \\
\tilde{P}(X=x) = \frac{v(X=x)}{N} \\
其中，v(X=x,Y=y)表示训练样本中（x，y）出现的频数，\\
v（X=x）表示训练样本中输入x出现的频数，\\
N表示训练样本容量。
$$
用特征函数f（x，y）描述输入x与输出y之间的某一事实，其定义为
$$
f(x,y) = 
\begin{cases}
1, x与y满足某一事实 \\
0, 否则
\end{cases} \\
二值函数，当x和y满足这个事实时取1，否则取0.
$$
特征函数f（x，y）关于经验分布$\tilde{P}(X,Y)$的期望值为
$$
E_{\tilde{P}}(f) = \sum_{x,y}\tilde{P}(x,y)f(x,y)
$$
特征函数f（x，y）关于模型P（Y|X）与经验分布$\tilde{P}(X)$的期望值为
$$
E_P(f) = \sum_{x,y}\tilde{P}(x)P(y|x)f(x,y)
$$
如果模型能够获取训练数据中的信息，那么就可以假设这两个期望值相等，即
$$
E_P(f) = E_{\tilde{P}}(f) \\
即 \\
\sum_{x,y}\tilde{P}(x)P(y|x)f(x,y) = \sum_{x,y}\tilde{P}(x,y)f(x,y)
$$
我们将上式作为模型学习的约束条件，假设有n个特征函数$f_i(x,y), i=1,2,\cdots, n$，那么就有n个约束条件。

得出，所有满足约束条件的模型集合为
$$
C = \{P | E_P(f_i) = E_{\tilde{P}}(f_i), i=1,2,\cdots,n\}
$$
定义在条件概率分布P（Y|X）上的条件熵为
$$
H(P) = -\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x)
$$
模型集合C中条件熵H（P）最大的模型称为最大熵模型。

最大熵模型学习可以形式化为约束最优化问题。

对于给定的训练数据集$T=\{(x_1,y_1), (x_2,y_2), \cdots, (x_N, y_N) \}$以及特征函数$f_i(x,y), i=1,2,\cdots,n$，最大熵模型的学习等价于约束最优化问题：
$$
\begin{align}
\underset{P\in C}{max} \text{ }   & H(P) = -\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x) \\
s.t. \text{    }&E_P(f_i) = E_{\tilde{P}}(f_i), i=1,2,\cdots,n \\
& \sum_yP(y|x) = 1
\end{align}
$$
### 最大熵模型求解

按照最优化问题的习惯，将求最大值的问题改成等价的求最小值问题：
$$
\begin{align}
\underset{P\in C}{min} \text{ }   & -H(P) = \sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x) \\
s.t. \text{    }&E_P(f_i) - E_{\tilde{P}}(f_i) =0, i=1,2,\cdots,n \\
& 1-  \sum_yP(y|x) = 0
\end{align}
$$
求解上式约束最优化问题所得出的解，就是最大熵模型学习的解。求解过程如下：

将约束最优化的原始问题转化为无约束最优化的对偶问题。

首先，引入拉格朗日乘子$w_0, w_1, \cdots,w_n$,定义拉格朗日函数L（P，w）：
$$
\begin{align}
L(P,w) & \equiv-H(P) + w_0(1-\sum_yP(y|x)) + \sum_{i=1}^nw_i(E_P(f_i) - E_{\tilde{P}}(f_i)) \\
& = \sum_{x \in X, y \in Y} \tilde{P}(x)P(y|x)logP(y|x) + w_0(1-\sum_{y \in Y}P(y|x)) \\
& + \sum_{i=1}^nw_i(\sum_{x \in X,y \in Y}\tilde{P}(x)P(y|x)f_i(x,y) - \sum_{x \in X,y \in Y}\tilde{P}(x,y)f_i(x,y))
\end{align}
$$

最优化的原始问题是
$$
\underset{P\in C}{min} \text{ } \underset{w}{max}L(P,w)
$$
对偶问题是
$$
\underset{w}{max}\text{ }\underset{P \in C}{min}L(P,w)
$$
由于拉格朗日函数L（P，w）是P的凸函数，因此原始问题与对偶问题是等价的。这样可以通过求解对偶问题来求原始问题。

首先，求解对偶问题的内部极小化问题，$\underset{P\in C}{min}L(P,w)$,得出解$P_w^*$（其可用w来表示,故加了下角标w）之后，拉格朗日函数可以认为是关于w的函数，然后再求解$w^*$使得外部函数极大化。具体求解过程如下：

设：
$$
P_w^* = arg\underset{P \in C}{min}L(P,w) \\
\Psi(w) = L(P_w^*, w) \\
w^* = arg \underset{w}{max}\Psi(w)
$$
一步一步来，先求$P_w^*$，（理解一下这里，P（y|x）可以认为是一个输入为（x，y）映射到该输入对应概率的函数，那么这里其实是对函数求偏导，与平常的对某个自变量求偏导略有不同，但无伤大雅，可以设：P=P（y|x))：
$$
\begin{align}
\frac{\partial L(P,w)}{\partial P(y|x)} &= \tilde{P}(x)(logP(y|x)+1) - w_0 + \sum_{i=1}^nw_i\tilde{P}(x)f_i(x,y) \\
\end{align} \\
这里的x和y为某一个x，y，而其他x，y偏导得0，因此无需用 \sum进行连加
$$
仔细考虑了下，李航的《统计学习方法》书中P.84页中偏导的推理，觉得并不严谨。我的推理如上。以下为根据李航求的偏导进行计算的过程：
$$
\sum_{x \in X, y \in Y}\tilde{P}(x)(logP(y|x)+1) = \sum_{y \in Y}w_0 - \sum_{i=1}^nw_i\sum_{x \in X, y\in Y}\tilde{P}(x)f_i(x,y) \\
=>\sum_{x \in X, y \in Y}\tilde{P}(x)(logP(y|x)+1) = \sum_{y \in Y} \left( \sum_{x\in X}\tilde{P}(x)\right) w_0 - \sum_{i=1}^nw_i\sum_{x \in X, y\in Y}\tilde{P}(x)f_i(x,y)  \\
=> \sum_{x \in X, y \in Y}\tilde{P}(x)(logP(y|x)+1) = \sum_{x \in X, y \in Y}\tilde{P}(x)\left( w_0 - \sum_{i=1}^nw_if_i(x,y) \right) \\
=> \sum_{y \in Y}(logP(y|x)+1) = \sum_{y \in Y}\left ( w_0 - \sum_iw_if_i(x,y)\right) \\
并不能推出 logP(y|x) + 1 = w_0 - \sum_iw_if_i(x,y) ?
$$
根据我的偏导进行计算，令其等于0得：
$$
\tilde{P}(x)\left(logP(y|x)+1\right) = w_0 - \sum_{i=1}^nw_i\tilde{P}(x)f_i(x,y) \\
=> logP(y|x) = \frac{ w_0 - \sum_{i=1}^nw_i\tilde{P}(x)f_i(x,y) }{\tilde{P}(x)} -1 \\
=> P(y|x) = exp\left(\frac{ w_0 - \sum_{i=1}^nw_i\tilde{P}(x)f_i(x,y) }{\tilde{P}(x)} -1  \right)
$$


其中用到如下导数公式：
$$
{lnx}' = \frac{1}{x} \\
{(xlnx)}' = (lnx + 1)
$$


## 参考

1. https://www.cnblogs.com/wxquare/p/5858008.html
2. http://www.cnblogs.com/ooon/p/5707889.html
3. https://www.cnblogs.com/ooon/p/5677098.html

















