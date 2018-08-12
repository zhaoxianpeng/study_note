# HMM

[TOC]

## 符号说明

| 符号                         | 含义                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| $Q=\{q^1,q^2,\cdots, q^T\}$  | 长度为T的状态序列                                            |
| $Y=\{y^1,y^2,\cdots,y^T\}$   | 长度为T的观测序列                                            |
| $S=\{s_1,s_2,\cdots,s_N\}$   | 状态变量$q^t$有N个可能取值的离散空间，状态取值的集合（又称状态空间） |
| $O=\{o_1,o_2, \cdots, o_M\}$ | 观测变量$y^t$有M个可能的取值空间，当然观测变量有可能是连续型的，为了便于讨论，这里我们仅考虑离散型观测变量 |
| $A_{N\times N}$              | $N\times N$维状态转移概率矩阵，其中 $a_{ij}=P(q^{t+1}=s_j | q^t=s_i), \quad 1 \leq i,j \leq N$表示任意时刻t，若状态为$s_j$，下一时刻转到状态$s_j$的概率 |
| $B_{N\times M}$              | $N \times M$维输出观测概率矩阵（又称发射矩阵），其中$b_{ij} = P(y^t=o_j | q^t=s_i),1\leq i \leq N, 1\leq j\leq M$表示任意时刻t，若状态为$s_i$则观测变量观测到$o_j$的概率 |
| $\pi_{1\times N}$            | N维初始状态概率向量，其中$\pi_i=P(q^1 = s_i), 1 \leq i \leq N$，表示模型的初始状态为$s_i$的概率。 |

## 基本概念

由上表可知一个隐马尔科夫模型（hidde Markov Model)由初始状态概率向量，状态转移概率矩阵和输出观测概率矩阵决定，这些参数确定了，那么这个隐马尔科夫模型也就确定了，我们用$\lambda=(A,B, \pi)$来表示一个HMM。

### 基本假设

1. 齐次马尔科夫假设：HMM在任意时刻t只依赖于前一时刻的状态，与其他时刻的状态和观测都无关，也和t无关：
   $$
   P(q^t|q^{t-1},y^{t-1}, \cdots, q^1,y^1) = P(q^t|q^{t-1})
   $$

2. 观测独立性假设：任意时刻t的观测只依赖于该时刻的状态，与其他观测和状态都无关，也和t无关：
   $$
   P(y^t|q^T,y^T, \cdots, q^t, q^{t-1},y^{t-1}, \cdots, q^1,y^1) = p(y^t|q^t)
   $$


### 基本问题

隐马尔科夫通常用来解决这3类问题：

1. 评估，评估模型与观测序列之间的匹配程度：给定模型$\lambda=(A,B, \pi)$和观测序列$Y=\{y^1,y^2,\cdots,y^T\}$，求在模型$\lambda$下观测序列Y出现的概率$P(Y|\lambda)$；
2. 学习，已知观测序列Y，估计模型的参数$\lambda$，使得在该模型下观测到Y序列的概率$P(Y|\lambda)$最大，即用最大似然估计法估计参数；
3. 预测，根据观测序列推断最有可能的隐藏的状态序列：给定模型$\lambda$，和观测序列Y，求状态序列Q使得$P(Q|O,\lambda)$最大。



## 评估算法

针对第1类问题：给定模型$\lambda=(A,B, \pi)$和观测序列$Y=\{y^1,y^2,\cdots,y^T\}$，求在模型$\lambda$下观测序列Y出现的概率$P(Y|\lambda)$；

### 直接计算

列举所有可能的长度为T的状态序列Q，计算$P(Y,Q|\lambda)$，然后对所有可能的状态序列求和来得到结果：
$$
P(Y|\lambda) = \sum_QP(Y,Q|\lambda) = \sum_QP(Y|Q, \lambda)P(Q|\lambda)
$$
其中：
$$
P(Y|Q,\lambda) = b_{q^1y^1}b_{q^2y^2}\cdots b_{q^Ty^T}\\
P(Q|\lambda) = \pi_{q^1}a_{q^1q^2}\cdots a_{q^{T-1}q^T} \\
P(Y,Q|\lambda) = P(Y|Q,\lambda)P(Q|\lambda) =  \pi_{q^1}b_{q^1y^1}a_{q^1q^2}b_{q^2y^2}\cdots  a_{q^{T-1}q^T}b_{q^Ty^T}\\
$$


展开上式得：
$$
P(Y|\lambda) =\sum_QP(Y|Q, \lambda)P(Q|\lambda) \\
= \sum_{q^1,q^2,\cdots,q^T} \pi_{q^1}b_{q^1y^1}a_{q^1q^2}b_{q^2y^2}\cdots  a_{q^{T-1}q^T}b_{q^Ty^T}
$$
上式计算量很大，长度为T的状态序列有$N^T$个。

### 前向算法

定义前向变量算子：
$$
\alpha^t(i) = P(y^1, y^2, \cdots, y^t, q^t = s_i|\lambda)
$$
表示给定HMM模型参数为$\lambda$到t时刻部分观测序列为$y^1,y^2,\cdots,y^t$且状态$q^t$为$s_i$的概率。

#### 递推

$$
\begin{align}
\alpha^{t+1}(i) & = P(y^1, y^2, \cdots, y^t, y^{t+1} , q^{t+1} = s_i|\lambda) \\
&=P(y^1, y^2, \cdots, y^t, y^{t+1} , q^{t+1} = s_i|\lambda) \\
&= P(y^1, y^2, \cdots, y^t, q^{t+1}=s_i|\lambda)P(y^{t+1}|\underbrace{y^1, y^2, \cdots, y^t}_{与y^{t+1}无关}, q^{t+1}=s_i) \\
&= P(y^1, y^2, \cdots, y^t, q^{t+1}=s_i|\lambda)P(y^{t+1}|q^{t+1}=s_i)  \quad 其中P(y^{t+1}|q^{t+1}=s_i)= b_{iy^{t+1}} \\
&= P(y^1, y^2, \cdots, y^t, q^{t+1}=s_i|\lambda) b_{iy^{t+1}}
\end{align}
$$

又由于：
$$
P(q^{t+1}=s_i|\lambda) = \sum_{j=1}^NP(q^t=s_j|\lambda)a_{ji} \quad 得到\\
P(y^1, y^2, \cdots, y^t, q^{t+1}=s_i|\lambda) = \sum_{j=1}^NP(y^1, y^2, \cdots, y^t, q^{t}=s_j|\lambda)a_ji \\
= \sum_{j=1}^N \alpha^t(j)a_{ji}
$$
合并上两式得到前向算子的递推公式：
$$
\alpha^{t+1}(i) =[\sum_{j=1}^N \alpha^t(j)a_{ji}]b_{iy^{t+1}}
$$


根据前向算子定义，有：
$$
\alpha^T(i) = P(y^1, y^2, \cdots, y^T, q^T = s_i|\lambda)
$$
而：
$$
P(Y|\lambda) = P(y^1,y^2,\cdots, y^T|\lambda) = \sum_{i=1}^NP(y^1,y^2,\cdots, y^T, q^T=s_i|\lambda) = \sum_{i=1}^N\alpha^T(i)
$$
至此，终于用前向算法求出$P(Y|\lambda)$，计算量比直接计算小多了。

### 后向算法



## 学习算法

学习算法是已知观测序列Y，估计模型的参数$\lambda$，使得在该模型下观测到Y序列的概率$P(Y|\lambda)$最大。

根据训练数据是否包含状态序列Q分监督学习和非监督学习。

### 监督学习

已知观测序列Y，状态序列Q，估计模型的参数$\lambda$，求$\lambda$使得在该模型下观测到Y序列的概率$P(Y|\lambda)$最大。

这种情况，未知变量只有$\lambda$，因此可以直接用极大似然估计来估计模型参数。

### 非监督学习

只知道观测序列Y，没有对应的状态序列Q，学习模型参数$\lambda$使得在该模型下观测到Y序列的概率$P(Y|\lambda)$最大。

由于含有隐藏变量，无法直接用极大似然估计求解，可以由EM算法实现。

1. 确定对数似然函数

   观测数据为Y，隐藏数据为Q，完全数据为(Y,Q), 完全数据的对数似然函数是$logP(Y,Q|\lambda)$。

2. E步：对对数似然函数$logP(Y,Q|\lambda)$求在隐变量分布$P(Q|Y,\lambda^t)$下的期望，在这里$\lambda^t$为已知变量
   $$
   Q(\lambda, \lambda^t) = \sum_QlogP(Y,Q|\lambda)P(Q|Y, \lambda^t) \quad \lambda^t为HMM参数的当前估计值 \\
   = \sum_QlogP(Y,Q|\lambda)P(Q,Y| \lambda^t)/P(Y|\lambda^t) \\
   $$
   由于$\lambda^t$为已知量，Y也是已知量，所以分母部分是常数，去掉常数项不影响优化结果: \
   $$
   Q(\lambda, \lambda^t) \propto \sum_QlogP(Y,Q|\lambda)P(Q,Y| \lambda^t) \\
   = E_{Q \sim P(Q,Y|\lambda^t)}logP(Y,Q|\lambda)
   $$
   由公式4得：
   $$
   P(Y,Q|\lambda)= \pi_{q^1}b_{q^1y^1}a_{q^1q^2}b_{q^2y^2}\cdots  a_{q^{T-1}q^T}b_{q^Ty^T}\\
   $$
   于是期望函数可以写成：
   $$
   Q(\lambda,\lambda^t) = E_{Q \sim P(Q,Y|\lambda^t)}log\left(\pi_{q^1}b_{q^1y^1}a_{q^1q^2}b_{q^2y^2}\cdots  a_{q^{T-1}q^T}b_{q^Ty^T}\right) \\
   = E_{Q \sim P(Q,Y|\lambda^t)}log\pi_{q^1} + E_{Q \sim P(Q,Y|\lambda^t)}\left[ \sum_{t=1}^{T-1}loga_{q^tq^{t+1}}\right] + E_{Q \sim P(Q,Y|\lambda^t)}\left[\sum_{t=1}^T log  b_{q^ty^t} \right]
   $$
   

3. M步：极大化Q函数



## 参考

1. https://blog.csdn.net/continueoo/article/details/77893587
2. https://www.cnblogs.com/skyme/p/4651331.html
