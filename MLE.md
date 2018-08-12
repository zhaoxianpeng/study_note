# 极大似然估计

## 目的

利用已知的样本，反推最有可能导致这些样本的模型参数值；当然服从的模型也要是已知的，正态分布还是伯努利分布或是其他分布。

## 似然函数

对于这个函数：
$$
P(x|\theta)
$$
有两个输入：$x, \theta$， x表示某个具体的输入数据，$\theta$表示模型参数。

如果$\theta$已知，那么这个函数就是概率函数，它描述的是对于不同的输入x出现的概率。

如果x是已知的，$\theta$为变量，那么我们叫这个函数为似然函数，它描述对于不同的模型参数，出现样本x的概率是多少。 

## 构造似然函数

给定样本数据集：
$$
X=\{x_1,x_2, \cdots, x_N\}, 其中x_i( i=1\cdots N)独立同分布
$$
似然函数为：
$$
L(\theta) = P(X|\theta) = P(x_1,x_2,\cdots, x_N|\theta) = \prod_{i=1}^NP(x_i|\theta) \quad 独立同分布可以写成联乘形式
$$
如果$\theta^*$能使似然函数最大，那么它就是最合适参数：
$$
\theta^* = arg\underset{\theta}{max}L(\theta)
$$

## 求解极大似然函数

为了方面计算，我们通常求解$\theta$使得对数似然函数最大：
$$
\theta^* = arg\underset{\theta}{max}log(L(\theta)) = arg\underset{\theta}{max}\sum_{i=1}^NlogP(x_i|\theta)
$$
求导数或梯度（当$\theta$为向量时）令其等于0解方程组即的最优解。



### 从极大似然估计到数学期望

从上面的极大似然函数公式（7）$\theta^* = arg\underset{\theta}{max}log(L(\theta)) = arg\underset{\theta}{max}\sum_{i=1}^NlogP(x_i|\theta)$，我们可以做如下变形：
$$
\theta^* = arg\underset{\theta}{max}log(L(\theta)) = arg\underset{\theta}{max}\sum_{i=1}^NlogP(x_i|\theta) \\
 =  arg\underset{\theta}{max}\frac{1}{N}\sum_{i=1}^NlogP(x_i|\theta)  \quad 除以常数N结果不变\\
 = arg\underset{\theta}{max}E_{x|\theta}[logP(x|\theta)]
$$
理论上，根据已有的数据，我们可以得到每个x的统计频率（又称经验分布）$\tilde{P}(x)$，那么上式等价位：
$$
\theta^* = arg\underset{\theta}{max}\sum_{x \in X}\tilde{P}(x)logP(x|\theta)
$$
又，若x为连续变量，则把求和换成积分，相应的概率函数变成概率密度函数即可，本质没区别。

### 有监督模型

我们假设输入为X，标签为Y，那么（X，Y）构成一个事件，于是根据公式8有：
$$
\theta^* = arg\underset{\theta}{max}E_{X,Y}[logP(X,Y|\theta)]
$$
对于分类问题，通常用P(Y|X)建模而不是P（X，Y），我们利用公式$P(X, Y) = P(X)P(Y|X)$转化上式：
$$
\begin{align}
\theta^* &= arg\underset{\theta}{max}E_{X,Y}[logP(X,Y|\theta)] \\
                & = arg\underset{\theta}{max}\sum_{x \in X, y \in Y}\tilde{P}(x,y)logP(x,y|\theta) \\
                & = arg\underset{\theta}{max}\sum_{x \in X, y \in Y}\tilde{P}(x,y)log[P(x|\theta)P(y|x,\theta)] \\
                & 由于x表示输入，与\theta没有关系，因此P(x|\theta)可以写为 \tilde{P}(x)，在输入确定的情况，它是常数，可以去掉 \\
                & = arg\underset{\theta}{max}\sum_{x \in X, y \in Y}\tilde{P}(x,y)logP(y|x,\theta) \\
\end{align}
$$
又有$\tilde{P}(x,y) = \tilde{P}(x)\tilde{P}(y|x)$，上式可得：
$$
\theta^* =arg\underset{\theta}{max}\sum_{x \in X, y \in Y}\tilde{P}(x)\tilde{P}(y|x)logP(y|x,\theta) \\ 
= arg\underset{\theta}{max}\sum_{x \in X}\tilde{P}(x)\left(\sum_{y \in Y}\tilde{P}(y|x) logP(y|x,\theta)\right) \\
$$
训练数据中对于给定输入x只有一个特定目标标签的$y_t$，那么这里括号里面的式子只有$\tilde{P}(y=y_t|x)$时为1，其他情况为0：
$$
\theta^*= arg\underset{\theta}{max}\sum_{x \in X}\tilde{P}(x)\left(logP(y_t|x,\theta)\right) \\
 = arg\underset{\theta}{max}E_X[logP(Y_t|X,\theta)]
$$

### 



