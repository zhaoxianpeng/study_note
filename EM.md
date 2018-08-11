# EM

[TOC]

## 极大似然

### 贝叶斯

贝叶斯公式表示如下：
$$
P(A|B) = \frac{P(A,B)}{P(B)} = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|C)P(C) \cdots}
$$
我们可以这么理解，把事件B看做结果，事件A，C，……导致这个结果可能的原因；那么贝叶斯公式可以看成已知结果B，求在众多可能导致这个结果的原因中是原因A导致的概率。

其中，P（原因）称为先验概率，P（原因|结果）称为后验概率；

有点绕口，我们可以写成如下形式：
$$
P(原因A|结果B) = \frac{P(结果B，原因A)}{P(结果B)} \\
=\frac{P(结果B|原因A)P（原因A）}{P(结果B|原因A)P（原因A）+ P(结果B|原其他原因)P（其他原因）}
$$

### 极大似然估计

#### 目的

利用已知的样本，反推最有可能导致这些样本的模型参数值；当然服从的模型也要是已知的，正态分布还是伯努利分布或是其他分布。

#### 似然函数

对于这个函数：
$$
P(x|\theta)
$$
有两个输入：$x, \theta$， x表示某个具体的输入数据，$\theta$表示模型参数。

如果$\theta$已知，那么这个函数就是概率函数，它描述的是对于不同的输入x出现的概率。

如果x是已知的，$\theta$为变量，那么我们叫这个函数为似然函数，它描述对于不同的模型参数，出现样本x的概率是多少。 

#### 构造似然函数

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

#### 求解极大似然函数

为了方面计算，我们通常求解$\theta$使得对数似然函数最大：
$$
\theta^* = arg\underset{\theta}{max}log(L(\theta)) = arg\underset{\theta}{max}\sum_{i=1}^NlogP(x_i|\theta)
$$
求导数或梯度（当$\theta$为向量时）令其等于0解方程组即的最优解。



## 交叉熵

## Jensen 不等式

## EM 推导

### 从极大似然估计到数学期望

从上面的极大似然函数公式（7）$\theta^* = arg\underset{\theta}{max}log(L(\theta)) = arg\underset{\theta}{max}\sum_{i=1}^NlogP(x_i|\theta)$，我们可以做如下变形
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

### 三硬币模型

### EM算法的思想

简版：猜（E-step）,反思（M-step）,重复； 

啰嗦版：

你知道一些东西（观察的到的数据）， 你不知道一些东西（观察不到的），你很好奇，想知道点那些不了解的东西。怎么办呢，你就根据一些假设（parameter）先猜（E-step），把那些不知道的东西都猜出来，假装你全都知道了; 然后有了这些猜出来的数据，你反思一下，更新一下你的假设（parameter）, 让你观察到的数据更加可能(Maximize likelihood; M-stemp); 然后再猜，在反思，最后，你就得到了一个可以解释整个数据的假设了。

1. 注意，你猜的时候，要尽可能的猜遍所有情况，然后求期望（Expected）；就是你不能仅仅猜一个个例，而是要猜出来整个宇宙；
2. 为什么要猜，因为反思的时候，知道全部的东西比较好。（就是P(X,Z)要比P(X)好优化一些。Z是hidden states）
3. 最后你得到什么了？你得到了一个可以解释数据的假设，可能有好多假设都能解释数据，可能别的假设更好。不过没关系，有总比没有强，知足吧。（你陷入到local minimum了）

 ###　EM算法导出

我们经常会从样本观察数据中找出样本的模型参数，常用的方法是极大化模型分布的对数似然函数。

但是在一些情况下，模型中含有隐藏变量不能被观测到时，就无法用上述方法进行求最优解。此时需要用EM算法来迭代寻找最优解。

下面通过近似求解观测数据的对数似然函数的极大化问题来导出ＥＭ算法。

我们已知如下数据：观测变量数据$Y = (y^1, y^2, \cdots, y^N)$，隐变量数据$Z=(z^1,z^2, \cdots, z^N)$，即每个样本属于哪个分布是未知的，这里$z^i=j$表示为第i个样本的隐变量对应的类别为类别j，也就是这里$z^i$不是标量，维度为隐变量类别的个数，意味这对于样本i属于各个类别的概率。（例如：在100男100女身高的混合高斯型中，隐变量的类别个数为2,男or女）。

求模型参数$\theta$

首先极大化观测数据（不完全数据）Ｙ关于参数$\theta$的对数似然函数，即极大化：
$$
\begin{align}
L(\theta) &= logP(Y|\theta) \\
	       & = \sum_{i=1}^NlogP(y^i|\theta) \\
	       & = \sum_{i=1}^Nlog\sum_{z^i}^{|z^i|}P(y^i, z^i|\theta) \quad 即，对某一个样本i，所有可能的类别集合z^i累加  \\
	       & = \sum_{i=1}^Nlog\sum_{z^i}^{|z^i|}P(y^i| z^i, \theta)P(z^i|\theta) \\
\end{align}
$$
极大化上式可能不容易，因为里面的未知变量除了$\theta$之外还有z，无法直接求解。

我们针对上式继续推导一下：
$$
\begin{align}
L( \theta) & =  \sum_{i=1}^Nlog\sum_{z^i}^{|z^i|}P(y^i| z^i, \theta)P(z^i|\theta) \\
	        & 我们设有Q_i(z^i)是第i个样本关于隐变量z^i的概率分布，显然有\sum_{z^i}Q_i(z^i) = 1 \\
                 & =  \sum_{i=1}^Nlog\sum_{z^i}^{|z^i|}Q_i(z^i)\frac{P(y^i| z^i, \theta)P(z^i|\theta)}{Q_i(z^i)} \\
                 & = \sum_{i=1}^Nlog  E_{z^i \sim Q_i(z^i)}[\frac{P(y^i| z^i, \theta)P(z^i|\theta)}{Q_i(z^i)}] \\
                 & \geq \sum_{i=1}^N  E_{z^i \sim Q_i(z^i)}log[\frac{P(y^i| z^i, \theta)P(z^i|\theta)}{Q_i(z^i)}] \quad log函数是凹函数，根据Jensen不等式得此 \\
                 & =  \sum_{i=1}^N\sum_{z^i}^{|z^i|}Q_i(z^i)log\frac{P(y^i| z^i, \theta)P(z^i|\theta)}{Q_i(z^i)} \\
\end{align} \\
$$
上式给出了$L(\theta)$的下界。

 优化方法：

1. 给定初始值 $\theta^0$

2. E步(Expectation):固定$\theta^t$, 调整Q（Z），也就是选择一种隐含变量Z的分布使得对数似然函数的下界上升，其实看公式就知道调整Z的概率分布函数使得对数似然函数关于Z的期望最大？！

   Jensen不等式等号成立时，下界达到最大，而等号成立的条件是对于的函数是常数值，也就是
   $$
   \frac{P(y^i| z^i, \theta)P(z^i|\theta)}{Q_i(z^i)} =c \\
   \sum_{z^i}Q_i(z^i) = 1 \\
   得到：\\
   Q_i(z^i) = \frac{P(y^i|z^i,\theta)P(z^i|\theta)}{\sum_{z^i}P(y^i|z^i,\theta)P(z^i|\theta)} \\
    = \frac{P(y^i,z^i|\theta)}{P(y^i, | \theta)} = P(z^i|y^i, \theta)
   $$

3. M步（Maximization):固定Q（Z），也就是固定隐含变量的概率分布，调整$\theta$， 使得下界达到最大值；
   $$
   \theta^* = arg\underset{\theta}{max}\sum_{i=1}^N\sum_{z^i}^{|z^i|}Q_i(z^i)log\frac{P(y^i| z^i, \theta)P(z^i|\theta)}{Q_i(z^i)} \\
   =  arg\underset{\theta}{max}\sum_{i=1}^N\sum_{z^i}^{|z^i|}Q_i(z^i)logP(y^i, z^i| \theta) - \underbrace{\sum_{i=1}^N\sum_{z^i}^{|z^i|}Q_i(z^i)log{Q_i(z^i)}}_{常量} \\
   $$
   更新$\theta = \theta^*$。

4. 重复2,3直至达到退出条件



### 证明EM算法收敛

 

 

 









## 问题

1. 为什么极大化下界就可以逼近最优值？

2. EM算法到底是已知什么求什么？

   已知联合分布$P(Y,Z|\theta)$，条件分布$P(Z|Y, \theta)$；求模型参数$\theta$；

   


## 参考

1. https://blog.csdn.net/zengxiantao1994/article/details/72787849
2. https://blog.csdn.net/u011508640/article/details/72815981
3. https://www.cnblogs.com/pinard/p/6912636.html
4. https://www.zhihu.com/question/27976634
5. https://wenku.baidu.com/view/3396bb4d6294dd88d0d26bee.html



