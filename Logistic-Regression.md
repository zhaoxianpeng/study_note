---
author: Xianpeng Zhao
Title: Logistic Regression
Date: 2018/8/1
---

# Logistic Regress

[TOC]

## 模型

在线性拟合基础上,我们用logistic实现二分类问题.

### sigmoid函数

我们有$sigmoid$函数如下,
$$
g(z) = \frac{1}{1+e^{-z}}
$$
该函数图像为:

![转自维基百科](http://omw27y2pe.bkt.clouddn.com/image/sigmoid/Logistic-curve.png)

sigmoid函数有如下性质

* 连续,光滑

* 严格单调

* x趋于负无穷时,y趋于0；x趋于正无穷时,y趋于1

* 定义域为$(-\infty, +\infty)$

* 值域为(0,1)

* 处处可导,并且导数为$g'(x) = g(x)(1-g(x))$

  $g(x) = (1+e^{-x})^{-1} $推导如下,根据复合函数求导法则:
  $$
  \begin{align}
  v & = -x \\
  u & = e^v  = e^{-x}\\
  w & = 1+u = 1 + e^{-x}\\
  g & = w^{-1} = (1+e^{-x})^{-1}\\
  & => \\
  g'(x) & = \frac{dg}{dw}\cdot \frac{dw}{du} \cdot \frac{du}{dv} \cdot \frac{dv}{dx} \\
  & =-w^{-2} \cdot 1 \cdot e^v \cdot -1 \\
  & = (1+e^{-x})^{-2} \cdot e^{-x} \\
  & = \frac{e^{-x}}{1+e^{-x}} \cdot \frac{1}{1+e^{-x}} \\
  & = g(x)\cdot(1-g(x))
  \end{align}
  $$



### 伯努力分布

那么,为什么sigmoid可以用到logistic回归问题上呢?

除了数学上更容易处理之外,还有其本身的推导特性; 对于分类问题,尤其是二分类问题,我们都假定是服从伯努力分布；伯努力分布的概率质量函数为(只是对于离散信号而言，我们一般比较熟悉的概率密度函数是对于连续信号。):
$$
P(y=1) = p \\ 
P(y=0)=1-p \\
f(y|p) = p^y\cdot(1-p)^{1-y}
$$
将其表示为指数分布族的一般形式为:
$$
f(y; \eta) = b(y)exp(\eta^TT(y)-a(\eta))
$$
其中，$\eta$是分布的**自然参数**（natural parameter）或**典范参数**（canonical parameter），T(y)叫做**充分统计量**（sufficient statistic），通常情况下T(y)=y；a($\eta$)是**对数分配函数**（log partition function），而a、b与T一般都是给定的，随着η的变化，会得到不同的分布。

伯努力分布可表示为:
$$
\begin{align}
f(y;p) & = exp(ln(p^y\cdot(1-p)^{1-y})) \\
& = exp(yln(p) + (1-y)ln(1-p)) \\
& =exp((ln(p)-ln(1-p)y + ln(1-p))) \\
& = exp(ln(\frac{p}{1-p})y + ln(1-p))
\end{align}
$$


对应标准表达式
$$
b(y)=1, \eta=ln(\frac{p}{1-p}), T(y)=y, a(\eta)=-ln(1-p)
$$
其中,由$\eta=ln(\frac{p}{1-p})$可以推出:
$$
\begin{align}
& \eta = ln(\frac{p}{1-p}) \\
& exp^{\eta} = \frac{p}{1-p} \\
& exp^{\eta} - pexp^{\eta} = p \\
& p = \frac{exp^{\eta}}{1+exp^{\eta}} \\
& p = \frac{1}{1+exp^{-\eta}} = g(\eta)
\end{align}
$$
由于广义线性模型中,自然参数$\eta$与输入特征x之间呈线性关系,即

实数时, $\eta = \theta^Tx$

向量时, $\eta_i=\theta_i^Tx$

因此,有

$f_{\theta}(x) = P(y=1|x;\theta) = p = g(\eta) = g(\theta^Tx) = \frac{1}{1+exp^{-\theta^Tx}}$



### logistic 回归模型

至此,我们要的模型终于浮出水面了,在给定了x和$\theta$时,y=1的概率如下:
$$
P(y=1|x;\theta) = h_{\theta}(x) = \frac{1}{1+exp^{-\theta^Tx}}
$$

## 策略

模型有了,那么下一步就是要采取什么策略来学习模型中的各个参数了.

假设现在有如下数据:

$D={(X_1, Y_1),(X_2,Y_2),\cdots, (X_m,Y_m)}$
$$
\begin{align}
X & = \begin{bmatrix} 1 & x_1^1 & x_2^1 & \cdots & x_n^1 \\ 1& x_1^2 & x_2^2 & \cdots & x_n^2 \\ \vdots  & \vdots & \vdots & \ddots & \vdots \\1&x_1^m & x_2^m & \cdots &x_n^m\end{bmatrix} ,
x^i_j表示第i个样本的第j个feature
\\
  Y &= \begin{bmatrix} y^1 \\ y^2 \\ \vdots \\ y^m\end{bmatrix} ,
   y^i \in \{0,1\} \\
  W&=\begin{bmatrix}w_0 \\ w_1 \\ \vdots \\ w_n \end{bmatrix}
 \end{align}
$$

### 极大似然估计

m个样本,似然函数为
$$
\begin{align}
L(\theta) &= \prod_{i=1}^mP(y^i|x^i;\theta) \\
\end{align}
$$
其中 
$$
P(y|x;\theta) = 
\begin{cases}
h_\theta(x) &\text{ if } y = 1 \\
1-h_\theta(x)  &\text{ if } y =0 \\
\end{cases}
\\ => 
p(y|x;\theta)=h_\theta(x)^y\cdot (1-h_\theta(x))^{1-y} \\
=> L(\theta) = \prod_{i=1}^mh_\theta(x^i)^{y^i}(1-h_\theta(x^i))^{1-y^i}
$$
为了方便计算,取对数似然函数:
$$
l(\theta) =log L(\theta) = \sum_{i=1}^m[y^ilogh_\theta(x^i) + (1-y^i)log(1-h_\theta(x^i))]
$$
然后求$\theta$使$l(\theta)$的值最大,即为极大似然估计；具体求解方法可用梯度下降,拟牛顿法来求最优值.

### 最大熵

#### 熵的定义

熵表示随机变量不确定性的度量. 设X是一个有限个值的离散随机变量,其概率分布为
$$
P(X=x_i) = p_i, i=1,2, \cdots, n
$$
则随机变量X的熵定义为
$$
H(X)=-\sum_{i=1}^np_ilogp_i
$$
熵越大不确定性越大,并且熵的取值范围为
$$
0 \leqslant H(p) \leqslant logn
$$

#### 条件熵

设又随机变量（X，Y），其联合概率分布为
$$
P(X=x_i, Y=y_j) = P_{ij}, i=1,2,\cdots,n; j=1,2,\cdots,m
$$
条件熵H（Y|X）表示在已知随机变量X的条件下随机变量Y的不确定性，定义为X给定条件下Y的条件概率分布的熵对X的数学期望
$$
H(Y|X) = \sum_{i=1}^np_iH(Y|X=x_i) \\
这里，p_i=P(X=x_i),i=1,2,\cdots,n
$$

#### 最大熵原理

学习概率模型时,在所有可能的概率模型中,熵最大的模型是最好的模型.

#### 最大熵模型

假设分类模型是一个条件概率分布P（Y|X）， $X\in \chi \subseteq R^n$ 表示输入，$Y \in y$表示输出，这个模型表示的是对于给定的输入X，以条件概率P（Y|X）输出Y。

对于给定训练数据，可以确定联合概率分布P（X，Y）的经验分布和边缘分布P（X）的经验分布：
$$
\tilde{P}(X=x,Y=y) = \frac{v(X=x, Y=y)}{N} \\
\tilde{P}(X=x) = \frac{v(X=x)}{N} \\
其中，v(X=x,Y=y)表示训练样本中（x，y）出现的频数，v（X=x）表示训练样本中输入x出现的频数，N表示训练样本容量。
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

#### 最大熵模型学习

最大熵模型学习可以形式化为约束最优化问题。

对于给定的训练数据集$T=\{(x_1,y_1), (x_2,y_2), \cdots, (x_N, y_N) \}$以及特征函数$f_i(x,y), i=1,2,\cdots,n$，最大熵模型的学习等价于约束最优化问题：
$$
\begin{align}
\underset{P\in C}{max} \text{ }   & H(P) = -\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x) \\
s.t. \text{    }&E_P(f_i) = E_{\tilde{P}}(f_i), i=1,2,\cdots,n \\
& \sum_yP(y|x) = 1
\end{align}
$$
按照最优化问题的习惯，将求最大值的问题改成等价的球最小值问题：
$$
\begin{align}
\underset{P\in C}{min} \text{ }   & -H(P) = \sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x) \\
s.t. \text{    }&E_P(f_i) - E_{\tilde{P}}(f_i) =0, i=1,2,\cdots,n \\
& 1-  \sum_yP(y|x) = 0
\end{align}
$$
求解上式约束最优化问题所得出的解，就是最大熵模型学习的解。求解过程如下：

将约束最优化的原始问题转化为无约束最优化的对偶问题。

首先引入拉格朗日乘子$w_0, w_1, \cdots,w_n$,定义拉格朗日函数L（P，w）：
$$
\begin{align}
L(P,w) & \equiv-H(P) + w_0(1-\sum_yP(y|x)) + \sum_{i=1}^nw_i(E_P(f_i) - E_{\tilde{P}}(f_i)) \\
& =
\end{align}
$$






### 构造损失函数





## 参考

1. https://blog.csdn.net/lizhe_dashuju/article/details/49864569
2. https://blog.csdn.net/saltriver/article/details/57531963
3. https://blog.csdn.net/baidu_15238925/article/details/81291247
4. https://www.cnblogs.com/wallacup/p/6024855.html?utm_source=itdadao&utm_medium=referral