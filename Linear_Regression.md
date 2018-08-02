---
title: 线性回归
date: 2018/8/1 11:05
author: Xianpeng Zhao
---

# 线性回归

[TOC]

## 模型

* 有n维特征
  $$
  h_w(x) = w_0 + w_1x_1 + \cdots + w_nx_n
  $$

* 向量表示


$$
W = \begin{bmatrix}w_0\\w_1\\{\vdots}\\w_n\end{bmatrix} X = \begin{bmatrix}1\\x_1\\{\vdots}\\x_n\end{bmatrix}
$$

$$
h_{w}(X)=\sum_{i=0}^{n}w_ix_i = W^TX
$$

  

* 训练数据

  $D={(X_1, Y_1),(X_2,Y_2),\cdots, (X_m,Y_m)}$


$$
  X = \begin{bmatrix} 1 & x_1^1 & x_1^2 & \cdots & x_1^n \\ 1& x_2^1 & x_2^2 & \cdots & x_2^n \\ \vdots  & \vdots & \vdots & \ddots & \vdots \\1&x_m^1 & x_m^2 & \cdots &x_m^n\end{bmatrix}
  Y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_m\end{bmatrix}
  W=\begin{bmatrix}w_0 \\ w_1 \\ \vdots \\ w_n \end{bmatrix}
$$
  预测数据为

  $\widehat{Y} = XW$

* 损失度量,平方损失函数

  $Loss(X_i) = (\widehat{Y_i}-Y_i)^2 = X_iW $

  整个训练集的损失为

  $J(W) = \sum_{i=1}^{m}Loss(X_i) = \sum_{i=1}^{m}(\widehat{Y_i}-Y_i)^2=\sum_{i=1}^m(X_iW-Y_i)^2$

  目标为求得一组$W$使$h_w(X_i)$ 与$Y_i$接近,即损失越小越好

* 数学表述

  $W^* = \underset{W}{argmin}J(W)$

  

## 优化

###　矩阵求导法

由于$XW-Y$为列向量,那么$\sum_{i=0}^m(X_iW-Y_i)^2$可以由$(XW-Y)^T(XW-Y)$求得,那么此时损失函数可以写为:
$$
\begin{align}
J(W) & = (XW-Y)^T(XW-Y) \\
& = W^TX^TXW -W^TX^TY-Y^TXW+Y^TY \\
& = W^TX^TXW -2W^TX^TY+Y^TY
\end{align}
$$
根据矩阵求导公式,$x​$为列向量
$$
\frac{d(u^Tv)}{dx}=\frac{d(u^t)}{dx}\cdot v + \frac{d(v^T)}{dx}\cdot u
$$
重要推论
$$
\frac{d(x^Tx)}{dx} = \frac{d(x^T)}{dx}\cdot x + \frac{d(x^T)}{dx}\cdot x = 2x \\
\frac{d(x^TAx)}{dx} = \frac{d(x^T)}{dx}\cdot Ax + \frac{d(Ax)^T}{dx}\cdot x = Ax+A^Tx
$$
其他常用公式
$$
\begin{align}
\frac{dx^T}{dx}& = I \\
\frac{d(Ax)^T}{dx} & = A^T \\
\frac{dx}{dx^T}& = I \\
\frac{d(Ax)}{dx^T}& = A
\end{align}
$$
$\frac{\partial J(W)}{\partial W} = X^TXW+(X^TX)^TW - 2X^TY $

因为$(X^TX)^T = X^TX$,所以,以上式子可化为 $\frac{\partial J(W)}{\partial W} = 2X^TXW- 2X^TY $

令其等于0,得
$$
X^TXW = X^TY \\
=> 
W = (X^TX)^{-1}X^TY
$$
前提是$X^TX$是可逆的.然而,好像有时伪逆也可.(为什么?附录)

### 梯度下降

#### 代数方法

* 损失函数可以写为

$$
\begin{align}
J(W) & = \sum_{i=1}^m(X_iW-Y_i)^2 \\
 & = \sum_{i=1}^{m}(\sum_{j=0}^nw^jx_i^j - y^i)^2
\end{align}
$$

​	对$w$求导得
$$
\frac{\partial J}{\partial w^j} = \sum_{i=1}^m(h_w(x_i)-y_i)x^j_i
$$
​	更新$w$,对 $j = 0, 1,2, \cdots n$:
$$
w^j=w^j-\alpha \frac{\partial J}{\partial w^j} = w^j - \alpha \sum_{i=1}^{m}(h_w(x_i)-y_i)x_i^j
$$

#### 向量方法计算梯度

* 残差向量 residual,维度为$m*1$, X维度为$m*n$, Y维度为$m*1$.

  $residual = \widehat{Y} - Y = XW-Y$

  根据矩阵乘法法则,以上式子可写为:
  $$
  \begin{align}
  \frac{\partial J}{\partial w_j}& = \sum_{i=1}^m(h_w(x_i)-y_i)x_{ij} \\
  & = residual^T \times x_j  \leftrightarrow 其中x_j为列向量 x_{1j},x_{2j}, \cdots, x_{mj}, 结果为标量 \\
  \end{align}
  $$
  所有$w$的梯度可以表示为$\bigtriangledown _WJ(W)$维度为(n*1)
  $$
  \bigtriangledown _W J(W) = (residual^T \times X)^T = ((XW-Y)^T X)^T
  $$
  W的更新规则为
  $$
  W=W - \alpha \bigtriangledown _WJ(W) = W - \alpha ((XW-Y)^T X)^T
  $$
  

### Normal Equation VS Gradient Descent

Normal Equation 跟 Gradient Descent（梯度下降）一样，可以用来求权重向量θ。但它与Gradient Descent相比，既有优势也有劣势。

优势：

Normal Equation可以不在意x特征的scale。比如，有特征向量X={x1, x2}, 其中x1的range为1 ~ 2000，而x2的range为1~ 4，可以看到它们的范围相差了500倍。如果使用Gradient Descent方法的话，会导致椭圆变得很窄很长，而出现梯度下降困难，甚至无法下降梯度（因为导数乘上步长后可能会冲出椭圆的外面）。但是，如果用Normal Equation方法的话，就不用担心这个问题了。因为它是纯粹的矩阵算法。

劣势：

相比于Gradient Descent，Normal Equation需要大量的矩阵运算，特别是求矩阵的逆。在矩阵很大的情况下，会大大增加计算复杂性以及对计算机内存容量的要求。



---

## 从概率层面解释模型的目标函数

​    基本上每个模型都会有一个对应的目标函数，可以通过不同的最优化求解方法（梯度下降，牛顿法等等）对这些对应的目标函数进行求解。线性回归模型，我们知道实际上是通过多个自变量对自变量进行曲线拟合。我们希望找到一条可以较好拟合的曲线，

​    那我们如何判断一条曲线的拟合程度的好坏。上面讲到，我们采用的是最小二乘法（预测值和真实值得误差的平方和），那为什么要用这个作为目标函数呢？

​         可以从中心极限定理、高斯分布来分析：

### 中心极限定理

**当样本量N逐渐趋于无穷大时，N个抽样样本的均值的频数逐渐趋于正态分布**，其对原总体的分布不做任何要求，意味着无论总体是什么分布，其抽样样本的均值的频数的分布都随着抽样数的增多而趋于正态分布。

事实上，如果初始条件相同，随着误差的逐渐叠加，最终将接近正态分布。

### 高斯分布

假的给定一个输入样本$x$，我们得到预测值$\widehat{y}$和真实值$y$间的存在的误差$\epsilon$，那么他们的关系如下：
$$
y = \widehat{y}+\epsilon
$$
其中$\epsilon$服从正态分布
$$
\epsilon \sim N(0, \sigma^2)
$$
因此有
$$
(y-\widehat{y}) \sim N(0, \sigma^2) \\
=> y \sim N(\widehat{y}, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}}exp(-\frac{(y-\widehat{y})^2}{2\sigma^2})
$$
其中
$$
\widehat{y} = w^Tx \\
=> y \sim N(w^Tx, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}}exp(-\frac{(y-w^Tx)^2}{2\sigma^2})
$$
要求w的极大似然估计，即是说，我们现在得到的真实存在的$y$在w取什么值时出现的概率最大，我们先来看这个概率：
$$
L(w) = P(y|x; w) = \prod_{i=1}^m\frac{1}{\sqrt{2\pi \sigma^2}}exp(-\frac{(y_i-\widehat{y_i})^2}{2\sigma^2})
$$
为了简化计算，取对数似然函数
$$
\begin{align}
l(w) & = logL(w) \\
& = \sum_{i=1}^mlog(\frac{1}{\sqrt{2\pi \sigma^2}}exp(-\frac{(y_i-\widehat{y_i})^2}{2\sigma^2})) \\
& = \sum_{i=1}^mlog(\frac{1}{\sqrt{2\pi}}) - \sum_{i=1}^mlog\sigma - \sum_{i=1}^{m}\frac{(y_i-\widehat{y_i})^2}{2\sigma^2}
\end{align}
$$
要让L(w)最大，即需要让$l(w)$最大，即让$\sum_{i=1}^{m}(y_i-\widehat{y_i})^2$最小。

综上，当误差函数定位平方时，参数w是样本的极大似然估计。



## 一些问题

1. 为什么在求$W$时$W = (X^TX)^{-1}X^TY$里面的$X^TX$不可逆也可以呢?

2. 梯度下降为什么下降最快的方向是负梯度方向?

3. 中心极限定理与大数定律的区别

   **大数定律**是说，n只要越来越大，我把这n个独立同分布的数加起来去除以n得到的这个样本均值（也是一个随机变量）会依概率收敛到真值u，但是样本均值的分布是怎样的我们不知道。

   **中心极限定理**是说，n只要越来越大，这n个数的样本均值会趋近于正态分布，并且这个正态分布以u为均值，sigma^2/n为方差。

   直观上来讲，想到大数定律的时候，你脑海里浮现的应该是一个样本，而想到中心极限定理的时候脑海里应该浮现出很多个样本。



## 参考

1. https://www.cnblogs.com/GuoJiaSheng/p/3928160.html
2. http://www.fuzihao.org/blog/2014/06/13/为什么最小二乘法对误差的估计要用平方/
3. 程序员的数学-概率统计
4. https://blog.csdn.net/huruzun/article/details/41493063
5. https://blog.csdn.net/lizhe_dashuju/article/details/49864569