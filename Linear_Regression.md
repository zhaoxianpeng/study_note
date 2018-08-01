---
title: 线性回归
date: 2018/8/1 11:05
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

* 残查向量 residual,维度为$m*1$, X维度为$m*n$, Y维度为$m*1$.

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
  

---

## 从概率层面解释模型的目标函数

​    基本上每个模型都会有一个对应的目标函数，可以通过不同的最优化求解方法（梯度下降，牛顿法等等）对这些对应的目标函数进行求解。线性回归模型，我们知道实际上是通过多个自变量对自变量进行曲线拟合。我们希望找到一条可以较好拟合的曲线，

​    那我们如何判断一条曲线的拟合程度的好坏。上面讲到，我们采用的是最小二乘法（预测值和真实值得误差的平方和），那为什么要用这个作为目标函数呢？

​         可以从中心极限定理、高斯分布来分析：

### 中心极限定理

**当样本量N逐渐趋于无穷大时，N个抽样样本的均值的频数逐渐趋于正态分布**，其对原总体的分布不做任何要求，意味着无论总体是什么分布，其抽样样本的均值的频数的分布都随着抽样数的增多而趋于正态分布



## 一些问题

1. 为什么在求$W$时$W = (X^TX)^{-1}X^TY$里面的$X^TX$不可逆也可以呢?

2. 梯度下降为什么下降最快的方向是负梯度方向?

3. 中心极限定理与大数定律的区别

   **大数定律**是说，n只要越来越大，我把这n个独立同分布的数加起来去除以n得到的这个样本均值（也是一个随机变量）会依概率收敛到真值u，但是样本均值的分布是怎样的我们不知道。

   **中心极限定理**是说，n只要越来越大，这n个数的样本均值会趋近于正态分布，并且这个正态分布以u为均值，sigma^2/n为方差。

   直观上来讲，想到大数定律的时候，你脑海里浮现的应该是一个样本，而想到中心极限定理的时候脑海里应该浮现出很多个样本。



## 参考

1. https://www.cnblogs.com/GuoJiaSheng/p/3928160.html