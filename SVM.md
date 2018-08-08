# SVM

## 模型

### 目标

有如下数据$D=\{x_i,y_i\}, i=1,2,\cdots,N; 其中 x\in R^{n}, y\in \{-1,1\}$，此处$x_i$为n维列向量；

对于给定training data（线性可分），分类模型$f(x)=w^Tx+b, (w \in R^n)$使得所有分类都正确，并且使得分类间隔最大化。此处

### 间隔最大化

某个点到分类平面的距离为：

$$d = \frac{\left|w^Tx+b\right|}{\left || w\right ||}$$

而对于所有的正负样本$(x_i, y_i)， i = 1,2,\cdots, N$，可以通过等比放缩w和b使得：
$$
w^Tx^++b >= +1 \\
w^Tx^-+b <=-1
$$
而由于对于正负样本y的取值分别为+1和-1，因此上式可统一写为如下形式：
$$
y_i(w^Tx_i + b) >= 1
$$
而对于分类间隔边缘的正负样本点$x_i^+$和$x_j^-$，有
$$
w^Tx_i+b = +1 \\
w^Tx_j+b =-1
$$
这时，分类间隔的值为，位于分类间隔的正负样本到分类平面的距离之和，即：
$$
\begin{align}
d &= d_i + d_j \\
& = \frac{\left|w^Tx_i + b\right|}{\left ||w \right||} + \frac{\left|w^Tx_j + b\right|}{\left||w \right||} \\
& = \frac{\left| +1 \right|}{\left ||w \right||} + \frac{\left|-1\right|}{\left ||w \right||} \\
& = \frac{2}{\left ||w \right||}
\end{align}
$$
此时，该问题可以表示成有约束的最优化问题，即
$$
\underset{w,b}{max} \frac{2}{\left||w\right||}\\
s.t. y_i(w^Tx_i+b) >= 1, i=1,2,\cdots, N
$$
优化上述问题，为了计算方便，通常我们写成等价的如下形式：
$$
\underset{w,b}{min} \frac{1}{2}\left || w\right ||^2 \\
s.t. 1-y_i(w^Tx_i+b) <= 0, i=1,2,\cdots, N
$$
求得最优解 $w^*, b^*​$, 则分类超平面为$w^*x+b=0​$，分类决策函数为$f(x) = sign(w^*x+b)​$。

## 优化

### 拉格朗日乘子法

拉格朗日乘子法是一种寻找多元函数在一组约束下的极值的方法。通过引入拉格朗日橙子，可将有d个变量与k个约束条件的最优化问题转化为具体d+k个变量的无约束最优化问题求解。

#### 等式约束

原始问题如下：
$$
\underset{x \in R^d}{min}f(x) \\
s.t. g_i(x) = 0, i=1,2,\cdots,k
$$
从几何角度看，该问题的目标是在由方程$g_i(x) = 0$确定的d-1维曲面上寻找能使目标函数f(x)最小化的点，此时不难得到如下结论：

* 对于约束曲面上的任意点x， 该点的梯度$\triangledown_xg_i(x)$正交于约束曲面；

  见参考2，函数z=f(x,y)在点p(x,y)的梯度的方向与过点p的等高线f(x,y)=c在这点的法线一个方向相同。梯度的方向与等高线切线方向垂直

* 在最优点$x^*$，目标函数在该点的梯度$\triangledown_xf(x^*)$正交于约束平面。

  可通过反证法证明：若梯度$\triangledown_xf(x^*)$与约束曲面不正交，则仍可在约束曲面上移动该点是f(x)的函数值进一步下降。

由此可知，在最优点$x^*$处，梯度$\triangledown_xg_i(x^*)和\triangledown_xf(x^*))$的方向必相同或相反，即存在$\lambda_i \neq 0$使得：
$$
\lambda_i\triangledown_xg_i(x^*) + \triangledown_xf(x^*) = 0
$$
$\lambda_i$称为拉格朗日乘子（对于等式约束 $\lambda$可能为正也可能为负）。

定义拉格朗日函数如下：
$$
L(x, \lambda) = f(x) + \sum_{i=1}^k \lambda_i g_i(x)
$$
不难发现：
$$
\frac{\part{L(x,\lambda)}}{\part x}=0，\\
 即：\triangledown_xf(x)+\sum_{i=1}^k\lambda_i\triangledown_xg_i(x) = 0 \\
 \frac{\part{L(x,\lambda)}}{\part \lambda}=0, \\
 即约束条件：g(x) = 0 \\
 注意，这里的x\in R^d，\lambda \in R^k都是向量而不是标量，\\
 也就是这里的0也不是标量，而是指0向量，即所有元素为0
$$
于是，原始约束最优化问题可转化为对拉格朗日函数$L(x,\lambda)​$的无约束最优化问题。

#### 不等式约束

现在考虑不等式约束，暂时只考虑一个不等式约束的情况。

假设有如下问题：
$$
\underset{x \in R^d}{min} f(x) \\
s.t. g(x) <= 0
$$
不等式约束的时候，类似等式约束，从几何角度看，该问题的目标是在由方程$g(x) <= 0$确定的d维空间的一部分（半空间？）上寻找能使目标函数f(x)最小化的点。最优点$x*$所在位置有两种可能:

1. 在边界g(x)=0的曲面上; 

   这种情况类似于等式约束,但是有一点不一样,此时 $\triangledown_xg_i(x^*)和\triangledown_xf(x^*))$的方向必相反,即存在常数$\lambda>0$使得$\lambda_i\triangledown_xg_i(x^*) + \triangledown_xf(x^*) = 0$. 

   至于为什么? 参考下图左边,函数f(x)在最优解$x^*$附近变化的趋势是在可行解区域内侧较大外侧较小,而与之对应的是函数g(x)在可行解区域内侧小于0,区域外侧大于0,所以在最优解$x^*$附近的变化趋势是内部较小外部较大,意味着两者梯度方向相反,因此可推断出$\lambda>0$.

2. 在g(x)<0的地方.

   这种情况的时候约束不起作用,等同于无约束的f(x)的最优解,可以直接通过$\triangledown_xf(x)=0$来获得最优点,这等价于将$\lambda$置0然后对$\triangledown_xL(x,\lambda)$置0得到最优点.

![v2-64dbe8af3d4e47e2586e883a1128b80f_hd](/home/xianpeng/workspace/study_note/v2-64dbe8af3d4e47e2586e883a1128b80f_hd.jpg)

整合两种情况,必满足$\lambda g(x) = 0$,因此在约束g(x)<=0的约束下最小化f(x)可转化为如下约束最小化拉格朗日函数:
$$
\underset{x \in R^d}{min}f(x) \\
s.t. g(x) <=0 \\
\lambda >= 0 \\
\lambda g(x) = 0
$$
上式称KKT条件.

#### 推广

推广到多个约束: m个等式约束和n个不等式约束,且可行域$D\subset R^d$非空的优化问题:
$$
\underset{x \in R^d}{min}f(x) \\
s.t. \\
h_i(x) =0, (i=1,2,\cdots,m) \\
g_j(x) <=0, (j=1,2,\cdots, n)
$$
上式称为主问题.

引入拉格朗日乘子$\lambda = (\lambda_1, \lambda_2, \cdots, \lambda_m)^T$和$\mu = (\mu_1, \mu_2, \cdots, \mu_n)^T$,相应的拉格朗日函数为:
$$
L(x, \lambda, \mu) = f(x) + \sum_{i=1}^m\lambda_ih_i(x) + \sum_{j=1}^m\mu_jg_j(x) \\
\begin{align}
& s.t. \\
& g_j(x) <=0, j=(1,2,\cdots,n) \\
& \mu_j >= 0, j=(1,2,\cdots,n)\\
& \mu_j g_j(x) = 0,j=(1,2,\cdots,n) \\
\end{align}
$$
以上为主问题的拉格朗日乘子函数,其拉格朗日对偶函数为:
$$
\Gamma(\lambda, \mu) = \underset{x \in D}{inf}L(x, \lambda, \mu) \\
 = \underset{x\in D}{inf}\left( f(x) + \sum_{i=1}^m\lambda_ih_i(x) + \sum_{j=1}^m\mu_jg_j(x) \right) \\
 这里inf的意思为下确界，每太理解，可想象成最小值min符号，但是又不是（数学上不严谨）
$$
此函数通常通过将拉格朗日乘子$L(x, \lambda, \mu)$对x求偏导并令其等于0来获得用$\lambda和\mu$来表示x的表达式,以此来获得对偶函数的表达形式.

若$\tilde{x} \in D$为主问题可行域中任意一点,则对于任意$\mu >= 0$和$\lambda$都有:
$$
\sum_{i=1}^m\lambda_ih_i(x) + \sum_{j=1}^m\mu_jg_j(x)  <=0
$$
进而有
$$
\Gamma(\lambda, \mu) = \underset{x \in D}{inf}L(x, \lambda, \mu) <= L(\tilde{x}, \lambda, \mu) <= f(\tilde{x})
$$
若主问题的最有值为$p^*$,则对于任意$\mu >= 0$和$\lambda$都有:
$$
\Gamma(\lambda, \mu) <= p^*
$$
即对偶函数给出了主问题最有值的下界.

显然,这个下界取决于$\mu$和$\lambda$的值,于是一个很自然的问题是:基于对偶函数能获得的最好下界是什么?这就引出了优化问题: 
$$
\underset{\lambda, \mu>=0}{max}{\Gamma(\lambda,\mu)}
$$
该式就是主问题的对偶问题,其中$\lambda$和$\mu$称为对偶变量,无论主问题的凸性如何,对偶问题始终是凸优化问题(问题1).

若对偶问题的最优值为$d^*$,显然有$d^* <= p^*$,这称为弱对偶性;若$d^* = p^*$称为强对偶性,此时由对偶问题能获得主问题的最优下界.

那么强对偶性什么时候成立呢?当如下条件满足时强对偶性成立:

1. 主问题为凸优化问题,即f(x)和$g_j(x)$均为凸函数,$h_i(x)$为仿射函数;
2. 可行域中至少有一点使不等式约束严格成立,即存在x对所有j有$g_j(x)<0$都成立.

在强对偶性成立时,将拉格朗日函数分别对原变量和对偶变量求导,再令导数等于0,即可得到原变量与对偶变量的数值关系.于是对偶问题解决了,主问题也就解决了.

##### 定理

对主问题和对偶问题，假设满足上述强对偶性条件；则$x^*和\lambda^*,\mu^*$分别是主问题和对偶问题的解的充分必要条件是$x^*, \lambda^*,\mu^*$满足下面的KKT条件
$$
\begin{align}
& \triangledown_xL(x^*, \lambda^*,\mu^*) = 0 \\
& \triangledown_\lambda L(x^*, \lambda^*,\mu^*) = 0 \\
& \triangledown_\mu L(x^*, \lambda^*,\mu^*) = 0 \\
& g_j(x) <=0, j=(1,2,\cdots,n) \\
& \mu_j >= 0, j=(1,2,\cdots,n)\\
& \mu_j g_j(x) = 0,j=(1,2,\cdots,n) \\
& h_i(x) =0, (i=1,2,\cdots,m) \\
\end{align}
$$

#### 优化方法

##### 二次规划(Quadratic Programming, QP)

##### 半正定规划(Semi-Definite Programming, SDP)



### SVM问题优化

好了，我们的目的是优化SVM分类时的最大间隔，回到我们的优化目标：
$$
\underset{w,b}{min} \frac{1}{2}\left || w\right ||^2 \\
s.t. \\
1-y_i(w^Tx_i+b) <= 0, i=1,2,\cdots, N
$$
构造拉格朗日函数$\lambda=(\lambda_1,\cdots,\lambda_N)^T$：
$$
L(w,b,\lambda) = \frac{1}{2}\left|\left|w\right|\right|^2 + \sum_{i=1}^N\lambda_i(1-y_i(w^Tx_i+b)) \\
s.t. \\
1-y_i(w^Tx_i+b) <= 0, i=1,2,\cdots, N \\
\lambda_i >= 0, i=1,2,\cdots, N \\
\lambda_i\left(1-y_i(w^Tx_i+b)\right)= 0, i=1,2,\cdots, N
$$


原始问题的对偶问题是极大极小问题：
$$
\underset{\lambda}{max} \underset{w,b}{min}L(w,b,\lambda) \\
严格来说应该是这样：
\underset{\lambda}{max} \underset{w,b}{inf}L(w,b,\lambda) \\
至于为什么可以写成min不确定，猜测可能是强对偶问题所以等价？问题部分。
$$
为了得到对偶问题的解，需要先求$L(w,b,\lambda)$对w,b的极小，再求对$\lambda$的极大。

1. 求$\underset{w,b}{min}L(w,b,\lambda)$

   将拉格朗日函数$L(w,b,\lambda)$分别对w和b求偏导并令其等于0：
   $$
   \triangledown_wL(w,b,\lambda) = w -\sum_{i=1}^N\lambda_iy_ix_i = 0 \\
   \triangledown_bL(w,b, \lambda) = -\sum_{i=1}^N\lambda_iy_i = 0
   $$
   用到如下求导公式
   $$
   \frac{dx^T}{dx} = I \\
   \frac{d(u^Tv)}{dx} = \frac{d(u^T)}{dx}\cdot v + \frac{d(v^T)}{dx}\cdot u \\
   \frac{d\left|\left|x\right|\right|^2}{dx} = \frac{d(x^Tx)}{dx} = \frac{dx^T}{dx}\cdot x +  \frac{dx^T}{dx}\cdot x=2x \\
   \frac{d(x^TA)}{dx} = \frac{d(x^T)}{dx}\cdot A + \frac{dA}{dx}\cdot x = A  , (A是与x维度相同的列向量,\frac{dA}{dx}=0)
   $$
   得：
   $$
   w^* = \sum_{i=1}^N\lambda_iy_ix_i \\
   \sum_{i=1}^N\lambda_iy_i = 0
   $$
   将w和b带入拉格朗日函数得：
   $$
   \begin{align}
   L(w,b,\lambda) & =\frac{1}{2}\left|\left|w\right|\right|^2 + \sum_{i=1}^N\lambda_i(1-y_i(w^Tx_i+b)) \\
   &= \frac{1}{2}\left(\sum_{i=1}^N\lambda_iy_ix_i\right)^T\left(\sum_{i=1}^N\lambda_iy_ix_i\right) + \sum_{i=1}^N\lambda_i(1-y_i(\left(\sum_{j=1}^N\lambda_jy_jx_j\right)^Tx_i + b)) \\
   & x_i为列向量，\lambda_i y_i都为标量，x_i^Tx_j 为标量等价于x_i \cdot x_j \\
   & = \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_j(x_i \cdot x_j) - \sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_j(x_i \cdot x_j) - \sum_{i=1}^N\lambda_iy_ib + \sum_{i=1}^N\lambda_i \\
   & = \sum_{i=1}^N\lambda_i - \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_j(x_i \cdot x_j) \\
   即：\\
   \underset{w,b}{min}L(w,b,\lambda) & =  \sum_{i=1}^N\lambda_i - \frac{1}
   {2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_j(x_i \cdot x_j) \\
   \end{align} \\
   $$

2. 求对偶函数$\underset{w,b}{min}L(w,b,\lambda)$对$\lambda$的极大：
   $$
   \underset{\lambda}{max}\underset{w,b}{min}L(w,b,\lambda)\\
   即：\\
   \underset{\lambda}{max} \sum_{i=1}^N\lambda_i - \frac{1}
   {2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_j(x_i \cdot x_j) \\ 
   s.t. \\
   \sum_{i=1}^N\lambda_iy_i = 0 \\
   \lambda_i >=0, i=1,2,\cdots,N
   $$
   将极大换成极小，得到下面等价的对偶最优化问题：
   $$
   \underset{\lambda}{min} \frac{1}
   {2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_j(x_i \cdot x_j) - \sum_{i=1}^N\lambda_i \\ 
   s.t. \\
   \sum_{i=1}^N\lambda_iy_i = 0 \\
   \lambda_i >=0, i=1,2,\cdots,N \\
   $$
   由于原始问题满足强对偶条件，所以存在$w^*,b^*,\lambda^*$,使$w^*,b^*$为原始问题的最优解，$\lambda^*$为对偶问题的解，意味着求解原始问题，可以转化为求解对偶问题。

### 引出核函数

对线性可分数据集，假设对偶问题的解为$\lambda^*=(\lambda_1,\lambda_2, \cdots,\lambda_N)^T$，并由$\lambda^*$求得原始问题的最优解为$w^*,b^*$.

根据拉格朗日定理（$x^*和\lambda^*,\mu^*$分别是主问题和对偶问题的解的充分必要条件是$x^*, \lambda^*,\mu^*$满足的KKT条件）可得：
$$
\triangledown_xL(w^*,b^*,\lambda^*) = w^*-\sum_{i=1}^N\lambda_iy_ix_i = 0 \\
\triangledown_bL(w^*,b^*,\lambda^*) = -\sum_{i=1}^N\lambda_iy_i=0 \\
1-y_i(w^Tx_i+b) <= 0, i=1,2,\cdots, N \\
\lambda_i >= 0, i=1,2,\cdots, N \\
\lambda_i\left(1-y_i(w^Tx_i+b)\right)= 0, i=1,2,\cdots, N
$$
由此得
$$
w^* =\sum_{i=1}^N\lambda_iy_ix_i
$$
其中至少有一个$\lambda_j>0$(反证法：若所有$\lambda^*$的元素都为0，则$w^*$为0，而$w^*=0$不是原始问题的解；为什么呢？因为如果w为0的情况，输出y与输入x无关，这不符合机器学习的目的)，对此j有如下成立：
$$
y_j(w^{*T}x_j+b^*) - 1 = 0
$$
将$w^* =\sum_{i=1}^N\lambda_iy_ix_i$带入该式得：
$$
y_j(\sum_{i=1}^N\lambda_iy_ix_i)^Tx_j + y_jb^* - 1 = 0 \\
b^* = 
\left\{\begin{matrix}
1 -(\sum_{i=1}^N\lambda_iy_ix_i)^Tx_j, \text{ if } y_j =1 \\
-1 -(\sum_{i=1}^N\lambda_iy_ix_i)^Tx_j, \text{ if } y_j =-1 
\end{matrix}\right. \\
整理上式得\\
b^*=y_j- (\sum_{i=1}^N\lambda_iy_ix_i)^Tx_j = y_j - \sum_{i=1}^N\lambda_iy_i(x_i \cdot x_j)
$$
现在w和b都有了，那么我们的分离超平面也有了：
$$
w^{*T}x+b^* = 0， 即\\
\sum_{i=1}^N\lambda_iy_i(x_i\cdot x) + \left(y_j - \sum_{i=1}^N\lambda_iy_i(x_i \cdot x_j)\right) = 0
$$
从上式可以看出，分类决策函数只依赖于输入x和训练样本输入的内机。



## 问题

1. 为什么无论主问题的凸性如何,对偶问题始终是凸优化问题?
2. 强对偶性成立条件的证明
3. $\underset{\lambda}{max} \underset{w,b}{min}L(w,b,\lambda) $严格来说应该是这样：$\underset{\lambda}{max} \underset{w,b}{inf}L(w,b,\lambda)$
   至于为什么可以写成min不确定，猜测可能是强对偶问题所以等价？

## 参考

1. http://netedu.xauat.edu.cn/jpkc/netedu/jpkc/gdsx/homepage/5jxsd/51/513/5308/530807.htm
2. https://blog.csdn.net/zxyhhjs2017/article/details/79162384
3. https://zhuanlan.zhihu.com/p/24638007
4. https://zhuanlan.zhihu.com/p/29865057
5. https://blog.csdn.net/ecnu18918079120/article/details/72971034
6. 《机器学习》周志华
7. 《统计机器学习》李航























