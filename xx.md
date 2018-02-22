contents

* [logistic regression as a neural network](#logistic-regression-as-a-neural-network)
  * [binary classification](#binary-classification)
  * [logistic regression](#logistic-regression)
  * [logistic regression cost function](#logistic-regression-cost-function)
  * [gradient descent](#gradient-descent)
  * [derivatives](#derivatives)
  * [more derivative examples](#more-derivative-examples)
  * [computation graph](#computation-graph)
  * [derivatives with a computation graph](#derivatives-with-a-computation-graph)
  * [logistic regression gradient descent](#logistic-regression-gradient-descent)
  * [gradient descent on m examples](#gradient-descent-on-m-examples)
* [python &amp; vectorization](#python--vectorization)
  * [vectorization](#vectorization)
  * [more examples of vectorization](#more-examples-of-vectorization)
  * [vectorizing logistic regression](#vectorizing-logistic-regression)
  * [vectorizing logistic regression's gradient output](#vectorizing-logistic-regressions-gradient-output)
  * [broadcasting in python](#broadcasting-in-python)
  * [a note on python/numpy vectors](#a-note-on-pythonnumpy-vectors)
  * [quick tour of jupyter/ipython notebooks](#quick-tour-of-jupyteripython-notebooks)
  * [explanation of logistic regression cost function](#explanation-of-logistic-regression-cost-function)


# logistic regression as a neural network

## binary classification

维度为(64, 64, 3)的图片 ===> img vector: x=维度为(64\*64\*3=12288, 1)的列向量。($n_x=12288$)

$$ (x,y), x \in R^{n_x}, y \in \\{0,1\\} $$
$m=m_{train}$个训练样本：${(x^{(1)}, y^{(1)}), ..., (x^{(m)}, y^{(m)})}, m_{test}$个测试样本。

$X$表示一个$n_x\*m$的训练样本矩阵,在python里就是```X.shape=(n_x,m)```
$Y$表示一个$1\*m$的向量,在python里是```Y.shape=(1,m)```

## logistic regression

given $x$, want $\hat y=P(y=1|x), x \in R^{n_x}$

params: $w \in R^{n_x}, b \in R$

output: $\hat y= \sigma(w^Tx+b), \sigma(z)=\frac{1}{1+e^(-z)}$

## logistic regression cost function

$(x^(i),y^(i))$ 表示第i个样本。

**Loss(error) function只针对一条训练样本：**

+ square error的loss function:
$$L(\hat y, y)=1/2*(\hat y - y)^2$$
+ logistic regression的loss function: 
$$L(\hat y, y)=-(ylog\hat y+(1-y)log(1-\hat y))$$

if $y=1, L(\hat y, y) = -log\hat y $, want $log\hat y$ large, want $\hat y$ large

if $y=0, L(\hat y, y) = -log(1-\hat y) $, want $\hat y$ small

**Cost function针对全体训练样本:**
$$J(w,b)=1/m\sum ^m_{i=1}L(\hat y^{(i)}, y^{(i)})=-1/m\sum^m_{i=1}[y^{(i)}log\hat y^{(i)}+(1-y^{(i)})log(1-\hat y^{(i)})]$$

## gradient descent

lr的$J(w,b)$是一个凸函数，所以有全局最优。 因为有全局最优，所以lr的初始化一般是0，不用随机。梯度下降：不断重复$ w=w-\alpha \frac{dJ(w)}{dw}$直到收敛。后续，用$dw$来指代$\frac{dJ(w)}{dw}$。梯度下降的公式：
$$w=w-\alpha dw$$
$$b=b-\alpha db$$

## derivatives

derivative = slope，就是$dy/dx=\Delta y/\Delta x$

## more derivative examples

$f(a)=log_e(a)=ln(a), df(a)/da=\frac{1}{a}$

## computation graph

正向计算图算出输出，算每个参数的梯度就反向算。

![computation graph](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/computation%20graph.png)

## derivatives with a computation graph
$$J=3v,v=a+u,u=bc$$
$$\frac {dJ}{dv}=3, \frac{dv}{da}=1$$
$$so, \frac {dJ}{da}=\frac {dJ}{dv}\frac {dv}{da}=3\*1=3$$
$$if\ b=2,then\ \frac{dJ}{dc}=\frac{dJ}{dv}\frac{dv}{du}\frac{du}{dc}=3\*1\*b=3\*1\*2=6$$
**写代码时，将$\frac{dFinalOutputVar}{dvar}$记为$dvar$(最后输出对这个变量的偏导)**

![derivatives in computation graph](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/computation%20graph-derivatives.png)

## logistic regression gradient descent

$$z=w^Tx+b$$
$$\hat y=a=\sigma (z)$$
$$L(a,y)=-(ylog(a)+(1-y)log(1-a))$$

![derivatives in computation graph in lr](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/computation%20graph-derivatives-lr.png)

## gradient descent on m examples

首先，根据J的公式，可以知道dJ/dw1其实就是对每个样本的dw1求和，然后/m。

![gradient_descent_lr_m_examples_djdw](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/gradient_descent_lr_m_examples_djdw.png)

每一次迭代，遍历m个样本，算出J/dw1/dw2/db，然后用这些梯度去更新一次w1/w2/b。

![gradient_descent_lr_m_examples](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/gradient_descent_lr_m_examples.png)

但这的for loop太多了。。所以我们需要vectorization!

# python & vectorization

## vectorization

![gradient_descent_lr_m_examples](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/vectorization-1.png)

对于两个100w维的向量进行点乘，vectorization(1.5ms) 比for loop(470ms+)快

## more examples of vectorization

![gradient_descent_lr_m_examples](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/vectorization-2.png)

![gradient_descent_lr_m_examples](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/vectorization-3.png)

![gradient_descent_lr_m_examples](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/vectorization-lr.png)
如上图，将$n_x$维的dw变为一个np.array即可干掉内层的for loop。

## vectorizing logistic regression

可见，整个求$Z$的过程可以变成一句话，而求A时，需要封装一个基于numpy的sigmoid函数。

![gradient_descent_lr_m_examples](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/vectorization-lr-2.png)

## vectorizing logistic regression's gradient output

![gradient_descent_lr_m_examples](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/vectorization-lr-3.png)

![gradient_descent_lr_m_examples](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/vectorization-lr-4.png)

## broadcasting in python

```python
A = ndarray([[1,2,3,4],[2,3,4,5],[3,4,5,6]]) # 3*4
calc = A.sum(axis=0) # A的每列求和,得到1*4
calc2 = A.sum(axis=1) # A的每行求和,得到3*1
A/calc.reshape(1,4) #得到一个3*4的矩阵，就是broadcasting。其实等价于A/calc，但为了保险，可以调用reshape(1,4)来确保无误
```
小结：

![broadcasting](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/broadcasting.png)

## a note on python/numpy vectors

```python
a=np.random.randn(5) # a.shape=(5,)是一个vector(rank 1 array)，不是一个矩阵，所以a.T还是(5,)，np.dot(a,a.T)=np.dot(a.T,a)是个1*1的数字

b=np.random.randn(5,1) # a.shape=(5,1),  a.T.shape=(1,5), np.dot(a,a.T)是一个5*5的，np.dot(a.T,a)是一个1*1的矩阵（类似array[[0.444]]））

## 可以加一句：
assert(a.shape == (5, 1))
## 如果不小心搞了个rank 1 array,也可以手动a.reshape((5,1))=a.reshape(5,1)
```

## quick tour of jupyter/ipython notebooks

## explanation of logistic regression cost function

单个样本的loss function, log越大，loss越小：

![lr-loss](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/lr-loss.png)

如果是iid（独立同分布），那么，m个样本的cost function，其实就叫对数似然。对他求极大似然估计，其实就是对m个样本求每个cost function的min:

![lr-cost-m-examples](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/lr-cost-m-examples.png)


## programming assignments

squeeze
```python
np.squeeze(a, axis=None)
## 删掉维数是一的部分，axis可以是Int/int数组，表示只去掉指定下标的部分，如果该部分维数不是1，会报错
x = np.array([[[0], [1], [2]]])
x.shape=(1,3,1)
np.squeeze(x)=array([0,1,2]) # shape=(3,)
np.squeeze(x, axis=(2,))=array([[0, 1, 2]]) # shape=(1,3)
```

把一个shape是(a,b,c,d)的array转成一个type是(b*c*d,a)的array:
```python
X_flatten = X.reshape(X.shape[0], -1).T 
```

图片的预处理：
+ Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
+ Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
+ "Standardize" the data: 对图片而言，所有元素除以255就可以了


contents

  * [neural networks overview](#neural-networks-overview)
  * [neural network representation](#neural-network-representation)
  * [computing a neural network's output](#computing-a-neural-networks-output)
  * [vectorizing across multiple examples](#vectorizing-across-multiple-examples)
  * [explanation for vectorized implementation](#explanation-for-vectorized-implementation)
  * [activation functions](#activation-functions)
  * [why do you need non-linear activation functions](#why-do-you-need-non-linear-activation-functions)
  * [derivatives of activation functions](#derivatives-of-activation-functions)
  * [gradient descent for neural networks](#gradient-descent-for-neural-networks)
  * [backpropagation intuition](#backpropagation-intuition)
  * [random initialization](#random-initialization)


## neural networks overview

其中，每个神经元完成了$z=w^Tx+b$以及$a=\sigma (z)$两个操作($a$表示activation)，每一层的数据用上标[i]表示。

![neural-networks-overview](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/neural-networks-overview.png)

## neural network representation

图示是一个2层nn（inputlayer不算在内，有hidden和output两层）。

如果输入的x有3维，在lr中，$shape(w)=(1,3)$，$shape(b)=(1,1)$。

而在nn中，$shape(w^{[1]})=(4,3)$因为有4个神经元，输入是3维。同理$shape(b^{[1]})=(4,1)$。

而$shape(w^{[2]})=(1,4)$，因为只有1个神经元，输入是3维。同理$shape(b^{[2]})=(1,1)$。

![neural-network-representation](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/neural-network-representation.png)

## computing a neural network's output

![compute-nn-output-1](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/compute-nn-output-1.png)
![compute-nn-output-1](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/compute-nn-output-2.png)

## vectorizing across multiple examples

![vectorizing-across-multiple-examples-for-loop](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/vectorizing-across-multiple-examples-for-loop.png)

矩阵$X$纵向是x的维数（行数），横向是training examples的个数（列数）。

矩阵$Z$、$A$纵向是hidden units的个数（行数），横向是training examples的个数（列数）。

![vectorizing-across-multiple-examples-vectorization](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/vectorizing-across-multiple-examples-vectorization.png)

## explanation for vectorized implementation

![justification-for-vectorized-implementation](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/justification-for-vectorized-implementation.png)
![vectorizing-across-multiple-examples-recap](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/vectorizing-across-multiple-examples-recap.png)

## activation functions

一般来说，$tanh$效果比$sigmoid$好，因为均值是0。但对于outputlayer而言，一般$y \in \\{0,1\\}$，所以希望$0\le \hat y \le 1$，所以会用$sigmoid$。

$ReLU(z)=max(0,z)$比$tanh$好，因为当$x\le 0$时，梯度为0。而$leaky ReLU$在$x\le 0$时，梯度是接近0，效果会好一点，但在实践中还是$ReLU$居多。当$x>0$时，梯度和$x\le0$差很多，所以训练速度会加快。

![activation-functions](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/activation-functions.png)

## why do you need non-linear activation functions

linear activation: 因为$z=wx+b$，所以激活函数$g(z)=z=wx+b$就叫linear activation function，也叫identity activatioin function。

不要在hidden layer用linear activation functions，因为多个linear嵌套，实质上还是linear。

例外，当进行回归时，$y\in R$，可以hidden layer用$ReLU$，但output layer用linear activation。

![why-use-non-linear-functions.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/why-use-non-linear-functions.png)

## derivatives of activation functions

sigmoid的导数
![derivative-of-sigmoid.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/derivative-of-sigmoid.png)

tanh的导数
![derivative-of-tanh.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/derivative-of-tanh.png)

relu和leaky relu的导数(z=0时不可导，但在工程上，直接归入z>0部分)
![derivative-of-tanh.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/derivative-of-tanh.png)

## gradient descent for neural networks

**记住每个W/b的shape!**
![gradient-descent-for-neural-networks](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/gradient-descent-for-neural-networks.png)

其中的keepdims=True表示，输出的$shape=(n^{[2]},1)$而非$(n^{[2]},)$
另外求dz时，两项之前是element-wise product(np.multiply)，其中第二项就是对激活函数求在$z^{[1]}$的导数
![forward-propagation-and-back-propagation](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/forward-propagation-and-back-propagation.png)

## backpropagation intuition

先回顾一下lr：
![back-propagation-lr](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/back-propagation-lr.png)

然后看nn：
![back-propagation-nn](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/back-propagation-nn.png)

扩展到m个examples，并进行vectorize：
![back-propagation-nn-vectorized](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/back-propagation-nn-vectorized.png)
![back-propagation-nn-vectorized-all](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/back-propagation-nn-vectorized-all.png)

**问：为何有的有1/m，有的没有。。。。。**

## random initialization

lr的训练，参数一般都初始化为0。但nn，如果初始化为0，会发现算出来的$a^{[1]}_1=a^{[1]}_2$，$dz^{[1]}_1=dz^{[1]}_2$，所以相当于每个神经元的influence是一样的(symetric)，所以$dw^{[1]}$这个矩阵的每一行都相等。
"Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons."
![if-initialize-zero](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/if-initialize-zero.png)

解决：随机初始化
只要w随机初始化了，b其实影响不大，0就可以了。如果用的是sigmoid/tanh的话，随机初始化时，尽量小，因为如果大的话，激活后会接近两端(无论是+无穷还是-无穷，梯度都接近0)，导致学习过程变得很慢。**对于浅层的网络，0.01就可以了，但如果是更深的网络，可能需要其他系数，这个选择过程，在后面的课程会讲。。。**

![random-initialization](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/random-initialization.png)


contents

  * [deep L-layer neural network](#deep-l-layer-neural-network)
  * [forward propagation in a deep network](#forward-propagation-in-a-deep-network)
  * [getting your matrix dimensions right](#getting-your-matrix-dimensions-right)
  * [why deep representations?](#why-deep-representations)
  * [building blocks of deep neural networks](#building-blocks-of-deep-neural-networks)
  * [forward and backward propagation](#forward-and-backward-propagation)
  * [parameters vs hyperparameters](#parameters-vs-hyperparameters)
  * [what does this have to do with the brain?](#what-does-this-have-to-do-with-the-brain)

## deep L-layer neural network

![deep-neural-network-notation.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/deep-neural-network-notation.png)

## forward propagation in a deep network

$z^{[l]\(i\)}$表示第l层的第i个训练样本(列向量)，$Z^{[l]}$表示将第l层的这$m$个训练样本全部水平地放在一起形成的矩阵。以此vectorization的方法，可以避免从1->$m$的这个for-loop。

另外，从第1层到第4层这个for-loop是无法避免的。

![forward-propagation-in-a-deep-network.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/forward-propagation-in-a-deep-network.png)

## getting your matrix dimensions right

![w-and-b-dimension.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/w-and-b-dimension.png)

vectorized之后，$b^{[l]}$仍然是$(n^{[l]},1)$维，只是因为broadcasting，才复制m遍，变成了$(n^{[l]},m)$维。

![vectorized-implementation.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/vectorized-implementation.png)

## why deep representations?

![intuition-of-deep-representation.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/intuition-of-deep-representation.png)

对于$(x_1,x_2,...x_n)$的异或（XOR）操作，如果用树型结构，n个叶子节点，则树深度是$log_2n+1$(深度为k的满二叉树的第i层上有$2^{i-1}$个结点,总共至多有$2^k-1个结点$)，即只需要$O(log_2n)$层的树就能完成。

而如果采用单隐层，则需要$2^{(n-1)}$个节点

![circuit-theory-and-deep-learning.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/circuit-theory-and-deep-learning.png)

## building blocks of deep neural networks

forward时，需要cache $Z^{[l]}$以供backward使用。

![forward-and-backward-functions.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/forward-and-backward-functions.png)

为了计算backward，其实需要cache的有$Z^{[l]}$、$W^{[l]}$以及$b^{[l]}$：

![forward-backward-functions.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/forward-backward-functions.png)

## forward and backward propagation

forward propagation for layer l:

![forward-propagation-for-layer-l.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/forward-propagation-for-layer-l.png)

backward propagation for layer l:

参考[C1W2的backward propagation intuition部分](https://github.com/daiwk/dl.ai/blob/master/c1/c1w3.md#backpropagation-intuition)：
注意：$da^{[l]}*g^{[l]'}(z^{[l]})$是element-wise product。

![backward-propagation-for-layer-l.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/backward-propagation-for-layer-l.png)

对于最后一层L，如果是sigmoid并采用logloss，那么：

![summary-forward-backward-propagation.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/summary-forward-backward-propagation.png)

## parameters vs hyperparameters

图中下方是：momentum，mini-batch size，regularization
![summary-forward-backward-propagation.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/parameters-vs-hyperparameters.png)

![apply-deep-learning-is-a-very-empirical-process.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/apply-deep-learning-is-a-very-empirical-process.png)

## what does this have to do with the brain?

![summary-and-brain.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/summary-and-brain.png)

## others

![shape-of-L-layer-nn.png](https://raw.githubusercontent.com/daiwk/dl.ai/master/c1/imgs/shape-of-L-layer-nn.png)
