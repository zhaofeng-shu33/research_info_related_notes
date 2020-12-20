### Graph embedding and community detection

We consider a graph embedding function $f(G)$ and graph label $Y$.

By using data clustering techniques (for example k-means clustering) we can get an estimator $\hat{Y}$ from $f(G)$.

Spectral clustering is such a two-stage method.

Our goal is to find the embedding function which can make the exact recovery possible.

$f: G \to X_{n \times k}$



Assumption: the edges are more densely connected within the community and

are loosely connected between different communities.



$Y$ is the node label (independent Bernoulli random variable).

$X_{ij}$ be the edge connection information, random variable depending only on node $i, j$. We can regard

$X$ as an adjacency matrix.

Suppose $f$ be a function which maps $n(n-1)/2$ vector to a $k$ vector and $g$ is a scalar function

which maps the node label to its higher ($k$) dimension embedding. Suppose there are $r$ communities,

then we should choose $k=r-1$. For example, if $r=2$, $g(1)=1,g(-1)=-1$. Our goal is to maximize the averaged correlation. The optimization problem should have some normalization constraint. We require that $E[f(X)]=0$ and $\textrm{Var}[f(X)] = 1$. 
$$
\max E[ g(Y_1) \cdot f(X) | Y_2 = 1]
$$

By Cauchy's inequality we know the maximal value is 1.

($Y_2$ and $Y_1$ are independent; $X$ and $Y$ are independent)

We can empirically estimate the maximum value of:
$$
\frac{1}{2} \sum_{x\in \mathcal{X}} f(x)[P(X=x|Y_1=1,Y_2=1) - P(X=x|Y_1=-1,Y_2=1)]
$$

Then we have $f(x)=\frac{1}{C}[P(X=x|Y_1=1,Y_2=1) - P(X=x|Y_1=-1,Y_2=1)]$

where $C$ is the normalization constant such that $\sum_{x\in \mathcal{X}} P(x)f^2(x)=1$.

## Simple example for $n=4$

Two parameters: $p,q$.

$|\mathcal{X}|=64$. The length of vector $X$ is 6.

Simulation: $p=0.7, q=0.3$, the optimal value is 0.44.



Now we consider $r>2$, then we should fix $Y_2 = v_2, Y_{r} = v_r$ where $v_1, v_2, \dots v_r \in \mathbb{R}^{r-1}$.

$||v_i||=1, \sum_{i=1}^r = v_i = 0, v_i \cdot v_j = \frac{-1}{r-1}$.

We are solving:
$$
\max E[ \sum_{i=2}^ r g_i(Y_1) \cdot f_i(X) | Y_2 = v_2, \dots, Y_r = v_r]
$$


### Useful property of HGR maximal correlation

If the HGR maximal correlation $\rho(X;Y)=0$, then $X$ and $Y$ are independent.

Proof:

For any $f,g$ we have $\mathbb{E}[f(x)g(y)] = \mathbb{E}[f(x)]\mathbb{E}[g(y)]$.

Choosing $f(x)=\mathbf{1}[X=x]$ and $g(y) = \mathbf{1}[Y=y]$,  we have

$P(X=x, Y=y) = P(X=x)P(Y=y)$.

That is, $X$ and $Y$ are independent.



Understanding HGR maximal correlation:

Suppose $Y, X$ are correlated binary distributions, specified by

$P(Y=1, X=1)=p, P(Y=0, X=1)=q, P(Y=1, X=0)= r - p, P(Y=0, X=0)=1-r-q$.

Then the B matrix of $X,Y$ can be written as:
$$
B=\begin{pmatrix}
\frac{1-r-q}{\sqrt{1-r}\sqrt{1-p-q}} & \frac{q}{\sqrt{1-r}\sqrt{p+q}} \\
\frac{r-p}{\sqrt{r}\sqrt{1-p-q}} & \frac{p}{\sqrt{r}\sqrt{p+q}}
\end{pmatrix}
$$
We can write the SVD decomposition of $B$ analytically:

$B=U\Sigma V^T$ where $\Sigma = \textrm{diag}(1, \sigma_2)$
$$
\sigma_2=\frac{|p-pr-qr|}{\sqrt{r(1-r)}\sqrt{p+q}\sqrt{1-p-q}}, U = \begin{pmatrix}
\sqrt{1-r} & \sqrt{r}\\
\sqrt{r} & -\sqrt{1-r}
\end{pmatrix}, V = \begin{pmatrix}
\sqrt{1-p-q} &  s\sqrt{p+q}\\
\sqrt{p+q} & -s \sqrt{1-p-q}
\end{pmatrix}
$$
The sign indicator $s$ is defined as:
$$
s = \begin{cases}
1 & p - pr - qr > 0 \\
-1 & p - pr - qr < 0
\end{cases}
$$

When $p=pr+qr$, $\sigma_2=0$ which means that $X$ and $Y$ are indepedent.

For this special example, the HGR maximal correlation $\sigma_2$ equals the Pearson correlation coefficient.

Binary Symmetric model is the special case of the above 2x2 B matrix. (See for example, https://github.com/zhaofeng-shu33/ace_cream/blob/master/example/BSC.py)



## Neural Network Approach

We can treat the $X$ as a zero-one image ($n\times n$) and $Y_1$ as its classification label. We consider the joint

distribution.



## Spectral Clustering Approach

To embed $Y_1$ given $Y_2=1$ in a numerical number.  We use the second smallest eigenvector of the unnormalized Laplacian function of the graph. We should make the eigenvector to have length 1

and the second component to be positive. Then we use the first component as the numerical embedding

of $Y_1$.



## Comments of Professor Huang

Optimal value is computational intractable. Try to find some sub-optimal algorithm and show that in some

special model they can approximate the optimal value in asymptotic regime.

SBM is not a very useful model, especially when modeling the community in real world.

Show that spectral embedding is a special kind of embedding under the framework of encoding-decoding view.