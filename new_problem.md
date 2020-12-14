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

then we should choose $k=r-1$. For example, if $k=2$, $g(1)=1,g(-1)=-1$. Our goal is to maximize the averaged correlation. The optimization problem should have some normalization constraint. We require that $E[f(X)]=0$ and $\textrm{Var}[f(X)] = 1$. 
$$
\max E[ g(Y_1) \cdot f(X) | Y_2 = 1]
$$

By Cauchy's inequality we know the maximal value is 1.

($Y_2$ and $Y_1$ are independent; $X$ and $Y$ are independent)

We can empirically estimate:
$$
\frac{1}{2} \sum_{x\in \mathcal{X}} f(x)[P(X=x|Y_1=1,Y_2=1) - P(X=x|Y_1=-1,Y_2=1)]
$$

Then we have $f(x)=\frac{1}{C}[P(X=x|Y_1=1,Y_2=1) - P(X=x|Y_1=-1,Y_2=1)]$

where $C$ is the normalization constant such that $\sum_{x\in \mathcal{X}} f(x)=1$.



### Useful property of HGR maximal correlation

If $\rho(X;Y)=0$, then $X$ and $Y$ are independent.



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



