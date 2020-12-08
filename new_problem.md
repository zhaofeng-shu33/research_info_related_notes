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

($Y_2$ and $Y_1$ are independent)





