# Erdos Renyi Graph

$G(n, p)$ where $p=c\frac{\log n}{n}$. $A$ is the adjacency matrix.

$|\sigma^T(A-\mathbb{E}[A|X])\sigma| \leq c \sqrt{\log n}$ with high probability while $||\sigma||_2=1$ ([2])

Then we can use some techniques to generalize this result to SBM graph([1]).

For the conclusion for ER graph, it is equivalent to say the second largest eigenvalue of $A$ is order $O(\log(n))$. The proof uses a lot of linear space of matrix, which cannot be generalized to $\sum_{i,j}I(\sigma_i, \sigma_j) A_{ij}$.

A proof with looser bound (not right?):

Consider $I(x,x) = (k-1)$ and $I(x, y) = -1$ if $x \neq y$.

We will show that $|\sum_{i,j}I(\sigma_i, \sigma_j)( A_{ij} -E[A_{ij}])| \leq c n \sqrt{\log n}$ with probabilty $1-\frac{k^2a}{n}$.

There are $n^2-n$ terms in the summation, Let $X = \sum_{i,j}I(\sigma_i, \sigma_j)( A_{ij} -E[A_{ij}])$.

The variance of $X$ is 
$$
\mathrm{Var}[X] = \sum_{i,j}I^2(\sigma_i, \sigma_j)\mathrm{Var}[A_{ij}] \leq (n^2-n)\frac{k^2a \log n}{n} \leq k^2a n \log n
$$


Using Chebyshev inequality we have
$$
\Pr(|\sum_{i,j}I(\sigma_i, \sigma_j)( M_{ij} -E[M_{ij}])| \geq \sqrt{c}) \leq \frac{\mathrm{Var}[X]}{c}
$$
Let $c = n^2 \log n$ we then have
$$
\Pr(|\sum_{i,j}I(\sigma_i, \sigma_j)( M_{ij} -E[M_{ij}])| \geq n\sqrt{\log n}) \leq \frac{k^2a}{n}
$$


[1] Distributed user profiling via spectral methods, 2010, ACM SIGMETRICS conference

[2] Spectral Techniques Applied to Sparse Random Graphs,2004, Random Structure Algorithms