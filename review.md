# Literature Review on Recent Developments on the topic of community discovery(theoretical literature)

## Ising Block Model (2019)

* No stochastic assumption about underlining community.

* Consider two communities

* Get the critical region involving $\alpha, \beta$.

Generating Model: Consider a partition of $V=\{1,\dots, p\}$ into $S$ and $\bar{S}$ with $|S|=\frac{p}{2}$, a determined graph $G$ is the combination of two complete graphs of $S$  and $\bar{S}$. Then we can generate a probability distribution
$$
P_{\sigma | G}(\sigma = \bar{\sigma}) = \frac{1}{Z_G(\alpha, \beta)}\exp\left(\frac{\beta}{2p} \sum_{\{i,j\}\in E(G)}\bar{\sigma}_i \bar{\sigma}_j + \frac{\alpha}{2p} \sum_{\{i,j\} \not\in E(G)} \bar{\sigma}_i \bar{\sigma}_j\right), \bar{\sigma}_i \in \{\pm 1 \}
$$
$\beta > \alpha$ can gurantee that inner community attraction is larger than intra-community one. ($\alpha$ can be positive or negative)

In this paper, the critical region of $\alpha$ and $\beta$ is got. Under these two regions, the sample complexity of

indepedent samples $n$ which is necessary to recover $S$ with probability 1 has been deducted.

## Hypergraph stochastic block model (2020)

Consider the recovery problem of k clustering for d-uniform hypergraph. That is, every edge connects $d$ nodes. This paper focus on the regime when $p,q$ are fixed while $n,k$ tends to infinity (dense community).

## Weighted Stochastic Block Model(2020)

Each edge is associated with a weight from a given density function $p(\cdot)$ (if the edge is within the community) or $q(\cdot)$ (if the edge joins two different communities).

The result is about general weighted model and some strange assumptions have to be used to make proof possible. For example, this theory requires each cluster should have at least $\frac{n}{\beta k}$ nodes. The extra parameter $\beta$ is introduced.

## Hierarchical Model proposed by Professor Huang

Consider the case with 2 attributes, there are four communities $S_1, S_2, S_3, S_4$, each with equal size $|S_i|=\frac{n}{4}$.

For $(i,j) \in S_i$, the probability of existence of an edge is $p$;

For $i \in S_1, j \in S_2$ or $i\in S_2, j\in S_3$ or $i\in S_3, j\in S_4$ or $i\in S_4, j\in S_1$ , the probability of existence of an edge is $q$;

For $i\in S_1, j\in S_3$ or $i\in S_2, j \in S_4$, the probability of existence of an edge is $r$. We require that $p>q>r$

Under which condition of $p,q,r$ can we find an efficient algorithm to recover the community as $n\to \infty$?

When $q=r$, the model reduces to SBM with $k=4$.

![](./S4.svg)

Practical Implications: We can think each node is associated with two binary attributes. When the two attributes are the same, they belong to one cluster $S_i$. When only one attribute differs, the relationship is

captured by $q$; When both the two attributes differ, the quantity is described by $r$.



[1] Berthet Q, Rigollet P, Srivastava P. Exact recovery in the Ising blockmodel. Ann Statist. 2019;47(4):1805-1834.

[2] Cole S, Zhu Y. Exact recovery in the hypergraph stochastic block model: A spectral algorithm. Linear Algebra and its Applications. 2020;593:45-73.

[3] Xu M, Jog V, Loh PL. OPTIMAL RATES FOR COMMUNITY ESTIMATION IN THE WEIGHTED STOCHASTIC BLOCK MODEL. Annals of Statistics. 2020;48(1):183-204.

# Literature Review on Semi-supervised SBM (Algorithm)

## Signed networks Model(2020)

* the edge takes values from $\{\pm 1\}$

* Variational Bayesian

[1] Liu, X., Song, W., Musial, K., Zhao, X., Zuo, W., Yang, B., 2020. Semi-supervised stochastic blockmodel for structure analysis of signed networks. Knowledge-Based Systems.. doi:10.1016/j.knosys.2020.105714

