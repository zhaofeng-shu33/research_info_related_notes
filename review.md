# Literature Review on Recent Developments on the topic of community discovery(theoretical literature)

## Ising Block Model (2019)

* No stochastic assumption about underlining community.

* Consider two communities

* Get the critical region involving $\alpha, \beta$.

Generating Model: Consider a partition of $V=\{1,\dots, p\}$ into $S$ and $\bar{S}$ with $|S|=\frac{p}{2}$, a determined graph $G$ is the combination of two complete graphs of $S$  and $\bar{S}$. Then we can generate a probability distribution
$$
P_{\sigma | G}(\sigma = \bar{\sigma}) = \frac{1}{Z_G(\alpha, \beta)}\exp\left(\frac{\beta}{2p} \sum_{\{i,j\}\in E(G)}\bar{\sigma}_i \bar{\sigma}_j + \frac{\alpha}{2p} \sum_{\{i,j\} \not\in E(G)} \bar{\sigma}_i \bar{\sigma}_j\right), \bar{\sigma}_i \in \{\pm 1 \}
$$
$\beta > \alpha$ can guarantee that inner community attraction is larger than intra-community one. ($\alpha$ can be positive or negative)

In this paper, the critical region of $\alpha$ and $\beta$ is got. Under these two regions, the sample complexity of

independent samples $n$ which is necessary to recover $S$ with probability 1 has been deducted.

## Hypergraph stochastic block model (2020)

Consider the recovery problem of k clustering for d-uniform hypergraph. That is, every edge connects $d$ nodes. This paper focus on the regime when $p,q$ are fixed while $n,k$ tends to infinity (dense community).

## Weighted Stochastic Block Model (2020)

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

## Signed networks Model (2020)

* partially labeled node
* the edge takes values from $\{\pm 1\}$
* Variational Bayesian




## Empirical Study of Phase transitions (2014)

- SBM with a fraction $\alpha$ nodes whose labels are known
- Study the detection versus non-detection regime using a belief propagation algorithm

## Belief propagation is optimal in some region when side information is provided (2016, Mossel)

Let $\sigma$ be true label, each component of $\sigma$ is sent into a BSC channel with probability $\alpha \in [0,1/2)$.

Then we can get a noisy label $\tilde{\sigma}$. Using the side information $\tilde{\sigma}$ and consider a SBM$(\frac{a}{n}, \frac{b}{n})$.

[1] Liu, X., Song, W., Musial, K., Zhao, X., Zuo, W., Yang, B., 2020. Semi-supervised stochastic blockmodel for structure analysis of signed networks. Knowledge-Based Systems.. doi:10.1016/j.knosys.2020.105714

[2] Zhang, Pan, Cristopher Moore, and Lenka Zdeborová. "Phase transitions in semisupervised clustering of sparse networks." *Physical Review E* 90.5 (2014): 052802.

[3] Mossel, Elchanan, and Jiaming Xu. "Local algorithms for block models with side information." *Proceedings of the 2016 ACM Conference on Innovations in Theoretical Computer Science*. 2016.



# Information Theoretic Approach in Community Detection

## pairwise measurement

The sample complexity (the number of edges required) to almost exactly recover the labels up to a constant

is of order $\frac{n\log n}{\mathrm{Hel}^{\mathrm{min}}_{1/2}}$.

Also, the information measure is difficult to estimate from one sampling of data.

[1] Chen, Yuxin, Changho Suh, and Andrea J. Goldsmith. "Information recovery from pairwise measurements." *IEEE Transactions on Information Theory* 62.10 (2016): 5881-5905.

## New model based on pairwise measurement

Consider ER graph with $p_{\textrm{obj}} \geq \log n /n$. Every node of the graph has a label $x_i$ coming from a cyclic group with order $M$. Suppose for every edge of the graph, there is an observation value $y$ coming from $\{0, 1, \dots, M\}$ and is prescribed by $p(y_{ij} | x_i - x_j = l) = P_{l}(y_{ij})$.  Also the discrete 

conditional distribution is only dependent on $l$.  Also for each node there are multiple independent samples (the number of sample is T) with probability $p(y_i | x_i = l)= P_l(y_i)$. The transfer probability is the same with that of edges. Notice node sample is different with edge sample. For each edge we only have one observation but for each node we have multiple observations. Then we investigate the error probability of ML algorithm $\psi$ to recover $X$. Also notice that the error probability is chosen as the maximum error:
$$
P_e(\psi) := P\{\textrm{dist}(\psi(y), x) \neq 0 | x \}
$$
The chosen of multiple observations for each node is to balance the contribution of nodes and that of edges. Compared to the original pairwise measurements model, we added noisy node observations, making the new model semi-supervised learning.

Some ideas to transform node observation to edge observation: Add new nodes with label zero. And add one edge from the new node to each old node.

Problems: the transformation is deterministic and the expanded graph does not satisfy the condition any more.

We take an approach to insert additional observations into the proof of Theorem 1 in the original paper of Chen, Yu Xin. 

$N_w = T n_0 + \frac{1}{2}(n^2 - \sum_{i=0}^{M-1} n_i^2 )$

In Chen's original paper,  there is an assumption of $n_0 \geq n_i$ for $i=1, \dots, M-1$. (Equation 91).

This assumption is possible due to the property of translational invariant. We can consider only

proposals with $n_0 \geq n_i$ and multiply the error probability by $M$.

### New model based on SBM

Consider $n$ nodes $Y_1, \dots, Y_n$ with binary labels. If  $Y_i  = Y_j$ there are probability $p$ that there is an edge between the two nodes; If $Y_i \neq Y_j$, there are probability $q$ that there is an edge between them. Also $p>q$.

For each node $Y_i$, we can generate $m$ i.i.d. observations $X_1^{(i)}, \dots, X_m^{(i)}$ from $P_0$ (if $Y_i = 0$) or $P_1$ (if $Y_i=1$).

Suppose given one sample of the graph and $X_j^{(i)}$ for $i=1, \dots, n$ and $j=1, \dots, m$,

What is the Chernoff Bound for ML method?

For SBM only, it is proved that the error probability $\leq c^{-\frac{1}{4}\epsilon}$.

If there is no graph structure, then we do a Hypothesis testing independently for each node, and the judging scheme is to compare $P(X^{(i)}_1, \dots, X^{(i)}_m | Y_i=0)=\prod_{j=1}^m p_0(x^{(i)}_j)$ and $P(X^{(i)}_1, \dots, X^{(i)}_n | Y_i=1) = \prod_{j=1}^m p_1(x_j^{(i)})$ and the error probability decreases as $\exp^{-n D}$ where $D$ is called the Chernoff component and can be computed from the two joint distribution.

Equivalence of ML in SBIM to bisection partition.

Let $z_{ij} \in \{0, 1\}$ to represent whether there is an edge between two nodes in a graph, then
$$
p(z | y) = \prod_{y_i = y_j} p^{z_{ij}} (1-p)^{1-z_{ij}} \prod_{y_i \neq y_j} q^{z_{ij}}(1-q)^{1-z_{ij}}
$$
Let $A$ to represent the number of edges between two parts $y_i=1$ and $y_i=0$.

Then 
$$
p(z|y)=p^{|E|-A}(1-p)^{\frac{n}{2}(\frac{n}{2}-1)-|E|+A} q^A (1-q)^{\frac{n^2}{4}-A}
$$
Suppose $p>q$, to maximize $p(z|y)$ is equivalent to minimize $A$. That is, to find a bisection which minimizes the number of edges across the cut.

Consider also $p(x|y)$ as:
$$
p(x|y) = \prod_{i=1}^n \prod_{j=1}^m p_0(x_j^{(i)})^{y_i} p_1(x_j^{(i)})^{1-y_i}
$$
We maximize $\log p(x,z | y) = \log p(z | y) + \log p(x|y) = C + \sum_{i=1}^n B_i y_i + (\log q - \log p - \log (1-q) + \log (1-p)) A$



The coefficient of $A$ is negative, $C$ is a constant, not involving $y_i$ and $B_i  = \sum_{j=1}^m \log \frac{p_0(x_j^{(i)})}{p_1(x_j^{(i)})}$.

Let $S$ be the adjacency matrix of the graph, then
$$
A = \sum_{(i,j) \in E(G)} (1-y_i) y_j + (1-y_j) y_i = 2 e^T S y - y^T S y
$$
where $e$ is the all-one vector and $y$ is the target $n$-dimensional $\{0,1\}$ vector. We can merge the coefficient of linear component of $y$ in $A$ with $B_i$ and get the final decision rule (ADMM):
$$
\begin{align}
\min\, & y^T S y- b^T y \\
s.t.\,\, & y \in \{0, 1\}^n \\
 &e^T y = \frac{n}{2}
\end{align}
$$
where $b_i = 2\textrm{deg}(i) + B_i$.

p(x, z | y_1)  is difficult to estimate, **sum** cannot be simplified.

We consider $p(y_1 | x, y_2, \dots, y_n ,z)$ instead. Assuming $y_i \sim Bern(0.5)$ That is, we do not require $e^t y = \frac{n}{2}$ exactly.

$H_0: p(x,z | y_2, \dots, y_n, y_1 = 0) = p_0(x) \prod_{i \in N_1(G)} p^{1-y_i} q^{y_i} \prod_{i \not \in N_1(G)} (1-p)^{1-y_i} (1-q)^{y_i} f(y_2, \dots, y_n)$

And

$H_1: p(x, z | y_2, \dots, y_n, y_1 = 1) = p_1(x) \prod_{i \in N_1(G)} p^{y_i} q^{1-y_i} \prod_{i \not \in N_1(G)} (1-p)^{y_i} (1-q)^{1-y_i} f(y_2, \dots, y_n)$

The decision rule to accept $H_0$ is $A(x, y_2, \dots, y_n, z) : = \{(x, y_2, \dots, y_n,z) | p(x,z | y_2, \dots, y_n, y_1 = 0) > p(x,z | y_2, \dots, y_n, y_1 = 1) \}$
$$
A(x, y_2, \dots, y_n, z) : =\{(x, y_2, \dots, y_n,z) | \sum_{i=1}^m \log \left(\frac{p_0(x_i^{(1)})}{p_1(x_i^{(1)})}\right) + \sum_{j=2}^n (1-2y_i)[z_{1j}\log\frac{p}{q} + (1-z_{1j})\log\frac{1-p}{1-q}] > 0 \}
$$


The type I error probability is $P_0((x, y_2, \dots, y_n) \not\in A )$ which can be bounded by $\exp^{-D(P_0 || P_1)}$(Chernoff Bound).

The Chernoff information term is

$D=m D(p_0 || p_1) + E_{P_{z | Y_1 =0, Y_2 = y_2, \dots, Y_n = y_n}}[\log \frac{P_0(z)}{P_1(z)}]$

$$
\begin{align}
P_0(z) & = \prod_{j=2}^n p^{z_{1j}(1-y_j)}q^{z_{1j}y_j}(1-p)^{(1-z_{1j})(1-y_j)}(1-q)^{(1-z_{1j})y_j} \prod_{i>j>1}f(z_{ij}, y_i, y_j) \\
P_1(z) & = \prod_{j=2}^n p^{z_{1j}y_j}q^{z_{1j}(1-y_j)}(1-p)^{(1-z_{1j})y_j}(1-q)^{(1-z_{1j})(1-y_j)} \prod_{i>j>1}f(z_{ij}, y_i, y_j) \\
\end{align}
$$
Products of Bernoulli distribution.

Only $z_{1j}, j=2,\dots, n$ are relevant.
$$
\begin{align*}
E_{P_{z | Y_1 =0, Y_2 = y_2, \dots, Y_n = y_n}}[\log \frac{P_0(z)}{P_1(z)}]
&= \sum_{j=2}^n E_{P_{z_{1j} | Y_1=0, Y_j = y_j}}[ \log \frac{P_0(z_{ij})}{P_1(z_{ij})}] \\
&= \sum_{j=2}^n [(1-y_j) D(p || q) + y_j D(q || p)]
\end{align*}
$$


## Huang's suggestion:

We suppose the prior distributions for $y_1, \dots, y_n$ are i.i.d Bern(0.5). $y_2, \dots, y_n$ should be hidden, to compare $p(x, z | y_1 =1)$ with $p(x,z| y_1 = 0)$ we only need

to compare $\sum_{y_2, \dots, y_n} p(x,z,y_1=1, y_2, \dots, y_n)$ with $\sum_{y_2, \dots, y_n} p(x,z,y_1=0, y_2, \dots, y_n)$.

A suboptimal algorithm:

To estimate $Y_1$, we first estimate $Y_i$ from sample $\{X^{(i)}_1, \dots, X^{(i)}_{m_i}\} $

The posterior distribution for $Y_i$ is $\Pr(Y_i=0 | x^{(i)}_1, \dots, x^{(i)}_{m_i}) = \frac{\prod_{j=1}^{m_i} p_0(x^{(i)}_j)}{\prod_{j=1}^{m_i} p_0(x^{(i)}_j)+\prod_{j=1}^{m_i} p_1(x^{(i)}_j)} $
$$
\Pr(Y_i=0 | x^{(i)}_1, \dots, x^{(i)}_{m_i})= \frac{1}{1+\exp\left(-\displaystyle\sum_{j=1}^{m_i} \log \frac{p_0(x^{(i)}_j)}{p_1(x^{(i)}_j)}\right)}
$$
Suppose $x_1^{(i)}, \dots x_{m_i}^{(i)}$ is sampled from $p_0$. Then as $m_i \to \infty$, $\Pr(Y_i=0 | x^{(i)}_1, \dots, x^{(i)}_{m_i}) = \frac{1}{1+\exp^{-m_i D(p_0 || p_1)}} \sim 1 - \exp^{-m_i D(p_0 || p_1)}$. Therefore, the type I error is $\Pr(Y_i=1 | x^{(i)}_1, \dots, x^{(i)}_{m_i}) \sim \exp^{-m_iD(p_0 || p_1)}$

A naive upper bound for this 2-stage method: $\sum_{i=2}^n \exp^{-m_i D(p_0 || p_1)} + \exp^{-m_1 D(p_0 || p_1) - n D(p || q)}$



## Chernoff Bound for SBM

[1] Zhou, Zhixin, and Ping Li. "Rate optimal Chernoff bound and application to community detection in the stochastic block models." *Electronic Journal of Statistics* 14.1 (2020): 1302-1347.

[2] Zhang, Anderson Y., and Harrison H. Zhou. "Minimax rates of community detection in stochastic block models." *The Annals of Statistics* 44.5 (2016): 2252-2280.

