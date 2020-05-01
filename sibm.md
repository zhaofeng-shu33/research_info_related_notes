## How to draw X in two communities

$X$ is draw from uniformly whose $\binom{n/2}{n}$ position takes value $+1$;

Each component of $X$ is draw from Bernoulli distrbituion with $p=0.5$.

Abbe said they were equivalent when $n$ is large.

## A simple illustration of Abbe's conclusion

$$
p=\frac{a\log n}{n}, q=\frac{b\log n}{n}
$$

Fully recovering requires $\sqrt{a} - \sqrt{b} > \sqrt{2}$

Problems with critical value?

Min Ye:
$$
P(B_i - A_i \geq 0) \leq n^{-\frac{(\sqrt{a} - \sqrt{b})^2}{2}}
$$
MNS16 (Theorem 2.5):

​      $X \sim B(\frac{n}{2}, p_n), Y \sim B(\frac{n}{2}, q_n), \Pr(Y\geq X) = o(n^{-1}) \iff $ fully recovery

## Extension of Ising model to three communities

$\sigma \in \{0,1,2\}^n$

Indicator function
$$
I(\sigma_i, \sigma_j) = \begin{cases} 1 & \sigma_i = \sigma_j \\
-1 & \sigma_i \neq \sigma_j\end{cases} 
$$
Ising model
$$
P_{\sigma | G}(\sigma = \bar{\sigma}) = \frac{1}{Z_G(\alpha, \beta)}\exp\left(\beta \sum_{\{i,j\}\in E(G)}I(\bar{\sigma}_i, \bar{\sigma}_j) - \frac{\alpha \log(n)}{n} \sum_{\{i,j\} \not\in E(G)} I(\bar{\sigma}_i,\bar{\sigma}_j)\right)
$$




For SBM with three communities and $p=\frac{a\log n}{n}, q=\frac{b\log n}{n}$

  $\sqrt{a} - \sqrt{b} > \sqrt{3}$ is sufficient to recover the community structure.

Computation of critical value $\beta^*$:

Examine $\mathbb{E}_G[\exp(2\beta (B_i - A_i))]$ when $A_i \sim B(\frac{n}{3}, \frac{a\log n}{n}), B_i \sim B(\frac{2n}{3}, \frac{b\log n}{n})$.

Empirically computing $\beta^*$ for this case?