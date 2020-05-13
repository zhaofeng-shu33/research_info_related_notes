Literature Review on Recent Developments on the topic of community discovery(theoretical literature)

## Ising Block Model (2019)

* No stochastic assumption about underlining community.

* Consider two communities

* Get the critical region involing $\alpha, \beta$.

Generating Model: Consider a partition of $V=\{1,\dots, p\}$ into $S$ and $\bar{S}$ with $|S|=\frac{p}{2}$, a determined graph $G$ is the combination of two complete graphs of $S$  and $\bar{S}$. Then we can generate a probability distribution
$$
P_{\sigma | G}(\sigma = \bar{\sigma}) = \frac{1}{Z_G(\alpha, \beta)}\exp\left(\frac{\beta}{2p} \sum_{\{i,j\}\in E(G)}\bar{\sigma}_i \bar{\sigma}_j + \frac{\alpha}{2p} \sum_{\{i,j\} \not\in E(G)} \bar{\sigma}_i \bar{\sigma}_j\right), \bar{\sigma}_i \in \{\pm 1 \}
$$
$\beta > \alpha$ can gurantee that inner community attraction is larger than intra-community one. ($\alpha$ can be positive or negative)

In this paper, the critical region of $\alpha$ and $\beta$ is got. Under these two regions, the sample complexity of

indepedent samples $n$ which is necessary to recover $S$ with probability 1 has been deducted.

## Hypergraph stochastic block model (2020)

