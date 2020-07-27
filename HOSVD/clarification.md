For Hypothesis Testing Problems, let the decision rule to accept $H_1: X \to P_1$ is

$A=\{x: P_1(x) >P_2(x)\}$, then the Type I error could be bounded by
$$
P_1(A^c) = P_1(\log \frac{P_2(x)}{P_1(x)} > 0) \leq E_{P_1}[\exp(\alpha \log \frac{P_2(x)}{P_1(x)})] = \int (P_2(x)/P_1(x))^\alpha P_1(x)dx = \int P_1^{1-\alpha}(x)P_2^{\alpha}(x)dx
$$
This result uses The Chernoff Bound Inequality. In [], the term$D_{\alpha}(P_1||P_2)=-\log\int P_1^{1-\alpha}(x)P_2^{\alpha}(x)dx$ is called Chernoff $\alpha-$divergence. Then we have $P_1(A) \leq \exp^{-D_{\alpha}(P_1||P_2)}$.

## Chernoff-Stein Lemma

This lemma deals with the case that when type I error $\alpha_n$ is controlled, type II error $\beta_n$ exponentially decays in $\exp^{-D(P_1||P_2)}$.

