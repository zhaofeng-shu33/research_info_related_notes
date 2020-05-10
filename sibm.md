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

### critical value of $\beta$

$\sigma \in \{1,\omega,\omega^2 \}^n$, root of $x^3-1$. cyclic group: $\sigma^3=1$

![](./case3.png)

Indicator function
$$
I(\sigma_i, \sigma_j) = \begin{cases} 2 & \sigma_i = \sigma_j \\
-1 & \sigma_i \neq \sigma_j\end{cases}
$$
Suppose $x_i, \dots, x_n$ are draw uniformly from $\{1,\omega,\omega^2\}^n$ (That is, there are $\frac{n}{3}$ items taking value $0$, $\frac{n}{3}$ items taking value 1...)

Thus gurantee $\sum_{i=1}^n I(x_i, \sigma) = 0$ for any $\sigma$

Ising model
$$
P_{\sigma | G}(\sigma = \bar{\sigma}) = \frac{1}{Z_G(\alpha, \beta)}\exp\left(\beta \sum_{\{i,j\}\in E(G)}I(\bar{\sigma}_i, \bar{\sigma}_j) - \frac{\alpha \log(n)}{n} \sum_{\{i,j\} \not\in E(G)} I(\bar{\sigma}_i,\bar{\sigma}_j)\right)
$$




For SBM with three communities and $p=\frac{a\log n}{n}, q=\frac{b\log n}{n}$

  $\sqrt{a} - \sqrt{b} > \sqrt{3}$ is sufficient to recover the community structure (Abbe).

Define
$$
\begin{align}
A_i &= |\{j \in [n]\backslash\{i\}: \{i,j\} \in E(G), X_j = X_i\}| \\
B_i &= |\{j \in [n]\backslash\{i\}: \{i,j\} \in E(G), X_j = \omega \cdot X_i\}| \\
C_i &= |\{j \in [n]\backslash\{i\}: \{i,j\} \in E(G), X_j = \omega^2 \cdot X_i\}|

\end{align}
$$


Computation of critical value $\beta^*$:
$$
\begin{align}
\frac{P_{\sigma |G}(\sigma_i \neq X_i)}{P_{\sigma | G}(\sigma = X)} &=
\frac{P_{\sigma |G}(\sigma_i = \omega \cdot X_i)}{P_{\sigma | G}(\sigma_i = X)}+\frac{P_{\sigma |G}(\sigma_i = \omega^2 \cdot X_i)}{P_{\sigma | G}(\sigma = X)}\\
&= \exp\left(3(\beta + \frac{\alpha \log n}{n})(B_i-A_i)-\frac{4\alpha \log n}{n}\right)\\
&+ \exp\left(3(\beta + \frac{\alpha \log n}{n})(C_i-A_i)-\frac{4\alpha \log n}{n}\right)
\end{align}
$$
Due to symmetry, when evaluation the expection, only

need to examine $\mathbb{E}_G[\exp(3\beta (B_i - A_i))]$ when $A_i \sim B(\frac{n}{3}-1, \frac{a\log n}{n}), B_i \sim B(\frac{n}{3}, \frac{b\log n}{n})$.
$$
g(\beta) = \frac{1}{3}(ae^{-3\beta}+be^{3\beta}) - \frac{a+b}{3} +1
$$
$g'(\beta)$ has unique root $\frac{1}{6}\log \frac{a}{b}$ and $g(\beta)$ is convex. and $g(\frac{1}{6} \log \frac{a}{b}) = \frac{1}{3}(2\sqrt{ab}-a-b) + 1$.

Let $g(\frac{1}{6} \log \frac{a}{b}) < 0$ we get the threshold: $\sqrt{a} - \sqrt{b} > \sqrt{3}$, which is same with Abbe's result.

The smaller root of $g(\beta)=0$ is
$$
\beta^* = \frac{1}{3}\log \frac{a+b-3 - \sqrt{(a+b-3)^2-4ab}}{2b}
$$

### Generalization of $\alpha = b \beta$

Empirical deduction: This result can be got by evaluating $\mathbb{E}_G \log(\Pr(\sigma = X)) = \mathbb{E}_G \log(\Pr(\sigma = 1)) $,

To be more precisely:

Consider the expection of edge numbers: 
$$
\mathbb{E}_{G}[|E|] = \frac{a \log n}{n} \frac{n}{2}(\frac{n}{2} - 1) + \frac{b\log n}{n}\left(\frac{n}{2}\right)^2 \approx \frac{a+b}{4} n \log n
$$
Then 
$$
\begin{align}
\mathbb{E}_G \log(\Pr(\sigma = 1)) &= \beta \mathbb{E}_{G}[|E|] - \frac{\alpha \log n}{n}(\frac{n^2-n}{2}-\mathbb{E}_{G}[|E|])-\log Z \approx (\frac{(a+b)\beta}{4} - \frac{\alpha}{2})n\log n \\
\mathbb{E}_G \log(\Pr(\sigma = X)) &= \frac{\beta}{2}\sum_{i=1}^n(\mathbb{E}[A_i - B_i]-\frac{\alpha \log n}{n}\mathbb{E}[(\frac{n}{2}-1-A_i)-(\frac{n}{2}-B_i)])\approx \frac{(a-b)\beta}{4}n\log n
\end{align}
$$
Therefore we can get $\alpha = b \beta$ as the critical condition.

Using the same method as above, we can get $\alpha = b \beta$ is also critical value for $k=3$.

With $\mathbb{E}_G[|E|] = \frac{n\log n}{3}(\frac{a}{2} + b)$, $\mathbb{E}_G \log(\Pr(\sigma = 1))\approx (\frac{a+2b}{3}\beta - \alpha) n \log n$ and $\mathbb{E}_G \log(\Pr(\sigma = X)) \approx \frac{\beta(a-b)}{3} n \log n $

To be more rigorous, we would like to estimate

 $\sum_{i,j} M_{ij} I(\sigma_i, \sigma_j)$  where $M_{ij}$ is defined as
$$
M_{ij} =
\begin{cases}
(a\beta - \alpha + a\alpha \frac{\log n}{n}) \frac{\log n}{n} &\textrm{ if } X_i = X_j \\
(b\beta - \alpha + b\alpha \frac{\log n}{n}) \frac{\log n}{n} &\textrm{ if } X_i \neq X_j 
\end{cases}
$$
Let
$$
\Xi = \begin{bmatrix}
u_1, u_2, u_3 \\
v_1, v_2, v_3 \\
w_1, w_2, w_3
\end{bmatrix}, Q = \begin{bmatrix}
Q_{11}, Q_{12}, Q_{13} \\
Q_{21}, Q_{22}, Q_{23} \\
Q_{31}, Q_{32}, Q_{33}
\end{bmatrix}
$$
which has 9 non-negative variables. Besides, each row of $\Xi$ sums to $\frac{n}{3}$.

The meaning of each variable:

$u_1 = |\{i\in [n]: X_i =1, \sigma_i = 1\}|$, $u_2 = |\{i\in [n]: X_i =1, \sigma_i = \omega\}|$, $u_3 = |\{i\in [n]: X_i =1, \sigma_i = \omega^2\}|$,$v_1 = |\{i\in [n]: X_i =\omega, \sigma_i = 1\}|$.

Suppose the subscript starts from 1: $\Xi_{ij} = |\{k \in [n]: X_k = \omega^{i-1}, \sigma_k = \omega^{j-1}\}|$.

Also
$$
Q_{ij} = (\Xi_{i1} - \Xi_{i2})(\Xi_{j1} - \Xi_{j2}) + (\Xi_{i1} - \Xi_{i3})(\Xi_{j1} - \Xi_{j3}) + (\Xi_{i2} - \Xi_{i3})(\Xi_{j2} - \Xi_{j3})
$$


$A =(a\beta - \alpha) \frac{\log n }{n}, B=(b\beta - \alpha) \frac{\log n}{n}$

We can write
$$
\sum_{i,j=1}^n M_{ij} I(\sigma_i, \sigma_j) = (Q_{11} + Q_{22} + Q_{33})(A-B) + (\sum_{i,j=1}^3 Q_{ij})B + \textrm{ high order term}
$$
Using the expression of $Q_{ij}$ we can show that the coeffient of $B$ is
$$
\begin{align}
\sum_{i,j=1}^3 Q_{ij} = &((u_1+v_1+w_1)-(u_2+v_2+w_2))^2 \\
+ &((u_1+v_1+w_1)-(u_3+v_3+w_3))^2 + ((u_2+v_2+w_2)-(u_3+v_3+w_3))^2
\end{align}
$$
which is non-negative.

Therefore, when $\alpha > b \beta (B<0)$, we should let $\sum_{i,j=1}^3 Q_{ij}=0$ to maximize $\sum_{i,j} M_{ij} I(\sigma_i, \sigma_j)$.

Actually we can maximize the coeffient of $A$ and $B$ respectively and these two maximal values can co-exist.

There are 6 kinds of possibilities for such $\Xi \in \{e, s, r, r^2, sr, sr^2\}$. $e=\frac{n}{3}I_3$ is the identity of the group.
$$
r = \frac{n}{3}\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0
\end{bmatrix}, s = \frac{n}{3} \begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$


This group is isomorphic to $S_3$.

This is equivalent to say when $\sigma  = f(X)$, $\Pr(X=\sigma)$ is maximized under the condition of $\alpha > b \beta$.

The function $f$ is applied to $X$ pairwisely. For each element, it is selected from the symmetric group $S_3$,

for example
$$
f=\binom{1, \omega, \omega^2}{\omega, 1, \omega^2}
$$
means $f(1) = \omega, f(\omega)=1, f(\omega^2) = \omega^2$.

When $\alpha < b \beta$, similarly we can show that when $\sigma$ all equals to $1, \omega$ or $\omega^2$, the maximum is taken. We represent it as $f_1$.

## How to show the concentration of $X_G$

$$
X_G = \beta \sum_{\{i,j\} \in E(G)} I(\sigma_i, \sigma_j) - \frac{\alpha \log n}{n}\sum_{\{i,j\} \not\in E(G)} I(\sigma_i, \sigma_j)
$$

For $k=3$, we have $I(\sigma_i, \sigma_j) = 2 \mathrm{Re}(\sigma_i \bar{\sigma}_j)$. $\bar{\sigma}$ is the complex conjugate of $\sigma$. 

This result cannot be extended to $k>3$ using complex plane.

A simple illustration for $k=4$ (the number is $\sigma_i \bar{\sigma}_j$ when $\sigma_i \neq \sigma_j$:

![](./impossible_4.svg)

Needs at least a tetrahedron in 3D.

## weak discovery in SIBM

Let $(X, G, \{\sigma^{(1)}, \dots, \sigma^{(m)}\}) \sim SIBM(n, \frac{a}{n}, \frac{b}{n}, \beta, m)$

If there exists an algorithm that takes $ \{\sigma^{(1)}, \dots, \sigma^{(m)}\}$ as inputs and outputs $\hat{X} =\hat{X}( \{\sigma^{(1)}, \dots, \sigma^{(m)}\})$ such

that

$ P(A(\hat{X}, X) \geq 1/2 + \epsilon)  \to 1$ as $n \to \infty$

For SBM with two communities, the problem is solvable if $(a-b)^2 > 2(a+b)$.

Exponential time algorithm: counting K-cycles;

