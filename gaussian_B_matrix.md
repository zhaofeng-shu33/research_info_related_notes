We investigate $||\widetilde{B}||_F^2$ for two Gaussian random variables: $X$ and $Y$. We denote their coefficient as $\epsilon$, which is a very small.

There are two methods to compute this quantity. One is using direct integration.

Since 

$$
P_{XY}(x,y) = P_X(x)P_Y(y) + \epsilon \sqrt{P_X(x)P_Y(y)} \phi(x,y)
$$

and $\widetilde{B}(x,y) = \phi(x,y) \epsilon$.

We assume both $X,Y$ are standard Gaussian distribution, then we can compute
$$
\phi(x,y) = \frac{xy}{\sqrt{2\pi}} \exp(-\frac{x^2+y^2}{4})
$$
Then $||\widetilde{B}||_F^2= \epsilon^2 \int \phi^2(x,y) dxdy$

