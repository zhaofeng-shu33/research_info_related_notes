We investigate $||\widetilde{B}||_F^2$ for two Gaussian random variables: $X$ and $Y$. We denote their coefficient as $\epsilon$, which is a very small.

There are two methods to compute this quantity. One is using direct integration.

Since 

$$
||\widetilde{B}||_F^2 = \int \frac{(p(x,y) - p(x) p (y))^2}{p(x)p(y)}dxdy
$$

Where 
$$
p(x,y) = \frac{1}{2\pi}\exp\left(\frac{1}{2(1-\rho^2)}(x^2+y^2 - 2\rho x y)\right)
$$
The result is $\frac{\rho^2}{1-\rho^2}$.



The second method is to use the Taylor expansion of $\frac{p(x,y)}{p(x)p(y)} = 1 + \sum_{i=1}^{\infty} \rho \pi_i(x) \pi_i(y)$, where

$\pi_i(x)$ is orthogonal polynomial w.r.t. the Gaussian kernel. 

This expansion is possible due to [Mehler's kernel](https://en.wikipedia.org/wiki/Mehler_kernel#Probability_version).

Notice that
$$
||\widetilde{B}||_F^2 = \int \left(\frac{p(x,y)}{p(x)p(y)} - 1\right)^2p(x)p(y)dxdy
$$
Therefore we have  
$$
||\widetilde{B}||_F^2 = \sum_{i=1}^{\infty}\rho^{2i} = \frac{\rho^2}{1-\rho^2}
$$






