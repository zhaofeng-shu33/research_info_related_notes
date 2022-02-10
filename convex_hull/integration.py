from sympy import symbols,integrate,sin,cos,pi
x = symbols('x')
a1 = integrate(x*sin(x)*cos(x)**5,(x,0,pi/2))
a2 = integrate(cos(x)**6,(x,0,pi/2))
a3 = integrate(cos(x)**8,(x,0,pi/2))
a1 + a2 - a3 / 3
