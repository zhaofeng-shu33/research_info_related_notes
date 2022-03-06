import argparse
import numpy as np
from scipy.optimize import fsolve
from scipy.spatial import ConvexHull
from scipy.integrate import dblquad
from scipy.special import gamma
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
NTRIAL = 1000
DISTRIBUTION = 'unit_circle'
DOF = 1.0

def random_points_2d_cauchy(n):
    y = np.random.random(n)
    r = np.sqrt(2 * y - y * y) / (1 - y)
    theta = np.random.random(n)
    theta *= 2 * np.pi
    return np.vstack((r * np.cos(theta), r * np.sin(theta))).T

def random_points_2d_t_distribution(n, v):
    y = np.random.random(n)
    C = 2 / v * gamma(v/2 + 1) / gamma(v/2)
    r = np.sqrt(np.power(1-y/C, -2/v) - 1)
    theta = np.random.random(n)
    theta *= 2 * np.pi
    return np.vstack((r * np.cos(theta), r * np.sin(theta))).T

def random_points_3d_cauchy(n):
    y = np.random.random(n)
    r = np.zeros(n)
    for i in range(n):
        f = lambda r: (-r/(2*(r*r+1))+np.arctan(r)/2)*4/np.pi - y[i]
        r[i] = fsolve(f, [1.0])[0]
    phi = 2 * np.pi * np.random.random(n)
    theta = np.pi * np.random.random(n)
    return np.vstack((r * np.cos(phi) * np.sin(theta),
                      r * np.sin(phi) * np.sin(theta),
                      r * np.cos(theta))).T

def random_points_in_unit_circle(n):
    r = np.random.random(n)
    r = np.sqrt(r)
    theta = np.random.random(n)
    theta *= 2 * np.pi
    return np.vstack((r * np.cos(theta), r * np.sin(theta))).T

def transform(n_list):
    if DISTRIBUTION == 'unit_circle':
        return n_list ** (1/3)
    elif DISTRIBUTION == 'gaussian':
        return np.sqrt(np.log(n_list))
    else:
        return n_list

def countVertex(n):
    if DISTRIBUTION == 'unit_circle':
        points = random_points_in_unit_circle(n)
    elif DISTRIBUTION == 'gaussian':
        points = np.random.normal(size=(n, 2))
    elif DISTRIBUTION == '2d-cauchy':
        points = random_points_2d_cauchy(n)
    elif DISTRIBUTION == '3d-cauchy':
        points = random_points_3d_cauchy(n)
    elif DISTRIBUTION == '2d-t-distribution':
        global DOF
        points = random_points_2d_t_distribution(n, DOF)
    else:
        raise ValueError("")
    hull = ConvexHull(points)
    return hull.nsimplex # number of (d-1) faces

def testN(n, nTrial=20):
    return np.array( [ countVertex(n) for i in range(nTrial) ])

def testAllN(n_list):
    
    nN = len(n_list)
    result = np.zeros(nN)
    for i, n in enumerate(n_list):
        print(i)
        result[i] = np.mean(testN(n, NTRIAL))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution',
            choices=['unit_circle', 'gaussian',
                 '2d-cauchy', '3d-cauchy', '2d-t-distribution'],
            default='unit_circle')
    parser.add_argument('--dof', help='degree of freedom for t distribution', default=1, type=float)
    args = parser.parse_args()
    DISTRIBUTION = args.distribution
    DOF = args.dof
    n_list = np.array(range(5, 100, 5))
    result = testAllN(n_list)
    transformed_n_list = transform(n_list)
    if args.distribution.find('cauchy') < 0:
        model = LinearRegression()        
        reg = model.fit(transformed_n_list.reshape(-1, 1), result)
        print(reg.coef_, reg.intercept_)
        print(reg.score(transformed_n_list.reshape(-1, 1), result))
    plt.plot(transformed_n_list, result)
    const_value = -1
    if args.distribution == '2d-cauchy':
        const_value = np.pi ** 2 / 2
    elif args.distribution == '3d-cauchy':
        const_value = 4 * np.pi ** 2 / 3
    elif args.distribution == '2d-t-distribution':
        v = DOF
        _int = dblquad(lambda x, y: np.abs(np.tan(x) - np.tan(y)) * np.power(np.cos(x) * np.cos(y), v),
            -np.pi/2, np.pi/2, -np.pi/2, np.pi/2)[0]
        const_value = 4 * np.sqrt(np.pi) * gamma(v+ 1/2) / gamma(v+1) * gamma(v/2 + 1) ** 2 / gamma((v+1)/2) ** 2
    if const_value > 0:
        plt.plot([0, 100], [const_value, const_value], color='red')

    plt.xlabel('N')
    plt.ylabel('$E(F_N)$')
    plt.title(args.distribution)
    plt.savefig(f'build/{args.distribution}.pdf')
    plt.show()
