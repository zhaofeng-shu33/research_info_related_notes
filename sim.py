import argparse

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

p_0 = 0.5
p_1 = 0.4

def f(a):
    C_11 = np.power(p_1, 1 - a) * np.power(p_0, a)
    C_12 = np.power(1 - p_1, 1 - a) * np.power(1 - p_0, a)
    C_21 = np.power(p_0, 1 - a) * np.power(p_1, a)
    C_22 = np.power(1 - p_0, 1 - a) * np.power(1 - p_1, a)
    res_1 = C_11 * np.log(p_0 / p_1)
    res_1 += C_12 * np.log((1 - p_0) / (1 - p_1))
    res_1 /= (C_11 + C_12)
    res_2 = C_21 * np.log(p_1 / p_0)
    res_2 += C_22 * np.log((1 - p_1) / (1 - p_0))
    res_2 /= (C_21 + C_22)
    res_3 = -1 * np.log(C_11 + C_12) - np.log(C_21 + C_22)
    return a * res_1 + a * res_2 +  res_3

def g(a):
    res = 2 * a * np.log(a / p_0)
    res += 2 * (1 - a) * np.log((1 - a) / (1 - p_0))
    res -= a * np.log(a / p_1)
    res -= (1 - a) * np.log((1 - a) / (1 - p_1))
    return res

def get_marginal_probability(x, p, q, y1=None):
    total_sum = 0
    for i in range(16):
        val = [int(j) for j in bin(i).lstrip('0b').zfill(4)]
        if val[1] == 0:
            continue
        if y1 is not None and val[0] != y1:
            continue
        # six terms
        val_base = []
        for i in range(4):
            for j in range(i + 1, 4):
                equal = int(val[i] == val[j])
                val_base.append(equal * p + (1 - equal) * q)        
        _sum = 1
        for i in range(6):
            _sum *= np.power(val_base[i], x[i]) * np.power(1 - val_base[i], 1 - x[i])
        total_sum += _sum
    if y1 is None:
        return total_sum / 8
    else:
        return total_sum / 4
    
def sbm_graph(n, k, a, b):
    if n % k != 0:
        raise ValueError('n %k != 0')
    elif a <= b:
        raise ValueError('a <= b')
    sizes = [int(n/k) for _ in range(k)]
    _p = np.log(n) * a / n
    _q = np.log(n) * b / n
    if _p > 1 or _q > 1:
        raise ValueError('')
    p = np.diag(np.ones(k) * (_p - _q)) + _q * np.ones([k, k])
    return nx.generators.community.stochastic_block_model(sizes, p)

def _get_embedding(affinity_matrix_, n_clusters=2):
    '''
    get low dimension features from embedded representation of data
    by taking the first k eigenvectors.
    k should be equal to self.n_clusters

    Parameters
    ----------
    norm_laplacian: bool, optional, default=False
        If True, then compute normalized Laplacian.

    Returns
    -------
    embedded feature:
        np.array, shape (num_samples, k)
    '''
    # start of your modification
    n = affinity_matrix_.shape[0]
    # compute the unnormalized Laplacian
    D = np.sum(affinity_matrix_, axis=0)
    L =  np.diag(D) - affinity_matrix_
    values, vectors = np.linalg.eig(L)
    Ls = [[i, values[i]] for i in range(n)]
    Ls.sort(key=lambda x:x[1])
    k = n_clusters
    selected_array = [Ls[i][0] for i in range(k)]
    print(Ls[1][1], Ls[2][1])
    return vectors[:, selected_array]

def get_first_embedding(x_list):
    x12, x13, x14, x23, x24, x34 = x_list
    A = np.zeros([4, 4])
    if x12 == 1:
        A[0, 1] = 1
    if x13 == 1:
        A[0, 2] = 1
    if x14 == 1:
        A[0, 3] = 1
    if x23 == 1:
        A[1, 2] = 1
    if x24 == 1:
        A[1, 3] = 1
    if x34 == 1:
        A[2, 3] = 1
    A = A + A.T
    n = 4
    D = np.sum(A, axis=0)
    L =  np.diag(D) - A
    values, vectors = np.linalg.eig(L)
    Ls = [[i, values[i]] for i in range(n)]
    Ls.sort(key=lambda x:x[1])
    if abs(Ls[2][1] - Ls[1][1]) < 1e-4:
        return 0
    if abs(Ls[0][1] - Ls[1][1]) > 1e-4:
        v = vectors[:, Ls[1][0]]
        v /= v[1]
        v /= np.linalg.norm(v)
        if np.abs(v[0]) < 1e-6:
            return 0        
        return v[0]
    v = vectors[:, Ls[1][0]]
    result = np.dot(v, np.array([1, 1, 1, 1]))
    if np.abs(result) < 1e-6:
        v_true = v
    else:
        alpha = -4 / result
        v_true = np.array([1, 1, 1, 1]) + alpha * v
    v_true /= v_true[1]
    v_true /= np.linalg.norm(v_true)
    if np.abs(v_true[0]) < 1e-6:
        return 0
    return v_true[0]

def exact_compare(labels):
    # return 1 if labels = X or -X
    n2 = int(len(labels) / 2)
    labels_inner = np.array(labels)
    return np.sum(labels_inner) == 0 and np.abs(np.sum(labels_inner[:n2])) == n2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectral', default=False, const=True, type=bool, nargs='?')
    args = parser.parse_args()
    if args.spectral:
        a = 11.6
        b = 4
        n = 200
        G = sbm_graph(n, 2, a, b)
        # print(b * np.log(n))
        A = np.zeros([n, n])
        for u,v in G.edges():
            A[u, v] = 1
        A = A + A.T
        features = _get_embedding(A)
        plt.scatter(features[:, 0], features[:, 1])
        labels = np.asarray(features[:, 1] > 0, dtype=int)
        labels = 2 * labels - 1
        print(exact_compare(labels))
        plt.show()
    else:
        p = 0.7
        q = 0.3
        prob = 0
        f_val = []
        prob_val = []
        for i in range(64):
            val = [int(j) for j in bin(i).lstrip('0b').zfill(6)]
            prob_val.append(get_marginal_probability(val, p, q))
            f_val.append(get_first_embedding(val))
        f_val = np.array(f_val)
        mean_val = np.dot(f_val, prob_val)
        std_var = np.sqrt(np.dot(f_val ** 2, prob_val) - mean_val ** 2)
        f_val = (f_val - mean_val) / std_var

        _prob_vector = []
        for i in range(64):
            val = [int(j) for j in bin(i).lstrip('0b').zfill(6)]
            _prob = get_marginal_probability(val, p, q, y1=1)
            _prob -= get_marginal_probability(val, p, q, y1=0)
            _prob /= 2
            _prob_vector.append(_prob)
        _prob_vector = np.array(_prob_vector)
        print(np.dot(_prob_vector, _prob_vector) / np.sqrt(np.dot(_prob_vector ** 2, prob_val)))
        print(np.dot(_prob_vector, f_val))