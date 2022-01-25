import numpy as np
for i in range(100):
    _weight = np.sort(-1.0 * np.random.random(3))
    _weight *= -1.0
    v1 = np.random.normal(size=3)
    v2 = np.random.normal(size=3)
    w = np.zeros(3)
    w[0] = v1[1] * v2[2] - v1[2] * v2[1]
    w[1] = -(v1[0] * v2[2] - v1[2] * v2[0])
    w[2] = v1[0] * v2[1] - v1[1] * v2[0]
    u = np.array([-w[1], w[0]])
    u = u / np.linalg.norm(u)
    if w[2] < 0:
        cos_theta = -w[2] / np.linalg.norm(w)
    else:
        u = -u
        cos_theta = w[2] / np.linalg.norm(w)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    # construct the rotational matrix
    R = np.zeros([3, 3])
    R[0, 0] = cos_theta + u[0] ** 2 * (1 - cos_theta)
    R[0, 1] = u[0] * u[1] * (1 - cos_theta)
    R[0, 2] = u[1] * sin_theta
    R[1, 0] = u[0] * u[1] * (1 - cos_theta)
    R[1, 1] = cos_theta + u[1] ** 2 * (1 - cos_theta)
    R[1, 2] = -1.0 * u[0] * sin_theta
    R[2, 0] = -u[1] * sin_theta
    R[2, 1] = u[0] * sin_theta
    R[2, 2] = cos_theta

    v_proj = R @ (v1 - v2)
    len_1 = np.linalg.norm(_weight * (v1 - v2))
    len_2 = np.linalg.norm(_weight *v_proj)
    if len_1 > len_2:
        import pdb
        pdb.set_trace()