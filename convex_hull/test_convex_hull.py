import numpy as np
from convex_hull import ConvexHull

def test_1():
    A = np.array([[0, 0], [1, 0], [0.5, 1], [0.5, -1]])
    ch = ConvexHull(A)
    assert len(ch.simplices) == 4

def test_2():
    A = np.array([[0, 0], [1, 0], [0.5, 1], [1.5, -1]])
    ch = ConvexHull(A)
    assert len(ch.vertices) == 3
