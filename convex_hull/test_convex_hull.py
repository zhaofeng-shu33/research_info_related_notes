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

def test_3():
    A = np.array([[-0.93914089,-0.05026096],
    [-0.66809079,-1.11161945],
    [ 0.89264514, 0.25255434],
    [-0.07901472,-1.1728504 ],
    [ 0.27926412,-0.28745604]])
    ch = ConvexHull(A)
    assert (len(ch.vertices) == 4)

def test_4():
    A = np.array([[-0.93914089,-0.05026096],
    [-0.66809079,-1.11161945],
    [ 0.89264514, 0.25255434],
    [-0.07901472,-1.1728504 ],
    [ 0.27926412,-0.28745604]])
    ch = ConvexHull(A, run=False)
    ch.run_2d_wrapper()
    assert (len(ch.vertices) == 4)