import numpy as np
from scipy.spatial import ConvexHull as ConvexHullRef

class ConvexHull:
    def __init__(self, coordinates):
        # coordinates are numpy array, we assume no three points are colinear
        self.simplices = []
        self.vertices = []
        self._coordinates = coordinates
        self.ndim = len(coordinates[0, :])
        if self.ndim == 2:
            self.run_2d()

    def run_2d(self):
        # find the vertex and facial representation in 2D

        # make sure the rotational direction is right-hand
        if self.check_left(0, 1, self._coordinates[2, :]):
            self.simplices = [[0, 1], [1, 2], [2, 0]]
        else:
            self.simplices = [[0, 2], [2, 1], [1, 0]]
        # incremental construction
        for i in range(3, len(self._coordinates)):
            # check whether the new point is outside of the existing convex hull
            if self.check_out(self._coordinates[i, :]):
                new_simplices = []
                start_found = False
                for indices in self.simplices:
                    if not self.check_left(indices[0], indices[1], self._coordinates[i, :]):
                        if not start_found:
                            start_index = indices[0]
                            end_index = indices[1]
                            start_found = True
                        else:
                            end_index = indices[1]
                    else:
                        new_simplices.append(indices)
                new_simplices.append([start_index, i])
                new_simplices.append([i, end_index])
                self.simplices = new_simplices
        for indices in self.simplices:
            self.vertices.append(indices[0])

    def check_left(self, i, j, coordinate):
        # check whether the point lies on the left-hand side of the vector i->j
        a = self._coordinates[j, :] - self._coordinates[i, :]
        b = coordinate - self._coordinates[i, :]
        if a[0] * b[1] > a[1] * b[0]:
            return True
        return False

    def check_out(self, coordinate):
        for indices in self.simplices:
            if not self.check_left(indices[0], indices[1], coordinate):
                return True
        return False


if __name__ == '__main__':
    np.random.seed(122)
    A = np.random.randn(5, 2)
    ch = ConvexHull(A)
    print(ch.simplices)
    print(ch.vertices)
    ch_ref = ConvexHullRef(A)
    print(ch_ref.simplices)
    print(ch_ref.vertices)
