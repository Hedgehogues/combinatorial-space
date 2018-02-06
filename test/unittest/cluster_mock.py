import numpy as np

from src.combinatorial_space.enums import CLUSTER


class ClusterMockNoneNotActive:
    def __init__(self,
                 in_sub_code=None, out_sub_code=None,
                 in_cluster_modify=None, out_cluster_modify=None,
                 binarization=None,
                 lr=None,
                 is_modify_lr=None):
        pass

    def predict_front(self, in_x):
        return 0, None, CLUSTER.NOT_ACTIVE

    def predict_back(self, out_x):
        return 0, None, CLUSTER.NOT_ACTIVE

    def modify(self, in_x, out_x):
        pass


class ClusterMock1None:
    def __init__(self,
                 in_sub_code=None, out_sub_code=None,
                 in_cluster_modify=None, out_cluster_modify=None,
                 binarization=None,
                 lr=None,
                 is_modify_lr=None):
        pass

    def predict_front(self, in_x):
        pass

    def predict_back(self, out_x):
        pass

    def modify(self, in_x, out_x):
        return CLUSTER.MODIFY


class ClusterMockCustom:
    def __init__(self,
                 in_sub_code=None, out_sub_code=None,
                 in_cluster_modify=None, out_cluster_modify=None,
                 binarization=None,
                 lr=None,
                 is_modify_lr=None):
        self.in_sub_code = in_sub_code
        self.out_sub_code = out_sub_code

    def predict_front(self, in_x):
        return 1, self.out_sub_code, CLUSTER.ACTIVE

    def predict_back(self, out_x):
        return 1, self.in_sub_code, CLUSTER.ACTIVE

    def modify(self, in_x, out_x):
        return CLUSTER.MODIFY


class ClusterMockCustomDot:
    def __init__(self,
                 in_sub_code=None, out_sub_code=None,
                 in_cluster_modify=None, out_cluster_modify=None,
                 binarization=None,
                 lr=None,
                 is_modify_lr=None):
        self.in_sub_code = in_sub_code
        self.out_sub_code = out_sub_code

    def predict_front(self, in_x):
        return np.dot(self.in_sub_code, in_x), self.out_sub_code, CLUSTER.ACTIVE

    def predict_back(self, out_x):
        return np.dot(self.in_sub_code, out_x), self.in_sub_code, CLUSTER.ACTIVE

    def modify(self, in_x, out_x):
        return CLUSTER.MODIFY


class ClusterMockWeight:
    def __init__(self,
                 in_sub_code, out_sub_code,
                 in_cluster_modify=5, out_cluster_modify=0,
                 binarization=0.1,
                 lr=0.01,
                 is_modify_lr=True):
        self.in_w, self.out_w = in_sub_code, out_sub_code

    def predict_front(self, in_x):
        return self.in_w

    def predict_back(self, out_x):
        return self.out_w

    def modify(self, in_x, out_x):
        return 1
