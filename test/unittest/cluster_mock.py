import numpy as np

class ClusterMock0None:
    def __init__(self,
                 base_in, base_out,
                 in_threshold_modify, out_threshold_modify,
                 threshold_bin,
                 base_lr,
                 is_modify_lr):
        pass

    def predict_front(self, in_x):
        return 0, None

    def predict_back(self, out_x):
        return 0, None

    def modify(self, in_x, out_x):
        return 0


class ClusterMockMinusNone:
    def __init__(self,
                 base_in, base_out,
                 in_threshold_modify, out_threshold_modify,
                 threshold_bin,
                 base_lr,
                 is_modify_lr):
        pass

    def predict_front(self, in_x):
        return -1, None

    def predict_back(self, out_x):
        return -1, None

    def modify(self, in_x, out_x):
        return -1


class ClusterMock1None:
    def __init__(self,
                 base_in, base_out,
                 in_threshold_modify, out_threshold_modify,
                 threshold_bin,
                 base_lr,
                 is_modify_lr):
        pass

    def predict_front(self, in_x):
        return 1, None

    def predict_back(self, out_x):
        return 1, None

    def modify(self, in_x, out_x):
        return 1


class ClusterMock1CustomBase:
    def __init__(self,
                 base_in, base_out,
                 in_threshold_modify, out_threshold_modify,
                 threshold_bin,
                 base_lr,
                 is_modify_lr):
        self.base_in = base_in
        self.base_out = base_out

    def predict_front(self, in_x):
        return 1, self.base_out

    def predict_back(self, out_x):
        return 1, self.base_in

    def modify(self, in_x, out_x):
        return 1


class ClusterMockGetDotCustomBase:
    def __init__(self,
                 base_in, base_out,
                 in_threshold_modify, out_threshold_modify,
                 threshold_bin,
                 base_lr,
                 is_modify_lr):
        self.base_in = base_in
        self.base_out = base_out

    def predict_front(self, in_x):
        return np.dot(self.base_in, in_x), self.base_out

    def predict_back(self, out_x):
        return np.dot(self.base_in, out_x), self.base_in

    def modify(self, in_x, out_x):
        return 1


class ClusterMockForPointWeight:
    def __init__(self,
                 base_in, base_out,
                 in_threshold_modify=5, out_threshold_modify=0,
                 threshold_bin=0.1,
                 base_lr=0.01,
                 is_modify_lr=True):
        self.in_w, self.out_w = base_in, base_out

    def predict_front(self, in_x):
        return self.in_w

    def predict_back(self, out_x):
        return self.out_w

    def modify(self, in_x, out_x):
        return 1
