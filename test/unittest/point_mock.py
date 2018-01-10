from random import randint
import numpy as np
from copy import deepcopy

from combinatorial_space.cluster import Cluster
from test.unittest.cluster_mock import ClusterMockForPointWeight


class PointMockInOutCode:
    def __init__(self,
                 in_threshold_modify=5, out_threshold_modify=0,
                 in_threshold_activate=5, out_threshold_activate=0,
                 threshold_bin=0.1,
                 in_random_bits=24, out_random_bits=10,
                 count_in_demensions=256, count_out_demensions=16,
                 base_lr=0.01, is_modify_lr=True,
                 max_cluster_per_point=100,
                 cluster_class=Cluster):
        self.clusters = []
        self.count_in_demensions = count_in_demensions
        self.count_out_demensions = count_out_demensions

    def predict_front(self, in_code, type_code=-1):
        return np.array(in_code)[:2]

    def predict_back(self, out_code, type_code=-1):
        return np.array(out_code)[:2]

    def add(self, in_code, out_code):
        return 0, 0, False


class PointMockOddEven:
    def __init__(self,
                 in_threshold_modify=5, out_threshold_modify=0,
                 in_threshold_activate=5, out_threshold_activate=0,
                 threshold_bin=0.1,
                 in_random_bits=24, out_random_bits=10,
                 count_in_demensions=256, count_out_demensions=16,
                 base_lr=0.01, is_modify_lr=True,
                 max_cluster_per_point=100,
                 cluster_class=Cluster):
        self.clusters = []
        self.count_in_demensions = count_in_demensions
        self.count_out_demensions = count_out_demensions

    def predict_front(self, in_code, type_code=-1):
        if np.random.rand() > 0.5:
            return np.array([np.random.randint(-1, 2) for _ in np.arange(self.count_out_demensions)])
        else:
            return None

    def predict_back(self, out_code, type_code=-1):
        if np.random.rand() > 0.5:
            return np.array([np.random.randint(-1, 2) for _ in np.arange(self.count_in_demensions)])
        else:
            return None

    def add(self, in_code, out_code):
        return 0, 0, False


class PointMockAssertDem2:
    def __init__(self,
                 in_threshold_modify=5, out_threshold_modify=0,
                 in_threshold_activate=5, out_threshold_activate=0,
                 threshold_bin=0.1,
                 in_random_bits=24, out_random_bits=10,
                 count_in_demensions=256, count_out_demensions=16,
                 base_lr=0.01, is_modify_lr=True,
                 max_cluster_per_point=100,
                 cluster_class=Cluster):
        self.clusters = []
        self.count_in_demensions = count_in_demensions
        self.count_out_demensions = count_out_demensions

    def predict_front(self, in_code, type_code=-1):
        return [1] * 100

    def predict_back(self, out_code, type_code=-1):
        return [1] * 100

    def add(self, in_code, out_code):
        return 0, 0, False


class PointMockNone:
    def __init__(self,
                 in_threshold_modify=5, out_threshold_modify=0,
                 in_threshold_activate=5, out_threshold_activate=0,
                 threshold_bin=0.1,
                 in_random_bits=24, out_random_bits=10,
                 count_in_demensions=256, count_out_demensions=16,
                 base_lr=0.01, is_modify_lr=True,
                 max_cluster_per_point=100,
                 cluster_class=Cluster):
        self.clusters = []

    def predict_front(self, in_code, type_code=-1):
        return None

    def predict_back(self, out_code, type_code=-1):
        return None

    def add(self, in_code, out_code):
        return 0, 0, False


class PointMockZeros:
    def __init__(self,
                 in_threshold_modify=5, out_threshold_modify=0,
                 in_threshold_activate=5, out_threshold_activate=0,
                 threshold_bin=0.1,
                 in_random_bits=24, out_random_bits=10,
                 count_in_demensions=256, count_out_demensions=16,
                 base_lr=0.01, is_modify_lr=True,
                 max_cluster_per_point=100,
                 cluster_class=Cluster):
        self.clusters = []

    def predict_front(self, in_code, type_code=-1):
        return np.array([0] * len(in_code))

    def predict_back(self, out_code, type_code=-1):
        return np.array([0] * len(out_code))

    def add(self, in_code, out_code):
        return 0, 0, False


class PointMockDoubleIdentical:
    def __init__(self,
                 in_threshold_modify=5, out_threshold_modify=0,
                 in_threshold_activate=5, out_threshold_activate=0,
                 threshold_bin=0.1,
                 in_random_bits=24, out_random_bits=10,
                 count_in_demensions=256, count_out_demensions=16,
                 base_lr=0.01, is_modify_lr=True,
                 max_cluster_per_point=100,
                 cluster_class=Cluster):
        self.clusters = []

    def predict_front(self, in_code, type_code=-1):
        in_code[in_code == 0] = -1
        return np.array(np.concatenate((in_code, in_code)))

    def predict_back(self, out_code, type_code=-1):
        out_local = deepcopy(out_code)
        out_local[out_local == 0] = -1
        return out_local[:4]

    def add(self, in_code, out_code):
        return 0, 0, False


class PointMockCodeAligment:
    def __init__(self,
                 in_threshold_modify=5, out_threshold_modify=0,
                 in_threshold_activate=5, out_threshold_activate=0,
                 threshold_bin=0.1,
                 in_random_bits=24, out_random_bits=10,
                 count_in_demensions=256, count_out_demensions=16,
                 base_lr=0.01, is_modify_lr=True,
                 max_cluster_per_point=100,
                 cluster_class=Cluster):
        self.clusters = []

    def predict_front(self, in_code, type_code=-1):
        in_code[in_code == 0] = -1
        return np.array(np.concatenate((in_code, in_code)))

    def predict_back(self, out_code, type_code=-1):
        return None

    def add(self, in_code, out_code):
        return 0, 0, False


class PointMockControversyIn:
    def __init__(self,
                 in_threshold_modify=5, out_threshold_modify=0,
                 in_threshold_activate=5, out_threshold_activate=0,
                 threshold_bin=0.1,
                 in_random_bits=24, out_random_bits=10,
                 count_in_demensions=256, count_out_demensions=16,
                 base_lr=0.01, is_modify_lr=True,
                 max_cluster_per_point=100,
                 cluster_class=Cluster):
        self.clusters = []

    def predict_front(self, in_code, type_code=-1):
        in_local = deepcopy(in_code)
        in_local[in_local == 0] = -1
        return np.array(np.concatenate((in_local, in_local)))

    def predict_back(self, out_code, type_code=-1):
        out_local = deepcopy(out_code)
        out_local[out_local == 0] = -1
        if np.random.rand() > 0.5:
            out_local[0] = -1
        else:
            out_local[0] = 1
        return out_local[:4]

    def add(self, in_code, out_code):
        return 0, 0, False
