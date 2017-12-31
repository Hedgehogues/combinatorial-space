from random import randint
import numpy as np
from combinatorial_space.cluster import Cluster


class PointMockOddEven:
    def __init__(self,
                 in_threshold_modify, out_threshold_modify,
                 in_threshold_activate, out_threshold_activate,
                 threshold_bin,
                 in_size, out_size,
                 count_in_demensions, count_out_demensions,
                 base_lr, is_modify_lr,
                 max_cluster_per_point,
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


class PointMockNone:
    def __init__(self,
                 in_threshold_modify, out_threshold_modify,
                 in_threshold_activate, out_threshold_activate,
                 threshold_bin,
                 in_size, out_size,
                 count_in_demensions, count_out_demensions,
                 base_lr, is_modify_lr,
                 max_cluster_per_point,
                 cluster_class=Cluster):
        self.clusters = []

    def predict_front(self, in_code, type_code=-1):
        return None

    def predict_back(self, out_code, type_code=-1):
        return None

    def add(self, in_code, out_code):
        return 0, 0, False
