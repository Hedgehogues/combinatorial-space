from combinatorial_space.cluster import Cluster


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
