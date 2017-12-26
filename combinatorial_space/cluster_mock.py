class ClusterMock0:
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
