from combinatorial_space.cluster import Cluster
from combinatorial_space.cluster_mock import ClusterMock0


class ClusterFactory:
    def __init__(self, mode='build'):
        self.__mode = mode

    def get(self):
        if self.__mode == 'build':
            return Cluster
        elif self.__mode == 'mock0':
            return ClusterMock0