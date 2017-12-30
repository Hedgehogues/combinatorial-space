# -*- encoding: utf-8 -*-

import unittest
import test.unittest.cluster_mock as cluster_mock
from combinatorial_space.minicolumn import Minicolumn
from test.unittest.point_mock import PointMockNone


class TestMinicolumn__init__(unittest.TestCase):
    def test_space_size(self):
        self.assertRaises(ValueError, Minicolumn, space_size=-10)
        self.assertRaises(ValueError, Minicolumn, space_size=None)

    def test_max_cluster_per_point(self):
        self.assertRaises(ValueError, Minicolumn, max_cluster_per_point=-100)
        self.assertRaises(ValueError, Minicolumn, max_cluster_per_point=None)

    def test_max_count_clusters(self):
        self.assertRaises(ValueError, Minicolumn, max_count_clusters=-1000000)
        self.assertRaises(ValueError, Minicolumn, max_count_clusters=None)

    def test_seed(self):
        self.assertRaises(ValueError, Minicolumn, seed=None)

    def test_threshold_modify(self):
        self.assertRaises(ValueError, Minicolumn, in_threshold_modify=-5)
        self.assertRaises(ValueError, Minicolumn, in_threshold_modify=None)
        self.assertRaises(ValueError, Minicolumn, out_threshold_modify=-5)
        self.assertRaises(ValueError, Minicolumn, out_threshold_modify=None)

    def test_threshold_activate(self):
        self.assertRaises(ValueError, Minicolumn, in_threshold_activate=-5)
        self.assertRaises(ValueError, Minicolumn, in_threshold_activate=None)
        self.assertRaises(ValueError, Minicolumn, out_threshold_activate=-5)
        self.assertRaises(ValueError, Minicolumn, out_threshold_activate=None)

    def test_threshold_bin(self):
        self.assertRaises(ValueError, Minicolumn, threshold_bin=-5)
        self.assertRaises(ValueError, Minicolumn, threshold_bin=None)

class TestPointBase(unittest.TestCase):
    def setUp(self):
        self.base_point_a = Point(
            in_threshold_modify=1, out_threshold_modify=1,
            in_threshold_activate=1, out_threshold_activate=1,
            threshold_bin=1,
            in_size=1, out_size=1,
            count_in_demensions=1, count_out_demensions=1,
            base_lr=0, is_modify_lr=True,
            max_cluster_per_point=1,
            cluster_class=cluster_mock.ClusterMock0None
        )

if __name__ == '__main__':
    unittest.main()