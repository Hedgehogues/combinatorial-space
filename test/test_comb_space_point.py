# -*- encoding: utf-8 -*-

import unittest

import numpy as np

from combinatorial_space.cluster_factory import ClusterFactory
from combinatorial_space.point import Point


class Test_comb_space_cluster(unittest.TestCase):
    # Вызывается перед каждым тестом
    def setUp(self):

        self.base_point_a = Point(
            in_threshold_modify=1, out_threshold_modify=1,
            in_threshold_activate=1, out_threshold_activate=1,
            threshold_bin=1,
            in_size=1, out_size=1,
            count_in_demensions=1, count_out_demensions=1,
            base_lr=0, is_modify_lr=True,
            max_cluster_per_point=1,
            cluster_factory=ClusterFactory('mock0')
        )

        self.base_point_b = Point(
            in_threshold_modify=1, out_threshold_modify=1,
            in_threshold_activate=0, out_threshold_activate=0,
            threshold_bin=1,
            in_size=1, out_size=1,
            count_in_demensions=1, count_out_demensions=1,
            base_lr=0, is_modify_lr=True,
            max_cluster_per_point=1,
            cluster_factory=ClusterFactory('mock0')
        )

    # TODO: Протестировать конструктор

    def test_predict_front_type_code_not_a_valid(self):
        self.assertRaises(ValueError, self.base_point_a.predict_front, [1], 1)

    def test_predict_front_not_activate(self):
        self.assertRaises(AssertionError, self.base_point_a.predict_front, [-1] * 1 + [0] * 9)

    def test_predict_back_type_code_not_a_valid(self):
        self.assertRaises(ValueError, self.base_point_a.predict_back, [1], 1)

    def test_predict_back_not_activate(self):
        self.assertRaises(AssertionError, self.base_point_a.predict_back, [-1] * 1 + [0] * 9)

    def test_predict_back_not_active(self):
        opt_in_code = self.base_point_a.predict_back([1])
        self.assertIsNone(opt_in_code, None)

    def test_predict_front_not_active(self):
        opt_in_code = self.base_point_a.predict_front([1])
        self.assertIsNone(opt_in_code, None)

    def test_predict_back_active_empty_cluster(self):
        opt_in_code = self.base_point_b.predict_back([1])
        self.assertIsNone(opt_in_code, None)

    def test_predict_front_active_empty_cluster(self):
        opt_in_code = self.base_point_b.predict_front([1])
        self.assertIsNone(opt_in_code, None)

if __name__ == '__main__':
    unittest.main()