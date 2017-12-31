# -*- encoding: utf-8 -*-

import unittest
import numpy as np
from combinatorial_space.minicolumn import Minicolumn
from test.unittest.point_mock import PointMockNone, PointMockOddEven


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

    def test_in_out_size(self):
        self.assertRaises(ValueError, Minicolumn, in_random_bits=-5)
        self.assertRaises(ValueError, Minicolumn, out_random_bits=-5)
        self.assertRaises(ValueError, Minicolumn, in_random_bits=None)
        self.assertRaises(ValueError, Minicolumn, out_random_bits=None)

    def test_base_lr(self):
        self.assertRaises(ValueError, Minicolumn, base_lr=-5)
        self.assertRaises(ValueError, Minicolumn, base_lr=None)

    def test_is_modify_lr(self):
        self.assertRaises(ValueError, Minicolumn, is_modify_lr=None)

    def test_count_demensions(self):
        self.assertRaises(ValueError, Minicolumn, count_in_demensions=-5)
        self.assertRaises(ValueError, Minicolumn, count_out_demensions=-5)
        self.assertRaises(ValueError, Minicolumn, count_in_demensions=None)
        self.assertRaises(ValueError, Minicolumn, count_out_demensions=None)

    def test_threshold_bits_controversy(self):
        self.assertRaises(ValueError, Minicolumn, threshold_bits_controversy=-5)
        self.assertRaises(ValueError, Minicolumn, threshold_bits_controversy=None)

    def test_out_non_zero_bits(self):
        self.assertRaises(ValueError, Minicolumn, out_non_zero_bits=-5)
        self.assertRaises(ValueError, Minicolumn, out_non_zero_bits=None)

    def test_class_point(self):
        self.assertRaises(ValueError, Minicolumn, class_point=None)

    def test_size_space(self):
        space_size = 100
        minicolumn = Minicolumn(space_size=space_size, class_point=PointMockNone)
        self.assertEqual(len(minicolumn.space), space_size)

    def test_size_random_bits_count_demensions(self):
        self.assertRaises(ValueError, Minicolumn, in_random_bits=11, count_in_demensions=10)
        self.assertRaises(ValueError, Minicolumn, out_random_bits=11, count_out_demensions=10)


class TestPointPredict(unittest.TestCase):
    def setUp(self):
        self.minicolumn_none = Minicolumn(
            in_random_bits=1,
            out_random_bits=1,
            count_in_demensions=1,
            count_out_demensions=1,
            class_point=PointMockNone
        )
        self.minicolumn = Minicolumn(
            space_size=20,
            in_random_bits=10,
            out_random_bits=10,
            count_in_demensions=10,
            count_out_demensions=10,
            seed=41,
            threshold_bits_controversy=0.05,
            class_point=PointMockOddEven
        )

    def test_front_assert_not_active_point(self):
        self.assertRaises(AssertionError, self.minicolumn_none.front_predict, [1])

    def test_back_assert_not_active_point(self):
        self.assertRaises(AssertionError, self.minicolumn_none.back_predict, [1])

    def test_front_assert_active_point(self):
        controversy, code = self.minicolumn.front_predict([1] * 10)
        np.testing.assert_array_equal(np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0]), code)
        self.assertEqual(2, controversy)

    def test_back_assert_active_point(self):
        controversy, code = self.minicolumn.back_predict([1] * 10)
        np.testing.assert_array_equal(np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0]), code)
        self.assertEqual(2, controversy)


class TestPointSleep(unittest.TestCase):
    def setUp(self):
        self.minicolumn = Minicolumn(
            in_random_bits=1,
            out_random_bits=1,
            count_in_demensions=1,
            count_out_demensions=1,
            class_point=PointMockNone
        )


if __name__ == '__main__':
    unittest.main()