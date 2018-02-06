# -*- encoding: utf-8 -*-

import unittest
import numpy as np
from src.combinatorial_space.minicolumn import Minicolumn, MINICOLUMN_LEARNING
from test.unittest.cluster_mock import ClusterMockWeight
from test.unittest.point_mock import PointMockNone, PointMockOddEven, PointMockZeros, \
    PointMockDoubleIdentical, PointMockCodeAligment, PointMockControversyIn, PointMockAssertDim2


class TestMinicolumnConstructor(unittest.TestCase):
    def test_code_alignment_count_dim(self):
        self.assertRaises(
            ValueError, Minicolumn,
            code_alignment=3, in_dimensions=4, out_dimensions=2,
            in_random_bits=1, out_random_bits=1
        )
        self.assertRaises(
            ValueError, Minicolumn,
            code_alignment=3, in_dimensions=3, out_dimensions=2,
            in_random_bits=1, out_random_bits=1
        )
        Minicolumn(
            space_size=10,
            code_alignment=2, in_dimensions=4, out_dimensions=4,
            in_random_bits=1, out_random_bits=1
        )

    def test_space_size(self):
        self.assertRaises(ValueError, Minicolumn, space_size=-10)
        self.assertRaises(ValueError, Minicolumn, space_size=None)

    def test_max_clusters_per_point(self):
        self.assertRaises(ValueError, Minicolumn, max_clusters_per_point=-100)
        self.assertRaises(ValueError, Minicolumn, max_clusters_per_point=None)

    def test_max_clusters(self):
        self.assertRaises(ValueError, Minicolumn, max_clusters=-1000000)
        self.assertRaises(ValueError, Minicolumn, max_clusters=None)

    def test_seed(self):
        self.assertRaises(ValueError, Minicolumn, seed=None)

    def test_threshold_modify(self):
        self.assertRaises(ValueError, Minicolumn, in_cluster_modify=-5)
        self.assertRaises(ValueError, Minicolumn, in_cluster_modify=None)
        self.assertRaises(ValueError, Minicolumn, out_cluster_modify=-5)
        self.assertRaises(ValueError, Minicolumn, out_cluster_modify=None)

    def test_threshold_activate(self):
        self.assertRaises(ValueError, Minicolumn, in_point_activate=-5)
        self.assertRaises(ValueError, Minicolumn, in_point_activate=None)
        self.assertRaises(ValueError, Minicolumn, out_point_activate=-5)
        self.assertRaises(ValueError, Minicolumn, out_point_activate=None)

    def test_binarization(self):
        self.assertRaises(ValueError, Minicolumn, binarization=-5)
        self.assertRaises(ValueError, Minicolumn, binarization=None)

    def test_in_out_size(self):
        self.assertRaises(ValueError, Minicolumn, in_random_bits=-5)
        self.assertRaises(ValueError, Minicolumn, out_random_bits=-5)
        self.assertRaises(ValueError, Minicolumn, in_random_bits=None)
        self.assertRaises(ValueError, Minicolumn, out_random_bits=None)

    def test_lr(self):
        self.assertRaises(ValueError, Minicolumn, lr=-5)
        self.assertRaises(ValueError, Minicolumn, lr=None)

    def test_is_modify_lr(self):
        self.assertRaises(ValueError, Minicolumn, is_modify_lr=None)

    def test_count_dimensions(self):
        self.assertRaises(ValueError, Minicolumn, in_dimensions=-5)
        self.assertRaises(ValueError, Minicolumn, out_dimensions=-5)
        self.assertRaises(ValueError, Minicolumn, in_dimensions=None)
        self.assertRaises(ValueError, Minicolumn, out_dimensions=None)

    def test_controversy(self):
        self.assertRaises(ValueError, Minicolumn, controversy=-5)
        self.assertRaises(ValueError, Minicolumn, controversy=None)

    def test_code_alignment(self):
        self.assertRaises(ValueError, Minicolumn, code_alignment=-5)
        self.assertRaises(ValueError, Minicolumn, code_alignment=None)

    def test_class_point(self):
        self.assertRaises(ValueError, Minicolumn, class_point=None)

    def test_size_space(self):
        space_size = 100
        minicolumn = Minicolumn(space_size=space_size, class_point=PointMockNone)
        self.assertEqual(len(minicolumn.space), space_size)

    def test_size_random_bits_count_dimensions(self):
        self.assertRaises(ValueError, Minicolumn, in_random_bits=11, in_dimensions=10)
        self.assertRaises(ValueError, Minicolumn, out_random_bits=11, out_dimensions=10)

    def test_count_active_point(self):
        self.assertRaises(ValueError, Minicolumn, min_active_points=-1)
        self.assertRaises(ValueError, Minicolumn, min_active_points=None)


class TestPointPredict(unittest.TestCase):
    def setUp(self):
        self.minicolumn_none = Minicolumn(
            space_size=100,
            in_random_bits=1,
            out_random_bits=1,
            in_dimensions=1,
            out_dimensions=1,
            code_alignment=1,
            class_point=PointMockNone
        )
        self.minicolumn = Minicolumn(
            space_size=20,
            in_random_bits=10,
            out_random_bits=10,
            in_dimensions=10,
            out_dimensions=10,
            seed=41,
            controversy=0.05,
            code_alignment=1,
            min_active_points=0,
            class_point=PointMockOddEven
        )
        self.minicolumn_front = Minicolumn(
            space_size=20,
            in_random_bits=10,
            out_random_bits=2,
            in_dimensions=10,
            out_dimensions=2,
            seed=41,
            controversy=0.05,
            code_alignment=1,
            class_point=PointMockOddEven
        )
        self.minicolumn_back = Minicolumn(
            space_size=20,
            in_random_bits=2,
            out_random_bits=10,
            in_dimensions=2,
            out_dimensions=10,
            seed=41,
            controversy=0.05,
            code_alignment=1,
            class_point=PointMockOddEven
        )
        self.minicolumn_assert_dim_2 = Minicolumn(
            space_size=20,
            in_random_bits=2,
            out_random_bits=10,
            in_dimensions=2,
            out_dimensions=10,
            seed=41,
            controversy=0.05,
            code_alignment=1,
            class_point=PointMockAssertDim2
        )

    def test_front_assert_not_valid_value(self):
        self.assertRaises(ValueError, self.minicolumn_front.front_predict, [-1] * 10)
        self.assertRaises(ValueError, self.minicolumn_front.front_predict, [2] * 10)
        self.assertRaises(ValueError, self.minicolumn_front.front_predict, [0.8] * 10)
        self.assertRaises(ValueError, self.minicolumn_front.front_predict, None)

    def test_back_assert_not_valid_value(self):
        self.assertRaises(ValueError, self.minicolumn_back.back_predict, [-1] * 10)
        self.assertRaises(ValueError, self.minicolumn_back.back_predict, [2] * 10)
        self.assertRaises(ValueError, self.minicolumn_back.back_predict, [0.8] * 10)
        self.assertRaises(ValueError, self.minicolumn_back.back_predict, None)

    def test_front_assert_not_valid_dim(self):
        self.assertRaises(AssertionError, self.minicolumn_front.front_predict, [1])

    def test_back_assert_not_valid_dim(self):
        self.assertRaises(AssertionError, self.minicolumn_back.back_predict, [1])

    def test_front_assert_not_active_point(self):
        code = [1]
        controversy, out_code, accept = self.minicolumn_none.front_predict(code)
        self.assertEqual([1], code)
        self.assertEqual(None, out_code)
        self.assertEqual(None, controversy)
        self.assertEqual(MINICOLUMN_LEARNING.INACTIVE_POINTS, accept)

    def test_back_assert_not_active_point(self):
        code = [1]
        controversy, in_code, accept = self.minicolumn_none.back_predict(code)
        self.assertEqual([1], code)
        self.assertEqual(None, in_code)
        self.assertEqual(None, controversy)
        self.assertEqual(MINICOLUMN_LEARNING.INACTIVE_POINTS, accept)

    def test_front_not_valid_dim(self):
        self.assertRaises(AssertionError, self.minicolumn_assert_dim_2.front_predict, [1] * 3)

    def test_back_assert_not_valid_dim(self):
        self.assertRaises(AssertionError, self.minicolumn_assert_dim_2.back_predict, [1] * 8)

    def test_front_assert_there_are(self):
        code = [1]
        controversy, out_code, accept = self.minicolumn_none.front_predict(code)
        self.assertEqual([1], code)
        self.assertEqual(None, out_code)
        self.assertEqual(None, controversy)
        self.assertEqual(MINICOLUMN_LEARNING.INACTIVE_POINTS, accept)

    def test_front_active_point(self):
        in_code = [1] * 10
        controversy, out_code, accept = self.minicolumn.front_predict(in_code)
        self.assertEqual([1] * 10, in_code)
        np.testing.assert_array_equal(np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0]), out_code)
        self.assertEqual(2, controversy)
        self.assertEqual(MINICOLUMN_LEARNING.ACCEPT, accept)

    def test_back_active_point(self):
        out_code = [1] * 10
        controversy, code, accept = self.minicolumn.back_predict(out_code)
        self.assertEqual([1] * 10, out_code)
        np.testing.assert_array_equal(np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0]), code)
        self.assertEqual(2, controversy)
        self.assertEqual(MINICOLUMN_LEARNING.ACCEPT, accept)


class TestPointSleep(unittest.TestCase):
    def setUp(self):
        self.minicolumn = Minicolumn(
            space_size=5,
            in_random_bits=1,
            out_random_bits=1,
            in_dimensions=1,
            out_dimensions=1,
            code_alignment=1,
            class_point=PointMockNone
        )
        self.minicolumn_activate = Minicolumn(
            space_size=5,
            in_random_bits=1,
            out_random_bits=1,
            in_point_activate=2,
            out_point_activate=2,
            in_dimensions=1,
            out_dimensions=1,
            code_alignment=1,
            in_code_activate=2,
            out_code_activate=2,
            class_point=PointMockNone
        )

    def test_assert_input_params(self):
        self.assertRaises(ValueError, self.minicolumn.sleep, activate=2)
        self.assertRaises(ValueError, self.minicolumn.sleep, activate=-1)
        self.assertRaises(ValueError, self.minicolumn.sleep, activate=None)

    def test_no_clusters(self):
        clusters, the_same_clusters = self.minicolumn.sleep()
        self.assertEqual(0, sum([len(cluster) for cluster in clusters]))
        self.assertEqual(0, the_same_clusters)

    def __constructor_active_mask_less_threshold(self, in_vec, out_vec):
        self.minicolumn_activate.count_clusters = 5 * 5
        for point in self.minicolumn_activate.space:
            for _ in range(5):
                point.clusters.append(ClusterMockWeight(in_vec, out_vec))

    def test_active_out_mask_less(self):
        self.__constructor_active_mask_less_threshold(np.array([0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1]))
        clusters, the_same_clusters = self.minicolumn_activate.sleep()
        self.assertEqual(0, sum([len(cluster) for cluster in clusters]))
        self.assertEqual(0, the_same_clusters)

    def test_active_in_mask_less(self):
        self.__constructor_active_mask_less_threshold(np.array([1, 1, 1, 1, 1]), np.array([0, 0, 0, 0, 0]))
        clusters, the_same_clusters = self.minicolumn_activate.sleep()
        self.assertEqual(0, sum([len(cluster) for cluster in clusters]))
        self.assertEqual(0, the_same_clusters)

    def test_sleep_0(self):
        self.__constructor_active_mask_less_threshold(
            np.array([1, 1, 1, 0, 0]),
            np.array([1, 1, 1, 0, 0])
        )

        points, the_same_clusters = self.minicolumn_activate.sleep()

        for clusters_number in points:
            np.testing.assert_array_equal([0, 1, 2, 3, 4], clusters_number)

        self.assertEqual(5, sum([len(point.clusters) for point in self.minicolumn_activate.space]))
        self.assertEqual(20, the_same_clusters)

        for point in self.minicolumn_activate.space:
            self.assertEqual(1, len(point.clusters))
            np.testing.assert_array_equal([1, 1, 1, 0, 0], point.clusters[0].in_w)
            np.testing.assert_array_equal([1, 1, 1, 0, 0], point.clusters[0].out_w)

    def test_sleep_1(self):
        self.minicolumn_activate.count_clusters = 5 * 5
        for point in self.minicolumn_activate.space:
            for i in range(5):
                point.clusters.append(ClusterMockWeight(
                    np.array([1] * i + [0] * (5 - i)),
                    np.array([1] * i + [0] * (5 - i))
                ))

        points, the_same_clusters = self.minicolumn_activate.sleep()

        for clusters_number in points:
            np.testing.assert_array_equal([2, 3, 4], clusters_number)

        self.assertEqual(15, sum([len(point.clusters) for point in self.minicolumn_activate.space]))
        self.assertEqual(0, the_same_clusters)

        for point in self.minicolumn_activate.space:
            self.assertEqual(3, len(point.clusters))
            np.testing.assert_array_equal([1, 1, 0, 0, 0], point.clusters[0].in_w)
            np.testing.assert_array_equal([1, 1, 1, 0, 0], point.clusters[1].in_w)
            np.testing.assert_array_equal([1, 1, 0, 0, 0], point.clusters[0].out_w)
            np.testing.assert_array_equal([1, 1, 1, 0, 0], point.clusters[1].out_w)


class TestPointUnsupervisedLearningException(unittest.TestCase):
    def setUp(self):
        self.minicolumn = Minicolumn(
            space_size=5,
            in_random_bits=1,
            out_random_bits=1,
            in_dimensions=1,
            out_dimensions=1,
            code_alignment=1,
            class_point=PointMockNone
        )

    def test_in_codes(self):
        self.assertRaises(ValueError, self.minicolumn.unsupervised_learning, in_codes=None)
        self.assertRaises(ValueError, self.minicolumn.unsupervised_learning, in_codes=[[0, None, -1, 1, 1]] * 3)
        self.assertRaises(ValueError, self.minicolumn.unsupervised_learning, in_codes=[[0, -1, -1, 1, 1]] * 3)
        self.assertRaises(ValueError, self.minicolumn.unsupervised_learning, in_codes=[[0, 1, 2]] * 3)
        self.assertRaises(ValueError, self.minicolumn.unsupervised_learning, in_codes=[[1, 0, 0.5, 1]] * 3)
        self.assertRaises(ValueError, self.minicolumn.unsupervised_learning, in_codes=[[0, 1, 0, 2.5, 1]] * 3)
        self.assertRaises(AssertionError, self.minicolumn.unsupervised_learning, in_codes=[[1] * 2] * 3)

    def test_controversy_out(self):
        self.assertRaises(ValueError, self.minicolumn.unsupervised_learning, in_codes=[[1]],
                          controversy_out=None)
        self.assertRaises(ValueError, self.minicolumn.unsupervised_learning, in_codes=[[1]],
                          controversy_out=-1)

    def test_controversy_in(self):
        self.assertRaises(ValueError, self.minicolumn.unsupervised_learning, in_codes=[[1]],
                          controversy_in=None)
        self.assertRaises(ValueError, self.minicolumn.unsupervised_learning, in_codes=[[1]],
                          controversy_in=-1)


class TestPointUnsupervisedLearning(unittest.TestCase):
    def setUp(self):
        self.minicolumn = Minicolumn(
            space_size=5,
            in_random_bits=1,
            out_random_bits=1,
            in_dimensions=1,
            out_dimensions=1,
            code_alignment=1,
            in_code_activate=0,
            class_point=PointMockNone
        )
        self.minicolumn_zeros = Minicolumn(
            space_size=5,
            in_random_bits=3,
            out_random_bits=3,
            in_dimensions=3,
            out_dimensions=3,
            code_alignment=2,
            seed=42,
            in_code_activate=0,
            class_point=PointMockZeros
        )
        self.minicolumn_controversy_out = Minicolumn(
            space_size=5,
            in_random_bits=4,
            out_random_bits=8,
            in_dimensions=4,
            out_dimensions=8,
            code_alignment=4,
            seed=42,
            controversy=2,
            class_point=PointMockDoubleIdentical
        )
        self.minicolumn_controversy_in = Minicolumn(
            space_size=200,
            in_random_bits=4,
            out_random_bits=8,
            in_dimensions=4,
            out_dimensions=8,
            code_alignment=2,
            seed=42,
            controversy=0.2,
            in_code_activate=0,
            class_point=PointMockControversyIn
        )
        self.minicolumn_code_alignment = Minicolumn(
            space_size=5,
            in_random_bits=4,
            out_random_bits=8,
            in_dimensions=4,
            out_dimensions=8,
            code_alignment=4,
            seed=42,
            controversy=0.1,
            in_code_activate=0,
            min_active_points=0,
            class_point=PointMockCodeAligment
        )

    def test_continue_zeros_codes(self):
        in_codes = [[0]] * 3
        min_in_code, min_out_code, min_ind_hamming = self.minicolumn.unsupervised_learning(in_codes=in_codes)
        self.assertEqual([[0]] * 3, in_codes)
        self.assertIsNone(None, min_out_code)
        self.assertIsNone(None, min_ind_hamming)
        self.assertIsNone(None, min_in_code)

    def test_out_not_detected(self):
        in_codes = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
        min_in_code, min_out_code, min_ind_hamming = self.minicolumn_zeros.unsupervised_learning(in_codes=in_codes)
        self.assertEqual([[1, 0, 1], [0, 1, 1], [1, 1, 0]], in_codes)
        np.testing.assert_array_equal([1, 1, 0], min_out_code)
        np.testing.assert_array_equal([1, 0, 1], min_in_code)
        self.assertEqual(0, min_ind_hamming)

    def test_controversy_out(self):
        in_codes = [[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]]
        min_in_code, min_out_code, min_ind_hamming = self.minicolumn_controversy_out.unsupervised_learning(in_codes=in_codes)
        self.assertEqual([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]], in_codes)
        self.assertIsNone(None, min_out_code)
        self.assertIsNone(None, min_ind_hamming)
        self.assertIsNone(None, min_in_code)

    def test_code_alignment_more(self):
        in_codes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        min_in_code, min_out_code, min_ind_hamming = self.minicolumn_code_alignment.unsupervised_learning(in_codes)
        self.assertEqual([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], in_codes)
        np.testing.assert_array_equal(np.array([1, 1, 0, 0, 0, 1, 0, 1]), min_out_code)
        np.testing.assert_array_equal(np.array([1, 1, 1, 1]), min_in_code)
        self.assertEqual(0, min_ind_hamming)
        self.assertEqual(self.minicolumn_code_alignment.alignment, np.sum(min_out_code))

    def test_code_alignment_less(self):
        in_codes = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
        min_in_code, min_out_code, min_ind_hamming = self.minicolumn_code_alignment.unsupervised_learning(in_codes)
        self.assertEqual([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], in_codes)
        np.testing.assert_array_equal(np.array([1, 1, 1, 0, 1, 0, 0, 0]), min_out_code)
        self.assertEqual([1, 0, 0, 0], min_in_code)
        self.assertEqual(self.minicolumn_code_alignment.alignment, np.sum(min_out_code))

    def test_code_alignment_eq(self):
        in_codes = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]
        min_in_code, min_out_code, min_ind_hamming = self.minicolumn_code_alignment.unsupervised_learning(in_codes)
        self.assertEqual([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]], in_codes)
        np.testing.assert_array_equal(np.array([1, 1, 0, 0, 1, 1, 0, 0]), min_out_code)
        np.testing.assert_array_equal(np.array([1, 1, 0, 0]), min_in_code)
        self.assertEqual(self.minicolumn_code_alignment.alignment, np.sum(min_out_code))

    def test_code_controversy_in(self):
        in_codes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        min_in_code, min_out_code, min_ind_hamming = self.minicolumn_controversy_in.unsupervised_learning(
            in_codes=in_codes, controversy_in=0
        )
        self.assertEqual([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], in_codes)
        self.assertIsNone(None, min_out_code)
        self.assertIsNone(None, min_ind_hamming)
        self.assertIsNone(None, min_in_code)

    def test_code_hamming_dist_more(self):
        in_codes = [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]]
        min_in_code, min_out_code, min_ind_hamming = self.minicolumn_controversy_in.unsupervised_learning(in_codes)
        self.assertEqual([[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]], in_codes)
        np.testing.assert_array_equal([1, 1, 1, 1], min_in_code)
        np.testing.assert_array_equal([0, 1, 0, 0, 0, 1, 0, 0], min_out_code)
        self.assertEqual(0, min_ind_hamming)

    def test_code_min_ind_hamming(self):
        in_codes = [[0], [1], [1]]
        min_in_code, min_out_code, min_ind_hamming = self.minicolumn.unsupervised_learning(in_codes)
        self.assertEqual([[0], [1], [1]], in_codes)
        np.testing.assert_array_equal([1], min_in_code)
        np.testing.assert_array_equal([1], min_out_code)
        self.assertEqual(1, min_ind_hamming)

    @unittest.skip('Не реализован. Выбор минимального оптимального кода')
    def test_code_hamming_min(self):
        pass


if __name__ == '__main__':
    unittest.main()