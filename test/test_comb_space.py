# -*- encoding: utf-8 -*-

import unittest
import numpy as np

from comb_space import Cluster


class Test_comb_space_cluster(unittest.TestCase):
    # Вызывается перед каждым тестом
    def setUp(self):

        n = 5
        m = 3
        self.base_cluster_a = Cluster(
                base_in=np.array([0.005] * n + [0.0006] * n),
                base_out=np.array([0.007] * m + [0.0008] * m),
                in_threshold_modify=0.5, out_threshold_modify=0.5,
                threshold_bin=0.1,
                base_lr=0.5,
                is_modify_lr=True
        )
        self.base_cluster_b = Cluster(
                base_in=np.array([0.2] * n + [0] * n),
                base_out=np.array([0.05] * m + [0] * m),
                in_threshold_modify=0.1, out_threshold_modify=0.1,
                threshold_bin=0.1,
                base_lr=0.5,
                is_modify_lr=True
        )
        self.base_cluster_c = Cluster(
                base_in=np.array([0.1] * n + [0.2] * n),
                base_out=np.array([0.3] * m + [0.4] * m),
                in_threshold_modify=0.1, out_threshold_modify=0.1,
                threshold_bin=0.25,
                base_lr=0.5,
                is_modify_lr=False
        )

    def test_predict_front_not_in_threshold_modify(self):
        dot, out_sub_code = self.base_cluster_a.predict_front([1] * 1 + [0] * 9)
        self.assertAlmostEqual(dot, 0, places=5)
        self.assertEqual(out_sub_code, None)

    def test_predict_front_modify_not_more_threshold_bin(self):
        dot, out_sub_code = self.base_cluster_b.predict_front(np.array([1] * 1 + [0] * 9))
        self.assertAlmostEqual(dot, 0.2, places=5)
        np.testing.assert_array_equal(out_sub_code, [0] * 6)

    def test_predict_back_modify_more_threshold_bin(self):
        dot, out_sub_code = self.base_cluster_c.predict_front(np.array([0] * 1 + [1] * 9))
        self.assertAlmostEqual(dot, 1.4, places=5)
        np.testing.assert_array_equal(out_sub_code, [1] * 6)

    def test_modify_not_threshold_modify(self):
        in_x = np.array([1] * 1 + [0] * 9)
        out_x = np.array([1] * 1 + [0] * 5)
        is_modify = self.base_cluster_a.modify(in_x, out_x)
        self.assertEqual(is_modify, 0)
        self.assertEqual(self.base_cluster_a.count_modifing, 0)
        np.testing.assert_array_almost_equal(
            self.base_cluster_a.in_w,
            np.array([0.005] * 5 + [0.0006] * 5),
            decimal=5
        )
        np.testing.assert_array_almost_equal(
            self.base_cluster_a.out_w,
            np.array([0.007] * 3 + [0.0008] * 3),
            decimal=5
        )

    def test_modify_false_in_true_out_threshold_modify(self):
        in_x = np.array([0] * 9 + [1] * 1)
        out_x = np.array([1] * 5 + [0] * 1)
        is_modify = self.base_cluster_b.modify(in_x, out_x)
        self.assertEqual(is_modify, 0)
        self.assertEqual(self.base_cluster_b.count_modifing, 0)
        np.testing.assert_array_almost_equal(
            self.base_cluster_b.in_w,
            np.array([0.2] * 5 + [0.0] * 5),
            decimal=5
        )
        np.testing.assert_array_almost_equal(
            self.base_cluster_b.out_w,
            np.array([0.05] * 3 + [0.0] * 3),
            decimal=5
        )

    def test_modify_true_in_false_out_threshold_modify(self):
        in_x = np.array([1] * 9 + [0] * 1)
        out_x = np.array([0] * 5 + [1] * 1)
        is_modify = self.base_cluster_b.modify(in_x, out_x)
        self.assertEqual(is_modify, 0)
        self.assertEqual(self.base_cluster_b.count_modifing, 0)
        np.testing.assert_array_almost_equal(
            self.base_cluster_b.in_w,
            np.array([0.2] * 5 + [0.0] * 5),
            decimal=5
        )
        np.testing.assert_array_almost_equal(
            self.base_cluster_b.out_w,
            np.array([0.05] * 3 + [0.0] * 3),
            decimal=5
        )

    def test_modify_threshold_modify_true_modify_lr(self):
        in_x = np.array([1] * 9 + [0] * 1)
        out_x = np.array([1] * 5 + [0] * 1)
        is_modify = self.base_cluster_b.modify(in_x, out_x)
        self.assertEqual(is_modify, 1)
        self.assertEqual(self.base_cluster_b.count_modifing, 1)
        np.testing.assert_array_almost_equal(
            self.base_cluster_b.in_w,
            np.array([[ 1.56524758, 1.56524758, 1.56524758, 1.56524758, 1.56524758, 1.11803399,
                        1.11803399, 1.11803399, 1.11803399, 0.]]),
            decimal=5

        )
        np.testing.assert_array_almost_equal(
            self.base_cluster_b.out_w,
            np.array([0.8660254, 0.57735027, 0.57735027, 0., 0., 0.]),
            decimal=5
        )

    def test_modify_threshold_modify_false_modify_lr(self):
        in_x = np.array([1] * 1 + [0] * 9)
        out_x = np.array([1] * 1 + [0] * 5)
        is_modify = self.base_cluster_c.modify(in_x, out_x)
        self.assertEqual(is_modify, 1)
        self.assertEqual(self.base_cluster_c.count_modifing, 1)
        np.testing.assert_array_almost_equal(
            self.base_cluster_c.in_w,
            np.array([0.3721042, 0.24806947, 0.24806947, 0.24806947, 0.24806947, 0.3721042, 0.3721042,
                      0.3721042, 0.3721042, 0.3721042]),
            decimal=5
        )
        np.testing.assert_array_almost_equal(
            self.base_cluster_c.out_w,
            np.array([0.48038446, 0.32025631, 0.32025631, 0.48038446, 0.48038446, 0.48038446]),
            decimal=5
        )

if __name__ == '__main__':
    unittest.main()