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
                base_in=np.array([0.004] * n + [0.0001] * n),
                base_out=np.array([0.004] * m + [0.0001] * m),
                in_threshold_modify=0.5, out_threshold_modify=0.5,
                threshold_bin=0.1,
                base_lr=0.5,
                is_modify_lr=True
        )
        self.base_cluster_b = Cluster(
                base_in=np.array([0.05] * n + [0] * n),
                base_out=np.array([0.05] * m + [0] * m),
                in_threshold_modify=0.1, out_threshold_modify=0.1,
                threshold_bin=0.1,
                base_lr=0.5,
                is_modify_lr=True
        )
        self.base_cluster_c = Cluster(
                base_in=np.array([0.2] * n + [0.3] * n),
                base_out=np.array([0.2] * m + [0.3] * m),
                in_threshold_modify=0.1, out_threshold_modify=0.1,
                threshold_bin=0.25,
                base_lr=0.5,
                is_modify_lr=False
        )

    def test_predict_front(self):
        o = [1]
        z = [0]

        # Условие 1
        corr, out_sub_code = self.base_cluster_a.predict_front(np.array(o * 1 + z * 9))
        self.assertAlmostEqual(corr, 0, places=5)
        self.assertEqual(out_sub_code, None)

        # Условие 2
        corr, out_sub_code = self.base_cluster_b.predict_front(np.array(o * 1 + z * 9))
        self.assertAlmostEqual(corr, 0.333333, places=5)
        np.testing.assert_array_equal(out_sub_code, z * 6)

        # Условие 3
        corr, out_sub_code = self.base_cluster_c.predict_front(np.array(o * 1 + z * 9))
        self.assertAlmostEqual(corr, -0.333333, places=5)
        np.testing.assert_array_equal(out_sub_code, z * 3 + o * 3)

    def test_predict_back(self):
        o = [1]
        z = [0]

        # Условие 1
        corr, in_sub_code = self.base_cluster_a.predict_back(np.array(o * 1 + z * 5))
        self.assertAlmostEqual(corr, 0, places=5)
        self.assertEqual(in_sub_code, None)

        # Условие 2
        corr, in_sub_code = self.base_cluster_b.predict_back(np.array(o * 1 + z * 5))
        self.assertAlmostEqual(corr, 0.4472135, places=5)
        np.testing.assert_array_equal(in_sub_code, z * 10)

        # Условие 3
        corr, in_sub_code = self.base_cluster_c.predict_back(np.array(o * 1 + z * 5))
        self.assertAlmostEqual(corr, -0.4472135, places=5)
        np.testing.assert_array_equal(in_sub_code, z * 5 + o * 5)

    def test_modify(self):
        o = [1]
        z = [0]

        # Условие 1
        in_x = np.array(o * 1 + z * 9)
        out_x = np.array(o * 1 + z * 5)
        is_modify = self.base_cluster_a.modify(in_x, out_x)
        self.assertEqual(is_modify, 0)
        self.assertEqual(self.base_cluster_a.count_modifing, 0)
        np.testing.assert_array_almost_equal(
            self.base_cluster_a.in_w,
            np.array([0.004] * 5 + [0.0001] * 5),
            decimal=5
        )
        np.testing.assert_array_almost_equal(
            self.base_cluster_a.out_w,
            np.array([0.004] * 3 + [0.0001] * 3),
            decimal=5
        )

        # Условие 2
        in_x = np.array(o * 1 + z * 9)
        out_x = np.array(o * 1 + z * 5)
        is_modify = self.base_cluster_b.modify(in_x, out_x)
        self.assertEqual(is_modify, 1)
        self.assertEqual(self.base_cluster_b.count_modifing, 1)
        np.testing.assert_array_almost_equal(
            self.base_cluster_b.in_w,
            np.array([0.67082039, 0.4472136, 0.4472136, 0.4472136, 0.4472136, 0., 0., 0., 0., 0.]),
            decimal=5

        )
        np.testing.assert_array_almost_equal(
            self.base_cluster_b.out_w,
            np.array([ 0.8660254, 0.57735027, 0.57735027, 0., 0., 0.]),
            decimal=5
        )

        # Условие 3
        in_x = np.array(o * 1 + z * 9)
        out_x = np.array(o * 1 + z * 5)
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