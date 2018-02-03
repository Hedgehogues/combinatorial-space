# -*- encoding: utf-8 -*-

import unittest
import numpy as np
from src.combinatorial_space.cluster import Cluster, ClusterAnswer


class TestCluster__init__(unittest.TestCase):

    def test_lr_less_0(self):
        n, m = 5, 3
        values = [-0.5, None]
        for value in values:
            self.assertRaises(
                ValueError, Cluster,
                in_sub_code=np.array([0.1] * n + [0.2] * n),
                out_sub_code=np.array([0.3] * m + [0.4] * m),
                lr=value
            )

    def test_binarization_less_0(self):
        n, m = 5, 3
        values = [-0.25, None]
        for value in values:
            self.assertRaises(
                ValueError, Cluster,
                in_sub_code=np.array([0.1] * n + [0.2] * n),
                out_sub_code=np.array([0.3] * m + [0.4] * m),
                binarization=value
            )

    def test_in_cluster_modify_less_0(self):
        n, m = 5, 3
        values = [-0.25, None]
        for value in values:
            self.assertRaises(
                ValueError, Cluster,
                in_sub_code=np.array([0.1] * n + [0.2] * n),
                out_sub_code=np.array([0.3] * m + [0.4] * m),
                in_cluster_modify=value
            )

    def test_out_cluster_modify_less_0(self):
        n, m = 5, 3
        values = [-0.25, None]
        for value in values:
            self.assertRaises(
                ValueError, Cluster,
                in_sub_code=np.array([0.1] * n + [0.2] * n),
                out_sub_code=np.array([0.3] * m + [0.4] * m),
                out_cluster_modify=value
            )


class TestClusterBase(unittest.TestCase):
    def setUp(self):
        n, m = 5, 3
        self.base_cluster_a = Cluster(
            in_sub_code=np.array([0.005] * n + [0.0006] * n),
            out_sub_code=np.array([0.007] * m + [0.0008] * m),
            in_cluster_modify=0.5, out_cluster_modify=0.5,
            binarization=0.1,
            lr=0.5,
            is_modify_lr=True
        )
        self.base_cluster_b = Cluster(
            in_sub_code=np.array([0.2] * n + [0] * n),
            out_sub_code=np.array([0.05] * m + [0] * m),
            in_cluster_modify=0.1, out_cluster_modify=0.1,
            binarization=0.005,
            lr=0.5,
            is_modify_lr=True
        )
        self.base_cluster_c = Cluster(
            in_sub_code=np.array([0.1] * n + [0.2] * n),
            out_sub_code=np.array([0.3] * m + [0.4] * m),
            in_cluster_modify=0.1, out_cluster_modify=0.1,
            binarization=0,
            lr=0.5,
            is_modify_lr=False
        )
        self.base_cluster_d = Cluster(
            in_sub_code=np.array([0.2] * n + [0] * n),
            out_sub_code=np.array([0.2] * m + [0] * m),
            in_cluster_modify=0.1, out_cluster_modify=0.1,
            binarization=0.1,
            lr=0.5,
            is_modify_lr=True
        )
        self.base_cluster_e = Cluster(
            in_sub_code=np.array([0.3] * n + [0] * n),
            out_sub_code=np.array([0.3] * m + [0] * m),
            in_cluster_modify=0.1, out_cluster_modify=0.1,
            binarization=0.1,
            lr=0.5,
            is_modify_lr=True
        )


class TestClusterException(TestClusterBase):
    def test_less_0_predict_front(self):
        self.assertRaises(ValueError, self.base_cluster_a.predict_front, [-1] * 1 + [0] * 9)

    def test_less_0_predict_back(self):
        self.assertRaises(ValueError, self.base_cluster_a.predict_back, [-1] * 1 + [0] * 5)

    def test_not_0_not_1_predict_front(self):
        self.assertRaises(ValueError, self.base_cluster_a.predict_front, [1.] * 1 + [0.2] * 5 + [0] * 4)

    def test_not_0_not_1_predict_back(self):
        self.assertRaises(ValueError, self.base_cluster_a.predict_back, [1.] * 1 + [0.] * 3 + [0.2] * 2)

    def test_not_0_not_1_predict_front_2(self):
        self.assertRaises(ValueError, self.base_cluster_a.predict_front, [-1.] * 1 + [0.2] * 5 + [0] * 4)

    def test_not_0_not_1_predict_back_2(self):
        self.assertRaises(ValueError, self.base_cluster_a.predict_back, [-1.] * 1 + [0] * 3 + [0.2] * 2)

    def test_less_0_modify_in(self):
        in_x = np.array([-1] * 9 + [0] * 1)
        out_x = np.array([1] * 5 + [0] * 1)
        self.assertRaises(ValueError, self.base_cluster_a.modify, in_x, out_x)

    def test_less_0_modify_out(self):
        in_x = np.array([1] * 9 + [0] * 1)
        out_x = np.array([-1] * 5 + [0] * 1)
        self.assertRaises(ValueError, self.base_cluster_a.modify, in_x, out_x)

    def test_predict_front_code_not_a_valid(self):
        self.assertRaises(AssertionError, self.base_cluster_a.predict_front, [1] * 1)

    def test_predict_back_code_not_a_valid(self):
        self.assertRaises(AssertionError, self.base_cluster_a.predict_back, [0] * 1)


class TestClusterPredict(TestClusterBase):
    def test_front_0(self):
        code = [1] * 1 + [0] * 9
        dot, out_sub_code, status = self.base_cluster_a.predict_front(code)
        self.assertEqual(status, ClusterAnswer.NOT_ACTIVE)
        self.assertEqual([1] * 1 + [0] * 9, code)
        self.assertEqual(dot, None)
        self.assertIsNone(out_sub_code)

    def test_back_0(self):
        code = [1] * 1 + [0] * 5
        dot, in_sub_code, status = self.base_cluster_a.predict_back(code)
        self.assertEqual(status, ClusterAnswer.NOT_ACTIVE)
        self.assertEqual([1] * 1 + [0] * 5, code)
        self.assertEqual(dot, None)
        self.assertIsNone(in_sub_code)

    def test_front_1(self):
        code = [1] * 1 + [0] * 9
        dot, out_sub_code, status = self.base_cluster_b.predict_front(code)
        self.assertEqual(status, ClusterAnswer.ACTIVE)
        self.assertEqual([1] * 1 + [0] * 9, code)
        self.assertAlmostEqual(dot, 0.2, places=5)
        np.testing.assert_array_equal(out_sub_code, [1] * 3 + [0] * 3)

    def test_back_1(self):
        code = [1] * 1 + [0] * 5
        dot, in_sub_code, status = self.base_cluster_d.predict_back(code)
        self.assertEqual(status, ClusterAnswer.ACTIVE)
        self.assertEqual([1] * 1 + [0] * 5, code)
        self.assertAlmostEqual(dot, 0.2, places=5)
        np.testing.assert_array_equal(in_sub_code, [1] * 5 + [0] * 5)

    def test_front_2(self):
        code = [0] * 1 + [1] * 9
        dot, out_sub_code, status = self.base_cluster_c.predict_front(code)
        self.assertEqual(status, ClusterAnswer.ACTIVE)
        self.assertEqual([0] * 1 + [1] * 9, code)
        self.assertAlmostEqual(dot, 1.4, places=5)
        np.testing.assert_array_equal(out_sub_code, [1] * 6)

    def test_back_2(self):
        code = [0] * 1 + [1] * 5
        dot, in_sub_code, status = self.base_cluster_e.predict_back(code)
        self.assertEqual(status, ClusterAnswer.ACTIVE)
        self.assertEqual([0] * 1 + [1] * 5, code)
        self.assertAlmostEqual(dot, 0.6, places=5)
        np.testing.assert_array_equal(in_sub_code, [1] * 5 + [0] * 5)


class TestClusterModify(TestClusterBase):
    def test_0(self):
        in_x = [1] * 1 + [0] * 9
        out_x = [1] * 1 + [0] * 5
        is_modify = self.base_cluster_a.modify(in_x, out_x)
        self.assertEqual([1] * 1 + [0] * 9, in_x)
        self.assertEqual([1] * 1 + [0] * 5, out_x)
        self.assertEqual(is_modify, ClusterAnswer.NOT_MODIFY)
        self.assertEqual(self.base_cluster_a.count_modify, 0)
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

    def test_1(self):
        in_x = [0] * 9 + [1] * 1
        out_x = [1] * 5 + [0] * 1
        is_modify = self.base_cluster_b.modify(in_x, out_x)
        self.assertEqual([0] * 9 + [1] * 1, in_x)
        self.assertEqual([1] * 5 + [0] * 1, out_x)
        self.assertEqual(is_modify, ClusterAnswer.NOT_MODIFY)
        self.assertEqual(self.base_cluster_b.count_modify, 0)
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

    def test_2(self):
        in_x = [1] * 9 + [0] * 1
        out_x = [0] * 5 + [1] * 1
        is_modify = self.base_cluster_b.modify(in_x, out_x)
        self.assertEqual([1] * 9 + [0] * 1, in_x)
        self.assertEqual([0] * 5 + [1] * 1, out_x)
        self.assertEqual(is_modify, ClusterAnswer.NOT_MODIFY)
        self.assertEqual(self.base_cluster_b.count_modify, 0)
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

    def test_3(self):
        in_x = [1] * 9 + [0] * 1
        out_x = [1] * 5 + [0] * 1
        is_modify = self.base_cluster_b.modify(in_x, out_x)
        self.assertEqual([1] * 9 + [0] * 1, in_x)
        self.assertEqual([1] * 5 + [0] * 1, out_x)
        self.assertEqual(is_modify, ClusterAnswer.MODIFY)
        self.assertEqual(self.base_cluster_b.count_modify, 1)
        np.testing.assert_array_almost_equal(
            self.base_cluster_b.in_w,
            np.array([1.56524758, 1.56524758, 1.56524758, 1.56524758, 1.56524758, 1.11803399,
                      1.11803399, 1.11803399, 1.11803399, 0.]),
            decimal=5
        )
        np.testing.assert_array_almost_equal(
            self.base_cluster_b.out_w,
            np.array([1.44337567, 1.44337567, 1.44337567, 0.8660254, 0.8660254, 0.]),
            decimal=5
        )

    def test_4(self):
        in_x = [1] * 9 + [0] * 1
        out_x = [1] * 5 + [0] * 1
        is_modify = self.base_cluster_c.modify(in_x, out_x)
        self.assertEqual([1] * 9 + [0] * 1, in_x)
        self.assertEqual([1] * 5 + [0] * 1, out_x)
        self.assertEqual(is_modify, ClusterAnswer.MODIFY)
        self.assertEqual(self.base_cluster_c.count_modify, 1)
        np.testing.assert_array_almost_equal(
            self.base_cluster_c.in_w,
            np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.7, 1.7, 1.7, 1.7, 0.4]),
            decimal=5
        )
        np.testing.assert_array_almost_equal(
            self.base_cluster_c.out_w,
            np.array([1.32790562, 1.32790562, 1.32790562, 1.44337567, 1.44337567, 0.46188022]),
            decimal=5
        )


if __name__ == '__main__':
    unittest.main()
