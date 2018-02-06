import unittest
from enum import Enum

import pandas as pd
import numpy as np

from src.combinatorial_space.enums import MINICOLUMN
from src.image import transformers
from src.combinatorial_space.minicolumn import Minicolumn
from src.context.transformer import ContextTransformer


class TestFunctionalMinicolumn(unittest.TestCase):

    class Answer(Enum):
        CONT_NUM = 0
        OPT_IND = 1
        OPT_OUT = 2
        STATUS = 3
        MEANS = 4
        EMPTY_CODE = 5

    @classmethod
    def setUpClass(cls):
        print('\n')
        print('#########')
        print('The executing of these tests can be a bit long')
        print('#########')

    def setUp(self):

        window_size = [4, 4]
        width_angle = np.pi / 2
        strength = 0
        non_zeros_bits = 5
        space_size = 500
        directs = 4
        self.minicolumn = Minicolumn(
            space_size=space_size,
            max_clusters=900,
            in_dimensions=64, in_random_bits=25,
            out_dimensions=20, out_random_bits=15,
            seed=42,
            code_alignment=5,
            in_point_activate=5,
            out_point_activate=4,
            in_cluster_modify=6,
            out_cluster_modify=3,
            lr=0.3, binarization=0.1
        )
        self.transformer = ContextTransformer(
            directs=directs, window_size=window_size, width_angle=width_angle, strength=strength,
            non_zeros_bits=non_zeros_bits
        )
        self.answer = [
            { 
                self.Answer.STATUS: MINICOLUMN.LEARN,
                self.Answer.OPT_OUT: np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]),
                self.Answer.MEANS: 0.628,
                self.Answer.CONT_NUM: 314,
                self.Answer.OPT_IND: 7,
                self.Answer.EMPTY_CODE: False
            },
            {
                self.Answer.STATUS: MINICOLUMN.LEARN,
                self.Answer.OPT_OUT: np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                self.Answer.MEANS: 1.154,
                self.Answer.CONT_NUM: 577,
                self.Answer.OPT_IND: 20,
                self.Answer.EMPTY_CODE: False
            },
            {
                self.Answer.STATUS: MINICOLUMN.LEARN,
                self.Answer.OPT_OUT: np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                self.Answer.MEANS: 1.408,
                self.Answer.CONT_NUM: 704,
                self.Answer.OPT_IND: 14,
                self.Answer.EMPTY_CODE: False
            },
            {
                self.Answer.STATUS: MINICOLUMN.LEARN,
                self.Answer.OPT_OUT: np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
                self.Answer.MEANS: 1.928,
                self.Answer.CONT_NUM: 964,
                self.Answer.OPT_IND: 29,
                self.Answer.EMPTY_CODE: False
            },
            {
                self.Answer.STATUS: MINICOLUMN.SLEEP,
                self.Answer.OPT_OUT: None,
                self.Answer.MEANS: 1.928,
                self.Answer.CONT_NUM: 964,
                self.Answer.OPT_IND: None,
                self.Answer.EMPTY_CODE: False
            },
            {
                self.Answer.STATUS: MINICOLUMN.SLEEP,
                self.Answer.OPT_OUT: None,
                self.Answer.MEANS: 1.928,
                self.Answer.CONT_NUM: 964,
                self.Answer.OPT_IND: None,
                self.Answer.EMPTY_CODE: False
            }
        ]

    def __add_to_dict(self, dict_var, key, value):
        dict_tmp = dict_var[-1]
        dict_tmp[key] = value
        dict_var[-1] = dict_tmp

    def __init_answ(self, answ):
        answ.append(
            {
                self.Answer.EMPTY_CODE: None,
                self.Answer.CONT_NUM: None,
                self.Answer.OPT_IND: None,
                self.Answer.OPT_OUT: None,
                self.Answer.STATUS: None,
                self.Answer.MEANS: None
            }
        )

    def test_minicolumn(self):

        df = pd.read_csv('../../data/test/test_image.csv', header=None, nrows=1)
        answer = []

        label, image = transformers.get_image(df, 0)
        self.assertEqual(9, label)
        for im_num in range(0, 6):
            self.__init_answ(answer)
            codes, context_numbes, image_sample = self.transformer.get_sample_codes(image)
            if codes is None:
                self.__add_to_dict(answer, self.Answer.EMPTY_CODE, True)
                continue
            else:
                self.__add_to_dict(answer, self.Answer.EMPTY_CODE, False)

            opt_ind, out_code, status = self.minicolumn.learn(codes, im_num, in_controversy=20, out_controversy=5)
            self.__add_to_dict(answer, self.Answer.OPT_IND, opt_ind)
            self.__add_to_dict(answer, self.Answer.OPT_OUT, out_code)
            self.__add_to_dict(answer, self.Answer.CONT_NUM, self.minicolumn.count_clusters)
            self.__add_to_dict(answer, self.Answer.STATUS, status)
            self.__add_to_dict(answer, self.Answer.MEANS, np.mean([len(p.clusters) for p in self.minicolumn.space]))

        for item in zip(self.answer, answer):
            for key in item[0]:
                np.testing.assert_equal(item[0][key], item[1][key])


if __name__ == '__main__':
    unittest.main()