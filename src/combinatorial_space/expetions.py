from enum import Enum

import numpy as np


class StatiscticsKey(Enum):
    pass


class CombSpaceExceptions:
    def __init__(self):
        pass

    @staticmethod
    def codes(codes, count_dimensions):
        CombSpaceExceptions.none(codes)
        for code in codes:
            CombSpaceExceptions.none(code)
            CombSpaceExceptions.code_value(code)
            CombSpaceExceptions.eq(len(code), count_dimensions, "Не совпадает размерность")
            CombSpaceExceptions.code_value(code)

    @staticmethod
    def none(obj, msg="Непредвиденное значение None"):
        if obj is None:
            raise ValueError(msg)

    @staticmethod
    def eq(obj_len, target_len, msg="Не совпадает размерность"):
        assert obj_len == target_len, msg

    @staticmethod
    def neq(obj_len, target_len, msg="Не совпадает размерность"):
        assert obj_len != target_len, msg

    @staticmethod
    def is_type(variable, type, msg="Неверное значение type_code"):
        if type(variable) is not type:
            raise TypeError(msg)

    @staticmethod
    def type_code(type_code, msg="Неверное значение type_code"):
        if not (type_code == -1 or type_code == 0):
            raise ValueError(msg)

    @staticmethod
    def more(obj, value, msg="Недопустимое значение переменной"):
        if obj > value:
            raise ValueError(msg)

    @staticmethod
    def less(obj, value, msg="Недопустимое значение переменной"):
        if obj < value:
            raise ValueError(msg)

    @staticmethod
    def less_or_equal(obj, value, msg="Недопустимое значение переменной"):
        if obj <= value:
            raise ValueError(msg)

    @staticmethod
    def more_or_equal(obj, value, msg="Недопустимое значение переменной"):
        if obj >= value:
            raise ValueError(msg)

    @staticmethod
    def code_value(code, msg="Значение аргумента может принимать значение 0 или 1"):
        # Значения выходного вектора могут быть равны 0 или 1
        if np.sum(np.uint8(np.logical_not(np.array(code) != 0) ^ (np.array(code) != 1))) > 0:
            raise ValueError(msg)
