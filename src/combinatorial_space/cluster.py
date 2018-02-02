from enum import Enum

import numpy as np

from src.combinatorial_space.expetions import CombSpaceExceptions

np.warnings.filterwarnings('ignore')


class ClusterAnswer(Enum):
    ACTIVE = 1
    NOT_ACTIVE = 0
    MODIFY = 2
    NOT_MODIFY = 3


class Cluster:
    """
        Кластер в точке комбинаторного пространства

        in_sub_code, out_sub_code - бинарный код, образующий кластер. Между образующим кодом и новым кодом
        будет вычисляться скалярное произведение
        in_cluster_modify, out_cluster_modify - порог активации кластера
        binarization - порог бинаризации кода
        вектора на новый вектор больше порога, то будет пересчитан веса кластера, выделяющие первую главную компоненту
        lr - начальное значение скорости обучения
        is_modify_lr - модификация скорости обучения пропорционально номер шага
    """
    def __init__(self,
                 in_sub_code, out_sub_code,
                 in_cluster_modify=5, out_cluster_modify=0,
                 binarization=0.1, lr=0.01, is_modify_lr=True):

        CombSpaceExceptions.none(in_cluster_modify)
        CombSpaceExceptions.none(out_cluster_modify)
        CombSpaceExceptions.none(binarization)
        CombSpaceExceptions.none(is_modify_lr)
        CombSpaceExceptions.none(lr)

        CombSpaceExceptions.less(in_cluster_modify, 0)
        CombSpaceExceptions.less(out_cluster_modify, 0)
        CombSpaceExceptions.less(binarization, 0)

        CombSpaceExceptions.is_type(is_modify_lr, bool)
        CombSpaceExceptions.is_type(in_sub_code, list)
        CombSpaceExceptions.is_type(out_sub_code, list)

        self.in_cluster_modify, self.out_cluster_modify = in_cluster_modify, out_cluster_modify
        self.base_lr = lr

        # Первые главные компоненты для входного и выходного векторов (Согласно правилу Хебба)
        self.in_w, self.out_w = in_sub_code, out_sub_code

        self.binarization = binarization
        self.is_modify_lr = is_modify_lr
        self.count_modify = 0

    def __predict(self, x, w_0, w_1, cluster_modify):

        CombSpaceExceptions.code_value(x)
        CombSpaceExceptions.none(x, 'Не определён аргумент')
        CombSpaceExceptions.eq(len(x), len(w_0), 'Не совпадает размерность')
        CombSpaceExceptions.is_type(x, list)

        dot = np.dot(x, w_0)
        if np.abs(dot) < cluster_modify:
            return None, None, ClusterAnswer.NOT_ACTIVE

        return dot, np.int8(w_1 >= self.binarization), ClusterAnswer.ACTIVE

    """
        Предсказание вперёд, т.е. предсказание входа по выходу

        in_x - входной вектор

        Возвращаются 
        (
            Значение похожести (число совпавших битов), 
            Предсказанный подвектор соответствующего размера,
            Код возврата
        )
    """
    def predict_front(self, in_x):
        return self.__predict(in_x, self.in_w, self.out_w, self.in_cluster_modify)

    """
        Предсказание назад, т.е. предсказание выхода по входу

        
        out_x - выходной sub-код требуемой размерности

        Возвращаются 
        (
            Значение похожести (число совпавших битов), 
            Предсказанный подвектор соответствующего размера,
            Код возврата
        ) 
    """
    def predict_back(self, out_x):
        return self.__predict(out_x, self.out_w, self.in_w, self.out_cluster_modify)

    """
        Получение величин delta, используемых в обучении Хебба (Ойо)
    """
    def __get_delta(self, in_x, out_x):
        in_y = np.dot(in_x, self.in_w)
        out_y = np.dot(out_x, self.out_w)

        lr = self.base_lr / self.count_modify if self.is_modify_lr else self.base_lr

        delta_in = np.array(lr * np.multiply(in_y, np.array(in_x)))
        delta_out = np.array(lr * np.multiply(out_y, np.array(out_x)))

        return delta_in, delta_out

    """
        Функция, производящая модификацию пары кодов кластера точки комбинаторного пространства

        in_x, out_x - входной и выходной бинарные sub-коды требуемых размерностей

        Возвращается ClusterAnswer.MODIFY, если была произведена модификация весов (т.е. кластер был активирован).
        В противном случае возвращается ClusterAnswer.NOT_MODIFY
    """
    def modify(self, in_x, out_x):
        CombSpaceExceptions.code_value(out_x)
        CombSpaceExceptions.code_value(in_x)
        CombSpaceExceptions.eq(len(out_x), len(self.out_w), 'Не совпадает размерность')
        CombSpaceExceptions.eq(len(in_x), len(self.in_w), 'Не совпадает размерность')
        CombSpaceExceptions.none(in_x, 'Не определён аргумент')
        CombSpaceExceptions.none(out_x, 'Не определён аргумент')

        CombSpaceExceptions.is_type(in_x, list)
        CombSpaceExceptions.is_type(out_x, list)

        if np.dot(in_x, np.uint8(np.abs(self.in_w) >= self.binarization)) < self.in_cluster_modify:
            return ClusterAnswer.NOT_MODIFY
        if np.dot(out_x, np.uint8(np.abs(self.out_w) >= self.binarization)) < self.out_cluster_modify:
            return ClusterAnswer.NOT_MODIFY

        # TODO: Правило Ойо почему-то расходится, поэтому используется лобовое решение
        #############
        self.count_modify += 1
        delta_in, delta_out = self.__get_delta(in_x, out_x)
        self.in_w = np.divide((self.in_w + delta_in), (np.sqrt(np.sum(np.square(self.in_w)))))
        self.out_w = np.divide((self.out_w + delta_out), (np.sqrt(np.sum(np.square(self.out_w)))))

        return ClusterAnswer.MODIFY
