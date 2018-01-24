from enum import Enum

import numpy as np

from src.combinatorial_space.expetions import CombSpaceExceptions

np.warnings.filterwarnings('ignore')


class ClusterAnswer(Enum):
    ACTIVE = 1
    NOT_ACTIVE = 0


class Cluster:
    """
        Кластер в точке комбинаторного пространства

        base_in_subvector, base_out_subvector - бинарный код, образующий кластер. Между образующим кодом и новым кодом
        будет вычисляться скалярное произведение
        in_threshold_modify, out_threshold_modify - порог активации кластера
        threshold_bin - порог бинаризации кода
        вектора на новый вектор больше порога, то будет пересчитан веса кластера, выделяющие первую главную компоненту
        base_lr - начальное значение скорости обучения
        is_modify_lr - модификация скорости обучения пропорционально номер шага
    """
    def __init__(self,
                 base_in, base_out,
                 in_threshold_modify=5, out_threshold_modify=0,
                 threshold_bin=0.1,
                 base_lr=0.01,
                 is_modify_lr=True):
        if in_threshold_modify is None or out_threshold_modify is None or \
                np.sum(np.uint8(np.array(base_in) is None)) or np.sum(np.uint8(np.array(base_out) is None)) or \
                threshold_bin is None or is_modify_lr is None or \
                in_threshold_modify < 0 or out_threshold_modify < 0 or base_lr < 0 or \
                np.sum(np.uint8(np.array(base_in) < 0)) or np.sum(np.uint8(np.array(base_out) < 0)) or \
                threshold_bin < 0 or type(is_modify_lr) is not bool:
                    raise ValueError("Неожиданное значение переменной")
        self.in_threshold_modify, self.out_threshold_modify = in_threshold_modify, out_threshold_modify
        self.base_lr = base_lr

        # Первые главные компоненты для входного и выходного векторов (Согласно правилу Хебба)
        self.in_w, self.out_w = base_in, base_out

        self.threshold_bin = threshold_bin
        self.is_modify_lr = is_modify_lr
        self.count_modifing = 0

    def __predict(self, x, w_0, w_1, threshold_modify):

        CombSpaceExceptions.code_value(x)
        CombSpaceExceptions.none(x, 'Не определён аргумент')
        CombSpaceExceptions.len(len(x), len(w_0), 'Не совпадает размерность')

        dot = np.dot(x, w_0)
        if np.abs(dot) > threshold_modify:
            return dot, np.int8(w_1 > self.threshold_bin), ClusterAnswer.ACTIVE
        else:
            return None, None, ClusterAnswer.NOT_ACTIVE

    """
        Предсказание вперёд, т.е. предсказание входа по выходу

        in_x - входной вектор

        Возвращается значение похожести (корелляция), предсказанный подвектор соответствующего размера. Если похожесть
        кластера на подвходной вектор маленькая, то возвращается нулевая корелляция и None 
    """
    def predict_front(self, in_x):
        return self.__predict(in_x, self.in_w, self.out_w, self.in_threshold_modify)

    """
        Предсказание назад, т.е. предсказание выхода по входу

        out_x - выходной вектор

        Возвращается значение похожести (корелляция), предсказанный подвектор соответствующего размера. Если похожесть
        кластера на подвходной вектор маленькая, то возвращается нулевая корелляция и None 
    """
    def predict_back(self, out_x):
        return self.__predict(out_x, self.out_w, self.in_w, self.out_threshold_modify)

    """
        Получение величин delta, используемых в обучении Хебба
    """

    def __get_delta(self, in_x, out_x):
        in_y = np.dot(in_x, self.in_w)
        out_y = np.dot(out_x, self.out_w)
        if self.is_modify_lr:
            delta_in = np.array((self.base_lr / self.count_modifing) * in_y * np.array(in_x))
            delta_out = np.array((self.base_lr / self.count_modifing) * out_y * np.array(out_x))
            # Правило Ойо почему-то расходится
            #                   self.in_w = self.in_w + (self.base_lr/self.count_modifing)*in_y*(in_x - in_y*self.in_w)
            #                   self.out_w = self.out_w + (self.base_lr/self.count_modifing)*out_y*(out_x - out_y*self.out_w)
        else:
            delta_in = np.array(self.base_lr * in_y * np.array(in_x))
            delta_out = np.array(self.base_lr * out_y * np.array(out_x))
            # Правило Ойо почему-то расходится
            #                   self.in_w = self.in_w + (self.base_lr*in_y*(in_x - in_y*self.in_w)
            #                   self.out_w = self.out_w + (self.base_lr*out_y*(out_x - out_y*self.out_w)
        return delta_in, delta_out

    """
        Функция, производящая модификацию пары кодов кластера точки комбинаторного пространства

        in_x, out_x - входной и выходной бинарные векторы подкодов соответствующих размерностей

        Возвращается 1, если была произведена модификация весов (т.е. кластер был активирован). В противном случае
        возвращается 0
    """
    def modify(self, in_x, out_x):
        CombSpaceExceptions.code_value(out_x)
        CombSpaceExceptions.code_value(in_x)
        CombSpaceExceptions.len(len(out_x), len(self.out_w), 'Не совпадает размерность')
        CombSpaceExceptions.len(len(in_x), len(self.in_w), 'Не совпадает размерность')
        CombSpaceExceptions.none(in_x, 'Не определён аргумент')
        CombSpaceExceptions.none(out_x, 'Не определён аргумент')

        in_dot = np.dot(in_x, self.in_w)
        out_dot = np.dot(out_x, self.out_w)

        if np.abs(in_dot) > self.in_threshold_modify and \
           np.abs(out_dot) > self.out_threshold_modify:
            self.count_modifing += 1
            delta_in, delta_out = self.__get_delta(in_x, out_x)
            self.in_w = np.divide((self.in_w + delta_in), (np.sum(self.in_w ** 2) ** (0.5)))
            self.out_w = np.divide((self.out_w + delta_out), (np.sum(self.out_w ** 2) ** (0.5)))

            return ClusterAnswer.ACTIVE
        else:
            return ClusterAnswer.NOT_ACTIVE
