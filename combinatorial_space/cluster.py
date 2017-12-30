import numpy as np

np.warnings.filterwarnings('ignore')

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


class Cluster:
    def __init__(self,
                 base_in, base_out,
                 in_threshold_modify, out_threshold_modify,
                 threshold_bin,
                 base_lr,
                 is_modify_lr):
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

    def __code_value_exeption(self, code):
        # Значения выходного вектора могут быть равны 0 или 1
        if np.sum(np.uint8(np.logical_not(np.array(code) != 0) ^ (np.array(code) != 1))) > 0:
            raise ValueError("Значение аргумента может принимать значение 0 или 1")

    def __len_exeption(self, obj_len, target_len):
        assert obj_len == target_len, "Не совпадает заданная размерность с поданой"

    """
        Предсказание вперёд, т.е. предсказание входа по выходу

        in_x - входной вектор

        Возвращается значение похожести (корелляция), предсказанный подвектор соответствующего размера. Если похожесть
        кластера на подвходной вектор маленькая, то возвращается нулевая корелляция и None 
    """
    def predict_front(self, in_x):
        self.__code_value_exeption(in_x)
        self.__len_exeption(len(in_x), len(self.in_w))
        dot = np.dot(in_x, self.in_w)
        if np.abs(dot) > self.in_threshold_modify:
            return dot, np.uint8(self.out_w > self.threshold_bin)
        else:
            return 0, None

    """
        Предсказание назад, т.е. предсказание выхода по входу

        out_x - выходной вектор

        Возвращается значение похожести (корелляция), предсказанный подвектор соответствующего размера. Если похожесть
        кластера на подвходной вектор маленькая, то возвращается нулевая корелляция и None 
    """

    def predict_back(self, out_x):
        self.__code_value_exeption(out_x)
        self.__len_exeption(len(out_x), len(self.out_w))
        dot = np.dot(out_x, self.out_w)
        if np.abs(dot) > self.out_threshold_modify:
            return dot, np.uint8(self.in_w > self.threshold_bin)
        else:
            return 0, None

    """
        Получение величин delta, используемых в обучении Хебба
    """

    def __get_delta(self, in_x, out_x):
        in_y = np.dot(in_x, self.in_w)
        out_y = np.dot(out_x, self.out_w)
        if self.is_modify_lr:
            delta_in = np.array((self.base_lr / self.count_modifing) * in_y * in_x)
            delta_out = np.array((self.base_lr / self.count_modifing) * out_y * out_x)
            # Правило Ойо почему-то расходится
            #                   self.in_w = self.in_w + (self.base_lr/self.count_modifing)*in_y*(in_x - in_y*self.in_w)
            #                   self.out_w = self.out_w + (self.base_lr/self.count_modifing)*out_y*(out_x - out_y*self.out_w)
        else:
            delta_in = np.array(self.base_lr * in_y * in_x)
            delta_out = np.array(self.base_lr * out_y * out_x)
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
        self.__code_value_exeption(out_x)
        self.__code_value_exeption(in_x)
        self.__len_exeption(len(out_x), len(self.out_w))
        self.__len_exeption(len(in_x), len(self.in_w))

        in_dot = np.dot(in_x, self.in_w)
        out_dot = np.dot(out_x, self.out_w)

        if np.abs(in_dot) > self.in_threshold_modify and \
           np.abs(out_dot) > self.out_threshold_modify:
            self.count_modifing += 1
            delta_in, delta_out = self.__get_delta(in_x, out_x)
            self.in_w = np.divide((self.in_w + delta_in), (np.sum(self.in_w ** 2) ** (0.5)))
            self.out_w = np.divide((self.out_w + delta_out), (np.sum(self.out_w ** 2) ** (0.5)))

            return 1
        return 0
