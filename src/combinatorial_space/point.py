import numpy as np

from src.combinatorial_space.cluster import Cluster
from src.combinatorial_space.enums import POINT_PREDICT, CLUSTER
from src.combinatorial_space.expetions import CombSpaceExceptions


class Point:
    """
        Точка комбинаторного пространства. Каждая точка содержит набор кластеров

        in_cluster_modify, out_cluster_modify - порог модификации кластера. Если скалярное произведение кластера
        на новый вектор больше порога, то будет пересчитаны веса кластера, выделяющие первую главную компоненту
        in_point_activate, out_point_activate - порог активации точки комбинаторного пространства. Если кол-во
        активных битов больше порога, то будет инициирован процесс модификации существующих кластеров или добавления
        нового кластера
        binarization - порог бинаризации кода. Скалярное произведение вычисляенся [w >= bin], где w - вес
        lr - начальное значение скорости обучения
        is_modify_lr - модификация скорости обучения пропорционально номеру шага
        in_random_bits, out_random_bits - количество случайных битов входного/выходного вектора для точки комб.
        in_dimensions, out_dimensions - размер входного и выходного векторов в точке комб. пространства
        max_cluster_per_point - максимальное количество кластеров в точке

        cluster_factory - фабричный метод для создания кластеров
    """
    def __init__(self,
                 in_cluster_modify=5, out_cluster_modify=0,
                 in_point_activate=5, out_point_activate=0,
                 binarization=0.1, lr=0.01, is_modify_lr=True,
                 in_random_bits=24, out_random_bits=10,
                 in_dimensions=256, out_dimensions=16,
                 max_clusters_per_point=100,
                 cluster_class=Cluster):

        CombSpaceExceptions.none(in_cluster_modify)
        CombSpaceExceptions.none(out_cluster_modify)
        CombSpaceExceptions.none(in_point_activate)
        CombSpaceExceptions.none(out_point_activate)
        CombSpaceExceptions.none(in_random_bits)
        CombSpaceExceptions.none(out_random_bits)
        CombSpaceExceptions.none(binarization)
        CombSpaceExceptions.none(is_modify_lr)
        CombSpaceExceptions.none(lr)
        CombSpaceExceptions.none(max_clusters_per_point)
        CombSpaceExceptions.none(in_dimensions)
        CombSpaceExceptions.none(out_dimensions)
        CombSpaceExceptions.none(cluster_class)

        CombSpaceExceptions.less(max_clusters_per_point, 0)
        CombSpaceExceptions.less(out_random_bits, 0)
        CombSpaceExceptions.less(in_random_bits, 0)
        CombSpaceExceptions.less(in_cluster_modify, 0)
        CombSpaceExceptions.less(out_cluster_modify, 0)
        CombSpaceExceptions.less(in_point_activate, 0)
        CombSpaceExceptions.less(out_point_activate, 0)
        CombSpaceExceptions.less(in_dimensions, 0)
        CombSpaceExceptions.less(out_dimensions, 0)
        CombSpaceExceptions.less(binarization, 0)
        CombSpaceExceptions.less(lr, 0)

        CombSpaceExceptions.is_type(is_modify_lr, bool)

        CombSpaceExceptions.more(in_random_bits, in_dimensions)
        CombSpaceExceptions.more(out_random_bits, out_dimensions)

        self.in_coords = np.sort(np.random.permutation(in_dimensions)[:in_random_bits])
        self.out_coords = np.sort(np.random.permutation(out_dimensions)[:out_random_bits])
        self.in_dimensions, self.out_dimensions = in_dimensions, out_dimensions
        self.clusters = []
        self.in_cluster_modify, self.out_cluster_modify = in_cluster_modify, out_cluster_modify
        self.in_point_activate, self.out_point_activate = in_point_activate, out_point_activate
        self.binarization = binarization
        self.lr = lr
        self.is_modify_lr = is_modify_lr
        self.max_clusters_per_point = max_clusters_per_point
        self.cluster_class = cluster_class

        self.statistics = []

    def __select_predict_function(self, cluster, sub_code, is_front):
        if is_front:
            return cluster.predict_front(sub_code)
        else:
            return cluster.predict_back(sub_code)

    def __predict(self, code, type_code, count_dimensions_0, count_dimensions_1, coords_0, coords_1,
                  point_activate, is_front):
        CombSpaceExceptions.none(code, "Не определён аргумент")
        CombSpaceExceptions.eq(len(code), count_dimensions_0, "Не совпадает размерность")
        CombSpaceExceptions.type_code(type_code)
        CombSpaceExceptions.code_value(code)

        if len(self.clusters) == 0:
            return None, POINT_PREDICT.NO_CLUSTERS

        sub_code = np.array(code)[coords_0]
        opt_dot = -np.inf
        opt_sub_code = None
        if np.sum(sub_code) >= point_activate:
            for cluster in self.clusters:
                dot, predicted_sub_code, status = self.__select_predict_function(cluster, sub_code, is_front)

                if status == CLUSTER.ACTIVE and dot > opt_dot:
                    CombSpaceExceptions.less(dot, 0, "Отрицательное значение скалярного произведения")
                    CombSpaceExceptions.none(dot, "Скалярное произведение None")
                    CombSpaceExceptions.none(predicted_sub_code, "Предсказанный вектор None")

                    opt_dot = dot
                    opt_sub_code = np.zeros(count_dimensions_1, dtype=np.int)
                    if type_code == -1:
                        predicted_sub_code[predicted_sub_code == 0] = -1
                    opt_sub_code[coords_1] = predicted_sub_code

        if opt_sub_code is None:
            return None, POINT_PREDICT.NOT_ACTIVE
        return opt_sub_code, POINT_PREDICT.ACTIVE

    """
        Осуществление выбора оптимального кластера при прямом предсказании 

        in_code - входной код 
        type_code - тип возвращаемого кода (с -1 или с 0). Каждый бит вектора может быть в 3х состояниях:
            * Активным (текущая точка наблюдает за битом и его значение равно 1)
            * Неактивным (текущая точка наблюдает за битом и его значение равно 0 (type_code = 0) или 
                          -1 (type_code = -1))
            * Ненаблюдаемым (текущая точка НЕ наблюдает за битом и его значение равно 0)
            
        Возвращается оптимальный выходной код
    """
    def predict_front(self, in_code, type_code=-1):
        return self.__predict(
            in_code, type_code,
            self.in_dimensions, self.out_dimensions,
            self.in_coords, self.out_coords,
            self.in_point_activate, True
        )

    """
        Осуществление выбора оптимального кластера при обратном предсказании 

        out_code - выходной код 
        type_code - тип возвращаемого кода (с -1 или с 0). Каждый бит вектора может быть в 3х состояниях:
            * Активным (текущая точка наблюдает за битом и его значение равно 1)
            * Неактивным (текущая точка наблюдает за битом и его значение равно 0 (type_code = 0) или 
                          -1 (type_code = -1))
            * Ненаблюдаемым (текущая точка НЕ наблюдает за битом и его значение равно 0)
            
        Возвращается оптимальный входной код
    """
    def predict_back(self, out_code, type_code=-1):
        return self.__predict(
            out_code, type_code,
            self.out_dimensions, self.in_dimensions,
            self.out_coords, self.in_coords,
            self.out_point_activate, False
        )

    """
        Функция, производящая добавление пары кодов в каждый кластер точки комбинаторного пространства

        in_code, out_code - входной и выходной бинарные векторы кодов соответствующих размерностей
        step_number - переменная, которая в текущей версии программы используется как отладочная

        Возвращается флаг добавления кластера (True - добавлен, False - не добавлен)
    """
    def add(self, in_code, out_code, step_number=None):

        CombSpaceExceptions.none(in_code, "Не определён аргумент")
        CombSpaceExceptions.none(out_code, "Не определён аргумент")
        CombSpaceExceptions.eq(len(out_code), self.out_dimensions, "Не совпадает размерность")
        CombSpaceExceptions.eq(len(in_code), self.in_dimensions, "Не совпадает размерность")
        CombSpaceExceptions.code_value(out_code)
        CombSpaceExceptions.code_value(in_code)

        if len(self.clusters) < self.max_clusters_per_point:

            in_x = np.array(in_code)[self.in_coords]
            out_x = np.array(out_code)[self.out_coords]

            is_modify_cluster = False

            # TODO: Возможно, проверять активацию не нужно, поскольку это будет отсекаться по скалярному
            # TODO: произведению при подсчёте корелляции
            if np.sum(in_x) >= self.in_point_activate and np.sum(out_x) >= self.out_point_activate:

                for cluster_id, cluster in enumerate(self.clusters):
                    if cluster.modify(in_x, out_x) is CLUSTER.MODIFY:
                        # tmp = self.statistics[cluster_id]
                        # tmp.append(step_number)
                        # self.statistics[cluster_id] = tmp
                        is_modify_cluster = True
                    # else:
                    #     tmp = self.statistics[cluster_id]
                    #     tmp.append(-1)
                    #     self.statistics[cluster_id] = tmp

                if not is_modify_cluster:
                    # self.statistics.append([step_number])
                    self.clusters.append(
                        self.cluster_class(
                            in_sub_code=in_x, out_sub_code=out_x,
                            in_cluster_modify=self.in_cluster_modify,
                            out_cluster_modify=self.out_cluster_modify,
                            binarization=self.binarization,
                            lr=self.lr, is_modify_lr=self.is_modify_lr
                        )
                    )
                    return True
        return False
