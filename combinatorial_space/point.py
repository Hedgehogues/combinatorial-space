import numpy as np
from combinatorial_space.cluster_factory import ClusterFactory

"""
    Точка комбинаторного пространства. Каждая точка содержит набор кластеров

    in_threshold_modify, out_threshold_modify - порог активации кластера. Если скалярное произведение базового 
    вектора кластера на новый вектор больше порога, то будет пересчитан веса кластера, выделяющие первую главную
    компоненту
    in_threshold_activate, out_threshold_activate - порог активации точки комбинаторного пространства. Если кол-во
    активных битов больше порога, то будет инициирован процесс модификации существующих кластеров, а также будет
    добавлен новый кластер
    threshold_bin - порог бинаризации кода
    count_in_demensions, count_out_demensions - размер входного и выходного векторов в точке комб. пространства
    in_size, out_size - количество случайных битов входного/выходного вектора
    base_lr - начальное значение скорости обучения
    is_modify_lr - модификация скорости обучения пропорционально номер шага
    max_cluster_per_point - максимальное количество кластеров в точке

    cluster_factory - фабричный метод для создания кластеров
"""


class Point:
    def __init__(self,
                 in_threshold_modify, out_threshold_modify,
                 in_threshold_activate, out_threshold_activate,
                 threshold_bin,
                 in_size, out_size,
                 count_in_demensions, count_out_demensions,
                 base_lr, is_modify_lr,
                 max_cluster_per_point,
                 cluster_factory=ClusterFactory()):
        # TODO: count_in_demensions <= 0
        # TODO: in_size <= 0
        self.in_coords = np.random.random_integers(0, in_size - 1, count_in_demensions)
        self.out_coords = np.random.random_integers(0, out_size - 1, count_out_demensions)
        self.count_in_demensions, self.count_out_demensions = count_in_demensions, count_out_demensions
        self.clusters = []
        self.in_threshold_modify, self.out_threshold_modify = in_threshold_modify, out_threshold_modify
        self.in_threshold_activate, self.out_threshold_activate = in_threshold_activate, out_threshold_activate
        self.threshold_bin = threshold_bin
        self.base_lr = base_lr
        self.is_modify_lr = is_modify_lr
        self.max_cluster_per_point = max_cluster_per_point
        self.cluster_factory = cluster_factory

    """
        Осуществление выбора оптимального кластера при прямом предсказании 

        in_code - входной вектор 
        type_code - тип возвращаемого кода (с -1 или с 0)

        Возвращается оптимальный выходной вектор
    """

    def predict_front(self, in_code, type_code=-1):
        assert len(in_code) == self.count_in_demensions, "Не совпадает заданная размерность с поданой"
        if not (type_code == -1 or type_code == 0):
            raise ValueError("Неверное значение type_code")

        in_x = np.array(in_code)[self.in_coords]
        is_active = np.sum(in_x) > self.in_threshold_activate
        opt_dot = -np.inf
        opt_out_code = None
        if is_active:
            for cluster in self.clusters:
                dot, out_x = cluster.predict_front(in_x)
                if dot > opt_dot:
                    opt_dot = dot
                    opt_out_code = np.array([0] * self.count_out_demensions)
                    if type_code == -1:
                        out_x[out_x == 0] = -1
                    opt_out_code[self.out_coords] = out_x
        return opt_out_code

    """
        Осуществление выбора оптимального кластера при обратном предсказании 

        out_code - выходной вектор
        type_code - тип возвращаемого кода (с -1 или с 0)

        Возвращается оптимальный входной вектор
    """

    def predict_back(self, out_code, type_code=-1):
        assert len(out_code) == self.count_in_demensions, "Не совпадает заданная размерность с поданой"
        if not (type_code == -1 or type_code == 0):
            raise ValueError("Неверное значение type_code")
        out_x = np.array(out_code)[self.out_coords]
        is_active = np.sum(out_x) > self.out_threshold_activate
        opt_dot = -np.inf
        opt_in_code = None
        if is_active:
            for cluster in self.clusters:
                dot, in_x = cluster.predict_back(out_x)
                if dot > opt_dot:
                    opt_dot = dot
                    opt_in_code = np.array([0] * self.count_in_demensions)
                    if type_code == -1:
                        in_x[in_x == 0] = -1
                    opt_in_code[self.in_coords] = in_x
        return opt_in_code

    """
        Функция, производящая добавление пары кодов в каждый кластер точки комбинаторного пространства

        in_code, out_code - входной и выходной бинарные векторы кодов соответствующих размерностей

        Возвращается количество произведённых модификаций внутри кластеров точки, флаг добавления кластера
        (True - добавлен, False - не добавлен)
    """

    def add(self, in_code, out_code=None):
        in_x = np.array(in_code)[self.in_coords]
        out_x = np.array(out_code)[self.out_coords]
        count_modify = 0
        count_fails = 0

        # Возможно, проверять активацию не нужно, поскольку это будет отсекаться по скалярному произведению
        # при подсчёте корелляции
        is_active = np.sum(in_x) > self.in_threshold_activate and \
                    np.sum(out_x) > self.out_threshold_activate
        if len(self.clusters) < self.max_cluster_per_point:
            if is_active:
                for cluster in self.clusters:
                    if cluster.modify(in_x, out_x):
                        count_modify += 1
                    else:
                        count_fails += 1
                return count_fails, count_modify, False
            else:
                self.clusters.append(
                    self.cluster_factory.get()(
                        base_in_subvector=in_x,
                        base_out_subvector=out_x,
                        in_threshold_modify=self.in_threshold_modify,
                        out_threshold_modify=self.out_threshold_modify,
                        base_lr=self.base_lr, is_modify_lr=self.is_modify_lr
                    )
                )
                return count_fails, count_modify, True
        return count_fails, count_modify, False