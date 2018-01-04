from copy import deepcopy
from enum import Enum

import Levenshtein
import numpy as np
from combinatorial_space.point import Point


class PREDICT_ENUM(Enum):
    THERE_ARE_NOT_ACTIVE_POINTS = 0
    THERE_ARE_NON_ACTIVE_POINTS = 1
    ACCEPT = 2


"""
    Миниколонка. Миниколонка - это набор точек комбинаторного пространства
    
    space_size - количество точек комбинаторного пространства
    max_cluster_per_point - максимальное количество кластеров в точке
    max_count_clusters - максмальное суммарное количество кластеров по всем точкам комбинаторного пространства
    in_threshold_modify, out_threshold_modify - порог активации кластера. Если скалярное произведение базового 
    вектора кластера на новый вектор больше порога, то будет пересчитан веса кластера, выделяющие первую главную
    компоненту
    threshold_bin - порог бинаризации кода
    in_threshold_activate, out_threshold_activate - порог активации точки комбинаторного пространства. Если кол-во
    активных битов больше порога, то будет инициирован процесс модификации существующих кластеров, а также будет
    добавлен новый кластер
    in_random_bits, out_random_bits - количество случайных битов входного/выходного вектора
    base_lr - начальное значение скорости обучения
    is_modify_lr - модификация скорости обучения пропорционально номер шага
    count_in_demensions, count_out_demensions - размер входного и выходного векторов
    threshold_bits_controversy - порог противоречия для битов кодов
    out_non_zero_bits - число ненулевых бит в выходном векторе
"""
class Minicolumn:
    
    def __init__(self, space_size=60000, max_cluster_per_point=100,
                 max_count_clusters=1000000, seed=42,
                 in_threshold_modify=5, out_threshold_modify=0,
                 in_threshold_activate=5, out_threshold_activate=0,
                 threshold_bin=0.1,
                 in_random_bits=24, out_random_bits=10,
                 base_lr=0.01, is_modify_lr=True,
                 count_in_demensions=256, count_out_demensions=16,
                 threshold_bits_controversy=0.1,
                 out_non_zero_bits=6, count_active_point=30, class_point=Point):

        if seed is None or \
            space_size is None or in_threshold_modify is None or out_threshold_modify is None or \
            in_threshold_activate is None or out_threshold_activate is None or \
            in_random_bits is None or out_random_bits is None or \
            threshold_bin is None or is_modify_lr is None or \
            base_lr is None or max_cluster_per_point is None or \
            count_in_demensions is None or count_out_demensions is None or \
            max_count_clusters is None or threshold_bits_controversy is None or \
            out_non_zero_bits is None or class_point is None or \
            count_active_point is None or count_active_point < 0 or \
            max_count_clusters <= 0 or space_size <= 0 or \
            in_random_bits > count_in_demensions or out_random_bits > count_out_demensions or \
            max_cluster_per_point < 0 or \
            out_random_bits < 0 or in_random_bits < 0 or \
            in_threshold_modify < 0 or out_threshold_modify < 0 or base_lr < 0 or \
            in_threshold_activate < 0 or out_threshold_activate < 0 or \
            count_in_demensions < 0 or count_out_demensions < 0 or \
            threshold_bin < 0 or type(is_modify_lr) is not bool or \
            threshold_bits_controversy < 0 or out_non_zero_bits < 0:
                raise ValueError("Неожиданное значение переменной")

        self.space = np.array(
            [
                class_point(
                    in_threshold_modify, out_threshold_modify,
                    in_threshold_activate, out_threshold_activate,
                    threshold_bin,
                    in_random_bits, out_random_bits,
                    count_in_demensions, count_out_demensions,
                    base_lr, is_modify_lr,
                    max_cluster_per_point
                ) for _ in range(space_size)
            ]
        )
        self.count_clusters = 0
        self.max_count_clusters = max_count_clusters
        self.count_in_demensions, self.count_out_demensions = count_in_demensions, count_out_demensions
        self.threshold_bits_controversy = threshold_bits_controversy
        self.out_non_zero_bits = out_non_zero_bits
        self.count_active_point = count_active_point

        self.__threshold_active = None
        self.__threshold_in_len = None
        self.__threshold_out_len = None
        self.__clusters_of_points = None
        self.__active_clusters = None
        
        np.random.seed(seed)

    def __none_exeption(self, obj):
        if obj is None:
            raise ValueError("Значение аргумента недопустимо")

    def __code_value_exeption(self, code):
        # Значения выходного вектора могут быть равны 0 или 1
        if np.sum(np.uint8(np.logical_not(np.array(code) != 0) ^ (np.array(code) != 1))) > 0:
            raise ValueError("Значение аргумента может принимать значение 0 или 1")

    def __len_exeption(self, obj_len, target_len):
        assert obj_len == target_len, "Не совпадает заданная размерность с поданой"

    """
        Получение выходного кода по входному. Прямое предсказание в каждой точке комбинаторного пространства
        
        in_code - входной код
        
        Возвращаемые значения: непротиворечивость, выходной код. В случае отсутствия хотя бы одной активной точки,
        возвращается бесконечное значение противоречивости
    """
    def front_predict(self, in_code):
        self.__none_exeption(in_code)
        self.__len_exeption(len(in_code), self.count_in_demensions)
        self.__code_value_exeption(in_code)

        out_code = [0] * self.count_out_demensions
        count = np.array([0] * self.count_out_demensions)
        for point in self.space:
            out_code_local = point.predict_front(in_code, -1)
            
            # Неактивная точка
            if out_code_local is None:
                continue

            count += np.uint8(out_code_local != 0)
            out_code += out_code_local

        # TODO: возможно, assert стоит заменить на
        # TODO: return np.inf, out_code
        # TODO: необходимо создавать новый кластер в обучении без учителя (а что делать в обучении с учителем?)
        non_zeros = np.sum(np.uint8(np.array(count) == 0))
        if non_zeros != 0:
            if non_zeros == self.count_out_demensions:
                return None, None, PREDICT_ENUM.THERE_ARE_NOT_ACTIVE_POINTS
            else:
                return None, None, PREDICT_ENUM.THERE_ARE_NON_ACTIVE_POINTS


        controversy = np.sum(np.uint8(np.abs(np.divide(out_code, count)) < self.threshold_bits_controversy))
        out_code[out_code <= 0] = 0
        out_code[out_code > 0] = 1
        return controversy, out_code, PREDICT_ENUM.ACCEPT
    
    """
        Получение входного кода по выходному. Обратное предсказание в каждой точке комбинаторного пространства
        
        out_code - выходной код
        
        Возвращаемые значения: непротиворечивость, входной код
    """
    def back_predict(self, out_code):
        self.__none_exeption(out_code)
        self.__len_exeption(len(out_code), self.count_out_demensions)
        self.__code_value_exeption(out_code)

        in_code = [0] * self.count_in_demensions
        count = [0] * self.count_in_demensions
        for point in self.space:
            in_code_local = point.predict_back(out_code, -1)
            
            # Неактивная точка
            if in_code_local is None:
                continue
            
            __count = np.uint8(in_code_local != 0)
            count += __count
            in_code += in_code_local

        # TODO: возможно, assert стоит заменить на
        # TODO: return np.inf, out_code
        # TODO: необходимо создавать новый кластер в обучении без учителя (а что делать в обучении с учителем?)
        non_zeros = np.sum(np.uint8(np.array(count) == 0))
        if non_zeros != 0:
            if non_zeros == self.count_in_demensions:
                return None, None, PREDICT_ENUM.THERE_ARE_NOT_ACTIVE_POINTS
            else:
                return None, None, PREDICT_ENUM.THERE_ARE_NON_ACTIVE_POINTS

        controversy = np.sum(np.uint8(np.abs(in_code / count) < self.threshold_bits_controversy))
        in_code[in_code <= 0] = 0
        in_code[in_code > 0] = 1
        return controversy, in_code, PREDICT_ENUM.ACCEPT

    def __sleep_process_clusters(self, point):

        for cluster_ind, cluster in enumerate(point.clusters):
            in_active_mask = np.abs(cluster.in_w) > self.__threshold_active
            out_active_mask = np.abs(cluster.out_w) > self.__threshold_active

            if len(cluster.in_w[in_active_mask]) > self.__threshold_in_len and \
                            len(cluster.out_w[out_active_mask]) > self.__threshold_out_len:

                # Подрезаем кластер
                cluster.in_w[~in_active_mask] = 0
                cluster.out_w[~out_active_mask] = 0

                self.__active_clusters.append(cluster)
                self.__clusters_of_points[-1].append(cluster_ind)
            else:
                self.count_clusters -= 1

    def __sleep_remove_the_same_clusters(self, point):

        # Удаляем одинаковые кластеры (те кластеры, у которых одинаковые базовые векторы)
        the_same_clusters = 0
        point.clusters = []
        for cluster_i in range(len(self.__active_clusters)):
            is_exist_the_same = False
            for cluster_j in range(cluster_i + 1, len(self.__active_clusters)):
                if np.sum(np.uint8(self.__active_clusters[cluster_i].in_w == \
                   self.__active_clusters[cluster_j].in_w)) == len(self.__active_clusters[cluster_i].in_w) \
                        and \
                   np.sum(np.uint8(self.__active_clusters[cluster_i].out_w == \
                   self.__active_clusters[cluster_j].out_w)) == len(self.__active_clusters[cluster_i].out_w):
                    is_exist_the_same = True
                    continue
            if not is_exist_the_same:
                point.clusters.append(self.__active_clusters[cluster_i])
            else:
                the_same_clusters += 1
                self.count_clusters -= 1
        return the_same_clusters

    """
        Этап сна
        
        threshold_active - порог активности бита внутри кластера (вес в преобразовании к первой главной компоненте), 
        выше которого активность остаётся
        threshold_in_len, threshold_out_len - порог количества ненулевых битов
        
        Возвращается количество одинаковых кластеров
    """    
    def sleep(self, threshold_active=0.75, threshold_in_len=4, threshold_out_len=0):
        self.__none_exeption(threshold_active)
        self.__none_exeption(threshold_in_len)
        self.__none_exeption(threshold_out_len)
        if threshold_active < 0 or threshold_active > 1:
            raise ValueError("Неожиданное значение переменной")
        if threshold_in_len < 0:
            raise ValueError("Неожиданное значение переменной")
        if threshold_out_len < 0:
            raise ValueError("Неожиданное значение переменной")

        the_same_clusters = 0

        self.__threshold_active = threshold_active
        self.__threshold_in_len = threshold_in_len
        self.__threshold_out_len = threshold_out_len
        self.__clusters_of_points = []

        for point_ind, point in enumerate(self.space):
            self.__clusters_of_points.append([])
            self.__active_clusters = []

            # Отбор наиболее информативных кластеров
            self.__sleep_process_clusters(point)
                    
            # Удаляем одинаковые кластеры (те кластеры, у которых одинаковые базовые векторы)
            the_same_clusters += self.__sleep_remove_the_same_clusters(point)
            
        return self.__clusters_of_points, the_same_clusters
        
    """
        Проверям: пора ли спать
    """
    def is_sleep(self):
        return self.count_clusters > self.max_count_clusters
    
    def __code_alignment(self, code):
        count_active_bits = np.sum(code)
        if count_active_bits > self.out_non_zero_bits:
            active_bits = np.where(code == 1)
            count_active_bits = active_bits.shape[0]
            stay_numbers = np.random.choice(
                count_active_bits, self.out_non_zero_bits, replace=False
            )
            active_bits = active_bits[stay_numbers]
            code_mod = np.zeros(code.shape[0])
            code_mod[active_bits] = 1
        elif count_active_bits < self.out_non_zero_bits:
            non_active_bits = np.where(code == 0)
            count_non_active_bits = non_active_bits.shape[0]
            count_active_bits = code.shape[0] - count_non_active_bits
            stay_numbers = np.random.choice(
                count_non_active_bits, self.out_non_zero_bits - count_active_bits, replace=False
            )
            non_active_bits = non_active_bits[stay_numbers]
            code_mod = deepcopy(code)
            code_mod[non_active_bits] = 1
        else:
            code_mod = deepcopy(code)
        return code_mod

    """
        Этап обучения без учителя
        
        Делается предсказание для всех переданных кодов и выбирается самый непротиворечивый из них, 
        либо констатируется, что такого нет.
        
        Для каждой активной точки выбирается наиболее подходящий кластер. Его предсказание учитывается в качестве
        ответа. Для конкретной точки все остальные кластеры учтены не будут.
        
        В качестве результата непротиворечивости берём среднее значение по ответам делёное на число активных точек.
        
        in_codes - входные коды в разных контекстах
        threshold_controversy_in, threshold_controversy_out - порого противоречивости для кодов
        
        Возвращается оптимальный код, порядковый номер контекста-победителя, 
        количество фэйлов во входном и выходном векторах
    """
    def unsupervised_learning(self, in_codes, threshold_controversy_in=3, threshold_controversy_out=3):

        self.__none_exeption(in_codes)
        self.__none_exeption(threshold_controversy_in)
        self.__none_exeption(threshold_controversy_out)

        self.__code_value_exeption(in_codes)
        if threshold_controversy_out < 0:
            raise ValueError("Неожиданное значение переменной")
        if threshold_controversy_in < 0:
            raise ValueError("Неожиданное значение переменной")
        self.__len_exeption(len(in_codes), self.count_in_demensions)
        self.__code_value_exeption(in_codes)

        min_hamming = np.inf
        min_ind_hamming = -1
        min_out_code = None
        out_fail = 0
        in_fail = 0
        for index in range(len(in_codes)):

            # Не обрабатываются полностью нулевые коды
            if np.sum(in_codes[index]) == 0:
                continue

            # TODO: холодный старт. С чего начинать?
            controversy_out, out_code = self.front_predict(in_codes[index])

            # TODO: что если все коды противоречивые? как быть?
            if controversy_out > threshold_controversy_out:
                out_fail += 1
                continue
                
            # Удаляем или добавляем единицы (если их мало или много)
            out_code = self.__code_alignment(out_code)
            
            controversy_in, in_code = self.back_predict(out_code)
            
            if controversy_in > threshold_controversy_in:
                in_fail += 1
                continue
                
            
            hamming_dist = Levenshtein.hamming(''.join(map(str, in_code)), ''.join(map(str, in_codes[index])))
            if min_hamming < hamming_dist:
                min_hamming = hamming_dist
                min_ind_hamming = index
                min_out_code = out_code
            
        return min_out_code, min_ind_hamming, in_fail, out_fail
    
    """
        Этап обучения без учителя
        
        Делается предсказание для всех переданных кодов и выбирается самый непротиворечивый из них, 
        либо констатируется, что такого нет.
        
        Для каждой активной точки выбирается наиболее подходящий кластер. Его предсказание учитывается в качестве
        ответа. Для конкретной точки все остальные кластеры учтены не будут.
        
        В качестве результата непротиворечивости берём среднее значение по ответам делёное на число активных точек.
        
        codes - входные коды в разных контекстах
        threshold_controversy_in, threshold_controversy_out - порого противоречивости для кодов
        
        Возвращается ...
    """
    def supervised_learning(self, in_codes, out_codes, threshold_controversy_out):
                       
        min_hamming = np.inf
        min_ind_hamming = -1
        min_out_code = None
        out_fail = 0
        for index in range(len(in_codes)):

            # Не обрабатываются полностью нулевые коды
            if np.sum(in_codes[index]) == 0:
                continue

            # TODO: холодный старт. С чего начинать?
            controversy_out, out_code = self.front_predict(in_codes[index])
            
            if controversy_out > threshold_controversy_out:
                out_fail += 1
                continue                
            
            hamming_dist = Levenshtein.hamming(''.join(map(str, out_code)), ''.join(map(str, out_codes[index])))
            if min_hamming < hamming_dist:
                min_hamming = hamming_dist
                min_ind_hamming = index
                min_out_code = out_code
            
        return min_out_code, min_ind_hamming, out_fail
    
    
    """
        Этап обучения с учителем
        
        Создание и модификация кластеров на основе пары кодов: входной и выходной
        
        in_code, out_code - входной и выходной коды
        threshold_controversy_in, threshold_controversy_out - пороги противоречивости на входной и выходной коды
        
        Возвращается количество точек, которые оказались неактивными; количество модификаций кластеров;
        количество новых кластеров
    """
    def learn(self, in_codes, out_codes=None, threshold_controversy_in=20, threshold_controversy_out=6):
        if self.is_sleep():
            return None, None, None
        
        if out_codes is not None:
            in_code, out_code = self.supervised_learning(in_codes, out_codes, threshold_controversy_out)
        else:
            in_code, out_code = self.unsupervised_learning(in_codes, threshold_controversy_in, threshold_controversy_out)
            
        
        count_fails = 0
        count_modify = 0
        count_adding = 0
        
        for point in self.space:
            __count_fails, __count_modify, __count_adding = point.add(in_code, out_code)
            count_modify += __count_modify
            count_fails += __count_fails
            count_adding += __count_adding
            self.count_clusters += np.uint(__count_adding)
        return count_fails, count_modify, count_adding
