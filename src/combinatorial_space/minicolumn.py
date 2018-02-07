from copy import deepcopy

import Levenshtein
import numpy as np

from src.combinatorial_space.enums import MINICOLUMN_LEARNING, MINICOLUMN
from src.combinatorial_space.expetions import CombSpaceExceptions
from src.combinatorial_space.point import Point, POINT_PREDICT


class Minicolumn:
    """
        Миниколонка. Миниколонка - это набор точек комбинаторного пространства

        space_size - количество точек комбинаторного пространства
        max_clusters_per_point - максимальное количество кластеров в точке
        max_clusters - максмальное суммарное количество кластеров по всем точкам комбинаторного пространства
        in_cluster_modify, out_cluster_modify - порог модификации кластера. Если скалярное произведение кластера
        на новый вектор больше порога, то будет пересчитаны веса кластера, выделяющие первую главную компоненту
        in_point_activate, out_point_activate - порог активации точки комбинаторного пространства. Если кол-во
        активных битов больше порога, то будет инициирован процесс модификации существующих кластеров или добавления
        нового кластера
        binarization - порог бинаризации кода. Скалярное произведение вычисляенся [w >= bin], где w - вес
        lr - начальное значение скорости обучения
        is_modify_lr - модификация скорости обучения пропорционально номеру шага
        in_random_bits, out_random_bits - количество случайных битов входного/выходного вектора для точки комб.
        пространства
        in_dimensions, out_dimensions - размер входного и выходного векторов
        controversy - порог противоречия для битов кодов
        code_alignment - число ненулевых бит в выходном векторе
        min_active_point - минимальное количество активных точек, необходимых, чтобы кластер был распознан
        in_code_activate, out_code_activate - количество активных битов во входном и выходном векторах
        seed - инициализация случайным состоянием
    """
    def __init__(self, space_size=10000, max_clusters_per_point=100,
                 max_clusters=1000000,
                 in_cluster_modify=5, out_cluster_modify=0,
                 in_point_activate=5, out_point_activate=0,
                 binarization=0.1, lr=0.01, is_modify_lr=True,
                 in_random_bits=24, out_random_bits=10,
                 in_dimensions=256, out_dimensions=16,
                 controversy=0.1,
                 code_alignment=6, min_active_points=80,
                 in_code_activate=14, out_code_activate=3,
                 seed=42, class_point=Point):

        CombSpaceExceptions.none(seed)
        CombSpaceExceptions.none(space_size)
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
        CombSpaceExceptions.none(controversy)
        CombSpaceExceptions.none(max_clusters)
        CombSpaceExceptions.none(code_alignment)
        CombSpaceExceptions.none(class_point)
        CombSpaceExceptions.none(min_active_points)

        CombSpaceExceptions.less_or_equal(max_clusters, 0)
        CombSpaceExceptions.less_or_equal(space_size, 0)

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
        CombSpaceExceptions.less(controversy, 0)
        CombSpaceExceptions.less(code_alignment, 0)
        CombSpaceExceptions.less(in_code_activate, 0)
        CombSpaceExceptions.less(out_code_activate, 0)
        CombSpaceExceptions.less(min_active_points, 0)

        CombSpaceExceptions.is_type(is_modify_lr, bool)

        CombSpaceExceptions.more(in_random_bits, in_dimensions)
        CombSpaceExceptions.more(out_random_bits, out_dimensions)
        CombSpaceExceptions.more(code_alignment, out_dimensions)

        np.random.seed(seed)

        self.space = [
            class_point(
                in_cluster_modify, out_cluster_modify,
                in_point_activate, out_point_activate,
                binarization, lr, is_modify_lr,
                in_random_bits, out_random_bits,
                in_dimensions, out_dimensions,
                max_clusters_per_point
            ) for _ in range(space_size)
        ]
        self.max_clusters = max_clusters
        self.controversy = controversy
        self.alignment = code_alignment
        self.min_active_points = min_active_points
        self.in_dimensions, self.out_dimensions = in_dimensions, out_dimensions
        self.in_code_activate, self.out_code_activate = in_code_activate, out_code_activate

        self.count_clusters = 0

        # For sleep methods
        self.__threshold_active = None
        self.__clusters_of_points = None
        self.__active_clusters = None

        # Statistics (debug) variables
        self.statistics = []

    def __predict_prepare_code(self, code, count):
        controversy = np.sum(
            np.uint8(np.abs(np.divide(code[count != 0], count[count != 0])) < self.controversy)
        )
        code[code <= 0] = 0
        code[code > 0] = 1

        return controversy

    def __select_predict_function(self, point, code, is_front):
        if is_front:
            return point.predict_front(code)
        else:
            return point.predict_back(code)

    def __predict(self, code, dimensions_0, dimensions_1, is_front):
        CombSpaceExceptions.none(code, 'Не определён входной аргумент')
        CombSpaceExceptions.eq(len(code), dimensions_0, 'Не совпадает размерность')
        CombSpaceExceptions.code_value(code)

        total_sub_code = np.zeros(dimensions_1, dtype=np.int)
        count = np.zeros(dimensions_1, dtype=np.int)
        active_points = 0
        # pool = multiprocessing.Pool(processes=4)

        for point in self.space:

            predicted_sub_code, status = self.__select_predict_function(point, code, is_front)

            if status is POINT_PREDICT.ACTIVE:
                active_points += 1
                CombSpaceExceptions.none(predicted_sub_code, 'Не определён входной аргумент')
                CombSpaceExceptions.eq(len(predicted_sub_code), dimensions_1, 'Не совпадает размерность')

                count += np.uint8(predicted_sub_code != 0)
                total_sub_code += predicted_sub_code

        if active_points < self.min_active_points:
            return None, None, MINICOLUMN_LEARNING.INACTIVE_POINTS

        controversy = self.__predict_prepare_code(total_sub_code, count)
        return controversy, total_sub_code, MINICOLUMN_LEARNING.ACCEPT

    """
        Получение выходного кода по входному. Прямое предсказание в каждой точке комбинаторного пространства
        
        in_code - входной код
        
        Возвращаемые значения: непротиворечивость, выходной код. В случае отсутствия хотя бы одной активной точки,
        возвращается бесконечное значение противоречивости
    """
    def front_predict(self, in_code):
        return self.__predict(in_code, self.in_dimensions, self.out_dimensions, True)
    
    """
        Получение входного кода по выходному. Обратное предсказание в каждой точке комбинаторного пространства
        
        out_code - выходной код
        
        Возвращаемые значения: непротиворечивость, входной код
    """
    def back_predict(self, out_code):
        return self.__predict(out_code, self.out_dimensions, self.in_dimensions, False)

    # TODO: перенести внутрь Clusters
    def __sleep_process_clusters(self, point):

        for cluster_ind, cluster in enumerate(point.clusters):
            in_active_mask = np.abs(cluster.in_w) >= self.__threshold_active
            out_active_mask = np.abs(cluster.out_w) >= self.__threshold_active

            if len(cluster.in_w[in_active_mask]) >= self.in_code_activate and \
                    len(cluster.out_w[out_active_mask]) >= self.out_code_activate:

                # Подрезаем кластер
                cluster.in_w[~in_active_mask] = 0
                cluster.out_w[~out_active_mask] = 0

                self.__active_clusters.append(cluster)
                self.__clusters_of_points[-1].append(cluster_ind)
            else:
                self.count_clusters -= 1

    # TODO: перенести внутрь Clusters
    def __sleep_remove_the_same_clusters(self, point):
        # Удаляем одинаковые кластеры (те кластеры, у которых одинаковые базовые векторы)
        the_same_clusters = 0
        point.clusters = []
        for cluster_i in range(len(self.__active_clusters)):
            is_exist_the_same = False
            for cluster_j in range(cluster_i + 1, len(self.__active_clusters)):
                in_equal = np.array_equal(
                    self.__active_clusters[cluster_i].in_w,
                    self.__active_clusters[cluster_j].in_w
                )
                out_equal = np.array_equal(
                    self.__active_clusters[cluster_i].out_w,
                    self.__active_clusters[cluster_j].out_w
                )
                # TODO: рассмотреть случай, в котором in_equal == True, out_equal == False
                # TODO: сейчас такой случай считаем одинаковыми кластерами
                if in_equal and out_equal or in_equal and not out_equal:
                    is_exist_the_same = True
                    break

            if not is_exist_the_same:
                point.clusters.append(self.__active_clusters[cluster_i])
            else:
                the_same_clusters += 1
                self.count_clusters -= 1

        return the_same_clusters

    """
        Этап сна
        
        active - порог активности бита внутри кластера (вес в преобразовании к первой главной компоненте), 
        выше которого активность остаётся
        
        Возвращается количество одинаковых кластеров
    """    
    def sleep(self, activate=0.75):
        CombSpaceExceptions.none(activate, 'Не определён аргумент')
        CombSpaceExceptions.less(activate, 0, 'Недопустимое значение аргумента')
        CombSpaceExceptions.more(activate, 1, 'Недопустимое значение аргумента')

        the_same_clusters = 0

        self.__threshold_active = activate
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
        return self.count_clusters > self.max_clusters
    
    def __code_alignment(self, code):
        count_active_bits = np.sum(code)
        if count_active_bits > self.alignment:
            active_bits = np.where(code == 1)[0]
            count_active_bits = active_bits.shape[0]
            stay_numbers = np.random.choice(
                count_active_bits, self.alignment, replace=False
            )
            code_mod = np.zeros(code.shape[0])
            code_mod[active_bits[stay_numbers]] = 1
        elif count_active_bits < self.alignment:
            non_active_bits = np.where(code == 0)[0]
            count_non_active_bits = non_active_bits.shape[0]
            count_active_bits = code.shape[0] - count_non_active_bits
            stay_numbers = np.random.choice(
                count_non_active_bits, self.alignment - count_active_bits, replace=False
            )
            code_mod = deepcopy(code)
            code_mod[non_active_bits[stay_numbers]] = 1
        else:
            code_mod = deepcopy(code)

        return np.int8(code_mod)

    """
        Этап обучения без учителя (не написан один тест. см. тесты для выбора максимума)
        
        Делается предсказание для всех переданных кодов и выбирается самый непротиворечивый из них, 
        либо констатируется, что такого нет. Если существует несколько непротиворечивых кодов, то выбирается тот,
        который лучше всего подходит под узнанную закономерность. Если же ни один код не является ни противоречивым,
        ни узнанным, то генерируется случайный выход для наиболее активного кода.
        
        in_codes - входные коды в разных контекстах
        controversy_in, controversy_out - порого противоречивости для кодов
        
        Возвращаются: (Оптимальный входной код, оптимальный выходной код, порядковый номер контекста-победителя)
    """
    def unsupervised_learning(self, in_codes, controversy_in=3, controversy_out=3):

        CombSpaceExceptions.codes(in_codes, self.in_dimensions)
        CombSpaceExceptions.none(controversy_in)
        CombSpaceExceptions.none(controversy_out)
        CombSpaceExceptions.less(controversy_in, 0)
        CombSpaceExceptions.less(controversy_out, 0)

        min_hamming = np.inf
        min_ind_hamming = None
        min_out_code = None
        all_codes_not_active = True
        for index in range(len(in_codes)):

            # Не обрабатываются коды из большого кол-ва нулей
            if np.sum(in_codes[index]) < self.in_code_activate:
                continue

            all_codes_not_active = False

            # TODO: холодный старт. С чего начинать?
            # TODO: если ни один код не распознан (status всегда принимает значения не равные ACCEPT),
            # TODO: то создаём случайный выходной вектор для первого кода из всех входных
            #############
            # TODO: что делать, если в одном из контекстов мы не можем распознать ничего, а в других можем?
            # TODO: на данном этапе забиваем на такие коды
            #############
            controversy_out_predict, out_code, status = self.front_predict(in_codes[index])
            if status is MINICOLUMN_LEARNING.INACTIVE_POINTS:
                continue

            # TODO: что если все коды противоречивые? как быть?
            # TODO: на данном этапе такой код не добавляем
            ############
            if controversy_out_predict >= controversy_out:
                continue
                
            # Удаляем или добавляем единицы (если их мало или много)
            out_code = self.__code_alignment(out_code)
            
            controversy_in_predict, in_code, status = self.back_predict(out_code)
            # TODO: при обратном предсказании не распознано, а при прямом -- распознано
            # TODO: на данном этапе забиваем на такие коды
            ############
            if status is MINICOLUMN_LEARNING.INACTIVE_POINTS:
                continue

            if controversy_in_predict >= controversy_in:
                continue

            hamming_dist = Levenshtein.hamming(''.join(map(str, in_code)), ''.join(map(str, in_codes[index])))
            if min_hamming > hamming_dist:
                min_hamming = hamming_dist
                min_ind_hamming = index
                min_out_code = out_code

        if all_codes_not_active:
            return None, None, None

        if min_ind_hamming is None:
            # TODO: выбирается наиболее активный код
            # TODO: правильно ли это?
            ############
            min_ind_hamming = 0
            max_ones = -1
            for ind, code in enumerate(in_codes):
                sum_ones = np.sum(code)
                if sum_ones > max_ones:
                    max_ones = sum_ones
                    min_ind_hamming = ind

            # Генерируем случайный код
            min_out_code = self.__code_alignment(np.zeros(self.out_dimensions, dtype=np.int))

        return in_codes[min_ind_hamming], min_out_code, min_ind_hamming
    
    """
        Этап обучения с учителем (не протестирован)
        
        Делается предсказание для всех переданных кодов и выбирается самый непротиворечивый из них, 
        либо констатируется, что такого нет. Если существует несколько непротиворечивых кодов, то выбирается тот,
        который лучше всего подходит под узнанную закономерность.
        
        in_codes - входные коды в разных контекстах
        controversy_in, controversy_out - порого противоречивости для кодов
        
        Возвращаются: (Оптимальный входной код, оптимальный выходной код, порядковый номер контекста-победителя)
    """
    def supervised_learning(self, in_codes, out_codes, out_controversy=3):

        CombSpaceExceptions.codes(in_codes, self.in_dimensions)
        CombSpaceExceptions.codes(out_codes, self.out_dimensions)
        CombSpaceExceptions.none(out_controversy, 'Неопределён аргумент')
        CombSpaceExceptions.less(out_controversy, 'Недопустимое значение переменной')

        min_hamming = np.inf
        min_ind_hamming = None
        all_codes_not_active = True
        for index in range(len(in_codes)):

            # Не обрабатываются коды из большого кол-ва нулей
            if np.sum(in_codes[index]) < self.in_code_activate or \
                    np.sum(in_codes[index]) < self.out_code_activate:
                continue

            all_codes_not_active = False

            # TODO: холодный старт. С чего начинать?
            # TODO: если ни один код не распознан (accept всегда принимает значения не равные ACCEPT),
            # TODO: то берём первый попавшийся код (например, нулевой по счёту)
            #############
            # TODO: что делать, если в одном из контекстов мы не можем распознать ничего, а в других можем?
            # TODO: на данном этапе забиваем на такие коды
            #############
            controversy_out_predict, out_code, status = self.front_predict(in_codes[index])
            if status is MINICOLUMN_LEARNING.INACTIVE_POINTS:
                continue

            # TODO: что если все коды противоречивые? как быть?
            # TODO: на данном этапе такой код не добавляем
            ############
            if controversy_out_predict >= out_controversy:
                continue
            
            hamming_dist = Levenshtein.hamming(''.join(map(str, out_code)), ''.join(map(str, out_codes[index])))
            if min_hamming < hamming_dist:
                min_hamming = hamming_dist
                min_ind_hamming = index

        if all_codes_not_active:
            return None, None, None

        # Если ни один код не определился, то берём самый первый
        if min_ind_hamming is None:
            # TODO: выбирается наиболее активный код (правильно ли это?)
            ############
            min_ind_hamming = 0
            max_ones = -1
            for ind, in_code, out_code in enumerate(zip(in_codes, out_codes)):
                sum_in_ones = np.sum(in_code)
                sum_out_ones = np.sum(out_code)
                # TODO: не совсем корректный способ выбирать максимум, поскольку количество нулей в out может
                # TODO: быть большим, а в in -- маленьким. И наоборот
                if sum_in_ones + sum_out_ones > max_ones and sum_in_ones > 0 and sum_out_ones > 0:
                    max_ones = sum_in_ones + sum_out_ones
                    min_ind_hamming = ind

        return in_codes[min_ind_hamming], out_codes[min_ind_hamming], min_ind_hamming

    """
        Этап обучения (не протестирована)
        
        Создание и модификация кластеров на основе пары кодов: входной и выходной
        
        in_code, out_code - входной и выходной коды
        threshold_controversy_in, threshold_controversy_out - пороги противоречивости на входной и выходной коды
        
        Возвращается количество точек, которые оказались неактивными; количество модификаций кластеров;
        количество новых кластеров
    """
    def learn(self, in_codes, step_number, out_codes=None, in_controversy=20, out_controversy=6):

        if self.is_sleep():
            return None, None, MINICOLUMN.SLEEP

        if out_codes is not None:
            in_code, out_code, opt_ind = self.supervised_learning(
                in_codes, out_codes, out_controversy
            )
        else:
            in_code, out_code, opt_ind = self.unsupervised_learning(
                in_codes, in_controversy, out_controversy
            )

        if opt_ind is None or out_code is None or in_code is None:
            return opt_ind, out_code, MINICOLUMN.BAD_CODES

        for ind, point in enumerate(self.space):
            self.count_clusters += point.add(in_code, out_code, step_number)

        return opt_ind, out_code, MINICOLUMN.LEARN
