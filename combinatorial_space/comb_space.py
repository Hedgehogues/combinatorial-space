from copy import deepcopy
import Levenshtein
import numpy as np
                

class Minicolumn:
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
    in_size, out_size - количество случайных битов входного/выходного вектора
    base_lr - начальное значение скорости обучения
    is_modify_lr - модификация скорости обучения пропорционально номер шага
    count_in_demensions, count_out_demensions - размер входного и выходного векторов в точке комб. пространства
    threshold_bits_controversy - порог противоречия для битов кодов
    out_non_zero_bits - число ненулевых бит в выходном векторе
    """
    
    def __init__(self, space_size=60000, max_cluster_per_point=100,
                 max_count_clusters=1000000, seed=42, 
                 in_threshold_modify=5, out_threshold_modify=0, 
                 in_threshold_activate=5, out_threshold_activate=0,
                 threshold_bin=0.1,
                 in_size=256, out_size=16,
                 base_lr=0.01, is_modify_lr=True,
                 count_in_demensions=24, count_out_demensions=10,
                 threshold_bits_controversy=0.1,
                 out_non_zero_bits=6):
        self.space = np.array(
            [
                Point(
                    in_threshold_modify, out_threshold_modify,
                    in_threshold_activate, out_threshold_activate,
                    threshold_bin,
                    count_in_demensions, count_out_demensions,
                    in_size, out_size,
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
        
        np.random.seed(seed)
        
    """
    Получение выходного кода по входному. Прямое предсказание в каждой точке комбинаторного пространства
    
    in_code - входной код
    
    Возвращаемые значения: непротиворечивость, выходной код. В случае отсутствия хотя бы одной активной точки,
    возвращается бесконечное значение противоречивости
    """
    def front_predict(self, in_code):
        out_code = [0] * self.count_out_demensions
        count = [0] * self.count_out_demensions
        for point in self.space:
            __out_code = point.predict_front(in_code, -1)
            
            # Неактивная точка
            if __out_code is None:
                continue
                
            __count = np.uint8(__out_code != 0)
            count += __count
            out_code += __out_code
        if np.sum(count == 0) > 0:
            raise ValueError("Не все биты входного вектора учитываются")
            controversy = np.sum(np.abs(out_code / count) < self.threshold_controversy)
            out_code[out_code <= 0] = 0
            out_code[out_code > 0] = 1
            return controversy, out_code
        else:
            return np.inf, out_code
    
    """
    Получение входного кода по выходному. Обратное предсказание в каждой точке комбинаторного пространства
    
    out_code - выходной код
    
    Возвращаемые значения: непротиворечивость, входной код
    """
    def back_predict(self, out_code):
        in_code = [0] * self.count_in_demensions
        count = [0] * self.count_in_demensions
        for point in self.space:
            __in_code = point.predict_back(out_code, -1)
            
            # Неактивная точка
            if __in_code is None:
                continue
            
            __count = np.uint8(__in_code != 0)
            count += __count
            in_code += __in_code
        if np.sum(count == np.nan):
            raise ValueError("Не все биты выходного вектора учитываются")
        controversy = np.sum(np.abs(in_code / count) < self.threshold_controversy)
        in_code[in_code <= 0] = 0
        in_code[in_code > 0] = 1
        return controversy, in_code
        
    """
    Этап сна
    
    threshold_active - порог активности бита внутри кластера (вес в преобразовании к первой главной компоненте), 
    выше которого активность остаётся
    threshold_in_len, threshold_out_len - порог количества ненулевых битов
    """    
    def sleep(self, threshold_active=0.75, threshold_in_len=4, threshold_out_len=0):
        clusters_of_points = []
        the_same_clusters = 0
        for point_ind, point in enumerate(self.space):
            clusters_of_points.append([])
            active_clusters = []
            for cluster_ind, cluster in enumerate(point.clusters):
                in_active_mask = np.abs(cluster.in_w) > threshold_active
                out_active_mask = np.abs(cluster.out_w) > threshold_active
                
                if len(cluster.in_w[in_active_mask]) > threshold_in_len and \
                    len(cluster.out_w[out_active_mask]) > threshold_out_len:
                        
                    # Подрезаем кластер
                    cluster.base_in_subvector[~in_active_mask] = 0
                    cluster.base_out_subvector[~out_active_mask] = 0
                    
                    active_clusters.append(cluster)
                    clusters_of_points[-1].append(cluster_ind)
                else:
                    self.count_clusters -= 1
                    
            # Удаляем одинаковые кластеры (те кластеры, у которых одинаковые базовые векторы)
            point.clusters = []
            for cluster_i in range(len(active_clusters)):
                is_exist_the_same = False
                for cluster_j in range(cluster_i+1, len(active_clusters)):
                    if np.sum(np.uint8(active_clusters[cluster_i].base_in_subvector == \
                        active_clusters[cluster_j].base_in_subvector)) \
                        and \
                        np.sum(np.uint8(active_clusters[cluster_i].base_out_subvector == \
                        active_clusters[cluster_j].base_out_subvector)):
                            is_exist_the_same = True
                            continue
                if not is_exist_the_same:
                    point.clusters.append(active_clusters[cluster_i])
                else:
                    the_same_clusters += 1
                    self.count_clusters -= 1
            
        return clusters_of_points, the_same_clusters
        
    """
    Проверям: пора ли спать
    """
    def is_sleep(self):
        return self.count_clusters > self.max_count_clusters
    
    def code_alignment(self, code):
        if self.count_active_bits > self.out_non_zero_bits:
            active_bits = np.where(code == 1)
            count_active_bits = active_bits.shape[0]
            stay_numbers = np.random.choice(
                count_active_bits, self.out_non_zero_bits, replace=False
            )
            active_bits = active_bits[stay_numbers]
            code_mod = np.zeros(code.shape[0])
            code_mod[active_bits] = 1
        elif self.count_active_bits < self.out_non_zero_bits:
            non_active_bits = np.where(code == 0)
            count_non_active_bits = non_active_bits.shape[0]
            count_active_bits = code.shape[0] - count_non_active_bits
            stay_numbers = np.random.choice(
                count_non_active_bits, self.out_non_zero_bits - count_active_bits, replace=False
            )
            non_active_bits = non_active_bits[stay_numbers]
            code_mod = deepcopy(code)
            code_mod[non_active_bits] = 1
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
    def unsupervised_learning(self, in_codes, threshold_controversy_in, threshold_controversy_out):
                       
        min_hamming = np.inf
        min_ind_hamming = -1
        min_out_code = None
        out_fail = 0
        in_fail = 0
        for index in range(len(in_codes)):

            # Не обрабатываются полностью нулевые коды
            if np.sum(in_codes[index]) == 0:
                continue

            controversy_out, out_code = self.front_predict(in_codes[index])
            
            if controversy_out > threshold_controversy_out:
                out_fail += 1
                continue
                
            # Удаляем или добавляем единицы (если их мало или много)
            out_code = self.code_alignment(out_code)
            
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
        in_fail = 0
        for index in range(len(in_codes)):

            # Не обрабатываются полностью нулевые коды
            if np.sum(in_codes[index]) == 0:
                continue
            
            controversy_out, out_code = self.front_predict(in_codes[index])
            
            if controversy_out > threshold_controversy_out:
                out_fail += 1
                continue                
            
            hamming_dist = Levenshtein.hamming(''.join(map(str, out_code)), ''.join(map(str, out_codes[index])))
            if min_hamming < hamming_dist:
                min_hamming = hamming_dist
                min_ind_hamming = index
                min_out_code = out_code
            
        return min_out_code, min_ind_hamming, in_fail, out_fail
    
    
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
