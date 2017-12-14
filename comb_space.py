# TODO: CUPY
# TODO: Правило Ойо не сходится по непонятным причинам

class Cluster:
    """
    Кластер в точке комбинаторного пространства
    
    base_in_subvector, base_out_subvector - бинарный код, образующий кластер. Между образующим кодом и новым кодом
    будет вычисляться скалярное произведение
    in_threshold_modify, out_threshold_modify - порог активации кластера. Если скалярное произведение базового 
    вектора на новый вектор больше порога, то будет пересчитан веса кластера, выделяющие первую главную компоненту
    base_lr - начальное значение скорости обучения
    is_modify_lr - модификация скорости обучения пропорционально номер шага
    """    
    def __init__(self, 
                 base_in_subvector, base_out_subvector,
                 in_threshold_modify, out_threshold_modify,
                 base_lr=0.5,
                 is_modify_lr=True):
        self.base_in_subvector = np.array(base_in_subvector)
        self.base_out_subvector = np.array(base_out_subvector)
        self.in_threshold_modify = in_threshold_modify
        self.out_threshold_modify = out_threshold_modify
        self.base_lr = base_lr
        self.in_w = np.random.rand(len(base_in_subvector))
        self.out_w = np.random.rand(len(base_out_subvector))
        self.is_modify_lr = is_modify_lr
        self.count_modifing = 0
        
    """
    Функция, производящая модификацию пары кодов кластера точки комбинаторного пространства
    
    in_x, out_x - входной и выходной бинарные векторы подкодов соответствующих размерностей
    
    Возвращается 1, если была произведена модификация весов (т.е. кластер был активирован). В противном случае
    возвращается 0
    """
    def modify(self, in_x, out_x):
        in_y = np.dot(in_x, self.in_w)
        out_y = np.dot(out_x, self.out_w)
        
        if np.dot(in_x, self.base_in_subvector) > self.in_threshold_modify and \
            np.dot(out_x, self.base_out_subvector) > self.out_threshold_modify:
                self.count_modifing += 1
                if self.is_modify_lr:
                    delta_in = np.array((self.base_lr/self.count_modifing)*in_y*in_x)
                    delta_out = np.array((self.base_lr/self.count_modifing)*out_y*out_x)
                    # Правило Ойо почему-то расходится
#                   self.in_w = self.in_w + (self.base_lr/self.count_modifing)*in_y*(in_x - in_y*self.in_w)
#                   self.out_w = self.out_w + (self.base_lr/self.count_modifing)*out_y*(out_x - out_y*self.out_w)
                else:
                    delta_in = np.array(self.base_lr*in_y*in_x)
                    delta_out = np.array(self.base_lr*out_y*out_x)
                    # Правило Ойо почему-то расходится
#                   self.in_w = self.in_w + (self.base_lr*in_y*(in_x - in_y*self.in_w)
#                   self.out_w = self.out_w + (self.base_lr*out_y*(out_x - out_y*self.out_w)
                self.in_w = np.divide((self.in_w + delta_in), (np.sum(self.in_w**2)**(0.5)))
                self.out_w = np.divide((self.out_w + delta_out), (np.sum(self.out_w**2)**(0.5)))

                return 1
        return 0
                
        
class Point:
    """
    Точка комбинаторного пространства. Каждая точка содержит набор кластеров
    
    in_threshold_modify, out_threshold_modify - порог активации кластера. Если скалярное произведение базового 
    вектора кластера на новый вектор больше порога, то будет пересчитан веса кластера, выделяющие первую главную
    компоненту
    in_threshold_activate, out_threshold_activate - порог активации точки комбинаторного пространства. Если кол-во
    активных битов больше порога, то будет инициирован процесс модификации существующих кластеров, а также будет
    добавлен новый кластер
    count_in_demensions, count_out_demensions - размер входного и выходного векторов в точке комб. пространства
    in_size, out_size - количество случайных битов входного/выходного вектора
    base_lr - начальное значение скорости обучения
    is_modify_lr - модификация скорости обучения пропорционально номер шага
    max_cluster_per_point - максимальное количество кластеров в точке
    """
    
    def __init__(self,
                 in_threshold_modify, out_threshold_modify,
                 in_threshold_activate, out_threshold_activate,
                 in_size, out_size,
                 count_in_demensions, count_out_demensions,
                 base_lr, is_modify_lr,
                 max_cluster_per_point):
        self.in_coords = np.random.random_integers(0, in_size-1, count_in_demensions)
        self.out_coords = np.random.random_integers(0, out_size-1, count_out_demensions)
        self.clusters = []
        self.in_threshold_modify = in_threshold_modify
        self.out_threshold_modify = out_threshold_modify
        self.in_threshold_activate = in_threshold_activate
        self.out_threshold_activate = out_threshold_activate
        self.base_lr = base_lr
        self.is_modify_lr = is_modify_lr
        self.max_cluster_per_point = max_cluster_per_point        
    
    """
    Функция, производящая добавление пары кодов в каждый кластер точки комбинаторного пространства
    
    in_code, out_code - входной и выходной бинарные векторы кодов соответствующих размерностей
    
    Возвращается количество произведённых модификаций внутри кластеров точки, флаг добавления кластера
    (True - добавлен, False - не добавлен)
    """
    def add(self, in_code, out_code):
        in_x = np.array(in_code)[self.in_coords]
        out_x = np.array(out_code)[self.out_coords]
        count_modify = 0
        count_fails = 0
        
        is_active = np.sum(in_x) > self.in_threshold_activate and np.sum(out_x) > self.out_threshold_activate
        if is_active and len(self.clusters) < self.max_cluster_per_point:
            for cluster in self.clusters:
                if cluster.modify(in_x, out_x):
                    count_modify += 1
                else:
                    count_fails += 1
            self.clusters.append(
                Cluster(
                    base_in_subvector=in_x,
                    base_out_subvector=out_x,
                    in_threshold_modify=self.in_threshold_modify, 
                    out_threshold_modify=self.out_threshold_modify,
                    base_lr=self.base_lr, is_modify_lr=self.is_modify_lr
                )
            )
            return count_fails, count_modify, True
        return count_fails, count_modify, False
                

class Minicolumn:
    """
    Миниколонка. Миниколонка - это набор точек комбинаторного пространства
    
    space_size - количество точек комбинаторного пространства
    max_cluster_per_point - максимальное количество кластеров в точке
    max_count_clusters - максмальное суммарное количество кластеров по всем точкам комбинаторного пространства
    in_threshold_modify, out_threshold_modify - порог активации кластера. Если скалярное произведение базового 
    вектора кластера на новый вектор больше порога, то будет пересчитан веса кластера, выделяющие первую главную
    компоненту
    in_threshold_activate, out_threshold_activate - порог активации точки комбинаторного пространства. Если кол-во
    активных битов больше порога, то будет инициирован процесс модификации существующих кластеров, а также будет
    добавлен новый кластер
    in_size, out_size - количество случайных битов входного/выходного вектора
    base_lr - начальное значение скорости обучения
    is_modify_lr - модификация скорости обучения пропорционально номер шага
    count_in_demensions, count_out_demensions - размер входного и выходного векторов в точке комб. пространства
    """
    
    def __init__(self, space_size=60000, max_cluster_per_point=100,
                 max_count_clusters=1000000, seed=42, 
                 in_threshold_modify=5, out_threshold_modify=0, 
                 in_threshold_activate=5, out_threshold_activate=0,
                 in_size=256, out_size=10,
                 base_lr=0.01, is_modify_lr=True,
                 count_in_demensions=24, count_out_demensions=10):
        self.space = np.array(
            [
                Point(
                    in_threshold_modify, out_threshold_modify,
                    in_threshold_activate, out_threshold_activate,
                    count_in_demensions, count_out_demensions,
                    in_size, out_size,
                    base_lr, is_modify_lr,
                    max_cluster_per_point
                ) for _ in range(space_size)
            ]
        )
        self.count_clusters = 0
        self.max_count_clusters = max_count_clusters
        
        np.random.seed(seed)
        
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
                    minicolumn.count_clusters -= 1
                    
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
    
    """
    Этап обучения
    
    Создание и модификация кластеров на основе пары кодов: входной и выходной
    
    in_code, out_code - входной и выходной коды
    
    Возвращается количество точек, которые оказались неактивными; количество модификаций кластеров;
    количество новых кластеров
    """
    def learn(self, in_code, out_code):
        if self.is_sleep():
            return None, None, None
        
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
