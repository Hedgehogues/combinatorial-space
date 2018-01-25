import numpy as np
import pandas as pd

from src import image_transformations, context_transform
from src.combinatorial_space.minicolumn import Minicolumn, LearnEnum, StatisicsMinicolumn

df = pd.read_csv('data/test_image.csv', header=None)

max_number = 5000
count_subimages_for_image = 100
window_size = [4, 4]
minicolumn = Minicolumn(max_count_clusters=6000, seed=10)
for image_number in range(max_number):
    label, image = image_transformations.get_image(df, 0)
    for subimage_number in range(0, count_subimages_for_image):
        x, y = np.random.random_integers(0, 27 - window_size[0], 2)
        image_sample = image[y:y + window_size[0], x:x + window_size[1]]

        if np.sum(np.sum(image_sample)) == 0:
            continue

        # Получаем коды во всех контекстах из подобласти 4х4
        # TODO:  (нужны правки)
        codes = context_transform.get_shift_context(image_sample)
        status = minicolumn.learn(codes)
        if status == LearnEnum.LEARN:
            stats = minicolumn.statistics
            print(
                'Изменения:', stats[StatisicsMinicolumn.MINICOLUMN_COUNT_MODIFY],
                '. Пропуски:', stats[StatisicsMinicolumn.MINICOLUMN_COUNT_NOT_MODIFY],
                '. Новые кластеры:', stats[StatisicsMinicolumn.MINICOLUMN_NEW_CLUSTERS],
                '. Всего кластеров:', stats[StatisicsMinicolumn.COUNT_CLUSTERS],
                '. Всего кластеров:', stats[StatisicsMinicolumn.COUNT_CLUSTERS]
            )
        else:
            print('Сон')
            clusters_of_points, the_same_clusters = minicolumn.sleep()
            sum_ = 0
            for ind, cluster in enumerate(clusters_of_points):
                if len(cluster) == 0:
                    sum_ += 1
            print(
                'Неактивные точки:', sum_,
                '. Одинаковые кластеры:', the_same_clusters
            )
