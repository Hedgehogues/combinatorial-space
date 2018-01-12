import numpy as np
import pandas as pd

import context_transform
import image_transformations
from combinatorial_space.minicolumn import Minicolumn

df = pd.read_csv('data/test_image.csv', header=None)

max_number = 5000
count_subimages_for_image = 7
window_size = [4, 4]
minicolumn = Minicolumn(max_count_clusters=200000)
for image_number in range(max_number):
    label, image = image_transformations.get_image(df, image_number)
    for subimage_number in range(0, 7):
        x, y = np.random.random_integers(0, 27 - window_size[0], 2)
        image_sample = image[y:y + window_size[0], x:x + window_size[1]]

        if np.sum(np.sum(image_sample)) == 0:
            continue

        # Получаем коды во всех контекстах из подобласти 4х4 (нужны правки)
        codes = context_transform.get_shift_context(image_sample)
        count_fails, count_modify, count_adding = minicolumn.learn(codes)
        if count_fails is not None:
            print(
                'Изменения:', int(count_modify),
                '. Пропуски:', int(count_fails),
                '. Новые кластеры:', int(count_adding),
                '. Всего кластеров:', int(minicolumn.count_clusters)
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