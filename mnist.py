import numpy as np
import pandas as pd

from src import image_transformations, context_transform
from src.combinatorial_space.minicolumn import Minicolumn, LearnEnum

df = pd.read_csv('data/MNIST/mnist_train.csv', header=None, nrows=100)

max_number = 5000
count_subimages_for_image = 100
window_size = [4, 4]
minicolumn = Minicolumn(
    space_size=10000,
    max_count_clusters=10000,
    count_in_dimensions=256, in_random_bits=30,
    count_out_dimensions=20, out_random_bits=10,
    seed=42,
    code_aligment_threshold=10,
    in_threshold_activate=6,
    out_threshold_activate=3,
    in_threshold_modify=6,
    out_threshold_modify=3
)
for image_number in range(max_number):
    label, image = image_transformations.get_image(df, 0)
    for subimage_number in range(0, count_subimages_for_image):
        # Получаем коды во всех контекстах из подобласти 4х4
        codes = context_transform.get_shift_context(image, window_size)
        if codes is None:
            continue
        status = minicolumn.learn(codes, threshold_controversy_in=20, threshold_controversy_out=5)
        if status == LearnEnum.LEARN:
            stats = minicolumn.statistics
            print('Всего кластеров:', minicolumn.count_clusters)
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
