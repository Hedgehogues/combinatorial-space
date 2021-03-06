import pandas as pd
import numpy as np
import pickle

from src.image import transformers
from src.combinatorial_space.minicolumn import Minicolumn, MINICOLUMN
from src.context.transformer import ContextTransformer


def sleep__(minicolumn):
    print('Сон')
    clusters_of_points, the_same_clusters = minicolumn.sleep(
        activate=0.75,
        threshold_in_len=3,
        threshold_out_len=2
    )
    sum_ = 0
    for ind, cluster in enumerate(clusters_of_points):
        if len(cluster) == 0:
            sum_ += 1
    print(
        'Неактивные точки:', sum_,
        '. Одинаковые кластеры:', the_same_clusters
    )


df = pd.read_csv('data/MNIST/mnist_train.csv', header=None, nrows=100)

max_number = 5000
count_subimages_for_image = 10
window_size = [4, 4]
space_size = 2000
minicolumn = Minicolumn(
    space_size=space_size,
    max_clusters=30000,
    in_dimensions=64, in_random_bits=25,
    out_dimensions=20, out_random_bits=15,
    seed=42,
    code_alignment=5,
    in_point_activate=5,
    out_point_activate=4,
    in_cluster_modify=6,
    out_cluster_modify=3,
    lr=0.3, binarization=0.1
)
transforms = ContextTransformer(directs=4)

for i in range(space_size, 10000):
    np.sort(np.random.permutation(64)[:25])
    np.sort(np.random.permutation(20)[:15])

means = []
delta = []
ind = 0
opt_context_numbes = []
opt_image_sample = []
opt_out = []
opt_ind_arr = []

for image_number in range(max_number):
    label, image = transformers.get_image(df, image_number)
    print(image_number, label)
    start = minicolumn.count_clusters
    for subimage_number in range(0, count_subimages_for_image):
        codes, context_numbes, image_sample = transforms.get_sample_codes(image)
        if codes is None:
            continue
        opt_ind, out_code, status = minicolumn.learn(
            codes,
            ind,
            in_controversy=20,
            out_controversy=5
        )
        if opt_ind is None:
            continue
        opt_context_numbes.append(context_numbes[opt_ind])
        opt_image_sample.append(image_sample)
        opt_out.append(out_code)
        opt_ind_arr.append(opt_ind)

        if status == MINICOLUMN.LEARN:
            print('№', ind, 'Всего кластеров:', minicolumn.count_clusters)
            means.append(np.mean([len(p.clusters) for p in minicolumn.space]))
        elif status == MINICOLUMN.SLEEP:
            sleep__(minicolumn)
        ind += 1
    print('Изменение кол-ва кластеров', minicolumn.count_clusters - start)
    delta.append(minicolumn.count_clusters - start)
    if minicolumn.is_sleep() or minicolumn.count_clusters - start < 200:
        sleep__(minicolumn)
        pickle.dump(minicolumn, open('data/minicolumn.pkl', 'wb'))
        break

print(opt_ind_arr[0], opt_ind_arr[1])
print(opt_out[0], opt_out[1])
