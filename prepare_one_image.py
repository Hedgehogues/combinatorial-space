import pandas as pd
import numpy as np

np.warnings.filterwarnings('ignore')

path_mnist = 'data/MNIST/'
conf_test_mnist = {
    'imgf': path_mnist+'t10k-images.idx3-ubyte',
    'labelf': path_mnist+'t10k-labels.idx1-ubyte',
    'n': 10000,
    'outf': path_mnist+'mnist_test.csv',
    'size': (28, 28)
}
conf_train_mnist = {
    'imgf': path_mnist+'train-images.idx3-ubyte',
    'labelf': path_mnist+'train-labels.idx1-ubyte',
    'n': 60000,
    'outf': path_mnist+'mnist_train.csv',
    'size': (28, 28)
}


df_test = pd.read_csv(conf_train_mnist['outf'], header=None)

pd.DataFrame(df_test.iloc[4]).T.to_csv('data/test_image.csv', sep=',', header=None, index=None)
