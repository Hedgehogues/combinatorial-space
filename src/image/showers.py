import matplotlib.pyplot as plt
from src.image import transformers


def show_image(image, label=None, ax=plt):
    ax.imshow(image, cmap='gray');
    ax.title('Цифра:'+str('' if label is None else label))
    ax.axis('off');


def show_image_data_frame(df, index, ax=plt):
    ax.imshow(transformers.get_image(df, index)[1], cmap='gray');
    ax.title('Цифра:'+str(df.loc[index, 0]))
    ax.axis('off');
