import numpy as np
import cv2


def get_image(df, index, size=(28, 28)):
    return df.loc[index, 0], df.loc[index, 1:].values.reshape(size).astype(np.uint8)


def resize_row_data_frame(row, input_size=(32, 32), output_size=(28, 28)):
    return cv2.resize(row.reshape(input_size).astype(np.uint8), output_size).ravel()


def rotate_image(image, alpha=0, scale=1):
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), alpha, scale)
    return cv2.warpAffine(image, M, (cols, rows))


def rotate_row_data_frame(row, alpha=0, scale=1, size=(28, 28)):
    label, image = row[0], row[1:].values.reshape(size).astype(np.uint8)
    return np.hstack([[label], rotate_image(image, alpha, scale).ravel()])


def rotate_image_data_frame(df, index, alpha=0, scale=1, size=(28, 28)):
    label, image = get_image(df, index, size)
    return label, rotate_image(image, alpha, scale)
