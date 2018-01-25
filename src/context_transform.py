import numpy as np
import cv2


# Получаем коды изображения во всех возможных контекстах
def get_shift_context(image, window_size, threshold_non_zeros=5):
    context_codes = []
    image_sample = get_sample(image, window_size)
    if np.sum(np.sum(np.uint8(image_sample) != 0)) > threshold_non_zeros:
        return None
    for context_y in np.arange(-window_size[0]+1, window_size[0], 1):
        for context_x in np.arange(-window_size[1]+1, window_size[1], 1):
            context_number = [context_y, context_x]
            context_image = get_context_image(
                context_number=context_number,
                image=image_sample,
                window_size=window_size
            )
            context_code = get_codes(context_image)
            context_codes.append(context_code.flatten())
    return context_codes


def get_sample(image, window_size):
    y = window_size[0] - 1 + np.random.randint(0, image.shape[0] - 2 * window_size[0] - 1)
    x = window_size[1] - 1 + np.random.randint(0, image.shape[1] - 2 * window_size[1] - 1)
    return image[
        y - window_size[0] + 1:y + 2 * window_size[0] - 1,
        x - window_size[1] + 1:x + 2 * window_size[1] - 1
    ]


def get_context_image(context_number, image, window_size):
    x0 = window_size[0] - 1
    y0 = window_size[1] - 1
    dx = context_number[0]
    dy = context_number[1]
    wdh = window_size[0]
    hgt = window_size[1]
    context_image = image[y0 + dy:y0 + dy + hgt, x0 + dx:x0 + dx + wdh]
    return np.uint8(context_image)


# Получаем код изображения
def get_codes(image, count_directs=16, width_angle=np.pi/2, strength_threshould=0):
    sobel_x = cv2.Sobel(image,cv2.CV_32F,1,0,ksize=1)
    sobel_y = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=1)
    
    start_angle = 0
    finish_angle = 2*np.pi
    
    step_angle = (finish_angle - start_angle) / count_directs
    central_angle = np.arange(start_angle, finish_angle, step_angle) + width_angle/2
    
    gamma_down = central_angle - width_angle/2
    gamma_up = central_angle + width_angle/2
    
    angle = np.arctan2(sobel_y, sobel_x) + np.pi
    strength = np.sqrt(sobel_y**2 + sobel_x**2)
    
    return np.array([
        np.uint(
            (
                (gamma_down[i] <= angle) & (angle <= gamma_up[i]) & (gamma_up[i] <= 2*np.pi) |
                (0 <= angle) & (angle <= gamma_up[i] - 2*np.pi) & (gamma_up[i] > 2*np.pi)
            )
            &
            (strength > strength_threshould)
        )
        for i in range(0, count_directs)
    ])
