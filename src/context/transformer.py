import numpy as np
import cv2


class ContextTransformer:

    def __init__(self, directs=16, width_angle=np.pi / 2, strength=0,
                 window_size=np.array([4, 4]), non_zeros_bits=5):
        self.directs = directs
        self.width_angle = width_angle
        self.strength = strength
        self.window_size = window_size
        self.non_zeros_bits = non_zeros_bits

    # Получаем коды изображения во всех возможных контекстах
    def get_all_codes(self, image):
        context_codes = []
        context_numbers = []
        image_sample = self.__get_sample(image)
        if np.sum(np.sum(np.uint8(image_sample != 0))) <= self.non_zeros_bits:
            return None, None, None
        for context_y in np.arange(-self.window_size[0]+1, self.window_size[0], 1):
            for context_x in np.arange(-self.window_size[1]+1, self.window_size[1], 1):
                context_numbers.append([context_y, context_x])
                context_image = self.get_context_image(
                    image=image_sample,
                    context_number=[context_y, context_x]
                )
                context_code = self.get_codes(context_image)
                context_codes.append(context_code.flatten())
        return [context_codes, context_numbers, image_sample]

    def __get_sample(self, image):
        y = self.window_size[0] - 1 + np.random.randint(0, image.shape[0] - 2 * self.window_size[0] - 1)
        x = self.window_size[1] - 1 + np.random.randint(0, image.shape[1] - 2 * self.window_size[1] - 1)
        return image[
            y - self.window_size[0] + 1:y + 2 * self.window_size[0] - 1,
            x - self.window_size[1] + 1:x + 2 * self.window_size[1] - 1
        ]

    def get_context_image(self, image, context_number):
        x0 = self.window_size[0] - 1
        y0 = self.window_size[1] - 1
        dx = context_number[0]
        dy = context_number[1]
        wdh = self.window_size[0]
        hgt = self.window_size[1]
        context_image = image[y0 + dy:y0 + dy + hgt, x0 + dx:x0 + dx + wdh]
        return np.uint8(context_image)

    def get_codes(self, image):
        sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

        start_angle = 0
        finish_angle = 2*np.pi

        step_angle = (finish_angle - start_angle) / self.directs
        central_angle = np.arange(start_angle, finish_angle, step_angle) + self.width_angle/2

        gamma_down = central_angle - self.width_angle/2
        gamma_up = central_angle + self.width_angle/2

        angle = np.arctan2(sobel_y, sobel_x) + np.pi
        strength = np.sqrt(sobel_y**2 + sobel_x**2)

        return np.array([
            np.uint(
                (
                    (gamma_down[i] <= angle) & (angle <= gamma_up[i]) & (gamma_up[i] <= 2*np.pi) |
                    (0 <= angle) & (angle <= gamma_up[i] - 2*np.pi) & (gamma_up[i] > 2*np.pi)
                )
                &
                (strength > self.strength)
            )
            for i in range(0, self.directs)
        ])
