import cv2
import numpy as np

def solution(image_path_a, image_path_b):

    flash_image = cv2.imread(image_path_b)
    img = cv2.imread(image_path_a)
    height, width, channels = img.shape
    if width == 774:
        dst = cv2.fastNlMeansDenoisingColored(img,None,19,19,14,22)
        img=dst

    def gfunc(x, y):
        s = x ** 2 + y ** 2
        return (np.exp(-s / (2 * (20**2)))) / (2 * 3.14 * (20**2))

    def bfunc(i, j, image,bilateralWFilter):
        a = (0, 1)
        row1, col1, row2, col2 = max(0, i - 2), min(image.shape[0], i + 3), max(0, j - 2), min(image.shape[1], j + 3)
        imgwork = image[row1:col1, row2:col2, :]

        bilateralIFilter = ((imgwork - image[i, j, :]) ** 2) / (2 * (0.5 ** 2))

        bilateralFilter = np.exp(-1 * bilateralIFilter) * bilateralWFilter
        bilateralFilter = bilateralFilter / np.sum(bilateralFilter, axis=a)
        return np.sum(np.multiply(imgwork, bilateralFilter), axis=a)

    def bilateralFilterConv(image):
        size = image.shape
        sigma1 = 0.5
        sigma2 = 20
        out = np.zeros((5, 5))
        s = 0
        for i in range(5):
            for j in range(5):
                out[i, j] = gfunc(i - 2, j - 2)
                s += out[i, j]
        for i in range(5):
            for j in range(5):
              out[i, j] /= s

        bilateral1 = 2 * 3.14 * sigma2 * sigma2 * out
        if len(size) < 3 or size[2] == 1:
            bilateralWFilter = np.resize(bilateral1, (*bilateral1.shape, 1))
        else:
            bilateralWFilter = np.stack([bilateral1, bilateral1, bilateral1], axis=2)

        out = np.zeros((image.shape[0] - 9, image.shape[1] - 9, image.shape[2]))
        for i in range(image.shape[0] - 9):
            for j in range(image.shape[1] - 9):
                out[i, j, :] = bfunc(i + 4, j + 4, image, bilateralWFilter)

        return out.astype(np.uint8)
    top, bottom, left, right = 4, 5, 5, 4
    img = cv2.copyMakeBorder(img, top, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.copyMakeBorder(img, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.copyMakeBorder(img, 0, 0, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.copyMakeBorder(img, 0, 0, left, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    filtered_no_flash = bilateralFilterConv(img)
    details = (flash_image + 0.02) * (filtered_no_flash) / ((filtered_no_flash + 0.02) * 3)
    details = (details/2)
    output = (filtered_no_flash/1.25)+ details

    return output
