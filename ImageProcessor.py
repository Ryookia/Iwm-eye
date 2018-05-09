import math

import cv2 as cv
import numpy as np
from skimage.filters import frangi


class ImageProcessor:
    test = False
    image = None
    expert_image = None
    gaussian_sigma = 2
    blur = 0.05
    median_value = 5
    dilation_value = 0.01
    erosion_value = 0.007

    def __init__(self):
        self.test = False

    def load_image(self, file_path, expert_path):
        self.image = cv.imread(file_path)
        self.expert_image = cv.imread(expert_path)

    def scale_image(self, height, width):
        self.image = cv.resize(self.image, (width, height))
        self.expert_image = cv.resize(self.expert_image, (width, height))

    def show_rgb_spectrum(self):
        bgr = cv.split(self.image)
        cv.imshow("Blue", bgr[0])
        cv.waitKey(0)

        cv.imshow("Green", bgr[1])
        cv.waitKey(0)

        cv.imshow("Red", bgr[2])
        cv.waitKey(0)

    def erase_channel(self, channel_array):
        shape = self.image.shape
        for i in channel_array:
            for j in range(0, shape[0]):
                for k in range(0, shape[1]):
                    self.image[j][k][i] = 0

    def to_grey_scale(self):
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self.expert_image = cv.cvtColor(self.expert_image, cv.COLOR_BGR2GRAY)

    def normalize_histogram(self):
        hist, bins = np.histogram(self.image, 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.image = cdf[self.image]

    @staticmethod
    def normalize(img):
        min_val = img.min()
        max_val = img.max()
        return (img - min_val) / (max_val - min_val)

    @staticmethod
    def to_binary_image(array, cut_value, max=1):
        rows = len(array[0])
        cols = len(array[1])
        for i in range(rows):
            for j in range(cols):
                if array[i][j] < cut_value:
                    array[i][j] = 0
                else:
                    array[i][j] = max
        return array

    @staticmethod
    def invert_image(img):
        return 255 - img

    # compares two binary images, where first one is user made and second one is created by expert
    @staticmethod
    def compare_images(img1, img2):
        if img1.shape[0] != img2.shape[0] or img1.shape[1] != img2.shape[1]:
            raise Exception("Compared image size is different")
        matrix = np.zeros(shape=(2, 2))
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                if img1[i][j] == 0:
                    if img2[i][j] == 0:
                        matrix[1][1] += 1
                    else:
                        matrix[1][0] += 1
                else:
                    if img2[i][j] == 0:
                        matrix[0][1] += 1
                    else:
                        matrix[0][0] += 1
        return matrix

    def sharp_image(self):
        avr = self.get_average_image_size()
        blurred = cv.GaussianBlur(self.image, (0, 0), avr * self.blur)
        cv.addWeighted(self.image, 1.5, blurred, -0.5, 0, self.image)

    def get_average_image_size(self):
        return (self.image.shape[0] + self.image.shape[1]) / 2

    def get_mask(self, file_path):
        mask = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        mask = cv.resize(mask, (self.image.shape[1], self.image.shape[0]))
        return mask

    def get_gabor_features(self):
        kernel = cv.getGaborKernel((11, 11), 2, math.pi / 4, 4.0, 0.6, 2)
        result = cv.filter2D(self.image, 32, kernel)
        self.show_given_image(result)

    @staticmethod
    def mask_image(image, mask):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i][j] == 0:
                    image[i][j] = 0
        return image

    def pre_process_image(self):
        self.sharp_image()
        self.erase_channel([0, 2])
        self.to_grey_scale()
        self.normalize_histogram()
        ## self.image = morphology.erosion(self.image, disk(4))
        ## self.image = morphology.dilation(self.image, disk(2))

        self.image = self.normalize(self.image) * 255
        self.image = self.image.astype(np.uint8)

        post_image = frangi(self.image, scale_range=(2, 5), beta1=0.1, beta2=1) * 255
        # post_image = self.normalize(post_image) * 255

        return post_image

    def show_image(self):
        cv.imshow("Image", self.image)
        cv.waitKey(0)

    @staticmethod
    def show_given_image(image):
        cv.imshow("Image", image)
        cv.waitKey(0)
