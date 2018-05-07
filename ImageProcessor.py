import cv2 as cv
import numpy as np
from skimage import filters
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.morphology import disk


class ImageProcessor:
    test = False
    image = None
    expert_image = None
    gaussian_sigma = 2
    median_value = 5
    dilation_value = 3
    erosion_value = 2

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

    @staticmethod
    def normalize(img):
        min_val = img.min()
        max_val = img.max()
        return (img - min_val) / (max_val - min_val)

    @staticmethod
    def to_binary_image(array, cut_value):
        rows = len(array[0])
        cols = len(array[1])
        for i in range(rows):
            for j in range(cols):
                if array[i][j] < cut_value:
                    array[i][j] = 0
                else:
                    array[i][j] = 1
        return array

    @staticmethod
    def invert_image(img):
        return 255 - img

    @staticmethod
    def get_accuracy(matrix):
        return (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])

    @staticmethod
    def get_sensitivity(matrix):
        return matrix[0][0] / (matrix[0][0] + matrix[1][0])

    @staticmethod
    def get_specificity(matrix):
        return matrix[1][1] / (matrix[1][1] + matrix[0][1])

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

    # idea was to find the eye radius on image to black out unnecessary parts (did not work as intended)
    def find_radius(self, img):
        height = img.shape[0]
        width = img.shape[1]
        top = 0
        left = 0
        right = width
        bottom = height
        for i in range(height):
            if top != 0:
                break
            for j in range(width):
                if img[i][j] > 0.1:
                    top = i

        for j in range(width):
            if left != 0:
                break
            for i in range(height):
                if img[i][j] > 0.1:
                    left = j

        for j in range(width - 1, -1, -1):
            if right != 0:
                break
            for i in range(height):
                if img[i][j] > 0.1:
                    right = j

        for i in range(height - 1, 0, -1):
            if bottom != 0:
                break
            for j in range(width):
                if img[i][j] > 0.1:
                    bottom = i
        max_result = height / 2 - top
        if width / 2 - left > max_result:
            max_result = width / 2 - left
        if right - width / 2 > max_result:
            max_result = right - width / 2
        if bottom - height / 2 > max_result:
            max_result = bottom - height / 2
        return max_result

    def pre_process_image(self):

        ## self.image = cv.split(self.image)
        ## self.image = self.image[1]

        self.image = filters.median(self.image, disk(self.median_value))
        self.image = filters.gaussian(self.image, sigma=self.gaussian_sigma)

        ## self.image = morphology.erosion(self.image, disk(4))
        ## self.image = morphology.dilation(self.image, disk(2))

        min = self.image.min()
        max = self.image.max()

        self.image = (self.image - min) / (max - min) * 255
        self.image = self.image.astype(np.uint8)

        self.image = self.image.astype(np.uint8)

        hxx, hyy, hxy = hessian_matrix(self.image, 4.0, order="xy")
        i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)

        i1 = self.normalize(i1)
        i2 = self.normalize(i2)

        i3 = i1 + self.invert_image(i2)
        i3 = self.normalize(i3)
        i3 = self.to_binary_image(i3, 0.3)
        self.expert_image = self.normalize(self.expert_image)
        self.expert_image = self.to_binary_image(self.expert_image, 0.5)
        result_matrix = self.compare_images(i3, self.expert_image)
        accuracy = self.get_accuracy(result_matrix)
        sensitivity = self.get_sensitivity(result_matrix)
        specificity = self.get_specificity(result_matrix)

        # i1 = self.to_binary_image(i1, 0.5)
        # i2 = self.to_binary_image(i2, 0.5)
        # i3 = self.to_binary_image(i3, 0.3)

        cv.imshow("i1", i1)
        cv.waitKey(0)
        cv.imshow("i2", i2)
        cv.waitKey(0)

        # i3 = morphology.dilation(i3, disk(self.dilation_value))
        # i3 = morphology.erosion(i3, disk(self.erosion_value))

        cv.imshow("i3", i3)
        cv.waitKey(0)
        cv.imshow("expert", self.expert_image)
        cv.waitKey(0)
        # self.image = cv.Canny(self.image, 0, 20)
        # self.image = morphology.dilation(self.image, disk(self.dilation_value))
        # self.image = morphology.erosion(self.image, disk(self.erosion_value))

    def show_image(self):
        cv.imshow("Image", self.image)
        cv.waitKey(0)
