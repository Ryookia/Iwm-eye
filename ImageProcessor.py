import random

import cv2 as cv
import numpy as np
from skimage import morphology
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import frangi
from skimage.morphology import disk
from sklearn.neural_network import MLPClassifier


class ImageProcessor:
    test = False
    image = None
    expert_image = None
    gaussian_sigma = 2
    median_value = 5
    dilation_value = 1
    erosion_value = 4

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
        cdf_normalized = cdf * hist.max() / cdf.max()
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

    def get_random_hu_moments(self, img, box_size=5):
        height = self.expert_image.shape[0]
        width = self.expert_image.shape[1]
        point = (random.randrange(box_size, height - box_size), random.randrange(box_size, width - box_size))
        box_range = int(box_size / 2)
        h_st = point[0] - box_range
        h_en = point[0] + box_range + 1
        w_st = point[1] - box_range
        w_en = point[1] + box_range + 1
        cut_image = img[h_st:h_en, w_st:w_en]
        humoments = cv.HuMoments(cv.moments(cut_image)).flatten()
        return self.expert_image[point[0]][point[1]] > 0, humoments

    def learn(self, data_set, class_set):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 2), random_state=1)
        clf.fit(data_set, class_set)
        return clf

    def get_learn_data(self, image, amount):
        data_array = []
        class_array = []
        for i in range(amount):
            result = self.get_random_hu_moments(image)
            if result[0]:
                class_array.append(1)
            else:
                class_array.append(0)
            if i == 0:
                data_array.append(66.666)
                data_array.append(66.666)
                data_array.append(66.666)
                data_array.append(66.666)
                data_array.append(66.666)
                data_array.append(66.666)
                data_array.append(66.666)
            else:
                for j in range(len(result[1])):
                    data_array.append(result[1][j])
        data_array = np.array(data_array)
        data_array = data_array.reshape(int(data_array.shape[0] / 7), 7)
        return data_array, class_array

    def get_predict_matrix(self, amount, image, clf):
        matrix = np.zeros((2, 2))
        for i in range(amount):
            point_moments = self.get_random_hu_moments(image)
            prediction = clf.predict(
                np.array(point_moments[1]).reshape(1, -1)
            )
            if prediction == point_moments[0]:
                if prediction == 1:
                    matrix[0][0] += 1
                else:
                    matrix[1][1] += 1
            else:
                if prediction == 1:
                    matrix[0][1] += 1
                else:
                    matrix[1][0] += 1
        return matrix

    def pre_process_image(self):
        ## self.image = cv.split(self.image)
        ## self.image = self.image[1]

        # self.image = filters.median(self.image, disk(self.median_value))
        # self.image = filters.gaussian(self.image, sigma=self.gaussian_sigma)

        ## self.image = morphology.erosion(self.image, disk(4))
        ## self.image = morphology.dilation(self.image, disk(2))

        self.image = self.normalize(self.image) * 255
        self.image = self.image.astype(np.uint8)
        self.image = self.invert_image(self.image)

        hxx, hyy, hxy = hessian_matrix(self.image, 3.0, order="xy")
        i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)

        i1 = self.normalize(i1)
        i2 = self.normalize(i2)

        i3 = i1 - i2
        i3 = self.normalize(i3)

        # self.expert_image = self.normalize(self.expert_image)
        # self.expert_image = self.to_binary_image(self.expert_image, 0.5)
        # result_matrix = self.compare_images(i3, self.expert_image)
        # accuracy = self.get_accuracy(result_matrix)
        # sensitivity = self.get_sensitivity(result_matrix)
        # specificity = self.get_specificity(result_matrix)

        # # i1 = self.to_binary_image(i1, 0.5)
        # # i2 = self.to_binary_image(i2, 0.5)
        # # i3 = self.to_binary_image(i3, 0.3)
        #
        # cv.imshow("i1", i1)
        # cv.waitKey(0)
        # cv.imshow("i1", i2)
        # cv.waitKey(0)
        #
        # # i3 = morphology.dilation(i3, disk(self.dilation_value))
        # # i3 = morphology.erosion(i3, disk(self.erosion_value))
        #
        # cv.imshow("i1", self.image)
        # cv.waitKey(0)
        # self.image = self.image / 255

        # cv.imshow("i1", self.image)
        # cv.waitKey(0)
        thresh = frangi(self.invert_image(self.image), beta2=5)
        thresh = self.normalize(thresh) * 255

        # #thresh = morphology.dilation(thresh, disk(self.dila# tion_value))
        thresh = morphology.erosion(thresh, disk(self.erosion_value))
        # cv.imshow("i1", thresh)
        # cv.waitKey(0)
        # cv.imshow("i1", self.expert_image)
        # cv.waitKey(0)
        # # self.image = cv.Canny(self.image, 0, 20)

        data_array, class_array = self.get_learn_data(thresh, 100000)
        clf = self.learn(data_array, class_array)
        matrix = self.get_predict_matrix(10000, thresh, clf)

        acc = self.get_accuracy(matrix)
        sen = self.get_sensitivity(matrix)
        spec = self.get_specificity(matrix)

        print("end")

    def show_image(self):
        cv.imshow("Image", self.image)
        cv.waitKey(0)
