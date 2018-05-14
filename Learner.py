import pickle
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

from ImageProcessor import ImageProcessor


class Learner:
    default_box_size = 49

    @staticmethod
    def get_features(img, expert_image, box_size):
        height = img.shape[0]
        width = img.shape[1]
        box_range = int(box_size / 2)
        point = (random.randrange(box_range, height - box_range), random.randrange(box_range, width - box_range))
        cut_image = ImageProcessor.crop_image(img, point, box_size)

        features = cut_image.flatten()
        # features = []
        # moments = cv.moments(cut_image)
        # hu_moments = cv.HuMoments(moments).flatten()
        # features = np.append(features, hu_moments)
        # moment_list = ["m00", "m01", "m02", "m03", "m10", "m11", "m12", "m20", "m21", "m30"]
        # for moment in moment_list:
        #     features = np.append(features, [moments[moment]])

        # features = np.append(cut_image.flatten(), features)
        # return expert_image[point[0]][point[1]] > 0, np.append(cut_image.flatten(), hu_moments)
        # return expert_image[point[0]][point[1]] > 0, cut_image.flatten()

        return expert_image[point[0]][point[1]] > 0, features, len(features)

    @staticmethod
    def learn(data_set, class_set):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 8), random_state=1,
                            learning_rate='constant', learning_rate_init=0.001)

        clf.fit(data_set, class_set)
        return clf

    @staticmethod
    def learn_forest(data_set, class_set):
        clf = RandomForestClassifier()
        clf.fit(data_set, class_set)
        return clf

    @staticmethod
    def generate_segmentation(clf, image, box_size):
        height = image.shape[0]
        width = image.shape[1]
        pixel_sum = height * width
        last_update = 0
        result_image = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                chunk = ImageProcessor.crop_image(image, (i, j), box_size)
                result_image[i][j] = clf.predict(chunk.flatten().reshape(1, -1))
                if (i * width + j) / pixel_sum > last_update + 0.01:
                    last_update = (i * width + j) / pixel_sum
                    print(last_update)
        return result_image

    @staticmethod
    def get_learn_data(self, image, expert_image, amount, box_size=49):
        data_array = []
        class_array = []
        feature_size = 1
        for i in range(amount):
            result = self.get_features(image, expert_image, box_size)
            if i == 0:
                feature_size = result[2]
            if result[0]:
                class_array.append(1)
            else:
                class_array.append(0)

            for j in range(len(result[1])):
                data_array.append(result[1][j])

        return data_array, class_array, feature_size

    @staticmethod
    def get_predict_matrix(self, amount, image, expert_image, clf, box_size=49):
        matrix = np.zeros((2, 2))
        for i in range(amount):
            point_moments = self.get_features(image, expert_image, box_size)
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

    @staticmethod
    def k_fold(data_array, class_array, k=10, progress_bar=None):
        folded = KFold(n_splits=k)

        folded.get_n_splits(data_array)
        clf = None
        accuracy_array = []
        split = folded.split(data_array)
        iteration = k
        i = 1
        matrix = np.zeros((2, 2))
        for train, test in split:
            X_train, X_test = data_array[train], data_array[test]
            y_train, y_test = class_array[train], class_array[test]
            if clf is None:
                clf = Learner.learn(X_train, y_train)
                clf.fit(X_train, y_train)
            else:
                clf.fit(X_train, y_train)
            accuracy_array.append(clf.score(X_test, y_test))
            if progress_bar is not None:
                progress_bar.setValue(30 + i / iteration * 70)
            i += 1
            for test_set, test_result in zip(X_test, y_test):

                prediction = clf.predict(
                    np.array(test_set).reshape(1, -1)
                )
                if prediction == test_result:
                    if prediction == 1:
                        matrix[0][0] += 1
                    else:
                        matrix[1][1] += 1
                else:
                    if prediction == 1:
                        matrix[0][1] += 1
                    else:
                        matrix[1][0] += 1

        return matrix, sum(accuracy_array) / len(accuracy_array), clf

    @staticmethod
    def get_accuracy(matrix):
        divider = (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
        if divider == 0:
            return 1
        return (matrix[0][0] + matrix[1][1]) / divider

    @staticmethod
    def get_sensitivity(matrix):
        divider = (matrix[0][0] + matrix[0][1])
        if divider == 0:
            return 1
        return matrix[0][0] / divider

    @staticmethod
    def get_specificity(matrix):
        divider = (matrix[1][1] + matrix[1][0])
        if divider == 0:
            return 1
        return matrix[1][1] / divider

    @staticmethod
    def get_precision(matrix):
        divider = (matrix[0][0] + matrix[1][0])
        if divider == 0:
            return 1
        return matrix[0][0] / divider

    @staticmethod
    def get_accuracy_average(matrix):
        divider = (matrix[1][0] + matrix[1][1])
        if divider == 0:
            return 1
        ratio = (matrix[0][0] + matrix[0][1]) / divider
        divider = (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
        if divider == 0:
            positive = 1
        else:
            positive = (matrix[0][0] + matrix[1][1]) / divider
        divider = (matrix[1][1] + matrix[1][0] + matrix[0][1] + matrix[0][0])
        if divider == 0:
            negative = 1
        else:
            negative = (matrix[1][1] + matrix[0][0]) / divider
        return (positive * ratio + negative) / (ratio + 1)

    @staticmethod
    def get_sensitivity_average(matrix):
        divider = (matrix[1][0] + matrix[1][1])
        if divider == 0:
            return 1
        ratio = (matrix[0][0] + matrix[0][1]) / divider
        divider = (matrix[0][0] + matrix[0][1])
        if divider == 0:
            positive = 1
        else:
            positive = matrix[0][0] / divider
        divider = (matrix[1][1] + matrix[1][0])
        if divider == 0:
            negative = 1
        else:
            negative = matrix[1][1] / divider
        return (positive * ratio + negative) / (ratio + 1)

    @staticmethod
    def get_specificity_average(matrix):
        divider = (matrix[1][0] + matrix[1][1])
        if divider == 0:
            return 1
        ratio = (matrix[0][0] + matrix[0][1]) / divider
        divider = (matrix[1][1] + matrix[1][0])
        if divider == 0:
            positive = 1
        else:
            positive = matrix[1][1] / divider
        divider = (matrix[0][0] + matrix[0][1])
        if divider == 0:
            negative = 1
        else:
            negative = matrix[0][0] / divider
        return (positive * ratio + negative) / (ratio + 1)

    @staticmethod
    def get_precision_average(matrix):
        divider = (matrix[1][0] + matrix[1][1])
        if divider == 0:
            return 1
        ratio = (matrix[0][0] + matrix[0][1]) / divider
        divider = (matrix[0][0] + matrix[1][0])
        if divider == 0:
            positive = 1
        else:
            positive = matrix[0][0] / divider
        divider = (matrix[1][1] + matrix[0][1])
        if divider == 0:
            negative = 1
        else:
            negative = matrix[1][1] / divider
        return (positive * ratio + negative) / (ratio + 1)

    @staticmethod
    def save_model(file_path, model):
        pickle.dump(model, open(file_path, 'wb'))

    @staticmethod
    def load_model(file_path):
        return pickle.load(open(file_path, 'rb'))
