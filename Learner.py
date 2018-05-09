import random

import cv2 as cv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier


class Learner:

    @staticmethod
    def get_random_hu_moments(img, expert_image, box_size=5):
        height = expert_image.shape[0]
        width = expert_image.shape[1]
        point = (random.randrange(box_size, height - box_size), random.randrange(box_size, width - box_size))
        box_range = int(box_size / 2)
        h_st = point[0] - box_range
        h_en = point[0] + box_range + 1
        w_st = point[1] - box_range
        w_en = point[1] + box_range + 1
        cut_image = img[h_st:h_en, w_st:w_en]
        moments = cv.moments(cut_image)
        hu_moments = cv.HuMoments(moments).flatten()
        moment_list = ["m00", "m01", "m02", "m03", "m10", "m11", "m12", "m20", "m21", "m30"]
        for moment in moment_list:
            hu_moments = np.append(hu_moments, [moments[moment]])
        # return expert_image[point[0]][point[1]] > 0, np.append(cut_image.flatten(), hu_moments)
        # return expert_image[point[0]][point[1]] > 0, cut_image.flatten()
        return expert_image[point[0]][point[1]] > 0, hu_moments


    @staticmethod
    def learn(data_set, class_set):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 8), random_state=1,
                            learning_rate='adaptive')
        clf.fit(data_set, class_set)
        return clf

    @staticmethod
    def get_learn_data(self, image, expert_image, amount):
        data_array = []
        class_array = []
        for i in range(amount):
            result = self.get_random_hu_moments(image, expert_image)
            if result[0]:
                class_array.append(1)
            else:
                class_array.append(0)

            for j in range(len(result[1])):
                data_array.append(result[1][j])

        return data_array, class_array

    @staticmethod
    def get_predict_matrix(self, amount, image, expert_image, clf):
        matrix = np.zeros((2, 2))
        for i in range(amount):
            point_moments = self.get_random_hu_moments(image, expert_image)
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
    def k_fold(data_array, class_array, k=10):
        folded = KFold(n_splits=k)
        # folded = LeaveOneOut()

        folded.get_n_splits(data_array)
        matrix = np.zeros((2, 2))
        clf = None
        for train, test in folded.split(data_array):
            X_train, X_test = data_array[train], data_array[test]
            y_train, y_test = class_array[train], class_array[test]
            if clf is None:
                clf = Learner.learn(X_train, y_train)
                clf.fit(X_train, y_train)
            else:
                clf.fit(X_train, y_train)
            clf.score(X_test, y_test)
            # for test_set, test_result in zip(X_test, y_test):
            #
            #     prediction = clf.predict(
            #         np.array(test_set).reshape(1, -1)
            #     )
            #     if prediction == test_result:
            #         if prediction == 1:
            #             matrix[0][0] += 1
            #         else:
            #             matrix[1][1] += 1
            #     else:
            #         if prediction == 1:
            #             matrix[0][1] += 1
            #         else:
            #             matrix[1][0] += 1
        return matrix, clf

    @staticmethod
    def get_accuracy(matrix):
        return (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])

    @staticmethod
    def get_sensitivity(matrix):
        return matrix[0][0] / (matrix[0][0] + matrix[1][0])

    @staticmethod
    def get_specificity(matrix):
        return matrix[1][1] / (matrix[1][1] + matrix[0][1])
