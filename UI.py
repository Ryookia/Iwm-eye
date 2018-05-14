import os

import PIL.ImageQt
import numpy as np
import scipy.misc as scmisc
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi

from ImageProcessor import ImageProcessor
from Learner import Learner

false = False
true = True


class UI(QMainWindow):
    processor = None
    files = None
    files_expert = None
    files_mask = None
    models = None
    listener = None
    images_array = None
    average_accuracy = 0
    matrix = np.zeros((2, 2))
    current_model = None

    current_image = None
    current_expert_image = None

    def __init__(self):
        super(UI, self).__init__()
        self.processor = ImageProcessor()
        loadUi("window_gui.ui", self)
        self.files = os.listdir("./database/images")
        self.files.sort()
        self.files_expert = os.listdir("./database/manual")
        self.files_expert.sort()
        self.files_mask = os.listdir("./database/mask")
        self.files_mask.sort()
        self.fileBox.addItems(self.files)

        self.models = os.listdir("./classifiers")
        self.modelList.addItems(self.models)

        self.generateButton.clicked.connect(self.generate_image)
        self.myImageButton.clicked.connect(self.show_my_image)
        self.expertImageButton.clicked.connect(self.show_expert_image)
        self.kfoldButton.clicked.connect(self.perform_kfold)
        self.balancedBox.stateChanged.connect(self.matrix_change)
        self.loadModelButton.clicked.connect(self.load_model)
        self.predictButton.clicked.connect(self.predict_image)
        self.predictButton.setVisible(False)
        self.generateBar.setVisible(False)

    def load_model(self):
        index = self.modelList.currentIndex()
        if index < 0:
            return
        file_path = "./classifiers/" + self.models[index]
        print(file_path)
        self.current_model = Learner.load_model(file_path)
        if self.current_model is not None:
            self.predictButton.setVisible(True)

    def predict_image(self):
        if self.current_model is None:
            self.predictButton.setVisible(False)
            return
        index = self.fileBox.currentIndex()
        self.processor.load_image("./database/images/" + self.files[index],
                                  "./database/manual/" + self.files_expert[index])
        self.processor.scale_image(self.heightValue.value(), self.widthValue.value())

        image = self.processor.pre_process_image()
        mask = self.processor.get_mask("./database/mask/" + self.files_mask[index])
        image = ImageProcessor.mask_image(image, mask)
        image = ImageProcessor.to_binary_image(image, 0.5)

        self.current_image = image
        self.current_expert_image = self.processor.expert_image
        image = Learner.generate_segmentation(self.current_model, self.current_image, 9)
        ImageProcessor.show_given_image(image, "After segmentation")
        ImageProcessor.show_given_image(self.current_expert_image, "Expert Image")

    def matrix_change(self):
        self.update_matrices()

    def show_my_image(self):
        if self.current_image is not None:
            self.processor.show_given_image(self.current_image, "Wytworzony obraz")

    def show_expert_image(self):
        if self.current_expert_image is not None:
            self.processor.show_given_image(self.current_expert_image, "Obraz eksperta")

    def perform_kfold(self):
        self.learnBar.setVisible(True)
        self.thread = LearnThread(self, self.imageCount.value())
        self.thread.start()
        self.thread.finished.connect(self.end_learn)
        # acc = Learner.get_accuracy(matrix_fold)
        # sen = Learner.get_sensitivity(matrix_fold)
        # spec = Learner.get_specificity(matrix_fold)

    # #

    def end_learn(self):
        self.learnBar.setVisible(False)
        self.accuracyLabel.setText(str(self.average_accuracy))

        self.update_matrices()

    def generate_image(self):
        self.generateBar.setVisible(True)

        index = self.fileBox.currentIndex()
        self.thread = ProcessThread(self.processor,
                                    self,
                                    self.files[index],
                                    self.files_expert[index],
                                    self.files_mask[index])
        self.thread.start()
        self.thread.finished.connect(self.end_generate)

    def end_generate(self):
        q_pixmap = QPixmap.fromImage(PIL.ImageQt.ImageQt(scmisc.toimage(self.current_image)))
        container_width = self.myImage.width()
        container_height = self.myImage.height()
        self.myImage.setPixmap(q_pixmap.scaled(
            container_width,
            container_height,
            QtCore.Qt.KeepAspectRatio))
        self.generateBar.setValue(90)
        q_pixmap = QPixmap.fromImage(PIL.ImageQt.ImageQt(scmisc.toimage(self.current_expert_image)))
        container_width = self.expertImage.width()
        container_height = self.expertImage.height()
        self.expertImage.setPixmap(q_pixmap.scaled(
            container_width,
            container_height,
            QtCore.Qt.KeepAspectRatio))

        self.matrix = np.zeros((2, 2))
        self.matrix += ImageProcessor.compare_images(self.current_image, self.current_expert_image)
        self.generateBar.setVisible(false)
        self.update_matrices()

    def update_matrices(self):

        use_balanced = self.balancedBox.isChecked()

        if use_balanced:
            acc = Learner.get_accuracy_average(self.matrix)
            sen = Learner.get_sensitivity_average(self.matrix)
            spec = Learner.get_specificity_average(self.matrix)
            prec = Learner.get_precision_average(self.matrix)
        else:
            acc = Learner.get_accuracy(self.matrix)
            sen = Learner.get_sensitivity(self.matrix)
            spec = Learner.get_specificity(self.matrix)
            prec = Learner.get_precision(self.matrix)

        self.labelTP.setText(str(self.matrix[0][0]))
        self.labelFN.setText(str(self.matrix[0][1]))
        self.labelFP.setText(str(self.matrix[1][0]))
        self.labelTN.setText(str(self.matrix[1][1]))

        self.labelSpec.setText(str(spec))
        self.labelPrec.setText(str(prec))
        self.labelSen.setText(str(sen))
        self.labelAcc.setText(str(acc))


class ProcessThread(QThread):

    def __init__(self, processor, context, image_file, expert_file, mask_file):
        QThread.__init__(self)
        self.signal = pyqtSignal()
        self.processor = processor
        self.context = context
        self.image_file = image_file
        self.expert_file = expert_file
        self.mask_file = mask_file

    def __del__(self):
        self.wait()

    def run(self):
        self.generate_image()

    def generate_image(self):
        self.update_progress(0)

        self.processor.load_image("./database/images/" + self.image_file,
                                  "./database/manual/" + self.expert_file)
        self.update_progress(20)
        self.processor.scale_image(self.context.heightValue.value(), self.context.widthValue.value())

        image = self.processor.pre_process_image()
        self.update_progress(40)
        mask = self.processor.get_mask("./database/mask/" + self.mask_file)
        image = ImageProcessor.mask_image(image, mask)
        self.update_progress(60)
        image = ImageProcessor.to_binary_image(image, 0.5)
        self.update_progress(80)

        self.context.current_image = image
        self.context.current_expert_image = self.processor.expert_image

    def update_progress(self, percentage):
        self.context.generateBar.setValue(percentage)


class LearnThread(QThread):

    def __init__(self, context, image_amount):
        QThread.__init__(self)
        self.signal = pyqtSignal()
        self.context = context
        self.image_amount = image_amount

    def __del__(self):
        self.wait()

    def run(self):
        self.perform_kfold()

    def perform_kfold(self):
        data_array = None
        class_array = None
        feature_size = 1
        self.context.learnBar.setValue(0)
        for i in range(self.image_amount):
            processor = ImageProcessor()
            processor.load_image("./database/images/" + self.context.files[i],
                                 "./database/manual/" + self.context.files_expert[i])
            processor.scale_image(self.context.heightValue.value(), self.context.widthValue.value())

            image = processor.pre_process_image()
            mask = processor.get_mask("./database/mask/" + self.context.files_mask[i])
            image = ImageProcessor.mask_image(image, mask)

            image = ImageProcessor.to_binary_image(image, 0.5)

            tmp_data, tmp_class, feature_size = Learner.get_learn_data(
                Learner,
                image,
                processor.expert_image,
                self.context.probCount.value(),
                self.context.boxSize.value()
            )
            if data_array is None:
                data_array = tmp_data
                class_array = tmp_class
            else:
                data_array = np.append(data_array, tmp_data)
                class_array = np.append(class_array, tmp_class)

            self.context.learnBar.setValue(30 * ((i + 1) / self.image_amount))

        data_array = np.array(data_array)
        data_array = data_array.reshape(int(data_array.shape[0] / feature_size), feature_size)

        matrix, average_accuracy, clf = Learner.k_fold(
            data_array,
            class_array,
            self.context.kCount.value(),
            progress_bar=self.context.learnBar
        )

        self.context.average_accuracy = average_accuracy
        self.context.matrix = matrix;

    def update_progress(self, percentage):
        self.context.generateBar.setValue(percentage)
