import os

import PIL.ImageQt
import scipy.misc as scmisc
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi

from ImageProcessor import ImageProcessor

false = False
true = True


class UI(QMainWindow):
    processor = None
    files = None
    files_expert = None
    files_mask = None

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
        self.generateButton.clicked.connect(self.generate_image)
        self.generateBar.setVisible(False)

    def generate_image(self):
        index = self.fileBox.currentIndex()
        self.thread = ProcessThread(self.processor,
                                    self,
                                    self.files[index],
                                    self.files_expert[index],
                                    self.files_mask[index])
        self.thread.start()


class ProcessThread(QThread):

    def __init__(self, processor, context, image_file, expert_file, mask_file):
        QThread.__init__(self)
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
        self.context.generateBar.setVisible(True)
        self.update_progress(0)

        self.processor.load_image("./database/images/" + self.image_file,
                                  "./database/manual/" + self.expert_file)
        self.update_progress(10)
        self.processor.scale_image(1000, 1000)

        image = self.processor.pre_process_image()
        self.update_progress(40)
        mask = self.processor.get_mask("./database/mask/" + self.mask_file)
        image = ImageProcessor.mask_image(image, mask)
        self.update_progress(60)
        image = ImageProcessor.to_binary_image(image, 0.5)
        self.update_progress(80)
        q_pixmap = QPixmap.fromImage(PIL.ImageQt.ImageQt(scmisc.toimage(image)))
        container_width = self.context.myImage.width()
        container_height = self.context.myImage.height()
        self.context.myImage.setPixmap(q_pixmap.scaled(
            container_width,
            container_height,
            QtCore.Qt.KeepAspectRatio))
        self.update_progress(90)
        q_pixmap = QPixmap.fromImage(PIL.ImageQt.ImageQt(scmisc.toimage(self.processor.expert_image)))
        container_width = self.context.expertImage.width()
        container_height = self.context.expertImage.height()
        self.context.expertImage.setPixmap(q_pixmap.scaled(
            container_width,
            container_height,
            QtCore.Qt.KeepAspectRatio))
        self.context.generateBar.setVisible(false)

    def update_progress(self, percentage):
        self.context.generateBar.setValue(percentage)
