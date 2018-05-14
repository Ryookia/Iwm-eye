import os
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

from ImageProcessor import ImageProcessor
from Learner import Learner
from UI import UI


def start_app():
    app = QApplication(sys.argv)
    window = UI()
    window.show()
    sys.exit(app.exec_())


def learn_stuff(image_width=1024, image_height=1024, image_amount=25, prob_count=500000, box_size=9):
    files = os.listdir("./database/images")
    files.sort()
    files_expert = os.listdir("./database/manual")
    files_expert.sort()
    files_mask = os.listdir("./database/mask")
    files_mask.sort()
    data_array = None
    class_array = None
    feature_size = 1
    for i in range(image_amount):
        processor = ImageProcessor()
        processor.load_image("./database/images/" + files[i],
                             "./database/manual/" + files_expert[i])
        processor.scale_image(image_height, image_width)

        image = processor.pre_process_image()
        mask = processor.get_mask("./database/mask/" + files_mask[i])
        image = ImageProcessor.mask_image(image, mask)

        # image = ImageProcessor.to_binary_image(image, 0.5)

        tmp_data, tmp_class, feature_size = Learner.get_learn_data(
            Learner,
            image,
            processor.expert_image,
            prob_count,
            box_size
        )
        if data_array is None:
            data_array = tmp_data
            class_array = tmp_class
        else:
            data_array = np.append(data_array, tmp_data)
            class_array = np.append(class_array, tmp_class)
        print("end " + str(i))

    data_array = np.array(data_array)
    data_array = data_array.reshape(int(data_array.shape[0] / feature_size), feature_size)

    clf = Learner.learn_forest(data_array, class_array)
    Learner.save_model("./classifiers/mad_man_real", clf)


if __name__ == "__main__":
    start_app()
    # learn_stuff()
