import numpy as np

from ImageProcessor import ImageProcessor
from Learner import Learner

images_path = [("database/images/01_dr.JPG", "database/manual/01_dr.tif"),
               ("database/images/02_dr.JPG", "database/manual/02_dr.tif"),
               ("database/images/03_dr.JPG", "database/manual/03_dr.tif")]

data_array = None
class_array = None
for image_path in images_path:
    processor = ImageProcessor()
    processor.load_image(image_path[0], image_path[1])
    processor.scale_image(1000, 1000)

    processor.erase_channel([0, 2])
    processor.to_grey_scale()
    processor.normalize_histogram()
    image = processor.pre_process_image()

    # ImageProcessor.show_given_image(image)

    tmp_data, tmp_class = Learner.get_learn_data(Learner, image, processor.expert_image, 10000)
    if data_array is None:
        data_array = tmp_data
        class_array = tmp_class
    else:
        data_array = np.append(data_array, tmp_data)
        class_array = np.append(class_array, tmp_class)

data_array = np.array(data_array)
data_array = data_array.reshape(int(data_array.shape[0] / 17), 17)

matrix = Learner.k_fold(data_array, class_array, 2)
# clf = Learner.learn(data_array, class_array)
#
# matrix = np.zeros((2, 2))
# matrix_retard = np.zeros((2, 2))
# for image_path in images_path:
#     processor = ImageProcessor()
#     processor.load_image(image_path[0], image_path[1])
#     processor.scale_image(1000, 1000)
#     processor.erase_channel([0, 2])
#     processor.to_grey_scale()
#     processor.normalize_histogram()
#
#     image = processor.pre_process_image()
#     matrix_retard += ImageProcessor.compare_images(image, processor.expert_image)
#     matrix += Learner.get_predict_matrix(Learner, 10000, image, processor.expert_image, clf)
# #
acc = Learner.get_accuracy(matrix)
sen = Learner.get_sensitivity(matrix)
spec = Learner.get_specificity(matrix)

print("Neural")
print("Acc: " + str(acc) + ",\nSen: " + str(sen) + ",\nSpec: " + str(spec))
#
# acc = Learner.get_accuracy(matrix_retard)
# sen = Learner.get_sensitivity(matrix_retard)
# spec = Learner.get_specificity(matrix_retard)
#
# print("Tard")
# print("Acc: " + str(acc) + ",\nSen: " + str(sen) + ",\nSpec: " + str(spec))
#
# print("end")
