import numpy as np

from ImageProcessor import ImageProcessor
from Learner import Learner

images_path = [("database/images/01_dr.JPG", "database/manual/01_dr.tif", "database/mask/01_dr_mask.tif"),
               ("database/images/02_dr.JPG", "database/manual/02_dr.tif", "database/mask/02_dr_mask.tif"),
               ("database/images/03_dr.JPG", "database/manual/03_dr.tif", "database/mask/03_dr_mask.tif"),
               ("database/images/04_dr.JPG", "database/manual/04_dr.tif", "database/mask/04_dr_mask.tif"),
               ("database/images/05_dr.JPG", "database/manual/05_dr.tif", "database/mask/05_dr_mask.tif")]

images_array = []

data_array = None
class_array = None
feature_size = 1
for image_path in images_path:
    print("process")
    processor = ImageProcessor()
    processor.load_image(image_path[0], image_path[1])
    processor.scale_image(1000, 1000)

    image = processor.pre_process_image()
    mask = processor.get_mask(image_path[2])
    image = ImageProcessor.mask_image(image, mask)

    image = ImageProcessor.to_binary_image(image, 0.5)
    print("copy")
    images_array.append((image, processor.to_binary_image(processor.expert_image, 0.5), mask))
    # test = processor.get_gabor_features()
    print("learn data")
    tmp_data, tmp_class, feature_size = Learner.get_learn_data(Learner, image, processor.expert_image, 1000)
    if data_array is None:
        data_array = tmp_data
        class_array = tmp_class
    else:
        data_array = np.append(data_array, tmp_data)
        class_array = np.append(class_array, tmp_class)

data_array = np.array(data_array)
data_array = data_array.reshape(int(data_array.shape[0] / feature_size), feature_size)

# matrix = Learner.k_fold(data_array, class_array, 2)
print("LEARN")
clf = Learner.learn(data_array, class_array)
matrix = np.zeros((2, 2))
matrix_retard = np.zeros((2, 2))
for image_array in images_array:
    print("Compare Start")
    matrix_retard += ImageProcessor.compare_images(image_array[0], image_array[1])
    # ImageProcessor.show_given_image(image_array[0])
    # ImageProcessor.show_given_image(image_array[1])
    print("Predict")
    matrix += Learner.get_predict_matrix(Learner, 10000, image_array[0], image_array[1], clf)
#
acc = Learner.get_accuracy(matrix)
sen = Learner.get_sensitivity(matrix)
spec = Learner.get_specificity(matrix)

print("Neural")
print("Acc: " + str(acc) + ",\nSen: " + str(sen) + ",\nSpec: " + str(spec))

acc = Learner.get_accuracy(matrix_retard)
sen = Learner.get_sensitivity(matrix_retard)
spec = Learner.get_specificity(matrix_retard)

print("Tard")
print("Acc: " + str(acc) + ",\nSen: " + str(sen) + ",\nSpec: " + str(spec))

# matrix_fold, clf = Learner.k_fold(data_array, class_array)

# acc = Learner.get_accuracy(matrix_fold)
# sen = Learner.get_sensitivity(matrix_fold)
# spec = Learner.get_specificity(matrix_fold)
#
# print("Fold")
# print("Acc: " + str(acc) + ",\nSen: " + str(sen) + ",\nSpec: " + str(spec))

# matrix = np.zeros((2, 2))
# for image_array in images_array:
#     matrix += Learner.get_predict_matrix(Learner, 10000, image_array[0], image_array[1], clf)
# #
# acc = Learner.get_accuracy(matrix)
# sen = Learner.get_sensitivity(matrix)
# spec = Learner.get_specificity(matrix)
#
# print("Neural after fold")
# print("Acc: " + str(acc) + ",\nSen: " + str(sen) + ",\nSpec: " + str(spec))

print("end")
