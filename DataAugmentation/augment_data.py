import numpy as np
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
from DataAugmentation_TimeseriesData import Augmentation_techniques
class Augmented_data:

    # START AUGMENTING
    def augment_data(train_set, train_label, function):
        train_set_one = train_set
        LABEL = []
        # select random indices
        number_of_rows = int(train_set_one.shape[0] * 0.1)

        # random indices has to be the same for every dimension so that the label can be accurate
        random_indices = np.sort(np.random.choice(train_set_one.shape[0] - 1, size=int(number_of_rows), replace=False))
        train_set_one = train_set_one[random_indices, :]

        train_set_one = train_set_one.transpose()
        if function == 'scale':
            train_set_one = Augmentation_techniques.DA_Scaling(train_set_one)
        elif function == 'jitter':
            train_set_one = Augmentation_techniques.DA_Jitter(train_set_one)
        elif function == 'magWarp':
            train_set_one = Augmentation_techniques.DA_MagWarp(train_set_one)
        elif function == 'timeWarp':
            train_set_one = Augmentation_techniques.DA_TimeWarp(train_set_one)
        elif function == 'rotation':
            train_set_one = Augmentation_techniques.DA_Rotation(train_set_one)
        elif function == 'permutation':
            train_set_one = Augmentation_techniques.DA_Permutation(train_set_one)
        else:
            print("Error no augmentation function")
            return -1
        train_set_one = train_set_one.transpose()

        # take the label and add them as the label for the new augmented data
        LABEL = np.array(train_label[random_indices])
        #     we have ARR which is of shape (6, row, col) with the augmented data
        #     and train_set which is of shape (6, row, col) with the non augmented data

        train_set_augmented = np.concatenate((train_set, train_set_one), axis=0)
        print(train_set[0, 0])
        print(train_set_one[0, 0])
        train_label = np.array(train_label)
        label_set_augmented = np.concatenate((train_label, LABEL))

        return train_set_augmented, label_set_augmented

