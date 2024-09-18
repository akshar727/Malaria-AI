import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pickle
import cv2

DATADIR = "cell_images"

CATAGORIES = ["Parasitized", "Uninfected"]

IMG_SIZE = 80

training_data = []

X = []
y = []

def create_training_data():
    for category in CATAGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATAGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y= np.array(y)
print(X.shape)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print("Successfully formatted Dataset for CNN training.")