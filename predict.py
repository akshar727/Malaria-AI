import tensorflow as tf
import cv2
import keras
# import matplotlib.pyplot as plt
import os
import shutil


CATEGORIES = ["Parasitized", "Uninfected"]

def prepare(path):
    IMG_SIZE = 80
    img_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


model = keras.models.load_model("32x3-CNN-64-95%.keras")
model.summary()
incorrect = 0
total = len(os.listdir("cell_images/Parasitized")) + len(os.listdir("cell_images/Uninfected"))

for image in os.listdir("cell_images/Parasitized"):
    try:
        prediction = model.predict([prepare("cell_images/Parasitized/"+image)])
        if CATEGORIES[int(prediction[0][0])] == "Uninfected":
            incorrect+= 1
    except Exception as e:
        continue

for image in os.listdir("cell_images/Uninfected"):
    try:
        prediction = model.predict([prepare("cell_images/Uninfected/"+image)])
        if CATEGORIES[int(prediction[0][0])] == "Parasitized":
            incorrect+= 1
    except Exception as e:
        continue

print(f"AI got {(total-incorrect)/total * 100}%")
# outputs = {}
# try:
#     shutil.rmtree("test_results")
# except:
#     pass
# for image in os.listdir("test"):
#     try:
#         prediction = model.predict([prepare("test/"+image)])
#         # print(CATEGORIES[int(prediction[0][0])])
#         outputs[image] = CATEGORIES[int(prediction[0][0])]
#     except Exception as e:
#         pass
# os.mkdir("test_results")
# os.mkdir("test_results/Parasitized")
# os.mkdir("test_results/Uninfected")
#
# for image in outputs:
#     shutil.copyfile("test/"+image, "test_results/"+outputs[image]+"/"+image)
# incorrect = 0
# for image in os.listdir("test_results/Parasitized"):
#     if not image.startswith("C3"):
#         incorrect+=1
#
# for image in os.listdir("test_results/Uninfected"):
#     if not image.startswith("C1"):
#         incorrect+=1
#
# print(f"Batch 64 (Best so far) AI got {len(outputs)-incorrect}/{len(outputs)} correct")
#
#
