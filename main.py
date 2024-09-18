import datetime
import os
import pickle
import shutil
import time

import tensorflow as tf
from keras import layers
from keras import Sequential
import matplotlib.pyplot as plt
from keras import callbacks, optimizers
import cv2




batch_sizes = [64]
conv_layers = [3]
layer_sizes = [64]
dense_layers = [0]
epochs = [15]


# BEST TWO COMBINATIONS; 96% ACCURACY NO AUGMENTATION
# batch_sizes = [32,64]
# conv_layers = [3]
# layer_sizes = [32]
# dense_layers = [0]

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X / 255.0

single_model_time = 0


data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal",
                            input_shape=(80,
                                        80,
                                        3))
    ]
)

def get_model_number(batch_size, conv_layer, layer_size, dense_layer):
    batch_size_index = batch_sizes.index(batch_size)
    conv_layer_index = conv_layers.index(conv_layer)
    layer_size_index = layer_sizes.index(layer_size)
    dense_layer_index = dense_layers.index(dense_layer)

    return (batch_size_index * len(conv_layers) * len(layer_sizes) * len(dense_layers) +
            conv_layer_index * len(layer_sizes) * len(dense_layers) +
            layer_size_index * len(dense_layers) +
            dense_layer_index + 1)

for batch_size in batch_sizes:
    for conv_layer in conv_layers:
        for layer_size in layer_sizes:
            for dense_layer in dense_layers:
                for epoch in epochs:
                    print("-" * 20 + "\n")
                    print("Training model with {} conv layers, {} nodes, and {} dense layers with {} batch size".format(
                        conv_layer, layer_size, dense_layer, batch_size))
                    print("Running model number {}/{}".format(
                        get_model_number(batch_size, conv_layer, layer_size, dense_layer),
                        len(conv_layers) * len(layer_sizes) * len(dense_layers) * len(batch_sizes)))
                    print()
                    print("-" * 20)
                    NAME = "{}-conv-{}-nodes-{}-dense-{}-batch".format(conv_layer, layer_size, dense_layer, batch_size)
                    tensorboard = callbacks.TensorBoard(log_dir="logs/{}".format(NAME))

                    model = Sequential([
                        layers.InputLayer(input_shape=(80, 80, 3)),
                        # data_augmentation,
                        layers.Conv2D(layer_size, 3, activation="relu"),
                        layers.MaxPooling2D()
                    ])

                    for l in range(conv_layer - 1):
                        model.add(layers.Conv2D(layer_size, 3, activation="relu"))
                        model.add(layers.MaxPooling2D())

                    model.add(layers.Flatten())



                    for l in range(dense_layer):
                        model.add(layers.Dense(layer_size, activation="relu"))

                    model.add(layers.Dense(1, activation="sigmoid"))

                    model.compile(loss="binary_crossentropy",
                                  optimizer=optimizers.Adam(),
                                  metrics=['accuracy'])

                    model.summary()
                    start = time.time()
                    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                  patience=1, min_lr=0.00001,mode="min",verbose=1)
                    history = model.fit(X, y, batch_size=batch_size, epochs=epoch, validation_split=0.25,callbacks=[reduce_lr])#,tensorboard])
                    end = time.time()
                    # if get_model_number(batch_size, conv_layer, layer_size, dense_layer) == 1:
                    #     single_model_time = end - start
                    #     print("Time to train first model: {} seconds".format(round(single_model_time)))
                    #     # format the time to train all models print to be in HH:MM:SS using a time formatter
                    #     print("Estimated time to train all models (considering varying batch size): {}".format(
                    #         datetime.timedelta(seconds=(1.5 * round(
                    #             single_model_time * len(conv_layers) * len(layer_sizes) * len(dense_layers))))))
                    # else:
                    #     print("Time to train this model: {} seconds".format(round(end - start)))

                    acc = history.history['accuracy']
                    val_acc = history.history['val_accuracy']

                    loss = history.history['loss']
                    val_loss = history.history['val_loss']
                    epochs_range = range(epoch)

                    plt.figure(figsize=(8, 8))
                    plt.subplot(1, 2, 1)
                    plt.plot(epochs_range, acc, label='Training Accuracy')
                    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
                    plt.legend(loc='lower right')
                    plt.title('Training and Validation Accuracy - {} epochs'.format(epoch))

                    plt.subplot(1, 2, 2)
                    plt.plot(epochs_range, loss, label='Training Loss')
                    plt.plot(epochs_range, val_loss, label='Validation Loss')
                    plt.legend(loc='upper right')
                    plt.title('Training and Validation Loss')
                    plt.show()

                    ## START TEST
                    CATEGORIES = ["Parasitized", "Uninfected"]

                    outputs = {}
                    def prepare(path):
                        IMG_SIZE = 80
                        img_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
                    try:
                        shutil.rmtree("test_results")
                    except:
                        pass
                    for image in os.listdir("test"):
                        try:
                            prediction = model.predict([prepare("test/" + image)])
                            outputs[image] = CATEGORIES[int(prediction[0][0])]
                        except Exception as e:
                            pass
                    os.mkdir("test_results")
                    os.mkdir("test_results/Parasitized")
                    os.mkdir("test_results/Uninfected")

                    for image in outputs:
                        shutil.copyfile("test/" + image, "test_results/" + outputs[image] + "/" + image)
                    incorrect = 0
                    for image in os.listdir("test_results/Parasitized"):
                        if not image.startswith("C3"):
                            incorrect += 1

                    for image in os.listdir("test_results/Uninfected"):
                        if not image.startswith("C1"):
                            incorrect += 1

                    print(f"Epoch {epoch} AI got {len(outputs) - incorrect}/{len(outputs)} correct")

                model.save(f"32x3-CNN-{batch_size}.keras")

