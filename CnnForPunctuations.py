import tensorflow as tf
from tensorflow import keras
# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from google.colab import drive
import os
import numpy
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10


def resize_image(input_image_path,
                 output_image_path,
                 size):
        original_image = Image.open(input_image_path)
        width, height = original_image.size
        print('The original image size is {wide} wide x {height} '
              'high'.format(wide=width, height=height))

        resized_image = original_image.resize(size)
        width, height = resized_image.size
        print('The resized image size is {wide} wide x {height} '
              'high'.format(wide=width, height=height))
        resized_image.show()
        resized_image.save(output_image_path)


os.mkdir('train1')
os.mkdir('callbacks')

dir_name = "/content/cars"
test = os.listdir(dir_name)
i = 0
for filename in test:
    if (filename == ".ipynb_checkpoints"):
        continue
    original_image = Image.open("/content/cars/" + filename)
    width, height = original_image.size
    if (width >= 300) and (height >= 300):
        original_image.save('/content/train1/' + filename)
        i += 1
        if (i % 100 == 0):
            print(i)
dir_name = "/content/SUVs"
test = os.listdir(dir_name)
i = 0
for filename in test:
        if (filename == ".ipynb_checkpoints"):
            continue
        original_image = Image.open("/content/SUVs/" + filename)
        width, height = original_image.size
        if (width >= 100) and (height >= 100):
            original_image.save('/content/train1/' + filename)
            i += 1
            if (i % 100 == 0):
                print(i)
image_files = []
dir_name = "/content/cars"
test = os.listdir(dir_name)
for filename in test:
    if filename.endswith(".jpg"):
        image_files.append(filename)
os.chdir("..")
with open("/content/train_cars.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")

image_files = []
dir_name = "/content/SUVs"
test = os.listdir(dir_name)
for filename in test:
    if filename.endswith(".jpg"):
        image_files.append(filename)
os.chdir("..")
with open("/content/train_SUVs.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")
with open('/content/train_cars.txt') as f1:
  cars = f1.readlines()
with open('/content/train_SUVs.txt') as f2:
  SUVs = f2.readlines()
train_labels = []
dir_name = "/content/train1"
w = 0
e = 0
test = os.listdir(dir_name)
for filename in test:
  if cars.count(filename + "\n"):
      train_labels.append([0])
  elif SUVs.count(filename + "\n"):
      train_labels.append([1])
  else:
    print("ERROR")
    print(filename)
for filename in test:
  if(filename == ".ipynb_checkpoints"):
    continue
  resize_image(input_image_path=('/content/train1/' + filename), output_image_path=('/content/train1/' + filename), size=(128, 128))
images_train = []
for filename in test:
  print(filename)
  if(filename == ".ipynb_checkpoints"):
    continue
  arr = np.array(Image.open('/content/train1/' + filename).convert("RGB"))
  images_train.append(arr)
X_train  = np.asarray(images_train)
y_train = np.asarray(train_labels)
X_train = X_train / 255.0
y_train = np_utils.to_categorical(y_train)
X_train.shape
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
batch_size  = 64
epochs = 18
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
model.save("/content/qqqqqq/model.h5")
test_patch = '/content/SUVs/clon000052.jpg'
resize_image(input_image_path=test_patch, output_image_path=test_patch, size=(128, 128))
arr = np.array(Image.open(test_patch).convert("RGB"))
img = np.asarray(arr)
img = (np.expand_dims(img,0))
predictions_single = model.predict(img)
print(predictions_single)