# "!pip3 install torch==1.6
# !pip3 install PyYAML==5.3
# !pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
# !pip3 install torchvision==0.7.0
# !pip3 install Cython
# !pip3 install numpy
# !pip install pillow==4.1.1
# %reload_ext autoreload
# %autoreload
# !git clone https://github.com/ria-com/nomeroff-net.git
# %cd nomeroff-net
# !git clone https://github.com/youngwanLEE/centermask2.git
# %cd /content
# !git clone https://github.com/AlexeyAB/darknet
# %cd darknet
# !sed -i 's/OPENCV=0/OPENCV=1/' Makefile
# !sed -i 's/GPU=0/GPU=1/' Makefile
# !sed -i 's/CUDNN=0/CUDNN=1/' Makefile
# !make


from keras.models import load_model
from google.colab import drive
import numpy as np
import re
from PIL import Image
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
from google.colab import files
import scipy.io
import numpy as np
import sys
import matplotlib.image as mpimg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
NOMEROFF_NET_DIR = os.path.abspath('/content/nomeroff-net')


# метод который переводит изображение в нужное разрешение
def resize_image(input_image_path,
                 output_image_path,
                 size):
        original_image = Image.open(input_image_path)
        width, height = original_image.size
        resized_image = original_image.resize(size)
        width, height = resized_image.size
        resized_image.show()
        resized_image.save(output_image_path)


# метод который выводит фото на экран
def imShow(path):
      image = cv2.imread(path)
      height, width = image.shape[:2]
      resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)
      fig = plt.gcf()
      fig.set_size_inches(18, 10)
      plt.axis("off")
      plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
      plt.show()


def get_min(arrPoints):
  min_x = 100000
  min_y = 100000
  max_x = 0
  max_y = 0
  for i in arrPoints:
    if i[0] < min_x:
       min_x = i[0]
    if i[0] > max_x:
       max_x = i[0]
    if i[1] < min_y:
       min_y = i[1]
    if i[1] > max_y:
       max_y = i[1]
  return (min_x, min_y, max_x, max_y)


#прогонка через нейронку если много фото
!./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights?dl=0 -ext_output -dont_show >result.txt < data/train.txt

#парсер из файла со всеми результатоми нахождения машин result.txt
dict_of_photos = {}
class ObjectVideo(object):
    def __init__(self, classification, left_x, top_y, width, height, patch):
        self.classification = classification
        self.left_x = left_x
        self.top_y = top_y
        self.width = width
        self.height = height
        self.patch = patch
    def __str__(self):
      return (str(self.classification) + " " + str(self.left_x) + " " + str(self.top_y) + " " + str(self.width) + " " + str(self.height) + " " + str(self.patch))
with open('result.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(text)
array = re.findall(r'data/obj2/[^&]+Enter', text)
print(array)
newArr = array[0].split('Enter')
newArr.pop()
for i in newArr:
    name = re.findall(r'data/obj2/[^:]+', i)
    name = name[0].split('/')
    name = name[2]
    dict_of_photos[name] = []
    cars = re.findall(r'car[^\n]+', i)
    if len(cars) != 0:
        for car in cars:
            coord = car.split('(')
            coord = coord[1].split(' ')
            coord = [x for x in coord if x]
            left_x = int(coord[1])
            top_y = int(coord[3])
            width = int(coord[5])
            heigth = coord[7]
            heigth = int(heigth[:len(heigth)-1])
            dict_of_photos[name].append(ObjectVideo('car', left_x, top_y, width, heigth, name))
    trucks = re.findall(r'truck[^\n]+', i)
    if len(trucks) != 0:
        for truck in trucks:
            coord = truck.split('(')
            coord = coord[1].split(' ')
            coord = [x for x in coord if x]
            left_x = int(coord[1])
            top_y = int(coord[3])
            width = int(coord[5])
            heigth = coord[7]
            heigth = int(heigth[:len(heigth)-1])
            dict_of_photos[name].append(ObjectVideo('truck', left_x, top_y, width, heigth, name))
    buses = re.findall(r'bus[^\n]+', i)
    if len(buses) != 0:
        for bus in buses:
            coord = bus.split('(')
            coord = coord[1].split(' ')
            coord = [x for x in coord if x]
            left_x = int(coord[1])
            top_y = int(coord[3])
            width = int(coord[5])
            heigth = coord[7]
            heigth = int(heigth[:len(heigth)-1])
            dict_of_photos[name].append(ObjectVideo('bus', left_x, top_y, width, heigth, name))

#обрезка фото
dict_classes = {"bus": 0, "car": 1, "truck": 2}
f = open('classes.txt', 'w')
for image1, value in dict_of_photos.items():
  print(image1)
  print(value)
  for i in range(len(value)):
    image = Image.open('/content/darknet/data/obj2/' + image1)
    if(dict_classes[value[i].classification] == 1):
      cropped = image.crop((value[i].left_x, value[i].top_y, value[i].left_x + value[i].width, value[i].top_y + value[i].height))
      orig_name = str(value[i].patch.replace(".jpg", ""))
      name_file = "clon" + str(i) + orig_name + ".jpg"
      open(name_file, 'tw')
      name_directory = "/content/darknet/data/clons/" + name_file
      cropped.save(name_directory)
      f.write(str(dict_classes[value[i].classification]) + "\n")
f.close()

#Нахождение всех машин на фото
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
!./darknet detector test /content/darknet/cfg/coco.data /content/darknet/cfg/yolov4.cfg yolov4.weights?dl=0 -ext_output -dont_show >result.txt /content/еуыеый.jpg
imShow('predictions.jpg')

from PIL import Image
image = Image.open('/content/1234567.tif')
new_image = image.resize((200, 200))

#Парсинг данных о машине из файла result.txt
dict_of_photos = {}
class ObjectVideo(object):
    def __init__(self, classification, left_x, top_y, width, height, patch, clon_name):
        self.classification = classification
        self.left_x = left_x
        self.top_y = top_y
        self.width = width
        self.height = height
        self.patch = patch
        self.clon_name = clon_name
    def __str__(self):
      return (str(self.classification) + " " + str(self.left_x) + " " + str(self.top_y) + " " + str(self.width) + " " + str(self.height) + " " + str(self.patch))
with open('result.txt', 'r', encoding='utf-8') as f:
    text = f.read()
name = re.findall(r'data/obj2/[^:]+', text)
name = name[0].split('/')
name = name[2]
dict_of_photos[name] = []
cars = re.findall(r'car[^\n]+', text)
if len(cars) != 0:
  for car in cars:
    coord = car.split('(')
    coord = coord[1].split(' ')
    coord = [x for x in coord if x]
    left_x = int(coord[1])
    top_y = int(coord[3])
    width = int(coord[5])
    heigth = coord[7]
    heigth = int(heigth[:len(heigth)-1])
    dict_of_photos[name].append(ObjectVideo('car', left_x, top_y, width, heigth, name, ''))
trucks = re.findall(r'truck[^\n]+', text)
if len(trucks) != 0:
  for truck in trucks:
    coord = truck.split('(')
    coord = coord[1].split(' ')
    coord = [x for x in coord if x]
    left_x = int(coord[1])
    top_y = int(coord[3])
    width = int(coord[5])
    heigth = coord[7]
    heigth = int(heigth[:len(heigth)-1])
    dict_of_photos[name].append(ObjectVideo('truck', left_x, top_y, width, heigth, name, ''))
buses = re.findall(r'bus[^\n]+', text)
if len(buses) != 0:
  for bus in buses:
    coord = bus.split('(')
    coord = coord[1].split(' ')
    coord = [x for x in coord if x]
    left_x = int(coord[1])
    top_y = int(coord[3])
    width = int(coord[5])
    heigth = coord[7]
    heigth = int(heigth[:len(heigth)-1])
    dict_of_photos[name].append(ObjectVideo('bus', left_x, top_y, width, heigth, name, ''))

#вырезание каждого автомобиля на фото отдельно
dict_classes = {"bus": 0, "car": 1, "truck": 2}
f = open('classes.txt', 'w')
for image1, value in dict_of_photos.items():

  for i in range(len(value)):
    image = Image.open('/content/darknet/data/obj2/' + image1)
    cropped = image.crop((value[i].left_x, value[i].top_y, value[i].left_x + value[i].width, value[i].top_y + value[i].height))
    orig_name = str(value[i].patch.replace(".jpg", ""))
    name_file = "clon" + str(i) + orig_name + ".jpg"
    open(name_file, 'tw')
    name_directory = "/content/darknet/data/clons/" + name_file
    value[i].clon_name = name_directory
    cropped.save(name_directory)
    f.write(str(dict_classes[value[i].classification]) + "\n")
f.close()

#скачивание и присваивание наших обученных нейронок
from keras.models import load_model
!wget https://www.dropbox.com/s/wmh5m6c4l3bcn6e/model.h5?dl=0
!wget https://www.dropbox.com/s/3nmk38hj0cz2uvl/model_nomers%20%281%29.h5?dl=0
!wget https://www.dropbox.com/s/m3sequxh8wo3qbv/model_lights%20%281%29.h5?dl=0
model = load_model('model.h5?dl=0')
model_for_nomers = load_model('model_nomers (1).h5?dl=0')
model_lights = load_model('model_lights (1).h5?dl=0')

# основной процесс с нахождением и вычислением типа номера и т.д
objects = {}
dict_classes_for_cars = {1: 'Внедорожник', 0: 'Легковой автомобиль'}
dict_classes_lights = {1: "Фары выключены", 2: "Фары включены"}
for key, value in dict_of_photos.items():
  for object_car in value:
    if object_car.classification == 'car':

      test_patch = object_car.clon_name
      number = clone_number(test_patch) #Нахождение и определение типа номера, метод будет ниже
      resize_image(input_image_path=test_patch, output_image_path=test_patch, size=(128, 128)) #изменение размеров изображение до 128х128
      arr = np.array(Image.open(test_patch).convert("RGB"))#перевод изображения в массив (128х128х3)
      img = np.asarray(arr)
      img = (np.expand_dims(img,0))
      predictions_single = model.predict(img)#предсказание типа кузова легковой/внедорожник
      prediction_lights = model_lights.predict(img)#предсказание включены ли фары
      objects[object_car.clon_name] = [dict_classes_for_cars[predictions_single.argmax()], dict_classes_lights[prediction_lights.argmax(1)[0]], number]
    else:
      test_patch = object_car.clon_name
      number = clone_number(test_patch)
      resize_image(input_image_path=test_patch, output_image_path=test_patch, size=(128, 128))
      arr = np.array(Image.open(test_patch).convert("RGB"))
      img = np.asarray(arr)
      img = (np.expand_dims(img,0))
      prediction_lights = model_lights.predict(img)
      if (object_car.classification == 'truck'):
        objects[object_car.clon_name] = ['грузовик', dict_classes_lights[prediction_lights.argmax(1)[0]], number]
      else:
        objects[object_car.clon_name] = ['автобус', dict_classes_lights[prediction_lights.argmax(1)[0]], number]


#удаление всех фоток чтобы не засорять память
folder = '/content/darknet/data/clons'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    if os.path.isfile(file_path):
        os.unlink(file_path)
folder = '/content/sae'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    if os.path.isfile(file_path):
        os.unlink(file_path)


#нахождение номера на фото с помощью библиотеки nomeroff-net
dict_classes_nomers = {1: "Дипломатический", 2: "Полицейский", 3: "Такси", 4: "Гражданский", 5: "Военный"}
def clone_number(img_path):
  orig_name = img_path.replace(".jpg", "")
  orig_name = orig_name.replace("/", "")
  img1 = mpimg.imread(img_path)

  cv_imgs_masks = nnet.detect_mask([img1])
  imgs = [img1]


  for img1, cv_img_masks in zip(imgs, cv_imgs_masks):
      try:
        arrPoints = rectDetector.detect(cv_img_masks)
      except:
        continue
      for i in range(len(arrPoints)):
        image = Image.open(img_path)
        cropped = image.crop(get_min(arrPoints[i]))
        name_file = "clon" + str(i) + orig_name + ".jpg"
        name_dir = '/content/sae/' + name_file
        cropped.save(name_dir)
        image_crop = Image.open(name_dir)
        if (i==0):
          width, height = image_crop.size
          final_name_dir = name_dir
        if (i!=0) and (image_crop.size[0] > width) and (image_crop.size[1] > height):
          width, height = image_crop.size
          final_name_dir = name_dir
      resize_image(input_image_path=final_name_dir, output_image_path=final_name_dir, size=(128, 128))
      arr2 = np.array(Image.open(final_name_dir).convert("RGB"))
      img2 = np.asarray(arr2)
      img2 = (np.expand_dims(img2,0))
      predictions_single = model_for_nomers.predict(img2)
      return dict_classes_nomers[predictions_single.argmax(1)[0]]


#отрисовка всехнадписей с данными об авто на фото
from PIL import Image, ImageColor
from PIL import ImageDraw
from PIL import ImageFont
image = Image.open('/content/darknet/data/obj2/krasnye-znaki.jpg')
font = ImageFont.truetype('/content/9041.ttf', 24)
draw = ImageDraw.Draw(image)
for key, value in dict_of_photos.items():
  for object_car in value:
    line = objects[object_car.clon_name][0] + '\n' + objects[object_car.clon_name][1] + '\n' + str(objects[object_car.clon_name][2])
    draw.line((object_car.left_x, object_car.top_y, object_car.left_x, object_car.top_y + object_car.height), width=2, fill='red')
    draw.line((object_car.left_x, object_car.top_y, object_car.left_x + object_car.width, object_car.top_y), width=2, fill='red')
    draw.line((object_car.left_x + object_car.width, object_car.top_y, object_car.left_x + object_car.width, object_car.top_y + object_car.height), width=2, fill='red')
    draw.line((object_car.left_x, object_car.top_y + object_car.height, object_car.left_x + object_car.width, object_car.top_y + object_car.height), width=2, fill='red')
    draw.text((object_car.left_x, object_car.top_y), line, fill='red', font = font)


image.save("empty.png", "PNG")



