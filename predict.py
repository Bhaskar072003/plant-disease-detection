"""
    This file used to load our model and predict batch of images from a directory.
"""
import os
import numpy as np
from tqdm import tqdm
import tensorflow.keras as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

model = K.models.load_model("root/cnn_model.keras")

types = ['Healthy','Powdery','Rust']
classes=  ['Healthy','Powdery','Rust']

img_path = os.listdir('predictions')
for i in tqdm(img_path):
    print('\nfn',i)
    fname = 'predictions'+'/'+i
    img = image.load_img(fname, target_size=(192, 192))
    x = img_to_array(img)

    prediction = model.predict(np.array([x]))[0]
    test_pred = np.argmax(prediction)
    print("=========================", classes[test_pred])

    # result = [(types[i], float(prediction[i]) * 100.0) for i in range(len(prediction))]
    # result.sort(reverse=True, key=lambda x: x[1])

    # print(f'Image name: {i}')
    # for j in range(6):
    #     (class_name, prob) = result[j]
    #     print("Top %d ====================" % (j + 1))
    #     print(class_name + ": %.2f%%" % (prob))
