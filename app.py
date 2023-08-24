import numpy as np
import cv2
# import keras
from keras.models import model_from_json

import tensorflow as tf
from string import ascii_uppercase
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)
directory=""
json_file = open(directory+"model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights(directory+"model-bw.h5")

cam = cv2.VideoCapture(0)

alpha_dict = {}
for i in range(10):
    alpha_dict[i]=str(i)
j=10
for i in ascii_uppercase:
   alpha_dict[j] = i
   j = j + 1

print(alpha_dict)



while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (319, 9), (620 + 1, 309), (0, 255, 0), 1)
    roi = frame[10:308, 320:620]

    # cv2.imshow("Frame", frame)
    gray= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
    final_image = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2.8)
    # ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("BW", final_image)
    final_image = cv2.resize(final_image, (200, 200))
    final_image = np.reshape(final_image, (1, final_image.shape[0], final_image.shape[1], 1))
    pred = loaded_model.predict(final_image)
    # print(alpha_dict[np.argmax(pred)])
    cv2.putText(frame,alpha_dict[np.argmax(pred)], (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    cv2.imshow("Frame", frame)
    if(cv2.waitKey(1) & 0XFF==ord("q")):
        break

cam.release()
cv2.destroyAllWindows()