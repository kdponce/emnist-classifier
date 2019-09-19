from keras.models import load_model
import numpy as np
import cv2
import tensorflow as tf


# Only allocates a subset of the available GPU Memory and take more as needed.
# Prevents "Failed to get convolution algorithm" error on Elementary OS Juno.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Load model
model = load_model('model.h5')

# Open Class labels dictionary. (human readable label given ID)
classes = eval(open('dataset/classes.txt', 'r').read())

# Load test image and modify size and color channels with opencv
img_path = 'test.jpg'
img = np.asarray(cv2.imread(img_path))[:, :, ::-1]
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.reshape((1, img.shape[0], img.shape[1], 1))


# Run prediction on test image
preds = model.predict(img)
print("Class is: " + classes[np.argmax(preds)])
print("Certainty is: " + str(preds[0][np.argmax(preds)]))
