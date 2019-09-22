from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from emnist_decoder import read_idx
from model import create_model
import tensorflow as tf
import numpy as np

# Only allocates a subset of the available GPU Memory and take more as needed.
# Prevents "Failed to get convolution algorithm" error on Elementary OS Juno.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Load EMNIST Alphabet data
train_images = read_idx('dataset/emnist-letters-train-images-idx3-ubyte.gz')
test_images = read_idx('dataset/emnist-letters-test-images-idx3-ubyte.gz')

# Rotate images to proper orientation
for x in range(len(train_images)):
    train_images[x] = np.rot90(np.fliplr(train_images[x]))
for x in range(len(test_images)):
    test_images[x] = np.rot90(np.fliplr(test_images[x]))

# Reshape images to fit model requirements
train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))

# Shift labels 0 - A, 25 - Z
train_labels = read_idx('dataset/emnist-letters-train-labels-idx1-ubyte.gz')
train_labels[:] = [x - 1 for x in train_labels]
train_labels = to_categorical(train_labels)
test_labels = read_idx('dataset/emnist-letters-test-labels-idx1-ubyte.gz')
test_labels[:] = [x - 1 for x in test_labels]
test_labels = to_categorical(test_labels)

# Create neural network
model = create_model()

# Compile and display model layers
model.compile(optimizer=SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Create Image Generator for Data Augmentation
train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.2, validation_split=0.2)
val_datagen = ImageDataGenerator(validation_split=0.2)

# Train & evaluate model on EMNIST Letters Dataset
train_generator = train_datagen.flow(train_images, train_labels, batch_size=32, subset='training')
val_generator = val_datagen.flow(train_images, train_labels, batch_size=32, subset='validation')

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_images) / 32, epochs=20,
                    validation_data=val_generator,
                    validation_steps=len(train_images) / 32)

results = model.evaluate(test_images, test_labels)

# Save model & print evaluation results
model.save('model.h5')
print('\nEvaluation Accuracy: {}\nEvaluation Loss: {}'.format(results[1], results[0]))
