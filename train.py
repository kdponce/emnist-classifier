from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D
from keras.utils import to_categorical
from keras.optimizers import SGD
from emnist_decoder import read_idx
import tensorflow as tf

# Only allocates a subset of the available GPU Memory and take more as needed.
# Prevents "Failed to get convolution algorithm" error on Elementary OS Juno.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Load EMNIST Alphabet data
train_images = read_idx('dataset/emnist-letters-train-images-idx3-ubyte.gz')
train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
train_labels = read_idx('dataset/emnist-letters-train-labels-idx1-ubyte.gz')
train_labels[:] = [x - 1 for x in train_labels]
train_labels = to_categorical(train_labels)

test_images = read_idx('dataset/emnist-letters-test-images-idx3-ubyte.gz')
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
test_labels = read_idx('dataset/emnist-letters-test-labels-idx1-ubyte.gz')
test_labels[:] = [x - 1 for x in test_labels]
test_labels = to_categorical(test_labels)

# Create model derived from LeNet-5
model = Sequential()
model.add(Conv2D(6, kernel_size=5, strides=1, activation='relu', input_shape=(28, 28, 1)))
model.add(AveragePooling2D(pool_size=2, strides=2))
model.add(Conv2D(16, kernel_size=5, strides=1, activation='relu'))
model.add(AveragePooling2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(26, activation='softmax'))

# Compile and display model layers
model.compile(optimizer=SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train & evaluate model on EMNIST Letters Dataset
model.fit(train_images, train_labels, batch_size=32, epochs=3, validation_split=0.2)
results = model.evaluate(test_images, test_labels)

# Save model & print evaluation results
model.save('model.h5')
print('\nEvaluation Accuracy: {}\nEvaluation Loss: {}'.format(results[1], results[0]))
