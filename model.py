from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D


# Create model derived from LeNet-5
def create_model():
    model = Sequential()
    model.add(Conv2D(6, kernel_size=5, strides=1, activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Conv2D(16, kernel_size=5, strides=1, activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Conv2D(120, kernel_size=5, strides=1, activation='relu', padding='valid'))
    model.add(Flatten())
    model.add(Dense(84, activation='relu'))
    model.add(Dense(26, activation='softmax'))
    return model
