"""
    A much better neural network built with Keras. Designed to be identical to
    the self coded neural network in terms of number of hidden layers/nodes,
    activation functions, stochastic gradient descent parameters, etc.
    Tim Coutinho
"""

from keras import layers, backend
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from mnist import MNIST
from tensorflow import ConfigProto, GPUOptions, Session
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

num_train, num_test = 60000, 10000  # Default to max possible
max_epochs = 50
batch_size = 100  # Number of images to go through before updating weights

mnist_data = MNIST('mnist-data')
X_train, y_train = mnist_data.load_training()
X_train, y_train = X_train[:num_train], y_train[:num_train]
X_train = np.reshape(X_train, (num_train, 28, 28)+(1,))
y_train = to_categorical(y_train)

X_test, y_test = mnist_data.load_testing()
X_test, y_test = X_test[:num_test], y_test[:num_test]
X_test = np.reshape(X_test, (num_test, 28, 28)+(1,))
y_test = to_categorical(y_test)

tf_config = ConfigProto(gpu_options=GPUOptions(allow_growth=True))
backend.set_session(Session(config=tf_config))

def main():
    # Functional
    img_input = layers.Input(shape=(28, 28, 1))
    # Additional layers really not necessary on this simple of a dataset
    # x = layers.Conv2D(16, 3, activation='relu')(img_input)
    # x = layers.MaxPooling2D(2)(x)
    # x = layers.Conv2D(32, 3, activation='relu')(x)
    # x = layers.MaxPooling2D(2)(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.Conv2D(64, 3, activation='relu')(x)
    # x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(img_input)
    x = layers.Dense(100, activation='sigmoid')(x)
    output = layers.Dense(10, activation='sigmoid')(x)
    model = Model(img_input, output)
    # Sequential
    # model = Sequential([layers.Flatten(input_shape=(28, 28, 1)),
    #                     layers.Dense(100, activation='sigmoid'), 
    #                     layers.Dense(10, activation='sigmoid')])
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.1, momentum=0.9), metrics=['acc'])
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    history = model.fit_generator(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=test_datagen.flow(X_test, y_test,
                                          batch_size=batch_size),
        steps_per_epoch=num_train/batch_size,
        validation_steps=num_test/batch_size,
        epochs=max_epochs, verbose=2).history
    epochs = range(max_epochs)
    sns.set_style('darkgrid')
    plt.plot(epochs, history['acc'], label='Training')
    plt.plot(epochs, history['val_acc'], label='Testing')
    plt.title('Training and testing accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, history['loss'], label='Training')
    plt.plot(epochs, history['val_loss'], label='Testing')
    plt.title('Training and testing loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
