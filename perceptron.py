"""
    Perceptron built to classify the MNIST dataset of handwritten digits.
    Tim Coutinho
"""

from pprint import pprint
from random import uniform

from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

num_train, num_test = 60000, 10000
num_inputs, num_outputs = 784, 10
max_value = 255
max_epochs = 70
batch_size = 100
weight_range = range(num_inputs+1)
init_weights = [[uniform(-.05, .05) for _ in weight_range]
                for _ in range(num_outputs)]
train_accuracies, test_accuracies = [[], [], []], [[], [], []]
acc_limit = 0.01
learning_rates = [.001, .01, .1]
confusion_matrix = [[[0 for _ in range(num_outputs)]
                     for _ in range(num_outputs)] for _ in range(3)]


def main():
    sns.set_style('darkgrid')
    _, ax = plt.subplots()
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    X_train, y_train, X_test, y_test = get_data()

    for n, learning_rate in enumerate(learning_rates):
        o_weights = [*init_weights]

        # Initial accuracy
        print('Epoch 0')
        accuracies = accuracy(X_train, y_train, X_test, y_test, o_weights)
        train_accuracies[n] += accuracies[0]
        test_accuracies[n] += accuracies[1]
        print(f'Train accuracy: {train_accuracies[n][0]}')
        print(f'Test accuracy: {test_accuracies[n][0]}')
        for epoch in range(1, max_epochs):
            # Train training data
            print(f'Epoch {epoch}')
            errors = [[] for _ in range(num_outputs)]
            for i, (image, target) in enumerate(zip(X_train, y_train)):
                for j, weights in enumerate(o_weights):
                    t = int(j == target)
                    y = int(image@weights > 0)
                    error = learning_rate*(t-y)*image
                    if i > 0 and i % batch_size == 0:
                        o_weights[j] += sum(errors[j])/batch_size
                        errors[j] = []
                    else:
                        errors[j] += [error]
            accuracies = accuracy(X_train, y_train, X_test, y_test, o_weights)
            train_accuracies[n] += accuracies[0]
            test_accuracies[n] += accuracies[1]
            print(f'Train accuracy: {train_accuracies[n][epoch]}')
            print(f'Test accuracy: {test_accuracies[n][epoch]}')
            if abs(train_accuracies[n][epoch]
                   - train_accuracies[n][epoch-1]) < acc_limit:
                break

        # Update confusion matrix
        for image, target in zip(X_test, y_test):
            predicted = np.argmax(o_weights@image)
            confusion_matrix[n][predicted][target] += 1
        ax.plot(train_accuracies[n], label=f'Training (LR={learning_rate})')
        ax.plot(test_accuracies[n], label=f'Testing (LR={learning_rate})')

    print('Confusion Matrix:')
    pprint(confusion_matrix)
    ax.legend()
    plt.show()


def accuracy(X_train, y_train, X_test, y_test, weights):
    correct_train = correct_test = 0
    for image, target in zip(X_train, y_train):
        predicted = np.argmax(weights@image)
        correct_train += int(predicted == target)
    for image, target in zip(X_test, y_test):
        predicted = np.argmax(weights@image)
        correct_test += int(predicted == target)
    return ([correct_train / num_train * 100], [correct_test / num_test * 100])


def get_data():
    mnist_data = MNIST('mnist-data')
    X_train, y_train = mnist_data.load_training()
    X_train, y_train = X_train[:num_train], y_train[:num_train]
    X_test, y_test = mnist_data.load_testing()
    X_test, y_test = X_test[:num_test], y_test[:num_test]
    # Biases, normalization
    X_train = np.hstack((np.ones((num_train, 1)),
                        [x/max_value for x in np.array(X_train, dtype='f')]))
    X_test = np.hstack((np.ones((num_test, 1)),
                        [x/max_value for x in np.array(X_test, dtype='f')]))
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    main()
