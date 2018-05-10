"""
    Neural network built to classify the MNIST dataset of handwritten digits.
    Numpy's @ operator is used for matrix multiplication and dot products.
    Tim Coutinho
"""

from pprint import pprint
from random import uniform

from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

num_train, num_test = 60000, 10000  # Default to max possible
num_inputs, num_outputs, num_hiddens = 784, 10, [20, 50, 100]
max_value = 255  # For normalizing the data
max_epochs = 50
batch_size = 100  # Number of images to go through before updating weights
num_batches = num_train//batch_size  # For splitting data into chunks
h_num_weights = range(num_inputs+1)
h_init_weights = [uniform(-.05, .05) for _ in h_num_weights]
train_accuracies, test_accuracies = [[], [], []], [[], [], []]
momentums = [0, 0.25, 0.5, 0.9]  # Experiment 2
train_fracs = [2, 4]  # Experiment 3
acc_limit = 0.01  # Accuracy change between epochs below this ends training
learning_rate = 0.1
confusion_matrix = [[[0 for _ in range(num_outputs)]
                     for _ in range(num_outputs)] for _ in range(3)]

mnist_data = MNIST('mnist-data')
x_train, y_train = mnist_data.load_training()
x_train, y_train = x_train[:num_train], np.array(y_train[:num_train])
x_test, y_test = mnist_data.load_testing()
x_test, y_test = x_test[:num_test], y_test[:num_test]
# Biases, normalization
x_train = np.hstack((np.ones((num_train, 1)),
                    [x/max_value for x in np.array(x_train, dtype='f')]))
x_test = np.hstack((np.ones((num_test, 1)),
                   [x/max_value for x in np.array(x_test, dtype='f')]))


def main():
    sns.set_style('darkgrid')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # experiment1()
    # experiment2()
    # experiment3()


def experiment1(num_hiddens=num_hiddens, momentum=momentums[3]):
    _, ax = plt.subplots()

    print('Performing experiment with varying number of hidden neurons.')
    for i, n in enumerate(num_hiddens):
        n += 1  # Bias
        o_num_weights = range(n)
        o_weights = np.array([[uniform(-.05, .05) for _ in o_num_weights]
                              for _ in range(num_outputs)])
        h_weights = [h_init_weights for _ in range(n)]

        # Initial accuracy
        print('Epoch 0')
        accuracies = accuracy(h_weights, o_weights)
        train_accuracies[i] += accuracies[0]
        test_accuracies[i] += accuracies[1]
        print(f'Train accuracy: {train_accuracies[i][0]}')
        print(f'Test accuracy: {test_accuracies[i][0]}')

        for epoch in range(1, max_epochs):
            # Train training data
            print(f'Epoch {epoch}')
            h_weights, o_weights = train(h_weights, o_weights, i, n, momentum)

            # Calculate accuracies
            accuracies = accuracy(h_weights, o_weights)
            train_accuracies[i] += accuracies[0]
            test_accuracies[i] += accuracies[1]
            print(f'Train accuracy: {train_accuracies[i][epoch]}')
            print(f'Test accuracy: {test_accuracies[i][epoch]}')
            if abs(train_accuracies[i][epoch]
                   - train_accuracies[i][epoch-1]) < acc_limit:
                break

        # Update confusion matrix
        for image, target in zip(x_test, y_test):
            predicted = np.argmax(sig(o_weights@sig(h_weights@image)))
            confusion_matrix[i][predicted][target] += 1
        ax.plot(train_accuracies[i], label=f'Training ({n-1} hiddens)')
        ax.plot(test_accuracies[i], label=f'Testing ({n-1} hiddens)')

    print('Confusion Matrix:')
    pprint(confusion_matrix)
    ax.legend()
    plt.show()


def experiment2(n=num_hiddens[2], momentums=momentums[:3]):
    _, ax = plt.subplots()
    n += 1  # Bias
    o_num_weights = range(n)
    o_init_weights = [[uniform(-.05, .05) for _ in o_num_weights]
                      for _ in range(num_outputs)]

    print('Performing experiment with varying momentum values.')
    for i, momentum in enumerate(momentums):
        o_weights = np.array(o_init_weights)  # Same initial weights each trial
        h_weights = [h_init_weights for _ in o_num_weights]

        # Initial accuracy
        print('Epoch 0')
        accuracies = accuracy(h_weights, o_weights)
        train_accuracies[i] += accuracies[0]
        test_accuracies[i] += accuracies[1]
        print(f'Train accuracy: {train_accuracies[i][0]}')
        print(f'Test accuracy: {test_accuracies[i][0]}')

        for epoch in range(1, max_epochs):
            # Train training data
            print(f'Epoch {epoch}')
            h_weights, o_weights = train(h_weights, o_weights, i, n, momentum)

            # Calculate accuracies
            accuracies = accuracy(h_weights, o_weights)
            train_accuracies[i] += accuracies[0]
            test_accuracies[i] += accuracies[1]
            print(f'Train accuracy: {train_accuracies[i][epoch]}')
            print(f'Test accuracy: {test_accuracies[i][epoch]}')
            if abs(train_accuracies[i][epoch]
                   - train_accuracies[i][epoch-1]) < acc_limit:
                break

        # Update confusion matrix
        for image, target in zip(x_test, y_test):
            predicted = np.argmax(sig(o_weights@sig(h_weights@image)))
            confusion_matrix[i][predicted][target] += 1
        ax.plot(train_accuracies[i], label=f'Training ({momentum} momentum)')
        ax.plot(test_accuracies[i], label=f'Testing ({momentum} momentum)')

    print('Confusion Matrix:')
    pprint(confusion_matrix)
    ax.legend()
    plt.show()


def experiment3(n=num_hiddens[2], momentum=momentums[3]):
    _, ax = plt.subplots()
    n += 1
    o_num_weights = range(n)
    o_init_weights = [[uniform(-.05, .05) for _ in o_num_weights]
                      for _ in range(num_outputs)]

    print('Performing experiment with varying number of training examples.')
    for i, frac in enumerate(train_fracs):
        num_train2 = num_train // frac
        o_weights = [*o_init_weights]  # Same initial weights each trial
        h_weights = [h_init_weights for _ in o_num_weights]

        # Initial accuracy
        print('Epoch 0')
        accuracies = accuracy(h_weights, o_weights, num_train=num_train2)
        train_accuracies[i] += accuracies[0]
        test_accuracies[i] += accuracies[1]
        print(f'Train accuracy: {train_accuracies[i][0]}')
        print(f'Test accuracy: {test_accuracies[i][0]}')

        for epoch in range(1, max_epochs):
            # Train training data
            print(f'Epoch {epoch}')
            h_weights, o_weights = train(h_weights, o_weights, i, n, momentum,
                                         x_train=x_train, y_train=y_train)

            # Calculate accuracies
            accuracies = accuracy(h_weights, o_weights)
            train_accuracies[i] += accuracies[0]
            test_accuracies[i] += accuracies[1]
            print(f'Train accuracy: {train_accuracies[i][epoch]}')
            print(f'Test accuracy: {test_accuracies[i][epoch]}')
            if abs(train_accuracies[i][epoch]
                   - train_accuracies[i][epoch-1]) < acc_limit:
                break

        # Update confusion matrix
        for image, target in zip(x_test, y_test):
            predicted = np.argmax(sig(o_weights@sig(h_weights@image)))
            confusion_matrix[i][predicted][target] += 1
        ax.plot(train_accuracies[i], label=f'Training ({num_train2} examples)')
        ax.plot(test_accuracies[i], label=f'Testing ({num_train2} examples)')

    print('Confusion Matrix:')
    pprint(confusion_matrix[:len(train_fracs)])
    ax.legend()
    plt.show()


def train(h_weights, o_weights, i, n, momentum,
          x_train=x_train, y_train=y_train):
    h_errors = [0] * n
    h_deltas = np.array([[0.0]*(num_inputs+1)]*n)
    h_prev_deltas = np.array(h_deltas)
    o_errors = [0] * num_outputs
    o_deltas = np.array([[0.0]*n]*num_outputs)
    o_prev_deltas = np.array(o_deltas)

    for j, (image, target) in enumerate(zip(x_train, y_train)):
        # Forward (calculate values)
        h_values = np.array(sig(h_weights@image))
        h_values[0] = 1.0  # Keep bias 1
        o_values = np.array(sig(o_weights@h_values))

        # Backward (calculate errors, update weights)
        o_errors = np.array([o*(1-o)*(0.9-o) if k == target
                             else o*(1-o)*(0.1-o)
                             for k, o in enumerate(o_values)])
        deltas = [learning_rate*error*h_values + momentum*delta
                  for error, delta in zip(o_errors, o_prev_deltas)]
        o_prev_deltas = np.array(deltas)
        if j > 0 and j % batch_size == 0:
            o_weights += o_deltas/batch_size
            o_deltas *= 0
        else:
            o_deltas += deltas

        h_o_weights = [[w[k] for w in o_weights] for k in range(n)]
        h_errors = np.array([h*(1-h)*(h_o_weights[k]@o_errors)
                             for k, h in enumerate(h_values)])
        deltas = [learning_rate*error*np.array(image) + momentum*delta
                  for error, delta in zip(h_errors, h_prev_deltas)]
        h_prev_deltas = np.array(deltas)
        if j > 0 and j % batch_size == 0:
            h_weights += h_deltas/batch_size
            h_deltas *= 0
        else:
            h_deltas += deltas

    return h_weights, o_weights


def accuracy(h_weights, o_weights, num_train=num_train):
    correct_train = correct_test = 0
    for image, target in zip(x_train, y_train):
        predicted = np.argmax(sig(o_weights@sig(h_weights@image)))
        correct_train += 1 if predicted == target else 0
    for image, target in zip(x_test, y_test):
        predicted = np.argmax(sig(o_weights@sig(h_weights@image)))
        correct_test += 1 if predicted == target else 0
    return [correct_train / num_train * 100], [correct_test / num_test * 100]


def sig(x):
    return 1/(1+np.exp(-x))


if __name__ == '__main__':
    main()
