# MNIST
Two implementations of machine learning algorithms trained to classify numbers using the MNIST dataset.

The perceptron implmentation is simply a neural network with no hidden layers, that uses a simple dot product as the activation function.
Three trials are done with it, one each for the learning rates 0.001, 0.01, and 0.1. Surprisingly, the lowest learning rate performed
the best; as can be seen in Figure 1, it has the smoothest curve and converges quite quickly, though steadily increases the whole way
through. The middle learning rate converges faster, but is cut short fairly quickly due to (perhaps unlucky) minimal increase in accuracy
between epochs. The highest learning rate is simply too large, and the curve simply bounces around the spot of convergence, unable to
increase slow enough without overshooting.
