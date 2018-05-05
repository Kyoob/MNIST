# MNIST
Two implementations of machine learning algorithms trained to classify numbers using the MNIST dataset.

The perceptron implmentation is simply a neural network with no hidden layers, that uses a simple dot product as the activation function.
Three trials are done with it, one each for the learning rates 0.001, 0.01, and 0.1. Surprisingly, the lowest learning rate performed
the best; as can be seen in Figure 1, it has the smoothest curve and converges quite quickly, though steadily increases the whole way
through. The middle learning rate converges faster, but is cut short fairly quickly due to (perhaps unlucky) minimal increase in accuracy
between epochs. The highest learning rate is simply too large, and the curve simply bounces around the spot of convergence, unable to
increase slow enough without overshooting.

The neural network implementation contains one hidden layer, and the code contains three separate experiments, each changing a different
variable for each trial. Experiment 1 changes the number of hidden neurons each trial, from 20, to 50, to 100. Experiment 2 changes the
momentum value, from 0, to 0.25, to 0.5. Experiment 3 changes the fraction of the training examples used during training, using one half
then one fourth of the 60,000 total. Each experiment uses an otherwise default value of the other changing variables, so that each
experiment only affects one variable. Feel free to change the values of the defaults in the definitions at the beginning of the file to
see what results come up. Be warned though, this code is ridiculously slow. Slow as in it takes up to 8 hours or so to run all three
trials on my machine, sometimes. Works, though, and gets accuracy of 98%+.
