import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

np.random.seed(0123)

##########################################
# Raw Neural Network using numpy
#
# Three layer neural network that predicts
# output of XOR gate
##########################################


# sigmoid acivation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# rectified linear unit activation function
def relu(x):
    np.maximum(0,x)

# get derivative of matrix
def derivative(x):
    return x * (1-x)

#######################################
# Load data and initialize weights
#######################################

X = np.array(
    [
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ]
)

Y = np.array(
    [
        [0],
        [1],
        [1],
        [0]
    ]
)

# initialize weights
w0 = np.random.random((2,4))
w1 = np.random.random((4,1))

##########################################
# Train network by updating weights
##########################################
plot_x = []
plot_y = []

for i in range(0,100000):

    # feed X matrix through three layers with two weighted layers
    l0 = X
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    # approximate loss on l2
    l2_loss = Y - l2
    plot_x.append(i)
    plot_y.append(np.mean(np.abs(l2_loss)))

    # get derivative of loss on third hidden layer with respect to second hidden layer
    d_l2 = l2_loss * derivative(l2)

    # get loss approximation of first hidden layer with respect to derivative
    # of l2 loss transposed with the weights of second hidden layer
    l1_loss = np.dot(d_l2,w1.T)

    # get derivative of loss on second hidden layer with respect to first hidden layer
    d_l1 = l1_loss * derivative(l1)

    # using 10 outputs, print mean loss
    if (i%10000) == 0:
        print 'Loss: ' + str(np.mean(np.abs(l2_loss)))

    # update weights of first two
    w1 += np.dot(l1.T,d_l2)
    w0 += np.dot(l0.T,d_l1)

plt.plot(plot_x,plot_y)
plt.show()
