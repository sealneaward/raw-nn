import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load data set into train, testm and validate
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# start interactive session to inerweave operations and to use optimized code outside of python
sess = tf.InteractiveSession()

# create input and output nodes for single layer network (images are 28 x 28 and are of 10 classes)
x = tf.placeholder(tf.float32, shape=[None,784]) # 28 x 28  = 784 vector size for image
y_ = tf.placeholder(tf.float32, shape=[None,10])

# define weights and biases for hidden layer
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# initalize variables
sess.run(tf.global_variables_initializer())

# define class prediction function at hidden layer
y = tf.matmul(x,W) + b

# define log loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))

# use gradient descent to minimize log loss in predictions
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# batch size is 100
for i in range(1000):
    batch = mnist.train.next_batch(100)
    # topimize weights for batch data
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # get accuracy by comparing the predictions and the ground truth
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})
