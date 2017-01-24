import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys

FLAGS = None

def train():
    # load data set into train, testm and validate
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True, fake_data=FLAGS.fake_data)

    # start interactive session to inerweave operations and to use optimized code outside of python
    sess = tf.InteractiveSession()

    # create input and output nodes for single layer network (images are 28 x 28 and are of 10 classes)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None,784]) # 28 x 28  = 784 vector size for image
        y_ = tf.placeholder(tf.float32, shape=[None,10])

    # create weight variable
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # output summary variable for display in tensorboard
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean',mean)
            with tf.name_scope('std_dev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('stddev', tf.reduce_min(var))
            tf.summary.histogram('histogram',var)

    # custom definition of hidden layer with relu activation function
    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        # create tensorboard variables to view
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim,output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor,weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    hidden_1 = nn_layer(x, 784, 500, 'layer_1')

    # apply dopout to hidden layer connections
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden_1, keep_prob)

    # apply fully connected layer with hidden layer with identity activation function
    y = nn_layer(dropped, 500, 10, 'layer_2', act= tf.identity)

    # apply loss function to fully connected layer
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(y,y_)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

    # apply adam optimizer on loss function in training step
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    # get accuracy measures when training
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # write all summary variables to log
    merged = tf.summary.merge_all()
    train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    # write summaries of batches during training
    def feed_dict(train):
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data = FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):
        if i%10 == 0:
            # print and write summaries of step
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print 'Accuracy at step: ' + str(i) + ' is: ' + str(acc)
        else:
            summary,_ =  sess.run([merged, accuracy], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
