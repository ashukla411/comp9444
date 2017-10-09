import tensorflow as tf

def input_placeholder():
	return tf.placeholder(dtype=tf.float32, shape=[None, 784],
                          name="image_input")

def target_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 10],
                          name="image_target_onehot")

def onelayer(X, Y, layersize=10):
    w = tf.Variable(tf.zeros([784, layersize]))
    b = tf.Variable(tf.zeros([layersize]))
    logits = tf.matmul(X, w) + b
    preds = tf.nn.softmax(logits)
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    batch_loss = tf.reduce_mean(batch_xentropy)
    return w, b, logits, preds, batch_xentropy, batch_loss

def twolayer(X, Y, hiddensize=30, outputsize=10):
	w1 = tf.Variable(tf.random_normal([784,hiddensize], stddev=0.15))
	b1 = tf.Variable(tf.zeros([hiddensize]))
	h1 = tf.nn.relu(tf.matmul(X,w1)+b1)
	w2 = tf.Variable(tf.random_normal([hiddensize,outputsize], stddev=0.15))
	b2 = tf.Variable(tf.zeros([outputsize]))
	logits = tf.matmul(h1,w2) + b2
	preds = tf.nn.softmax(logits)
	batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
	batch_loss = tf.reduce_mean(batch_xentropy)
	return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss

def convnet(X, Y, convlayer_sizes=[10, 10],filter_shape=[3, 3], outputsize=10, padding="same"):
	conv1 = tf.layers.conv2d(X, convlayer_sizes[0], filter_shape, padding=padding, activation=tf.nn.relu)
	conv2 = tf.layers.conv2d(conv1, convlayer_sizes[1], filter_shape, padding=padding, activation=tf.nn.relu)
	fullyconnected = tf.reshape(conv2, [-1,784 * 10])
	w = tf.Variable(tf.zeros([7840, outputsize]))
	b = tf.Variable(tf.zeros([outputsize]))
	logits = tf.matmul(fullyconnected, w) + b
	preds = tf.nn.softmax(logits)
	batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
	batch_loss = tf.reduce_mean(batch_xentropy)
	return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss

def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):
    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    return train_result, loss, summary
