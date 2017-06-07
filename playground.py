# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the playground network"""
import math
import input
import tensorflow as tf
import numpy as np

NUM_CLASSES = 2
LEARNING_RATE = 0.001
NUM_CELL = 6
BATCH_SIZE = 500
NUM_TRAIN_STEP = 3000
GAUSS = "Gauss"
CIRCLE = "Circle"
XOR = "Xor"
SPIRAL = "Spiral"

""" Returns the next batch of data points """
def next_batch(data):
    if data == CIRCLE:
        points = input.classifyCircleData(BATCH_SIZE)
    if data == GAUSS:
        points = input.classifyGaussData(BATCH_SIZE)
    if data == XOR:
        points = input.classifyXORData(BATCH_SIZE)
    if data == SPIRAL:
        points = input.classifySpiralData(BATCH_SIZE)
        
    xy = points[:,[0,1]]
    labels = points[:,2]
    return xy, labels
    

    
def inference(input, hidden1_units, hidden2_units, hidden3_units):
    """Build the playground model up to where it may be used for inference.
    Args:
    input: Input points placeholder, from input.
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
    hidden3_units: Size of the third hidden layer.
    Returns:
    softmax_linear: Output tensor with the computed logits.
    """
    
    tf.set_random_seed(2)
    
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([NUM_CLASSES, hidden1_units],stddev=1.0 / math.sqrt(NUM_CLASSES)), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.tanh(tf.matmul(input, weights) + biases)
        tf.summary.histogram('hidden1', hidden1)
        tf.summary.tensor_summary('hidden1-s', hidden1)
        
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.tanh(tf.matmul(hidden1, weights) + biases)
        tf.summary.histogram('hidden2', hidden2)
        tf.summary.tensor_summary('hidden2-s', hidden2)
        
    # Hidden 3
    with tf.name_scope('hidden3'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, hidden3_units],stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
        biases = tf.Variable(tf.zeros([hidden3_units]), name='biases')
        hidden3 = tf.nn.tanh(tf.matmul(hidden2, weights) + biases)
        tf.summary.histogram('hidden3', hidden3)
        tf.summary.tensor_summary('hidden3-s', hidden3)
        
        
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden3_units, NUM_CLASSES], stddev=1.0 / math.sqrt(NUM_CLASSES)), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden3, weights) + biases
        tf.summary.histogram('softmax', logits)
        tf.summary.tensor_summary('softmax-s', logits)
        
    return logits


def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    Returns:
    train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size] 
    Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """

    correct = tf.nn.in_top_k(logits, labels, 1)
                # Return the number of true entries.
    return tf.reduce_mean(tf.cast(correct, tf.float32))
    
def train():
    """Train playground for a number of steps."""
    experiment = SPIRAL
    points, labels = next_batch(data = experiment)
    
    x = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    y_ = tf.placeholder(tf.int64, [None])
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = inference(x, 8,8,5)

    # Calculate loss.
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=logits, name='xentropy')
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)
    
    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())
    
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_step = optimizer.minimize(cross_entropy)
    
    accuracy = evaluation(logits, labels = y_)
    tf.summary.scalar('accuracy', accuracy)

    # Start running operations on the Graph.
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/Users/jackyhan/Desktop/' + '/train',
                                      sess.graph)
    ckpt = tf.train.get_checkpoint_state('/Users/jackyhan/Desktop/' + '/train/')
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = 200
      
    #  
    for step in range(NUM_TRAIN_STEP):
        batch_xs, batch_ys = next_batch(data=experiment)
        
        _, loss_value, summary = sess.run([train_step, loss, merged], feed_dict={x: batch_xs, y_: batch_ys})
        
                
        ac = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        
        format_str = ('step %d, loss = %.2f, accuracy = %.2f')
        print (format_str % (step, loss_value, ac))
        train_writer.add_summary(summary, step)
        
        if step % 200 == 0 or (step + 1) == NUM_TRAIN_STEP:
            saver.save(sess, '/Users/jackyhan/Desktop/' + '/train/' + 'model.ckpt', global_step=step)


def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
