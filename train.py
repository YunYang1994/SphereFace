# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import model
# import matplotlib.pyplot as plt
from utils import visualize, create_gif
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

"""
@platform: vim
@author:   YunYang1994
Created on sunday July 15  16:25:45 2018

    -->  -->
       ==
##########################################################################

This is quick re-implementation of asoftmax loss proposed in this paper:
    'SphereFace: Deep Hypersphere Embedding for Face Recognition. '
see https://arxiv.org/pdf/1704.08063.pdf
if you have any questions, please contact with me, I am very happy to
discuss them with you, my email is 'dreameryangyun@sjtu.edu.cn'

#########################################################################
"""

# prepare mnist data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=False, reshape=False)

# define training parameters
lr = 0.001
epochs = 40
batch_size = 256
train_batchs = 40 # the number of batchs per epoch
test_batchs  = 20
embedding_dim = 2 # 3
loss_type = 1



def train(loss_type):
    """
    original_softmax_loss_network = network(0)
    modified_softmax_loss_network = network(1)
    angular_softmax_loss_network  = network(2)
    """
    # define input placeholder for network
    images = tf.placeholder(tf.float32, shape=[batch_size,28,28,1], name='input')
    labels = tf.placeholder(tf.int64, [batch_size,])
    global_step = tf.Variable(0, trainable=False)
    add_step_op = tf.assign_add(global_step, tf.constant(1))
    # about network
    network = model.Model(images, labels, embedding_dim, loss_type)
    accuracy = network.accuracy
    loss = network.loss
    # define optimizer and learning rate
    decay_lr = tf.train.exponential_decay(lr, global_step, 500, 0.9)
    optimizer = tf.train.AdamOptimizer(decay_lr)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # training process
    network.loss_type = loss_type
    print('training loss type %d' %loss_type)
    if loss_type ==0: f = open('./result/oringal_softmax_result.txt', 'w')
    if loss_type ==1: f = open('./result/modified_softmax_result.txt', 'w')
    if loss_type ==2: f = open('./result/angular_softmax_result.txt', 'w')
    # summary = tf.summary.FileWriter("./image/", sess.graph)

    for epoch in range(epochs):
        nlabels = np.zeros((train_batchs*batch_size,), dtype=np.int32)
        embeddings = np.zeros((train_batchs*batch_size, embedding_dim), dtype=np.float32)
        train_acc = 0.
        for batch in tqdm(range(train_batchs)):
            i,j = batch*batch_size, (batch+1)*batch_size
            batch_images, batch_labels = mnist.train.next_batch(batch_size)
            feed_dict = {images:batch_images, labels:batch_labels}
            _, _, batch_loss, batch_acc, embeddings[i:j,:] = sess.run([train_op, add_step_op, loss, accuracy, network.embeddings], feed_dict)
            nlabels[i:j] = batch_labels
            f.write(" ".join(map(str,[batch_acc, batch_loss]))+ "\n")
            # print(batch_acc)
            train_acc += batch_acc
        train_acc /= train_batchs
        print("epoch %2d---------------------------train accuracy:%.4f" %(epoch+1, train_acc))
        visualize(embeddings, nlabels, epoch, train_acc, picname="./image/%d/%d.jpg"%(loss_type, epoch))
    # testing process
    test_acc = 0.
    embeddings = np.zeros((test_batchs*batch_size, embedding_dim), dtype=np.float32)
    nlabels = np.zeros(shape=(test_batchs*batch_size,), dtype=np.int32)
    for batch in range(test_batchs):
        i,j = batch*batch_size, (batch+1)*batch_size
        batch_images, batch_labels = mnist.test.next_batch(batch_size)
        feed_dict = {images:batch_images, labels:batch_labels}
        _, batch_loss, batch_acc, embeddings[i:j,:] = sess.run([train_op, loss, accuracy, network.embeddings], feed_dict)
        nlabels[i:j] = batch_labels
        test_acc += batch_acc
    test_acc /= test_batchs
    print("test accuracy: %.4f" %test_acc)
    return test_acc, embeddings, nlabels



if __name__ == "__main__":

    gif = ['original_softmax_loss.gif', 'modified_softmax_loss.gif', 'angular_softmax_loss.gif']
    path = './image/%d/' %loss_type
    gif_name = './image/%s' %gif[loss_type]
    train(loss_type=loss_type)
    create_gif(gif_name, path)



