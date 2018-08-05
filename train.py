# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import model
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from tensorflow.examples.tutorials.mnist import input_data
from mpl_toolkits.mplot3d import Axes3D

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
epochs = 20
batch_size = 512
train_batchs = 100 # the number of batchs per epoch
test_batchs  = 20
embedding_dim = 2 # 3



def train():
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
    network = model.Model(images, labels, embedding_dim)
    accuracy = network.accuracy
    loss = network.loss
    # define optimizer and learning rate
    decay_lr = tf.train.exponential_decay(lr, global_step, 500, 0.9)
    optimizer = tf.train.AdamOptimizer(decay_lr)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # training process
    for loss_type in range(3):
        network.loss_type = loss_type
        train_acc = []; train_loss = []
        print('training loss type %d' %loss_type)
        for epoch in range(epochs):
            for batch in tqdm(range(train_batchs)):
                batch_images, batch_labels = mnist.train.next_batch(batch_size)
                feed_dict = {images:batch_images, labels:batch_labels}
                _, _, batch_loss, batch_acc = sess.run([train_op, add_step_op, loss, accuracy], feed_dict)
                train_acc.append(batch_acc)
                train_loss.append(batch_loss)
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
        yield train_acc, train_loss, test_acc, embeddings, nlabels


def visualize(embedding, label, test_acc=0., picname=''):

    batch_size, embedding_dim = embedding.shape
    if embedding_dim == 2:
        """
        visualize embedding in 2D
        """
        fig,ax = plt.subplots()
        X, Y = embedding[:,0], embedding[:,1]
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        for x,y,l in zip(X,Y,label):
            c = cm.rainbow(int(255 *l/ 9))
            ax.text(x, y, l, color=c)
        plt.title("test accuracy: %.4f" %test_acc)
        plt.axis('off')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./image/2D_'+picname)

    if embedding_dim == 3:
        """
        visualize embedding in 3D
        """
        fig = plt.figure(); ax = Axes3D(fig)
        X, Y, Z = embedding[:, 0], embedding[:, 1], embedding[:, 2]
        for x, y, z, s in zip(X, Y, Z, label):
            c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
        ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
        plt.title("test accuracy: %.4f" %test_acc)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./image/3D_'+picname)



index = [0,1,2]  # loss type index
result = list(train())

train_acc  = [result[i][0] for i in index]
train_loss = [result[i][1] for i in index]
test_acc   = [result[i][2] for i in index]
embeddings = [result[i][3] for i in index]
labels     = [result[i][4] for i in index]

plt.plot(train_acc[0], label='original softmax (test acc: %.4f)' %test_acc[0])
plt.plot(train_acc[1], label='modified softmax (test acc: %.4f)' %test_acc[1])
plt.plot(train_acc[2], label='angular  softmax (test acc: %.4f)' %test_acc[2])
plt.xlabel("batch")
plt.ylabel("train accuracy")
plt.legend()
plt.savefig("./image/train_acc.jpg")
plt.close()

plt.plot(train_loss[0], label='Original Softmax (test acc: %.4f)' %test_acc[0])
plt.plot(train_loss[1], label='Modified Softmax (test acc: %.4f)' %test_acc[1])
plt.plot(train_loss[2], label='Angular  Softmax (test acc: %.4f)' %test_acc[2])
plt.xlabel("batch")
plt.ylabel("train loss")
plt.legend()
plt.savefig("./image/train_loss.jpg")
plt.close()

visualize(embeddings[0], labels[0], test_acc[0], picname='Original_Softmax_Loss_embeddings.jpg')
visualize(embeddings[1], labels[1], test_acc[1], picname='Modified_Softmax_Loss_embeddings.jpg')
visualize(embeddings[2], labels[2], test_acc[2], picname='Angular_Softmax_Loss_embeddings.jpg')



