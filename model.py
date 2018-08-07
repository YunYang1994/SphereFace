# -*- coding: utf-8 -*-
# import math
import tensorflow as tf

"""
@platform: vim
@author:   YunYang1994
@email:    dreameryangyun@sjtu.edu.cn
"""

class Model(object):
    """
    Created on sunday July 15  15:25:45 2018
        -->  -->
           ==
    """
    def __init__(self, images, labels, embedding_dim,loss_type = 0):
        self.images = images
        self.labels = labels
        self.embedding_dim = embedding_dim
        self.loss_type = loss_type
        self.embeddings = self.__get_embeddings()
        self.pred_prob, self.loss = self.__get_loss()
        self.predictions = self.__get_pred()
        self.accuracy = self.__get_accuracy()


    def __get_embeddings(self):
        return self.network(inputs=self.images, embedding_dim=self.embedding_dim)

    def __get_loss(self):
        if self.loss_type == 0: return self.Original_Softmax_Loss(self.embeddings, self.labels)
        if self.loss_type == 1: return self.Modified_Softmax_Loss(self.embeddings, self.labels)
        if self.loss_type == 2: return self.Angular_Softmax_Loss( self.embeddings, self.labels)

    def __get_pred(self):
        return tf.argmax(self.pred_prob, axis=1)

    def __get_accuracy(self):
        correct_predictions = tf.equal(self.predictions, self.labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
        return accuracy

    @staticmethod
    def network(inputs, embedding_dim=2):

        def prelu(inputs, name=''):
            alpha = tf.get_variable(name, shape=inputs.get_shape(),
                                    initializer=tf.constant_initializer(0.0), dtype=inputs.dtype)
            return tf.maximum(alpha*inputs, inputs)

        def conv(inputs, filters, kernel_size, strides, w_init, padding='same', suffix='', scope=None):
            conv_name = 'conv'+suffix
            relu_name = 'relu'+suffix

            with tf.name_scope(name=scope):
                if w_init == 'xavier':   w_init = tf.contrib.layers.xavier_initializer(uniform=True)
                if w_init == 'gaussian': w_init = tf.contrib.layers.xavier_initializer(uniform=False)
                input_shape = inputs.get_shape().as_list()
                net = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding=padding,
                                    kernel_initializer=w_init, name=conv_name)
                output_shape=net.get_shape().as_list()
                print("=================================================================================")
                print("layer:%8s    input shape:%8s   output shape:%8s" %(conv_name, str(input_shape), str(output_shape)))
                print("---------------------------------------------------------------------------------")
                net = prelu(net, name=relu_name)
                return net

        def resnet_block(net, blocks, suffix=''):
            n = len(blocks)
            for i in range(n):
                if n == 2 and i == 0: identity = net
                net = conv(inputs=net,
                           filters=blocks[i]['filters'],
                           kernel_size=blocks[i]['kernel_size'],
                           strides=blocks[i]['strides'],
                           w_init=blocks[i]['w_init'],
                           padding=blocks[i]['padding'],
                           suffix=suffix+'_'+blocks[i]['suffix'],
                           scope='conv'+suffix+'_'+blocks[i]['suffix'])

                if n == 3 and i == 0: identity = net
            return identity + net

        res1_3=[
            {'filters':64, 'kernel_size':3, 'strides':2, 'w_init':'xavier',   'padding':'same', 'suffix':'1'},
            {'filters':64, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'2'},
            {'filters':64, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'3'},
        ]

        res2_3=[
            {'filters':128, 'kernel_size':3, 'strides':2, 'w_init':'xavier',   'padding':'same', 'suffix':'1'},
            {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'2'},
            {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'3'},
        ]

        res2_5=[
            {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'4'},
            {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'5'},
        ]

        res3_3=[
            {'filters':256, 'kernel_size':3, 'strides':2, 'w_init':'xavier',   'padding':'same', 'suffix':'1'},
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'2'},
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'3'},
        ]

        res3_5=[
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'4'},
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'5'},
        ]

        res3_7=[
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'6'},
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'7'},
        ]

        res3_9=[
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'8'},
            {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'9'},
        ]

        res4_3=[
            {'filters':512, 'kernel_size':3, 'strides':2, 'w_init':'xavier',   'padding':'same', 'suffix':'1'},
            {'filters':512, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'2'},
            {'filters':512, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'3'},
        ]

        net = inputs
        for suffix, blocks in zip(('1','2','2','3','3','3','3','4'),
                                  (res1_3,res2_3,res2_5,res3_3,res3_5,res3_7,res3_9,res4_3)):
            net = resnet_block(net, blocks, suffix=suffix)

        net = tf.layers.flatten(net)
        embeddings = tf.layers.dense(net, units=embedding_dim, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return embeddings


    # @staticmethod
    # def network(inputs, embedding_dim=2, weight_decay=0.0):
        # """
        # This is a simple convolutional neural network to extract features from images
        # @inputs: images (batch_size, 28, 28, 1); embedding_dim , the num of dimension of embeddings
        # @return: embeddings (batch_size, embedding_dim)
        # """
        # w_init = tf.contrib.layers.xavier_initializer(uniform=False)
        # with tf.name_scope('conv1.x'):
            # net = tf.layers.conv2d(inputs, 32, [5,5], strides=1, padding='same', kernel_initializer=w_init)
            # net = tf.layers.conv2d(net,    32, [5,5], strides=2, padding='same', kernel_initializer=w_init)
            # net = tf.nn.relu(net)
        # with tf.name_scope('conv2.x'):
            # net = tf.layers.conv2d(net,    64, [5,5], strides=1, padding='same', kernel_initializer=w_init)
            # net = tf.layers.conv2d(net,    64, [5,5], strides=2, padding='same', kernel_initializer=w_init)
            # net = tf.nn.relu(net)
        # with tf.name_scope('conv3.x'):
            # net = tf.layers.conv2d(net,   128, [5,5], strides=1, padding='same',kernel_initializer=w_init)
            # net = tf.layers.conv2d(net,   128, [5,5], strides=2, padding='same',kernel_initializer=w_init)
            # net = tf.nn.relu(net)
        # net = tf.layers.flatten(net)
        # embeddings = tf.layers.dense(net, units=embedding_dim, kernel_initializer=w_init)
        # return embeddings

    @staticmethod
    def Original_Softmax_Loss(embeddings, labels):
        """
        This is the orginal softmax loss, nothing to say
        """
        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights',
                                      shape=[embeddings.get_shape().as_list()[-1], 10],
                                      initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(embeddings, weights)
            pred_prob = tf.nn.softmax(logits=logits) # output probability
            # define cross entropy
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            return pred_prob, loss

    @staticmethod
    def Modified_Softmax_Loss(embeddings, labels):
        """
        This kind of loss is slightly different from the orginal softmax loss. the main difference
        lies in that the L2-norm of the weights are constrained  to 1, then the
        decision boundary will only depends on the angle between weights and embeddings.
        """
        # # normalize embeddings
        # embeddings_norm = tf.norm(embeddings, axis=1, keepdims=True)
        # embeddings = tf.div(embeddings, embeddings_norm, name="normalize_embedding")
        """
        the abovel commented-out code would lead loss to divergence, maybe you can try it.
        """
        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights',
                                      shape=[embeddings.get_shape().as_list()[-1], 10],
                                      initializer=tf.contrib.layers.xavier_initializer())
            # normalize weights
            weights_norm = tf.norm(weights, axis=0, keepdims=True)
            weights = tf.div(weights, weights_norm, name="normalize_weights")
            logits = tf.matmul(embeddings, weights)
            pred_prob = tf.nn.softmax(logits=logits)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            return pred_prob, loss

    @staticmethod
    def Angular_Softmax_Loss(embeddings, labels, margin=4):
        """
        Note:(about the value of margin)
        as for binary-class case, the minimal value of margin is 2+sqrt(3)
        as for multi-class  case, the minimal value of margin is 3

        the value of margin proposed by the author of paper is 4.
        here the margin value is 4.
        """
        l = 0.
        embeddings_norm = tf.norm(embeddings, axis=1)

        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights',
                                      shape=[embeddings.get_shape().as_list()[-1], 10],
                                      initializer=tf.contrib.layers.xavier_initializer())
            weights = tf.nn.l2_normalize(weights, axis=0)
            # cacualting the cos value of angles between embeddings and weights
            orgina_logits = tf.matmul(embeddings, weights)
            N = embeddings.get_shape()[0] # get batch_size
            single_sample_label_index = tf.stack([tf.constant(list(range(N)), tf.int64), labels], axis=1)
            # N = 128, labels = [1,0,...,9]
            # single_sample_label_index:
            # [ [0,1],
            #   [1,0],
            #   ....
            #   [128,9]]
            selected_logits = tf.gather_nd(orgina_logits, single_sample_label_index)
            cos_theta = tf.div(selected_logits, embeddings_norm)
            cos_theta_power = tf.square(cos_theta)
            cos_theta_biq = tf.pow(cos_theta, 4)
            sign0 = tf.sign(cos_theta)
            sign3 = tf.multiply(tf.sign(2*cos_theta_power-1), sign0)
            sign4 = 2*sign0 + sign3 -3
            result=sign3*(8*cos_theta_biq-8*cos_theta_power+1) + sign4

            margin_logits = tf.multiply(result, embeddings_norm)
            f = 1.0/(1.0+l)
            ff = 1.0 - f
            combined_logits = tf.add(orgina_logits, tf.scatter_nd(single_sample_label_index,
                                                           tf.subtract(margin_logits, selected_logits),
                                                           orgina_logits.get_shape()))
            updated_logits = ff*orgina_logits + f*combined_logits
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=updated_logits))
            pred_prob = tf.nn.softmax(logits=updated_logits)
            return pred_prob, loss



