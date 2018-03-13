import numpy as np
import tensorflow as tf


class Network(object):
    
    def __init__(self, input_dim, hidden_units,
                 dropout, eta):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = eta
        self.num_classes = 2
        self.num_layers = len(hidden_units)
        self.create_network()
        self.sess = None

    def create_network(self):
        tf.reset_default_graph()
        self.X = tf.placeholder(tf.float32, [None, self.input_dim])
        self.Y = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        self.weights = self.define_weights()
        self.biases = self.define_biases()
        self.layers = self.multilayer_perceptron()
        self.output = tf.nn.dropout(tf.matmul(
                self.layers['l_'+str(self.num_layers-1)], self.weights['w_out'])+\
                self.biases['b_out'], self.keep_prob)
        self.probs = tf.sigmoid(self.output[:, 1])
        self.loss = tf.losses.log_loss(self.Y, self.probs)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def define_weights(self):
        weights = {
            'w_0': tf.get_variable('w_0', [self.input_dim, self.hidden_units[0]],
                initializer=tf.contrib.layers.xavier_initializer())
        }
        if self.num_layers > 1:
            for i, u in enumerate(self.hidden_units[1:]):
                dim1 = weights['w_'+str(i)].shape[1] 
                dim2 = u
                weights['w_'+str(i+1)] = tf.get_variable('w_'+str(i+1), [dim1, dim2],
                        initializer=tf.contrib.layers.xavier_initializer()) 

            weights['w_out'] = tf.get_variable('w_out', [dim2, self.num_classes])
        else:
            weights['w_out'] = tf.get_variable('w_out',
                    [self.hidden_units[0], self.num_classes])

        return weights

    def define_biases(self):
        biases = {
            'b_0': tf.get_variable('b_0', [1, self.hidden_units[0]],
                initializer=tf.contrib.layers.xavier_initializer())
        }
        if self.num_layers > 1:
            for i, u in enumerate(self.hidden_units[1:]):
                dim1 = 1
                dim2 = u
                biases['b_'+str(i+1)] = tf.get_variable('b_'+str(i+1), [dim1, dim2],
                    initializer=tf.contrib.layers.xavier_initializer()) 

            biases['b_out'] = tf.get_variable('b_out',
                    [1, self.num_classes])

        else:
            biases['b_out'] = tf.get_variable('b_out',
                    [1, self.num_classes])
        return biases

    def multilayer_perceptron(self):
        layers = {
            'l_0': tf.matmul(self.X, self.weights['w_0']) + self.biases['b_0']
        }
        for i in range(self.num_layers-1):
            layers['l_'+str(i+1)] = tf.nn.dropout(
                    tf.nn.relu(tf.matmul(layers['l_'+str(i)],
                               self.weights['w_'+str(i+1)])\
                               + self.biases['b_'+str(i+1)]),
                    self.keep_prob)
        
        return layers


