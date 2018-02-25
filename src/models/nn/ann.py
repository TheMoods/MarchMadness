import tensorflow as tf
from sklearn.model_selection import KFold


class ANN(object):
    
    def __init__(self, input_dim, batch_size, hidden_units,
                 dropout, eta, num_epochs):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = eta
        self.num_epochs = num_epochs
        self.num_classes = 2
        self.num_layers = len(hidden_units)
        self.create_network()
        self.sess = None

    def train(self, X, Y, X_eval=None, Y_eval=None, verbose=False):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            self.sess = sess
            self.sess.run(init)
            # Training cycle
            for epoch in range(self.num_epochs):
                avg_loss = .0
                num_batches = int(X.shape[0]/self.batch_size)
                for i in range(num_batches):
                    s_ix, f_ix = (i*self.batch_size), (i+1)*self.batch_size
                    batch_X, batch_Y = X[s_ix:f_ix], Y[s_ix:f_ix]
                    nodes = [self.probs, self.train_op, self.loss]
                    feed_dict = {self.X: batch_X, self.Y: batch_Y,
                                 self.keep_prob: 1 - self.dropout}  
                    probs, _, loss = sess.run(nodes, feed_dict=feed_dict)
                    avg_loss += loss/len(batch_X)

                    # Compute average loss
                # Display logs per epoch step
                if epoch % 5 == 0:
                    if X_eval is not None:
                        eval_loss = self.validate(X_eval, Y_eval)
                        if verbose:
                            print("Epoch:", '%03d' % (epoch),
                                  "train_loss={:.3f}".format(avg_loss),
                                  "eval_loss={:.3f}".format(eval_loss))
                    else:
                        if verbose:
                            print("Epoch:", '%03d' % (epoch),
                                  "train_loss={:.3f}".format(avg_loss))

            return avg_loss, eval_loss

            '''
            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})):
            '''
    
    def validate(self, X, Y):
        nodes = [self.loss]
        feed_dict = {self.X: X, self.Y: Y}
        eval_loss = self.sess.run(nodes, feed_dict=feed_dict)
        return eval_loss[0]

    def create_network(self):
        tf.reset_default_graph()
        self.X = tf.placeholder(tf.float32, [None, self.input_dim])
        self.Y = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        self.weights = self.define_weights()
        self.biases = self.define_biases()
        self.layers = self.multilayer_perceptron()
        self.output = tf.matmul(
                self.layers['l_'+str(self.num_layers-1)], self.weights['w_out'])+\
                self.biases['b_out']
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


