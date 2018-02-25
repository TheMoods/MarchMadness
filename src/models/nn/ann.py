import tensorflow as tf
from sklearn.model_selection import KFold


class ANN(object):
    
    def __init__(self, input_dim, batch_size, hidden_units, eta, num_epochs):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.learning_rate = eta
        self.num_epochs = num_epochs
        self.num_classes = 2
        self.num_layers = len(hidden_units)
        self.create_network()

    def train(self, X, Y):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(self.num_epochs):
                avg_loss = .0
                num_batches = int(X.shape[0]/self.batch_size)
                for i in num_batches:
                    batch_ix = range(i*self.batch_size, (i+1)*self.batch_size)
                    batch_X, batch_Y = X[batch_ix], Y[batch_ix]
                    nodes = [self.optimize, self.loss]
                    feed_dict = {X: batch_X, Y: batch_Y}  
                    _, loss = sess.run(nodes, feed_dict=feed_dict)
                    avg_loss += loss/len(batch_X)
                    # Compute average loss
                # Display logs per epoch step
                if epoch % 5 == 0:
                    print("Epoch:", '%04d' % (epoch+1), 
                          "cost={:.9f}".format(loss))
            '''
            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})):
            '''

    def create_network(self):
        self.X = tf.placeholder(tf.float32, [None, self.input_dim])
        self.Y = tf.placeholder(tf.int32, [None])

        self.weights = self.define_weights()
        self.biases = self.define_biases()
        self.layers = self.multilayer_perceptron()
        self.output = tf.matmul(
                self.layers['l_'+str(self.num_layers-1)], self.weights['w_out'])+\
                self.biases['b_out']
        self.loss = tf.losses.log_loss(self.Y, self.output)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def define_weights(self):
        weights = {
            'w_0': tf.get_variable('w_0', [self.input_dim, self.hidden_units[0]])
        }
        for i, u in enumerate(self.hidden_units[1:]):
            dim1 = weights['w_'+str(i)].shape[1] 
            dim2 = u
            weights['w_'+str(i+1)] = tf.get_variable('w_'+str(i+1), [dim1, dim2]) 

        weights['w_out'] = tf.get_variable('w_out', [dim2, self.num_classes])
        return weights

    def define_biases(self):
        biases = {
            'b_0': tf.get_variable('b_0', [self.input_dim, self.hidden_units[0]])
        }
        for i, u in enumerate(self.hidden_units[1:]):
            dim1 = 1
            dim2 = u
            biases['b_'+str(i+1)] = tf.get_variable('b_'+str(i+1), [dim1, dim2]) 

        biases['b_out'] = tf.get_variable('b_out', [dim1, self.num_classes])
        return biases

    def multilayer_perceptron(self):
        layers = {
            'l_0': tf.matmul(self.X, self.weights['w_0']) + self.biases['b_0']
        }
        for i in range(self.num_layers-1):
            layers['l_'+str(i+1)] = tf.matmul(layers['l_'+str(i)],
                                            self.weights['w_'+str(i+1)])\
                                            + self.biases['b_0']
        
        return layers

if __name__ == '__main__':

    input_dim = input_dim
    batch_size = batch_size
    hidden_units = hidden_units
    learning_rate = eta
    num_epochs = num_epochs
    num_classes = 2
    num_layers = len(hidden_units)
    ann = ANN(10, [12, 12, 12])
