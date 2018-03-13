import time
import numpy as np
import tensorflow as tf
from src.models.nn.network import Network


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
        self.network = Network(input_dim, hidden_units, dropout, eta)
        self.sess = None

    def train(self, X, Y, X_eval=None, Y_eval=None, verbose=False):
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)
        # Training cycle
        for epoch in range(self.num_epochs):
            stime = time.time()
            avg_loss = list()
            num_batches = int(X.shape[0]/self.batch_size)
            for i in range(num_batches):
                s_ix, f_ix = (i*self.batch_size), (i+1)*self.batch_size
                batch_X, batch_Y = X[s_ix:f_ix], Y[s_ix:f_ix]
                nodes = [
                    self.network.probs, 
                    self.network.train_op,
                    self.network.loss
                ]
                feed_dict = {self.network.X: batch_X, self.network.Y: batch_Y,
                             self.network.keep_prob: 1 - self.dropout}  
                probs, _, loss = self.sess.run(nodes, feed_dict=feed_dict)
                avg_loss.append(loss)
                epoch_time = (time.time() - stime)
                # Compute average loss
            # Display logs per epoch step
            if epoch % 5 == 0:
                if X_eval is not None:
                    print(probs[:5])
                    eval_loss, probs_test = self.validate(X_eval, Y_eval, 
                                                          1-self.dropout)
                    print(probs_test[:5])
                    if verbose:
                        print("Epoch:", '%03d' % (epoch),
                              "train_loss={:.3f}".format(np.mean(avg_loss)),
                              "eval_loss={:.3f}".format(eval_loss),
                              "--- time per epoch={:.2f} seconds"\
                                      .format(epoch_time))
                else:
                    if verbose:
                        print("Epoch:", '%03d' % (epoch),
                              "train_loss={:.3f}".format(avg_loss))
        self.saver.save(self.sess, "src/saved_models/model.ckpt")
        return avg_loss, eval_loss

    def validate(self, X, Y, k_p=1):
        nodes = [self.network.loss, self.network.probs]
        feed_dict = {
            self.network.X: X,
            self.network.Y: Y,
            self.network.keep_prob: k_p
        }
        eval_loss, probs = self.sess.run(nodes, feed_dict=feed_dict)
        return eval_loss, probs

    def predict(self, X, samples=None):
        self.saver.restore(self.sess, "src/saved_models/model.ckpt")
        if samples is None:
            nodes = [self.network.output]
            feed_dict = {self.network.X: X}
            predictions = self.sess.run(nodes, feed_dict=feed_dict)
        else:
            predictions = list()
            for _ in range(samples):
                nodes = [self.network.probs]
                feed_dict = {
                    self.network.X: X,
                    self.network.keep_prob: 1 - self.network.dropout
                }
                preds = self.sess.run(nodes, feed_dict=feed_dict)[0]
                predictions.append(preds)

            predictions = np.column_stack((predictions))

        return predictions
