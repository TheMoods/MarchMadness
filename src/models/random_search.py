import random
import numpy as np
import pandas as pd


class RandomSearch(object):
    def __init__(self, sample_space, model, X, Y, X_eval, Y_eval, iterations):
        if isinstance(sample_space, str):
            if sample_space == 'nn':
                self.sample_params = self.sample_params_nn
            else:
                print('Please provide an appropriate sample space')
        else:
            self.sample_params = sample_space

        self.model = model
        self.X = X
        self.Y = Y
        self.X_eval = X_eval
        self.Y_eval = Y_eval
        self.iterations = iterations
    
    def search(self):
        performance_list = list() 
        for _ in range(self.iterations):
            params = self.sample_params()
            fit = self.model(**params)
            train_loss, eval_loss = fit.train(self.X, self.Y, self.X_eval, 
                                              self.Y_eval)
            params['train_loss'] = train_loss  # make this general
            params['eval_loss'] = eval_loss
            performance_list.append(params)
            print(params)

        return pd.DataFrame.from_dict(performance_list)

    def sample_params_nn(self):
        params = {
            'input_dim': self.X.shape[1],
            'eta': random.choice(np.linspace(0.01, 0.05, 1000)),
            'dropout': random.choice(np.linspace(0, 1, 1000)),
            'hidden_units': [random.choice(range(4, 128)) 
                             for _ in range(random.choice(range(1, 4)))],
            'batch_size': random.choice(range(4, 128)),
            'num_epochs': random.choice(range(100, 500)),
        }
        return params

