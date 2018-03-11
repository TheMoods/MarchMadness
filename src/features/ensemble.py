import numpy as np
from pandas import read_csv, merge
from src.features.feature import Feature


class EnsembleFeatures(Feature):
    pred_index_cols = ['team_a', 'team_b',
                       'Season', 'DayNum']

    def __init__(self):
        super().__init__()

    def load_preds(self, path, name):
        pred_value_col = 'Pred'
        preds = read_csv(path, dtype={
            'team_a': str,
            'team_b': str,
            'Season': int,
            'DayNum': int
        })[self.pred_index_cols + [pred_value_col]]
        preds.set_index(self.pred_index_cols, inplace=True)
        preds.rename(columns={pred_value_col: name}, inplace=True)
        return preds

    def model_out_preds(self, df, path, name='model_out_preds'):
        df = merge(df, self.load_preds(path, name),
                   left_on=self.pred_index_cols,
                   right_index=True)
        return df
