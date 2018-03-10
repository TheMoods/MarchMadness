import matplotlib.pyplot as plt
from copy import deepcopy
from functools import partial
from numpy import ceil
from pandas import DataFrame
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from src.utils import load_data_template
from src.features import GameFeatures, GameDetailedFeatures,\
    SeedFeatures, EnsembleFeatures
from xgboost import XGBClassifier


class GameModel(object):
    Estimator = XGBClassifier
    index_dtypes = {
        'team_a': str,
        'team_b': str,
        'Season': int,
        'DayNum': int
    }
    drop_for_features = ['Season', 'a_win', 'in_target',
                         'DayNum', 'team_a', 'team_b']
    target_cols = ['team_a', 'team_b', 'Season', 'DayNum', 'a_win', 'game_set']

    def __init__(self, pred_data_temp):
        self.pred_data_temp = pred_data_temp
        self.load_data()

    def feature_pipeline(self, data):
        game_feat = GameFeatures(default_lags=3)
        data = game_feat\
            .per_team_wrapper(data,
                              game_feat.last_games_won_in_season,
                              combine='subtract',
                              fillna=0)
        data = game_feat\
            .per_team_wrapper(data,
                              game_feat.last_games_won_in_tourney,
                              combine='subtract',
                              fillna=0)
        data = game_feat\
            .per_team_wrapper(data,
                              game_feat.last_games_won_against_opponent,
                              per_game=True,
                              combine='subtract',
                              fillna=0)
        data = game_feat\
            .per_team_wrapper(data,
                              game_feat.games_won_in_tourney_against_opponent,
                              per_game=True,
                              combine='subtract',
                              fillna=0)
        game_detail_feat = GameDetailedFeatures(default_lags=3)
        data = game_detail_feat\
            .per_team_wrapper(data,
                              game_detail_feat.detail_features_by_game,
                              per_day=True,
                              combine='subtract')
        seed_feat = SeedFeatures(default_lags=0)
        data = seed_feat.per_team_wrapper(data,
                                          seed_feat.team_seeds,
                                          combine='subtract',
                                          fillna=0)
        return data

    def load_fit_features(self):
        data = self.get_fit_data_temp()
        data = self.feature_pipeline(data)
        self.fit_features = data\
            .drop(self.drop_for_features, axis=1)
        self.fit_targets = deepcopy(data[self.target_cols])
        print('Fit Features Loaded: {}'
              .format(self.fit_features.shape))

    def load_pred_features(self):
        data = self.feature_pipeline(self.pred_data_temp)
        self.pred_features = data\
            .drop(self.drop_for_features, axis=1)
        self.pred_targets = deepcopy(data[self.target_cols])
        self.pred_targets.loc[:, 'a_win'] = 'not_predicted'
        print('Pred Features Loaded: {}'
              .format(self.pred_features.shape))

    def load_data(self):
        self.load_fit_features()
        self.load_pred_features()

    def fit(self):
        self.estimator = self.Estimator()
        self.estimator.fit(self.fit_features, self.fit_targets['a_win'])

    def predict(self):
        pred = self.estimator.predict_proba(self.pred_features)
        self.pred_targets['b_win'] = pred[:, 0]
        self.pred_targets['a_win'] = pred[:, 1]

    def cross_validate(self, n=1, n_splits=3, show_hist=False,
                       estimator_params={}):
        X = self.fit_features
        y = self.fit_targets
        cv_results = {
                'log_loss': []
        }
        # COMPETITION SPECIFIC CODE START
        cv_results['ncaa_log_loss'] = []
        # COMPETITION SPECIFIC CODE END
        for i in range(ceil(n / n_splits).astype(int)):
            kf = KFold(n_splits=n_splits, shuffle=True)
            for tr_i, t_i in kf.split(X):
                X_tr, y_tr = X.iloc[tr_i], y.iloc[tr_i].a_win
                X_t, y_t = X.iloc[t_i], y.iloc[t_i].a_win.astype(int)
                estimator = self.Estimator(**estimator_params)
                estimator.fit(X_tr.values, y_tr)
                preds = estimator.predict_proba(X_t.values)
                log_loss_metric = log_loss(y_t.values, preds,
                                           labels=(0, 1))
                cv_results['log_loss'].append(log_loss_metric)
                # COMPETITION SPECIFIC CODE START
                ncaa_true = y.iloc[t_i][y.iloc[t_i].game_set == 0]
                ncaa_pred = preds[y.iloc[t_i].reset_index().game_set == 0]
                ncaa_log_loss = log_loss(ncaa_true.a_win.astype(int),
                                         ncaa_pred, labels=(0, 1))
                cv_results['ncaa_log_loss'].append(ncaa_log_loss)
                # COMPETITION SPECIFIC CODE END

        cv_results = DataFrame(cv_results)
        if show_hist:
            for c in cv_results.columns:
                plt.figure(figsize=(20, 4))
                # plt.xlim([0, .25])
                plt.hist(self.cv_results[c].values,
                         bins=ceil(n / 10).astype(int))
                plt.show()

        self.cv_results = cv_results
        return self.cv_results.agg(['mean', 'std', 'min', 'max'])

    def get_fit_data_temp(self):
        fit_temp = load_data_template(season=False)
        fit_temp.dropna(subset=['a_win'], inplace=True)
        fit_temp = fit_temp.astype(self.index_dtypes)
        return fit_temp
