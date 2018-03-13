import matplotlib.pyplot as plt
from datetime import datetime as dt
from copy import deepcopy
from functools import partial
from numpy import ceil, nan
from pandas import DataFrame, to_datetime, concat
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from src.utils import load_target_sample, load_data_template
from src.features import *
from xgboost import XGBClassifier


class GameModel(object):
    index_dtypes = {
        'team_a': str,
        'team_b': str,
        'Season': int,
        'DayNum': int
    }
    drop_for_features = ['Season', 'a_win', 'in_target',
                         'DayNum', 'team_a', 'team_b']
    target_cols = ['team_a', 'team_b', 'Season', 'DayNum', 'a_win', 'game_set']

    def __init__(self, pred_data_temp=None, preload=True,
                 Estimator=None, feature_pipeline=None,
                 with_season_games=False):
        for p in ['pred_data_temp', 'Estimator',
                  'feature_pipeline', 'with_season_games']:
            if locals()[p] is not None:
                setattr(self, p, locals()[p])
            elif not hasattr(self, p):
                raise Exception('''
                    Need to set `{}`
                    in sublass or pass as argument
                '''.format(p))
        if preload:
            self.load_data()
        self.cv_history = []

    def feature_pipeline(self, data):
        print('Running Feature Pipeline')
        start = dt.now()

        print('-- Seeds --')
        seed_feat = SeedFeatures()
        data = seed_feat.per_team_wrapper(
            data, seed_feat.team_seeds,
            per_game=False, per_day=False)
        print(data.shape)

        print('Feature Pipeline Clock: {} Seconds'
              .format((dt.now() - start).seconds))
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

    def fit_predict(self, ep={}):
        self.fit(ep=ep)
        return self.predict()

    def fit(self, ep={}):
        self.estimator = self.Estimator(**ep)
        self.estimator.fit(self.fit_features,
                           self.fit_targets['a_win'].astype(int))

    def predict(self):
        pred = self.estimator.predict_proba(self.pred_features)
        self.pred_targets['b_win'] = pred[:, 0]
        self.pred_targets['a_win'] = pred[:, 1]
        return self.pred_targets

    def cross_validate(self, n=1, n_splits=3, show_histogram=False,
                       ep={}):
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
                X_tr, y_tr = X.iloc[tr_i], y.iloc[tr_i].a_win.astype(int)
                X_t, y_t = X.iloc[t_i], y.iloc[t_i].a_win.astype(int)
                estimator = self.Estimator(**ep)
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
        cv_results['timestamp'] = to_datetime(dt.now())
        cv_results.set_index('timestamp', inplace=True)
        for name, value in ep.items():
            cv_results['ep_{}'.format(name)] = value

        self.cv_history.append(cv_results)
        if show_histogram:
            c = 'log_loss'
            plt.figure(figsize=(20, 4))
            plt.hist(cv_results[c].values,
                     bins=ceil(n / 7).astype(int))
            plt.show()

        self.cv_results = cv_results
        return self.cv_results.agg(['mean', 'std', 'min', 'max'])

    def get_cv_history(self):
        return concat(self.cv_history)

    def get_fit_data_temp(self):
        fit_temp = load_data_template(season=self.with_season_games)
        fit_temp.dropna(subset=['a_win'], inplace=True)
        fit_temp = fit_temp.astype(self.index_dtypes)
        return fit_temp


class NCAAModel(GameModel):

    def __init__(self, **kw_args):
        self.load_pred_data_template()
        super().__init__(**kw_args)

    def load_pred_data_template(self):
        temp = load_target_sample()
        temp['Season'] = temp['Season'].astype(int)
        temp['a_win'] = 'not predicted'
        temp['DayNum'] = 366
        temp['game_set'] = 0
        self.pred_data_temp = temp


class NCAAModel4Bets(GameModel):
    Estimator = XGBClassifier

    def feature_pipeline(self, data):
        print('Running Feature Pipeline')
        start = dt.now()

        print('-- Seeds --')
        seed_feat = SeedFeatures()
        data = seed_feat.per_team_wrapper(
            data, seed_feat.team_seeds,
            per_game=False, per_day=False)
        print(data.shape)

#         print('-- Game Features --')
#         game_feat = GameFeatures()
#         data = game_feat.per_team_wrapper(
#             data, game_feat.last_games_won_in_season)
#         data = game_feat.per_team_wrapper(
#             data, game_feat.last_games_won_in_tourney)
#         data = game_feat.per_team_wrapper(
#             data, game_feat.last_games_won_against_opponent,
#             per_game=True)
#         data = game_feat.per_team_wrapper(
#             data, game_feat.games_won_in_tourney_against_opponent,
#             per_game=True)
#         data.fillna(0, inplace=True)
#         print(data.shape)

#         print('-- Game Detailed Features --')
#         game_detail_feat = GameDetailedFeatures(default_lags=1)
#         data = game_detail_feat.per_team_wrapper(
#             data, game_detail_feat.detail_features_by_game,
#             per_day=True)
#         data.dropna(inplace=True)
#         print(data.shape)

#         print('-- Rankings --')
#         rank_feat = RankingFeatures()
#         # data = rank_feat.per_team_wrapper(
#         #     data, rank_feat.pca_variables_rankings,
#         #     per_game=False, per_day=False)
#         data = rank_feat.per_team_wrapper(
#             data, rank_feat.elos_season,
#             per_game=False, per_day=False)
#         data.fillna(0, inplace=True)
#         print(data.shape)

        print('Feature Pipeline Clock: {} Seconds'
              .format((dt.now() - start).seconds))
        return data
