import os
from betfairlightweight import APIClient
from copy import deepcopy
from functools import partial
from numpy import nan
from pandas import Series, concat
from src.bets.api import BetSet
from src.features.feature import Feature
from src.features.games import GameFeatures
from src.features.games_detailed import GameDetailedFeatures
from src.features.seeds import SeedFeatures
from src.utils import load_data_template
from xgboost import XGBClassifier


class GameSetModel(object):
    Estimator = partial(XGBClassifier, max_depth=7, subsample=.8)
    drop_for_features = ['Season', 'a_win', 'in_target',
                         'DayNum', 'team_a', 'team_b']
    target_cols = ['team_a', 'team_b', 'Season', 'DayNum', 'a_win', 'game_set']
    index_dtypes = {
        'team_a': str,
        'team_b': str,
        'Season': int,
        'DayNum': int
    }

    def __init__(self, bet_set=None, game_features=None):
        self.bet_set = bet_set or self.get_bet_set()
        self.load_data()

    def feature_pipeline(self, data):
        game_feat = GameFeatures(default_lags=3)
        data = game_feat\
            .per_team_wrapper(data,
                              game_feat.last_games_won_in_season,
                              combine='subtract')
        return data

    def load_fit_features(self):
        data = self.get_fit_data_temp()
        data = self.feature_pipeline(data)
        self.fit_features = data\
            .drop(self.drop_for_features, axis=1)
        self.fit_targets = deepcopy(data[self.target_cols])

    def load_pred_features(self):
        data = self.get_pred_data_temp()
        data = self.feature_pipeline(data)
        self.pred_features = data\
            .drop(self.drop_for_features, axis=1)
        self.pred_targets = deepcopy(data[self.target_cols])
        self.pred_targets.loc[:, 'a_win'] = 'not_predicted'

    def load_data(self):
        self.load_fit_features()
        self.load_pred_features()

    def fit(self):
        self.est = self.Estimator()
        self.est.fit(self.fit_features, self.fit_targets['a_win'])

    def predict(self):
        pred = self.est.predict_proba(self.pred_features)
        self.pred_targets['b_win'] = pred[:, 0]
        self.pred_targets['a_win'] = pred[:, 1]

    def load_prediction_table(self):
        self.fit()
        self.predict()
        predictions = concat([
            self.pred_targets.set_index('team_a')['a_win'],
            self.pred_targets.set_index('team_b')['b_win']
        ]).to_frame().rename(columns={0: 'pred'})
        table = self.bet_set.odds\
            .reset_index()\
            .astype({'external_id': str})\
            .set_index('external_id')\
            .join(predictions)
        table['E(r)'] = table['pred'] * table['price_max']
        self.prediction_table = table

    def get_prediction_table(self, refit=False):
        if refit or not hasattr(self, 'prediction_table'):
            self.load_prediction_table()

        return self.prediction_table

    def get_pred_data_temp(self):
        pred_temp = self.bet_set.runners\
            .groupby(['eventName', 'marketTime'])\
            .external_id.unique().reset_index().set_index('eventName')

        def get_team_cols(ids):
            ids = sorted(ids)
            return Series({'team_a': ids[0], 'team_b': ids[1]})

        pred_temp = concat([
            pred_temp,
            pred_temp.external_id.apply(get_team_cols)
        ], axis=1)
        pred_temp['Season'] = 2017
        pred_temp['DayNum'] = 366
        pred_temp['a_win'] = nan
        pred_temp['game_set'] = 1
        pred_temp['in_target'] = True
        pred_temp.reset_index(inplace=True)
        pred_temp.drop(['external_id', 'marketTime', 'eventName'],
                       axis=1, inplace=True)
        pred_temp = pred_temp.astype(self.index_dtypes)
        return pred_temp

    def get_bet_set(self):
        uname = os.environ['BETFAIR_USERNAME']
        pword = os.environ['BETFAIR_PASSWORD']
        key = os.environ['BETFAIR_API_KEY']
        certs_dir = os.environ['BETFAIR_API_CERTS_DIR']
        return BetSet(APIClient(uname, pword, app_key=key,
                                certs=certs_dir,
                                locale='spain',
                                lightweight=True),
                      league_name='NCAAB',
                      bet_type='MATCH_ODDS',
                      currency='EUR',
                      allow_local_load=True)

    def get_fit_data_temp(self):
        fit_temp = load_data_template(season=False)
        fit_temp.dropna(subset=['a_win'], inplace=True)
        fit_temp = fit_temp.astype(self.index_dtypes)
        return fit_temp
