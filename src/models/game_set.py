import os
from betfairlightweight import APIClient
from numpy import nan
from pandas import Series, concat
from src.bets.api import BetSet
from src.models.game import GameModel


class GameSetModel(object):
    index_dtypes = {
        'team_a': str,
        'team_b': str,
        'Season': int,
        'DayNum': int
    }

    def __init__(self, bet_set=None, model=None):
        self.bet_set = bet_set or self.get_bet_set()
        self.model = model or self.get_model()

    def load_prediction_table(self):
        self.model.fit()
        self.model.predict()
        predictions = concat([
            self.model.pred_targets.set_index('team_a')['a_win'],
            self.model.pred_targets.set_index('team_b')['b_win']
        ]).to_frame().rename(columns={0: 'pred'})
        table = self.bet_set.odds\
            .reset_index()\
            .astype({'external_id': str})\
            .join(predictions, on='external_id')
        table['E(r)'] = table['pred'] * table['price_max']
        self.prediction_table = table

    def get_prediction_table(self, refit=False):
        if refit or not hasattr(self, 'prediction_table'):
            self.load_prediction_table()

        return self.prediction_table

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
                      allow_local_load=False)

    def get_model(self):
        return GameModel(pred_data_temp=self.get_pred_data_temp())

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
