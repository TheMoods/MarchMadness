from copy import deepcopy
from pandas import concat, read_csv
from src.features.feature import Feature
from src.features.games import GameFeatures


class GameDetailedFeatures(GameFeatures):
    detail_feature_cols = ['{}FGM', '{}FGA', '{}FGM3', '{}FGA3',
                           '{}FTM', '{}FTA', '{}OR', '{}DR',
                           '{}Ast', '{}TO', '{}Stl', '{}Blk', '{}PF']

    def __init__(self, default_lags=1):
        super().__init__()
        self.default_lags = default_lags
        self.season_games = self\
            .load_game_data('RegularSeasonDetailedResults_Prelim2018.csv')
        self.tourney_games = self\
            .load_game_data('NCAATourneyDetailedResults.csv')
        self.all_games = concat([
            self.season_games,
            self.tourney_games
        ])

    def load_game_data(self, path):
        games = read_csv('{}{}'.format(self.data_path, path))
        games = games.astype({
            'LTeamID': str,
            'WTeamID': str,
            'Season': int,
            'DayNum': int
            })
        games['diff'] = games['WScore'] - games['LScore']
        games.DayNum.fillna(366, inplace=True)
        return games

    def lag_features(self, df, drop_unlagged, lags=None):
        if lags is None:
            lags = self.default_lags
        for c in df.columns:
            for l in range(1, lags+1):
                df['{}_lag-{}'.format(c, l)] = df[[c]]\
                        .shift(l).fillna(0)

            if drop_unlagged:
                df.drop(c, inplace=True, axis=1)

        return df

    def detail_features_by_game(self, df, team):
        games = self.all_games.set_index(['WTeamID', 'LTeamID'])
        time_cols = ['Season', 'DayNum']
        winner_cols = [c.format('W') for c in self.detail_feature_cols]
        winner_features = deepcopy(games[winner_cols + time_cols])
        winner_features.columns = [c[1:] + '_' + team
                                   for c in winner_cols] + time_cols
        winner_features.index = winner_features.index.droplevel(1)
        looser_cols = [c.format('L') for c in self.detail_feature_cols]
        looser_features = deepcopy(games[looser_cols + time_cols])
        looser_features.columns = [c[1:] + '_' + team
                                   for c in looser_cols] + time_cols
        looser_features.index = looser_features.index.droplevel(0)
        feats = concat([looser_features, winner_features])\
            .reset_index().rename(columns={'index': 'team_id'})
        last_days = feats\
            .groupby(['team_id', 'Season']).last()\
            .reset_index()
        last_days['DayNum'] = 366
        feats = \
            concat([feats,
                    last_days[last_days.Season == last_days.Season.max()]])
        feats.set_index(['team_id', 'Season', 'DayNum'], inplace=True)
        feats = self.lag_features(feats, drop_unlagged=True)
        return feats
