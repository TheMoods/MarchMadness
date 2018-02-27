import copy as cp
import pandas as pd
from src.features.feature import Feature
from src.features.games import GameFeatures


class GameDetailedFeatures(GameFeatures):
    detail_feature_cols = ['{}FGM', '{}FGA', '{}FGM3', '{}FGA3',
                           '{}FTM', '{}FTA', '{}OR', '{}DR', 
                           '{}Ast', '{}TO', '{}Stl', '{}Blk', '{}PF']

    def __init__(self, default_lags=1):
        super().__init__()
        self.default_lags = default_lags
        self.tourney_games = self\
                .load_game_data('NCAATourneyDetailedResults.csv')
        self.season_games = self\
                .load_game_data('RegularSeasonDetailedResults.csv')

    def load_game_data(self, path):
        games = pd.read_csv('{}{}'.format(self.data_path, path))
        games = games.astype({
            'LTeamID': str,
            'WTeamID': str,
            'Season': str
            })
        games['diff'] = games['WScore'] - games['LScore']
        return games

    def lag_features(self, df, drop_unlagged, lags=None):
        if lags is None:
            lags = self.default_lags
        group_columns = [c for c in df.index.names if c not in 'Season']
        for c in df.columns:
            for l in range(1, lags+1):
                df['{}_lag-{}'.format(c, l)] = df.groupby(group_columns)[[c]]\
                        .shift(l).fillna(0)

            if drop_unlagged:
                df.drop(c, inplace=True, axis=1)

        return df

    def tourney_detail_features_summed(self, df, team):
        tourney_detail_features_summed = pd.concat([
            self.tourney_games\
                .groupby(['WTeamID', 'Season']).sum()\
                [[c.format('W') for c in self.detail_feature_cols]]\
                .rename(columns={
                    c.format('W'): c.format('') + '_tourney_' + team 
                    for c in self.detail_feature_cols
                }),
            self.tourney_games.drop('LTeamID', axis=1)\
                .rename(columns={'LTeamID': 'WTeamID'})\
                .groupby(['WTeamID', 'Season']).sum()\
                [[c.format('L') for c in self.detail_feature_cols]]\
                .rename(columns={
                    c.format('L'): c.format('') + '_tourney_' + team 
                    for c in self.detail_feature_cols
                })
        ]).reset_index().groupby(['WTeamID', 'Season']).sum()
        tourney_detail_features_summed = self.lag_features(tourney_detail_features_summed,
                drop_unlagged=True)
        return tourney_detail_features_summed

    def season_detail_features_summed(self, df, team):
        season_detail_features_summed = pd.concat([
            self.season_games\
                .groupby(['WTeamID', 'Season']).sum()\
                [[c.format('W') for c in self.detail_feature_cols]]\
                .rename(columns={
                    c.format('W'): c.format('') + '_season_' + team 
                    for c in self.detail_feature_cols
                }),
            self.season_games.drop('LTeamID', axis=1)\
                .rename(columns={'LTeamID': 'WTeamID'})\
                .groupby(['WTeamID', 'Season']).sum()\
                [[c.format('L') for c in self.detail_feature_cols]]\
                .rename(columns={
                    c.format('L'): c.format('') + '_season_' + team 
                    for c in self.detail_feature_cols
                })
        ]).reset_index().groupby(['WTeamID', 'Season']).sum()
        season_detail_features_summed = self.lag_features(season_detail_features_summed,
                drop_unlagged=False)
        return season_detail_features_summed

        new_df['a_win'] = df['a_win']
        return new_df
