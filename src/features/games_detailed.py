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
        self.season_games = self\
            .load_game_data('RegularSeasonDetailedResults_Prelim2018.csv')
        self.tourney_games = self\
            .load_game_data('NCAATourneyDetailedResults.csv')
        self.all_games = pd.concat([
            self.season_games,
            self.tourney_games
        ])

    def load_game_data(self, path):
        games = pd.read_csv('{}{}'.format(self.data_path, path))
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
        detail_features_by_game = pd.concat([
            self.all_games.set_index(['WTeamID', 'Season', 'DayNum'])\
                [[c.format('W') for c in self.detail_feature_cols]]\
                .sort_index(ascending=True)\
                .rename(columns={
                    c.format('W'): c.format('') + '_game_' + team 
                    for c in self.detail_feature_cols
                }),
            self.all_games.drop('WTeamID', axis=1)\
                .rename(columns={'LTeamID': 'WTeamID'})\
                .set_index(['WTeamID', 'Season', 'DayNum'])\
                [[c.format('L') for c in self.detail_feature_cols]]\
                .sort_index(ascending=True)\
                .rename(columns={
                    c.format('L'): c.format('') + '_game_' + team 
                    for c in self.detail_feature_cols
                })
        ])
        detail_features_by_game = self.lag_features(detail_features_by_game,
                drop_unlagged=True)
        return detail_features_by_game

    def tourney_detail_features_summed_over_season(self, df, team):
        tourney_detail_features_summed_over_season = pd.concat([
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
        tourney_detail_features_summed_over_season = self.lag_features(tourney_detail_features_summed_over_season,
                drop_unlagged=True)
        return tourney_detail_features_summed_over_season

    def season_detail_features_summed_over_season(self, df, team):
        season_detail_features_summed_over_season = pd.concat([
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
        season_detail_features_summed_over_season = self.lag_features(season_detail_features_summed_over_season,
                drop_unlagged=False)
        return season_detail_features_summed_over_season

        new_df['a_win'] = df['a_win']
        return new_df
