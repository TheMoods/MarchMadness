import pandas as pd
from src.features.feature import Feature


class GameFeatures(Feature):

    def __init__(self, default_lags=3):
        super().__init__()
        self.default_lags = default_lags
        self.tourney_games = self\
            .load_game_data('NCAATourneyCompactResults.csv')
        self.season_games = self\
            .load_game_data('RegularSeasonCompactResults.csv')
        self.all_games = pd.concat([
            self.tourney_games, self.season_games
        ])

    def load_game_data(self, path):
        games = pd.read_csv('{}{}'.format(self.data_path, path))
        games = games.astype({
            'LTeamID': str,
            'WTeamID': str,
            'Season': int,
            'DayNum': int
            })
        if any(c in games.columns for c in ['WScore', 'LScore']):
            games['diff'] = games['WScore'] - games['LScore']
        else:
            games['diff'] = 0

        games.DayNum.fillna(366, inplace=True)
        return games

    def games_won_in_season(self, df, team,
                            name='games_won_in_season'):
        feats = self.season_games\
            .groupby(['WTeamID', 'Season']).count()[['diff']]\
            .rename(columns={'diff': '{}_{}'.format(name, team)})\
            .fillna(0)
        feats = self\
            .lag_features(feats,
                          drop_unlagged=False,
                          fill_missing_dates=True,
                          missing_date_fill_method=None,
                          missing_date_min_max=[1985, 2017])
        return feats

    def last_games_won_in_season(self, df, team):
        name = 'last_games_won_in_season'
        feats = self.season_games\
            .groupby(['WTeamID', 'Season']).count()[['diff']]\
            .rename(columns={'diff': '{}_{}'.format(name, team)})\
            .fillna(0)
        feats = self\
            .lag_features(feats,
                          drop_unlagged=False)
        return feats

    def games_won_in_tourney(self, df, team,
                             name='games_won_in_tourney'):
        feats = self.tourney_games\
            .groupby(['WTeamID', 'Season']).count()[['LTeamID']]\
            .rename(columns={'LTeamID': '{}_{}'.format(name, team)})\
            .fillna(0)
        feats = self\
            .lag_features(feats,
                          drop_unlagged=True,
                          fill_missing_dates=True,
                          missing_date_fill_method=None,
                          missing_date_min_max=[1985, 2017])
        return feats

    def last_games_won_in_tourney(self, df, team):
        name = 'last_games_won_in_tourney'
        feats = self.tourney_games\
            .groupby(['WTeamID', 'Season']).count()[['LTeamID']]\
            .rename(columns={'LTeamID': '{}_{}'.format(name, team)})\
            .fillna(0)
        feats = self\
            .lag_features(feats,
                          drop_unlagged=True)
        return feats

    def last_games_won_against_opponent(self, df, team):
        name = 'last_games_won_in_year_against_opponent'
        feats = self.all_games\
            .groupby(['WTeamID', 'LTeamID', 'Season']).count()[['diff']]\
            .rename(columns={'diff': '{}_{}'.format(name, team)})\
            .fillna(0)
        feats = self\
            .lag_features(feats,
                          drop_unlagged=True)
        return feats

    def games_won_in_tourney_against_opponent(self, df, team):
        name = 'games_won_in_tourney_against_opponent'
        feats = self.tourney_games\
            .groupby(['WTeamID', 'LTeamID', 'Season']).count()[['diff']]\
            .rename(columns={'diff': '{}_{}'.format(name, team)})\
            .fillna(0)
        feats = self\
            .lag_features(feats,
                          drop_unlagged=True)
        return feats
