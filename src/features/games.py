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
        games_won_in_season = self.season_games\
            .groupby(['WTeamID', 'Season']).count()[['diff']]\
            .rename(columns={'diff': '{}_{}'.format(name, team)})
        games_won_in_season = self\
            .lag_features(games_won_in_season,
                          drop_unlagged=False,
                          fill_missing_dates=True,
                          missing_date_fill_method=None,
                          missing_date_min_max=[1985, 2017])
        return games_won_in_season

    def last_games_won_in_season(self, df, team):
        name = 'last_games_won_in_season'
        last_games_won_in_season = self.season_games\
            .groupby(['WTeamID', 'Season']).count()[['diff']]\
            .rename(columns={'diff': '{}_{}'.format(name, team)})
        last_games_won_in_season = self\
            .lag_features(last_games_won_in_season,
                          drop_unlagged=False)
        return last_games_won_in_season

    def games_won_in_tourney(self, df, team,
                             name='games_won_in_tourney'):
        games_won_in_tourney = self.tourney_games\
            .groupby(['WTeamID', 'Season']).count()[['LTeamID']]\
            .rename(columns={'LTeamID': '{}_{}'.format(name, team)})
        games_won_in_tourney = self\
            .lag_features(games_won_in_tourney,
                          drop_unlagged=True,
                          fill_missing_dates=True,
                          missing_date_fill_method=None,
                          missing_date_min_max=[1985, 2017])
        return games_won_in_tourney

    def last_games_won_in_tourney(self, df, team):
        name = 'last_games_won_in_tourney'
        last_games_won_in_tourney = self.tourney_games\
            .groupby(['WTeamID', 'Season']).count()[['LTeamID']]\
            .rename(columns={'LTeamID': '{}_{}'.format(name, team)})
        last_games_won_in_tourney = self\
            .lag_features(last_games_won_in_tourney,
                          drop_unlagged=True)
        return last_games_won_in_tourney

    def last_games_won_against_opponent(self, df, team):
        name = 'last_games_won_in_year_against_opponent'
        last_games_won__against_opponent = self.all_games\
            .groupby(['WTeamID', 'LTeamID', 'Season']).count()[['diff']]\
            .rename(columns={'diff': '{}_{}'.format(name, team)})
        last_games_won__against_opponent = self\
            .lag_features(last_games_won__against_opponent,
                          drop_unlagged=True)
        return last_games_won__against_opponent

    def games_won_in_tourney_against_opponent(self, df, team):
        name = 'games_won_in_tourney_against_opponent'
        games_won_in_tourney_against_opponent = self.tourney_games\
            .groupby(['WTeamID', 'LTeamID', 'Season']).count()[['diff']]\
            .rename(columns={'diff': '{}_{}'.format(name, team)})
        games_won_in_tourney_against_opponent = self\
            .lag_features(games_won_in_tourney_against_opponent,
                          drop_unlagged=True)
        return games_won_in_tourney_against_opponent
