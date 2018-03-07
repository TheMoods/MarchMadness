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
        
    def load_game_data(self, path):
        games = pd.read_csv('{}{}'.format(self.data_path, path))
        games = games.astype({
            'LTeamID': str,
            'WTeamID': str,
            'Season': str
            })
        games['diff'] = games['WScore'] - games['LScore']
        return games
    
    def games_won_in_season(self, df, team,
            name='games_won_in_season'):
        games_won_in_season = self.season_games\
                .groupby(['WTeamID', 'Season']).count()[['diff']]\
                .rename(columns={'diff': '{}_{}'.format(name, team)})
        games_won_in_season = self.lag_features(games_won_in_season,
                drop_unlagged=False)
        return games_won_in_season

    def games_won_in_season_against_opponent(self, df, team,
            name='games_won_in_season_against_opponent'):
        games_won_in_season_against_opponent = self.season_games\
                .groupby(['WTeamID', 'LTeamID', 'Season']).count()[['diff']]\
                .rename(columns={'diff': '{}_{}'.format(name, team)})
        games_won_in_season_against_opponent = self.lag_features(games_won_in_season_against_opponent,
                drop_unlagged=False)
        return games_won_in_season_against_opponent

    def games_won_in_tourney(self, df, team,
            name='games_won_in_tourney'):
        games_won_in_tourney = self.tourney_games\
                .groupby(['WTeamID', 'Season']).count()[['diff']]\
                .rename(columns={'diff': '{}_{}'.format(name, team)})
        games_won_in_tourney = self.lag_features(games_won_in_tourney,
                drop_unlagged=True)
        return games_won_in_tourney

    def games_won_in_tourney_against_opponent(self, df, team,
            name='games_won_in_tourney_against_opponent'):
        games_won_in_tourney_against_opponent = self.tourney_games\
                .groupby(['WTeamID', 'LTeamID', 'Season']).count()[['diff']]\
                .rename(columns={'diff': '{}_{}'.format(name, team)})
        games_won_in_tourney_against_opponent = self.lag_features(games_won_in_tourney_against_opponent,
                drop_unlagged=True)
        return games_won_in_tourney_against_opponent
