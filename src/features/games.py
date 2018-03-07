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
        self.season_games = self\
                .load_game_data('RegularSeasonCompactResults.csv')
        self.conference_games = self\
                .load_game_data('ConferenceTourneyGames.csv')
        self.average_rankings = self\
            .load_ranks('MasseyOrdinals.csv')
        
    def load_game_data(self, path):
        games = pd.read_csv('{}{}'.format(self.data_path, path))
        games = games.astype({
            'LTeamID': str,
            'WTeamID': str,
            'Season': str,
            'DayNum': str
            })
        if any(c in games.columns for c in ['WScore', 'LScore']):
            games['diff'] = games['WScore'] - games['LScore']
        else:
            games['diff'] = 0

        games.DayNum.fillna(366, inplace=True)
        return games
    
    def load_ranks(self, path):
        ranks = pd.read_csv('{}{}'.format(self.data_path, path))
        ranks = ranks.astype({
            'TeamID': str,
            'Season': str
        })
        return ranks

    def games_won_in_season(self, df, team,
            name='games_won_in_season'):
        games_won_in_season = self.season_games\
                .groupby(['WTeamID', 'Season']).count()[['diff']]\
                .rename(columns={'diff': '{}_{}'.format(name, team)})
        games_won_in_season = self.lag_features(games_won_in_season,
                drop_unlagged=False)
        return games_won_in_season

    def games_won_against_opponent(self, df, team,
            name='games_won_against_opponent'):
        games_won_against_opponent = self.season_games\
                .groupby(['WTeamID', 'LTeamID', 'Season']).count()[['diff']]\
                .rename(columns={'diff': '{}_{}'.format(name, team)})
        games_won_against_opponent = self.lag_features(games_won_against_opponent,
                drop_unlagged=False)
        return games_won_against_opponent

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

    def games_won_in_conference(self, df, team,
            name='games_won_in_conference'):
        games_won_in_conference = self.conference_games\
                .groupby(['WTeamID', 'Season']).count()[['diff']]\
                .rename(columns={'diff': '{}_{}'.format(name, team)})
        games_won_in_conference = self.lag_features(games_won_in_conference,
                drop_unlagged=False)
        return games_won_in_conference

    def games_won_in_conference_against_opponent(self, df, team,
            name='games_won_in_conference_against_opponent'):
        games_won_in_conference_against_opponent = self.conference_games\
                .groupby(['WTeamID', 'LTeamID', 'Season']).count()[['diff']]\
                .rename(columns={'diff': '{}_{}'.format(name, team)})
        games_won_in_conference_against_opponent = self.lag_features(games_won_in_conference_against_opponent,
                drop_unlagged=False)
        return games_won_in_conference_against_opponent

    def average_ranking_team(self, df, team, 
             name='average_ranking'):
        average_ranking_team = self.average_rankings\
                .groupby(['TeamID','Season',]).mean()[['OrdinalRank']]\
                .rename(columns={'OrdinalRank': '{}_{}'.format(name, team)})
        average_ranking_team = self.lag_features(average_ranking_team,
                drop_unlagged=False)
        return average_ranking_team
    
    def sd_ranking_team(self, df, team, 
                        name='sd_rankings'):
        sd_ranking_team = self.average_rankings\
                .groupby(['TeamID', 'Season',]).std()[['OrdinalRank']]\
                .rename(columns={'OrdinalRank': '{}_{}'.format(name, team)})
        sd_ranking_team = self.lag_features(sd_ranking_team,
                drop_unlagged=False)
        return sd_ranking_team
