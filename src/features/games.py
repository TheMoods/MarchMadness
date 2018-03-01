import copy as cp
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
        self.average_rankings = self\
                .load_ranks('MasseyOrdinals.csv')
            
    def load_game_data(self, path):
        games = pd.read_csv('{}{}'.format(self.data_path, path))
        games = games.astype({
            'LTeamID': str,
            'WTeamID': str,
            'Season': str
            })
        games['diff'] = games['WScore'] - games['LScore']
        return games
    
    def load_ranks(self, path):
        ranks = pd.read_csv('{}{}'.format(self.data_path, path))
        ranks = ranks.astype({
            'TeamID': str,
            'Season': str
        })
        return ranks

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
    
    def per_team_wrapper(self, df, feature_func, per_game=False, fillna=None, **kw_args):
        new_df = cp.deepcopy(df)
        for team, opponent in [('team_a', 'team_b'), ('team_b', 'team_a')]:
            if per_game:
                left_merge_cols = [team, opponent, 'Season']
            else:
                left_merge_cols = [team, 'Season']

            new_df = pd.merge(new_df, feature_func(df, team, **kw_args),
                    left_on=left_merge_cols, right_index=True,
                    how='left')

        if fillna is not None:
            new_df.fillna(fillna, inplace=True)

        new_df['a_win'] = df['a_win']
        return new_df
