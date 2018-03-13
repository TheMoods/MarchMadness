import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from src.features.feature import Feature

class RankingFeatures(Feature):

    def __init__(self, default_lags=3):
        super().__init__()
        self.default_lags = default_lags
        self.rankings = self\
                .load_rankings('MasseyOrdinals_thruSeason2018_Day128.csv')
        self.elos = self\
                .load_elos('season_elos.csv')

    def load_rankings(self, path):
       rankings = pd.read_csv('{}{}'.format(self.data_path, path))
       rankings = rankings.astype({
           'TeamID': str,
           'Season': int,
           'RankingDayNum': int
       }).rename(columns={'RankingDayNum': 'DayNum'})
       return rankings

    def load_elos(self, path):
        elos = pd.read_csv('{}{}'.format(self.data_path, path), header=0, names=['Season', 'Season_elo', 'TeamID'])
        elos = elos.astype({
             'TeamID':str,
             'Season':int,
             'Season_elo':float
             })
        return elos

    def pca_variables_rankings(self, df, team, n_components=20,
            name = 'pca_variables_rankings'):
        pca_variables_team = self.rankings\
                     .pivot_table(values = 'OrdinalRank',
                                  index = ['TeamID', 'Season'], # 'DayNum'
                                  columns = 'SystemName').fillna(0)

        X = pca_variables_team.values
        pca = PCA(n_components)
        X_pca = pca.fit_transform(X)
        col_names = ['PC' + str(pcs+1) + '_' + team for pcs in np.arange(X_pca.shape[1])]
        pca_variables_team = pd.DataFrame(X_pca, columns=col_names,
                                          index=pca_variables_team.index)


#        pca_variables_team = self\
#           .lag_features(pca_variables_team,
#                         drop_unlagged=False,
#                         fill_missing_dates=True,
#                         missing_date_fill_method='ffill',
#                         time_indices={
#                             'Season': [1, 13]
#                             },
#                         lags=None)
        return pca_variables_team

    def elos_season(self, df, team,
        name = 'elos_season'):
        elos_season = self\
                 .elos.set_index(['TeamID','Season'])\
                 .rename(columns={'Season_elo':'{}_{}'.format(name, team)})\
                 .fillna(0)
#        elos_season = self\
#               .lag_features(elos_season,
##                   drop_unlagged=False,
##                   filling_missing_dates=True,
##                   missing_date_fill_method='ffill',
##                   time_indices={
##                       'Season': [1, 13]
##                   },
#                    lags=None)
        return elos_season

