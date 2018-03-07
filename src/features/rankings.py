import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from src.features.feature import Feature

class RankingFeatures(Feature):

    def __init__(self, default_lags=3, rows=1000):
        super().__init__()
        self.default_lags = default_lags
        self.rows = rows
        self.rankings = self\
                .load_rankings('MasseyOrdinals.csv')

    def load_rankings(self, path):
       rankings = pd.read_csv('{}{}'.format(self.data_path, path))
       rankings = rankings.astype({
           'TeamID': str,
           'Season': str,
        })
       return rankings

    def pca_variables_rankings(self, df, team, n_components=20,
            name = 'pca_variables_rankings'):
        pca_variables_team = self.rankings\
                     .pivot_table(values = 'OrdinalRank',
                                  index = ['TeamID', 'Season', 'RankingDayNum'],
                                  columns = 'SystemName').fillna(0)

        X = pca_variables_team.iloc[:, 3:].values
        pca = PCA(n_components)
        X_pca = pca.fit_transform(X)
        col_names = ['PC' + str(pcs+1) + '_' + team for pcs in np.arange(X_pca.shape[1])]
        pca_variables_team = pd.DataFrame(X_pca, columns=col_names,
                                          index=pca_variables_team.index)
        pca_variables_team = self.lag_features(pca_variables_team,
                                         drop_unlagged=False)
        return pca_variables_team






