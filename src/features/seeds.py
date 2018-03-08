import pandas as pd
import numpy as np
from src.features.feature import Feature


class SeedFeatures(Feature):

    def __init__(self, default_lags=3):
        super().__init__()
        self.default_lags = default_lags
        self.seeds = self\
            .load_seeds('NCAATourneySeeds.csv')

    def load_seeds(self, path):
        seeds = pd.read_csv('{}{}'.format(self.data_path, path))
        seeds = seeds.astype({
            'TeamID': str,
            'Season': int
        })
        seeds['Seed'] = seeds['Seed'].apply(lambda s: int(s[1:3]))
        return seeds

    def team_seeds(self, df, team):
        team_seeds = self\
            .seeds.set_index(['TeamID', 'Season'])\
            .rename(columns={
                'Seed': 'seed_{}'.format(team)
            })
        team_seeds = self\
            .lag_features(team_seeds,
                          drop_unlagged=False)
        return team_seeds