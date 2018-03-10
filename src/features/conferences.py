import pandas as pd
from src.features.feature import Feature
from sklearn.preprocessing import LabelEncoder

class ConferenceFeatures(Feature):

    def __init__(self, default_lags=3, rows=1000):
        super().__init__()
        self.default_lags = default_lags
        self.conferences = self\
                .load_conferences('TeamConferences.csv')

    def load_conferences(self, path):

        conferences = pd.read_csv('{}{}'.format(self.data_path, path))
        # Encode event types as numeric (instead of string)
        le = LabelEncoder()
        le.fit(conferences['ConfAbbrev'])
        conferences['ConfAbbrev'] = le.transform(conferences['ConfAbbrev'])
        conferences = conferences.astype({
            'TeamID':str,
            'Season':int,
            })
        return conferences

    def conference_games(self, df, team):
        conferences = self\
                    .conferences.set_index(['TeamID','Season'])\
                    .rename(columns={
                        'ConfAbbrev':'conference_{}'.format(team)
                        })
        conferences = self\
                .lag_features(conferences,
                        drop_unlagged = False)
        return conferences
