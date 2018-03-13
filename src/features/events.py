import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from src.features.feature import Feature


class EventFeatures(Feature):

    def __init__(self, default_lags=3, rows=None):
        super().__init__()
        self.default_lags = default_lags
        self.rows = rows
        self.events = self.load_event_data()

    def load_event_data(self):
        file_names = [file_ for file_ in os.listdir('data/')
                      if 'Events' in file_]
        events = [pd.read_csv('data/'+file_, nrows=self.rows)
                  for file_ in file_names]
        events = pd.concat(events).reset_index(drop=True)
        # Encode event types as numeric (instead of string)
        le = LabelEncoder()
        le.fit(events['EventType'])
        events['EventNum'] = le.transform(events['EventType'])
        # One-hot-encode the numeric types
        ohe = OneHotEncoder()
        ohe.fit(events['EventNum'].values.reshape([-1, 1]))
        ohe_events = pd.DataFrame(ohe.transform(
            events['EventNum'].values.reshape([-1, 1])).toarray())
        # Assign the original strings as column names
        ohe_events.columns = le.classes_
        events = pd.concat([events, ohe_events], axis=1)
        events = events.astype({
            'EventTeamID': str,
            'Season': int
        })
        return events

    def steals_in_season(self, df, team, name='steals_in_season'):
        steals = self.events.groupby(['EventTeamID', 'Season'])['steal'].sum()
        steals = pd.DataFrame(steals).rename(
                columns={'steal': '{}_{}'.format(name, team)})
        return steals
        steals = self.lag_features(steals, drop_unlagged=True)
        return steals
