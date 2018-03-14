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
        non_event_columns = ['EventID', 'Season', 'DayNum',
            'WPoints', 'LPoints', 'WTeamID', 'LTeamID',
            'ElapsedSeconds', 'EventTeamID',
            'EventPlayerID', 'EventType', 'EventNum']
        self.event_columns = [col for col in self.events.columns
                              if col not in non_event_columns]

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
            'Season': int,
        })
        return events

    def total_events_in_season(self, df, team, name='total'):
        total_events = self.events\
                .groupby(['EventTeamID', 'Season', 'DayNum']).sum()\
                .groupby(['EventTeamID', 'Season'])[self.event_columns].sum()
        total_events = pd.DataFrame(total_events).rename(
            columns={key: '{}_{}_{}'.format(key, name, team) 
                     for key in self.event_columns})
        total_events = self.lag_features(total_events, drop_unlagged=True)
        return total_events

    def average_events_in_season(self, df, team, name='avg'):
        total_events = self.events\
                .groupby(['EventTeamID', 'Season', 'DayNum']).sum()\
                .groupby(['EventTeamID', 'Season'])[self.event_columns].mean()
        total_events = pd.DataFrame(total_events).rename(
            columns={key: '{}_{}_{}'.format(key, name, team)
                     for key in self.event_columns})
        total_events = self.lag_features(total_events, drop_unlagged=True)
        return total_events

    def total_events_by_game(self, df, team, name='total'):
        total_events = self.events\
                .groupby(['EventTeamID', 'Season', 'DayNum'])\
                [self.event_columns].sum()
        total_events = pd.DataFrame(total_events).rename(
            columns={key: '{}_{}_{}'.format(key, name, team) 
                     for key in self.event_columns})
        total_events = self.lag_features(total_events, drop_unlagged=True)
        return total_events

    def average_events_by_game(self, df, team, name='avg'):
        total_events = self.events\
                .groupby(['EventTeamID', 'Season', 'DayNum'])\
                [self.event_columns].mean()
        total_events = pd.DataFrame(total_events).rename(
            columns={key: '{}_{}_{}'.format(key, name, team)
                     for key in self.event_columns})
        total_events = self.lag_features(total_events, drop_unlagged=True)
        return total_events
