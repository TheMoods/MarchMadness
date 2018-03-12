import pandas as pd
from src.features.feature import Feature
from sklearn.preprocessing import LabelEncoder

class CoachFeatures(Feature):

    def __init__(self, default_lags=3, rows=1000):
        super().__init__()
        self.default_lags = default_lags
        self.coaches = self\
                .load_coaches('TeamCoaches.csv')

    def load_coaches(self, path):
        coaches = pd.read_csv('{}{}'.format(self.data_path, path))
        # Encode coach types as numeric (instead of string)
        le = LabelEncoder()
        le.fit(coaches['CoachName'])
        coaches['CoachName'] = le.transform(coaches['CoachName'])
        coaches = coaches.astype({
            'TeamID':str,
            'Season':int,
            'FirstDayNum':int,
            'LastDayNum':int
            })
        return coaches

    def coach_func(self, df, team):
        coaches = self\
                .coaches.set_index(['TeamID', 'Season'])\
                .rename(columns={
                    'CoachName':'coach_{}'.format(team)
                    })
        coaches = coaches.drop(['FirstDayNum','LastDayNum'], axis = 1)
       # coaches = self\
       #         .lag_features(coaches,
       #         drop_unlagged = False)
       # print(coaches)
       # return coaches
    # Unfinished. TODO: I need to count the years that a coach has been on a team.
    def coach_years(self, df, team):
        coach_years = self\
                .coaches.groupby(['TeamID'])\
                .agg('count')[['CoachName']]\
                .rename(columns={'CoachName':'coach_experience_{}'.format(team)})\
                .reset_index(['TeamID','Season'], inplace=False)
        print(coach_years)
        coach_years = self\
                .lag_features(coach_years,
                    drop_unlagged=False)
        return coach_years
