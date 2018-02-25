import copy as cp
import pandas as pd


def load_target_sample():
    target = pd.read_csv('data/SampleSubmissionStage1.csv').set_index('ID')\
        .drop('Pred', axis=1)
    target['Season'] = target.index.map(lambda i: i[:4])
    target['team_a'] = target.index.map(lambda i: i[5:9])
    target['team_b'] = target.index.map(lambda i: i[10:14])
    target['in_target'] = True
    return target


def load_data_template():
    tourney_games = pd.read_csv('data/NCAATourneyCompactResults.csv')
    target = load_target_sample()
    data = cp.deepcopy(tourney_games)
    data['team_a'] = data[['WTeamID', 'LTeamID']]\
        .apply(lambda t: t[0] if int(t[0]) < int(t[1]) else t[1], axis=1)
    data['team_b'] = data[['WTeamID', 'LTeamID']]\
        .apply(lambda t: t[0] if int(t[0]) > int(t[1]) else t[1], axis=1)
    data['a_win'] = data['WTeamID'] == data['team_a']
    data = data[['Season', 'team_a', 'team_b', 'a_win', 'DayNum']]
    data = pd.concat([data, target.reset_index(drop=True)])\
        .astype({'a_win': bool})
    return data
