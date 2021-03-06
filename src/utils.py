import pandas as pd
from numpy import nan
from sklearn.preprocessing import LabelEncoder


def load_target_sample():
    target = pd.read_csv('data/SampleSubmissionStage2.csv').set_index('ID')\
        .drop('Pred', axis=1)
    target['Season'] = target.index.map(lambda i: i[:4])
    target['team_a'] = target.index.map(lambda i: i[5:9])
    target['team_b'] = target.index.map(lambda i: i[10:14])
    target['in_target'] = True
    target['game_set'] = 'ncaa'
    return target


def load_data_template(season=False):
    tourney_games = pd.read_csv('data/NCAATourneyCompactResults.csv')
    tourney_games['game_set'] = 'ncaa'
    data = [tourney_games]
    if season:
        season_games = pd.read_csv('data/RegularSeasonCompactResults_Prelim2018.csv')
        season_games['game_set'] = 'season'
        data.append(season_games)

    data = pd.concat(data).astype({
        'Season': str, 'WTeamID': str,
        'LTeamID': str, 'DayNum': int
    })
    data['team_a'] = data[['WTeamID', 'LTeamID']]\
        .apply(lambda t: t[0] if int(t[0]) < int(t[1]) else t[1], axis=1)
    data['team_b'] = data[['WTeamID', 'LTeamID']]\
        .apply(lambda t: t[0] if int(t[0]) > int(t[1]) else t[1], axis=1)
    data['a_win'] = data['WTeamID'] == data['team_a']
    data = data[['Season', 'team_a', 'team_b', 'a_win', 'DayNum', 'game_set']]
    target = load_target_sample()
    target_index = target.index.tolist()
    data['in_target'] = data[['Season', 'team_a', 'team_b']]\
        .apply(lambda r: '_'.join(r.values) in target_index, axis=1)
    data = pd.merge(target, data,
                    on=['Season', 'team_a', 'team_b', 'game_set', 'in_target'],
                    how='outer')
    data['DayNum'].fillna(366, inplace=True)
    data = data.astype({
        'Season': int, 'DayNum': int,
        'team_a': str, 'team_b': str,
        'in_target': bool
    })
    data['game_set'] = LabelEncoder().fit_transform(data['game_set'])
    return data
