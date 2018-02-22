import pandas as pd


def load_target_sample():
    target = pd.read_csv('data/SampleSubmissionStage1.csv').set_index('ID')\
        .drop('Pred', axis=1)
    target['Season'] = target.index.map(lambda i: i[:4])
    target['team_a'] = target.index.map(lambda i: i[5:9])
    target['team_b'] = target.index.map(lambda i: i[10:14])
    target['in_target'] = True
    return target
