import copy as cp
import pandas as pd


class Feature(object):

    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.default_lags = None
    
    def per_team_wrapper(self, df, feature_func,
                         per_game=False, per_day=False,
                         fillna=None, **kw_args):
        new_df = cp.deepcopy(df)
        for team, opponent in [('team_a', 'team_b'), ('team_b', 'team_a')]:
            if per_game:
                left_merge_cols = [team, opponent, 'Season']
            else:
                left_merge_cols = [team, 'Season']

            if per_day:
                left_merge_cols.append('DayNum')

            new_df = pd.merge(new_df, feature_func(df, team, **kw_args),
                    left_on=left_merge_cols, right_index=True,
                    how='left')

        if fillna is not None:
            new_df.fillna(fillna, inplace=True)

        new_df['a_win'] = df['a_win']
        return new_df
    
    def lag_features(self, df, drop_unlagged, lags=None):
        if lags is None:
            lags = self.default_lags

        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)

        group_columns = [c for c in df.index.names if c not in 'Season']
        for c in df.columns:
            for l in range(1, lags+1):
                df['{}_lag-{}'.format(c, l)] = df.groupby(group_columns)[[c]]\
                        .shift(l).fillna(0)

            if drop_unlagged:
                df.drop(c, inplace=True, axis=1)

        return df
