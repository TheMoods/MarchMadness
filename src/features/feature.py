import copy as cp
import numpy as np
from pandas import DataFrame, MultiIndex, Series, merge


class Feature(object):

    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.default_lags = None

    def per_team_wrapper(self, df, feature_func,
                         per_game=False, per_day=False,
                         combine=None, **kw_args):
        new_df = cp.deepcopy(df)
        for team, opponent in [('team_a', 'team_b'), ('team_b', 'team_a')]:
            if per_game:
                left_merge_cols = [team, opponent, 'Season']
            else:
                left_merge_cols = [team, 'Season']

            if per_day:
                left_merge_cols.append('DayNum')

            merge_df = feature_func(df, team, **kw_args)
            new_df = merge(new_df, merge_df,
                           left_on=left_merge_cols, right_index=True,
                           how='left')

        if combine is not None:
            comb_cols = [c.replace('team_b', 'combined')
                         for c in merge_df.columns]
            b_cols = merge_df.columns
            a_cols = [c.replace('team_b', 'team_a') for c in merge_df.columns]
            if combine == 'subtract':
                for c, a, b in zip(comb_cols, a_cols, b_cols):
                    new_df[c] = new_df[a] - new_df[b].values

            new_df.drop(a_cols, axis=1, inplace=True)
            new_df.drop(b_cols, axis=1, inplace=True)

        new_df['a_win'] = df['a_win']
        return new_df

    def lag_features(self, df, drop_unlagged,
                     fill_missing_dates=False,
                     missing_date_fill_method='ffill',
                     time_indices={
                         'Season': [1, 13]
                     },
                     lags=None):
        if lags is None:
            lags = self.default_lags

        if isinstance(df, Series):
            df = DataFrame(df)

        time_col = df.index.names[len(df.index.names)-1]
        group_cols = df.index.names[:-1]
        if fill_missing_dates:
            time_range = np\
                .arange(*missing_date_min_max)
            time_index = MultiIndex.from_product([
                *df.index.levels[:-1],
                time_range
            ])
            time_index.names = df.index.names
            df = df.reindex(time_index, fill_value=np.nan)
            if missing_date_fill_method is not None:
                df.fillna(method=missing_date_fill_method, inplace=True)

        for c in df.columns:
            for l in range(1, lags+1):
                df['{}_lag-{}'.format(c, l)] = df\
                        .sort_index()\
                        .groupby(group_cols)[[c]]\
                        .shift(l).fillna(0)

            df.dropna(subset=[c], inplace=True)
            if drop_unlagged:
                df.drop(c, inplace=True, axis=1)

        return df
