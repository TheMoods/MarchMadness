import json
import time
import difflib as diff
import pprint as pp
import regex as re
from copy import deepcopy
from betfairlightweight import filters as fi
from datetime import datetime
from functools import partial
from numpy import nan
from numpy.random import choice
from pandas import DataFrame, Series, concat, merge, read_csv


def expand_dict_value_col(c):
    """
    Takes a pandas series of dicts or lists of dicts
    and expands the first level of contained values into a new dataframe.
    Works only with single index at the moment.
    Note: Also clears any missing values out and removes those rows
    """
    c = c.apply(lambda v: nan if not v else v).dropna()
    assert len(c.index.names) == 1
    if type(c.iloc[0]) == list:
        df = concat(
            [concat([Series({**e, c.index.names[0]: i}) for e in r],
                    axis=1) for i, r in c.iteritems()],
            axis=1).transpose()
    elif type(c.iloc[0]) == dict:
        df = concat(
            [Series({**e, c.index.names[0]: i}) for i, e in c.iteritems()],
            axis=1).transpose()
    else:
        raise Exception('Please pass a dict or list of dicts, not {}'
                        .format(type(c.iloc[0])))

    df.set_index(c.index.names[0], inplace=True)
    return df


def dump_json_cols(df, cols):
    for col in cols:
        df[col] = df[col].apply(json.dumps)

    return df


def load_json_cols(df, cols):
    for col in cols:
        df[col] = df[col].apply(json.loads)

    return df


class BetsAPI(object):
    market_projection = [
        'RUNNER_METADATA', 'MARKET_DESCRIPTION',
        'EVENT', 'COMPETITION',
        'EVENT_TYPE', 'MARKET_START_TIME'
    ]
    match_projection = 'NO_ROLLUP'
    order_projection = 'ALL'
    price_projection = {'priceData': ['EX_BEST_OFFERS']}
    cat_json_cols = ['runners', 'description', 'event']
    book_json_cols = ['runners']

    def __init__(self, client, allow_local_load=True,
                 league_name='NCAAB', bet_type='MATCH_ODDS', currency='EUR',
                 name_to_id_path='data/Teams.csv'):
        """
            Notes:
                * `league` corresponds to `competition` in the Betfair API
                * `runner` corresponds to `event` in the Betfair API
                * `team` corresponds to `runner` on Betfair
        """
        self.bet_type = bet_type
        self.client = client
        self.currency = currency
        self.league_name = league_name
        self.name_to_id_path = name_to_id_path

        loaded = False
        if allow_local_load:
            loaded = self.load_last_data()
        if not loaded:
            self.request_data()

        self.build_secondary_tables()

    def update_data(self):
        self.request_data()
        self.build_secondary_tables()
        return self

    def listen_data(self, tick=5):
        while True:
            print('Updating data')
            self.update_data()
            time.sleep(tick)

    def build_secondary_tables(self):
        descriptions = expand_dict_value_col(self.mkt_cat['description'])
        events = expand_dict_value_col(self.mkt_cat['event'])
        events.rename(columns={'id': 'eventId', 'name': 'eventName'},
                      inplace=True)
        runners_cat = expand_dict_value_col(self.mkt_cat['runners'])
        runners_cat['runnerId'] = runners_cat.metadata\
            .apply(lambda d: d['runnerId'])
        runners_cat.drop('metadata', axis=1, inplace=True)
        runners_book = expand_dict_value_col(self.mkt_book['runners'])
        runners = merge(runners_cat, runners_book,
                        on=['selectionId', 'handicap'])
        runners.set_index('runnerId', inplace=True)
        prices = expand_dict_value_col(runners['ex'])
        back = expand_dict_value_col(prices['availableToBack'])
        lay = expand_dict_value_col(prices['availableToLay'])
        runners = runners_cat\
            .join(descriptions)\
            .join(events)\
            .join(concat([back, lay]), how='left', on='runnerId')
        if self.name_to_id_path:
            name2id = read_csv(self.name_to_id_path)
            name_col = next(c for c in name2id.columns
                            if re.search('name', c.lower()))
            id_col = next(c for c in name2id.columns
                          if re.search('id', c.lower()))
            ext_teams = name2id[name_col].values
            get_ext_name = partial(diff.get_close_matches,
                                   possibilities=ext_teams, n=1, cutoff=.01)
            runners['external_name'] = runners['runnerName']\
                .apply(lambda n: get_ext_name(n)[0])
            runners = merge(runners, name2id,
                            left_on=['external_name'], right_on=[name_col],
                            how='left')
            runners.rename(columns={id_col: 'external_id'}, inplace=True)

        odds_index_cols = ['eventName', 'sortPriority',
                           'runnerName', 'external_id']
        self.odds = runners\
            .groupby(odds_index_cols)\
            .agg({'price': ['min', 'max']})\
            .sort_index()
        self.runners = runners

    def save_data(self):
        mkt_book = deepcopy(self.mkt_book)
        mkt_book = dump_json_cols(mkt_book, self.book_json_cols)
        mkt_book.to_csv(
            'data/mkt_book_{league_name}_{bet_type}_{currency}.csv'
            .format(**vars(self)))
        mkt_cat = deepcopy(self.mkt_cat)
        mkt_cat = dump_json_cols(mkt_cat, self.cat_json_cols)
        mkt_cat.to_csv(
            'data/mkt_cat_{league_name}_{bet_type}_{currency}.csv'
            .format(**vars(self)))

    def load_last_data(self):
        try:
            mkt_book = read_csv(
                'data/mkt_book_{league_name}_{bet_type}_{currency}.csv'
                .format(**vars(self)),
                index_col='marketId')
            self.mkt_book = load_json_cols(mkt_book, self.book_json_cols)
            mkt_cat = read_csv(
                'data/mkt_cat_{league_name}_{bet_type}_{currency}.csv'
                .format(**vars(self)),
                index_col='marketId')
            self.mkt_cat = load_json_cols(mkt_cat, self.cat_json_cols)
            return True
        except FileNotFoundError:
            return False

    def request_data(self):
        self.client.login()
        comp_list = self.client.betting.list_competitions()

        def is_league(c):
            return c['competition']['name'] == self.league_name

        try:
            self.league = next(c for c in comp_list if is_league(c))
        except StopIteration:
            raise StopIteration(
                """
                    It seems like the requested league with name '{}'
                    is currently not available as no matching league
                    can be found on Betfair
                    Available leagues (competitions) are: {}
                """.format(self.league_name,
                           [c['competition']['name'] for c in comp_list])
            )
        self.market_filter = fi\
            .market_filter(competition_ids=[self.league['competition']['id']],
                           market_type_codes=[self.bet_type])
        market_type_list = self.client.betting\
            .list_market_types(filter=self.market_filter)
        market_types = DataFrame(market_type_list)
        market_types.set_index('marketType', inplace=True)

        max_results = int(market_types.loc[self.bet_type, 'marketCount'])
        mkt_cat = self.client.betting\
            .list_market_catalogue(filter=self.market_filter,
                                   max_results=max_results,
                                   market_projection=self.market_projection)
        mkt_cat = DataFrame(mkt_cat)
        mkt_cat['marketType'] = mkt_cat.description\
            .apply(lambda d: d['marketType'])
        mkt_cat = mkt_cat[mkt_cat['marketType'] == self.bet_type]
        mkt_cat.set_index('marketId', inplace=True)
        mkt_cat['retrieved'] = datetime.now().isoformat()

        mkt_book = self.client.betting\
            .list_market_book(market_ids=mkt_cat.index,
                              currency_code=self.currency,
                              price_projection=self.price_projection,
                              order_projection=self.order_projection,
                              match_projection=self.match_projection)
        mkt_book = DataFrame(mkt_book).set_index('marketId')
        mkt_book['retrieved'] = datetime.now().isoformat()

        self.mkt_cat = mkt_cat
        self.mkt_book = mkt_book
        self.save_data()
