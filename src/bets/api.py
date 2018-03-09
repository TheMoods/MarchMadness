import json
import pprint as pp
from betfairlightweight import filters as fi
from copy import deepcopy
from pandas import DataFrame, Series, concat, merge, read_csv


def expand_dict_value_col(c):
    """
    Takes a pandas series of dicts or lists of dicts
    and expands the first level of contained values into a new dataframe.
    Works only with single index at the moment.
    """
    assert len(c.index.names) == 1
    if type(c.iloc[0]) == list:
        df = concat([concat([Series({**e, 'index': i}) for e in r], axis=1)
                    for i, r in c.iteritems()],
                    axis=1).transpose()
    elif type(c.iloc[0]) == dict:
        df = concat([Series({**e, 'index': i}) for i, e in c.iteritems()],
                    axis=1).transpose()
    else:
        raise Exception('Please pass a dict or list of dicts, not {}'
                        .format(type(c.iloc[0])))

    df.set_index('index', inplace=True)
    return df


class BetsAPI(object):
    market_projection = [
        "RUNNER_METADATA", "MARKET_DESCRIPTION",
        "EVENT", "COMPETITION",
        "EVENT_TYPE", "MARKET_START_TIME"
    ]
    match_projection = [
        "NO_ROLLUP"
    ]
    price_projection = fi\
        .price_projection(price_data=fi.price_data(["EX_BEST_OFFERS"]))

    def __init__(self, client, allow_reload=True,
                 league_name='NCAAB', bet_type='MATCH_ODDS', currency='EUR'):
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

        loaded = False
        if allow_reload:
            loaded = self.load_last_data()
        if not loaded:
            self.request_data()

        runners = expand_dict_value_col(self.mkt_cat['runners'])
        runners['runnerId'] = runners.metadata.apply(lambda d: d['runnerId'])
        runners.drop('metadata', axis=1, inplace=True)
        self.runners = runners

        odds = expand_dict_value_col(self.mkt_book['runners'])
        # odds['runnerId'] = odds.metadata.apply(lambda d: d['runnerId'])
        # odds.drop('metadata', axis=1, inplace=True)
        self.odds = odds

    def load_last_data(self):
        try:
            self.mkt_book = read_csv(
                'data/mkt_book_{league_name}_{bet_type}_{currency}.csv'
                .format(**vars(self)),
                index_col='marketId')
            self.mkt_book.runners = self.mkt_book.runners.apply(json.loads)
            self.mkt_cat = read_csv(
                'data/mkt_cat_{league_name}_{bet_type}_{currency}.csv'
                .format(**vars(self)),
                index_col='marketId')
            self.mkt_cat.runners = self.mkt_cat.runners.apply(json.loads)
            return True
        except FileNotFoundError:
            return False

    def save_data(self):
        mkt_book = deepcopy(self.mkt_book)
        mkt_book.runners = mkt_book.runners.apply(json.dumps)
        mkt_book.to_csv(
            'data/mkt_book_{league_name}_{bet_type}_{currency}.csv'
            .format(**vars(self)))
        mkt_cat = deepcopy(self.mkt_cat)
        mkt_cat.runners = mkt_cat.runners.apply(json.dumps)
        mkt_cat.to_csv(
            'data/mkt_cat_{league_name}_{bet_type}_{currency}.csv'
            .format(**vars(self)))

    def request_data(self):
        self.client.login()
        comp_list = self.client.betting.list_competitions()

        def is_league(c):
            return c['competition']['name'] == self.league_name

        try:
            self.league = next(c for c in comp_list if is_league(c))
        except StopIteration:
            raise StopIteration(
                '''
                    It seems like the requested league with name '{}'
                    is currently not available as no matching league
                    can be found on Betfair
                    Available leagues (competitions) are: {}
                '''.format(self.league_name,
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

        mkt_book = self.client.betting\
            .list_market_book(market_ids=mkt_cat.index,
                              currency_code=self.currency,
                              price_projection=self.price_projection)
        mkt_book = DataFrame(mkt_book).set_index('marketId')

        self.mkt_cat = mkt_cat
        self.mkt_book = mkt_book
        self.save_data()
