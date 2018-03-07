import pprint as pp
from pandas import DataFrame, Series, concat, merge
from betfairlightweight import filters as fi


class BetsAPI(object):

    def __init__(self, client,
                 league_name='NCAAB', bet_type='MATCH_ODDS', currency='EUR'):
        """
            Notes:
                * `league` corresponds to `competition` in the Betfair API
                * `game` corresponds to `event` in the Betfair API
                * `team` corresponds to `runner` on Betfair
        """
        self.client = client
        self.currency = currency
        self.client.login()

        comp_list = self.client.betting.list_competitions()

        def is_league(c):
            return c['competition']['name'] == league_name

        try:
            self.league = next(c for c in comp_list if is_league(c))
        except StopIteration:
            pass
            raise StopIteration(
                '''
                    It seems like the requested league with name '{}'
                    is currently not available as no matching league
                    can be found on Betfair
                    Available leagues (competitions) are: {}
                '''.format(league_name,
                           [c['competition']['name'] for c in comp_list])
            )
        self.league_filter = fi\
            .market_filter(competition_ids=[self.league['competition']['id']])

        event_list = self.client.betting\
            .list_events(filter=self.league_filter)
        games = DataFrame([g['event'] for g in event_list])
        games.set_index('id', inplace=True)

        market_type_list = self.client.betting\
            .list_market_types(filter=self.league_filter)
        market_types = DataFrame(market_type_list)
        market_types.set_index('marketType', inplace=True)

        max_results = int(market_types.loc[bet_type, 'marketCount'])
        market_catalogue = self.client.betting\
            .list_market_catalogue(filter=self.league_filter,
                                   max_results=max_results,
                                   market_projection=["RUNNER_METADATA"])
        market_catalogue = DataFrame(market_catalogue)
        market_catalogue.set_index('marketId', inplace=True)

        self.price_projection = fi\
            .price_projection(price_data=fi.price_data(["EX_BEST_OFFERS"]))
        market_book = self.client.betting\
            .list_market_book(market_ids=market_catalogue.index,
                              currency_code=self.currency,
                              price_projection=self.price_projection)
        market_book = DataFrame(market_book).set_index('marketId')

        self.client = client
        self.games = games
        self.market_types = market_types
        self.market_catalogue = market_catalogue
        self.market_book = market_book
        self.teams = self.get_teams(market_catalogue)

    def get_teams(self, df, col='runners'):
        teams = concat([concat([Series(r) for r in row], axis=1)
                       for row in df[col].values],
                       axis=1).transpose()
        teams['runnerId'] = teams.metadata.apply(lambda d: d['runnerId'])
        teams.drop('metadata', axis=1, inplace=True)
        teams.set_index('runnerId', inplace=True)
        return teams
