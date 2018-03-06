from pandas import DataFrame
from betfairlightweight import filters as fi


class BetsAPI(object):

    def __init__(self, client, league_name='NCAAB', bet_type='MATCH_ODDS'):
        """
            Notes:
                * `league`, corresponds to `competition` in the Betfair API
                * `game`, corresponds to `event` in the Betfair API
        """
        client.login()

        comp_list = client.betting.list_competitions()

        def is_league(c):
            return c['competition']['name'] == league_name

        self.league = next(c for c in comp_list if is_league(c))
        self.league_filter = fi\
            .market_filter(competition_ids=[self.league['competition']['id']])

        event_list = client.betting\
            .list_events(filter=self.league_filter)
        games = DataFrame([g['event'] for g in event_list])
        games.set_index('id', inplace=True)

        market_type_list = client.betting\
            .list_market_types(filter=self.league_filter)
        market_types = DataFrame(market_type_list)
        print(market_types.columns)
        market_types.set_index('marketType', inplace=True)

        max_results = int(market_types.loc[bet_type, 'marketCount'])
        market_catalogue = client.betting\
            .list_market_catalogue(filter=self.league_filter,
                                   max_results=max_results,
                                   market_projection=["RUNNER_METADATA"])
        market_catalogue = DataFrame(market_catalogue)
        market_catalogue.set_index('marketId', inplace=True)

        self.client = client
        self.market_types = market_types
        self.market_catalogue = market_catalogue
        self.games = games
