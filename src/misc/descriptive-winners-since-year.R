
list.of.packages <- c('data.table', 'magrittr', 'ggplot2', 'gridExtra')
new.packages     <- list.of.packages[!(list.of.packages %in% installed.packages()[, "Package"])]
if(length(new.packages)) install.packages(new.packages)


theme_set(theme_bw())

descriptive_stats_winners_after_year <- function(year){
    #  Function to plot the winners of tournaments and other statistics
    teams <- fread('data/Teams.csv')
    seasons <- fread('data/Seasons.csv')
    seeds <- fread('data/NCAATourneySeeds.csv')
    seas_results <- fread('data/RegularSeasonCompactResults.csv')
    tour_results <- fread('data/NCAATourneyCompactResults.csv')
    
    seasons <- seasons[seasons$Season > year]
    seeds   <- seeds[seeds$Season > year]
    seas_results <- seas_results[seas_results$Season > year]
    tour_results <- tour_results[tour_results$Season > year]
    
    setkey(teams, TeamID)
    setkey(seeds, TeamID)
    
    g1 <-
        teams[seeds][, one_seed := as.numeric(substr(Seed, 2, 3)) == 1][, sum(one_seed), by = TeamName][order(V1, decreasing = T)][1:15,] %>%
        ggplot(aes(x = reorder(TeamName, V1), y = V1)) +
        geom_bar(stat = 'identity', fill = 'darkred') +
        labs(x = '', y = 'No 1 seeds', title = paste('No. 1 Seeds since ', year, sep = '')) +
        coord_flip()
    
    setkey(seas_results, WTeamID)
    
    g2 <-
        seas_results[teams][, .(wins = .N), by = TeamName][order(-wins)][1:15,] %>%
        ggplot(aes(x = reorder(TeamName, wins), y = wins)) +
        geom_bar(stat = 'identity', fill = 'darkred') +
        labs(x = '', y = 'Wins', title = paste('Regular Season Wins since ', year, sep = '')) +
        coord_flip()
    
    setkey(tour_results, WTeamID)
    
    g3 <-
        tour_results[teams][, .(wins = .N), by = TeamName][order(-wins)][1:15,] %>%
        ggplot(aes(x = reorder(TeamName, wins), y = wins)) +
        geom_bar(stat = 'identity', fill = 'darkred') +
        labs(x = '', y = 'Wins', title = paste('Tournament Wins since ', year, sep = '')) +
        coord_flip()
    
    g4 <-
        tour_results[teams][DayNum == 154, .(wins = .N), by = TeamName][order(-wins)][1:(2017-year),] %>%
        ggplot(aes(x = reorder(TeamName, wins), y = wins)) +
        geom_bar(stat = 'identity', fill = 'darkred') +
        labs(x = '', y = 'Championships', title = paste('Tournament Championships since ', year, sep = '')) +
        coord_flip()
    # Throws warning and positions NA on top row, but doesn't affect
    
    return(grid.arrange(g1, g2, g3, g4, nrow = 2))
}

# Notebook copied from - >
# https://www.kaggle.com/captcalculator/a-very-extensive-ncaa-exploratory-analysis
# Nice explanations of the intuitions are provided there.

teams        <- fread('data/Teams.csv')
seasons      <- fread('data/Seasons.csv')
seeds        <- fread('data/NCAATourneySeeds.csv')
seas_results <- fread('data/RegularSeasonCompactResults.csv')
tour_results <- fread('data/NCAATourneyCompactResults.csv')

setkey(teams, TeamID)
setkey(seeds, TeamID)

g1 <-
    teams[seeds][, one_seed := as.numeric(substr(Seed, 2, 3)) == 1][, sum(one_seed), by = TeamName][order(V1, decreasing = T)][1:15,] %>%
    ggplot(aes(x = reorder(TeamName, V1), y = V1)) +
    geom_bar(stat = 'identity', fill = 'darkred') +
    labs(x = '', y = 'No 1 seeds', title = 'No. 1 Seeds since 1985') +
    coord_flip()

setkey(seas_results, WTeamID)

g2 <-
    seas_results[teams][, .(wins = .N), by = TeamName][order(-wins)][1:15,] %>%
    ggplot(aes(x = reorder(TeamName, wins), y = wins)) +
    geom_bar(stat = 'identity', fill = 'darkred') +
    labs(x = '', y = 'Wins', title = 'Regular Season Wins since 1985') +
    coord_flip()

setkey(tour_results, WTeamID)

g3 <-
    tour_results[teams][, .(wins = .N), by = TeamName][order(-wins)][1:15,] %>%
    ggplot(aes(x = reorder(TeamName, wins), y = wins)) +
    geom_bar(stat = 'identity', fill = 'green') +
    labs(x = '', y = 'Wins', title = 'Tournament Wins since 1985') +
    coord_flip()

g4 <-
    tour_results[teams][DayNum == 154, .(wins = .N), by = TeamName][order(-wins)][1:15,] %>%
    ggplot(aes(x = reorder(TeamName, wins), y = wins)) +
    geom_bar(stat = 'identity', fill = 'green') +
    labs(x = '', y = 'Championships', title = 'Tournament Championships since 1985') +
    coord_flip()

grid.arrange(g1, g2, g3, g4, nrow = 2)

# Tournament wins by regular season wins

wins_s <- seas_results[, .(rsW = .N), by = c('WTeamID', 'Season')]

wins_t <- tour_results[, .(tW = .N), by = c('WTeamID', 'Season')]

wins_teams <- wins_s[wins_t][teams]

wins_teams[!is.na(Season), ] %>%
    ggplot(aes(x = rsW, y = tW)) + 
    geom_point() + 
    geom_smooth(method = 'lm') + 
    facet_wrap( ~ as.factor(Season)) + 
    labs(x = 'Regular season wins', y = 'Tournament wins', title = 'Tournament Wins by Regular Season Wins')

# Tournament Wins by regular season points per game

wins <- seas_results[, .(n_games = .N, sum_score = sum(WScore)), by = c('WTeamID', 'Season')]

losses <- seas_results[, .(n_games = .N, sum_score = sum(LScore)), by = c('LTeamID', 'Season')]

all_games <- rbindlist(list(wins, losses))

all_games <- all_games[, .(rs_ppg = sum(sum_score) / sum(n_games)), by = c('WTeamID', 'Season')]

all_games[wins_t, on = c('WTeamID', 'Season')] %>%
    ggplot(aes(x = rs_ppg, y = tW)) + 
    geom_point() + 
    geom_smooth(method = 'lm') + 
    facet_wrap( ~ as.factor(Season)) + 
    labs(x = 'Regular season average score', y = 'Tournament wins', title = 'Tournament Wins by Regular Season Point per Game')

# Tournament wins by Seed

seeds[, .(Season, WTeamID = TeamID, seed_num = as.numeric(substr(Seed, 2, 3)))][wins_t, on = c('Season', 'WTeamID')] %>%
    ggplot(aes(x = seed_num, y = tW)) + 
    geom_jitter(width = 0.2, height = 0.2) + 
    geom_smooth(method = 'lm') + 
    labs(x = 'Seed', y = 'Tournament Wins', title = 'Tournament Wins by Seed')


