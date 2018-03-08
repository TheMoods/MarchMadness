# MarchMadness

## General Info

https://en.wikipedia.org/wiki/NCAA_Division_I_Men's_Basketball_Tournament

### Kaggle

https://www.kaggle.com/c/mens-machine-learning-competition-2018/data


## Data

### Indexing

Alwyas maintain the following data types for indices when building features:
```
   {
     'Season': int,
     'DayNum': int,
     'Team': str # (refers to any occurance of the team id e.g. `WTeamID`, `team_b`)
   }
```
