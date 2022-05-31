#LIMPIAR


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

class FeatureEngineeringDf:
  def __init__(self, df):
    self.df = df
    self.top_leagues = ['Ligue1', 'PremierLeague', 'LaLiga', 'SerieA', 'Bundesliga']
    self.top2_leagues = ['Eredivisie', 'LigaNOS', 'MLS', 'SérieA', 'LigaMXClausura', 'PremierLiga','JupilerProLeague', '']
    self.all_top_leagues = self.top2_leagues + self.top_leagues
    self.top_nations = ['Brazil', 'Spain', 'France', 'Germany', 'Belgium', 'Argentina', 'Italy','England', 'Portugal','Mexico','Netherlands','Denmark']
    self.top2_nations = ['Uruguay', 'Switzerland', 'USA', 'Croatia', 'Colombia', 'Wales', 'Sweden','Senegal', 'Iran','Peru','Japan','Morocco', 'Serbia', 'Poland', 'Chile']
    self.all_top_nations = self.top_nations + self.top2_nations
    self.worst_leagues = []
    self.worst_nations = []
    self.condlist_top_leagues = [
        self.df['league'].isin(self.top_leagues),
        self.df['league'].isin(self.top2_leagues),
        ~self.df['league'].isin(self.all_top_leagues),
    ]
    self.condlist_top_nations = [
        self.df['nation'].isin(self.top_nations),
        self.df['nation'].isin(self.top2_nations),
        ~self.df['nation'].isin(self.all_top_nations),
    ]
    #self.init_dtypes()

  def init_dtypes(self):
    for col in self.df.columns:
      if col != "pollutant":
          if is_string_dtype(self.df[col]):
              print("str", col)
              self.df[col] = self.df[col].astype('category').cat.codes
          elif is_numeric_dtype(self.df[col]):
              print("int", col)
              self.df[col] = self.df[col].astype(np.int64)
    
  def add_goals_value(self):
    choicelist_cup = [
        self.df['goal_cup']*2,
        self.df['goal_cup']*1.5,
        self.df['goal_cup'],
    ]
    self.df['score_goal_cup'] = np.select(self.condlist_top_leagues, choicelist_cup, default=0)
    choicelist_champ = [
        self.df['goal_champ']*2,
        self.df['goal_champ']*1.5,
        self.df['goal_champ'],
    ]
    self.df['score_goal_champ'] = np.select(self.condlist_top_leagues, choicelist_champ, default=0)

    choicelist_champ = [
        self.df['goals_selection']*2,
        self.df['goals_selection']*1.5,
        self.df['goals_selection'],
    ]
    self.df['score_goals_selection'] = np.select(self.condlist_top_nations, choicelist_champ, default=0)
  
  def add_assists_value(self):
    choicelist_cup = [
        self.df['assist_cup']*2,
        self.df['assist_cup']*1.5,
        self.df['assist_cup'],
    ]
    self.df['score_assist_cup'] = np.select(self.condlist_top_leagues, choicelist_cup, default=0)
    choicelist_champ = [
        self.df['assist_champ']*2,
        self.df['assist_champ']*1.5,
        self.df['assist_champ'],
    ]
    self.df['score_assist_champ'] = np.select(self.condlist_top_leagues, choicelist_champ, default=0)

  def add_champions_score(self):
    # dividir continents --> TO DO
      choicelist_champ = [
          self.df['assist_continent']*2,
          self.df['assist_continent']*1.5,
          self.df['assist_continent'],
      ]
      self.df['score_assist_continent'] = np.select(self.condlist_top_nations, choicelist_champ, default=0)

      choicelist_champ = [
          self.df['goal_continent']*2,
          self.df['goal_continent']*1.5,
          self.df['goal_continent'],
      ]
      self.df['score_goal_continent'] = np.select(self.condlist_top_nations, choicelist_champ, default=0)

  
  def add_nations_value(self):
    choicelist_champ = [
        self.df['selections_nation']*2,
        self.df['selections_nation']*1.5,
        self.df['selections_nation'],
    ]
    self.df['score_selections_nation'] = np.select(self.condlist_top_nations, choicelist_champ, default=0)
  
  def filter_by_position(self, filter=0):
    if filter == 0:
      return self.df[self.df.position.isin(['Goalkeeper'])]
    elif filter == 1:
      return self.df[self.df.position.isin([ 'DefensiveMidfield','Defender'])]
    elif filter == 2:
      return self.df[self.df.position.isin(['LeftMidfield' , 'CentralMidfield', 'Midfielder', 'AttackingMidfield','RightMidfield'])]
    else:
      return self.df[self.df.position.isin(['SecondStriker', 'Forward','LeftWinger','RightWinger'])]
  
  def filter_top_leagues(self, filter_positions=True, positions = ['SecondStriker', 'Forward']):
    top_league_players = self.df[self.df.league.isin(self.top_leagues)]
    if filter_positions:
      top_league_players[top_league_players.position.isin(positions)]
    return top_league_players
  
  def data_correlated(self,label_col_name='price'):
      corr = self.df.corr().abs()['price']
      corr = corr[corr != 1]
      corr = corr[corr > 0.05]
      select_columns = list(corr.index)
      #select_columns.remove('Unnamed: 0')
      df_select_columns = self.df[select_columns]
      y = self.df[label_col_name]
      return df_select_columns.copy(), y.copy()