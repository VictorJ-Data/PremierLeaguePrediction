import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

premier_league_df = pd.read_csv("premier-league-matches.csv")

print(premier_league_df.shape)
print(premier_league_df.head())
print(premier_league_df.dtypes)
print(premier_league_df.isnull().sum())

premier_league_df["results"] = premier_league_df["FTR"].map({"H": "Home Win", "A": "Away Win", "D": "Draw"})

print(premier_league_df["results"].value_counts())
print(premier_league_df["results"].value_counts(normalize=True).round(3))

premier_league_df.describe()

le_temp = LabelEncoder()
premier_league_df["results_encoded"] = le_temp.fit_transform(premier_league_df["results"])
premier_league_df.corr(numeric_only=True)["results_encoded"].sort_values

premier_league_df["HomeGoals"].hist(bins=10)
plt.title('distribution but à domicile')
plt.show()


premier_league_df["Date"] = pd.to_datetime(premier_league_df["Date"])

premier_league_df["home_goals_avg"] = premier_league_df.groupby("Home")["HomeGoals"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
premier_league_df["away_goals_avg"] = premier_league_df.groupby("Away")["AwayGoals"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

premier_league_df["home_conceded_avg"] = premier_league_df.groupby("Home")["AwayGoals"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
premier_league_df["away_conceded_avg"] = premier_league_df.groupby("Away")["HomeGoals"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

premier_league_df = premier_league_df.dropna()

premier_league_df["home_encoded"] = le_temp.fit_transform(premier_league_df["Home"])
premier_league_df["away_encoded"] = le_temp.fit_transform(premier_league_df["Away"])
premier_league_df["results_encoded"] = le_temp.fit_transform(premier_league_df["results"])

print(premier_league_df)
