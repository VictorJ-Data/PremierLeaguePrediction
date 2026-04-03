import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

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
premier_league_df.corr(numeric_only=True)["results_encoded"]
