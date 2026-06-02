import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

premier_league_df = pd.read_csv("premier-league-matches.csv")

print(premier_league_df.shape)
print(premier_league_df.head())
print(premier_league_df.dtypes)
print(premier_league_df.isnull().sum())

premier_league_df["results"] = premier_league_df["FTR"].map({"H": "Home Win", "A": "Away Win", "D": "Draw"})

#print(premier_league_df["results"].value_counts())
#print(premier_league_df["results"].value_counts(normalize=True).round(3))

premier_league_df.describe()

le_temp = LabelEncoder()

premier_league_df["results_encoded"] = le_temp.fit_transform(premier_league_df["results"])
premier_league_df.corr(numeric_only=True)["results_encoded"].sort_values

premier_league_df["HomeGoals"].hist(bins=10)
plt.title('distribution but à domicile')
#plt.show()


premier_league_df["Date"] = pd.to_datetime(premier_league_df["Date"])

premier_league_df["home_goals_avg"] = premier_league_df.groupby("Home")["HomeGoals"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
premier_league_df["away_goals_avg"] = premier_league_df.groupby("Away")["AwayGoals"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

premier_league_df["home_conceded_avg"] = premier_league_df.groupby("Home")["AwayGoals"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
premier_league_df["away_conceded_avg"] = premier_league_df.groupby("Away")["HomeGoals"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

premier_league_df = premier_league_df.dropna()

le_home = LabelEncoder()
le_away = LabelEncoder()
le_res = LabelEncoder()

premier_league_df["home_encoded"] = le_home.fit_transform(premier_league_df["Home"])
premier_league_df["away_encoded"] = le_away.fit_transform(premier_league_df["Away"])
premier_league_df["results_encoded"] = le_res.fit_transform(premier_league_df["results"])

features = ["home_encoded", "away_encoded",
            "home_goals_avg", "away_goals_avg",
            "home_conceded_avg","away_conceded_avg"]

X = premier_league_df[features]
y = premier_league_df["results_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
pred_labels = le_res.inverse_transform(predictions[:10])




print("\n","#"*25, "START", "#"*25,"\n")
print("Voici les label prédit")
print(pred_labels)
print("\n","#"*25, "END", "#"*25)

print("\n","#"*25, "START", "#"*25,"\n")
print("Précision du score du modèle")
print(accuracy_score(y_test, predictions))
print("\n","#"*25, "END", "#"*25)


print("\n","#"*25, "START", "#"*25,"\n")
print("Rapport de classification")
print(classification_report(y_test, predictions, target_names=le_res.classes_))
print("\n","#"*25, "END", "#"*25)

cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Away Win", "Draw", "Home Win"],
            yticklabels=["Away Win", "Draw", "Home Win"])
