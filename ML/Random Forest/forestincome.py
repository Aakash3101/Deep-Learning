import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

income_data = pd.read_csv("income.csv", header=0, delimiter=", ", engine='python')
#print(income_data.iloc[0])

income_data["sex-int"] = np.nan
income_data["sex-int"] = income_data["sex"].apply(lambda row: 1 if row == "Female" else 0)
income_data["native-int"] = np.nan
income_data["native-int"] = income_data["native-country"].apply(lambda row: 1 if row == "United-States" else 0)
#print(income_data.iloc[0])

labels = income_data["income"]
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "native-int"]]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels,train_size = 0.7, test_size = 0.3, random_state = 1)

classifier = RandomForestClassifier(random_state=1, n_estimators=500)
classifier.fit(train_data, train_labels)

print(classifier.score(test_data, test_labels))