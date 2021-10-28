import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("us_insurance_costs.csv")
df.columns = df.columns.str.replace("smoker", "smoking_status")

# Convert categorical values to numerical
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df["smoking_status"] = le.fit_transform(df["smoking_status"])
df["region"] = le.fit_transform(df["region"])

# Feature selection
x_t = df[["age", "bmi", "smoking_status", "children", "region"]]
y_t = df["charges"]
best_features = SelectKBest(score_func=f_regression, k=3)
fit = best_features.fit(x_t, y_t)
df_scores = pd.DataFrame(best_features.scores_)
df_columns = pd.DataFrame(x_t.columns)
feature_scores = pd.concat([df_columns, df_scores], axis=1)
feature_scores.columns = ["feature_name", "score"]
print(feature_scores.nlargest(20, "score"))

# Features
x = df[["age", "bmi", "smoking_status"]]

# Target variable
y = df["charges"]

# Using LazyRegressor to find the best performing model
offset = int(x.shape[0] * 0.9)
x_train, y_train = x[:offset], y[:offset]
x_test, y_test = x[offset:], y[offset:]
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(x_train, x_test, y_train, y_test)
print(models)

# Splitting dataset into training set and test set
from sklearn.metrics import mean_absolute_error
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
params = {"n_estimators": 500,
          "max_depth": 4,
          "min_samples_split": 5,
          "learning_rate": 0.01,
          "loss": "ls"}

# Fitting the regression model
model = ensemble.GradientBoostingRegressor(**params)
model.fit(x_train, y_train)
y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)
mae = mean_absolute_error(y_test, model.predict(x_test))
print("The mean absolute error on the test set: {:.4f}".format(mae))

# Plotting the deviance - training set vs test set
test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(model.staged_predict(x_test)):
    test_score[i] = model.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params["n_estimators"]) + 1, model.train_score_, "b-",
         label="Training Set Deviance")
plt.plot(np.arange(params["n_estimators"]) + 1, test_score, "r-",
         label="Test Set Deviance")
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()

import pickle
pickl = {"model": model}
pickle.dump(pickl, open("model_file" + ".p", "wb"))

file_name = "model_file.p"
with open(file_name, "rb") as pickled:
    data = pickle.load(pickled)
    model = data["model"]

model.predict(x_test.iloc[1, :].values.reshape(1, -1))

list(x_test.iloc[1, :])