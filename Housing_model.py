import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def ensure_dataset():
    if os.path.exists("USA_Housing.csv") and os.path.getsize("USA_Housing.csv") > 50:
        return

    try:
        import kagglehub
    except Exception:
        print("python3 -m pip install kagglehub")
        raise

    path = kagglehub.dataset_download("kanths028/usa-housing")

    found = None
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".csv"):
                found = os.path.join(root, f)
                break
        if found:
            break

    if not found:
        raise FileNotFoundError("No CSV found")

    shutil.copy(found, "USA_Housing.csv")

ensure_dataset()

df = pd.read_csv("USA_Housing.csv")

df = df.drop("Address", axis=1)

print(df.head())
print(df.info())
print(df.describe())

sns.pairplot(df)
plt.show()

df["Price"].hist(bins=25)
plt.show()

df["Price"].plot.density()
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.intercept_)
print(model.coef_)

coef_df = pd.DataFrame(model.coef_, index=X.columns, columns=["Coefficient"])
print(coef_df)

train_pred = model.predict(X_train)
print(round(metrics.r2_score(y_train, train_pred), 3))

test_pred = model.predict(X_test)

print(metrics.mean_absolute_error(y_test, test_pred))
print(metrics.mean_squared_error(y_test, test_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, test_pred)))
print(round(metrics.r2_score(y_test, test_pred), 3))

plt.scatter(y_test, test_pred)
plt.show()

residuals = y_test - test_pred

plt.hist(residuals, bins=30)
plt.show()

plt.scatter(test_pred, residuals)
plt.axhline(0, linestyle="--")
plt.show()
