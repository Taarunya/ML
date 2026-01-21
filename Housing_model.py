import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def ensure_dataset():
    if os.path.exists("USA_Housing.csv") and os.path.getsize("USA_Housing.csv") > 50:
        return

    try:
        import kagglehub
    except Exception:
        print("kagglehub not installed.")
        print("Run: python3 -m pip install kagglehub")
        raise

    print("✅ Downloading dataset...")
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
        raise FileNotFoundError("❌ No CSV file found inside downloaded Kaggle dataset folder.")

    shutil.copy(found, "USA_Housing.csv")
    print("✅ Dataset saved as USA_Housing.csv")


ensure_dataset()

df = pd.read_csv("USA_Housing.csv", engine="python")

print(df.head())
print(df.shape)
df.info(verbose=True)
print(df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
print(df.columns)

sns.pairplot(df)
plt.show()

df["Price"].plot.hist(bins=25, figsize=(8, 4))
plt.title("Histogram of House Prices")
plt.xlabel("Price")
plt.show()

df["Price"].plot.density()
plt.title("Density Plot of House Prices")
plt.xlabel("Price")
plt.show()

print(df.corr())

plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, linewidths=2)
plt.title("Correlation Heatmap")
plt.show()

l_column = list(df.columns)
len_feature = len(l_column)

X = df[l_column[0:len_feature - 2]]
y = df[l_column[len_feature - 2]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

lm = LinearRegression()
lm.fit(X_train, y_train)

print(lm.intercept_)
print(lm.coef_)

cdf = pd.DataFrame(data=lm.coef_, index=X_train.columns, columns=["Coefficients"])
print(cdf)

n = X_train.shape[0]
k = X_train.shape[1]
dfN = n - k

train_pred = lm.predict(X_train)
residuals = y_train - train_pred

sigma2 = np.sum(residuals ** 2) / dfN
XTX_inv = np.linalg.inv(np.dot(X_train.T, X_train))
var_b = sigma2 * np.diag(XTX_inv)
std_err = np.sqrt(var_b)

t_stats = lm.coef_ / std_err
cdf["t-statistic"] = t_stats
print(cdf.sort_values(by="t-statistic", ascending=False))

print(round(metrics.r2_score(y_train, train_pred), 3))

predictions = lm.predict(X_test)

plt.figure(figsize=(10, 7))
plt.title("Actual vs Predicted House Prices", fontsize=20)
plt.scatter(y_test, predictions, alpha=0.6)
plt.xlabel("Actual test set house prices", fontsize=14)
plt.ylabel("Predicted house prices", fontsize=14)
plt.show()

residuals_test = y_test - predictions
plt.figure(figsize=(10, 7))
plt.title("Histogram of Residuals", fontsize=20)
plt.hist(residuals_test, bins=30)
plt.xlabel("Residuals", fontsize=14)
plt.show()

plt.figure(figsize=(10, 7))
plt.title("Residuals vs Predicted Values", fontsize=20)
plt.scatter(predictions, residuals_test, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted house prices", fontsize=14)
plt.ylabel("Residuals", fontsize=14)
plt.show()

print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print(round(metrics.r2_score(y_test, predictions), 3))

min_val = np.min(predictions / 6000)
max_val = np.max(predictions / 6000)

print(min_val)
print(max_val)

L = (100 - min_val) / (max_val - min_val)
print(L)
