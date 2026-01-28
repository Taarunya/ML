import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


CSV_FILE = "kidney_disease.csv"


# -------------------------
# Dataset loader (Kaggle style)
# -------------------------
def ensure_dataset():
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 50:
        print("✅ Dataset already exists")
        return

    try:
        import kagglehub
    except Exception:
        print("kagglehub not installed.")
        print("Run: pip install kagglehub")
        raise

    print("⬇️ Downloading CKD dataset from Kaggle...")

    path = kagglehub.dataset_download("mansoordaku/ckdisease")

    found = None
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".csv"):
                found = os.path.join(root, f)
                break
        if found:
            break

    if not found:
        raise FileNotFoundError("❌ No CSV file found in Kaggle dataset")

    import shutil
    shutil.copy(found, CSV_FILE)

    print("✅ Dataset saved as kidney_disease.csv")


# -------------------------
# Load dataset
# -------------------------
ensure_dataset()

df = pd.read_csv(CSV_FILE)

print(df.head())
print(df.shape)
df.info()
print(df.describe())
print(df.columns)

# -------------------------
# Cleaning
# -------------------------
df = df.replace("?", np.nan)
df = df.dropna()

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

df_num = df.select_dtypes(include=["int64", "float64"])

# -------------------------
# Choose numeric target
# -------------------------
target = "sc"   # serum creatinine

# -------------------------
# Improved Pairplot (reduced features)
# -------------------------
pairplot_cols = ["age", "bp", "bgr", "bu", "sc", "hemo"]

sns.pairplot(df_num[pairplot_cols])
plt.show()

# -------------------------
# Histogram
# -------------------------
df_num[target].plot.hist(bins=25, figsize=(8, 4))
plt.title("Histogram of Serum Creatinine")
plt.xlabel(target)
plt.show()

# -------------------------
# Density plot
# -------------------------
df_num[target].plot.density()
plt.title("Density Plot of Serum Creatinine")
plt.xlabel(target)
plt.show()

# -------------------------
# Heatmap
# -------------------------
plt.figure(figsize=(10, 7))
sns.heatmap(df_num.corr(), annot=True, linewidths=2)
plt.title("Correlation Heatmap")
plt.show()

# -------------------------
# Model
# -------------------------
X = df_num.drop(columns=[target])
y = df_num[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

lm = LinearRegression()
lm.fit(X_train, y_train)

print("\nIntercept:", lm.intercept_)
print("Coefficients:", lm.coef_)

cdf = pd.DataFrame(data=lm.coef_, index=X_train.columns, columns=["Coefficients"])
print("\nCoefficients Table:")
print(cdf)

# -------------------------
# Training statistics
# -------------------------
train_pred = lm.predict(X_train)
print("\nTrain R2:", round(metrics.r2_score(y_train, train_pred), 3))

# -------------------------
# Test evaluation
# -------------------------
predictions = lm.predict(X_test)

plt.figure(figsize=(10, 7))
plt.title("Actual vs Predicted Serum Creatinine", fontsize=16)
plt.scatter(y_test, predictions, alpha=0.6)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()

residuals_test = y_test - predictions

plt.figure(figsize=(10, 7))
plt.title("Histogram of Residuals", fontsize=16)
plt.hist(residuals_test, bins=30)
plt.show()

plt.figure(figsize=(10, 7))
plt.title("Residuals vs Predicted Values", fontsize=16)
plt.scatter(predictions, residuals_test, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.show()

print("\nMAE:", metrics.mean_absolute_error(y_test, predictions))
print("MSE:", metrics.mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Test R2:", round(metrics.r2_score(y_test, predictions), 3))

print("\n✅ Program finished successfully")
