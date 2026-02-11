# ==========================================================
# CHRONIC KIDNEY DISEASE – LINEAR REGRESSION PIPELINE
# (Improved version based on USA Housing workflow)
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

CSV_FILE = "kidney_disease.csv"

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------
df = pd.read_csv(CSV_FILE)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Column Names ---")
print(df.columns)

# ----------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

# Convert columns to numeric where possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# Remove rows with missing values
df.dropna(inplace=True)

# Keep only numeric columns
df_num = df.select_dtypes(include=["int64", "float64"])

print("\nClean dataset shape:", df_num.shape)

# ----------------------------------------------------------
# 3. CHOOSE TARGET VARIABLE
# ----------------------------------------------------------
target = "sc"   # Serum Creatinine (kidney health indicator)

# ----------------------------------------------------------
# 4. CORRELATION HEATMAP
# ----------------------------------------------------------
plt.figure(figsize=(10,7))
sns.heatmap(df_num.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------------------------------------
# 5. DISTRIBUTION OF TARGET
# ----------------------------------------------------------
plt.figure(figsize=(8,4))
df_num[target].hist(bins=25)
plt.title("Histogram of Serum Creatinine")
plt.xlabel(target)
plt.show()

plt.figure(figsize=(8,4))
df_num[target].plot.density()
plt.title("Density Plot of Serum Creatinine")
plt.show()

# ----------------------------------------------------------
# 6. FEATURE & TARGET SEPARATION ⭐
# ----------------------------------------------------------
X = df_num.drop(columns=[target])
y = df_num[target]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ----------------------------------------------------------
# 7. TRAIN TEST SPLIT ⭐
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------------------------------------
# 8. FEATURE SCALING ⭐ (VERY IMPORTANT ADDITION)
# ----------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------------
# 9. TRAIN LINEAR REGRESSION MODEL
# ----------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------------------------
# 10. PREDICTIONS
# ----------------------------------------------------------
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# ----------------------------------------------------------
# 11. MODEL EVALUATION ⭐ IMPROVED
# ----------------------------------------------------------
print("\n========== MODEL PERFORMANCE ==========")

print("\nR2 Score (Train):", r2_score(y_train, train_pred))
print("R2 Score (Test):", r2_score(y_test, test_pred))
print("Mean Squared Error:", mean_squared_error(y_test, test_pred))

# ----------------------------------------------------------
# 12. COEFFICIENT INTERPRETATION ⭐
# ----------------------------------------------------------
coef_df = pd.DataFrame(
    model.coef_,
    index=X.columns,
    columns=["Coefficient"]
)

print("\n--- Feature Importance ---")
print(coef_df.sort_values(by="Coefficient", ascending=False))

# ----------------------------------------------------------
# 13. ACTUAL VS PREDICTED PLOT ⭐
# ----------------------------------------------------------
plt.figure(figsize=(7,6))
plt.scatter(y_test, test_pred)
plt.xlabel("Actual Serum Creatinine")
plt.ylabel("Predicted Serum Creatinine")
plt.title("Actual vs Predicted")
plt.show()

# ----------------------------------------------------------
# 14. RESIDUAL ANALYSIS (Model Diagnostics)
# ----------------------------------------------------------
residuals = y_test - test_pred

plt.figure(figsize=(7,6))
plt.hist(residuals, bins=30)
plt.title("Histogram of Residuals")
plt.show()

plt.figure(figsize=(7,6))
plt.scatter(test_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()

print("\n✅ Program Finished Successfully")
