# ==========================================================
# LAB 04 – CKD LOGISTIC REGRESSION WITH SAVED PLOTS
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----------------------------------------------------------
# CREATE OUTPUT FOLDER FOR IMAGES
# ----------------------------------------------------------
OUTPUT_FOLDER = "lab04"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------
df = pd.read_csv("kidney_disease.csv")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

# ----------------------------------------------------------
# DATA CLEANING
# ----------------------------------------------------------
df.replace("?", np.nan, inplace=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

df.dropna(inplace=True)

print("\nClean dataset shape:", df.shape)

# ----------------------------------------------------------
# CONVERT TARGET → CLASSIFICATION
# ----------------------------------------------------------
print("\nUnique class values:", df["classification"].unique())

df["classification"] = df["classification"].map({
    "ckd": 1,
    "notckd": 0
})

# Keep only numeric columns
df_num = df.select_dtypes(include=["int64", "float64"])

# ----------------------------------------------------------
# CORRELATION HEATMAP (SAVE)
# ----------------------------------------------------------
plt.figure(figsize=(10,7))
sns.heatmap(df_num.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")

plt.savefig(os.path.join(OUTPUT_FOLDER, "heatmap.png"))
plt.show()

# ----------------------------------------------------------
# FEATURE & TARGET SPLIT
# ----------------------------------------------------------
X = df_num.drop("classification", axis=1)
y = df_num["classification"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ----------------------------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------------------------------------
# FEATURE SCALING
# ----------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------------
# TRAIN LOGISTIC REGRESSION MODEL
# ----------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------------------------------------
# PREDICTIONS
# ----------------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------------
# MODEL EVALUATION
# ----------------------------------------------------------
print("\n========== MODEL PERFORMANCE ==========")

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------
# CONFUSION MATRIX (SAVE)
# ----------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"))
plt.show()

# ----------------------------------------------------------
# FEATURE IMPORTANCE
# ----------------------------------------------------------
coef_df = pd.DataFrame(
    model.coef_[0],
    index=X.columns,
    columns=["Importance"]
)

print("\n--- Feature Importance ---")
print(coef_df.sort_values(by="Importance", ascending=False))

print(f"\n📁 All plots saved inside '{OUTPUT_FOLDER}' folder")
print("\n✅ Program Finished Successfully")
