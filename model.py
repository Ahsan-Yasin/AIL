import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv('Exam_Score_Prediction.csv')

# ---------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------

# A. Drop Irrelevant Columns
# The column '1' acts as a student ID, which is useless for prediction.
if '1' in df.columns:
    df = df.drop('1', axis=1)
    print("-> Dropped irrelevant column '1' (Student ID)")

# B. Remove Duplicates
# Removes rows that are exact copies of each other
df = df.drop_duplicates()
print("-> Removed duplicates")

# C. Remove Empty/Missing Values
# Removes any row that has a missing value (NaN)
df = df.dropna()
print("-> Removed empty rows")

# ---------------------------------------------------------
# 3. PREPROCESSING (Encoding & Standardization)
# ---------------------------------------------------------

# A. Encode Ordinal Data (Order matters: Low < Medium < High)
quality_map = {'poor': 0, 'average': 1, 'good': 2}
difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
facility_map = {'low': 0, 'medium': 1, 'high': 2}
internet_map = {'no': 0, 'yes': 1}

df['sleep_quality'] = df['sleep_quality'].map(quality_map)
df['exam_difficulty'] = df['exam_difficulty'].map(difficulty_map)
df['facility_rating'] = df['facility_rating'].map(facility_map)
df['internet_access'] = df['internet_access'].map(internet_map)

# B. Encode Nominal Data (Order doesn't matter: Male/Female)
df = pd.get_dummies(df, columns=['gender', 'course', 'study_method'], drop_first=True)

# C. Split Data
X = df.drop('exam_score', axis=1) # Inputs
y = df['exam_score']              # Output (Score)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# D. Standardize (Normalize) Values
# This scales values (like age=20 and study_hours=5) to a common range so the model doesn't get confused.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Fit only on training data
X_test = scaler.transform(X_test)       # Transform test data using the same scale

# ---------------------------------------------------------
# 4. TRAIN MODELS & SHOW "ACCURACY"
# ---------------------------------------------------------

print("\n" + "="*30)
print("     MODEL PERFORMANCE")
print("="*30)

# --- Model 1: Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = r2_score(y_test, lr_pred) * 100  # Convert to percentage

print(f"\n1. Linear Regression")
print(f"   Accuracy (R2 Score): {lr_acc:.2f}%")
print(f"   Error (MAE):         {mean_absolute_error(y_test, lr_pred):.2f}")


# --- Model 2: Decision Tree ---
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = r2_score(y_test, dt_pred) * 100

print(f"\n2. Decision Tree")
print(f"   Accuracy (R2 Score): {dt_acc:.2f}%")
print(f"   Error (MAE):         {mean_absolute_error(y_test, dt_pred):.2f}")


# --- Model 3: Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = r2_score(y_test, rf_pred) * 100

print(f"\n3. Random Forest")
print(f"   Accuracy (R2 Score): {rf_acc:.2f}%")
print(f"   Error (MAE):         {mean_absolute_error(y_test, rf_pred):.2f}")

# ---------------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------------

# Graph 1: Accuracy Comparison
plt.figure(figsize=(8, 5))
models = ['Linear Regression', 'Decision Tree', 'Random Forest']
accuracies = [lr_acc, dt_acc, rf_acc]
colors = ['skyblue', 'lightgreen', 'salmon']

plt.bar(models, accuracies, color=colors)
plt.title('Model Accuracy Comparison ($R^2$ Score)')
plt.ylabel('Accuracy %')
plt.ylim(0, 100)
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
plt.show()

# Graph 2: Actual vs Predicted (Random Forest)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_pred, alpha=0.6, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Random Forest: Actual vs Predicted')
plt.legend()
plt.show()