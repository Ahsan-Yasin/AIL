

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn for processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv('Exam_Score_Prediction.csv')

# ---------------------------------------------------------
# 2. DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------

# A. Drop Irrelevant Columns ('1' is Student ID)
if '1' in df.columns:
    df = df.drop('1', axis=1)

# B. Drop Duplicates and Empty Rows
df = df.drop_duplicates()
df = df.dropna()

# C. Encode Ordinal Variables
quality_map = {'poor': 0, 'average': 1, 'good': 2}
difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
facility_map = {'low': 0, 'medium': 1, 'high': 2}
internet_map = {'no': 0, 'yes': 1}

df['sleep_quality'] = df['sleep_quality'].map(quality_map)
df['exam_difficulty'] = df['exam_difficulty'].map(difficulty_map)
df['facility_rating'] = df['facility_rating'].map(facility_map)
df['internet_access'] = df['internet_access'].map(internet_map)

# D. Encode Nominal Variables (One-Hot Encoding)
df = pd.get_dummies(df, columns=['gender', 'course', 'study_method'], drop_first=True)

# ---------------------------------------------------------
# 3. SPLIT & STANDARDIZE
# ---------------------------------------------------------
X = df.drop('exam_score', axis=1).values # Convert to numpy array for TensorFlow
y = df['exam_score'].values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Normalize/Standardize Data (CRITICAL for Neural Networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Input features shape: {X_train.shape}")

# ---------------------------------------------------------
# 4. BUILD NEURAL NETWORK (TensorFlow)
# ---------------------------------------------------------

# Define the model architecture
model = Sequential([
    # Input Layer & 1st Hidden Layer (64 Neurons, ReLU activation)
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    
    # 2nd Hidden Layer (32 Neurons, ReLU activation)
    Dense(32, activation='relu'),
    
    # Output Layer (1 Neuron for the Exam Score, Linear activation)
    Dense(1, activation='linear') 
])

# Compile the model
# Optimizer: Adam (Standard for most problems)
# Loss: Mean Squared Error (Standard for Regression)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# ---------------------------------------------------------
# 5. TRAIN THE MODEL
# ---------------------------------------------------------
print("\nTraining Neural Network...")
history = model.fit(
    X_train, y_train,
    epochs=100,          # How many times to go through the data
    batch_size=32,       # Update weights after every 32 rows
    validation_split=0.2, # Use 20% of training data to validate while training
    verbose=0            # Hides the long log output
)
print("Training Complete.")

# ---------------------------------------------------------
# 6. EVALUATE & PREDICT
# ---------------------------------------------------------
y_pred = model.predict(X_test)

# Calculate Accuracy (R2 Score)
r2_acc = r2_score(y_test, y_pred) * 100
mae_err = mean_absolute_error(y_test, y_pred)

print("\n" + "="*30)
print(f" NEURAL NETWORK RESULTS")
print("="*30)
print(f"Accuracy (R2 Score): {r2_acc:.2f}%")
print(f"Mean Absolute Error: {mae_err:.2f}")

# ---------------------------------------------------------
# 7. VISUALIZATION
# ---------------------------------------------------------

# Graph 1: Loss Curve (Did the model learn?)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Learning Curve (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()

# Graph 2: Actual vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.6, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Perfect prediction line
plt.title(f'Actual vs Predicted (Acc: {r2_acc:.1f}%)')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')

plt.tight_layout()
plt.show() 