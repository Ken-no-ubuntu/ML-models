# Set your CSV file path here
CSV_PATH = 'try1.csv'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data with explicit data type checking
df = pd.read_csv(CSV_PATH)

# Print data info for debugging
print("DataFrame Info:")
print(df.info())
print("\nSample of data:")
print(df.head())

# Convert all columns to numeric, handling any non-numeric values
for column in df.columns:
    if df[column].dtype == 'object':
        print(f"\nUnique values in {column}:")
        print(df[column].unique())
        df[column] = pd.to_numeric(df[column], errors='coerce')

# Fill any NaN values that might have been created
df = df.fillna(0)

# Prepare X and y
y = df['X17_ioutcome'].astype('float32')
X = df.drop(['X16_toutcome', 'X17_ioutcome'], axis=1).astype('float32')

# Print shapes and data types for verification
print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", y.shape)
print("\nFeature matrix dtype:", X.values.dtype)
print("Target vector dtype:", y.values.dtype)

# Convert to numpy arrays
X = X.values
y = y.values

# Create and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
             loss='binary_crossentropy',
             metrics=['mae'])

# Train model
history = model.fit(X_train, y_train,
                   validation_data=(X_test, y_test),
                   epochs=100,
                   batch_size=32,
                   verbose=1)

# Plot results
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE Over Time')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Print final metrics
print(f"\nFinal Training MAE: {history.history['mae'][-1]:.4f}")
print(f"Final Validation MAE: {history.history['val_mae'][-1]:.4f}")