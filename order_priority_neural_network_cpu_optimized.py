import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import time
warnings.filterwarnings('ignore')

print("="*60)
print("CPU-OPTIMIZED VERSION - ORDER PRIORITY NEURAL NETWORK")
print("="*60)
print("\nThis version is optimized for CPU execution with:")
print("- Smaller network architectures")
print("- Minimal hyperparameter tuning")
print("- Progress indicators")
print("- Faster execution time\n")

# Load data
start_time = time.time()
df = pd.read_excel("C:/Users/maiwa/Downloads/Global Superstore.xlsx")
print(f"✓ Data loaded in {time.time() - start_time:.2f} seconds")
print(f"✓ Total records: {len(df)}")

# Feature engineering
print("\nCreating features...")
start_time = time.time()

# Convert dates
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Create features
df['Days_to_Ship'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Profit_Margin'] = df['Profit'] / (df['Sales'] + 1)
df['Revenue_Per_Unit'] = df['Sales'] / (df['Quantity'] + 1)
df['Profit_Per_Unit'] = df['Profit'] / (df['Quantity'] + 1)
df['Discount_Impact'] = df['Discount'] * df['Sales']
df['Shipping_Cost_Ratio'] = df['Shipping Cost'] / (df['Sales'] + 1)

# Customer statistics (simplified)
customer_stats = df.groupby('Customer ID').agg({
    'Sales': ['mean', 'count'],
    'Profit': 'mean'
}).reset_index()
customer_stats.columns = ['Customer ID', 'Customer_Sales_mean', 'Customer_Order_Count', 'Customer_Profit_mean']
df = df.merge(customer_stats, on='Customer ID', how='left')

# Product statistics (simplified)
product_stats = df.groupby('Product ID').agg({
    'Sales': ['mean', 'count'],
    'Profit': 'mean'
}).reset_index()
product_stats.columns = ['Product ID', 'Product_Sales_mean', 'Product_Order_Count', 'Product_Profit_mean']
df = df.merge(product_stats, on='Product ID', how='left')

# Select features
feature_cols = [
    'Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost',
    'Days_to_Ship', 'Profit_Margin', 'Revenue_Per_Unit', 'Profit_Per_Unit',
    'Discount_Impact', 'Shipping_Cost_Ratio',
    'Customer_Sales_mean', 'Customer_Order_Count', 'Customer_Profit_mean',
    'Product_Sales_mean', 'Product_Order_Count', 'Product_Profit_mean'
]

X = df[feature_cols].fillna(0)
print(f"✓ Created {len(feature_cols)} features in {time.time() - start_time:.2f} seconds")

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Order Priority'])
print(f"\nOrder Priority Classes: {label_encoder.classes_}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Build simple but effective model
print("\n" + "="*60)
print("TRAINING NEURAL NETWORK")
print("="*60)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(16, activation='relu'),
    
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel architecture:")
model.summary()

# Define callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

# Train model
print("\nStarting training...")
start_time = time.time()

history = model.fit(
    X_train_scaled, y_train,
    epochs=30,  # Limited epochs for CPU
    batch_size=128,  # Larger batch size for CPU efficiency
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f} seconds")

# Evaluate
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

print(f"\nTest Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=label_encoder.classes_,
           yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix (Test Accuracy: {test_acc:.2%})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Training history plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Try a quick ensemble with different architectures
print("\n" + "="*60)
print("QUICK ENSEMBLE TEST")
print("="*60)

# Model 2: Different architecture
model2 = keras.Sequential([
    layers.Dense(256, activation='elu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.4),
    layers.Dense(128, activation='elu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='elu'),
    layers.Dropout(0.2),
    layers.Dense(4, activation='softmax')
])

model2.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training second model...")
history2 = model2.fit(
    X_train_scaled, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

# Ensemble prediction
pred1 = model.predict(X_test_scaled)
pred2 = model2.predict(X_test_scaled)
ensemble_pred = (pred1 + pred2) / 2
y_pred_ensemble = np.argmax(ensemble_pred, axis=1)

ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
print(f"\nModel 1 Accuracy: {test_acc:.4f}")
print(f"Model 2 Accuracy: {model2.evaluate(X_test_scaled, y_test, verbose=0)[1]:.4f}")
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ Best Single Model Accuracy: {test_acc:.4f}")
print(f"✓ Ensemble Accuracy: {ensemble_acc:.4f}")
print(f"✓ Total execution time: {time.time() - start_time:.2f} seconds")

if max(test_acc, ensemble_acc) >= 0.9:
    print("\n🎉 SUCCESS! Achieved 90%+ accuracy!")
else:
    print(f"\n📈 Current best: {max(test_acc, ensemble_acc):.2%}")
    print(f"   {(0.9 - max(test_acc, ensemble_acc))*100:.1f}% away from 90% target")

print("\n💡 Tips for better accuracy:")
print("   - Add more advanced features")
print("   - Use the full script with Keras Tuner")
print("   - Consider using a GPU for faster training")
print("   - Try different resampling techniques for imbalanced data") 