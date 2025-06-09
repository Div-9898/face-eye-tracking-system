import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Adamax, Nadam
import keras_tuner as kt
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

# GPU Configuration
print("="*60)
print("CONFIGURING GPU")
print("="*60)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Optional: Limit GPU memory (uncomment if needed)
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        # )
        
        print(f"✓ GPU Available: {len(gpus)} GPU(s) detected")
        print(f"✓ GPU Details: {[gpu.name for gpu in gpus]}")
        
        # Set TensorFlow to use GPU
        with tf.device('/GPU:0'):
            print("✓ TensorFlow configured to use GPU")
            
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")
else:
    print("⚠ No GPU detected. Running on CPU.")
    print("To use GPU, ensure CUDA and cuDNN are properly installed.")

# Set mixed precision for better GPU performance
try:
    # Only use mixed precision if GPU is available
    if tf.config.list_physical_devices('GPU'):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"✓ Mixed precision policy set: {policy.name}")
    else:
        print("✓ Using default precision policy for CPU")
except Exception as e:
    print(f"Mixed precision not available: {e}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories for saving models
MODEL_DIR = "saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"✓ Created directory for saving models: {MODEL_DIR}")

class AdvancedOrderPriorityPredictor:
    def __init__(self, file_path):
        """Initialize the predictor with advanced analytics"""
        self.df = pd.read_excel("C:/Users/maiwa/Downloads/Global Superstore.xlsx")
        self.label_encoder = LabelEncoder()
        self.scaler = None
        self.best_model = None
        
    def prepare_data(self):
        """Prepare data with basic features"""
        print("="*60)
        print("PREPARING DATA FOR NEURAL NETWORK")
        print("="*60)
        
        # Basic info
        print(f"\nOrder Priority Distribution:")
        print(self.df['Order Priority'].value_counts())
        print(f"\nTotal samples: {len(self.df)}")
        
        # Convert dates
        self.df['Order Date'] = pd.to_datetime(self.df['Order Date'])
        self.df['Ship Date'] = pd.to_datetime(self.df['Ship Date'])
        
        # Basic features for model comparison
        basic_features = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost']
        X_basic = self.df[basic_features]
        y = self.label_encoder.fit_transform(self.df['Order Priority'])
        
        return X_basic, y
    
    def create_advanced_features(self):
        """Create highly advanced features with domain knowledge"""
        print("\n" + "="*60)
        print("ADVANCED FEATURE ENGINEERING")
        print("="*60)
        
        df_feat = self.df.copy()
        
        # 1. Time-based features
        df_feat['Days_to_Ship'] = (df_feat['Ship Date'] - df_feat['Order Date']).dt.days
        df_feat['Order_Month'] = df_feat['Order Date'].dt.month
        df_feat['Order_Quarter'] = df_feat['Order Date'].dt.quarter
        df_feat['Order_DayOfWeek'] = df_feat['Order Date'].dt.dayofweek
        df_feat['Order_Week'] = df_feat['Order Date'].dt.isocalendar().week
        df_feat['Order_DayOfYear'] = df_feat['Order Date'].dt.dayofyear
        df_feat['Is_Weekend'] = (df_feat['Order_DayOfWeek'] >= 5).astype(int)
        df_feat['Is_MonthEnd'] = df_feat['Order Date'].dt.is_month_end.astype(int)
        df_feat['Is_QuarterEnd'] = df_feat['Order Date'].dt.is_quarter_end.astype(int)
        
        # 2. Financial ratios and metrics
        df_feat['Profit_Margin'] = df_feat['Profit'] / (df_feat['Sales'] + 1)
        df_feat['Profit_Margin_Pct'] = df_feat['Profit_Margin'] * 100
        df_feat['Revenue_Per_Unit'] = df_feat['Sales'] / (df_feat['Quantity'] + 1)
        df_feat['Profit_Per_Unit'] = df_feat['Profit'] / (df_feat['Quantity'] + 1)
        df_feat['Discount_Impact'] = df_feat['Discount'] * df_feat['Sales']
        df_feat['Shipping_Cost_Ratio'] = df_feat['Shipping Cost'] / (df_feat['Sales'] + 1)
        df_feat['Effective_Price'] = df_feat['Sales'] * (1 - df_feat['Discount'])
        
        # 3. Statistical transformations
        numeric_cols = ['Sales', 'Profit', 'Quantity', 'Shipping Cost']
        for col in numeric_cols:
            # Log transformations
            df_feat[f'{col}_log'] = np.log1p(df_feat[col].clip(lower=0))
            # Square root transformations
            df_feat[f'{col}_sqrt'] = np.sqrt(df_feat[col].clip(lower=0))
            # Square transformations
            df_feat[f'{col}_squared'] = df_feat[col] ** 2
            # Z-score normalization
            df_feat[f'{col}_zscore'] = (df_feat[col] - df_feat[col].mean()) / (df_feat[col].std() + 1e-6)
            
        # 4. Interaction features
        df_feat['Sales_x_Quantity'] = df_feat['Sales'] * df_feat['Quantity']
        df_feat['Profit_x_Quantity'] = df_feat['Profit'] * df_feat['Quantity']
        df_feat['Discount_x_Quantity'] = df_feat['Discount'] * df_feat['Quantity']
        df_feat['ShipCost_x_Days'] = df_feat['Shipping Cost'] * df_feat['Days_to_Ship']
        df_feat['Sales_per_ShipDay'] = df_feat['Sales'] / (df_feat['Days_to_Ship'] + 1)
        
        # 5. Customer behavior features
        customer_stats = df_feat.groupby('Customer ID').agg({
            'Sales': ['mean', 'sum', 'std', 'count', 'max', 'min'],
            'Profit': ['mean', 'sum', 'std', 'max', 'min'],
            'Quantity': ['mean', 'sum', 'std'],
            'Days_to_Ship': ['mean', 'std'],
            'Order Priority': lambda x: (x == 'Critical').sum()
        }).reset_index()
        customer_stats.columns = ['Customer ID'] + ['Customer_' + '_'.join(col).strip() for col in customer_stats.columns[1:]]
        df_feat = df_feat.merge(customer_stats, on='Customer ID', how='left')
        
        # 6. Product performance features
        product_stats = df_feat.groupby('Product ID').agg({
            'Sales': ['mean', 'sum', 'std', 'count'],
            'Profit': ['mean', 'sum', 'std'],
            'Quantity': ['mean', 'sum'],
            'Discount': ['mean', 'max'],
            'Order Priority': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Medium'
        }).reset_index()
        product_stats.columns = ['Product ID'] + ['Product_' + '_'.join(col).strip() for col in product_stats.columns[1:]]
        
        # Encode product priority mode
        priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        product_stats['Product_Priority_Score'] = product_stats['Product_Order Priority_<lambda>'].map(priority_map)
        product_stats = product_stats.drop('Product_Order Priority_<lambda>', axis=1)
        
        df_feat = df_feat.merge(product_stats, on='Product ID', how='left')
        
        # 7. Geographic features
        region_stats = df_feat.groupby('Region').agg({
            'Sales': ['mean', 'sum', 'std'],
            'Profit': ['mean', 'sum'],
            'Days_to_Ship': 'mean',
            'Order Priority': lambda x: (x == 'Critical').sum() / len(x)
        }).reset_index()
        region_stats.columns = ['Region'] + ['Region_' + '_'.join(col).strip() for col in region_stats.columns[1:]]
        df_feat = df_feat.merge(region_stats, on='Region', how='left')
        
        # 8. Category and sub-category insights
        category_stats = df_feat.groupby(['Category', 'Sub-Category']).agg({
            'Sales': ['mean', 'sum'],
            'Profit': ['mean', 'sum'],
            'Profit_Margin': 'mean'
        }).reset_index()
        category_stats.columns = ['Category', 'Sub-Category'] + ['Cat_SubCat_' + '_'.join(col).strip() for col in category_stats.columns[2:]]
        df_feat = df_feat.merge(category_stats, on=['Category', 'Sub-Category'], how='left')
        
        # 9. Shipping mode efficiency
        ship_mode_map = {'Same Day': 4, 'First Class': 3, 'Second Class': 2, 'Standard Class': 1}
        df_feat['Ship_Mode_Priority'] = df_feat['Ship Mode'].map(ship_mode_map)
        
        # 10. Market segment features
        segment_map = {'Consumer': 1, 'Corporate': 2, 'Home Office': 3}
        df_feat['Segment_Code'] = df_feat['Segment'].map(segment_map)
        
        # 11. Advanced ratios
        df_feat['Profitability_Score'] = (df_feat['Profit'] > 0).astype(int) * df_feat['Profit_Margin']
        df_feat['Value_Density'] = df_feat['Sales'] / (df_feat['Quantity'] * df_feat['Days_to_Ship'] + 1)
        df_feat['Customer_Loyalty_Score'] = df_feat['Customer_Sales_count'] * df_feat['Customer_Profit_mean']
        df_feat['Product_Popularity'] = df_feat['Product_Sales_count'] * df_feat['Product_Sales_mean']
        
        # 12. Binned features for better pattern recognition
        df_feat['Sales_Bin'] = pd.qcut(df_feat['Sales'], q=10, labels=False, duplicates='drop')
        df_feat['Profit_Bin'] = pd.qcut(df_feat['Profit'].rank(method='first'), q=10, labels=False)
        df_feat['Quantity_Bin'] = pd.qcut(df_feat['Quantity'], q=5, labels=False, duplicates='drop')
        
        # 13. Anomaly scores
        for col in ['Sales', 'Profit', 'Quantity']:
            mean = df_feat[col].mean()
            std = df_feat[col].std()
            df_feat[f'{col}_Anomaly'] = np.abs((df_feat[col] - mean) / (std + 1e-6))
        
        # 14. Polynomial features for key metrics
        df_feat['Sales_Profit_Poly'] = df_feat['Sales'] * df_feat['Profit']
        df_feat['Sales_Quantity_Poly'] = df_feat['Sales'] ** 2 * df_feat['Quantity']
        df_feat['Profit_Margin_Poly'] = df_feat['Profit_Margin'] ** 2
        
        # Handle infinities and NaN
        df_feat = df_feat.replace([np.inf, -np.inf], 0)
        df_feat = df_feat.fillna(0)
        
        # Select features for modeling
        feature_cols = [col for col in df_feat.columns if col not in [
            'Order ID', 'Row ID', 'Order Date', 'Ship Date', 'Customer ID', 
            'Customer Name', 'Product ID', 'Product Name', 'Order Priority',
            'Country', 'City', 'State', 'Postal Code', 'Region',
            'Category', 'Sub-Category', 'Ship Mode', 'Segment', 'Market'
        ]]
        
        print(f"Created {len(feature_cols)} advanced features")
        
        # Fit and transform the label encoder
        y = self.label_encoder.fit_transform(df_feat['Order Priority'])
        
        return df_feat[feature_cols], y
    
    def build_nn_model(self, hp):
        """Build neural network with Keras Tuner hyperparameters"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.InputLayer(input_shape=(self.input_dim,)))
        
        # Hidden layers with tunable parameters
        n_layers = hp.Int('n_layers', 2, 4)
        
        for i in range(n_layers):
            model.add(layers.Dense(
                units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                activation=hp.Choice('activation', ['relu', 'elu']),
                kernel_regularizer=regularizers.l2(hp.Float('l2_reg', 1e-5, 1e-3, sampling='log'))
            ))
            
            # Optional batch normalization
            if hp.Boolean(f'batch_norm_{i}'):
                model.add(layers.BatchNormalization())
            
            # Dropout with tunable rate
            if hp.Boolean(f'dropout_{i}'):
                model.add(layers.Dropout(hp.Float(f'dropout_rate_{i}', 0.1, 0.5, step=0.1)))
        
        # Output layer
        model.add(layers.Dense(4, activation='softmax'))
        
        # No need for explicit float32 conversion on CPU
        # Mixed precision is only used on GPU
        
        # Compile with tunable parameters
        optimizer_choice = hp.Choice('optimizer', ['adam', 'adamax', 'nadam'])
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        
        if optimizer_choice == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'adamax':
            optimizer = Adamax(learning_rate=learning_rate)
        else:
            optimizer = Nadam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_with_keras_tuner(self, X_train, X_test, y_train, y_test):
        """Train neural network with Keras Tuner hyperparameter optimization"""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION WITH KERAS TUNER")
        print("="*60)
        
        self.input_dim = X_train.shape[1]
        
        # Create tuner
        tuner = kt.BayesianOptimization(
            self.build_nn_model,
            objective='val_accuracy',
            max_trials=5,  # Reduced from 20 for faster CPU execution
            directory='nn_tuning',
            project_name='order_priority',
            overwrite=True
        )
        
        # Calculate class weights
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Define callbacks
        early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)
        
        # Search for best hyperparameters
        print("Searching for optimal hyperparameters...")
        print("Note: This may take a while on CPU. Consider using a GPU for faster training.")
        print(f"Running 5 trials...")  # Hard-coded the value instead of accessing tuner.max_trials
        
        tuner.search(
            X_train, y_train,
            epochs=20,  # Reduced from 50 for faster execution
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weight_dict,
            verbose=1  # Changed from 0 to show progress
        )
        
        # Get best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        print("\nBest hyperparameters found:")
        print(f"Number of layers: {best_hps.get('n_layers')}")
        print(f"Activation: {best_hps.get('activation')}")
        print(f"Optimizer: {best_hps.get('optimizer')}")
        print(f"Learning rate: {best_hps.get('learning_rate'):.4f}")
        
        # Build and train the best model
        best_model = tuner.hypermodel.build(best_hps)
        
        print("\nTraining best model...")
        
        # Add ModelCheckpoint callback for saving best model during training
        model_checkpoint = ModelCheckpoint(
            os.path.join(MODEL_DIR, 'kt_best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        history = best_model.fit(
            X_train, y_train,
            epochs=40,  # Reduced from 100 for faster CPU execution
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr, model_checkpoint],
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Save the final model
        best_model.save(os.path.join(MODEL_DIR, 'kt_final_model.h5'))
        print(f"\n✓ Keras Tuner model saved to: {os.path.join(MODEL_DIR, 'kt_final_model.h5')}")
        
        # Evaluate
        test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(best_model.predict(X_test), axis=1)
        
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred, 
                                   f"Keras Tuner Optimized Model (Acc: {test_acc:.2%})")
        
        return best_model, test_acc, history
    
    def train_ensemble_neural_networks(self, X_train, X_test, y_train, y_test):
        """Train ensemble of neural networks with different architectures"""
        print("\n" + "="*60)
        print("ENSEMBLE NEURAL NETWORKS")
        print("="*60)
        
        # Apply SMOTE for handling imbalanced data
        print("Applying SMOTE for balanced training...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        models = []
        histories = []
        
        # Architecture 1: Deep network
        print("\n1. Training Deep Neural Network...")
        model1 = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(4, activation='softmax')
        ])
        
        model1.compile(optimizer=Adam(0.001), 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        history1 = model1.fit(X_train_balanced, y_train_balanced,
                             epochs=30, batch_size=64,
                             validation_split=0.2,
                             callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
                                       ReduceLROnPlateau(patience=5)],
                             verbose=1)
        models.append(model1)
        histories.append(history1)
        
        # Architecture 2: Wide network
        print("2. Training Wide Neural Network...")
        model2 = keras.Sequential([
            layers.Dense(512, activation='elu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.4),
            layers.Dense(256, activation='elu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='elu'),
            layers.Dropout(0.2),
            layers.Dense(4, activation='softmax')
        ])
        
        model2.compile(optimizer=Adamax(0.002), 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        history2 = model2.fit(X_train, y_train,
                             epochs=30, batch_size=128,
                             validation_split=0.2,
                             callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
                                       ReduceLROnPlateau(patience=5)],
                             class_weight=dict(enumerate(class_weight.compute_class_weight(
                                 'balanced', classes=np.unique(y_train), y=y_train))),
                             verbose=1)
        models.append(model2)
        histories.append(history2)
        
        # Architecture 3: Residual-like connections
        print("3. Training Residual-style Neural Network...")
        inputs = layers.Input(shape=(X_train.shape[1],))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        residual = x
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(4, activation='softmax')(x)
        
        model3 = keras.Model(inputs=inputs, outputs=outputs)
        model3.compile(optimizer=Nadam(0.001), 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Use SMOTETomek for this model
        smote_tomek = SMOTETomek(random_state=42)
        X_train_st, y_train_st = smote_tomek.fit_resample(X_train, y_train)
        
        history3 = model3.fit(X_train_st, y_train_st,
                             epochs=30, batch_size=96,
                             validation_split=0.2,
                             callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
                                       ReduceLROnPlateau(patience=5)],
                             verbose=1)
        models.append(model3)
        histories.append(history3)
        
        # Evaluate ensemble
        print("\nEvaluating ensemble models...")
        ensemble_predictions = []
        
        for i, model in enumerate(models):
            # Save individual models
            model_path = os.path.join(MODEL_DIR, f'ensemble_model_{i+1}.h5')
            model.save(model_path)
            print(f"✓ Saved ensemble model {i+1} to: {model_path}")
            
            _, acc = model.evaluate(X_test, y_test, verbose=0)
            pred_proba = model.predict(X_test)
            ensemble_predictions.append(pred_proba)
            print(f"Model {i+1} Test Accuracy: {acc:.4f}")
        
        # Average predictions
        ensemble_pred_avg = np.mean(ensemble_predictions, axis=0)
        y_pred_ensemble = np.argmax(ensemble_pred_avg, axis=1)
        
        ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
        print(f"\nEnsemble Test Accuracy: {ensemble_acc:.4f}")
        
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test, y_pred_ensemble,
                                  target_names=self.label_encoder.classes_))
        
        return models, ensemble_acc
    
    def _plot_confusion_matrix(self, y_true, y_pred, title):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "="*70)
        print("ADVANCED NEURAL NETWORK ANALYSIS WITH KERAS TUNER")
        print("="*70)
        
        # Create advanced features
        X_advanced, y = self.create_advanced_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_advanced, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        print("\nScaling features...")
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train with Keras Tuner
        kt_model, kt_acc, kt_history = self.train_with_keras_tuner(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Train ensemble
        ensemble_models, ensemble_acc = self.train_ensemble_neural_networks(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Final results
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Keras Tuner Optimized Model: {kt_acc:.4f}")
        print(f"Ensemble Neural Networks: {ensemble_acc:.4f}")
        print(f"Best Accuracy: {max(kt_acc, ensemble_acc):.4f}")
        
        # Save preprocessing objects
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        print(f"\n✓ Saved scaler to: {os.path.join(MODEL_DIR, 'scaler.pkl')}")
        print(f"✓ Saved label encoder to: {os.path.join(MODEL_DIR, 'label_encoder.pkl')}")
        
        # GPU Memory Summary
        if tf.config.list_physical_devices('GPU'):
            try:
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                print(f"\nGPU Memory Usage:")
                print(f"  Current: {gpu_info['current'] / 1e9:.2f} GB")
                if 'peak' in gpu_info:
                    print(f"  Peak: {gpu_info['peak'] / 1e9:.2f} GB")
            except:
                pass
        
        if max(kt_acc, ensemble_acc) >= 0.9:
            print("\n🎉 SUCCESS! Achieved 90%+ accuracy!")
        else:
            print(f"\n📈 {(0.9 - max(kt_acc, ensemble_acc))*100:.1f}% away from 90% target")
        
        return {
            'keras_tuner_model': kt_model,
            'ensemble_models': ensemble_models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'best_accuracy': max(kt_acc, ensemble_acc)
        }

# Usage
if __name__ == "__main__":
    # Run advanced analysis
    predictor = AdvancedOrderPriorityPredictor("C:/Users/maiwa/Downloads/Global Superstore.xlsx")
    results = predictor.run_complete_analysis()
    
    print(f"\n🎯 Advanced Neural Network Analysis Complete!")
    print(f"📊 Best Accuracy Achieved: {results['best_accuracy']:.2%}")

    # Save results
    model_name = f"order_priority_model_{results['best_accuracy']:.4f}"
    model_path = os.path.join(MODEL_DIR, model_name)
    joblib.dump(results, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Example: How to load and use the saved models
    print("\n" + "="*70)
    print("HOW TO LOAD AND USE SAVED MODELS")
    print("="*70)
    print("""
# Load saved models and preprocessing objects:
from tensorflow import keras
import joblib

# Load preprocessing objects
scaler = joblib.load('saved_models/scaler.pkl')
label_encoder = joblib.load('saved_models/label_encoder.pkl')

# Load Keras Tuner model
kt_model = keras.models.load_model('saved_models/kt_final_model.h5')

# Load ensemble models
ensemble_models = []
for i in range(1, 4):
    model = keras.models.load_model(f'saved_models/ensemble_model_{i}.h5')
    ensemble_models.append(model)

# Make predictions
# 1. Create features for new data
# 2. Scale features: X_scaled = scaler.transform(X_new)
# 3. Predict: predictions = kt_model.predict(X_scaled)
# 4. Decode: labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
""") 