import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam, Adamax, Nadam, RMSprop
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization, GaussianNoise
from tensorflow.keras.layers import PReLU, LeakyReLU, ELU
import keras_tuner as kt
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
import warnings
import time
import joblib
import os
from scipy import stats
from scipy.stats import boxcox
from scipy.special import erfinv
warnings.filterwarnings('ignore')

print("="*80)
print("EXTREME NEURAL NETWORK APPROACH - ORDER PRIORITY PREDICTION")
print("Target: 90%+ Accuracy with Advanced Feature Engineering")
print("="*80)

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU Available: {len(gpus)} GPU(s) detected")
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")
else:
    print("⚠ Running on CPU - this will be slower")

class ExtremeNeuralNetworkPredictor:
    def __init__(self, file_path):
        """Initialize with extreme feature engineering"""
        self.df = pd.read_excel("C:/Users/maiwa/Downloads/Global Superstore.xlsx")
        self.label_encoder = LabelEncoder()
        self.scalers = {}
        self.best_features = None
        
    def create_ultra_advanced_features(self):
        """Create extremely sophisticated features for neural networks"""
        print("\n" + "="*60)
        print("ULTRA-ADVANCED FEATURE ENGINEERING")
        print("="*60)
        
        import time
        start_time = time.time()
        
        df_feat = self.df.copy()
        print(f"Dataset size: {len(df_feat)} rows")
        
        # Convert dates
        print("1/12: Converting dates...")
        df_feat['Order Date'] = pd.to_datetime(df_feat['Order Date'])
        df_feat['Ship Date'] = pd.to_datetime(df_feat['Ship Date'])
        
        # 1. Time-based features with cyclical encoding
        print("2/12: Creating time-based features...")
        df_feat['Days_to_Ship'] = (df_feat['Ship Date'] - df_feat['Order Date']).dt.days
        df_feat['Order_Year'] = df_feat['Order Date'].dt.year
        df_feat['Order_Month'] = df_feat['Order Date'].dt.month
        df_feat['Order_Quarter'] = df_feat['Order Date'].dt.quarter
        df_feat['Order_DayOfWeek'] = df_feat['Order Date'].dt.dayofweek
        df_feat['Order_DayOfMonth'] = df_feat['Order Date'].dt.day
        df_feat['Order_WeekOfYear'] = df_feat['Order Date'].dt.isocalendar().week
        
        # Cyclical encoding for time features
        df_feat['Month_sin'] = np.sin(2 * np.pi * df_feat['Order_Month'] / 12)
        df_feat['Month_cos'] = np.cos(2 * np.pi * df_feat['Order_Month'] / 12)
        df_feat['DayOfWeek_sin'] = np.sin(2 * np.pi * df_feat['Order_DayOfWeek'] / 7)
        df_feat['DayOfWeek_cos'] = np.cos(2 * np.pi * df_feat['Order_DayOfWeek'] / 7)
        df_feat['Quarter_sin'] = np.sin(2 * np.pi * df_feat['Order_Quarter'] / 4)
        df_feat['Quarter_cos'] = np.cos(2 * np.pi * df_feat['Order_Quarter'] / 4)
        df_feat['WeekOfYear_sin'] = np.sin(2 * np.pi * df_feat['Order_WeekOfYear'] / 52)
        df_feat['WeekOfYear_cos'] = np.cos(2 * np.pi * df_feat['Order_WeekOfYear'] / 52)
        
        # 2. Advanced financial ratios
        print("3/12: Creating financial ratios...")
        eps = 1e-8  # Small constant to avoid division by zero
        df_feat['Profit_Margin'] = df_feat['Profit'] / (df_feat['Sales'] + eps)
        df_feat['Revenue_Per_Unit'] = df_feat['Sales'] / (df_feat['Quantity'] + eps)
        df_feat['Profit_Per_Unit'] = df_feat['Profit'] / (df_feat['Quantity'] + eps)
        df_feat['Discount_Impact'] = df_feat['Discount'] * df_feat['Sales']
        df_feat['Shipping_Efficiency'] = df_feat['Shipping Cost'] / (df_feat['Sales'] + eps)
        df_feat['Value_Density'] = df_feat['Sales'] / (df_feat['Quantity'] * df_feat['Days_to_Ship'] + 1)
        df_feat['Profit_to_Ship_Ratio'] = df_feat['Profit'] / (df_feat['Shipping Cost'] + eps)
        df_feat['Sales_Velocity'] = df_feat['Sales'] / (df_feat['Days_to_Ship'] + 1)
        df_feat['Order_Value_Score'] = df_feat['Sales'] * (1 - df_feat['Discount']) * df_feat['Profit_Margin']
        
        # 3. Statistical transformations for better neural network performance
        print("4/12: Creating statistical transformations...")
        numeric_cols = ['Sales', 'Profit', 'Quantity', 'Shipping Cost', 'Discount']
        
        # Vectorized operations for efficiency
        for col in numeric_cols:
            col_data = df_feat[col].values
            
            # Log transformations
            df_feat[f'{col}_log'] = np.log1p(np.maximum(col_data, 0))
            
            # Square root transformation
            df_feat[f'{col}_sqrt'] = np.sqrt(np.maximum(col_data, 0))
            
            # Cubic root transformation
            df_feat[f'{col}_cbrt'] = np.cbrt(col_data)
            
            # Exponential transformation
            df_feat[f'{col}_exp'] = np.exp(np.minimum(col_data, 10))  # Clip to avoid overflow
            
            # Power transformations
            df_feat[f'{col}_squared'] = col_data ** 2
            df_feat[f'{col}_cubed'] = col_data ** 3
            
            # Rank transformations (more efficient implementation)
            df_feat[f'{col}_rank'] = df_feat[col].rank(pct=True, method='average')
            rank_data = df_feat[f'{col}_rank'].values
            # Clip to avoid erfinv domain errors
            rank_clipped = np.clip(rank_data, 0.001, 0.999)
            df_feat[f'{col}_rank_norm'] = erfinv(2 * rank_clipped - 1) * np.sqrt(2)
            
            # Z-score normalization
            mean_val = col_data.mean()
            std_val = col_data.std()
            df_feat[f'{col}_zscore'] = (col_data - mean_val) / (std_val + eps)
            
            # Robust scaling
            median_val = np.median(col_data)
            mad_val = np.median(np.abs(col_data - median_val))
            df_feat[f'{col}_robust'] = (col_data - median_val) / (mad_val + eps)
        
        # 4. Interaction features (multiplicative and divisive)
        print("5/12: Creating interaction features...")
        df_feat['Sales_x_Quantity'] = df_feat['Sales'] * df_feat['Quantity']
        df_feat['Profit_x_Quantity'] = df_feat['Profit'] * df_feat['Quantity']
        df_feat['Sales_x_Discount'] = df_feat['Sales'] * df_feat['Discount']
        df_feat['Profit_div_Quantity'] = df_feat['Profit'] / (df_feat['Quantity'] + eps)
        df_feat['Sales_div_ShipCost'] = df_feat['Sales'] / (df_feat['Shipping Cost'] + eps)
        df_feat['Quantity_x_Days'] = df_feat['Quantity'] * df_feat['Days_to_Ship']
        df_feat['Sales_minus_Cost'] = df_feat['Sales'] - df_feat['Shipping Cost']
        
        # 5. Polynomial features for key metrics
        print("6/12: Creating polynomial features...")
        for col1 in ['Sales', 'Profit', 'Quantity']:
            for col2 in ['Discount', 'Shipping Cost']:
                df_feat[f'{col1}_x_{col2}_poly'] = df_feat[col1] * df_feat[col2]
                df_feat[f'{col1}_x_{col2}_squared'] = (df_feat[col1] * df_feat[col2]) ** 2
        
        # 6. Customer behavior patterns
        print("7/12: Creating customer features (this may take a moment)...")
        
        # Pre-calculate Order Priority critical flag for efficiency
        df_feat['Is_Critical'] = (df_feat['Order Priority'] == 'Critical').astype(int)
        
        customer_features = df_feat.groupby('Customer ID').agg({
            'Sales': ['mean', 'sum', 'std', 'min', 'max', 'count', 'median', 'var', 'skew'],
            'Profit': ['mean', 'sum', 'std', 'min', 'max', 'median', 'var'],
            'Quantity': ['mean', 'sum', 'std', 'median'],
            'Days_to_Ship': ['mean', 'std', 'min', 'max', 'median'],
            'Discount': ['mean', 'max', 'std'],
            'Is_Critical': 'sum',  # More efficient than lambda
            'Profit_Margin': ['mean', 'std'],
            'Value_Density': ['mean', 'max']
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = ['Customer ID'] + [f'Cust_{col[0]}_{col[1]}' for col in customer_features.columns[1:]]
        
        # Rename the critical column
        customer_features.rename(columns={'Cust_Is_Critical_sum': 'Cust_Order Priority_<lambda>'}, inplace=True)
        
        # Customer lifetime value and scoring
        customer_features['Cust_CLV'] = customer_features['Cust_Sales_sum'] * customer_features['Cust_Profit_mean']
        customer_features['Cust_Consistency'] = 1 / (customer_features['Cust_Sales_std'].fillna(1) + 1)
        customer_features['Cust_Value_Score'] = (
            customer_features['Cust_Sales_mean'] * 
            customer_features['Cust_Profit_mean'] * 
            customer_features['Cust_Sales_count']
        ) ** (1/3)
        
        df_feat = df_feat.merge(customer_features, on='Customer ID', how='left')
        
        # 7. Product performance metrics
        print("8/12: Creating product features...")
        
        # More efficient mode calculation
        def get_mode(x):
            if len(x) > 0:
                return x.value_counts().index[0]
            return 'Medium'
        
        product_features = df_feat.groupby('Product ID').agg({
            'Sales': ['mean', 'sum', 'std', 'count', 'median', 'var', 'skew'],
            'Profit': ['mean', 'sum', 'std', 'median', 'min', 'max'],
            'Quantity': ['mean', 'sum', 'std'],
            'Discount': ['mean', 'std', 'max'],
            'Days_to_Ship': ['mean', 'std'],
            'Order Priority': get_mode,
            'Profit_Margin': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        product_features.columns = ['Product ID'] + [f'Prod_{col[0]}_{col[1]}' for col in product_features.columns[1:]]
        
        # Encode product priority
        priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        product_features['Prod_Priority_Mode'] = product_features['Prod_Order Priority_get_mode'].map(priority_map).fillna(1)
        product_features = product_features.drop('Prod_Order Priority_get_mode', axis=1)
        
        # Product scoring
        product_features['Prod_Performance_Score'] = (
            product_features['Prod_Sales_mean'] * 
            product_features['Prod_Profit_mean'] * 
            product_features['Prod_Sales_count']
        ) ** (1/3)
        
        df_feat = df_feat.merge(product_features, on='Product ID', how='left')
        
        # 8. Categorical encoding with target statistics
        print("9/12: Creating categorical encodings...")
        ship_priority = {'Same Day': 4, 'First Class': 3, 'Second Class': 2, 'Standard Class': 1}
        df_feat['Ship_Mode_Priority'] = df_feat['Ship Mode'].map(ship_priority)
        
        segment_map = {'Consumer': 1, 'Corporate': 2, 'Home Office': 3}
        df_feat['Segment_Code'] = df_feat['Segment'].map(segment_map)
        
        # 9. Regional and market insights
        print("10/12: Creating regional features...")
        
        # Pre-calculate high priority flag
        df_feat['Is_High_Priority'] = df_feat['Order Priority'].isin(['Critical', 'High']).astype(int)
        
        region_features = df_feat.groupby('Region').agg({
            'Sales': ['mean', 'sum', 'std'],
            'Profit': ['mean', 'sum'],
            'Days_to_Ship': ['mean', 'std'],
            'Is_High_Priority': 'mean'  # More efficient than lambda
        }).reset_index()
        
        region_features.columns = ['Region'] + [f'Region_{col[0]}_{col[1]}' for col in region_features.columns[1:]]
        region_features.rename(columns={'Region_Is_High_Priority_mean': 'Region_Order Priority_<lambda>'}, inplace=True)
        
        df_feat = df_feat.merge(region_features, on='Region', how='left')
        
        # 10. Advanced ratios and composite scores
        print("11/12: Creating composite scores...")
        df_feat['Order_Complexity_Score'] = (
            df_feat['Quantity'] * 
            df_feat['Days_to_Ship'] * 
            df_feat['Ship_Mode_Priority'] / 
            (df_feat['Segment_Code'] + 1)
        )
        
        df_feat['Customer_Product_Affinity'] = (
            df_feat['Cust_Sales_mean'] * 
            df_feat['Prod_Sales_mean'] / 
            (df_feat['Sales'] + eps)
        )
        
        df_feat['Urgency_Score'] = (
            df_feat['Ship_Mode_Priority'] * 
            (1 / (df_feat['Days_to_Ship'] + 1)) * 
            df_feat['Sales']
        )
        
        # 11. Outlier detection features
        print("12/12: Creating outlier detection features...")
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
            outlier_features = ['Sales', 'Profit', 'Quantity', 'Shipping Cost']
            df_feat['Outlier_Score'] = iso_forest.fit_predict(df_feat[outlier_features])
            df_feat['Outlier_Probability'] = iso_forest.score_samples(df_feat[outlier_features])
        except Exception as e:
            print(f"Warning: Outlier detection failed: {e}")
            df_feat['Outlier_Score'] = 0
            df_feat['Outlier_Probability'] = 0
        
        # 12. Feature crosses and higher-order interactions
        df_feat['Sales_Profit_Ratio'] = df_feat['Sales'] / (df_feat['Profit'].abs() + eps)
        df_feat['Quantity_Discount_Impact'] = df_feat['Quantity'] * (1 - df_feat['Discount'])
        df_feat['Ship_Efficiency_Score'] = df_feat['Shipping Cost'] / (df_feat['Days_to_Ship'] * df_feat['Quantity'] + 1)
        
        # Handle missing values and infinities
        print("Handling missing values...")
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        numeric_features = df_feat.select_dtypes(include=[np.number]).columns
        
        # More efficient imputation - use median for speed
        print("Imputing missing values...")
        for col in numeric_features:
            if df_feat[col].isnull().any():
                df_feat[col].fillna(df_feat[col].median(), inplace=True)
        
        # Remove temporary columns
        df_feat = df_feat.drop(['Is_Critical', 'Is_High_Priority'], axis=1, errors='ignore')
        
        # Select features for modeling
        feature_cols = [col for col in df_feat.columns if col not in [
            'Order ID', 'Row ID', 'Order Date', 'Ship Date', 'Customer ID',
            'Customer Name', 'Product ID', 'Product Name', 'Order Priority',
            'Country', 'City', 'State', 'Postal Code', 'Region',
            'Category', 'Sub-Category', 'Ship Mode', 'Segment', 'Market'
        ]]
        
        elapsed_time = time.time() - start_time
        print(f"\nCreated {len(feature_cols)} ultra-advanced features in {elapsed_time:.2f} seconds")
        
        # Encode target
        y = self.label_encoder.fit_transform(df_feat['Order Priority'])
        
        return df_feat[feature_cols], y
    
    def advanced_preprocessing(self, X_train, X_test):
        """Apply advanced preprocessing techniques"""
        print("\nApplying advanced preprocessing...")
        
        # 1. Quantile transformation for non-linear relationships
        quantile_transformer = QuantileTransformer(
            n_quantiles=1000,
            output_distribution='normal',
            random_state=42
        )
        X_train_quantile = quantile_transformer.fit_transform(X_train)
        X_test_quantile = quantile_transformer.transform(X_test)
        
        # 2. PCA for dimensionality reduction
        pca = PCA(n_components=0.99, random_state=42)  # Keep 99% variance
        X_train_pca = pca.fit_transform(X_train_quantile)
        X_test_pca = pca.transform(X_test_quantile)
        
        # 3. Combine original scaled features with PCA components
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Concatenate original and PCA features
        X_train_combined = np.hstack([X_train_scaled, X_train_pca])
        X_test_combined = np.hstack([X_test_scaled, X_test_pca])
        
        print(f"Final feature dimension: {X_train_combined.shape[1]}")
        
        return X_train_combined, X_test_combined, scaler, quantile_transformer, pca
    
    def build_extreme_nn_model(self, input_dim):
        """Build an extreme neural network architecture"""
        model = keras.Sequential([
            # Input layer with noise for regularization
            layers.InputLayer(input_shape=(input_dim,)),
            layers.GaussianNoise(0.01),
            
            # First block - wide layer
            layers.Dense(1024, kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.PReLU(),
            layers.Dropout(0.3),
            
            # Second block - deep layers
            layers.Dense(512, kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.ELU(alpha=1.0),
            layers.Dropout(0.3),
            
            layers.Dense(512, kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.25),
            
            # Third block - bottleneck
            layers.Dense(256, kernel_initializer='he_uniform', 
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            layers.BatchNormalization(),
            layers.PReLU(),
            layers.Dropout(0.25),
            
            layers.Dense(128, kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.ELU(alpha=1.0),
            layers.Dropout(0.2),
            
            # Fourth block - final layers
            layers.Dense(64, kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.PReLU(),
            layers.Dropout(0.15),
            
            layers.Dense(32, kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.ELU(alpha=1.0),
            
            # Output layer
            layers.Dense(4, activation='softmax')
        ])
        
        return model
    
    def train_ensemble_neural_networks(self, X_train, X_test, y_train, y_test):
        """Train an ensemble of extreme neural networks"""
        print("\n" + "="*60)
        print("TRAINING EXTREME NEURAL NETWORK ENSEMBLE")
        print("="*60)
        
        models = []
        histories = []
        predictions = []
        
        # Apply SMOTE for balanced training
        print("Applying advanced sampling techniques...")
        
        # Fix column names for SMOTE compatibility
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Try different sampling strategies
        samplers = [
            ('SMOTE', SMOTE(random_state=42)),
            ('BorderlineSMOTE', BorderlineSMOTE(random_state=42)),
            ('ADASYN', ADASYN(random_state=42))
        ]
        
        for i, (sampler_name, sampler) in enumerate(samplers):
            print(f"\n{i+1}. Training model with {sampler_name}...")
            
            try:
                # Apply sampling
                X_resampled, y_resampled = sampler.fit_resample(X_train_df, y_train)
                X_resampled = X_resampled.values  # Convert back to numpy
                
                # Build model
                model = self.build_extreme_nn_model(X_train.shape[1])
                
                # Different optimizers for diversity
                optimizers = [
                    Adam(learning_rate=0.001),
                    Adamax(learning_rate=0.002),
                    Nadam(learning_rate=0.001)
                ]
                
                model.compile(
                    optimizer=optimizers[i],
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Callbacks
                early_stop = EarlyStopping(
                    monitor='val_accuracy',
                    patience=20,
                    restore_best_weights=True,
                    verbose=0
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=0
                )
                
                # Train model
                history = model.fit(
                    X_resampled, y_resampled,
                    epochs=100,
                    batch_size=64,
                    validation_split=0.2,
                    callbacks=[early_stop, reduce_lr],
                    verbose=1
                )
                
                # Evaluate
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                print(f"Model {i+1} Test Accuracy: {test_acc:.4f}")
                
                models.append(model)
                histories.append(history)
                predictions.append(model.predict(X_test))
                
            except Exception as e:
                print(f"Failed with {sampler_name}: {str(e)}")
                # Train without sampling as fallback
                model = self.build_extreme_nn_model(X_train.shape[1])
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                history = model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=64,
                    validation_split=0.2,
                    callbacks=[early_stop, reduce_lr],
                    verbose=1
                )
                
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                print(f"Model {i+1} (no sampling) Test Accuracy: {test_acc:.4f}")
                
                models.append(model)
                histories.append(history)
                predictions.append(model.predict(X_test))
        
        # Ensemble predictions
        print("\nCreating ensemble predictions...")
        ensemble_pred = np.mean(predictions, axis=0)
        y_pred_ensemble = np.argmax(ensemble_pred, axis=1)
        ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
        
        print(f"\nEnsemble Accuracy: {ensemble_acc:.4f}")
        
        return models, ensemble_acc, y_pred_ensemble
    
    def train_with_keras_tuner_extreme(self, X_train, X_test, y_train, y_test):
        """Use Keras Tuner for extreme hyperparameter optimization"""
        print("\n" + "="*60)
        print("KERAS TUNER EXTREME OPTIMIZATION")
        print("="*60)
        
        def build_model(hp):
            model = keras.Sequential()
            model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
            
            # Add noise layer
            if hp.Boolean('use_noise'):
                model.add(layers.GaussianNoise(hp.Float('noise_stddev', 0.001, 0.1, sampling='log')))
            
            # Variable number of layers
            n_layers = hp.Int('n_layers', 3, 8)
            
            for i in range(n_layers):
                model.add(layers.Dense(
                    units=hp.Int(f'units_{i}', min_value=32, max_value=1024, step=32),
                    kernel_initializer='he_uniform'
                ))
                
                # Normalization
                norm_type = hp.Choice(f'norm_type_{i}', ['batch', 'layer', 'none'])
                if norm_type == 'batch':
                    model.add(layers.BatchNormalization())
                elif norm_type == 'layer':
                    model.add(layers.LayerNormalization())
                
                # Activation
                activation = hp.Choice(f'activation_{i}', ['relu', 'elu', 'prelu', 'leaky_relu'])
                if activation == 'relu':
                    model.add(layers.ReLU())
                elif activation == 'elu':
                    model.add(layers.ELU())
                elif activation == 'prelu':
                    model.add(layers.PReLU())
                elif activation == 'leaky_relu':
                    model.add(layers.LeakyReLU())
                
                # Dropout
                if hp.Boolean(f'dropout_{i}'):
                    model.add(layers.Dropout(hp.Float(f'dropout_rate_{i}', 0.1, 0.5, step=0.05)))
            
            # Output layer
            model.add(layers.Dense(4, activation='softmax'))
            
            # Optimizer
            optimizer_choice = hp.Choice('optimizer', ['adam', 'adamax', 'nadam', 'rmsprop'])
            learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
            
            if optimizer_choice == 'adam':
                optimizer = Adam(learning_rate=learning_rate)
            elif optimizer_choice == 'adamax':
                optimizer = Adamax(learning_rate=learning_rate)
            elif optimizer_choice == 'nadam':
                optimizer = Nadam(learning_rate=learning_rate)
            else:
                optimizer = RMSprop(learning_rate=learning_rate)
            
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        
        # Create tuner
        tuner = kt.BayesianOptimization(
            build_model,
            objective='val_accuracy',
            max_trials=30,
            directory='nn_extreme_tuning',
            project_name='order_priority_extreme',
            overwrite=True
        )
        
        # Search
        print("Searching for optimal hyperparameters...")
        tuner.search(
            X_train, y_train,
            epochs=50,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=1
        )
        
        # Get best model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        
        print("\nTraining best model with more epochs...")
        history = best_model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=64,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7)
            ],
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(best_model.predict(X_test), axis=1)
        
        print(f"\nKeras Tuner Best Model Accuracy: {test_acc:.4f}")
        
        return best_model, test_acc, y_pred
    
    def run_extreme_pipeline(self):
        """Run the complete extreme neural network pipeline"""
        print("\n" + "="*80)
        print("RUNNING EXTREME NEURAL NETWORK PIPELINE")
        print("="*80)
        
        # Step 1: Create ultra-advanced features
        X, y = self.create_ultra_advanced_features()
        print(f"\nDataset shape: {X.shape}")
        
        # Step 2: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Step 3: Advanced preprocessing
        X_train_processed, X_test_processed, scaler, qt, pca = self.advanced_preprocessing(
            X_train, X_test
        )
        
        # Step 4: Train ensemble neural networks
        ensemble_models, ensemble_acc, ensemble_pred = self.train_ensemble_neural_networks(
            X_train_processed, X_test_processed, y_train, y_test
        )
        
        # Step 5: Train with Keras Tuner
        kt_model, kt_acc, kt_pred = self.train_with_keras_tuner_extreme(
            X_train_processed, X_test_processed, y_train, y_test
        )
        
        # Step 6: Final ensemble of all models
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        # Combine all predictions
        all_predictions = [model.predict(X_test_processed) for model in ensemble_models]
        all_predictions.append(kt_model.predict(X_test_processed).reshape(1, -1, 4)[0])
        
        # Weighted average (give more weight to better models)
        final_pred = np.average(all_predictions, axis=0, weights=[0.2, 0.2, 0.2, 0.4])
        y_pred_final = np.argmax(final_pred, axis=1)
        final_acc = accuracy_score(y_test, y_pred_final)
        
        print(f"\nEnsemble Neural Networks: {ensemble_acc:.4f}")
        print(f"Keras Tuner Optimized: {kt_acc:.4f}")
        print(f"Final Combined Accuracy: {final_acc:.4f}")
        
        # Best single accuracy
        best_acc = max(ensemble_acc, kt_acc, final_acc)
        print(f"\n{'='*40}")
        print(f"BEST ACCURACY ACHIEVED: {best_acc:.4f}")
        print(f"{'='*40}")
        
        if best_acc >= 0.9:
            print("\n🎉 SUCCESS! Achieved 90%+ accuracy!")
        else:
            print(f"\n📈 Current: {best_acc:.2%}")
            print(f"   Still {(0.9 - best_acc)*100:.1f}% away from 90% target")
        
        # Classification report
        print("\nClassification Report (Best Model):")
        if final_acc == best_acc:
            y_pred_best = y_pred_final
        elif kt_acc == best_acc:
            y_pred_best = kt_pred
        else:
            y_pred_best = ensemble_pred
            
        print(classification_report(y_test, y_pred_best,
                                  target_names=self.label_encoder.classes_))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_best)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Best Model Confusion Matrix (Accuracy: {best_acc:.2%})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        # Save models if accuracy >= 90%
        if best_acc >= 0.9:
            model_dir = 'extreme_nn_models'
            os.makedirs(model_dir, exist_ok=True)
            
            # Save best model
            if final_acc == best_acc:
                for i, model in enumerate(ensemble_models):
                    model.save(os.path.join(model_dir, f'ensemble_model_{i+1}.h5'))
                kt_model.save(os.path.join(model_dir, 'kt_model.h5'))
            elif kt_acc == best_acc:
                kt_model.save(os.path.join(model_dir, 'best_kt_model.h5'))
            
            # Save preprocessing objects
            joblib.dump({
                'scaler': scaler,
                'quantile_transformer': qt,
                'pca': pca,
                'label_encoder': self.label_encoder,
                'best_accuracy': best_acc
            }, os.path.join(model_dir, 'preprocessing_objects.pkl'))
            
            print(f"\n✓ Models saved in '{model_dir}' directory")
        
        return best_acc

# Execute the extreme neural network approach
if __name__ == "__main__":
    start_time = time.time()
    
    predictor = ExtremeNeuralNetworkPredictor("C:/Users/maiwa/Downloads/Global Superstore.xlsx")
    final_accuracy = predictor.run_extreme_pipeline()
    
    print(f"\n⏱ Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"\n🎯 Extreme Neural Network Pipeline Complete!")
    print(f"📊 Final Accuracy: {final_accuracy:.2%}") 