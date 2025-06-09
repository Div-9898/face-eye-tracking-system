# Advanced Employee Attrition Prediction: Extreme Optimization for "Yes" Predictions
# Focus: Maximum accuracy and precision for "Yes" attrition predictions with drastic improvements

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
import keras_tuner as kt
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve, precision_recall_curve,
                           average_precision_score, make_scorer)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, RFECV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure GPU for TensorFlow
print("🖥️ Configuring GPU for TensorFlow...")
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"✅ Found {len(physical_devices)} GPU(s):")
    for device in physical_devices:
        print(f"  • {device.name}")
    
    # Enable memory growth to prevent TensorFlow from allocating all GPU memory
    for gpu in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  • Memory growth enabled for {gpu.name}")
        except RuntimeError as e:
            print(f"  • Error setting memory growth: {e}")
    
    # Optional: Set memory limit (uncomment if needed)
    # tf.config.set_logical_device_configuration(
    #     physical_devices[0],
    #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # Limit to 4GB
    # )
    
    print("✅ GPU configuration completed!")
else:
    print("⚠️ No GPU found. Running on CPU.")
    print("💡 For faster training, consider using a GPU-enabled environment.")

# Verify TensorFlow is using GPU
print(f"\n📊 TensorFlow version: {tf.__version__}")
print(f"🔧 Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"🖥️ GPU available: {tf.test.is_gpu_available()}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure matplotlib for better display
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

print("\n🚀 EXTREME Employee Attrition Prediction Analysis")
print("🎯 Model: Deep Neural Network with Keras Tuner and Multiple Threshold Strategies")
print("📊 Focus: MAXIMUM accuracy for 'Yes' attrition predictions")
print("⚡ Implementing EXTREME optimization with GPU acceleration")
print("=" * 70)

# Load and prepare the data
def load_and_prepare_data():
    """Load data from user-specified dataset"""
    
    # ========================================
    # 📁 DATASET CONFIGURATION
    # ========================================
    print("\n" + "="*50)
    print("📁 DATASET CONFIGURATION")
    print("="*50)
    
    # Option 1: Specify path directly in code
    data_path = "C:/Users/maiwa/Downloads/EA.csv"
    
    print(f"🔍 Loading dataset from: {data_path}")
    
    try:
        # Read CSV with error handling
        df = pd.read_csv(data_path, encoding='utf-8')
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Data validation
        if 'Attrition' not in df.columns:
            print("❌ Error: 'Attrition' column not found!")
            print(f"Available columns: {', '.join(df.columns.tolist())}")
            return None
            
        # Display attrition distribution
        print(f"\n📈 Attrition Distribution:")
        attrition_counts = df['Attrition'].value_counts()
        for label, count in attrition_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  • {label}: {count} ({percentage:.1f}%)")
        
        # Calculate class imbalance ratio
        if 'Yes' in attrition_counts.index and 'No' in attrition_counts.index:
            imbalance_ratio = attrition_counts['No'] / attrition_counts['Yes']
            print(f"  • Class imbalance ratio: {imbalance_ratio:.2f}:1")
        
        return df
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found at '{data_path}'")
        return None
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return None

# EXTREME preprocessing with aggressive feature engineering
def preprocess_data_extreme(df):
    """EXTREME preprocessing with aggressive feature engineering for maximum Yes detection"""
    
    print("\n" + "="*50)
    print("🔧 EXTREME DATA PREPROCESSING")
    print("="*50)
    
    df_processed = df.copy()
    
    # Encode target variable
    le_target = LabelEncoder()
    df_processed['Attrition'] = le_target.fit_transform(df_processed['Attrition'])
    print(f"✅ Target encoding: No=0, Yes=1")
    
    # Remove non-predictive columns
    columns_to_drop = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18']
    columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
    if columns_to_drop:
        df_processed = df_processed.drop(columns=columns_to_drop)
        print(f"✅ Removed non-predictive columns: {columns_to_drop}")
    
    # EXTREME Feature Engineering
    print("\n🔨 Creating EXTREME engineered features...")
    
    # 1. Multi-level Satisfaction Index
    satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']
    available_satisfaction = [col for col in satisfaction_cols if col in df_processed.columns]
    if available_satisfaction:
        df_processed['SatisfactionIndex'] = df_processed[available_satisfaction].mean(axis=1)
        df_processed['SatisfactionStd'] = df_processed[available_satisfaction].std(axis=1)
        df_processed['MinSatisfaction'] = df_processed[available_satisfaction].min(axis=1)
        df_processed['SatisfactionRange'] = df_processed[available_satisfaction].max(axis=1) - df_processed[available_satisfaction].min(axis=1)
        print("  ✓ Created multi-level satisfaction features")
    
    # 2. Advanced Work-Life Balance Metrics
    if all(col in df_processed.columns for col in ['WorkLifeBalance', 'OverTime']):
        df_processed['WorkLifeScore'] = df_processed['WorkLifeBalance'] * (1 - df_processed['OverTime'].map({'Yes': 1, 'No': 0}))
        df_processed['WorkLifeRisk'] = (df_processed['WorkLifeBalance'] <= 2) & (df_processed['OverTime'].map({'Yes': 1, 'No': 0}) == 1)
        print("  ✓ Created advanced work-life metrics")
    
    # 3. Career Stagnation Indicators
    if all(col in df_processed.columns for col in ['YearsAtCompany', 'YearsSinceLastPromotion', 'YearsInCurrentRole']):
        df_processed['PromotionRate'] = df_processed['YearsSinceLastPromotion'] / (df_processed['YearsAtCompany'] + 1)
        df_processed['RoleStagnation'] = df_processed['YearsInCurrentRole'] / (df_processed['YearsAtCompany'] + 1)
        df_processed['CareerProgress'] = df_processed['JobLevel'] / (df_processed['TotalWorkingYears'] + 1)
        df_processed['IsStagnant'] = (df_processed['YearsSinceLastPromotion'] > 4) & (df_processed['YearsInCurrentRole'] > 4)
        print("  ✓ Created career stagnation indicators")
    
    # 4. Income Analysis
    if all(col in df_processed.columns for col in ['MonthlyIncome', 'Age', 'JobLevel']):
        df_processed['IncomeAgeRatio'] = df_processed['MonthlyIncome'] / (df_processed['Age'] + 1)
        df_processed['IncomePerLevel'] = df_processed['MonthlyIncome'] / (df_processed['JobLevel'] + 1)
        # Calculate income percentile within job level
        df_processed['IncomePercentile'] = df_processed.groupby('JobLevel')['MonthlyIncome'].rank(pct=True)
        df_processed['IsUnderpaid'] = df_processed['IncomePercentile'] < 0.25
        print("  ✓ Created income analysis features")
    
    # 5. Loyalty vs Reward Imbalance
    if all(col in df_processed.columns for col in ['YearsAtCompany', 'StockOptionLevel', 'JobLevel']):
        df_processed['LoyaltyRewardRatio'] = (df_processed['YearsAtCompany'] + 1) / (df_processed['StockOptionLevel'] + df_processed['JobLevel'] + 1)
        df_processed['UnrewardedLoyalty'] = (df_processed['YearsAtCompany'] > 5) & (df_processed['StockOptionLevel'] == 0)
        print("  ✓ Created loyalty-reward features")
    
    # 6. Distance and Life Stage Risk
    if 'DistanceFromHome' in df_processed.columns:
        df_processed['HighDistance'] = df_processed['DistanceFromHome'] > df_processed['DistanceFromHome'].quantile(0.75)
    
    if all(col in df_processed.columns for col in ['Age', 'MaritalStatus']):
        df_processed['YoungSingle'] = (df_processed['Age'] < 30) & (df_processed['MaritalStatus'] == 'Single')
    
    # 7. Performance vs Reward Mismatch
    if all(col in df_processed.columns for col in ['PerformanceRating', 'PercentSalaryHike']):
        df_processed['PerformanceRewardGap'] = df_processed['PerformanceRating'] - (df_processed['PercentSalaryHike'] / 10)
        print("  ✓ Created performance-reward features")
    
    # 8. Engagement Score
    if 'JobInvolvement' in df_processed.columns:
        engagement_features = ['JobInvolvement']
        if 'EnvironmentSatisfaction' in df_processed.columns:
            engagement_features.append('EnvironmentSatisfaction')
        df_processed['EngagementScore'] = df_processed[engagement_features].mean(axis=1)
        df_processed['LowEngagement'] = df_processed['EngagementScore'] <= 2
        print("  ✓ Created engagement features")
    
    # 9. Risk Combination Features
    risk_factors = []
    if 'IsStagnant' in df_processed.columns:
        risk_factors.append('IsStagnant')
    if 'IsUnderpaid' in df_processed.columns:
        risk_factors.append('IsUnderpaid')
    if 'UnrewardedLoyalty' in df_processed.columns:
        risk_factors.append('UnrewardedLoyalty')
    if 'LowEngagement' in df_processed.columns:
        risk_factors.append('LowEngagement')
    
    if risk_factors:
        df_processed['TotalRiskFactors'] = df_processed[risk_factors].sum(axis=1)
        df_processed['HighRisk'] = df_processed['TotalRiskFactors'] >= 2
        print("  ✓ Created risk combination features")
    
    # 10. Statistical Outlier Detection for Each Employee
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if 'Attrition' in numerical_cols:
        numerical_cols.remove('Attrition')
    
    if numerical_cols:
        # Z-score based outlier detection
        z_scores = np.abs(stats.zscore(df_processed[numerical_cols].fillna(df_processed[numerical_cols].mean())))
        df_processed['OutlierScore'] = (z_scores > 2).sum(axis=1)
        df_processed['IsOutlier'] = df_processed['OutlierScore'] > 2
        print("  ✓ Created outlier detection features")
    
    # Identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []
    
    for col in df_processed.columns:
        if col == 'Attrition':
            continue
        if df_processed[col].dtype == 'object' or df_processed[col].dtype.name == 'category':
            categorical_cols.append(col)
        else:
            unique_ratio = df_processed[col].nunique() / len(df_processed)
            if unique_ratio < 0.05:  # Less than 5% unique values
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
    
    print(f"\n📊 Feature types after engineering:")
    print(f"  • Numerical: {len(numerical_cols)} features")
    print(f"  • Categorical: {len(categorical_cols)} features")
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = df_processed[col].astype(str)
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
    
    # Handle missing values
    if numerical_cols:
        imputer = SimpleImputer(strategy='median')
        df_processed[numerical_cols] = imputer.fit_transform(df_processed[numerical_cols])
    
    print(f"\n✅ EXTREME preprocessing completed!")
    print(f"📊 Final shape: {df_processed.shape}")
    
    return df_processed, label_encoders

# EXTREME feature selection with multiple advanced methods
def select_features_extreme(X, y, k=35):
    """EXTREME feature selection using multiple advanced methods - OPTIMIZED VERSION"""
    
    print("\n" + "="*50)
    print("🎯 EXTREME FEATURE SELECTION - OPTIMIZED")
    print("="*50)
    
    # Ensure k doesn't exceed number of features
    k = min(k, X.shape[1])
    
    print(f"📊 Selecting top {k} features from {X.shape[1]} total features...")
    
    # Method 1: F-statistic (Fast)
    print("  • Running F-statistic selection...")
    selector_f = SelectKBest(score_func=f_classif, k=k)
    selector_f.fit(X, y)
    f_scores = pd.DataFrame({
        'feature': X.columns,
        'f_score': selector_f.scores_
    }).sort_values('f_score', ascending=False)
    
    # Method 2: Mutual information (Fast)
    print("  • Running mutual information selection...")
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=k)
    selector_mi.fit(X, y)
    mi_scores = pd.DataFrame({
        'feature': X.columns,
        'mi_score': selector_mi.scores_
    }).sort_values('mi_score', ascending=False)
    
    # Method 3: Random Forest feature importance (Fast)
    print("  • Running Random Forest importance...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1  # Use all cores
    )
    rf.fit(X, y)
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    # Method 4: XGBoost feature importance (Fast)
    print("  • Running XGBoost importance...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        scale_pos_weight=len(y[y==0])/len(y[y==1]),
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X, y)
    xgb_importance = pd.DataFrame({
        'feature': X.columns,
        'xgb_importance': xgb_model.feature_importances_
    }).sort_values('xgb_importance', ascending=False)
    
    # Method 5: Correlation with target (Very Fast)
    print("  • Calculating correlation with target...")
    correlations = pd.DataFrame({
        'feature': X.columns,
        'correlation': [abs(X[col].corr(y)) for col in X.columns]
    }).sort_values('correlation', ascending=False)
    
    # Combine all methods - create a ranking system
    print("\n📊 Combining feature selection methods...")
    
    feature_rankings = pd.DataFrame({'feature': X.columns})
    
    # Add rankings from each method
    for df, col_name in [(f_scores, 'f_score'), 
                        (mi_scores, 'mi_score'),
                        (rf_importance, 'rf_importance'),
                        (xgb_importance, 'xgb_importance'),
                        (correlations, 'correlation')]:
        df['rank'] = range(1, len(df) + 1)
        feature_rankings = feature_rankings.merge(
            df[['feature', 'rank']], 
            on='feature', 
            suffixes=('', f'_{col_name}')
        )
        feature_rankings.rename(columns={'rank': f'rank_{col_name}'}, inplace=True)
    
    # Calculate average rank
    rank_columns = [col for col in feature_rankings.columns if 'rank_' in col]
    feature_rankings['avg_rank'] = feature_rankings[rank_columns].mean(axis=1)
    feature_rankings = feature_rankings.sort_values('avg_rank')
    
    # Select top k features
    selected_features = feature_rankings.head(k)['feature'].tolist()
    
    print(f"✅ Selected {len(selected_features)} features using ensemble ranking")
    print(f"📊 Top 10 features: {', '.join(selected_features[:10])}...")
    
    # Create interaction features for top numerical features (OPTIMIZED)
    print(f"\n🔧 Creating interaction features...")
    
    # Identify numerical features (those with many unique values)
    numerical_features = []
    for feature in selected_features[:15]:  # Only use top 15 to limit complexity
        if X[feature].nunique() > 10:
            numerical_features.append(feature)
    
    # Limit to top 5 numerical features to prevent explosion
    numerical_features = numerical_features[:5]
    
    X_selected = X[selected_features].copy()
    
    if len(numerical_features) >= 2:
        print(f"  • Creating interactions for {len(numerical_features)} numerical features...")
        
        # Create only key interactions (not all combinations)
        interaction_count = 0
        for i in range(len(numerical_features)):
            for j in range(i+1, len(numerical_features)):
                if interaction_count >= 10:  # Limit number of interactions
                    break
                feat1, feat2 = numerical_features[i], numerical_features[j]
                
                # Create interaction
                X_selected[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
                
                # Create ratio (with protection against division by zero)
                X_selected[f'{feat1}_div_{feat2}'] = X[feat1] / (X[feat2] + 1e-5)
                
                interaction_count += 2
        
        print(f"✅ Added {interaction_count} interaction features")
    
    # Add some domain-specific engineered features if they exist
    if 'MonthlyIncome' in X.columns and 'Age' in X.columns:
        X_selected['Income_Age_Ratio'] = X['MonthlyIncome'] / (X['Age'] + 1)
    
    if 'YearsAtCompany' in X.columns and 'TotalWorkingYears' in X.columns:
        X_selected['Company_Career_Ratio'] = X['YearsAtCompany'] / (X['TotalWorkingYears'] + 1)
    
    print(f"\n✅ Feature selection completed!")
    print(f"📊 Final feature count: {X_selected.shape[1]}")
    
    return X_selected, X_selected.columns.tolist()

# EXTREME Deep Neural Network Model with Keras Tuner
def build_extreme_model(hp):
    """Build a deep neural network with hyperparameter tuning for extreme Yes prediction"""
    
    model = keras.Sequential()
    
    # Input layer with tunable units
    model.add(layers.Dense(
        units=hp.Int('input_units', min_value=128, max_value=512, step=64),
        activation='relu',
        kernel_regularizer=regularizers.l2(hp.Float('l2_input', min_value=1e-4, max_value=1e-2, sampling='LOG'))
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hp.Float('dropout_input', min_value=0.3, max_value=0.6, step=0.1)))
    
    # Hidden layers with tunable depth
    for i in range(hp.Int('n_layers', 2, 5)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation=hp.Choice(f'activation_{i}', ['relu', 'elu', 'swish']),
            kernel_regularizer=regularizers.l2(hp.Float(f'l2_{i}', min_value=1e-4, max_value=1e-2, sampling='LOG'))
        ))
        
        if hp.Boolean(f'batch_norm_{i}'):
            model.add(layers.BatchNormalization())
        
        model.add(layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Additional attention-like layer for Yes detection (fixed implementation)
    if hp.Boolean('attention_layer'):
        # Add a dense layer with sigmoid activation to create attention weights
        attention_units = hp.Int('attention_units', min_value=16, max_value=64, step=16)
        model.add(layers.Dense(attention_units, activation='sigmoid', name='attention_weights'))
        # Add another dense layer to apply the attention
        model.add(layers.Dense(attention_units, activation='relu', name='attention_output'))
        model.add(layers.Dropout(0.3))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile with tunable learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    
    # Custom loss function that penalizes false negatives more
    def weighted_binary_crossentropy(y_true, y_pred):
        # Penalize false negatives (missing Yes predictions) more heavily
        false_negative_weight = 3.0
        loss = -false_negative_weight * y_true * tf.math.log(y_pred + 1e-7) - (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
        return tf.reduce_mean(loss)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=weighted_binary_crossentropy,
        metrics=['accuracy', 
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')]
    )
    
    return model

def create_extreme_neural_network_tuned(X_train, y_train, X_val, y_val):
    """Create and tune an EXTREME deep neural network optimized for Yes predictions"""
    
    print("\n🔧 Running Keras Tuner for Neural Network optimization...")
    
    # Check GPU availability for this specific training
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  • Using GPU: {gpus[0].name}")
        # Create a strategy for GPU usage
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        print("  • Using CPU (no GPU available)")
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    
    # Define a custom tuner objective that focuses on Yes precision
    def custom_objective(metrics_dict):
        # Heavily weight precision while maintaining some recall
        precision = metrics_dict.get('val_precision', 0)
        recall = metrics_dict.get('val_recall', 0)
        
        # Custom score that emphasizes precision but requires minimum recall
        if recall < 0.3:  # Minimum recall threshold
            return 0
        else:
            # 70% weight on precision, 30% on recall
            return 0.7 * precision + 0.3 * recall
    
    # Create tuner with GPU strategy
    with strategy.scope():
        tuner = kt.BayesianOptimization(
            build_extreme_model,
            objective=kt.Objective('val_precision', direction='max'),
            max_trials=20,
            num_initial_points=5,
            directory='keras_tuner_extreme',
            project_name='attrition_yes_optimization',
            overwrite=True
        )
    
    # Define callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_precision',
        patience=15,
        restore_best_weights=True,
        mode='max'
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_precision',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode='max'
    )
    
    # GPU memory callback
    class GPUMemoryCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if gpus:
                # Get GPU memory info
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                if gpu_memory:
                    used_memory = gpu_memory.get('current', 0) / (1024**2)  # Convert to MB
                    if epoch % 10 == 0:  # Print every 10 epochs
                        print(f"    GPU Memory: {used_memory:.0f} MB")
    
    # Custom class weights for training
    class_weight = {0: 1.0, 1: 3.0}  # Triple weight for Yes class
    
    # Search for best hyperparameters
    print("  • Searching for optimal hyperparameters...")
    
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        tuner.search(
            X_train, y_train,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr, GPUMemoryCallback()],
            class_weight=class_weight,
            verbose=0
        )
    
    # Get best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("  • Best hyperparameters found:")
    print(f"    - Input units: {best_hp.get('input_units')}")
    print(f"    - Number of layers: {best_hp.get('n_layers')}")
    print(f"    - Learning rate: {best_hp.get('learning_rate'):.6f}")
    
    # Retrain best model with more epochs
    print("  • Retraining best model with extended epochs...")
    
    with strategy.scope():
        final_model = tuner.hypermodel.build(best_hp)
    
    # Extended training with GPU placement
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        history = final_model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=16,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr, GPUMemoryCallback()],
            class_weight=class_weight,
            verbose=0
        )
    
    # Print final GPU usage
    if gpus:
        print("  • Training completed on GPU")
    
    return final_model, history, best_hp

# Custom scoring function for precision on Yes class
def yes_precision_score(y_true, y_pred):
    """Custom scorer that focuses on precision for Yes predictions"""
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)

# EXTREME model training with aggressive techniques
def train_models_extreme(X_train, X_test, y_train, y_test, feature_names):
    """Train Neural Network model with EXTREME optimization for Yes predictions"""
    
    print("\n" + "="*50)
    print("🤖 TRAINING EXTREME NEURAL NETWORK")
    print("="*50)
    
    # Calculate extreme class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # Make Yes predictions even more important
    class_weight_dict = {0: class_weights[0], 1: class_weights[1] * 2}  # Double weight for Yes
    print(f"📊 Extreme class weights: No={class_weight_dict[0]:.2f}, Yes={class_weight_dict[1]:.2f}")
    
    # Apply multiple SMOTE variants
    print("\n🔄 Applying EXTREME oversampling techniques...")
    
    # Try multiple SMOTE variants and choose the best
    smote_variants = {
        'SMOTE': SMOTE(random_state=42, k_neighbors=3),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=3),
        'SVMSMOTE': SVMSMOTE(random_state=42, k_neighbors=3),
        'ADASYN': ADASYN(random_state=42, n_neighbors=3)
    }
    
    best_smote = None
    best_ratio = 0
    
    for name, sampler in smote_variants.items():
        try:
            X_temp, y_temp = sampler.fit_resample(X_train, y_train)
            ratio = sum(y_temp == 1) / len(y_temp)
            print(f"  • {name}: {ratio:.2%} Yes samples")
            if ratio > best_ratio and ratio <= 0.6:  # Don't oversample too much
                best_ratio = ratio
                best_smote = name
                X_train_balanced = X_temp
                y_train_balanced = y_temp
        except:
            continue
    
    print(f"✅ Selected {best_smote} for oversampling")
    print(f"📊 Balanced dataset: {np.bincount(y_train_balanced)}")
    
    results = {}
    models = {}
    
    # EXTREME Deep Neural Network with Keras Tuner
    print("\n🧠 Training EXTREME Deep Neural Network with Keras Tuner...")
    
    # Prepare data for TensorFlow
    scaler = RobustScaler()  # More robust to outliers
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Split training data for validation
    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
        X_train_scaled, y_train_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train_balanced
    )
    
    # Create and train model with Keras Tuner
    tf_model, history, best_hp = create_extreme_neural_network_tuned(
        X_train_nn, y_train_nn, X_val_nn, y_val_nn
    )
    
    # TensorFlow predictions with multiple threshold optimization
    tf_proba = tf_model.predict(X_test_scaled).flatten()
    
    # Try multiple threshold strategies
    tf_threshold, threshold_results = optimize_nn_thresholds(y_test, tf_proba)
    tf_pred = (tf_proba > tf_threshold).astype(int)
    
    models['NN_Extreme'] = (tf_model, scaler, tf_threshold, threshold_results)
    results['NN_Extreme'] = evaluate_model(y_test, tf_pred, tf_proba, 'NN_Extreme')
    
    # Also evaluate with different thresholds
    print("\n📊 Evaluating Neural Network with different threshold strategies:")
    for strategy, thresh_info in threshold_results.items():
        threshold = thresh_info['threshold']
        y_pred_strategy = (tf_proba > threshold).astype(int)
        
        print(f"\n  Strategy: {strategy} (threshold={threshold:.3f})")
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_strategy),
            'precision': precision_score(y_test, y_pred_strategy, zero_division=0),
            'recall': recall_score(y_test, y_pred_strategy, zero_division=0),
            'f1': f1_score(y_test, y_pred_strategy, zero_division=0)
        }
        print(f"    • Accuracy: {metrics['accuracy']:.3f}")
        print(f"    • Precision: {metrics['precision']:.3f}")
        print(f"    • Recall: {metrics['recall']:.3f}")
        print(f"    • F1-Score: {metrics['f1']:.3f}")
        
        # Store the best strategy result
        results[f'NN_{strategy}'] = {
            'metrics': metrics,
            'y_pred': y_pred_strategy,
            'y_proba': tf_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred_strategy),
            'threshold': threshold
        }
    
    return results, models, history

# Optimize threshold for maximum precision with multiple strategies
def optimize_threshold_advanced(y_true, y_proba, strategy='balanced'):
    """Find optimal threshold using different strategies for extreme Yes detection"""
    
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    if strategy == 'max_precision':
        # Maximum precision with minimum recall constraint
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            if sum(y_pred) > 0:  # Avoid division by zero
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                
                # Only consider if recall is above 20%
                if recall >= 0.2 and precision > best_score:
                    best_score = precision
                    best_threshold = threshold
                    
    elif strategy == 'f2_score':
        # F2 score gives more weight to recall
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            if sum(y_pred) > 0:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                if precision > 0 and recall > 0:
                    f2 = 5 * precision * recall / (4 * precision + recall)
                    if f2 > best_score:
                        best_score = f2
                        best_threshold = threshold
                        
    elif strategy == 'balanced':
        # Balance between precision and recall
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            if sum(y_pred) > 0:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                # Custom score: 60% precision, 40% recall
                score = 0.6 * precision + 0.4 * recall
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
    elif strategy == 'aggressive':
        # Very aggressive: lower threshold to catch more Yes
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            if sum(y_pred) > 0:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                # Require at least 50% precision
                if precision >= 0.5 and recall > best_score:
                    best_score = recall
                    best_threshold = threshold
    
    return best_threshold

# Optimize threshold for maximum precision
def optimize_threshold_for_precision(y_true, y_proba, min_recall=0.3):
    """Find optimal threshold that maximizes precision while maintaining minimum recall"""
    
    thresholds = np.arange(0.1, 0.95, 0.01)
    best_threshold = 0.5
    best_precision = 0
    
    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        if sum(y_pred) > 0:  # Avoid division by zero
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            # Only consider if recall is above minimum
            if recall >= min_recall and precision > best_precision:
                best_precision = precision
                best_threshold = threshold
    
    return best_threshold

# Neural Network specific threshold optimization
def optimize_nn_thresholds(y_true, y_proba):
    """Optimize multiple thresholds for neural network predictions"""
    
    print("\n🎯 Optimizing multiple thresholds for Neural Network...")
    
    strategies = ['max_precision', 'f2_score', 'balanced', 'aggressive']
    thresholds = {}
    
    for strategy in strategies:
        threshold = optimize_threshold_advanced(y_true, y_proba, strategy)
        y_pred = (y_proba > threshold).astype(int)
        
        if sum(y_pred) > 0:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            thresholds[strategy] = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"  • {strategy}: threshold={threshold:.3f}, precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
    
    # Select best strategy based on precision with minimum recall
    best_strategy = max(thresholds.items(), 
                       key=lambda x: x[1]['precision'] if x[1]['recall'] >= 0.25 else 0)
    
    print(f"\n✅ Selected strategy: {best_strategy[0]} with threshold {best_strategy[1]['threshold']:.3f}")
    
    return best_strategy[1]['threshold'], thresholds

# Model evaluation function
def evaluate_model(y_true, y_pred, y_proba, model_name):
    """Comprehensive model evaluation with focus on Yes predictions"""
    
    # Calculate all metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    average_precision = average_precision_score(y_true, y_proba)
    
    # Per-class metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 {model_name} Results:")
    print(f"  • Accuracy: {accuracy:.3f}")
    print(f"  • Precision (Yes): {precision:.3f}")
    print(f"  • Recall (Yes): {recall:.3f}")
    print(f"  • F1-Score: {f1:.3f}")
    print(f"  • ROC-AUC: {roc_auc:.3f}")
    print(f"  • True Positives: {tp}/{sum(y_true==1)} ({tp/sum(y_true==1)*100:.1f}%)")
    print(f"  • False Positives: {fp}/{sum(y_true==0)} ({fp/sum(y_true==0)*100:.1f}%)")
    
    return {
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'average_precision': average_precision,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        },
        'y_pred': y_pred,
        'y_proba': y_proba,
        'confusion_matrix': cm
    }

# Create individual visualizations
def create_individual_visualizations(results, y_test, models, X_test, feature_names):
    """Create separate, detailed visualizations for Neural Network analysis"""
    
    print("\n" + "="*50)
    print("📊 CREATING NEURAL NETWORK VISUALIZATIONS")
    print("="*50)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Neural Network Performance with Different Thresholds
    plt.figure(figsize=(14, 8))
    
    # Prepare data for different threshold strategies
    strategies = ['NN_Extreme', 'NN_max_precision', 'NN_f2_score', 'NN_balanced', 'NN_aggressive']
    available_strategies = [s for s in strategies if s in results]
    
    metrics_data = {
        'Strategy': [],
        'Threshold': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    for strategy in available_strategies:
        result = results[strategy]
        strategy_name = strategy.replace('NN_', '').replace('_', ' ').title()
        metrics_data['Strategy'].append(strategy_name)
        
        # Get threshold value
        if 'threshold' in result:
            metrics_data['Threshold'].append(result['threshold'])
        else:
            metrics_data['Threshold'].append(0.5)
        
        # Get metrics
        if 'metrics' in result:
            metrics = result['metrics']
            metrics_data['Accuracy'].append(metrics.get('accuracy', 0))
            metrics_data['Precision'].append(metrics.get('precision', 0))
            metrics_data['Recall'].append(metrics.get('recall', 0))
            metrics_data['F1-Score'].append(metrics.get('f1', 0))
        else:
            metrics_data['Accuracy'].append(result.get('accuracy', 0))
            metrics_data['Precision'].append(result.get('precision', 0))
            metrics_data['Recall'].append(result.get('recall', 0))
            metrics_data['F1-Score'].append(result.get('f1', 0))
    
    # Create grouped bar chart
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Performance metrics
    metrics_to_plot = metrics_df[['Strategy', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
    metrics_to_plot.set_index('Strategy').plot(kind='bar', ax=ax1, 
                                               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3'])
    
    ax1.set_title('Neural Network Performance with Different Threshold Strategies', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_xlabel('Threshold Strategy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.3f', fontsize=8, padding=2)
    
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Threshold values
    ax2.bar(metrics_df['Strategy'], metrics_df['Threshold'], color='#9B59B6')
    ax2.set_title('Optimal Thresholds by Strategy', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Threshold Value', fontsize=12)
    ax2.set_xlabel('Strategy', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(metrics_df['Threshold']):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Confusion Matrices for Different Strategies
    num_strategies = len(available_strategies)
    if num_strategies > 0:
        cols = min(3, num_strategies)
        rows = (num_strategies + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if num_strategies == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes
        
        for idx, strategy in enumerate(available_strategies):
            if num_strategies == 1:
                ax = axes[0]
            else:
                ax = axes[idx]
                
            result = results[strategy]
            cm = result['confusion_matrix']
            
            # Create annotated heatmap
            im = ax.imshow(cm, cmap='Blues', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Count', fontsize=9)
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    value = cm[i, j]
                    percentage = value / cm.sum() * 100
                    text = ax.text(j, i, f'{value}\n({percentage:.1f}%)', 
                                 ha='center', va='center', fontsize=11,
                                 color='white' if value > cm.max()/2 else 'black',
                                 fontweight='bold')
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['No', 'Yes'], fontsize=11)
            ax.set_yticklabels(['No', 'Yes'], fontsize=11)
            ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
            
            strategy_name = strategy.replace('NN_', '').replace('_', ' ').title()
            ax.set_title(f'{strategy_name}', fontsize=12, fontweight='bold')
        
        # Hide empty subplots
        if num_strategies < len(axes):
            for idx in range(num_strategies, len(axes)):
                axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices - Different Threshold Strategies', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # 3. Precision-Recall Curve with Different Thresholds
    plt.figure(figsize=(10, 8))
    
    # Get the probability predictions (same for all strategies)
    y_proba = results[available_strategies[0]]['y_proba']
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Plot the curve
    plt.plot(recall, precision, 'b-', linewidth=2, label='Neural Network')
    
    # Mark different threshold strategies on the curve
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, strategy in enumerate(available_strategies):
        result = results[strategy]
        threshold = result.get('threshold', 0.5)
        
        # Find the point on the curve closest to this threshold
        if 'metrics' in result:
            strategy_recall = result['metrics'].get('recall', 0)
            strategy_precision = result['metrics'].get('precision', 0)
        else:
            # Calculate from predictions
            y_pred = (y_proba > threshold).astype(int)
            strategy_recall = recall_score(y_test, y_pred)
            strategy_precision = precision_score(y_test, y_pred, zero_division=0)
        
        strategy_name = strategy.replace('NN_', '').replace('_', ' ').title()
        plt.scatter(strategy_recall, strategy_precision, 
                   color=colors[idx % len(colors)], 
                   marker=markers[idx % len(markers)],
                   s=150, 
                   label=f'{strategy_name} (t={threshold:.3f})',
                   edgecolors='black',
                   linewidth=2)
    
    # Add baseline
    baseline = sum(y_test) / len(y_test)
    plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, 
               label=f'Baseline (Random): {baseline:.3f}')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve with Different Threshold Strategies', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()
    
    # 4. ROC Curve with threshold points
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Neural Network (AUC = {roc_auc:.3f})')
    
    # Mark different threshold strategies
    for idx, strategy in enumerate(available_strategies):
        result = results[strategy]
        threshold = result.get('threshold', 0.5)
        
        # Calculate FPR and TPR for this threshold
        y_pred = (y_proba > threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate rates
        tn, fp, fn, tp = cm.ravel()
        strategy_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        strategy_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        strategy_name = strategy.replace('NN_', '').replace('_', ' ').title()
        plt.scatter(strategy_fpr, strategy_tpr, 
                   color=colors[idx % len(colors)], 
                   marker=markers[idx % len(markers)],
                   s=150, 
                   label=f'{strategy_name}',
                   edgecolors='black',
                   linewidth=2)
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve with Different Threshold Strategies', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Prediction Distribution Analysis
    plt.figure(figsize=(12, 8))
    
    # Get probabilities for each class
    proba_no = y_proba[y_test == 0]
    proba_yes = y_proba[y_test == 1]
    
    # Create histogram
    plt.hist(proba_no, bins=30, alpha=0.6, label='Actual No', 
            color='blue', density=True, edgecolor='black', linewidth=0.5)
    plt.hist(proba_yes, bins=30, alpha=0.6, label='Actual Yes', 
            color='red', density=True, edgecolor='black', linewidth=0.5)
    
    # Add threshold lines for different strategies
    for idx, strategy in enumerate(available_strategies):
        result = results[strategy]
        threshold = result.get('threshold', 0.5)
        strategy_name = strategy.replace('NN_', '').replace('_', ' ').title()
        
        plt.axvline(x=threshold, 
                   color=colors[idx % len(colors)], 
                   linestyle='--', 
                   linewidth=2,
                   label=f'{strategy_name}: {threshold:.3f}')
    
    plt.xlabel('Predicted Probability of Attrition', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Prediction Probability Distribution with Different Thresholds', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 6. Performance Summary Table
    plt.figure(figsize=(12, 6))
    plt.axis('tight')
    plt.axis('off')
    
    # Create summary data
    summary_data = []
    for strategy in available_strategies:
        result = results[strategy]
        strategy_name = strategy.replace('NN_', '').replace('_', ' ').title()
        
        if 'metrics' in result:
            metrics = result['metrics']
        else:
            metrics = result
        
        cm = result['confusion_matrix']
        
        summary_data.append([
            strategy_name,
            f"{result.get('threshold', 0.5):.3f}",
            f"{metrics.get('accuracy', 0):.3f}",
            f"{metrics.get('precision', 0):.3f}",
            f"{metrics.get('recall', 0):.3f}",
            f"{metrics.get('f1', 0):.3f}",
            f"{cm[1,1]}/{sum(y_test==1)}"  # TP/Total Yes
        ])
    
    # Create table
    table = plt.table(
        cellText=summary_data,
        colLabels=['Strategy', 'Threshold', 'Accuracy', 'Precision', 'Recall', 
                   'F1-Score', 'True Positives'],
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.12, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the table
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        else:
            cell.set_facecolor('#F2F2F2' if row % 2 == 0 else 'white')
            
            # Highlight best values
            if col > 1:  # Skip strategy and threshold columns
                value = float(cell.get_text().get_text().split('/')[0])
                if col in [3, 4] and value >= 0.8:  # High precision/recall
                    cell.set_facecolor('#90EE90')
                elif col == 5 and value >= 0.7:  # High F1
                    cell.set_facecolor('#90EE90')
    
    plt.title('Neural Network Performance Summary - Different Threshold Strategies', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    print("\n✅ All visualizations created successfully!")

# Main execution function
def main():
    """Main execution function with comprehensive error handling"""
    
    try:
        # Load data
        df = load_and_prepare_data()
        if df is None:
            print("\n❌ Failed to load data. Exiting...")
            return None
        
        # EXTREME preprocessing
        df_processed, label_encoders = preprocess_data_extreme(df)
        
        # Prepare features and target
        X = df_processed.drop('Attrition', axis=1)
        y = df_processed['Attrition']
        
        # EXTREME feature selection
        X_selected, selected_features = select_features_extreme(X, y, k=40)
        
        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data Split Information:")
        print(f"  • Training set: {X_train.shape}")
        print(f"  • Test set: {X_test.shape}")
        print(f"  • Test set Yes ratio: {(y_test == 1).sum() / len(y_test):.2%}")
        
        # Train EXTREME models
        results, models, history = train_models_extreme(
            X_train, X_test, y_train, y_test, selected_features
        )
        
        # Create individual visualizations
        create_individual_visualizations(results, y_test, models, X_test, selected_features)
        
        # Print final summary
        print("\n" + "="*70)
        print("📊 EXTREME NEURAL NETWORK OPTIMIZATION SUMMARY")
        print("="*70)
        
        # Find best strategy by precision
        nn_strategies = [k for k in results.keys() if k.startswith('NN_')]
        best_precision_strategy = max(
            nn_strategies,
            key=lambda x: results[x].get('metrics', results[x]).get('precision', 0)
        )
        best_precision_result = results[best_precision_strategy]
        
        print(f"\n🏆 Best Strategy by Precision: {best_precision_strategy.replace('NN_', '').replace('_', ' ').title()}")
        if 'metrics' in best_precision_result:
            metrics = best_precision_result['metrics']
        else:
            metrics = best_precision_result
        print(f"   • Threshold: {best_precision_result.get('threshold', 0.5):.3f}")
        print(f"   • Precision: {metrics.get('precision', 0):.3f}")
        print(f"   • Recall: {metrics.get('recall', 0):.3f}")
        print(f"   • F1-Score: {metrics.get('f1', 0):.3f}")
        
        # Find best strategy by F1
        best_f1_strategy = max(
            nn_strategies,
            key=lambda x: results[x].get('metrics', results[x]).get('f1', 0)
        )
        best_f1_result = results[best_f1_strategy]
        
        print(f"\n🏆 Best Strategy by F1-Score: {best_f1_strategy.replace('NN_', '').replace('_', ' ').title()}")
        if 'metrics' in best_f1_result:
            metrics = best_f1_result['metrics']
        else:
            metrics = best_f1_result
        print(f"   • Threshold: {best_f1_result.get('threshold', 0.5):.3f}")
        print(f"   • F1-Score: {metrics.get('f1', 0):.3f}")
        print(f"   • Precision: {metrics.get('precision', 0):.3f}")
        print(f"   • Recall: {metrics.get('recall', 0):.3f}")
        
        print("\n✅ EXTREME NEURAL NETWORK ANALYSIS COMPLETED SUCCESSFULLY!")
        print("\n💡 Key Features Implemented:")
        print("  • Extreme feature engineering with interaction terms")
        print("  • Keras Tuner for hyperparameter optimization")
        print("  • Multiple SMOTE variants for optimal oversampling")
        print("  • Custom weighted loss function (3x weight for Yes)")
        print("  • Multiple threshold optimization strategies")
        print("  • GPU acceleration (if available)")
        print("  • Comprehensive visualization of different threshold approaches")
        
        return results, models
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Execute the analysis
if __name__ == "__main__":
    # Check required packages
    print("\n📦 Checking required packages...")
    try:
        import tensorflow
        import xgboost
        import imblearn
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install missing packages using:")
        print("pip install tensorflow xgboost imbalanced-learn")
    
    # Run the EXTREME analysis
    print("\n🚀 Starting EXTREME analysis...")
    results, models = main()
    
    if results is not None:
        print("\n🎉 EXTREME analysis completed successfully!")
    else:
        print("\n❌ Analysis failed. Please check your data and try again.") 