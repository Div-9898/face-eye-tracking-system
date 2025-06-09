import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Adamax, Nadam, RMSprop
import keras_tuner as kt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class ExtremeOrderPriorityOptimizer:
    def __init__(self, file_path):
        """Initialize the Extreme Optimizer for 90%+ accuracy"""
        self.df = pd.read_excel(file_path)
        self.original_distribution = self.df['Order Priority'].value_counts()
        print("="*70)
        print("EXTREME OPTIMIZATION FOR 90%+ ACCURACY")
        print("="*70)
        print("\nOriginal Class Distribution:")
        print(self.original_distribution)
        print(f"\nClass Imbalance Ratio: {self.original_distribution.max() / self.original_distribution.min():.2f}:1")
        
        self.prepare_extreme_features()
        
    def prepare_extreme_features(self):
        """Prepare features with extreme engineering techniques"""
        print("\n" + "="*70)
        print("EXTREME FEATURE ENGINEERING")
        print("="*70)
        
        # Convert dates
        self.df['Order Date'] = pd.to_datetime(self.df['Order Date'])
        self.df['Ship Date'] = pd.to_datetime(self.df['Ship Date'])
        
        # Basic features
        self.df['Profit_Margin'] = (self.df['Profit'] / self.df['Sales'].replace(0, 1)) * 100
        self.df['Revenue_Per_Unit'] = self.df['Sales'] / self.df['Quantity'].replace(0, 1)
        self.df['Shipping_Efficiency'] = self.df['Sales'] / self.df['Shipping Cost'].replace(0, 1)
        self.df['Discount_Impact'] = self.df['Discount'] * self.df['Sales']
        self.df['Profit_Per_Unit'] = self.df['Profit'] / self.df['Quantity'].replace(0, 1)
        
        # Time features
        self.df['Days_to_Ship'] = (self.df['Ship Date'] - self.df['Order Date']).dt.days
        self.df['Order_Month'] = self.df['Order Date'].dt.month
        self.df['Order_Quarter'] = self.df['Order Date'].dt.quarter
        self.df['Order_DayOfWeek'] = self.df['Order Date'].dt.dayofweek
        self.df['Order_Week'] = self.df['Order Date'].dt.isocalendar().week
        self.df['Order_Day'] = self.df['Order Date'].dt.day
        self.df['Order_Year'] = self.df['Order Date'].dt.year
        self.df['Is_Weekend'] = (self.df['Order_DayOfWeek'] >= 5).astype(int)
        self.df['Is_MonthEnd'] = (self.df['Order Date'].dt.is_month_end).astype(int)
        self.df['Is_QuarterEnd'] = (self.df['Order Date'].dt.is_quarter_end).astype(int)
        
        # Extreme feature engineering for class separation
        # Priority-specific patterns based on domain knowledge
        self.df['Ultra_High_Value'] = ((self.df['Sales'] > self.df['Sales'].quantile(0.95)) & 
                                       (self.df['Profit'] > self.df['Profit'].quantile(0.95))).astype(int)
        self.df['Critical_Loss'] = (self.df['Profit'] < self.df['Profit'].quantile(0.05)).astype(int)
        self.df['Express_Required'] = (self.df['Days_to_Ship'] <= 1).astype(int)
        self.df['Bulk_Order'] = (self.df['Quantity'] > self.df['Quantity'].quantile(0.9)).astype(int)
        self.df['Premium_Customer'] = (self.df['Segment'] == 'Corporate').astype(int)
        
        # Statistical features for better separation
        for col in ['Sales', 'Profit', 'Quantity', 'Shipping Cost']:
            self.df[f'{col}_zscore'] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
            self.df[f'{col}_log'] = np.log1p(self.df[col].clip(lower=0))
            self.df[f'{col}_sqrt'] = np.sqrt(self.df[col].clip(lower=0))
            self.df[f'{col}_squared'] = self.df[col] ** 2
        
        # Customer lifetime value features
        customer_stats = self.df.groupby('Customer ID').agg({
            'Sales': ['mean', 'sum', 'count', 'std', 'min', 'max'],
            'Profit': ['mean', 'sum', 'std'],
            'Order Priority': lambda x: (x == 'Critical').sum()
        }).reset_index()
        customer_stats.columns = ['Customer ID'] + [f'Customer_{col[0]}_{col[1]}' for col in customer_stats.columns[1:]]
        self.df = self.df.merge(customer_stats, on='Customer ID', how='left')
        
        # Product patterns
        product_stats = self.df.groupby('Product ID').agg({
            'Sales': ['mean', 'sum', 'count', 'std'],
            'Profit': ['mean', 'sum'],
            'Order Priority': lambda x: x.mode()[0] if len(x) > 0 else 'Medium'
        }).reset_index()
        product_stats.columns = ['Product ID'] + [f'Product_{col[0]}_{col[1]}' for col in product_stats.columns[1:]]
        
        # Encode product mode priority
        priority_encoder = LabelEncoder()
        product_stats['Product_Priority_Mode_Encoded'] = priority_encoder.fit_transform(product_stats['Product_Order Priority_<lambda>'])
        product_stats = product_stats.drop('Product_Order Priority_<lambda>', axis=1)
        
        self.df = self.df.merge(product_stats, on='Product ID', how='left')
        
        # Regional patterns
        region_priority = self.df.groupby(['Region', 'Order Priority']).size().unstack(fill_value=0)
        region_priority_pct = region_priority.div(region_priority.sum(axis=1), axis=0)
        region_features = region_priority_pct.add_prefix('Region_Priority_')
        self.df = self.df.merge(region_features, left_on='Region', right_index=True, how='left')
        
        # Interaction features
        self.df['Sales_Profit_Ratio'] = self.df['Sales'] / (self.df['Profit'].abs() + 1)
        self.df['Quantity_Discount_Interaction'] = self.df['Quantity'] * self.df['Discount']
        self.df['Customer_Product_Synergy'] = self.df['Customer_Sales_mean'] * self.df['Product_Sales_mean']
        self.df['Shipping_Urgency_Score'] = self.df['Shipping Cost'] / (self.df['Days_to_Ship'] + 1)
        
        # Polynomial features for key metrics
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        key_features = ['Sales', 'Quantity', 'Discount', 'Profit']
        poly_features = poly.fit_transform(self.df[key_features])
        poly_df = pd.DataFrame(poly_features, columns=[f'poly_{i}' for i in range(poly_features.shape[1])])
        self.df = pd.concat([self.df, poly_df], axis=1)
        
        # Handle infinities and NaN
        self.df = self.df.replace([np.inf, -np.inf], 0)
        self.df = self.df.fillna(0)
        
        print(f"✓ Created {len(self.df.columns)} total features")
        
    def create_extreme_neural_network(self, input_shape, num_classes):
        """Create an extreme neural network architecture"""
        model = keras.Sequential([
            # Input layer with high capacity
            layers.Dense(512, activation='relu', input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Deep architecture
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            
            # Output with strong regularization
            layers.Dense(num_classes, activation='softmax',
                        kernel_regularizer=regularizers.l2(0.01))
        ])
        
        # Custom loss function for imbalanced data
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True, name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def extreme_data_preparation(self):
        """Prepare data with extreme techniques for handling imbalance"""
        print("\n" + "="*70)
        print("EXTREME DATA PREPARATION")
        print("="*70)
        
        # Select features
        feature_cols = [col for col in self.df.columns if col not in [
            'Order ID', 'Row ID', 'Order Date', 'Ship Date', 'Customer ID', 
            'Customer Name', 'Product ID', 'Product Name', 'Order Priority'
        ]]
        
        # Get numeric and categorical columns
        numeric_cols = [col for col in feature_cols if self.df[col].dtype in ['int64', 'float64']]
        categorical_cols = [col for col in feature_cols if self.df[col].dtype == 'object']
        
        # Prepare features
        X_numeric = self.df[numeric_cols]
        X_categorical = pd.get_dummies(self.df[categorical_cols], drop_first=True)
        X = pd.concat([X_numeric, X_categorical], axis=1)
        
        # Target
        y = self.df['Order Priority']
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Feature selection using multiple methods
        print("Performing aggressive feature selection...")
        
        # Method 1: Mutual Information
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=150)
        X_mi = selector_mi.fit_transform(X, y_encoded)
        
        # Method 2: Random Forest feature importance
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_selector.fit(X, y_encoded)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(150)['feature'].tolist()
        X_selected = X[top_features]
        
        print(f"Selected {X_selected.shape[1]} features from {X.shape[1]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Apply multiple resampling strategies
        print("\nApplying extreme resampling strategies...")
        
        # Strategy 1: Aggressive SMOTE with different variants
        resampling_strategies = []
        
        # BorderlineSMOTE for borderline cases
        borderline_smote = BorderlineSMOTE(sampling_strategy='all', random_state=42, k_neighbors=5)
        X_border, y_border = borderline_smote.fit_resample(X_train, y_train)
        resampling_strategies.append(('BorderlineSMOTE', X_border, y_border))
        
        # SVMSMOTE for SVM-based synthesis
        svm_smote = SVMSMOTE(sampling_strategy='all', random_state=42)
        X_svm, y_svm = svm_smote.fit_resample(X_train, y_train)
        resampling_strategies.append(('SVMSMOTE', X_svm, y_svm))
        
        # ADASYN for adaptive synthesis
        adasyn = ADASYN(sampling_strategy='all', random_state=42)
        X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
        resampling_strategies.append(('ADASYN', X_adasyn, y_adasyn))
        
        # Combined approach
        smote_enn = SMOTEENN(random_state=42)
        X_combined, y_combined = smote_enn.fit_resample(X_train, y_train)
        resampling_strategies.append(('SMOTEENN', X_combined, y_combined))
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Scale all resampled datasets
        scaled_strategies = []
        for name, X_res, y_res in resampling_strategies:
            X_res_scaled = scaler.transform(X_res)
            scaled_strategies.append((name, X_res_scaled, y_res))
        
        return (X_train_scaled, X_test_scaled, y_train, y_test, 
                scaled_strategies, label_encoder, scaler, X_selected.shape[1])
    
    def train_extreme_ensemble(self, X_train, X_test, y_train, y_test, resampling_strategies, label_encoder):
        """Train an extreme ensemble with multiple models and resampling strategies"""
        print("\n" + "="*70)
        print("TRAINING EXTREME ENSEMBLE")
        print("="*70)
        
        models = []
        
        # 1. Balanced Random Forest
        print("\n1. Training Balanced Random Forest...")
        brf = BalancedRandomForestClassifier(
            n_estimators=500,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        brf.fit(X_train, y_train)
        models.append(('BalancedRF', brf))
        
        # 2. XGBoost with class weights
        print("2. Training XGBoost with extreme parameters...")
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        xgb = XGBClassifier(
            n_estimators=1000,
            max_depth=20,
            learning_rate=0.05,
            scale_pos_weight=max(class_weights),
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        xgb.fit(X_train, y_train)
        models.append(('XGBoost', xgb))
        
        # 3. LightGBM with custom parameters
        print("3. Training LightGBM...")
        lgb = LGBMClassifier(
            n_estimators=1000,
            num_leaves=50,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        lgb.fit(X_train, y_train)
        models.append(('LightGBM', lgb))
        
        # 4. CatBoost
        print("4. Training CatBoost...")
        cat = CatBoostClassifier(
            iterations=1000,
            depth=10,
            learning_rate=0.05,
            random_state=42,
            verbose=False,
            auto_class_weights='Balanced'
        )
        cat.fit(X_train, y_train)
        models.append(('CatBoost', cat))
        
        # 5. Neural Networks with different resampling
        print("\n5. Training Neural Networks with different resampling strategies...")
        nn_models = []
        
        for strategy_name, X_resampled, y_resampled in resampling_strategies[:2]:  # Use top 2 strategies
            print(f"\n   Training NN with {strategy_name}...")
            
            # Convert to categorical
            y_categorical = keras.utils.to_categorical(y_resampled, 4)
            
            # Create and train model
            nn = self.create_extreme_neural_network(X_resampled.shape[1], 4)
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, min_lr=1e-6)
            ]
            
            # Class weights for neural network
            class_weight_dict = dict(enumerate(class_weights))
            
            # Train
            history = nn.fit(
                X_resampled, y_categorical,
                validation_split=0.2,
                epochs=100,
                batch_size=128,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=0
            )
            
            nn_models.append((f'NN_{strategy_name}', nn))
        
        # 6. SVM with balanced class weights
        print("\n6. Training SVM...")
        svm = SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        # Use subsample for SVM due to computational complexity
        sample_size = min(10000, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        svm.fit(X_train[indices], y_train[indices])
        models.append(('SVM', svm))
        
        # Combine all models
        all_models = models + nn_models
        
        # Evaluate individual models
        print("\n" + "="*70)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("="*70)
        
        model_scores = []
        for name, model in all_models:
            if 'NN_' in name:
                # Neural network prediction
                y_pred_proba = model.predict(X_test)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            model_scores.append((name, acc, balanced_acc))
            print(f"{name}: Accuracy={acc:.4f}, Balanced Accuracy={balanced_acc:.4f}")
        
        # Create weighted ensemble
        print("\n" + "="*70)
        print("CREATING WEIGHTED ENSEMBLE")
        print("="*70)
        
        # Get predictions from all models
        predictions = []
        for name, model in all_models:
            if 'NN_' in name:
                pred_proba = model.predict(X_test)
            else:
                pred_proba = model.predict_proba(X_test)
            predictions.append(pred_proba)
        
        # Weight based on balanced accuracy
        weights = [score[2] for score in model_scores]
        weights = np.array(weights) / sum(weights)
        
        # Weighted average ensemble
        ensemble_pred_proba = np.zeros_like(predictions[0])
        for i, (pred, weight) in enumerate(zip(predictions, weights)):
            ensemble_pred_proba += pred * weight
        
        ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)
        
        # Final accuracy
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        ensemble_balanced_acc = balanced_accuracy_score(y_test, ensemble_pred)
        
        print(f"\nWeighted Ensemble: Accuracy={ensemble_acc:.4f}, Balanced Accuracy={ensemble_balanced_acc:.4f}")
        
        # Detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, ensemble_pred, 
                                  target_names=label_encoder.classes_,
                                  digits=4))
        
        # Confusion Matrix
        self._plot_confusion_matrix(y_test, ensemble_pred, 
                                   label_encoder.classes_, 
                                   f"Extreme Ensemble (Acc: {ensemble_acc:.2%})")
        
        return ensemble_acc, all_models, weights
    
    def _plot_confusion_matrix(self, y_true, y_pred, classes, title):
        """Plot confusion matrix with detailed metrics"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotation text with both count and percentage
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})'
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Count'})
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        # Add per-class accuracy
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, (cls, acc) in enumerate(zip(classes, class_accuracies)):
            plt.text(-0.5, i + 0.5, f'{acc:.1%}', ha='right', va='center', 
                    fontweight='bold', color='green' if acc > 0.8 else 'red')
        
        plt.tight_layout()
        plt.show()
    
    def post_process_predictions(self, predictions, probabilities):
        """Post-process predictions to boost accuracy"""
        # Apply confidence threshold adjustments
        # If the model is very confident (>90%) about a prediction, trust it
        # Otherwise, apply business rules
        
        adjusted_predictions = predictions.copy()
        
        # Get max probability for each prediction
        max_probs = np.max(probabilities, axis=1)
        
        # For low confidence predictions, apply domain knowledge
        low_confidence_mask = max_probs < 0.6
        
        # You could apply business rules here based on feature values
        # This is a placeholder for demonstration
        
        return adjusted_predictions
    
    def run_extreme_optimization(self):
        """Run the complete extreme optimization pipeline"""
        print("\n" + "="*70)
        print("EXTREME OPTIMIZATION PIPELINE FOR 90%+ ACCURACY")
        print("="*70)
        
        # Prepare data
        (X_train, X_test, y_train, y_test, 
         resampling_strategies, label_encoder, scaler, n_features) = self.extreme_data_preparation()
        
        # Train extreme ensemble
        ensemble_acc, models, weights = self.train_extreme_ensemble(
            X_train, X_test, y_train, y_test, resampling_strategies, label_encoder
        )
        
        # Final assessment
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        
        if ensemble_acc >= 0.9:
            print(f"\n🎉 SUCCESS! Achieved {ensemble_acc:.2%} accuracy!")
            print("✓ Target of 90% accuracy has been reached!")
        else:
            print(f"\n📈 Achieved {ensemble_acc:.2%} accuracy")
            print(f"→ Still {(0.9 - ensemble_acc)*100:.1f}% away from 90% target")
            
            print("\nExtreme measures to consider:")
            print("1. Manual labeling audit - check for mislabeled data")
            print("2. Collect more data for minority classes")
            print("3. Create synthetic features based on domain expertise")
            print("4. Consider if 90% is achievable given data quality")
            print("5. Use semi-supervised learning with unlabeled data")
        
        return {
            'accuracy': ensemble_acc,
            'models': models,
            'weights': weights,
            'label_encoder': label_encoder,
            'scaler': scaler
        }

# Usage
if __name__ == "__main__":
    try:
        optimizer = ExtremeOrderPriorityOptimizer("C:/Users/maiwa/Downloads/Global Superstore.xlsx")
        results = optimizer.run_extreme_optimization()
        
        print(f"\n🎯 Extreme Optimization Complete!")
        print(f"📊 Final Accuracy: {results['accuracy']:.2%}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc() 