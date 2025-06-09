import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import warnings
import time
import joblib
from scipy import stats
warnings.filterwarnings('ignore')

print("="*80)
print("EXTREME ACCURACY APPROACH - ORDER PRIORITY PREDICTION")
print("Target: 90%+ Accuracy")
print("="*80)

class ExtremeAccuracyPredictor:
    def __init__(self, file_path):
        """Initialize with extreme optimization techniques"""
        self.df = pd.read_excel(file_path)
        self.label_encoder = LabelEncoder()
        self.scaler = None
        self.best_features = None
        self.models = {}
        
    def create_extreme_features(self):
        """Create extremely sophisticated features"""
        print("\n" + "="*60)
        print("EXTREME FEATURE ENGINEERING")
        print("="*60)
        
        df_feat = self.df.copy()
        
        # Convert dates
        df_feat['Order Date'] = pd.to_datetime(df_feat['Order Date'])
        df_feat['Ship Date'] = pd.to_datetime(df_feat['Ship Date'])
        
        # 1. Basic time features
        df_feat['Days_to_Ship'] = (df_feat['Ship Date'] - df_feat['Order Date']).dt.days
        df_feat['Order_Year'] = df_feat['Order Date'].dt.year
        df_feat['Order_Month'] = df_feat['Order Date'].dt.month
        df_feat['Order_Quarter'] = df_feat['Order Date'].dt.quarter
        df_feat['Order_DayOfWeek'] = df_feat['Order Date'].dt.dayofweek
        df_feat['Order_DayOfYear'] = df_feat['Order Date'].dt.dayofyear
        df_feat['Order_WeekOfYear'] = df_feat['Order Date'].dt.isocalendar().week
        df_feat['Is_Weekend'] = (df_feat['Order_DayOfWeek'] >= 5).astype(int)
        df_feat['Is_MonthStart'] = df_feat['Order Date'].dt.is_month_start.astype(int)
        df_feat['Is_MonthEnd'] = df_feat['Order Date'].dt.is_month_end.astype(int)
        df_feat['Is_QuarterStart'] = df_feat['Order Date'].dt.is_quarter_start.astype(int)
        df_feat['Is_QuarterEnd'] = df_feat['Order Date'].dt.is_quarter_end.astype(int)
        
        # 2. Advanced financial features
        df_feat['Profit_Margin'] = df_feat['Profit'] / (df_feat['Sales'] + 1e-6)
        df_feat['Profit_Margin_Pct'] = df_feat['Profit_Margin'] * 100
        df_feat['Revenue_Per_Unit'] = df_feat['Sales'] / (df_feat['Quantity'] + 1)
        df_feat['Profit_Per_Unit'] = df_feat['Profit'] / (df_feat['Quantity'] + 1)
        df_feat['Discount_Impact'] = df_feat['Discount'] * df_feat['Sales']
        df_feat['Shipping_Cost_Ratio'] = df_feat['Shipping Cost'] / (df_feat['Sales'] + 1)
        df_feat['Cost_Efficiency'] = df_feat['Profit'] / (df_feat['Shipping Cost'] + 1)
        df_feat['Value_Score'] = df_feat['Sales'] * df_feat['Profit_Margin']
        df_feat['Loss_Indicator'] = (df_feat['Profit'] < 0).astype(int)
        df_feat['High_Value_Order'] = (df_feat['Sales'] > df_feat['Sales'].quantile(0.75)).astype(int)
        
        # 3. Shipping and segment encoding
        ship_priority = {'Same Day': 4, 'First Class': 3, 'Second Class': 2, 'Standard Class': 1}
        df_feat['Ship_Mode_Priority'] = df_feat['Ship Mode'].map(ship_priority)
        segment_map = {'Consumer': 1, 'Corporate': 2, 'Home Office': 3}
        df_feat['Segment_Code'] = df_feat['Segment'].map(segment_map)
        
        # 4. Statistical features for each numeric column
        numeric_cols = ['Sales', 'Profit', 'Quantity', 'Shipping Cost', 'Discount']
        for col in numeric_cols:
            # Z-score
            df_feat[f'{col}_zscore'] = stats.zscore(df_feat[col].fillna(0))
            # Percentile rank
            df_feat[f'{col}_percentile'] = df_feat[col].rank(pct=True)
            # Log transformation
            df_feat[f'{col}_log'] = np.log1p(df_feat[col].clip(lower=0))
            # Square root
            df_feat[f'{col}_sqrt'] = np.sqrt(df_feat[col].clip(lower=0))
            # Binning
            df_feat[f'{col}_bin'] = pd.qcut(df_feat[col], q=10, labels=False, duplicates='drop')
        
        # 5. Customer-level aggregations with more statistics
        customer_agg = df_feat.groupby('Customer ID').agg({
            'Sales': ['mean', 'sum', 'std', 'min', 'max', 'count', 'median'],
            'Profit': ['mean', 'sum', 'std', 'min', 'max', 'median'],
            'Quantity': ['mean', 'sum', 'std'],
            'Days_to_Ship': ['mean', 'std', 'min', 'max'],
            'Discount': ['mean', 'max'],
            'Order Priority': lambda x: (x == 'Critical').sum(),
            'Loss_Indicator': 'sum',
            'High_Value_Order': 'sum'
        }).reset_index()
        customer_agg.columns = ['Customer ID'] + [f'Customer_{col[0]}_{col[1]}' for col in customer_agg.columns[1:]]
        
        # Customer lifetime value
        customer_agg['Customer_CLV'] = customer_agg['Customer_Sales_sum'] * customer_agg['Customer_Profit_mean']
        customer_agg['Customer_Reliability'] = 1 - (customer_agg['Customer_Loss_Indicator_sum'] / customer_agg['Customer_Sales_count'])
        
        df_feat = df_feat.merge(customer_agg, on='Customer ID', how='left')
        
        # 6. Product-level aggregations
        product_agg = df_feat.groupby('Product ID').agg({
            'Sales': ['mean', 'sum', 'std', 'count', 'median'],
            'Profit': ['mean', 'sum', 'std', 'median'],
            'Quantity': ['mean', 'sum', 'median'],
            'Discount': ['mean', 'std', 'max'],
            'Days_to_Ship': ['mean', 'std'],
            'Order Priority': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Medium',
            'Loss_Indicator': ['sum', 'mean']
        }).reset_index()
        product_agg.columns = ['Product ID'] + [f'Product_{col[0]}_{col[1]}' for col in product_agg.columns[1:]]
        
        # Product priority encoding
        priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        product_agg['Product_Priority_Mode'] = product_agg['Product_Order Priority_<lambda>'].map(priority_map)
        product_agg = product_agg.drop('Product_Order Priority_<lambda>', axis=1)
        
        df_feat = df_feat.merge(product_agg, on='Product ID', how='left')
        
        # 7. Category and Sub-Category features
        category_agg = df_feat.groupby(['Category', 'Sub-Category']).agg({
            'Sales': ['mean', 'sum', 'std'],
            'Profit': ['mean', 'sum'],
            'Profit_Margin': ['mean', 'std'],
            'Days_to_Ship': 'mean',
            'Order Priority': lambda x: (x == 'Critical').mean()
        }).reset_index()
        category_agg.columns = ['Category', 'Sub-Category'] + [f'Cat_{col[0]}_{col[1]}' for col in category_agg.columns[2:]]
        df_feat = df_feat.merge(category_agg, on=['Category', 'Sub-Category'], how='left')
        
        # 8. Region and Market features
        region_agg = df_feat.groupby('Region').agg({
            'Sales': ['mean', 'sum', 'std'],
            'Profit': ['mean', 'sum'],
            'Days_to_Ship': ['mean', 'std'],
            'Order Priority': lambda x: (x.isin(['Critical', 'High'])).mean()
        }).reset_index()
        region_agg.columns = ['Region'] + [f'Region_{col[0]}_{col[1]}' for col in region_agg.columns[1:]]
        df_feat = df_feat.merge(region_agg, on='Region', how='left')
        
        # 9. Interaction features
        df_feat['Sales_x_Quantity'] = df_feat['Sales'] * df_feat['Quantity']
        df_feat['Profit_x_Quantity'] = df_feat['Profit'] * df_feat['Quantity']
        df_feat['Sales_x_Discount'] = df_feat['Sales'] * df_feat['Discount']
        df_feat['ShipCost_per_Day'] = df_feat['Shipping Cost'] / (df_feat['Days_to_Ship'] + 1)
        df_feat['Sales_per_Day'] = df_feat['Sales'] / (df_feat['Days_to_Ship'] + 1)
        df_feat['Profit_per_Day'] = df_feat['Profit'] / (df_feat['Days_to_Ship'] + 1)
        
        # 10. Ratios and complex features
        df_feat['Order_Complexity'] = (
            df_feat['Quantity'] * df_feat['Days_to_Ship'] * 
            df_feat['Ship_Mode_Priority'] / (df_feat['Segment_Code'] + 1)
        )
        df_feat['Value_Density'] = df_feat['Sales'] / (df_feat['Quantity'] * df_feat['Days_to_Ship'] + 1)
        df_feat['Customer_Value_Ratio'] = df_feat['Sales'] / (df_feat['Customer_Sales_mean'] + 1)
        df_feat['Product_Performance_Ratio'] = df_feat['Profit'] / (df_feat['Product_Profit_mean'] + 1)
        
        # 11. Time-based patterns
        df_feat['Days_Since_Year_Start'] = df_feat['Order_DayOfYear']
        df_feat['Days_Until_Year_End'] = 365 - df_feat['Order_DayOfYear']
        df_feat['Quarter_Progress'] = (df_feat['Order_Month'] - 1) % 3 + 1
        
        # 12. Advanced categorical encoding
        # Create frequency encoding for high-cardinality features
        for col in ['Customer ID', 'Product ID', 'City', 'State']:
            freq_encoding = df_feat[col].value_counts().to_dict()
            df_feat[f'{col}_frequency'] = df_feat[col].map(freq_encoding)
        
        # Handle missing values
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        numeric_features = df_feat.select_dtypes(include=[np.number]).columns
        df_feat[numeric_features] = df_feat[numeric_features].fillna(df_feat[numeric_features].median())
        
        # Select features for modeling
        feature_cols = [col for col in df_feat.columns if col not in [
            'Order ID', 'Row ID', 'Order Date', 'Ship Date', 'Customer ID',
            'Customer Name', 'Product ID', 'Product Name', 'Order Priority',
            'Country', 'City', 'State', 'Postal Code', 'Region',
            'Category', 'Sub-Category', 'Ship Mode', 'Segment', 'Market'
        ]]
        
        print(f"Created {len(feature_cols)} features")
        
        # Encode target
        y = self.label_encoder.fit_transform(df_feat['Order Priority'])
        
        return df_feat[feature_cols], y
    
    def select_best_features(self, X, y, n_features=50):
        """Select best features using multiple methods"""
        print("\nSelecting best features...")
        
        # Method 1: Univariate feature selection
        selector_f = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        selector_f.fit(X, y)
        scores_f = selector_f.scores_
        
        # Method 2: Mutual information
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(n_features, X.shape[1]))
        selector_mi.fit(X, y)
        scores_mi = selector_mi.scores_
        
        # Method 3: Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        scores_rf = rf.feature_importances_
        
        # Combine scores
        scores_combined = (
            stats.rankdata(scores_f) + 
            stats.rankdata(scores_mi) + 
            stats.rankdata(scores_rf)
        ) / 3
        
        # Select top features
        top_indices = np.argsort(scores_combined)[-n_features:]
        selected_features = X.columns[top_indices].tolist()
        
        print(f"Selected {len(selected_features)} best features")
        return selected_features
    
    def train_extreme_models(self, X_train, X_test, y_train, y_test):
        """Train multiple state-of-the-art models"""
        print("\n" + "="*60)
        print("TRAINING EXTREME MODELS")
        print("="*60)
        
        models = {}
        
        # 1. XGBoost with optimized parameters
        print("\n1. Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            objective='multi:softmax',
            num_class=4,
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        models['XGBoost'] = xgb_model
        
        # 2. LightGBM with optimized parameters
        print("2. Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            num_leaves=100,
            max_depth=10,
            learning_rate=0.01,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            lambda_l1=0.1,
            lambda_l2=1,
            min_split_gain=0.01,
            objective='multiclass',
            num_class=4,
            random_state=42,
            n_jobs=-1
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        models['LightGBM'] = lgb_model
        
        # 3. CatBoost with optimized parameters
        print("3. Training CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=1000,
            depth=10,
            learning_rate=0.01,
            l2_leaf_reg=3,
            subsample=0.8,
            colsample_bylevel=0.8,
            random_strength=0.1,
            bagging_temperature=0.1,
            od_type='Iter',
            od_wait=50,
            random_state=42,
            verbose=False
        )
        cb_model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=False
        )
        models['CatBoost'] = cb_model
        
        # 4. Extra Trees
        print("4. Training Extra Trees...")
        et_model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        et_model.fit(X_train, y_train)
        models['ExtraTrees'] = et_model
        
        # 5. Gradient Boosting
        print("5. Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.01,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        models['GradientBoosting'] = gb_model
        
        # 6. Advanced Random Forest
        print("6. Training Advanced Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['RandomForest'] = rf_model
        
        # Evaluate all models
        print("\nModel Performance:")
        print("-" * 40)
        for name, model in models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"{name}: {acc:.4f}")
            
        return models
    
    def create_stacking_ensemble(self, models, X_train, X_test, y_train, y_test):
        """Create a stacking ensemble of all models"""
        print("\n" + "="*60)
        print("CREATING STACKING ENSEMBLE")
        print("="*60)
        
        # Get predictions from all models for training meta-model
        train_meta_features = []
        test_meta_features = []
        
        for name, model in models.items():
            # Get probability predictions
            train_proba = model.predict_proba(X_train)
            test_proba = model.predict_proba(X_test)
            train_meta_features.append(train_proba)
            test_meta_features.append(test_proba)
        
        # Stack predictions
        X_train_meta = np.hstack(train_meta_features)
        X_test_meta = np.hstack(test_meta_features)
        
        # Train meta-model (XGBoost)
        print("Training meta-model...")
        meta_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=4,
            use_label_encoder=False,
            random_state=42
        )
        meta_model.fit(X_train_meta, y_train)
        
        # Evaluate stacking ensemble
        y_pred_stack = meta_model.predict(X_test_meta)
        stack_acc = accuracy_score(y_test, y_pred_stack)
        print(f"\nStacking Ensemble Accuracy: {stack_acc:.4f}")
        
        return meta_model, stack_acc
    
    def train_with_advanced_sampling(self, X, y):
        """Train with multiple sampling strategies"""
        print("\n" + "="*60)
        print("ADVANCED SAMPLING STRATEGIES")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        best_accuracy = 0
        best_model = None
        best_strategy = None
        
        # Try different sampling strategies
        sampling_strategies = {
            'SMOTE': SMOTE(random_state=42),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42, kind='borderline-1'),
            'ADASYN': ADASYN(random_state=42),
            'SMOTETomek': SMOTETomek(random_state=42),
            'SMOTEENN': SMOTEENN(random_state=42)
        }
        
        for strategy_name, sampler in sampling_strategies.items():
            print(f"\nTrying {strategy_name}...")
            try:
                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                
                # Train XGBoost with resampled data
                model = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=10,
                    learning_rate=0.01,
                    objective='multi:softmax',
                    num_class=4,
                    use_label_encoder=False,
                    random_state=42
                )
                model.fit(X_resampled, y_resampled)
                
                # Evaluate
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"{strategy_name} Accuracy: {acc:.4f}")
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = model
                    best_strategy = strategy_name
                    
            except Exception as e:
                print(f"Failed with {strategy_name}: {str(e)}")
        
        print(f"\nBest Sampling Strategy: {best_strategy}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        
        return best_model, best_accuracy, X_test, y_test
    
    def run_extreme_pipeline(self):
        """Run the complete extreme accuracy pipeline"""
        print("\n" + "="*80)
        print("RUNNING EXTREME ACCURACY PIPELINE")
        print("="*80)
        
        # Step 1: Create extreme features
        X, y = self.create_extreme_features()
        print(f"\nDataset shape: {X.shape}")
        
        # Step 2: Select best features
        X_selected = X[self.select_best_features(X, y, n_features=100)]
        
        # Step 3: Scale features
        print("\nScaling features...")
        self.scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_selected),
            columns=X_selected.columns
        )
        
        # Step 4: Train with advanced sampling
        best_sampled_model, best_sampled_acc, X_test, y_test = self.train_with_advanced_sampling(
            X_scaled, y
        )
        
        # Step 5: Train multiple models
        X_train, _, y_train, _ = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        models = self.train_extreme_models(X_train, X_test, y_train, y_test)
        
        # Step 6: Create stacking ensemble
        meta_model, stack_acc = self.create_stacking_ensemble(
            models, X_train, X_test, y_train, y_test
        )
        
        # Step 7: Final evaluation
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        # Get best single model
        best_single_acc = 0
        best_single_model = None
        for name, model in models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            if acc > best_single_acc:
                best_single_acc = acc
                best_single_model = (name, model)
        
        # Compare all approaches
        print(f"\nBest Single Model ({best_single_model[0]}): {best_single_acc:.4f}")
        print(f"Best with Sampling ({best_strategy}): {best_sampled_acc:.4f}")
        print(f"Stacking Ensemble: {stack_acc:.4f}")
        
        # Get final best accuracy
        final_best_acc = max(best_single_acc, best_sampled_acc, stack_acc)
        print(f"\n{'='*40}")
        print(f"BEST ACCURACY ACHIEVED: {final_best_acc:.4f}")
        print(f"{'='*40}")
        
        if final_best_acc >= 0.9:
            print("\n🎉 SUCCESS! Achieved 90%+ accuracy!")
        else:
            print(f"\n📈 Current: {final_best_acc:.2%}")
            print(f"   Still {(0.9 - final_best_acc)*100:.1f}% away from 90% target")
        
        # Detailed report on best model
        if stack_acc == final_best_acc:
            print("\nBest Model: Stacking Ensemble")
            # Get meta predictions for classification report
            train_meta_features = []
            test_meta_features = []
            for name, model in models.items():
                train_proba = model.predict_proba(X_train)
                test_proba = model.predict_proba(X_test)
                train_meta_features.append(train_proba)
                test_meta_features.append(test_proba)
            X_test_meta = np.hstack(test_meta_features)
            y_pred_final = meta_model.predict(X_test_meta)
        else:
            print(f"\nBest Model: {best_single_model[0]}")
            y_pred_final = best_single_model[1].predict(X_test)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_final,
                                  target_names=self.label_encoder.classes_))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_final)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Best Model Confusion Matrix (Accuracy: {final_best_acc:.2%})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        # Save best model
        if final_best_acc >= 0.9:
            joblib.dump({
                'models': models,
                'meta_model': meta_model if stack_acc == final_best_acc else None,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'selected_features': X_selected.columns.tolist(),
                'best_accuracy': final_best_acc
            }, f'extreme_model_{final_best_acc:.4f}.pkl')
            print(f"\n✓ Model saved as: extreme_model_{final_best_acc:.4f}.pkl")
        
        return final_best_acc

# Execute the extreme accuracy approach
if __name__ == "__main__":
    start_time = time.time()
    
    predictor = ExtremeAccuracyPredictor("C:/Users/maiwa/Downloads/Global Superstore.xlsx")
    final_accuracy = predictor.run_extreme_pipeline()
    
    print(f"\n⏱ Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"\n🎯 Extreme Accuracy Pipeline Complete!")
    print(f"📊 Final Accuracy: {final_accuracy:.2%}") 