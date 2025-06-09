import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Adamax, Nadam
import keras_tuner as kt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class AdvancedOrderPriorityPredictor:
    def __init__(self, file_path):
        """Initialize the Advanced Order Priority Predictor"""
        self.df = pd.read_excel(file_path)
        self.prepare_advanced_features()
        
    def prepare_advanced_features(self):
        """Prepare advanced features with extensive engineering"""
        print("="*70)
        print("ADVANCED FEATURE ENGINEERING FOR 90%+ ACCURACY")
        print("="*70)
        
        # Convert dates
        self.df['Order Date'] = pd.to_datetime(self.df['Order Date'])
        self.df['Ship Date'] = pd.to_datetime(self.df['Ship Date'])
        
        # Core financial metrics
        self.df['Profit_Margin'] = (self.df['Profit'] / self.df['Sales'].replace(0, 1)) * 100
        self.df['Revenue_Per_Unit'] = self.df['Sales'] / self.df['Quantity'].replace(0, 1)
        self.df['Shipping_Efficiency'] = self.df['Sales'] / self.df['Shipping Cost'].replace(0, 1)
        self.df['Discount_Impact'] = self.df['Discount'] * self.df['Sales']
        self.df['Profit_Per_Unit'] = self.df['Profit'] / self.df['Quantity'].replace(0, 1)
        
        # Advanced temporal features
        self.df['Days_to_Ship'] = (self.df['Ship Date'] - self.df['Order Date']).dt.days
        self.df['Order_Month'] = self.df['Order Date'].dt.month
        self.df['Order_Quarter'] = self.df['Order Date'].dt.quarter
        self.df['Order_DayOfWeek'] = self.df['Order Date'].dt.dayofweek
        self.df['Order_Week'] = self.df['Order Date'].dt.isocalendar().week
        self.df['Is_Weekend'] = (self.df['Order_DayOfWeek'] >= 5).astype(int)
        self.df['Is_MonthEnd'] = (self.df['Order Date'].dt.is_month_end).astype(int)
        
        # Performance indicators
        self.df['Is_Profitable'] = (self.df['Profit'] > 0).astype(int)
        self.df['High_Value_Order'] = (self.df['Sales'] > self.df['Sales'].quantile(0.75)).astype(int)
        self.df['Large_Quantity'] = (self.df['Quantity'] > self.df['Quantity'].quantile(0.75)).astype(int)
        self.df['High_Discount'] = (self.df['Discount'] > 0.2).astype(int)
        self.df['Express_Shipping'] = (self.df['Days_to_Ship'] <= 2).astype(int)
        
        # Customer behavior patterns
        customer_stats = self.df.groupby('Customer ID').agg({
            'Sales': ['mean', 'sum', 'count', 'std'],
            'Profit': ['mean', 'sum'],
            'Quantity': ['mean', 'sum'],
            'Days_to_Ship': 'mean'
        }).reset_index()
        customer_stats.columns = ['Customer ID', 'Customer_Avg_Sales', 'Customer_Total_Sales', 
                                 'Customer_Order_Count', 'Customer_Sales_Std',
                                 'Customer_Avg_Profit', 'Customer_Total_Profit',
                                 'Customer_Avg_Quantity', 'Customer_Total_Quantity',
                                 'Customer_Avg_Ship_Days']
        self.df = self.df.merge(customer_stats, on='Customer ID', how='left')
        
        # Product intelligence
        product_stats = self.df.groupby('Product ID').agg({
            'Sales': ['mean', 'sum', 'count'],
            'Profit': ['mean', 'sum'],
            'Quantity': ['mean', 'sum'],
            'Discount': 'mean',
            'Days_to_Ship': 'mean'
        }).reset_index()
        product_stats.columns = ['Product ID', 'Product_Avg_Sales', 'Product_Total_Sales',
                               'Product_Order_Count', 'Product_Avg_Profit', 'Product_Total_Profit',
                               'Product_Avg_Quantity', 'Product_Total_Quantity',
                               'Product_Avg_Discount', 'Product_Avg_Ship_Days']
        self.df = self.df.merge(product_stats, on='Product ID', how='left')
        
        # Category and segment patterns
        category_stats = self.df.groupby('Category').agg({
            'Sales': 'mean',
            'Profit': 'mean',
            'Days_to_Ship': 'mean'
        }).reset_index()
        category_stats.columns = ['Category', 'Category_Avg_Sales', 'Category_Avg_Profit', 'Category_Avg_Ship_Days']
        self.df = self.df.merge(category_stats, on='Category', how='left')
        
        # Geographic patterns
        region_stats = self.df.groupby('Region').agg({
            'Sales': 'mean',
            'Days_to_Ship': 'mean'
        }).reset_index()
        region_stats.columns = ['Region', 'Region_Avg_Sales', 'Region_Avg_Ship_Days']
        self.df = self.df.merge(region_stats, on='Region', how='left')
        
        # Interaction features
        self.df['Sales_Quantity_Interaction'] = self.df['Sales'] * self.df['Quantity']
        self.df['Profit_Shipping_Ratio'] = self.df['Profit'] / (self.df['Shipping Cost'] + 1)
        self.df['Customer_Product_Affinity'] = self.df['Customer_Avg_Sales'] * self.df['Product_Avg_Sales']
        
        # Handle infinities and NaN
        self.df = self.df.replace([np.inf, -np.inf], 0)
        self.df = self.df.fillna(0)
        
        print(f"✓ Created {len(self.df.columns)} total features")
        print(f"✓ Dataset shape: {self.df.shape}")
        
    def create_model_builder(self, hp):
        """Create model with Keras Tuner hyperparameter optimization"""
        model = keras.Sequential()
        
        # Input layer with tunable units
        model.add(layers.Dense(
            units=hp.Int('first_layer', min_value=128, max_value=512, step=64),
            activation=hp.Choice('activation', ['relu', 'elu', 'swish']),
            input_shape=(self.input_shape,),
            kernel_regularizer=regularizers.l1_l2(
                l1=hp.Float('l1_reg', 1e-5, 1e-2, sampling='log'),
                l2=hp.Float('l2_reg', 1e-5, 1e-2, sampling='log')
            )
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))
        
        # Hidden layers with tunable depth
        for i in range(hp.Int('n_layers', 2, 5)):
            model.add(layers.Dense(
                units=hp.Int(f'layer_{i}_units', min_value=64, max_value=256, step=32),
                activation=hp.Choice(f'activation_{i}', ['relu', 'elu', 'swish']),
                kernel_regularizer=regularizers.l1_l2(
                    l1=hp.Float(f'l1_reg_{i}', 1e-5, 1e-2, sampling='log'),
                    l2=hp.Float(f'l2_reg_{i}', 1e-5, 1e-2, sampling='log')
                )
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(hp.Float(f'dropout_{i+2}', 0.1, 0.4, step=0.1)))
        
        # Output layer
        model.add(layers.Dense(4, activation='softmax'))
        
        # Compile with tunable optimizer
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
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True)]
        )
        
        return model
    
    def advanced_data_preparation(self):
        """Prepare data with advanced techniques"""
        print("\nPreparing data with advanced techniques...")
        
        # Select all numeric features
        numeric_features = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64'] 
                           and col not in ['Order ID', 'Row ID']]
        
        # Categorical features
        categorical_features = ['Segment', 'Category', 'Sub-Category', 'Region', 
                               'Market', 'Ship Mode', 'Country', 'City', 'State']
        
        # Prepare features
        X_numeric = self.df[numeric_features]
        X_categorical = pd.get_dummies(self.df[categorical_features], drop_first=True)
        X = pd.concat([X_numeric, X_categorical], axis=1)
        
        # Target
        y = self.df['Order Priority']
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Feature selection using mutual information
        print("Performing feature selection...")
        selector = SelectKBest(score_func=mutual_info_classif, k=min(100, X.shape[1]))
        X_selected = selector.fit_transform(X, y_encoded)
        selected_features = X.columns[selector.get_support()]
        print(f"Selected {len(selected_features)} most important features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Apply SMOTE-Tomek for better class balance
        print("Applying SMOTE-Tomek for class balancing...")
        smote_tomek = SMOTETomek(random_state=42)
        X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
        print(f"Balanced training set size: {X_train_balanced.shape[0]}")
        
        # Robust scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to categorical
        y_train_categorical = keras.utils.to_categorical(y_train_balanced, 4)
        y_test_categorical = keras.utils.to_categorical(y_test, 4)
        
        self.input_shape = X_train_scaled.shape[1]
        
        return (X_train_scaled, X_test_scaled, y_train_categorical, 
                y_test_categorical, label_encoder, scaler)
    
    def create_ensemble_model(self, X_train, y_train, X_test, y_test, label_encoder):
        """Create an ensemble of models for better accuracy"""
        print("\n" + "="*70)
        print("CREATING ENSEMBLE MODEL FOR MAXIMUM ACCURACY")
        print("="*70)
        
        # Convert categorical back to labels for ensemble
        y_train_labels = np.argmax(y_train, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        # 1. Best Neural Network from Keras Tuner
        print("\n1. Training optimized Neural Network...")
        tuner = kt.RandomSearch(
            self.create_model_builder,
            objective=kt.Objective("val_accuracy", direction="max"),
            max_trials=20,
            executions_per_trial=2,
            directory='keras_tuner',
            project_name='order_priority'
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)
        ]
        
        # Search for best hyperparameters
        print("Searching for best hyperparameters...")
        tuner.search(X_train, y_train, 
                    epochs=50, 
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1)
        
        # Get best model
        best_nn = tuner.get_best_models(num_models=1)[0]
        
        # Train the best model for more epochs
        print("\nTraining best model configuration...")
        history = best_nn.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=64,
            callbacks=[
                EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, min_lr=1e-6),
                ModelCheckpoint('best_nn_model.h5', monitor='val_accuracy', save_best_only=True)
            ],
            verbose=1
        )
        
        # 2. Random Forest for comparison
        print("\n2. Training Random Forest classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train_labels)
        
        # 3. Gradient Boosting
        print("\n3. Training Gradient Boosting classifier...")
        from sklearn.ensemble import GradientBoostingClassifier
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )
        gb_model.fit(X_train, y_train_labels)
        
        # Ensemble predictions
        print("\n4. Creating ensemble predictions...")
        
        # Get predictions from all models
        nn_pred_proba = best_nn.predict(X_test)
        rf_pred_proba = rf_model.predict_proba(X_test)
        gb_pred_proba = gb_model.predict_proba(X_test)
        
        # Weighted ensemble (giving more weight to better performing models)
        ensemble_pred_proba = (0.5 * nn_pred_proba + 
                              0.3 * rf_pred_proba + 
                              0.2 * gb_pred_proba)
        ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)
        
        # Individual model accuracies
        nn_pred = np.argmax(nn_pred_proba, axis=1)
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)
        
        nn_acc = accuracy_score(y_test_labels, nn_pred)
        rf_acc = accuracy_score(y_test_labels, rf_pred)
        gb_acc = accuracy_score(y_test_labels, gb_pred)
        ensemble_acc = accuracy_score(y_test_labels, ensemble_pred)
        
        print(f"\nModel Accuracies:")
        print(f"Neural Network: {nn_acc:.4f}")
        print(f"Random Forest: {rf_acc:.4f}")
        print(f"Gradient Boosting: {gb_acc:.4f}")
        print(f"Ensemble: {ensemble_acc:.4f}")
        
        # Detailed classification report for ensemble
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test_labels, ensemble_pred, 
                                  target_names=label_encoder.classes_))
        
        # Confusion Matrix
        self._plot_confusion_matrix(y_test_labels, ensemble_pred, 
                                   label_encoder.classes_, "Ensemble Model")
        
        return ensemble_acc, best_nn, rf_model, gb_model
        
    def _plot_confusion_matrix(self, y_true, y_pred, classes, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'label': 'Count'})
        plt.title(f'{model_name} - Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(0.5, -0.1, f'Accuracy: {accuracy:.2%}', 
                transform=plt.gca().transAxes, 
                ha='center', fontsize=14, weight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def custom_focal_loss(self, y_true, y_pred, gamma=2.0, alpha=0.25):
        """Focal loss for handling class imbalance"""
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -tf.math.log(p_t)
        weight = alpha_t * tf.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    def run_advanced_analysis(self):
        """Run the complete advanced analysis"""
        print("\n" + "="*70)
        print("ADVANCED ORDER PRIORITY PREDICTION - TARGETING 90%+ ACCURACY")
        print("="*70)
        
        # Prepare data
        (X_train, X_test, y_train, y_test, 
         label_encoder, scaler) = self.advanced_data_preparation()
        
        # Create and train ensemble
        ensemble_acc, nn_model, rf_model, gb_model = self.create_ensemble_model(
            X_train, y_train, X_test, y_test, label_encoder
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
            print(f"→ {(0.9 - ensemble_acc)*100:.1f}% away from 90% target")
            
            # Suggest next steps
            print("\nNext steps to reach 90%:")
            print("1. Collect more training data")
            print("2. Engineer more domain-specific features")
            print("3. Try deep learning architectures (LSTM for sequential patterns)")
            print("4. Implement active learning to focus on hard examples")
        
        return {
            'ensemble_accuracy': ensemble_acc,
            'neural_network': nn_model,
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'scaler': scaler,
            'label_encoder': label_encoder
        }

# Usage
if __name__ == "__main__":
    try:
        predictor = AdvancedOrderPriorityPredictor("C:/Users/maiwa/Downloads/Global Superstore.xlsx")
        results = predictor.run_advanced_analysis()
        
        print(f"\n🎯 Advanced Analysis Complete!")
        print(f"📊 Final Ensemble Accuracy: {results['ensemble_accuracy']:.2%}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc() 