import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import calendar
warnings.filterwarnings('ignore')

# Set style for clear and professional visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

class GlobalSuperstoreAnalyzer:
    def __init__(self, file_path):
        """Initialize the analyzer for data analytics and visualization"""
        self.df = pd.read_excel("C:/Users/maiwa/Downloads/Global Superstore.xlsx")
        self.prepare_data()
        
    def prepare_data(self):
        """Basic data preparation and feature engineering"""
        print("\n" + "="*60)
        print("DATA PREPARATION AND OVERVIEW")
        print("="*60)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumn Types:")
        print(self.df.dtypes.value_counts())
        
        # Handle missing values
        self._handle_missing_values()
        
        # Basic feature engineering for analysis
        self._create_analytical_features()
        
        # Display basic statistics
        self._display_basic_stats()
        
    def _handle_missing_values(self):
        """Handle missing values appropriately"""
        missing_values = self.df.isnull().sum()
        missing_pct = (missing_values / len(self.df)) * 100
        
        if missing_values.any():
            print(f"\nMissing Values Analysis:")
            missing_df = pd.DataFrame({
                'Missing Count': missing_values[missing_values > 0],
                'Percentage': missing_pct[missing_values > 0]
            })
            print(missing_df)
            
            # Handle missing values
            for col in self.df.columns:
                if self.df[col].dtype in ['float64', 'int64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna('Unknown', inplace=True)
                    
    def _create_analytical_features(self):
        """Create key features for analysis"""
        print("\nCreating Analytical Features...")
        
        # Financial metrics
        self.df['Profit_Margin'] = (self.df['Profit'] / self.df['Sales'].replace(0, 1)) * 100
        self.df['Revenue_Per_Unit'] = self.df['Sales'] / self.df['Quantity'].replace(0, 1)
        self.df['Shipping_Efficiency'] = self.df['Sales'] / self.df['Shipping Cost'].replace(0, 1)
        
        # Performance indicators
        self.df['Is_Profitable'] = (self.df['Profit'] > 0).astype(int)
        self.df['High_Value_Order'] = (self.df['Sales'] > self.df['Sales'].quantile(0.75)).astype(int)
        self.df['Discount_Category'] = pd.cut(self.df['Discount'], 
                                              bins=[0, 0.1, 0.2, 0.3, 1.0],
                                              labels=['No/Low', 'Medium', 'High', 'Very High'])
        
        print(f"✓ Created {6} analytical features")
        
    def _display_basic_stats(self):
        """Display basic statistics"""
        print("\nKey Business Metrics:")
        print(f"Total Revenue: ${self.df['Sales'].sum():,.2f}")
        print(f"Total Profit: ${self.df['Profit'].sum():,.2f}")
        print(f"Overall Profit Margin: {(self.df['Profit'].sum() / self.df['Sales'].sum() * 100):.2f}%")
        print(f"Number of Orders: {self.df['Order ID'].nunique():,}")
        print(f"Number of Customers: {self.df['Customer ID'].nunique():,}")
        print(f"Number of Products: {self.df['Product ID'].nunique():,}")
        
    def sales_performance_analysis(self):
        """Comprehensive sales performance analysis"""
        print("\n" + "="*60)
        print("SALES PERFORMANCE ANALYSIS")
        print("="*60)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Sales by Category
        ax1 = plt.subplot(2, 3, 1)
        category_sales = self.df.groupby('Category')['Sales'].sum().sort_values(ascending=True)
        category_sales.plot(kind='barh', ax=ax1, color='steelblue')
        ax1.set_title('Total Sales by Category', fontweight='bold')
        ax1.set_xlabel('Sales ($)')
        
        # 2. Profit Margin by Category
        ax2 = plt.subplot(2, 3, 2)
        category_margins = self.df.groupby('Category')['Profit_Margin'].mean().sort_values(ascending=True)
        colors = ['red' if x < 0 else 'green' for x in category_margins.values]
        category_margins.plot(kind='barh', ax=ax2, color=colors)
        ax2.set_title('Average Profit Margin by Category', fontweight='bold')
        ax2.set_xlabel('Profit Margin (%)')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Sales vs Profit Scatter
        ax3 = plt.subplot(2, 3, 3)
        scatter = ax3.scatter(self.df['Sales'], self.df['Profit'], 
                             c=self.df['Discount'], cmap='RdYlGn_r', 
                             alpha=0.6, s=30)
        ax3.set_xlabel('Sales ($)')
        ax3.set_ylabel('Profit ($)')
        ax3.set_title('Sales vs Profit (colored by Discount)', fontweight='bold')
        plt.colorbar(scatter, ax=ax3, label='Discount')
        
        # 4. Top 10 Products by Sales
        ax4 = plt.subplot(2, 3, 4)
        top_products = self.df.groupby('Product Name')['Sales'].sum().nlargest(10).sort_values()
        top_products.plot(kind='barh', ax=ax4, color='darkblue')
        ax4.set_title('Top 10 Products by Sales', fontweight='bold')
        ax4.set_xlabel('Sales ($)')
        
        # 5. Order Priority Distribution
        ax5 = plt.subplot(2, 3, 5)
        priority_counts = self.df['Order Priority'].value_counts()
        colors_priority = ['#ff4444', '#ff8800', '#ffcc00', '#44ff44']
        priority_counts.plot(kind='pie', ax=ax5, autopct='%1.1f%%', 
                            colors=colors_priority, startangle=90)
        ax5.set_title('Order Priority Distribution', fontweight='bold')
        ax5.set_ylabel('')
        
        # 6. Monthly Sales Trend (if we have date data)
        ax6 = plt.subplot(2, 3, 6)
        # Convert to datetime and extract month
        self.df['Order Date'] = pd.to_datetime(self.df['Order Date'])
        monthly_sales = self.df.groupby(self.df['Order Date'].dt.to_period('M'))['Sales'].sum()
        monthly_sales.plot(ax=ax6, color='navy', linewidth=2)
        ax6.set_title('Monthly Sales Trend', fontweight='bold')
        ax6.set_xlabel('Month')
        ax6.set_ylabel('Sales ($)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print insights
        print("\nKey Insights:")
        print(f"• Best performing category: {category_sales.index[-1]} (${category_sales.iloc[-1]:,.2f})")
        print(f"• Highest profit margin category: {category_margins.index[-1]} ({category_margins.iloc[-1]:.2f}%)")
        print(f"• {(self.df['Is_Profitable'].sum() / len(self.df) * 100):.1f}% of orders are profitable")
        
    def customer_segmentation_analysis(self):
        """Customer and market segmentation analysis"""
        print("\n" + "="*60)
        print("CUSTOMER SEGMENTATION ANALYSIS")
        print("="*60)
        
        fig = plt.figure(figsize=(18, 10))
        
        # 1. Sales by Customer Segment
        ax1 = plt.subplot(2, 3, 1)
        segment_sales = self.df.groupby('Segment')['Sales'].sum()
        segment_sales.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Sales by Customer Segment', fontweight='bold')
        ax1.set_ylabel('Sales ($)')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # 2. Profit by Market
        ax2 = plt.subplot(2, 3, 2)
        market_profit = self.df.groupby('Market')['Profit'].sum().sort_values()
        colors_market = ['red' if x < 0 else 'green' for x in market_profit.values]
        market_profit.plot(kind='barh', ax=ax2, color=colors_market)
        ax2.set_title('Profit by Market', fontweight='bold')
        ax2.set_xlabel('Profit ($)')
        
        # 3. Sales Distribution by Region
        ax3 = plt.subplot(2, 3, 3)
        region_sales = self.df.groupby('Region')['Sales'].sum().sort_values(ascending=False).head(10)
        region_sales.plot(kind='bar', ax=ax3, color='teal')
        ax3.set_title('Top 10 Regions by Sales', fontweight='bold')
        ax3.set_ylabel('Sales ($)')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Shipping Method Analysis
        ax4 = plt.subplot(2, 3, 4)
        ship_mode_sales = self.df.groupby('Ship Mode')['Sales'].mean().sort_values()
        ship_mode_sales.plot(kind='barh', ax=ax4, color='purple')
        ax4.set_title('Average Sales by Shipping Mode', fontweight='bold')
        ax4.set_xlabel('Average Sales ($)')
        
        # 5. Customer Concentration (Top customers)
        ax5 = plt.subplot(2, 3, 5)
        top_customers = self.df.groupby('Customer ID')['Sales'].sum().nlargest(20)
        customer_pct = (top_customers.sum() / self.df['Sales'].sum()) * 100
        other_pct = 100 - customer_pct
        ax5.pie([customer_pct, other_pct], labels=[f'Top 20 Customers\n({customer_pct:.1f}%)', 
                                                   f'Others\n({other_pct:.1f}%)'],
                colors=['#ff6b6b', '#4ecdc4'], autopct='%1.1f%%', startangle=90)
        ax5.set_title('Customer Concentration', fontweight='bold')
        
        # 6. Segment Profitability
        ax6 = plt.subplot(2, 3, 6)
        segment_profit_margin = self.df.groupby('Segment')['Profit_Margin'].mean()
        segment_profit_margin.plot(kind='bar', ax=ax6, color=['#e74c3c', '#3498db', '#2ecc71'])
        ax6.set_title('Average Profit Margin by Segment', fontweight='bold')
        ax6.set_ylabel('Profit Margin (%)')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Print insights
        print("\nKey Insights:")
        print(f"• Most valuable segment: {segment_sales.idxmax()} (${segment_sales.max():,.2f})")
        print(f"• Most profitable market: {market_profit.idxmax()} (${market_profit.max():,.2f})")
        print(f"• Top 20 customers contribute {customer_pct:.1f}% of total sales")
        
    def advanced_pca_clustering(self):
        """Advanced PCA analysis with 95% variance and clustering"""
        print("\n" + "="*60)
        print("ADVANCED PCA & CLUSTERING ANALYSIS")
        print("="*60)
        
        # Prepare features for PCA
        feature_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost',
                       'Profit_Margin', 'Revenue_Per_Unit', 'Shipping_Efficiency']
        
        X = self.df[feature_cols].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        # Find components for 95% variance
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
        
        print(f"Components needed for 95% variance: {n_components_95} out of {len(feature_cols)}")
        print(f"Variance explained by first {n_components_95} components: {cumsum_variance[n_components_95-1]:.2%}")
        
        # Perform PCA with optimal components
        pca = PCA(n_components=n_components_95)
        X_pca = pca.fit_transform(X_scaled)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        self.df['Cluster'] = clusters
        
        # Visualization
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Scree plot with 95% threshold
        ax1 = plt.subplot(2, 3, 1)
        components_to_show = min(20, len(pca_full.explained_variance_ratio_))
        ax1.bar(range(1, components_to_show + 1), 
                pca_full.explained_variance_ratio_[:components_to_show], 
                alpha=0.7, color='steelblue', label='Individual')
        ax1.plot(range(1, components_to_show + 1), 
                cumsum_variance[:components_to_show], 
                'ro-', linewidth=2, label='Cumulative')
        ax1.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% threshold')
        ax1.axvline(x=n_components_95, color='red', linestyle='--', alpha=0.7, 
                   label=f'{n_components_95} components')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained')
        ax1.set_title('PCA Scree Plot (95% Variance)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. PCA visualization (handle case when only 1 component)
        ax2 = plt.subplot(2, 3, 2)
        if n_components_95 >= 2:
            scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                                 c=clusters, cmap='viridis', 
                                 alpha=0.6, s=30)
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        else:
            # If only 1 component, create a histogram
            ax2.hist(X_pca[:, 0], bins=50, alpha=0.7, color='steelblue')
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax2.set_ylabel('Frequency')
        ax2.set_title('PCA Visualization', fontweight='bold')
        
        # 3. Feature contributions to PC1
        ax3 = plt.subplot(2, 3, 3)
        loadings = pd.Series(pca.components_[0], index=feature_cols).sort_values()
        colors_loading = ['red' if x < 0 else 'green' for x in loadings.values]
        loadings.plot(kind='barh', ax=ax3, color=colors_loading)
        ax3.set_title('Feature Contributions to PC1', fontweight='bold')
        ax3.set_xlabel('Loading Value')
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # 4. Cluster characteristics
        ax4 = plt.subplot(2, 3, 4)
        cluster_means = self.df.groupby('Cluster')[['Sales', 'Profit', 'Quantity']].mean()
        cluster_means.plot(kind='bar', ax=ax4)
        ax4.set_title('Cluster Characteristics (Averages)', fontweight='bold')
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Average Value')
        ax4.legend(['Sales', 'Profit', 'Quantity'])
        
        # 5. Cluster sizes
        ax5 = plt.subplot(2, 3, 5)
        cluster_sizes = self.df['Cluster'].value_counts().sort_index()
        colors_cluster = plt.cm.viridis(np.linspace(0, 1, len(cluster_sizes)))
        cluster_sizes.plot(kind='pie', ax=ax5, autopct='%1.1f%%', 
                          colors=colors_cluster, startangle=90)
        ax5.set_title('Cluster Distribution', fontweight='bold')
        ax5.set_ylabel('')
        
        # 6. Profit margin by cluster
        ax6 = plt.subplot(2, 3, 6)
        cluster_margins = self.df.groupby('Cluster')['Profit_Margin'].mean().sort_values()
        cluster_margins.plot(kind='bar', ax=ax6, color=plt.cm.RdYlGn(cluster_margins / cluster_margins.max()))
        ax6.set_title('Average Profit Margin by Cluster', fontweight='bold')
        ax6.set_xlabel('Cluster')
        ax6.set_ylabel('Profit Margin (%)')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Print cluster insights
        print("\nCluster Analysis:")
        for i in range(4):
            cluster_data = self.df[self.df['Cluster'] == i]
            print(f"\nCluster {i} ({len(cluster_data)} items, {len(cluster_data)/len(self.df)*100:.1f}%):")
            print(f"  • Average Sales: ${cluster_data['Sales'].mean():,.2f}")
            print(f"  • Average Profit: ${cluster_data['Profit'].mean():,.2f}")
            print(f"  • Average Profit Margin: {cluster_data['Profit_Margin'].mean():.2f}%")
            
    def discount_impact_analysis(self):
        """Analyze the impact of discounts on business metrics"""
        print("\n" + "="*60)
        print("DISCOUNT IMPACT ANALYSIS")
        print("="*60)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Discount vs Profit Margin
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(self.df['Discount'] * 100, self.df['Profit_Margin'], 
                   alpha=0.5, s=20, c=self.df['Sales'], cmap='viridis')
        ax1.set_xlabel('Discount (%)')
        ax1.set_ylabel('Profit Margin (%)')
        ax1.set_title('Discount vs Profit Margin', fontweight='bold')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # 2. Sales by Discount Category
        ax2 = plt.subplot(2, 3, 2)
        discount_sales = self.df.groupby('Discount_Category')['Sales'].mean()
        discount_sales.plot(kind='bar', ax=ax2, color=['green', 'yellow', 'orange', 'red'])
        ax2.set_title('Average Sales by Discount Level', fontweight='bold')
        ax2.set_ylabel('Average Sales ($)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # 3. Profitability by Discount Category
        ax3 = plt.subplot(2, 3, 3)
        discount_profitable = self.df.groupby('Discount_Category')['Is_Profitable'].mean() * 100
        discount_profitable.plot(kind='bar', ax=ax3, color=['darkgreen', 'lightgreen', 'orange', 'red'])
        ax3.set_title('Profitability Rate by Discount Level', fontweight='bold')
        ax3.set_ylabel('Profitable Orders (%)')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5)
        
        # 4. Quantity vs Discount relationship
        ax4 = plt.subplot(2, 3, 4)
        discount_quantity = self.df.groupby(pd.cut(self.df['Discount'], bins=10))['Quantity'].mean()
        discount_quantity.plot(kind='line', ax=ax4, marker='o', linewidth=2, markersize=8, color='navy')
        ax4.set_title('Average Quantity vs Discount Range', fontweight='bold')
        ax4.set_xlabel('Discount Range')
        ax4.set_ylabel('Average Quantity')
        ax4.grid(True, alpha=0.3)
        
        # 5. Revenue impact of discounts
        ax5 = plt.subplot(2, 3, 5)
        no_discount_revenue = self.df[self.df['Discount'] == 0]['Sales'].sum()
        discounted_revenue = self.df[self.df['Discount'] > 0]['Sales'].sum()
        revenues = [no_discount_revenue, discounted_revenue]
        labels = ['No Discount', 'With Discount']
        colors = ['#2ecc71', '#e74c3c']
        ax5.pie(revenues, labels=labels, colors=colors, autopct=lambda p: f'${revenues[int(p/100*len(revenues))]/(10**6):.1f}M\n({p:.1f}%)', 
                startangle=90)
        ax5.set_title('Revenue Split: Discount vs No Discount', fontweight='bold')
        
        # 6. Discount effectiveness by category
        ax6 = plt.subplot(2, 3, 6)
        category_discount_impact = self.df[self.df['Discount'] > 0].groupby('Category')['Profit_Margin'].mean()
        category_discount_impact.plot(kind='bar', ax=ax6, color=['#3498db', '#e74c3c', '#f39c12'])
        ax6.set_title('Avg Profit Margin on Discounted Items by Category', fontweight='bold')
        ax6.set_ylabel('Profit Margin (%)')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Print insights
        print("\nDiscount Impact Insights:")
        print(f"• Orders with no discount: {(self.df['Discount'] == 0).sum()} ({(self.df['Discount'] == 0).sum()/len(self.df)*100:.1f}%)")
        print(f"• Average profit margin without discount: {self.df[self.df['Discount'] == 0]['Profit_Margin'].mean():.2f}%")
        print(f"• Average profit margin with discount: {self.df[self.df['Discount'] > 0]['Profit_Margin'].mean():.2f}%")
        print(f"• Revenue from discounted orders: ${discounted_revenue:,.2f} ({discounted_revenue/(no_discount_revenue+discounted_revenue)*100:.1f}%)")
        
    def comprehensive_eda(self):
        """Comprehensive Exploratory Data Analysis with deep insights"""
        print("\n" + "="*60)
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Statistical summary
        print("\nDetailed Statistical Summary:")
        print("-" * 40)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in ['Sales', 'Profit', 'Quantity', 'Discount', 'Shipping Cost']:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(f"  Mean: {self.df[col].mean():.2f}")
                print(f"  Median: {self.df[col].median():.2f}")
                print(f"  Std Dev: {self.df[col].std():.2f}")
                print(f"  Skewness: {self.df[col].skew():.2f}")
                print(f"  Kurtosis: {self.df[col].kurtosis():.2f}")
        
        fig = plt.figure(figsize=(24, 18))
        
        # 1. Box and Whiskers Plot for Profit by Segment (as requested)
        ax1 = plt.subplot(4, 4, 1)
        segment_order = self.df.groupby('Segment')['Profit'].median().sort_values().index
        self.df.boxplot(column='Profit', by='Segment', ax=ax1, patch_artist=True)
        ax1.set_title('Profit Distribution by Segment (Box & Whiskers)', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Segment')
        ax1.set_ylabel('Profit ($)')
        ax1.set_xticklabels(segment_order)
        plt.sca(ax1)
        plt.xticks(rotation=0)
        ax1.get_figure().suptitle('')  # Remove automatic title
        
        # 2. Correlation Heatmap
        ax2 = plt.subplot(4, 4, 2)
        corr_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost', 
                     'Profit_Margin', 'Revenue_Per_Unit']
        correlation_matrix = self.df[corr_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', ax=ax2, square=True, linewidths=1)
        ax2.set_title('Feature Correlation Matrix', fontweight='bold')
        
        # 3. Distribution of Key Metrics
        ax3 = plt.subplot(4, 4, 3)
        self.df['Sales'].hist(bins=50, ax=ax3, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(self.df['Sales'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: ${self.df["Sales"].median():.0f}')
        ax3.axvline(self.df['Sales'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: ${self.df["Sales"].mean():.0f}')
        ax3.set_title('Sales Distribution', fontweight='bold')
        ax3.set_xlabel('Sales ($)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.set_yscale('log')
        
        # 4. Profit Distribution with Outliers
        ax4 = plt.subplot(4, 4, 4)
        profit_q1 = self.df['Profit'].quantile(0.25)
        profit_q3 = self.df['Profit'].quantile(0.75)
        profit_iqr = profit_q3 - profit_q1
        outliers = self.df[(self.df['Profit'] < profit_q1 - 1.5*profit_iqr) | 
                          (self.df['Profit'] > profit_q3 + 1.5*profit_iqr)]
        
        self.df['Profit'].hist(bins=50, ax=ax4, color='lightgreen', alpha=0.7, edgecolor='black')
        ax4.axvline(0, color='red', linestyle='-', linewidth=2, alpha=0.7)
        ax4.set_title(f'Profit Distribution ({outliers.shape[0]} outliers)', fontweight='bold')
        ax4.set_xlabel('Profit ($)')
        ax4.set_ylabel('Frequency')
        
        # 5. Time Series Decomposition
        ax5 = plt.subplot(4, 4, 5)
        daily_sales = self.df.groupby(self.df['Order Date'].dt.date)['Sales'].sum()
        daily_sales.rolling(window=30).mean().plot(ax=ax5, label='30-day MA', linewidth=2)
        daily_sales.rolling(window=7).mean().plot(ax=ax5, label='7-day MA', alpha=0.7)
        ax5.set_title('Sales Moving Averages', fontweight='bold')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Sales ($)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Seasonal Pattern Analysis
        ax6 = plt.subplot(4, 4, 6)
        self.df['Month'] = self.df['Order Date'].dt.month
        self.df['DayOfWeek'] = self.df['Order Date'].dt.dayofweek
        monthly_pattern = self.df.groupby('Month')['Sales'].mean()
        monthly_pattern.plot(kind='bar', ax=ax6, color=plt.cm.viridis(np.linspace(0, 1, 12)))
        ax6.set_title('Average Sales by Month', fontweight='bold')
        ax6.set_xlabel('Month')
        ax6.set_ylabel('Average Sales ($)')
        
        # 7. Product Category Performance Heatmap
        ax7 = plt.subplot(4, 4, 7)
        cat_subcat = pd.crosstab(self.df['Category'], self.df['Sub-Category'], 
                                values=self.df['Profit'], aggfunc='mean')
        sns.heatmap(cat_subcat, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=ax7)
        ax7.set_title('Avg Profit: Category vs Sub-Category', fontweight='bold')
        
        # 8. Geographic Performance (Top Countries)
        ax8 = plt.subplot(4, 4, 8)
        top_countries = self.df.groupby('Country')['Sales'].sum().nlargest(15)
        top_countries.plot(kind='barh', ax=ax8, color='navy')
        ax8.set_title('Top 15 Countries by Sales', fontweight='bold')
        ax8.set_xlabel('Total Sales ($)')
        
        # 9. Order Size Distribution
        ax9 = plt.subplot(4, 4, 9)
        order_sizes = self.df.groupby('Order ID')['Sales'].sum()
        order_size_bins = pd.cut(order_sizes, bins=[0, 100, 500, 1000, 5000, float('inf')],
                                labels=['<$100', '$100-500', '$500-1K', '$1K-5K', '>$5K'])
        order_size_bins.value_counts().sort_index().plot(kind='bar', ax=ax9, color='darkgreen')
        ax9.set_title('Order Size Distribution', fontweight='bold')
        ax9.set_xlabel('Order Size Range')
        ax9.set_ylabel('Number of Orders')
        ax9.set_xticklabels(ax9.get_xticklabels(), rotation=45)
        
        # 10. Profit Margin Distribution by Market
        ax10 = plt.subplot(4, 4, 10)
        market_margins = self.df.boxplot(column='Profit_Margin', by='Market', ax=ax10, 
                                        patch_artist=True, return_type='dict')
        ax10.set_title('Profit Margin Distribution by Market', fontweight='bold')
        ax10.set_xlabel('Market')
        ax10.set_ylabel('Profit Margin (%)')
        plt.sca(ax10)
        plt.xticks(rotation=45)
        ax10.get_figure().suptitle('')
        
        # 11. Customer Purchase Frequency
        ax11 = plt.subplot(4, 4, 11)
        customer_frequency = self.df.groupby('Customer ID').size()
        freq_dist = customer_frequency.value_counts().head(10)
        freq_dist.plot(kind='bar', ax=ax11, color='orange')
        ax11.set_title('Customer Purchase Frequency Distribution', fontweight='bold')
        ax11.set_xlabel('Number of Orders per Customer')
        ax11.set_ylabel('Number of Customers')
        
        # 12. Shipping Cost Efficiency
        ax12 = plt.subplot(4, 4, 12)
        ax12.scatter(self.df['Shipping Cost'], self.df['Sales'], 
                    c=self.df['Ship Mode'].astype('category').cat.codes, 
                    alpha=0.5, s=20, cmap='Set1')
        ax12.set_xlabel('Shipping Cost ($)')
        ax12.set_ylabel('Sales ($)')
        ax12.set_title('Shipping Cost vs Sales by Ship Mode', fontweight='bold')
        
        # 13. Discount Effectiveness Curve
        ax13 = plt.subplot(4, 4, 13)
        discount_bins = pd.cut(self.df['Discount'], bins=20)
        discount_profit = self.df.groupby(discount_bins)['Profit'].mean()
        discount_sales = self.df.groupby(discount_bins)['Sales'].mean()
        
        ax13_twin = ax13.twinx()
        discount_profit.plot(ax=ax13, color='red', linewidth=2, marker='o', label='Avg Profit')
        discount_sales.plot(ax=ax13_twin, color='blue', linewidth=2, marker='s', label='Avg Sales')
        ax13.set_xlabel('Discount Range')
        ax13.set_ylabel('Average Profit ($)', color='red')
        ax13_twin.set_ylabel('Average Sales ($)', color='blue')
        ax13.set_title('Discount Impact on Sales & Profit', fontweight='bold')
        ax13.tick_params(axis='y', labelcolor='red')
        ax13_twin.tick_params(axis='y', labelcolor='blue')
        
        # 14. Product Lifecycle Analysis
        ax14 = plt.subplot(4, 4, 14)
        product_lifecycle = self.df.groupby([self.df['Order Date'].dt.to_period('Q'), 'Category'])['Sales'].sum().unstack()
        product_lifecycle.plot(ax=ax14, linewidth=2, marker='o')
        ax14.set_title('Category Sales Trend by Quarter', fontweight='bold')
        ax14.set_xlabel('Quarter')
        ax14.set_ylabel('Sales ($)')
        ax14.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 15. Profitability Segments
        ax15 = plt.subplot(4, 4, 15)
        self.df['Profitability_Segment'] = pd.cut(self.df['Profit_Margin'], 
                                                  bins=[-float('inf'), -10, 0, 10, 20, float('inf')],
                                                  labels=['Heavy Loss', 'Loss', 'Low Profit', 'Medium Profit', 'High Profit'])
        profit_segments = self.df['Profitability_Segment'].value_counts()
        colors_profit = ['darkred', 'red', 'yellow', 'lightgreen', 'darkgreen']
        profit_segments.plot(kind='pie', ax=ax15, colors=colors_profit, autopct='%1.1f%%', startangle=90)
        ax15.set_title('Orders by Profitability Segment', fontweight='bold')
        ax15.set_ylabel('')
        
        # 16. Return on Investment by Category
        ax16 = plt.subplot(4, 4, 16)
        roi_by_category = self.df.groupby('Category').apply(
            lambda x: (x['Profit'].sum() / x['Sales'].sum() * 100) if x['Sales'].sum() > 0 else 0
        )
        roi_by_category.plot(kind='bar', ax=ax16, color=['red' if x < 0 else 'green' for x in roi_by_category])
        ax16.set_title('ROI by Category', fontweight='bold')
        ax16.set_ylabel('ROI (%)')
        ax16.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Advanced Statistical Insights
        print("\n" + "="*60)
        print("ADVANCED STATISTICAL INSIGHTS")
        print("="*60)
        
        # Outlier Analysis
        print(f"\nOutlier Analysis:")
        print(f"• Profit outliers: {outliers.shape[0]} ({outliers.shape[0]/len(self.df)*100:.2f}%)")
        print(f"• Extreme losses (< -$1000): {(self.df['Profit'] < -1000).sum()}")
        print(f"• Extreme gains (> $5000): {(self.df['Profit'] > 5000).sum()}")
        
        # Customer Insights
        print(f"\nCustomer Behavior:")
        print(f"• Repeat customers: {(customer_frequency > 1).sum()} ({(customer_frequency > 1).sum()/customer_frequency.shape[0]*100:.1f}%)")
        print(f"• Average orders per customer: {customer_frequency.mean():.2f}")
        print(f"• Top customer orders: {customer_frequency.max()} orders")
        
        # Seasonal Insights
        print(f"\nSeasonal Patterns:")
        best_month = monthly_pattern.idxmax()
        worst_month = monthly_pattern.idxmin()
        print(f"• Best month: {calendar.month_name[best_month]} (${monthly_pattern.max():,.2f} avg)")
        print(f"• Worst month: {calendar.month_name[worst_month]} (${monthly_pattern.min():,.2f} avg)")
        
        # Efficiency Metrics
        print(f"\nOperational Efficiency:")
        print(f"• Average shipping cost ratio: {(self.df['Shipping Cost']/self.df['Sales']*100).mean():.2f}%")
        print(f"• Orders with shipping > 10% of sales: {((self.df['Shipping Cost']/self.df['Sales']) > 0.1).sum()}")
        
    def statistical_hypothesis_testing(self):
        """Perform statistical hypothesis testing and advanced analytics"""
        print("\n" + "="*60)
        print("STATISTICAL HYPOTHESIS TESTING")
        print("="*60)
        
        from scipy import stats
        
        # Test 1: Is there a significant difference in profit margins between segments?
        segments = ['Consumer', 'Corporate', 'Home Office']
        segment_margins = [self.df[self.df['Segment'] == seg]['Profit_Margin'].dropna() for seg in segments]
        
        # Remove infinite values
        segment_margins = [margins[~np.isinf(margins)] for margins in segment_margins]
        
        # ANOVA test
        f_stat, p_value = stats.f_oneway(*segment_margins)
        print(f"\n1. ANOVA Test - Profit Margins across Segments:")
        print(f"   F-statistic: {f_stat:.4f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Result: {'Significant difference' if p_value < 0.05 else 'No significant difference'} between segments")
        
        # Test 2: Correlation between discount and quantity
        discount_clean = self.df['Discount'][self.df['Discount'] > 0]
        quantity_clean = self.df.loc[discount_clean.index, 'Quantity']
        corr, p_val_corr = stats.pearsonr(discount_clean, quantity_clean)
        print(f"\n2. Pearson Correlation - Discount vs Quantity:")
        print(f"   Correlation coefficient: {corr:.4f}")
        print(f"   P-value: {p_val_corr:.4f}")
        print(f"   Result: {'Significant' if p_val_corr < 0.05 else 'Not significant'} correlation")
        
        # Test 3: Chi-square test for profitability and shipping mode
        contingency_table = pd.crosstab(self.df['Ship Mode'], self.df['Is_Profitable'])
        chi2, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\n3. Chi-Square Test - Ship Mode vs Profitability:")
        print(f"   Chi-square statistic: {chi2:.4f}")
        print(f"   P-value: {p_val_chi2:.4f}")
        print(f"   Result: {'Significant' if p_val_chi2 < 0.05 else 'Not significant'} association")
        
        # Visualization of statistical tests
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Segment profit margin distributions
        ax1 = axes[0, 0]
        for i, (seg, margins) in enumerate(zip(segments, segment_margins)):
            ax1.hist(margins, alpha=0.5, label=seg, bins=30)
        ax1.set_title('Profit Margin Distributions by Segment', fontweight='bold')
        ax1.set_xlabel('Profit Margin (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Plot 2: Discount vs Quantity scatter with regression
        ax2 = axes[0, 1]
        ax2.scatter(self.df['Discount'], self.df['Quantity'], alpha=0.3, s=20)
        z = np.polyfit(self.df['Discount'], self.df['Quantity'], 1)
        p = np.poly1d(z)
        ax2.plot(self.df['Discount'].sort_values(), p(self.df['Discount'].sort_values()), 
                "r--", linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
        ax2.set_title(f'Discount vs Quantity (r={corr:.3f})', fontweight='bold')
        ax2.set_xlabel('Discount')
        ax2.set_ylabel('Quantity')
        ax2.legend()
        
        # Plot 3: Profitability by Ship Mode
        ax3 = axes[1, 0]
        ship_profit_pct = self.df.groupby('Ship Mode')['Is_Profitable'].mean() * 100
        ship_profit_pct.plot(kind='bar', ax=ax3, color=['red' if x < 50 else 'green' for x in ship_profit_pct])
        ax3.set_title('Profitability Rate by Shipping Mode', fontweight='bold')
        ax3.set_ylabel('Profitable Orders (%)')
        ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        
        # Plot 4: Market performance comparison
        ax4 = axes[1, 1]
        market_stats = self.df.groupby('Market').agg({
            'Sales': 'mean',
            'Profit': 'mean',
            'Profit_Margin': 'mean'
        })
        x = np.arange(len(market_stats))
        width = 0.25
        ax4.bar(x - width, market_stats['Sales'], width, label='Avg Sales', color='blue')
        ax4.bar(x, market_stats['Profit'], width, label='Avg Profit', color='green')
        ax4.bar(x + width, market_stats['Profit_Margin'], width, label='Avg Margin %', color='orange')
        ax4.set_xlabel('Market')
        ax4.set_ylabel('Value')
        ax4.set_title('Market Performance Comparison', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(market_stats.index, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
    def run_complete_analysis(self):
        """Run all analysis components including new deep analytics"""
        print("\n" + "="*70)
        print("GLOBAL SUPERSTORE COMPREHENSIVE DATA ANALYTICS - ENHANCED")
        print("="*70)
        
        # Run all analyses
        self.sales_performance_analysis()
        self.customer_segmentation_analysis()
        self.comprehensive_eda()  # New comprehensive EDA
        self.statistical_hypothesis_testing()  # New statistical testing
        self.advanced_pca_clustering()
        self.discount_impact_analysis()
        
        print("\n" + "="*70)
        print("ENHANCED ANALYSIS COMPLETE")
        print("="*70)
        print("\nAnalysis Components Completed:")
        print("✓ Sales Performance Analysis")
        print("✓ Customer Segmentation Analysis")
        print("✓ Comprehensive Exploratory Data Analysis")
        print("✓ Box & Whiskers Plot for Profit by Segment")
        print("✓ Statistical Hypothesis Testing")
        print("✓ PCA with 95% Variance & Clustering")
        print("✓ Discount Impact Analysis")
        print("\nAll visualizations have been displayed.")
        
        return self.df

# Usage
if __name__ == "__main__":
    try:
        analyzer = GlobalSuperstoreAnalyzer("Global Superstore.xlsx")
        results = analyzer.run_complete_analysis()
        
        print(f"\n🎯 Analysis complete!")
        print(f"📊 Analyzed {len(analyzer.df):,} records")
        
    except FileNotFoundError:
        print("Error: Could not find the Global Superstore file")
        print("Please ensure the file path is correct")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()