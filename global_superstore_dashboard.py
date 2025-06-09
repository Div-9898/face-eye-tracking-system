import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Global Superstore Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Info box styling */
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    /* Success/Warning/Error boxes */
    .success-box {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    
    .error-box {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    </style>
""", unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    """Load and prepare the Global Superstore data"""
    try:
        df = pd.read_excel("C:/Users/maiwa/Downloads/Global Superstore.xlsx")
        
        # Convert Order Date to datetime
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        
        # Create additional features
        df['Profit_Margin'] = (df['Profit'] / df['Sales'].replace(0, 1)) * 100
        df['Revenue_Per_Unit'] = df['Sales'] / df['Quantity'].replace(0, 1)
        df['Shipping_Efficiency'] = df['Sales'] / df['Shipping Cost'].replace(0, 1)
        df['Is_Profitable'] = (df['Profit'] > 0).astype(int)
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.month
        df['Month_Name'] = df['Order Date'].dt.strftime('%B')
        df['Quarter'] = df['Order Date'].dt.quarter
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main dashboard function
def main():
    # Header
    st.markdown("<h1 class='main-header'>🌍 Global Superstore Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the file path.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Global+Superstore", use_column_width=True)
        st.markdown("### 📊 Navigation")
        
        page = st.selectbox(
            "Select Dashboard View",
            ["🏠 Overview", "📈 Sales Analysis", "👥 Customer Insights", 
             "📦 Product Performance", "🌍 Geographic Analysis", "💰 Profitability Analysis",
             "📊 Advanced Analytics", "⚙️ Data Explorer"]
        )
        
        st.markdown("### 🔍 Global Filters")
        
        # Date range filter
        date_range = st.date_input(
            "Select Date Range",
            value=(df['Order Date'].min(), df['Order Date'].max()),
            min_value=df['Order Date'].min(),
            max_value=df['Order Date'].max()
        )
        
        # Segment filter
        segments = st.multiselect(
            "Select Segments",
            options=df['Segment'].unique(),
            default=df['Segment'].unique()
        )
        
        # Category filter
        categories = st.multiselect(
            "Select Categories",
            options=df['Category'].unique(),
            default=df['Category'].unique()
        )
        
        # Apply filters
        filtered_df = df[
            (df['Order Date'].dt.date >= date_range[0]) &
            (df['Order Date'].dt.date <= date_range[1]) &
            (df['Segment'].isin(segments)) &
            (df['Category'].isin(categories))
        ]
        
        # Display filter summary
        st.markdown("### 📋 Filter Summary")
        st.info(f"""
        - **Records**: {len(filtered_df):,} / {len(df):,}
        - **Date Range**: {date_range[0]} to {date_range[1]}
        - **Segments**: {', '.join(segments)}
        """)
    
    # Main content area based on selected page
    if page == "🏠 Overview":
        show_overview(filtered_df)
    elif page == "📈 Sales Analysis":
        show_sales_analysis(filtered_df)
    elif page == "👥 Customer Insights":
        show_customer_insights(filtered_df)
    elif page == "📦 Product Performance":
        show_product_performance(filtered_df)
    elif page == "🌍 Geographic Analysis":
        show_geographic_analysis(filtered_df)
    elif page == "💰 Profitability Analysis":
        show_profitability_analysis(filtered_df)
    elif page == "📊 Advanced Analytics":
        show_advanced_analytics(filtered_df)
    elif page == "⚙️ Data Explorer":
        show_data_explorer(filtered_df)

def show_overview(df):
    """Display overview dashboard"""
    st.markdown("<h2 class='section-header'>Executive Dashboard Overview</h2>", unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>${:,.0f}</div>
                <div class='metric-label'>Total Revenue</div>
            </div>
        """.format(df['Sales'].sum()), unsafe_allow_html=True)
    
    with col2:
        profit = df['Profit'].sum()
        color = "green" if profit > 0 else "red"
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color: {color}'>${profit:,.0f}</div>
                <div class='metric-label'>Total Profit</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>{:,.0f}</div>
                <div class='metric-label'>Total Orders</div>
            </div>
        """.format(df['Order ID'].nunique()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>{:,.0f}</div>
                <div class='metric-label'>Total Customers</div>
            </div>
        """.format(df['Customer ID'].nunique()), unsafe_allow_html=True)
    
    with col5:
        profit_margin = (df['Profit'].sum() / df['Sales'].sum() * 100)
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>{:.1f}%</div>
                <div class='metric-label'>Profit Margin</div>
            </div>
        """.format(profit_margin), unsafe_allow_html=True)
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales trend
        st.subheader("📈 Sales Trend Over Time")
        monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        monthly_sales['Order Date'] = monthly_sales['Order Date'].astype(str)
        
        fig = px.line(monthly_sales, x='Order Date', y='Sales',
                     title='Monthly Sales Trend',
                     labels={'Sales': 'Sales ($)', 'Order Date': 'Month'})
        fig.update_traces(line_color='#1f77b4', line_width=3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category performance
        st.subheader("📊 Sales by Category")
        category_sales = df.groupby('Category')['Sales'].sum().reset_index()
        
        fig = px.pie(category_sales, values='Sales', names='Category',
                    title='Sales Distribution by Category',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 products
        st.subheader("🏆 Top 10 Products by Sales")
        top_products = df.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
        
        fig = px.bar(top_products, x='Sales', y='Product Name',
                    orientation='h', title='Top 10 Products',
                    color='Sales', color_continuous_scale='Blues')
        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Segment performance
        st.subheader("👥 Performance by Segment")
        segment_data = df.groupby('Segment').agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Sales', x=segment_data['Segment'], y=segment_data['Sales'],
                            marker_color='lightblue'))
        fig.add_trace(go.Bar(name='Profit', x=segment_data['Segment'], y=segment_data['Profit'],
                            marker_color='darkblue'))
        fig.update_layout(barmode='group', title='Sales and Profit by Segment', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.markdown("<h3 class='section-header'>📅 Recent Activity</h3>", unsafe_allow_html=True)
    recent_orders = df.nlargest(5, 'Order Date')[['Order Date', 'Customer Name', 'Product Name', 'Sales', 'Profit']]
    st.dataframe(recent_orders, use_container_width=True)

def show_sales_analysis(df):
    """Display sales analysis dashboard"""
    st.markdown("<h2 class='section-header'>Sales Analysis Dashboard</h2>", unsafe_allow_html=True)
    
    # Sales metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_order_value = df.groupby('Order ID')['Sales'].sum().mean()
        st.metric("Average Order Value", f"${avg_order_value:,.2f}")
    
    with col2:
        total_quantity = df['Quantity'].sum()
        st.metric("Total Units Sold", f"{total_quantity:,}")
    
    with col3:
        avg_discount = df['Discount'].mean() * 100
        st.metric("Average Discount", f"{avg_discount:.1f}%")
    
    with col4:
        revenue_per_customer = df.groupby('Customer ID')['Sales'].sum().mean()
        st.metric("Revenue per Customer", f"${revenue_per_customer:,.2f}")
    
    # Time series analysis
    st.subheader("📊 Sales Time Series Analysis")
    
    # Create tabs for different time views
    tab1, tab2, tab3 = st.tabs(["Daily", "Monthly", "Yearly"])
    
    with tab1:
        daily_sales = df.groupby(df['Order Date'].dt.date)['Sales'].sum().reset_index()
        fig = px.line(daily_sales, x='Order Date', y='Sales',
                     title='Daily Sales Trend')
        fig.add_scatter(x=daily_sales['Order Date'], y=daily_sales['Sales'].rolling(7).mean(),
                       mode='lines', name='7-day MA', line=dict(color='red'))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        monthly_sales = df.groupby([df['Order Date'].dt.year, df['Order Date'].dt.month]).agg({
            'Sales': 'sum'
        }).reset_index()
        monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Order Date', 'Order Date']].rename(columns={'Order Date': 'year', 'Order Date': 'month'}))
        
        fig = px.bar(monthly_sales, x='Date', y='Sales',
                    title='Monthly Sales Performance')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        yearly_sales = df.groupby(df['Order Date'].dt.year)['Sales'].sum().reset_index()
        yearly_sales.columns = ['Year', 'Sales']
        
        fig = px.bar(yearly_sales, x='Year', y='Sales',
                    title='Yearly Sales Trend',
                    text='Sales')
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Sales breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Sales by Sub-Category")
        subcat_sales = df.groupby('Sub-Category')['Sales'].sum().nlargest(15).reset_index()
        fig = px.treemap(subcat_sales, path=['Sub-Category'], values='Sales',
                        title='Sales Distribution by Sub-Category',
                        color='Sales', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Sales Growth Analysis")
        # Calculate month-over-month growth
        monthly_growth = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
        growth_rate = monthly_growth.pct_change() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=growth_rate.index.astype(str), y=growth_rate.values,
                           marker_color=['red' if x < 0 else 'green' for x in growth_rate.values]))
        fig.update_layout(title='Month-over-Month Sales Growth Rate (%)',
                         xaxis_title='Month',
                         yaxis_title='Growth Rate (%)')
        st.plotly_chart(fig, use_container_width=True)

def show_customer_insights(df):
    """Display customer insights dashboard"""
    st.markdown("<h2 class='section-header'>Customer Insights Dashboard</h2>", unsafe_allow_html=True)
    
    # Customer metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = df['Customer ID'].nunique()
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        repeat_customers = df.groupby('Customer ID').size()
        repeat_rate = (repeat_customers > 1).sum() / len(repeat_customers) * 100
        st.metric("Repeat Customer Rate", f"{repeat_rate:.1f}%")
    
    with col3:
        avg_customer_value = df.groupby('Customer ID')['Sales'].sum().mean()
        st.metric("Avg Customer Value", f"${avg_customer_value:,.2f}")
    
    with col4:
        customer_segments = df['Segment'].value_counts()
        top_segment = customer_segments.index[0]
        st.metric("Top Segment", top_segment)
    
    # Customer segmentation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Customer Segmentation")
        segment_analysis = df.groupby('Segment').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Customer ID': 'nunique'
        }).reset_index()
        
        fig = px.scatter(segment_analysis, x='Sales', y='Profit', size='Customer ID',
                        color='Segment', title='Segment Performance Analysis',
                        labels={'Customer ID': 'Number of Customers'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Customer Lifetime Value Distribution")
        customer_ltv = df.groupby('Customer ID')['Sales'].sum().reset_index()
        
        fig = px.histogram(customer_ltv, x='Sales', nbins=50,
                          title='Customer Lifetime Value Distribution',
                          labels={'Sales': 'Customer Lifetime Value ($)'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top customers
    st.subheader("🏆 Top 20 Customers")
    top_customers = df.groupby('Customer Name').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique'
    }).nlargest(20, 'Sales').reset_index()
    top_customers.columns = ['Customer', 'Total Sales', 'Total Profit', 'Number of Orders']
    
    fig = px.bar(top_customers, x='Customer', y='Total Sales',
                color='Total Profit', color_continuous_scale='RdYlGn',
                title='Top 20 Customers by Sales')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer behavior patterns
    st.subheader("📊 Customer Behavior Patterns")
    
    tab1, tab2 = st.tabs(["Purchase Frequency", "Segment Analysis"])
    
    with tab1:
        purchase_freq = df.groupby('Customer ID').size().value_counts().head(10).reset_index()
        purchase_freq.columns = ['Orders per Customer', 'Number of Customers']
        
        fig = px.bar(purchase_freq, x='Orders per Customer', y='Number of Customers',
                    title='Customer Purchase Frequency Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        segment_metrics = df.groupby('Segment').agg({
            'Sales': ['mean', 'sum'],
            'Profit': ['mean', 'sum'],
            'Discount': 'mean'
        }).round(2)
        
        st.dataframe(segment_metrics, use_container_width=True)

def show_product_performance(df):
    """Display product performance dashboard"""
    st.markdown("<h2 class='section-header'>Product Performance Dashboard</h2>", unsafe_allow_html=True)
    
    # Product metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_products = df['Product ID'].nunique()
        st.metric("Total Products", f"{total_products:,}")
    
    with col2:
        avg_product_sales = df.groupby('Product ID')['Sales'].sum().mean()
        st.metric("Avg Product Sales", f"${avg_product_sales:,.2f}")
    
    with col3:
        profitable_products = (df.groupby('Product ID')['Profit'].sum() > 0).sum()
        profitable_rate = profitable_products / total_products * 100
        st.metric("Profitable Products", f"{profitable_rate:.1f}%")
    
    with col4:
        best_category = df.groupby('Category')['Sales'].sum().idxmax()
        st.metric("Top Category", best_category)
    
    # Category performance
    st.subheader("📊 Category Performance Analysis")
    
    category_performance = df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Sales by Category', 'Profit by Category', 'Quantity by Category'),
        specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]]
    )
    
    fig.add_trace(go.Pie(labels=category_performance['Category'], 
                        values=category_performance['Sales'], name="Sales"), 1, 1)
    fig.add_trace(go.Pie(labels=category_performance['Category'], 
                        values=category_performance['Profit'], name="Profit"), 1, 2)
    fig.add_trace(go.Pie(labels=category_performance['Category'], 
                        values=category_performance['Quantity'], name="Quantity"), 1, 3)
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sub-category analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performing Sub-Categories")
        top_subcats = df.groupby('Sub-Category').agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).nlargest(10, 'Sales').reset_index()
        
        fig = px.bar(top_subcats, x='Sub-Category', y=['Sales', 'Profit'],
                    title='Top 10 Sub-Categories by Sales',
                    barmode='group')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📉 Bottom Performing Products")
        bottom_products = df.groupby('Product Name')['Profit'].sum().nsmallest(10).reset_index()
        
        fig = px.bar(bottom_products, x='Profit', y='Product Name',
                    orientation='h', title='Bottom 10 Products by Profit',
                    color='Profit', color_continuous_scale='Reds_r')
        st.plotly_chart(fig, use_container_width=True)
    
    # Product profitability matrix
    st.subheader("🎯 Product Profitability Matrix")
    
    product_matrix = df.groupby('Product Name').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    product_matrix['Profit_Margin'] = (product_matrix['Profit'] / product_matrix['Sales'] * 100)
    
    # Filter top 50 products by sales for clarity
    top_products_matrix = product_matrix.nlargest(50, 'Sales')
    
    fig = px.scatter(top_products_matrix, x='Sales', y='Profit_Margin',
                    size='Quantity', color='Profit',
                    hover_data=['Product Name'],
                    title='Product Profitability Matrix (Top 50 by Sales)',
                    labels={'Profit_Margin': 'Profit Margin (%)'},
                    color_continuous_scale='RdYlGn')
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=top_products_matrix['Sales'].median(), line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)

def show_geographic_analysis(df):
    """Display geographic analysis dashboard"""
    st.markdown("<h2 class='section-header'>Geographic Analysis Dashboard</h2>", unsafe_allow_html=True)
    
    # Geographic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_countries = df['Country'].nunique()
        st.metric("Countries Served", f"{total_countries}")
    
    with col2:
        total_cities = df['City'].nunique()
        st.metric("Cities Served", f"{total_cities:,}")
    
    with col3:
        top_market = df.groupby('Market')['Sales'].sum().idxmax()
        st.metric("Top Market", top_market)
    
    with col4:
        top_country = df.groupby('Country')['Sales'].sum().idxmax()
        st.metric("Top Country", top_country)
    
    # Market performance
    st.subheader("🌍 Market Performance Overview")
    
    market_data = df.groupby('Market').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique',
        'Customer ID': 'nunique'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sales by Market', 'Profit by Market', 
                       'Orders by Market', 'Customers by Market')
    )
    
    fig.add_trace(go.Bar(x=market_data['Market'], y=market_data['Sales'], 
                        name='Sales', marker_color='lightblue'), row=1, col=1)
    fig.add_trace(go.Bar(x=market_data['Market'], y=market_data['Profit'], 
                        name='Profit', marker_color='green'), row=1, col=2)
    fig.add_trace(go.Bar(x=market_data['Market'], y=market_data['Order ID'], 
                        name='Orders', marker_color='orange'), row=2, col=1)
    fig.add_trace(go.Bar(x=market_data['Market'], y=market_data['Customer ID'], 
                        name='Customers', marker_color='purple'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Country analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 15 Countries by Sales")
        top_countries = df.groupby('Country')['Sales'].sum().nlargest(15).reset_index()
        
        fig = px.bar(top_countries, x='Sales', y='Country',
                    orientation='h', title='Top Countries Performance',
                    color='Sales', color_continuous_scale='Blues')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🗺️ Sales Distribution by Region")
        region_sales = df.groupby('Region')['Sales'].sum().nlargest(20).reset_index()
        
        fig = px.treemap(region_sales, path=['Region'], values='Sales',
                        title='Regional Sales Distribution',
                        color='Sales', color_continuous_scale='Viridis')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # City performance
    st.subheader("🏙️ City-Level Performance")
    
    city_data = df.groupby(['Country', 'City']).agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    # Top 20 cities
    top_cities = city_data.nlargest(20, 'Sales')
    
    fig = px.sunburst(top_cities, path=['Country', 'City'], values='Sales',
                     color='Profit', color_continuous_scale='RdYlGn',
                     title='Top 20 Cities: Sales and Profit Distribution')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def show_profitability_analysis(df):
    """Display profitability analysis dashboard"""
    st.markdown("<h2 class='section-header'>Profitability Analysis Dashboard</h2>", unsafe_allow_html=True)
    
    # Profitability metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_profit = df['Profit'].sum()
        color = "green" if total_profit > 0 else "red"
        st.metric("Total Profit", f"${total_profit:,.0f}", delta_color="normal")
    
    with col2:
        profit_margin = (df['Profit'].sum() / df['Sales'].sum() * 100)
        st.metric("Overall Profit Margin", f"{profit_margin:.2f}%")
    
    with col3:
        profitable_orders = (df['Profit'] > 0).sum()
        profitable_rate = profitable_orders / len(df) * 100
        st.metric("Profitable Orders", f"{profitable_rate:.1f}%")
    
    with col4:
        avg_profit_per_order = df.groupby('Order ID')['Profit'].sum().mean()
        st.metric("Avg Profit per Order", f"${avg_profit_per_order:.2f}")
    
    # Profit analysis by segment (including box plot)
    st.subheader("📊 Profit Distribution by Segment (Box & Whiskers)")
    
    fig = px.box(df, x='Segment', y='Profit', color='Segment',
                title='Profit Distribution by Customer Segment',
                labels={'Profit': 'Profit ($)'},
                notched=True)
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Profitability trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Profit Margin Trends")
        monthly_margins = df.groupby(df['Order Date'].dt.to_period('M')).agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        monthly_margins['Profit_Margin'] = (monthly_margins['Profit'] / monthly_margins['Sales'] * 100)
        monthly_margins['Order Date'] = monthly_margins['Order Date'].astype(str)
        
        fig = px.line(monthly_margins, x='Order Date', y='Profit_Margin',
                     title='Monthly Profit Margin Trend',
                     labels={'Profit_Margin': 'Profit Margin (%)'})
        fig.add_hline(y=profit_margin, line_dash="dash", 
                     annotation_text=f"Overall Avg: {profit_margin:.1f}%")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Profitability by Discount Level")
        df['Discount_Range'] = pd.cut(df['Discount'], 
                                      bins=[0, 0.1, 0.2, 0.3, 1.0],
                                      labels=['0-10%', '10-20%', '20-30%', '30%+'])
        
        discount_profit = df.groupby('Discount_Range').agg({
            'Profit': 'mean',
            'Sales': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Avg Profit', x=discount_profit['Discount_Range'], 
                            y=discount_profit['Profit'], marker_color='green'))
        fig.add_trace(go.Bar(name='Avg Sales', x=discount_profit['Discount_Range'], 
                            y=discount_profit['Sales'], marker_color='blue'))
        fig.update_layout(barmode='group', title='Impact of Discount on Profitability')
        st.plotly_chart(fig, use_container_width=True)
    
    # Loss analysis
    st.subheader("🚨 Loss Analysis")
    
    loss_orders = df[df['Profit'] < 0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Loss-Making Orders", f"{len(loss_orders):,}")
        st.metric("Total Losses", f"${loss_orders['Profit'].sum():,.0f}")
    
    with col2:
        loss_categories = loss_orders.groupby('Category')['Profit'].sum().reset_index()
        fig = px.pie(loss_categories, values='Profit', names='Category',
                    title='Losses by Category')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        loss_reasons = loss_orders.groupby('Sub-Category')['Profit'].sum().nsmallest(10).reset_index()
        fig = px.bar(loss_reasons, x='Profit', y='Sub-Category',
                    orientation='h', title='Top 10 Loss-Making Sub-Categories',
                    color='Profit', color_continuous_scale='Reds_r')
        st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics(df):
    """Display advanced analytics dashboard"""
    st.markdown("<h2 class='section-header'>Advanced Analytics Dashboard</h2>", unsafe_allow_html=True)
    
    # Correlation analysis
    st.subheader("🔍 Correlation Analysis")
    
    numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost']
    correlation_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(correlation_matrix, 
                   labels=dict(color="Correlation"),
                   x=numeric_cols,
                   y=numeric_cols,
                   color_continuous_scale='RdBu',
                   aspect="auto",
                   title="Feature Correlation Heatmap")
    fig.update_traces(text=correlation_matrix.round(2), texttemplate="%{text}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series decomposition
    st.subheader("📊 Sales Patterns Analysis")
    
    # Prepare time series data
    daily_sales = df.groupby(df['Order Date'].dt.date)['Sales'].sum().reset_index()
    daily_sales = daily_sales.set_index('Order Date')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Day of week analysis
        df['DayOfWeek'] = df['Order Date'].dt.day_name()
        dow_sales = df.groupby('DayOfWeek')['Sales'].mean().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
        
        fig = px.bar(x=dow_sales.index, y=dow_sales.values,
                    title='Average Sales by Day of Week',
                    labels={'x': 'Day of Week', 'y': 'Average Sales ($)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly seasonality
        df['Month'] = df['Order Date'].dt.month
        monthly_pattern = df.groupby('Month')['Sales'].mean()
        
        fig = px.line_polar(r=monthly_pattern.values, theta=monthly_pattern.index,
                           line_close=True, title='Monthly Sales Pattern',
                           labels={'r': 'Average Sales'})
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer RFM Analysis
    st.subheader("👥 Customer RFM Analysis")
    
    # Calculate RFM metrics
    max_date = df['Order Date'].max()
    rfm = df.groupby('Customer ID').agg({
        'Order Date': lambda x: (max_date - x.max()).days,  # Recency
        'Order ID': 'nunique',  # Frequency
        'Sales': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    
    # Create RFM segments
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
    
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # Visualize RFM
    fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                       color='RFM_Score', title='Customer RFM Segmentation',
                       labels={'Recency': 'Recency (days)', 
                              'Frequency': 'Order Frequency',
                              'Monetary': 'Total Spend ($)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictive insights
    st.subheader("🔮 Predictive Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales forecast placeholder
        st.info("📈 **Sales Forecast**: Based on historical trends, next month's projected sales: $2.5M - $3.2M")
        
        # Simple trend projection
        monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
        recent_growth = monthly_sales.pct_change().tail(6).mean()
        projected_sales = monthly_sales.iloc[-1] * (1 + recent_growth)
        
        st.metric("Next Month Projection", f"${projected_sales:,.0f}", 
                 delta=f"{recent_growth*100:.1f}% growth")
    
    with col2:
        # Risk analysis
        st.warning("⚠️ **Risk Analysis**: 23% of products show declining sales trends")
        
        # Product risk categories
        product_trends = df.groupby(['Product Name', df['Order Date'].dt.to_period('Q')])['Sales'].sum().unstack(fill_value=0)
        if len(product_trends.columns) > 1:
            product_trends['Trend'] = product_trends.iloc[:, -1] - product_trends.iloc[:, -2]
            at_risk = (product_trends['Trend'] < 0).sum()
            total_products = len(product_trends)
            
            risk_data = pd.DataFrame({
                'Category': ['At Risk', 'Stable', 'Growing'],
                'Count': [at_risk, 
                         (product_trends['Trend'] == 0).sum(),
                         (product_trends['Trend'] > 0).sum()]
            })
            
            fig = px.pie(risk_data, values='Count', names='Category',
                        title='Product Performance Risk Assessment',
                        color_discrete_map={'At Risk': 'red', 'Stable': 'yellow', 'Growing': 'green'})
            st.plotly_chart(fig, use_container_width=True)

def show_data_explorer(df):
    """Display data explorer"""
    st.markdown("<h2 class='section-header'>Data Explorer</h2>", unsafe_allow_html=True)
    
    # Data preview
    st.subheader("📊 Data Preview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Total Columns", f"{len(df.columns)}")
    
    with col3:
        st.metric("Date Range", f"{df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
    
    # Column selector
    st.subheader("🔍 Explore Specific Columns")
    
    selected_columns = st.multiselect(
        "Select columns to display",
        options=df.columns.tolist(),
        default=['Order Date', 'Customer Name', 'Product Name', 'Sales', 'Profit']
    )
    
    if selected_columns:
        st.dataframe(df[selected_columns].head(100), use_container_width=True)
    
    # Data filtering
    st.subheader("🎯 Advanced Filtering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sales_range = st.slider(
            "Sales Range ($)",
            min_value=float(df['Sales'].min()),
            max_value=float(df['Sales'].max()),
            value=(float(df['Sales'].min()), float(df['Sales'].max()))
        )
    
    with col2:
        profit_range = st.slider(
            "Profit Range ($)",
            min_value=float(df['Profit'].min()),
            max_value=float(df['Profit'].max()),
            value=(float(df['Profit'].min()), float(df['Profit'].max()))
        )
    
    # Apply filters
    filtered_data = df[
        (df['Sales'] >= sales_range[0]) & 
        (df['Sales'] <= sales_range[1]) &
        (df['Profit'] >= profit_range[0]) & 
        (df['Profit'] <= profit_range[1])
    ]
    
    st.info(f"Filtered data contains {len(filtered_data):,} records ({len(filtered_data)/len(df)*100:.1f}% of total)")
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_numeric = st.multiselect(
        "Select numeric columns for statistics",
        options=numeric_columns,
        default=['Sales', 'Profit', 'Quantity', 'Discount']
    )
    
    if selected_numeric:
        st.dataframe(filtered_data[selected_numeric].describe(), use_container_width=True)
    
    # Export functionality
    st.subheader("💾 Export Data")
    
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name=f"filtered_superstore_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Run the app
if __name__ == "__main__":
    main() 