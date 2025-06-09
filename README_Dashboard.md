# Global Superstore Analytics Dashboard

A beautiful, interactive Streamlit dashboard for comprehensive analysis of Global Superstore data.

## Features

### 📊 Dashboard Views
1. **🏠 Overview** - Executive dashboard with key metrics and trends
2. **📈 Sales Analysis** - Deep dive into sales performance and trends
3. **👥 Customer Insights** - Customer segmentation and behavior analysis
4. **📦 Product Performance** - Product and category analysis
5. **🌍 Geographic Analysis** - Market and regional performance
6. **💰 Profitability Analysis** - Profit margins and loss analysis (includes Box & Whiskers plot)
7. **📊 Advanced Analytics** - Correlation analysis, RFM segmentation, and predictive insights
8. **⚙️ Data Explorer** - Interactive data exploration and export functionality

### ✨ Key Features
- **Interactive Visualizations** using Plotly
- **Real-time Filtering** by date range, segments, and categories
- **Beautiful UI/UX** with custom CSS styling
- **Responsive Design** that works on all screen sizes
- **Data Export** functionality
- **Statistical Analysis** including hypothesis testing
- **Predictive Insights** based on historical trends

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure your data file is in the correct location:**
   - Default path: `C:/Users/maiwa/Downloads/Global Superstore.xlsx`
   - Or modify line 115 in `global_superstore_dashboard.py` to your file path

## Running the Dashboard

1. **Open a terminal in the project directory**

2. **Run the Streamlit app:**
   ```bash
   streamlit run global_superstore_dashboard.py
   ```

3. **The dashboard will open in your default browser** at `http://localhost:8501`

## Usage Guide

### Navigation
- Use the **sidebar** to switch between different dashboard views
- Apply **global filters** to analyze specific segments, categories, or date ranges

### Interactive Features
- **Hover** over charts to see detailed information
- **Click and drag** to zoom into specific areas
- **Double-click** to reset zoom
- Use **tabs** in certain views to see different perspectives

### Key Analyses Available
- Sales trends and forecasting
- Customer segmentation and RFM analysis
- Product profitability matrix
- Geographic performance maps
- Profit distribution by segment (Box & Whiskers)
- Correlation heatmaps
- Seasonal pattern analysis

## Customization

### Changing Data Source
Edit line 115 in `global_superstore_dashboard.py`:
```python
df = pd.read_excel("YOUR_FILE_PATH_HERE")
```

### Modifying Colors/Theme
The dashboard uses custom CSS for styling. You can modify the CSS in the markdown section (lines 20-90) to change colors, fonts, or layout.

### Adding New Visualizations
Each dashboard view is a separate function (e.g., `show_overview()`, `show_sales_analysis()`). You can add new visualizations within these functions or create new dashboard views.

## Performance Tips

1. **Data Caching**: The app uses Streamlit's caching to load data once and reuse it
2. **Filtering**: Use filters to reduce data volume for faster rendering
3. **Large Datasets**: Consider sampling or aggregating data for very large files

## Troubleshooting

### Common Issues:
1. **File not found error**: Check that the Excel file path is correct
2. **Module not found**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
3. **Port already in use**: Try `streamlit run global_superstore_dashboard.py --server.port 8502`

## Screenshots

The dashboard includes:
- Executive overview with key metrics
- Interactive time series analysis
- Customer segmentation visualizations
- Product performance matrices
- Geographic distribution maps
- Profitability analysis with box plots
- Advanced analytics and predictions

## Support

For issues or questions:
1. Check that all dependencies are installed
2. Verify the data file path is correct
3. Ensure your data has the required columns

Enjoy exploring your Global Superstore data! 🚀 