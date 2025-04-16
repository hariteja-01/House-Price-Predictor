import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import re

# Set page configuration
st.set_page_config(
    page_title="Advanced Real Estate Price Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme toggle
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
    <style>
    body { background-color: #1E1E1E; color: #FFFFFF; }
    .stApp { background-color: #1E1E1E; color: #FFFFFF; }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏠 Advanced Real Estate Price Prediction System")
st.markdown("""
This system provides cutting-edge real estate price predictions, market insights, and geographical analysis.
Using data from 2001-2022, it offers precise predictions, advanced visualizations, and real-time market trends.
""")

# Load the dataset with caching for performance
@st.cache_data(ttl=3600)
def load_data():
    data = pd.read_csv('Real_Estate_Sales_2001-2022_GL.csv', nrows=100000)
    numeric_cols = ['Assessed Value', 'Sale Amount', 'Sales Ratio']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['Date Recorded'] = pd.to_datetime(data['Date Recorded'], errors='coerce')
    data['Year'] = data['Date Recorded'].dt.year.astype('Int64')
    data = data.dropna(subset=['Sale Amount', 'Assessed Value', 'Property Type', 'Town'])
    data['Price_to_Assessment_Ratio'] = data['Sale Amount'] / data['Assessed Value']
    
    if 'Location' in data.columns:
        def extract_coords(location):
            if pd.isna(location):
                return None, None
            match = re.search(r'\(([^,]+),\s*([^\)]+)\)', str(location))
            if match:
                return float(match.group(1)), float(match.group(2))
            return None, None
        data[['Latitude', 'Longitude']] = data['Location'].apply(lambda x: pd.Series(extract_coords(x)))
    
    return data

with st.spinner('Loading and processing data...'):
    data = load_data()

# Sidebar filters
st.sidebar.header("Search Properties")
towns = sorted(data['Town'].unique())
selected_town = st.sidebar.selectbox("Select Town", ['All'] + list(towns))
property_types = sorted(data['Property Type'].unique())
selected_property_type = st.sidebar.selectbox("Property Type", ['All'] + list(property_types))
min_price = int(data['Sale Amount'].min())
max_price = int(data['Sale Amount'].max())
price_range = st.sidebar.slider("Price Range ($)", min_price, max_price, (min_price, max_price))
years = sorted(data['Year'].dropna().unique())
year_range = st.sidebar.slider("Year Range", min(years), max(years), (min(years), max(years)))
if 'Residential Type' in data.columns:
    residential_types = sorted(data['Residential Type'].dropna().unique())
    selected_residential_type = st.sidebar.selectbox("Residential Type", ['All'] + list(residential_types))
else:
    selected_residential_type = 'All'

# Reset filters button
if st.sidebar.button("Reset Filters"):
    selected_town = 'All'
    selected_property_type = 'All'
    selected_residential_type = 'All'
    price_range = (min_price, max_price)
    year_range = (min(years), max(years))

# Apply filters
filtered_data = data.copy()
if selected_town != 'All':
    filtered_data = filtered_data[filtered_data['Town'] == selected_town]
if selected_property_type != 'All':
    filtered_data = filtered_data[filtered_data['Property Type'] == selected_property_type]
if selected_residential_type != 'All':
    filtered_data = filtered_data[filtered_data['Residential Type'] == selected_residential_type]
filtered_data = filtered_data[
    (filtered_data['Sale Amount'] >= price_range[0]) & 
    (filtered_data['Sale Amount'] <= price_range[1]) &
    (filtered_data['Year'] >= year_range[0]) & 
    (filtered_data['Year'] <= year_range[1])
]

# Feature Engineering and Model Training
def prepare_model_data(data):
    features = ['Assessed Value', 'Town', 'Property Type', 'Year']
    if 'Residential Type' in data.columns:
        features.append('Residential Type')
    X = data[features].copy()
    y = data['Sale Amount']
    encoders = {}
    encoders['town'] = LabelEncoder()
    encoders['property'] = LabelEncoder()
    X['Town_Encoded'] = encoders['town'].fit_transform(X['Town'])
    X['Property_Type_Encoded'] = encoders['property'].fit_transform(X['Property Type'])
    if 'Residential Type' in X.columns:
        encoders['residential'] = LabelEncoder()
        X['Residential_Type_Encoded'] = encoders['residential'].fit_transform(X['Residential Type'].fillna('None'))
        X = X.drop(['Residential Type'], axis=1)
    X = X.drop(['Town', 'Property Type'], axis=1)
    return X, y, encoders

@st.cache_resource
def train_model(data):
    X, y, encoders = prepare_model_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model': model,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
            'y_test': y_test,
            'y_pred': y_pred
        }
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    return results, best_model[0], encoders

st.write(f"Found {len(filtered_data)} properties matching your criteria")

if len(filtered_data) >= 50:
    with st.spinner('Training prediction models...'):
        model_results, best_model_name, encoders = train_model(filtered_data)
    
    tabs = st.tabs([
        "Price Prediction", "Market Insights", "Model Performance", 
        "Feature Importance", "Geographical Analysis", "Price Trends & Correlations", "Real-Time Estimations"
    ])
    
    # Tab 1: Price Prediction
    with tabs[0]:
        st.header("Price Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            assessed_value = st.number_input("Assessed Value ($)", min_value=10000, max_value=1000000, value=100000, step=10000)
        with col2:
            predict_town = st.selectbox("Town", sorted(filtered_data['Town'].unique()))
        with col3:
            predict_property_type = st.selectbox("Property Type", sorted(filtered_data['Property Type'].unique()))
        
        col1, col2 = st.columns(2)
        with col1:
            predict_year = st.slider("Year", min(years), max(years), max(years))
        with col2:
            predict_residential_type = None
            if 'Residential Type' in filtered_data.columns:
                predict_residential_type = st.selectbox("Residential Type", sorted(filtered_data['Residential Type'].dropna().unique()))
        
        # Live price range update
        st.write(f"Current Price Range Selected: ${price_range[0]:,} - ${price_range[1]:,}")
        
        if st.button("Predict Price"):
            input_data = {
                'Assessed Value': [assessed_value],
                'Year': [predict_year],
                'Town_Encoded': [encoders['town'].transform([predict_town])[0]],
                'Property_Type_Encoded': [encoders['property'].transform([predict_property_type])[0]]
            }
            if 'residential' in encoders and predict_residential_type:
                input_data['Residential_Type_Encoded'] = [encoders['residential'].transform([predict_residential_type])[0]]
            
            input_df = pd.DataFrame(input_data)
            best_model = model_results[best_model_name]['model']
            predicted_price = best_model.predict(input_df)[0]
            rmse = model_results[best_model_name]['rmse']
            lower_bound = predicted_price - 1.96 * rmse
            upper_bound = predicted_price + 1.96 * rmse
            
            st.success(f"**Predicted Price:** ${predicted_price:,.2f}")
            st.info(f"**95% Confidence Interval:** ${max(0, lower_bound):,.2f} to ${upper_bound:,.2f}")
            
            with st.expander("Similar Properties"):
                similar = filtered_data[
                    (filtered_data['Town'] == predict_town) &
                    (filtered_data['Property Type'] == predict_property_type)
                ]
                if len(similar) > 0:
                    similar_stats = {
                        'Average Price': similar['Sale Amount'].mean(),
                        'Median Price': similar['Sale Amount'].median(),
                        'Min Price': similar['Sale Amount'].min(),
                        'Max Price': similar['Sale Amount'].max(),
                        'Count': len(similar)
                    }
                    cols = st.columns(5)
                    for i, (label, value) in enumerate(similar_stats.items()):
                        if 'Price' in label:
                            cols[i].metric(label, f"${value:,.2f}")
                        else:
                            cols[i].metric(label, value)
    
    # Tab 2: Market Insights
    with tabs[1]:
        st.header("Market Insights")
        col1, col2 = st.columns(2)
        with col1:
            town_avg = filtered_data.groupby('Town')['Sale Amount'].mean().sort_values(ascending=False).head(10)
            fig_bar = px.bar(town_avg, x=town_avg.index, y='Sale Amount', title="Top 10 Towns by Avg Price")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            prop_counts = filtered_data['Property Type'].value_counts()
            fig_pie = px.pie(values=prop_counts.values, names=prop_counts.index, title="Property Type Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with st.expander("Download Filtered Data"):
            csv = filtered_data.to_csv(index=False)
            st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")
    
    # Tab 3: Model Performance
    with tabs[2]:
        st.header("Model Performance")
        metrics_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'RMSE': [model_results[m]['rmse'] for m in model_results],
            'MAE': [model_results[m]['mae'] for m in model_results],
            'R²': [model_results[m]['r2'] for m in model_results],
            'MAPE (%)': [model_results[m]['mape'] for m in model_results]
        })
        st.dataframe(metrics_df.style.format({'RMSE': '{:,.2f}', 'MAE': '{:,.2f}', 'R²': '{:.3f}', 'MAPE (%)': '{:.2f}%'}))
        st.success(f"**Best Model:** {best_model_name}")
        
        fig_scatter = px.scatter(x=model_results[best_model_name]['y_test'], y=model_results[best_model_name]['y_pred'], 
                                 labels={'x': 'Actual Price', 'y': 'Predicted Price'}, title="Actual vs Predicted Prices")
        fig_scatter.add_trace(go.Scatter(x=[min(years), max(years)], y=[min(years), max(years)], mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tab 4: Feature Importance
    with tabs[3]:
        st.header("Feature Importance")
        rf_model = model_results['Random Forest']['model']
        X, _, _ = prepare_model_data(filtered_data)
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=False)
        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Geographical Analysis
    with tabs[4]:
        st.header("Geographical Analysis")
        if 'Latitude' in filtered_data.columns and 'Longitude' in filtered_data.columns:
            fig_map = px.scatter_mapbox(filtered_data.dropna(subset=['Latitude', 'Longitude']),
                                        lat="Latitude", lon="Longitude", color="Sale Amount", size="Assessed Value",
                                        hover_data=["Town", "Property Type"], zoom=8, height=600,
                                        title="Property Locations by Sale Amount")
            fig_map.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Geographical data not available in the dataset.")
    
    # Tab 6: Price Trends & Correlations
    with tabs[5]:
        st.header("Price Trends & Correlations")
        col1, col2 = st.columns(2)
        with col1:
            yearly_avg = filtered_data.groupby('Year')['Sale Amount'].mean().reset_index()
            fig_trend = px.line(yearly_avg, x='Year', y='Sale Amount', title="Average Price Trend Over Years")
            st.plotly_chart(fig_trend, use_container_width=True)
        with col2:
            corr = filtered_data[['Sale Amount', 'Assessed Value', 'Sales Ratio', 'Year']].corr()
            fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Tab 7: Real-Time Estimations
    with tabs[6]:
        st.header("Real-Time Estimations")
        with st.expander("Inflation-Adjusted Price Prediction"):
            inflation_rate = st.slider("Annual Inflation Rate (%)", 0.0, 10.0, 3.0) / 100
            current_year = 2025
            years_diff = current_year - predict_year
            adjusted_price = predicted_price * (1 + inflation_rate) ** years_diff
            st.success(f"**Inflation-Adjusted Price (2025):** ${adjusted_price:,.2f}")
            st.info(f"Based on a {inflation_rate*100:.1f}% annual inflation rate from {predict_year} to {current_year}")
else:
    st.warning("Not enough data to train the model. Please adjust your filters to include more properties.")

# Footer
st.markdown("---")
st.caption("Created by Hari Teja Patnala | Enhanced by xAI | Updated: April 06, 2025")