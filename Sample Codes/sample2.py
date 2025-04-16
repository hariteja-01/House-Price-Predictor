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

# Set page configuration
st.set_page_config(
    page_title="Real Estate Price Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏠 Real Estate Price Prediction System")
st.markdown("""
This system predicts real estate prices based on your preferences and provides market insights.
Using data from 2001-2022, it offers accurate predictions with confidence intervals.
""")

# Load the dataset with caching for performance
@st.cache_data(ttl=3600)
def load_data():
    # Load only first 100,000 rows for performance
    data = pd.read_csv('Real_Estate_Sales_2001-2022_GL.csv', nrows=100000)
    
    # Convert columns to appropriate types
    numeric_cols = ['Assessed Value', 'Sale Amount']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Convert date
    data['Date Recorded'] = pd.to_datetime(data['Date Recorded'], errors='coerce')
    data['Year'] = data['Date Recorded'].dt.year
    
    # Drop rows with missing critical values
    data = data.dropna(subset=['Sale Amount', 'Assessed Value', 'Property Type'])
    
    # Feature engineering
    data['Price_to_Assessment_Ratio'] = data['Sale Amount'] / data['Assessed Value']
    
    return data

# Display loading spinner
with st.spinner('Loading and processing data...'):
    data = load_data()

# Sidebar filters
st.sidebar.header("Search Properties")

# Town filter
towns = sorted(data['Town'].unique())
selected_town = st.sidebar.selectbox("Select Town", ['All'] + towns)

# Property type filter
property_types = sorted(data['Property Type'].unique())
selected_property_type = st.sidebar.selectbox("Property Type", ['All'] + property_types)

# Price range
min_price = int(data['Sale Amount'].min())
max_price = int(data['Sale Amount'].max())
price_range = st.sidebar.slider(
    "Price Range ($)", 
    min_price, 
    max_price, 
    (min_price, max_price)
)

# Year range
years = sorted(data['Year'].unique())
year_range = st.sidebar.slider(
    "Year Range",
    int(min(years)),
    int(max(years)),
    (int(min(years)), int(max(years)))
)

# Residential type filter if available
if 'Residential Type' in data.columns:
    residential_types = sorted(data['Residential Type'].dropna().unique())
    selected_residential_type = st.sidebar.selectbox("Residential Type", ['All'] + list(residential_types))
else:
    selected_residential_type = 'All'

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
    # Select features
    features = ['Assessed Value', 'Town', 'Property Type', 'Year']
    if 'Residential Type' in data.columns:
        features.append('Residential Type')
    
    X = data[features].copy()
    y = data['Sale Amount']
    
    # Encode categorical features
    town_encoder = LabelEncoder()
    property_encoder = LabelEncoder()
    X['Town_Encoded'] = town_encoder.fit_transform(X['Town'])
    X['Property_Type_Encoded'] = property_encoder.fit_transform(X['Property Type'])
    
    residential_encoder = None
    if 'Residential Type' in X.columns:
        residential_encoder = LabelEncoder()
        X['Residential_Type_Encoded'] = residential_encoder.fit_transform(X['Residential Type'].fillna('None'))
    
    # Drop original categorical columns
    X = X.drop(['Town', 'Property Type'], axis=1)
    if 'Residential Type' in X.columns:
        X = X.drop(['Residential Type'], axis=1)
    
    # Return encoders dictionary including residential encoder
    encoders = {
        'town': town_encoder, 
        'property': property_encoder
    }
    
    if residential_encoder is not None:
        encoders['residential'] = residential_encoder
    
    return X, y, encoders

# Train model with caching for performance
@st.cache_resource
def train_model(data):
    X, y, encoders = prepare_model_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    
    return results, best_model[0], encoders

# Show filtered data count
st.write(f"Found {len(filtered_data)} properties matching your criteria")

# Train model if enough data
if len(filtered_data) >= 50:
    with st.spinner('Training prediction models...'):
        model_results, best_model_name, encoders = train_model(filtered_data)
    
    # Display tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Price Prediction", "Market Insights", "Model Performance", "Feature Importance"])
    
    with tab1:
        st.header("Price Prediction")
        
        # User input for prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            assessed_value = st.number_input("Assessed Value ($)", min_value=10000, max_value=1000000, value=100000, step=10000)
        
        with col2:
            predict_town = st.selectbox("Town", sorted(filtered_data['Town'].unique()))
        
        with col3:
            predict_property_type = st.selectbox("Property Type", sorted(filtered_data['Property Type'].unique()))
        
        # Additional inputs
        col1, col2 = st.columns(2)
        
        with col1:
            predict_year = st.slider("Year", min(years), max(years), max(years))
        
        with col2:
            predict_residential_type = None
            if 'Residential Type' in filtered_data.columns:
                predict_residential_type = st.selectbox("Residential Type", 
                                                      sorted(filtered_data['Residential Type'].dropna().unique()))
        
        # Make prediction
        if st.button("Predict Price"):
            # Prepare input data with all required features
            input_data = {
                'Assessed Value': [assessed_value],
                'Year': [predict_year],
                'Town_Encoded': [encoders['town'].transform([predict_town])[0]],
                'Property_Type_Encoded': [encoders['property'].transform([predict_property_type])[0]]
            }
            
            # Add Residential_Type_Encoded if it was used during training
            if 'residential' in encoders and predict_residential_type is not None:
                input_data['Residential_Type_Encoded'] = [encoders['residential'].transform([predict_residential_type])[0]]
            
            # Create DataFrame from the input data dictionary
            input_df = pd.DataFrame(input_data)
            
            # Get best model
            best_model = model_results[best_model_name]['model']
            
            # Make prediction
            predicted_price = best_model.predict(input_df)[0]
            
            # Calculate confidence interval (using model's RMSE for simplicity)
            rmse = model_results[best_model_name]['rmse']
            lower_bound = predicted_price - 1.96 * rmse
            upper_bound = predicted_price + 1.96 * rmse
            
            # Display prediction
            st.success(f"**Predicted Price:** ${predicted_price:,.2f}")
            st.info(f"**95% Confidence Interval:** ${max(0, lower_bound):,.2f} to  ${upper_bound:,.2f}")
            
            # Similar properties
            st.subheader("Similar Properties")
            similar = filtered_data[
                (filtered_data['Town'] == predict_town) &
                (filtered_data['Property Type'] == predict_property_type)
            ].sort_values(by='Sale Amount')
            
            if len(similar) > 0:
                similar_stats = {
                    'Average Price': similar['Sale Amount'].mean(),
                    'Median Price': similar['Sale Amount'].median(),
                    'Min Price': similar['Sale Amount'].min(),
                    'Max Price': similar['Sale Amount'].max(),
                    'Count': len(similar)
                }
                
                # Create columns for stats
                cols = st.columns(len(similar_stats))
                for i, (label, value) in enumerate(similar_stats.items()):
                    if 'Price' in label:
                        cols[i].metric(label, f"${value:,.2f}")
                    else:
                        cols[i].metric(label, value)
                
                # Show comparison to similar properties
                if similar['Sale Amount'].mean() > 0:
                    price_diff_pct = (predicted_price - similar['Sale Amount'].mean()) / similar['Sale Amount'].mean() * 100
                    st.metric("Price Comparison", 
                              f"${predicted_price:,.2f}", 
                              f"{price_diff_pct:.1f}% compared to average")
    
    with tab2:
        st.header("Market Insights")
        
        # Price trends over time
        yearly_avg = filtered_data.groupby('Year')['Sale Amount'].mean().reset_index()
        
        fig_trend = px.line(
            yearly_avg, 
            x='Year', 
            y='Sale Amount',
            title='Average Sale Price by Year',
            labels={'Sale Amount': 'Average Sale Price ($)', 'Year': 'Year'}
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Price distribution by town
        col1, col2 = st.columns(2)
        
        with col1:
            town_avg = filtered_data.groupby('Town')['Sale Amount'].mean().sort_values(ascending=False).head(10).reset_index()
            fig_town = px.bar(
                town_avg,
                x='Town',
                y='Sale Amount',
                title='Top 10 Towns by Average Sale Price',
                labels={'Sale Amount': 'Average Sale Price ($)'}
            )
            st.plotly_chart(fig_town, use_container_width=True)
        
        with col2:
            # Distribution of sale amounts
            fig_dist = px.histogram(
                filtered_data,
                x='Sale Amount',
                title='Distribution of Sale Prices',
                labels={'Sale Amount': 'Sale Price ($)'},
                nbins=50
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Price by property type
        property_pivot = filtered_data.pivot_table(
            index='Property Type',
            values='Sale Amount',
            aggfunc=['mean', 'median', 'count']
        ).reset_index()
        
        # Flatten column names
        property_pivot.columns = ['Property Type', 'Mean Price', 'Median Price', 'Count']
        
        # Sort by count
        property_pivot = property_pivot.sort_values('Count', ascending=False)
        
        st.subheader("Price by Property Type")
        st.dataframe(property_pivot.style.format({
            'Mean Price': '${:,.2f}',
            'Median Price': '${:,.2f}',
            'Count': '{:,}'
        }))
    
    with tab3:
        st.header("Model Performance")
        
        # Display metrics for all models
        st.subheader("Model Comparison")
        
        # Create metrics table
        metrics_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'RMSE': [model_results[m]['rmse'] for m in model_results],
            'MAE': [model_results[m]['mae'] for m in model_results],
            'R²': [model_results[m]['r2'] for m in model_results],
            'MAPE (%)': [model_results[m]['mape'] for m in model_results]
        })
        
        st.dataframe(metrics_df.style.format({
            'RMSE': '${:,.2f}',
            'MAE': '${:,.2f}',
            'R²': '{:.3f}',
            'MAPE (%)': '{:.2f}%'
        }))
        
        # Highlight best model
        st.success(f"**Best Model:** {best_model_name}")
        
        # Actual vs Predicted plot
        st.subheader("Actual vs. Predicted Prices")
        fig = px.scatter(
            x=model_results[best_model_name]['y_test'],
            y=model_results[best_model_name]['y_pred'],
            labels={'x': 'Actual Price', 'y': 'Predicted Price'},
            title=f"Actual vs. Predicted Prices ({best_model_name})"
        )
        
        # Add perfect prediction line
        fig.add_trace(go.Scatter(
            x=[model_results[best_model_name]['y_test'].min(), model_results[best_model_name]['y_test'].max()],
            y=[model_results[best_model_name]['y_test'].min(), model_results[best_model_name]['y_test'].max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error distribution
        errors = model_results[best_model_name]['y_test'] - model_results[best_model_name]['y_pred']
        fig_errors = px.histogram(
            x=errors,
            title='Error Distribution',
            labels={'x': 'Prediction Error ($)'}
        )
        st.plotly_chart(fig_errors, use_container_width=True)
    
    with tab4:
        st.header("Feature Importance")
        
        # Get feature importance from Random Forest model
        rf_model = model_results['Random Forest']['model']
        
        # Get feature names
        X, _, _ = prepare_model_data(filtered_data)
        feature_names = X.columns
        
        # Calculate feature importance
        importance = rf_model.feature_importances_
        
        # Create dataframe for feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            labels={'Importance': 'Importance Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature explanation
        st.subheader("What These Features Mean")
        
        feature_explanations = {
            'Assessed Value': "The value of the property as determined by tax assessors. Higher assessed values generally indicate higher property prices.",
            'Year': "The year when the property was sold. This captures market trends and inflation over time.",
            'Town_Encoded': "The location of the property. Location is typically one of the most important factors in real estate pricing.",
            'Property_Type_Encoded': "The type of property (residential, commercial, etc.). Different property types have different market values."
        }
        
        if 'Residential_Type_Encoded' in feature_importance['Feature'].values:
            feature_explanations['Residential_Type_Encoded'] = "The specific type of residential property (single family, condo, etc.). This affects pricing based on demand for different housing types."
        
        for feature, explanation in feature_explanations.items():
            if feature in feature_importance['Feature'].values:
                importance_score = feature_importance[feature_importance['Feature'] == feature]['Importance'].values[0]
                st.markdown(f"**{feature}** (Importance: {importance_score:.4f}): {explanation}")
else:
    st.warning("Not enough data to train the model. Please adjust your filters to include more properties.")

# Footer
st.markdown("---")
st.caption("Created by Hari Teja Patnala")