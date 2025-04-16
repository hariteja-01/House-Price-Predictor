import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# Aggressive caching to improve performance
@st.cache_data(persist=True)
def load_and_preprocess_data():
    # Read data with optimized parameters - ONLY FIRST 100,000 ROWS
    data = pd.read_csv('Real_Estate_Sales_2001-2022_GL.csv', 
                       nrows=100000,     # LIMIT TO 100K ROWS
                       low_memory=False,  
                       parse_dates=['Date Recorded'],
                       infer_datetime_format=True)
    
    # Aggressive preprocessing
    data['Sale Amount'] = pd.to_numeric(data['Sale Amount'], errors='coerce')
    data['Assessed Value'] = pd.to_numeric(data['Assessed Value'], errors='coerce')
    
    # Drop completely unnecessary columns
    columns_to_drop = ['Serial Number', 'OPM remarks', 'Assessor Remarks']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    
    # Quick cleaning
    data = data.dropna(subset=['Sale Amount', 'Assessed Value', 'Property Type'])
    
    return data

# Fastest possible encoding
def fast_encode(data):
    le = LabelEncoder()
    categorical_cols = ['Property Type', 'Residential Type', 'Town', 'Non Use Code']
    
    encoded_data = data.copy()
    for col in categorical_cols:
        if col in encoded_data.columns:
            encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
    
    return encoded_data

# Optimized model training
@st.cache_resource
def train_fast_model(data):
    # Feature selection
    features = ['Assessed Value', 'Sales Ratio', 'Property Type', 'Residential Type', 'Town']
    target = 'Sale Amount'
    
    # Fast encoding
    encoded_data = fast_encode(data)
    
    # Prepare features and target
    X = encoded_data[features]
    y = encoded_data[target]
    
    # Split with stratification for better representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use Gradient Boosting for faster, more efficient prediction
    model = GradientBoostingRegressor(
        n_estimators=100,  # Balanced between speed and accuracy
        max_depth=5,       # Limit depth for faster training
        learning_rate=0.1, 
        random_state=42
    )
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predictions and metrics
    y_pred = model.predict(X_test)
    
    # Store encoders for prediction
    encoders = {}
    for col in ['Property Type', 'Residential Type', 'Town']:
        le = LabelEncoder()
        le.fit(data[col].astype(str))
        encoders[col] = le
    
    return {
        'model': model,
        'encoders': encoders,
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

# Streamlit App with Performance Optimizations
def main():
    st.set_page_config(
        page_title="Real Estate Price Predictor", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title('⚡ Fast Real Estate Price Predictor (100K Sample)')
    
    # Load and preprocess data
    with st.spinner('Loading data (first 100,000 records)...'):
        data = load_and_preprocess_data()
    
    # Train model
    with st.spinner('Training model...'):
        model_results = train_fast_model(data)
    
    # Display dataset size info
    st.info(f"Using {len(data):,} records from the dataset (limited to first 100,000 rows)")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header('Prediction Inputs')
        
        assessed_value = st.number_input('Assessed Value ($)', min_value=0, value=100000, step=10000)
        sales_ratio = st.slider('Sales Ratio', 0.0, 1.0, 0.8, 0.05)
        
        property_type = st.selectbox('Property Type', data['Property Type'].unique())
        residential_type = st.selectbox('Residential Type', data['Residential Type'].unique())
        town = st.selectbox('Town', data['Town'].unique())
        
        predict_button = st.button('Predict Sale Amount')
    
    # Main area
    col1, col2, col3 = st.columns(3)
    col1.metric('Mean Absolute Error', f'${model_results["mae"]:,.2f}')
    col2.metric('Mean Squared Error', f'${model_results["mse"]:,.2f}')
    col3.metric('R-squared', f'{model_results["r2"]:.2%}')
    
    # Prediction
    if predict_button:
        # Create input for prediction
        input_data = pd.DataFrame({
            'Assessed Value': [assessed_value],
            'Sales Ratio': [sales_ratio],
            'Property Type': [model_results['encoders']['Property Type'].transform([str(property_type)])[0]],
            'Residential Type': [model_results['encoders']['Residential Type'].transform([str(residential_type)])[0]],
            'Town': [model_results['encoders']['Town'].transform([str(town)])[0]]
        })
        
        # Make prediction
        predicted_sale_amount = model_results['model'].predict(input_data)[0]
        st.success(f'🏠 Predicted Sale Amount: ${predicted_sale_amount:,.2f}')
    
    # Visualizations
    st.header('Data Visualizations')
    
    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Property Analysis", "Model Performance"])
    
    with tab1:
        # Price distribution
        fig_price = px.histogram(
            data, 
            x='Sale Amount',
            nbins=50,
            title='Distribution of Sale Amounts',
            labels={'Sale Amount': 'Sale Amount ($)'}
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Average price by property type
            property_avg = data.groupby('Property Type')['Sale Amount'].mean().reset_index()
            fig_property = px.bar(
                property_avg,
                x='Property Type',
                y='Sale Amount',
                title='Average Sale Amount by Property Type',
                labels={'Sale Amount': 'Average Sale Amount ($)'}
            )
            st.plotly_chart(fig_property, use_container_width=True)
        
        with col2:
            # Top towns by average price
            town_avg = data.groupby('Town')['Sale Amount'].mean().sort_values(ascending=False).head(10).reset_index()
            fig_town = px.bar(
                town_avg,
                x='Town',
                y='Sale Amount',
                title='Top 10 Towns by Average Sale Amount',
                labels={'Sale Amount': 'Average Sale Amount ($)'}
            )
            st.plotly_chart(fig_town, use_container_width=True)
    
    with tab3:
        # Actual vs Predicted
        pred_df = pd.DataFrame({
            'Actual': model_results['y_test'],
            'Predicted': model_results['y_pred']
        })
        
        fig_pred = px.scatter(
            pred_df, 
            x='Actual', 
            y='Predicted',
            title='Actual vs Predicted Sale Amounts',
            labels={'Actual': 'Actual Sale Amount ($)', 'Predicted': 'Predicted Sale Amount ($)'}
        )
        
        # Add perfect prediction line
        fig_pred.add_shape(
            type='line',
            x0=pred_df['Actual'].min(),
            y0=pred_df['Actual'].min(),
            x1=pred_df['Actual'].max(),
            y1=pred_df['Actual'].max(),
            line=dict(color='red', dash='dash')
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)

if __name__ == '__main__':
    main()