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

# Custom CSS to ensure consistent font styling
st.markdown("""
<style>
    .font-consistent {
        font-family: 'Arial', sans-serif;
        font-size: 16px;
    }
    .metric-container {
        border-radius: 5px;
        background-color: #f9f9f9;
        padding: 10px;
        margin: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .metric-value {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .graph-container {
        border-radius: 10px;
        background-color: #ffffff;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .graph-explanation {
        background-color: #f0f7ff;
        padding: 10px;
        border-radius: 5px;
        margin-top: 5px;
        border-left: 3px solid #3b82f6;
    }
    .confidence-interval {
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        padding: 15px;
        background-color: #f0f7ff;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .confidence-number {
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        font-weight: bold;
        padding: 0 8px;
    }
</style>
""", unsafe_allow_html=True)

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
    data['Month'] = data['Date Recorded'].dt.month
    data['Quarter'] = data['Date Recorded'].dt.quarter
    
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
    features = ['Assessed Value', 'Town', 'Property Type', 'Year', 'Month', 'Quarter']
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
st.write(f"Found {len(filtered_data):,} properties matching your criteria")

# Train model if enough data
if len(filtered_data) >= 50:
    with st.spinner('Training prediction models...'):
        model_results, best_model_name, encoders = train_model(filtered_data)
    
    # Display tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price Prediction", "Market Insights", "Seasonal Analysis", "Model Performance", "Feature Importance"])
    
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
        col1, col2, col3 = st.columns(3)
        
        with col1:
            predict_year = st.slider("Year", min(years), max(years), max(years))
        
        with col2:
            predict_month = st.slider("Month", 1, 12, 6)
            
        with col3:
            predict_quarter = st.selectbox("Quarter", [1, 2, 3, 4])
            predict_residential_type = None
            if 'Residential Type' in filtered_data.columns:
                predict_residential_type = st.selectbox("Residential Type", 
                                                      sorted(filtered_data['Residential Type'].dropna().unique()))
        
        # Make prediction
        if st.button("Predict Price"):
            # Prepare input data
            input_data = {
                'Assessed Value': [assessed_value],
                'Year': [predict_year],
                'Month': [predict_month],
                'Quarter': [predict_quarter],
                'Town_Encoded': [encoders['town'].transform([predict_town])[0]],
                'Property_Type_Encoded': [encoders['property'].transform([predict_property_type])[0]]
            }
            
            if 'residential' in encoders and predict_residential_type is not None:
                input_data['Residential_Type_Encoded'] = [encoders['residential'].transform([predict_residential_type])[0]]
            
            input_df = pd.DataFrame(input_data)
            
            # Get best model
            best_model = model_results[best_model_name]['model']
            
            # Make prediction
            predicted_price = best_model.predict(input_df)[0]
            
            # Calculate confidence interval
            rmse = model_results[best_model_name]['rmse']
            lower_bound = predicted_price - 1.96 * rmse
            upper_bound = predicted_price + 1.96 * rmse
            
            # Display prediction
            st.success(f"**Predicted Price:** ${predicted_price:,.2f}")
            
            st.markdown(
                f"""
                <div class="confidence-interval">
                    <span>95% Confidence Interval:</span>
                    <span class="confidence-number">${max(0, lower_bound):,.2f}</span>
                    <span>to</span>
                    <span class="confidence-number">${upper_bound:,.2f}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
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
                
                cols = st.columns(len(similar_stats))
                for i, (label, value) in enumerate(similar_stats.items()):
                    if 'Price' in label:
                        cols[i].markdown(
                            f"""
                            <div class="metric-container">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">${value:,.2f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        cols[i].markdown(
                            f"""
                            <div class="metric-container">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{value:,}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                if similar['Sale Amount'].mean() > 0:
                    price_diff_pct = (predicted_price - similar['Sale Amount'].mean()) / similar['Sale Amount'].mean() * 100
                    
                    st.markdown(
                        f"""
                        <div class="metric-container">
                            <div class="metric-label">Price Comparison</div>
                            <div class="metric-value">${predicted_price:,.2f}</div>
                            <div>{'▲' if price_diff_pct > 0 else '▼'} {abs(price_diff_pct):.1f}% compared to average</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    comparison_data = pd.DataFrame({
                        'Type': ['Predicted', 'Average', 'Median', 'Min', 'Max'],
                        'Price': [
                            predicted_price, 
                            similar['Sale Amount'].mean(), 
                            similar['Sale Amount'].median(),
                            similar['Sale Amount'].min(),
                            similar['Sale Amount'].max()
                        ]
                    })
                    
                    fig_comp = px.bar(
                        comparison_data, 
                        x='Type', 
                        y='Price',
                        title='Price Comparison',
                        color='Type',
                        text_auto='.2s',
                        labels={'Price': 'Price ($)', 'Type': ''}
                    )
                    fig_comp.update_traces(texttemplate='$%{y:,.2f}', textposition='outside')
                    
                    st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                    st.plotly_chart(fig_comp, use_container_width=True)
                    st.markdown(
                        '<div class="graph-explanation">This graph compares your predicted property price with statistics from similar properties in the same area and of the same type.</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("Market Insights")
        
        yearly_avg = filtered_data.groupby('Year')['Sale Amount'].mean().reset_index()
        
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        fig_trend = px.line(
            yearly_avg, 
            x='Year', 
            y='Sale Amount',
            title='Average Sale Price by Year',
            labels={'Sale Amount': 'Average Sale Price ($)', 'Year': 'Year'},
            markers=True
        )
        
        fig_trend.add_trace(
            go.Scatter(
                x=yearly_avg['Year'],
                y=yearly_avg['Sale Amount'].rolling(window=3).mean(),
                mode='lines',
                name='3-Year Rolling Average',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig_trend.update_layout(
            xaxis=dict(tickmode='linear', dtick=1),
            yaxis=dict(title='Average Sale Price ($)', tickformat='$,.0f'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown(
            '<div class="graph-explanation">This graph shows the yearly average sale prices over time. The trend line helps identify market cycles and long-term price trends.</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            town_avg = filtered_data.groupby('Town')['Sale Amount'].mean().sort_values(ascending=False).head(10).reset_index()
            fig_town = px.bar(
                town_avg,
                x='Town',
                y='Sale Amount',
                title='Top 10 Towns by Average Sale Price',
                labels={'Sale Amount': 'Average Sale Price ($)'},
                color='Sale Amount',
                color_continuous_scale='Viridis',
                text_auto='.2s'
            )
            fig_town.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
            fig_town.update_layout(yaxis=dict(tickformat='$,.0f'))
            
            st.plotly_chart(fig_town, use_container_width=True)
            st.markdown(
                '<div class="graph-explanation">The chart identifies the towns with the highest average property prices.</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            fig_dist = px.histogram(
                filtered_data,
                x='Sale Amount',
                title='Distribution of Sale Prices',
                labels={'Sale Amount': 'Sale Price ($)', 'count': 'Number of Properties'},
                nbins=50,
                marginal='box'
            )
            fig_dist.update_layout(xaxis=dict(tickformat='$,.0f'))
            
            fig_dist.add_vline(
                x=filtered_data['Sale Amount'].mean(), 
                line_dash="dash", 
                line_color="red",
                annotation_text="Mean",
                annotation_position="top"
            )
            fig_dist.add_vline(
                x=filtered_data['Sale Amount'].median(), 
                line_dash="dash", 
                line_color="green",
                annotation_text="Median",
                annotation_position="top"
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown(
                '<div class="graph-explanation">This histogram shows the distribution of property prices. The vertical lines mark the mean (red) and median (green) prices.</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        if len(filtered_data) > 0:
            heatmap_data = filtered_data.pivot_table(
                index='Town',
                columns='Property Type',
                values='Sale Amount',
                aggfunc='mean'
            ).fillna(0)
            
            heatmap_data = heatmap_data.loc[(heatmap_data.sum(axis=1) > 0), (heatmap_data.sum(axis=0) > 0)]
            
            if not heatmap_data.empty and heatmap_data.shape[0] > 1 and heatmap_data.shape[1] > 1:
                fig_heatmap = px.imshow(
                    heatmap_data,
                    text_auto='$,.0f',
                    aspect="auto",
                    title="Average Sale Price by Town and Property Type",
                    labels=dict(x="Property Type", y="Town", color="Sale Price"),
                    color_continuous_scale='Viridis'
                )
                fig_heatmap.update_layout(height=max(400, len(heatmap_data) * 30))
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.markdown(
                    '<div class="graph-explanation">This heatmap displays average property prices by town and property type.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("Not enough data variation to create a meaningful heatmap.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        property_pivot = filtered_data.pivot_table(
            index='Property Type',
            values='Sale Amount',
            aggfunc=['mean', 'median', 'count']
        ).reset_index()
        
        property_pivot.columns = ['Property Type', 'Mean Price', 'Median Price', 'Count']
        
        property_pivot = property_pivot.sort_values('Count', ascending=False)
        
        st.subheader("Price by Property Type")
        
        fig_property = px.bar(
            property_pivot,
            x='Property Type',
            y=['Mean Price', 'Median Price'],
            title='Mean and Median Price by Property Type',
            barmode='group',
            labels={'value': 'Price ($)', 'variable': 'Metric'},
            text_auto='$,.0f',
            hover_data=['Count']
        )
        fig_property.update_traces(textposition='outside')
        fig_property.update_layout(yaxis=dict(tickformat='$,.0f'))
        
        st.plotly_chart(fig_property, use_container_width=True)
        st.markdown(
            '<div class="graph-explanation">This chart shows the average and median prices for different property types.</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.dataframe(property_pivot.style.format({
            'Mean Price': '${:,.2f}',
            'Median Price': '${:,.2f}',
            'Count': '{:,}'
        }), use_container_width=True)
    
    with tab3:
        st.header("Seasonal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            monthly_avg = filtered_data.groupby('Month')['Sale Amount'].mean().reset_index()
            monthly_count = filtered_data.groupby('Month')['Sale Amount'].count().reset_index()
            monthly_count.columns = ['Month', 'Count']
            monthly_data = monthly_avg.merge(monthly_count, on='Month')
            
            fig_monthly = go.Figure()
            
            fig_monthly.add_trace(
                go.Bar(
                    x=monthly_data['Month'],
                    y=monthly_data['Sale Amount'],
                    name='Avg. Price',
                    text=monthly_data['Sale Amount'].apply(lambda x: f'${x:,.0f}'),
                    textposition='outside',
                    marker_color='lightblue'
                )
            )
            
            fig_monthly.add_trace(
                go.Scatter(
                    x=monthly_data['Month'],
                    y=monthly_data['Count'],
                    name='Number of Sales',
                    mode='lines+markers',
                    marker=dict(color='red'),
                    yaxis='y2'
                )
            )
            
            fig_monthly.update_layout(
                title='Monthly Price Trends and Sales Volume',
                xaxis=dict(
                    title='Month',
                    tickvals=list(range(1, 13)),
                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ),
                yaxis=dict(
                    title='Average Price ($)',
                    tickformat='$,.0f',
                    side='left'
                ),
                yaxis2=dict(
                    title='Number of Sales',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
            st.markdown(
                '<div class="graph-explanation">This chart shows how property prices and sales volume vary by month.</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            quarterly_data = filtered_data.groupby(['Year', 'Quarter'])['Sale Amount'].agg(['mean', 'count']).reset_index()
            quarterly_data['YearQuarter'] = quarterly_data['Year'].astype(str) + '-Q' + quarterly_data['Quarter'].astype(str)
            
            recent_years = sorted(filtered_data['Year'].unique())[-5:]
            recent_quarterly = quarterly_data[quarterly_data['Year'].isin(recent_years)]
            
            fig_quarterly = px.line(
                recent_quarterly,
                x='YearQuarter',
                y='mean',
                markers=True,
                title=f'Quarterly Price Trends (Last {len(recent_years)} Years)',
                labels={'mean': 'Average Price ($)', 'YearQuarter': 'Year-Quarter'},
                text='mean'
            )
            
            fig_quarterly.update_traces(
                texttemplate='$%{y:,.0f}',
                textposition='top center'
            )
            
            fig_quarterly.update_layout(
                xaxis=dict(tickangle=45),
                yaxis=dict(tickformat='$,.0f')
            )
            
            st.plotly_chart(fig_quarterly, use_container_width=True)
            st.markdown(
                '<div class="graph-explanation">This chart displays quarterly price trends over the most recent years.</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        year_month_pivot = filtered_data.pivot_table(
            index='Year',
            columns='Month',
            values='Sale Amount',
            aggfunc='mean'
        )
        
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        
        year_month_pivot.columns = [month_names[month] for month in year_month_pivot.columns]
        
        fig_year_month = px.imshow(
            year_month_pivot,
            text_auto='$,.0f',
            aspect="auto",
            title="Average Sale Price by Year and Month",
            labels=dict(x="Month", y="Year", color="Average Price ($)"),
            color_continuous_scale='Viridis'
        )
        fig_year_month.update_layout(height=max(400, len(year_month_pivot) * 30))
        
        st.plotly_chart(fig_year_month, use_container_width=True)
        st.markdown(
            '<div class="graph-explanation">This heatmap shows average sale prices across years and months.</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.header("Model Performance")
        
        st.subheader("Model Comparison")
        metrics_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'RMSE': [results['rmse'] for results in model_results.values()],
            'MAE': [results['mae'] for results in model_results.values()],
            'R² Score': [results['r2'] for results in model_results.values()],
            'MAPE (%)': [results['mape'] for results in model_results.values()]
        })
        
        st.dataframe(metrics_df.style.format({
            'RMSE': '{:,.2f}',
            'MAE': '{:,.2f}',
            'R² Score': '{:.3f}',
            'MAPE (%)': '{:.2f}'
        }).highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='lightgreen')
        .highlight_max(subset=['R² Score'], color='lightgreen'), 
        use_container_width=True)
        
        st.markdown(
            '<div class="graph-explanation">This table compares different models based on various metrics.</div>',
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        best_results = model_results[best_model_name]
        fig_actual_pred = go.Figure()
        
        fig_actual_pred.add_trace(
            go.Scatter(
                x=best_results['y_test'],
                y=best_results['y_pred'],
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', opacity=0.5)
            )
        )
        
        min_val = min(min(best_results['y_test']), min(best_results['y_pred']))
        max_val = max(max(best_results['y_test']), max(best_results['y_pred']))
        fig_actual_pred.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig_actual_pred.update_layout(
            title=f'Actual vs Predicted Prices ({best_model_name})',
            xaxis_title='Actual Price ($)',
            yaxis_title='Predicted Price ($)',
            xaxis=dict(tickformat='$,.0f'),
            yaxis=dict(tickformat='$,.0f'),
            showlegend=True
        )
        
        st.plotly_chart(fig_actual_pred, use_container_width=True)
        st.markdown(
            f'<div class="graph-explanation">This scatter plot shows how well the {best_model_name} predictions align with actual prices.</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.header("Feature Importance")
        
        best_model = model_results[best_model_name]['model']
        feature_names = prepare_model_data(filtered_data)[0].columns
        
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            fig_importance = px.bar(
                feature_importance_df,
                x='Feature',
                y='Importance',
                title=f'Feature Importance ({best_model_name})',
                text_auto='.3f',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig_importance.update_traces(textposition='outside')
            fig_importance.update_layout(xaxis_title='Features', yaxis_title='Importance Score')
            
            st.plotly_chart(fig_importance, use_container_width=True)
            st.markdown(
                '<div class="graph-explanation">This chart shows which features most influence price predictions. Higher importance scores indicate stronger impact on the model’s predictions.</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.dataframe(feature_importance_df.style.format({'Importance': '{:.3f}'}), 
                        use_container_width=True)
        else:
            st.info(f"Feature importance is not available for {best_model_name}")

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666;'>
            Real Estate Price Prediction System | Data from 2001-2022 | Last updated: {datetime.now().strftime('%B %d, %Y')}
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.warning("Not enough data to train models (minimum 50 properties required). Please adjust filters.")