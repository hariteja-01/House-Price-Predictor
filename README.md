# Real Estate House Price & Analytics Predictor

![Real Estate Analytics](https://img.shields.io/badge/Streamlit-App-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

The **Real Estate House Price & Analytics Predictor** is an advanced, interactive web application built with Streamlit, designed to provide real estate professionals, investors, market analysts, and property developers with data-driven insights and accurate price predictions. Leveraging a robust dataset of real estate sales from 2001 to 2022, this tool combines cutting-edge machine learning algorithms, statistical analysis, and geospatial visualizations to deliver a comprehensive platform for property valuation and market trend analysis.

> This project integrates multiple Python libraries for data processing, visualization, and predictive modeling, offering users an intuitive interface to explore historical data, estimate property values, and uncover market dynamics through interactive charts, maps, and detailed reports.

## Features

### 1. Data Exploration & Filtering
- **Comprehensive Filters**: Filter data by town, property type, residential type, price range, year range, months, seasons, and advanced metrics like sales ratio and property flips.
- **Interactive Sidebar**: User-friendly controls to refine datasets dynamically.
- **Dataset Insights**: Real-time metrics on filtered properties, median prices, town diversity, and property type counts.

### 2. Price Prediction
- **Machine Learning Models**: Employs Random Forest, Gradient Boosting, Extra Trees, and Ridge Regression for robust price predictions.
- **Property Value Estimator**: Input property details (e.g., town, property type, assessed value) to get estimated sale prices with prediction intervals.
- **Feature Importance**: Visualizes key factors influencing property values using feature importance and SHAP analysis.

### 3. Market Trends Analysis
- **Time-Based Trends**: Analyze price trends over years and months with dual-axis charts showing price and sales volume.
- **Seasonal Insights**: Explore seasonal price variations and sales distribution by season.
- **Growth Metrics**: Calculate total growth, annualized growth, and recent trends for strategic insights.

### 4. Geospatial Analysis
- **Interactive Maps**: Visualize town-level price distributions using Folium maps with color-coded markers.
- **Town Statistics**: Detailed tables of median prices and sales counts per town.

### 5. Model Performance Evaluation
- **Model Comparison**: Compare multiple models based on R² score, RMSE, MAE, MAPE, and prediction interval coverage.
- **Prediction Accuracy**: Scatter plots of predicted vs. actual prices with residual analysis.
- **Prediction Intervals**: Assess the reliability of predictions with confidence intervals.

### 6. Advanced Insights
- **Raw Data Access**: View and download filtered datasets as CSV files.
- **Statistical Summary**: Detailed descriptive statistics of the dataset.
- **Correlation Analysis**: Heatmap of correlations between numeric features.

## Tech Stack

- **Programming Language**: Python 3.8+
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly, Folium
- **Machine Learning**: Scikit-learn, SHAP
- **Caching**: Streamlit Cache, Joblib
- **File Handling**: Pathlib, OS
- **Styling**: Custom CSS via Streamlit Markdown

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- A compatible IDE (e.g., VS Code, PyCharm)

### Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download the Dataset

*   Obtain the Real\_Estate\_Sales\_2001-2022\_GL.csv dataset (e.g., from a public source or your own data repository).
    
*   Place it in the project root directory.
    

### Run the Application

```bash
streamlit run RealEstateHousePricePredictor.py
```

> Open your browser and navigate to http://localhost:8501.

Requirements
------------

Create a requirements.txt file with the following dependencies:

```bash
streamlit==1.20.0
pandas==1.5.0
numpy==1.23.0
matplotlib==3.6.0
seaborn==0.12.0
plotly==5.10.0
folium==0.13.0
streamlit-folium==0.6.0
scikit-learn==1.1.0
shap==0.41.0
joblib==1.2.0
```

Usage
-----

1.  **Launch the App**: Run the script as described above.
    
2.  **Apply Filters**: Use the sidebar to filter the dataset by location, price, time, and advanced criteria.
    
3.  **Explore Tabs**:
    
    *   **Market Overview**: View price distributions and town comparisons.
        
    *   **Price Estimations**: Estimate property values and analyze influencing factors.
        
    *   **Market Trends**: Study temporal and seasonal trends.
        
    *   **Geographic Analysis**: Explore town-level price maps.
        
    *   **Model Performance**: Evaluate model accuracy and residuals.
        
    *   **Advanced Insights**: Access raw data and statistical summaries.
        
4.  **Interact**: Adjust filters, select chart types, and download data as needed.
    

Dataset
-------

The application uses the Real\_Estate\_Sales\_2001-2022\_GL.csv dataset, which includes:

*   **Columns**: Town, Property Type, Sale Amount, Assessed Value, Date Recorded, Location, etc.
    
*   **Size**: Over 100,000 records (configurable via nrows in the code).
    
*   **Preprocessing**: Handles missing values, extracts coordinates, calculates sales ratios, and creates advanced features like price-to-median ratios and property flip indicators.
    

> **Note**: Ensure the dataset is available in the project directory or update the file path in the code.

Project Structure
-----------------

```bash
real-estate-predictor/
├── app.py # Main application script
├── cache/ # Directory for cached processed data
├── Real_Estate_Sales_2001-2022_GL.csv # Dataset from data.gov
├── requirements.txt # Dependencies
├── README.md # This file
```

Customization
-------------

*   **Theme**: Uncomment and modify the theme toggle in the sidebar for Light, Dark, or Modern Blue styles.
    
*   **Model Tuning**: Adjust hyperparameters in the train\_model function (e.g., n\_estimators, alpha).
    
*   **Map Coordinates**: Replace the placeholder town\_coords dictionary with actual latitude/longitude data for accurate mapping.
    
*   **Data Source**: Modify the load\_data function to use a different dataset or API.
    

Performance Optimization
------------------------

*   **Caching**: Uses @st.cache\_data and @st.cache\_resource to cache data loading and model training, reducing computation time.
    
*   **Data Sampling**: Limits SHAP analysis and large visualizations to manageable sample sizes for faster rendering.
    
*   **Outlier Removal**: Filters extreme sale amounts using IQR to improve model accuracy.
    

Limitations
-----------

*   **Dataset Dependency**: Requires the specific CSV file format; adapt the code for other datasets.
    
*   **Geospatial Accuracy**: Current town coordinates are placeholders; real coordinates are needed for precise mapping.
    
*   **Scalability**: Performance may degrade with very large datasets (>1M rows) without further optimization.
    
*   **SHAP Analysis**: Limited to Random Forest due to computational constraints; may fail with small datasets.
    

Future Enhancements
-------------------

*   **API Integration**: Fetch real-time real estate data from online sources.
    
*   **Advanced Models**: Incorporate deep learning (e.g., TensorFlow) for improved predictions.
    
*   **User Authentication**: Add login features for personalized dashboards.
    
*   **Export Options**: Support PDF reports and additional file formats.
    
*   **Mobile Optimization**: Enhance responsiveness for mobile devices.
    

Contributing
------------

Contributions are welcome! To contribute:

1.  Fork the repository.
    
2.  Create a feature branch (git checkout -b feature/your-feature).
    
3.  Commit your changes (git commit -m "Add your feature").
    
4.  Push to the branch (git push origin feature/your-feature).
    
5.  Open a Pull Request.
    

> Please ensure your code follows PEP 8 guidelines and includes appropriate documentation.

## Project's Screenshots 
#### Landing Page 
<img src="https://github.com/hariteja-01/House-Price-Predictor/blob/main/python_landing_page.png" alt="flowchart" width="500"/>

#### Price Predictions Output 
<img src="https://github.com/hariteja-01/House-Price-Predictor/blob/main/python_price_predctions.png" alt="flowchart" width="500"/> 

#### Market Trend Visualization 
<img src="https://github.com/hariteja-01/House-Price-Predictor/blob/main/python_market_trends.png" alt="flowchart" width="500"/>

#### GeoSpacial Map
<img src="https://github.com/hariteja-01/House-Price-Predictor/blob/main/python_geospacial_map.png" alt="flowchart" width="500"/>

#### Model Performance Metrics
<img src="https://github.com/hariteja-01/House-Price-Predictor/blob/main/python_model_performance.png" alt="flowchart" width="500"/>


Acknowledgments
---------------

*   **Streamlit Team**: For an amazing framework to build interactive apps.
    
*   **Scikit-learn & SHAP**: For robust machine learning and interpretability tools.
    
*   **Plotly & Folium**: For powerful visualization capabilities.
    
*   **Data Providers**: For the real estate sales dataset (https://catalog.data.gov/dataset/real-estate-sales-2001-2018).
    

Contact
-------

For questions, suggestions, or collaboration, reach out via:

*   **GitHub Issues**: [Submit an Issue](https://github.com/hariteja-01/House-Price-Predictor/issues)
    
*   **Email**: \[[Click here!](mailto:patnalahariteja@gmail.com)\]
