# **Marketing Budget Optimization Dashboard**

A comprehensive data analytics and machine learning platform designed to optimize marketing budget allocation through advanced lead quality analysis, predictive modeling, and performance tracking.

## **Key Features**

### Data Analytics & Visualization
- Interactive data statistics and exploratory analysis
- Lead funnel visualization with conversion metrics
- Time-based trend analysis (daily, weekly patterns)
- Marketing channel performance matrix
- Lead quality score distribution and segmentation

### Machine Learning Capabilities
- Competitive analysis of multiple ML models:
  - Random Forest
  - XGBoost
  - LightGBM
- Advanced model evaluation metrics:
  - ROC curves and AUC scores
  - Confusion matrices
  - Precision-Recall analysis
  - Feature importance visualization

### Lead Quality Analysis
- Composite Lead Quality Scoring (0-100):
  - Interest Level (40 points)
  - Demo Status (40 points)
  - Source Quality (20 points)
- Quality segmentation and trend analysis
- Channel-wise quality distribution

### Interactive Features
- Real-time filtering capabilities:
  - Date range selection
  - Lead owner filtering
  - Marketing source selection
- Dynamic metric updates
- Interactive visualizations
- Custom view options

## **Technical Stack**

### Core Technologies
- Python 3.11.5
- Streamlit (Web Interface)
- Pandas & NumPy (Data Processing)
- Plotly (Interactive Visualizations)
- Scikit-learn (Machine Learning)
- XGBoost & LightGBM (Advanced ML Models)

### Key Libraries
```txt
streamlit
pandas
numpy
plotly
scikit-learn
xgboost
lightgbm
seaborn
matplotlib
```

## **Installation**

1. Clone this repository:
```bash
git clone <your-repository-url>
cd Marketing-Budget-Optimization-Dashboard
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## **Usage**

1. Prepare your data:
   - Ensure your `Marketing_Data.csv` file is in the `data` directory
   - Required columns:
     - Lead Id
     - Marketing Source
     - Lead Owner
     - Interest Level
     - Demo Status
     - Creation Source
     - Lead created
     - Lead Last Update time

2. Launch the dashboard:
```bash
streamlit run src/app.py
```

3. Access the dashboard:
   - Open your web browser
   - Navigate to http://localhost:8501
   - Use the sidebar filters to customize your view

## **Dashboard Sections**

1. **Overview Metrics**
   - Total leads
   - High-interest leads
   - Demo progress
   - Active marketing channels

2. **Data Statistics & Exploratory Analysis**
   - Data overview
   - Missing values analysis
   - Column distributions
   - Key correlations

3. **Lead Funnel Analysis**
   - Conversion stages
   - Stage-wise metrics
   - Conversion rates

4. **Time Analysis**
   - Daily trends
   - Weekly patterns
   - Response time analysis

5. **Marketing Channel Performance**
   - Performance matrix
   - Channel comparison
   - ROI analysis
   - Quality metrics

6. **Lead Quality Analysis**
   - Score distribution
   - Channel-wise quality
   - Temporal trends
   - Segmentation analysis

7. **Machine Learning Insights**
   - Model comparison
   - Performance metrics
   - Feature importance
   - Predictive analytics

## **Data Requirements**

The `Marketing_Data.csv` should contain the following columns:
- Lead Id (unique identifier)
- Marketing Source (channel information)
- Lead Owner (sales representative)
- Interest Level (lead interest category)
- Demo Status (demo completion status)
- Creation Source (lead origin)
- Lead created (timestamp)
- Lead Last Update time (timestamp)
- Additional metadata columns (optional)

## **Acknowledgments**

- Built with Streamlit
- Powered by advanced ML algorithms
- Designed for marketing professionals and data analysts