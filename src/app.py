import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Marketing Budget Optimization Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Marketing Budget Optimization Dashboard")
st.markdown("""
This enhanced dashboard provides comprehensive analysis of marketing data with advanced analytics and machine learning capabilities.

Key features:
- Detailed data statistics and exploratory analysis
- Lead funnel analysis and conversion metrics
- Marketing channel performance evaluation
- ML model competitive analysis with multiple algorithms
- Feature importance and correlation insights
- Advanced visualizations for budget optimization
""")

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        # Try multiple possible paths for the CSV file
        possible_paths = [
            "data/Marketing_Data.csv",  # Current path in the code
            "../data/Marketing_Data.csv",  # One directory up
            "./data/Marketing_Data.csv",  # Explicit current directory
            "Marketing_Data.csv",  # Direct filename
            str(Path(__file__).parent.parent / "data" / "Marketing_Data.csv"),  # Absolute path
            r"c:\Users\jhadi\Desktop\Divyansh\Daksh\Marketing Budget Optimization Dashboard\data\Marketing_Data.csv"  # Specific full path
        ]
        
        # Try each path until we find the file
        df = None
        successful_path = None
        
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                successful_path = path
                st.success(f"Successfully loaded data from: {path}")
                break
            except Exception:
                continue
        
        if df is None:
            # Last resort - try to find the file using glob
            import glob
            data_files = glob.glob("**/Marketing_Data.csv", recursive=True)
            if data_files:
                try:
                    df = pd.read_csv(data_files[0])
                    st.success(f"Successfully loaded data from discovered path: {data_files[0]}")
                except Exception as e:
                    st.error(f"Found file at {data_files[0]} but couldn't read it: {str(e)}")
            
            if df is None:
                st.error("Could not find the data file. Please ensure 'Marketing_Data.csv' is in the data directory.")
                st.info("Current working directory: " + os.getcwd())
                st.info("Please check if the file is present in this directory or its subdirectories.")
                return None
        
        # Handle missing values
        categorical_columns = ['Marketing Source', 'Lead Owner', 'Interest Level', 
                             'Creation Source', 'What do you do currently ?', 'Demo Status']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Convert date columns to datetime
        date_columns = ['Lead created', 'Lead Last Update time', 'Demo Date', 'Closure date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Create derived features
        if 'Lead created' in df.columns:
            df['hour_of_day'] = df['Lead created'].dt.hour.astype('Int64')
            df['day_of_week'] = df['Lead created'].dt.dayofweek.astype('Int64')
            df['month'] = df['Lead created'].dt.month.astype('Int64')
            df['year'] = df['Lead created'].dt.year.astype('Int64')
            df['week'] = df['Lead created'].dt.isocalendar().week.astype('Int64')
            
            # Add day names for better readability
            day_map = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 
                      4:'Friday', 5:'Saturday', 6:'Sunday'}
            df['day_name'] = df['day_of_week'].map(day_map)
        
        return df
    except Exception as e:
        st.error(f"Error reading the CSV file: {str(e)}")
        return None

@st.cache_data
def generate_data_statistics(df):
    """
    Generate comprehensive statistics about the dataset
    """
    stats = {}
    
    # Basic dataset info
    stats['total_rows'] = len(df)
    stats['total_columns'] = len(df.columns)
    
    # Missing values analysis
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    stats['missing_values'] = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Count': missing_values.values,
        'Missing Percent': missing_percent.values.round(2)
    }).sort_values('Missing Count', ascending=False)
    
    # Categorical columns analysis
    cat_columns = df.select_dtypes(include=['object']).columns
    stats['categorical_stats'] = {}
    
    for col in cat_columns:
        if col in df.columns:
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = [col, 'Count']
            value_counts['Percentage'] = (value_counts['Count'] / len(df) * 100).round(2)
            stats['categorical_stats'][col] = value_counts.head(10)  # Top 10 values
    
    # Numerical columns analysis
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_columns) > 0:
        stats['numerical_stats'] = df[num_columns].describe().T
        
    # Date columns analysis
    date_columns = df.select_dtypes(include=['datetime64']).columns
    stats['date_stats'] = {}
    
    for col in date_columns:
        if col in df.columns:
            stats['date_stats'][col] = {
                'min_date': df[col].min(),
                'max_date': df[col].max(),
                'range_days': (df[col].max() - df[col].min()).days if not pd.isna(df[col].max()) and not pd.isna(df[col].min()) else None
            }
    
    return stats

@st.cache_data
def ml_model_comparison(X_train, X_test, y_train, y_test, feature_cols, class_weight=None):
    """
    Train multiple ML models and compare their performance
    """
    results = {}
    feature_importance_dict = {}
    model_predictions = {}
    confusion_matrices = {}
    
    # Initialize models with balanced class weights
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=class_weight if class_weight else 1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            class_weight='balanced',
            objective='binary',
            random_state=42
        )
    }
    
    # Performance metrics to track
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'train_time', 'inference_time']
    
    # Train and evaluate each model
    for name, model in models.items():
        try:
            # Time the training process
            start_time = time.time()
            
            # Train model
            if name == "XGBoost":
                model.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            train_time = time.time() - start_time
            
            # Time the inference process
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time
            
            # Store predictions for later analysis
            model_predictions[name] = y_pred
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices[name] = cm
            
            # Get probability predictions if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                y_pred_proba = proba[:, 1] if proba.shape[1] >= 2 else y_pred
            else:
                y_pred_proba = y_pred
            
            # Calculate performance metrics
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "train_time": train_time,
                "inference_time": inference_time
            }
            
            # ROC curve
            try:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                results[name]["roc"] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
            except Exception:
                results[name]["roc"] = None
            
            # Precision-Recall curve
            try:
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                results[name]["pr_curve"] = {"precision": precision, "recall": recall}
            except Exception:
                results[name]["pr_curve"] = None
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                # Direct feature importance
                feature_importance_dict[name] = {
                    'direct': pd.Series(
                        model.feature_importances_,
                        index=feature_cols
                    ).sort_values(ascending=False)
                }
                
                # Permutation importance for more robust feature evaluation
                try:
                    perm_importance = permutation_importance(
                        model, X_test, y_test, n_repeats=5, random_state=42
                    )
                    feature_importance_dict[name]['permutation'] = pd.Series(
                        perm_importance.importances_mean,
                        index=feature_cols
                    ).sort_values(ascending=False)
                except Exception:
                    # If permutation importance fails, just use direct importance
                    feature_importance_dict[name]['permutation'] = feature_importance_dict[name]['direct']
        
        except Exception as model_error:
            # Log the error but continue with other models
            print(f"Error training {name} model: {str(model_error)}")
            continue
    
    return {
        'results': results,
        'feature_importance': feature_importance_dict,
        'predictions': model_predictions,
        'confusion_matrices': confusion_matrices
    }

try:
    df = load_data()
    
    if df is None:
        st.warning("Could not automatically load the data file. Please upload it manually.")
        uploaded_file = st.file_uploader("Upload Marketing_Data.csv", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Data successfully loaded from uploaded file!")
    
    if df is not None:
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Lead Owner filter
        if 'Lead Owner' in df.columns:
            selected_owners = st.sidebar.multiselect(
                "Select Lead Owners",
                options=sorted(df['Lead Owner'].unique()),
                default=sorted(df['Lead Owner'].unique())[:5]
            )
        
        # Marketing Source filter
        if 'Marketing Source' in df.columns:
            selected_sources = st.sidebar.multiselect(
                "Select Marketing Sources",
                options=sorted(df['Marketing Source'].unique()),
                default=sorted(df['Marketing Source'].unique())[:5]
            )
        
        # Date range filter
        if 'Lead created' in df.columns:
            min_date = df['Lead created'].min().date()
            max_date = df['Lead created'].max().date()
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        # Apply filters
        filtered_df = df.copy()
        if 'Lead Owner' in df.columns and selected_owners:
            filtered_df = filtered_df[filtered_df['Lead Owner'].isin(selected_owners)]
        if 'Marketing Source' in df.columns and selected_sources:
            filtered_df = filtered_df[filtered_df['Marketing Source'].isin(selected_sources)]
        if 'Lead created' in df.columns and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['Lead created'].dt.date >= date_range[0]) &
                (filtered_df['Lead created'].dt.date <= date_range[1])
            ]

        # Calculate Lead Quality Score early
        def calculate_lead_score(row):
            score = 0
            
            # Interest Level contribution (max 40 points)
            if 'Interest Level' in row and pd.notna(row['Interest Level']):
                interest_level = str(row['Interest Level']).lower()
                if 'high' in interest_level:
                    score += 40
                elif 'medium' in interest_level:
                    score += 30
                elif 'low' in interest_level:
                    score += 20
                else:
                    score += 10  # Base score for any interest level
            
            # Demo Status contribution (max 40 points)
            if 'Demo Status' in row and pd.notna(row['Demo Status']):
                demo_status = str(row['Demo Status']).lower()
                if 'complete' in demo_status or 'done' in demo_status:
                    score += 40
                elif 'schedule' in demo_status or 'book' in demo_status:
                    score += 30
                elif 'pending' in demo_status or 'request' in demo_status:
                    score += 20
                else:
                    score += 10  # Base score for any demo status
            
            # Source Quality contribution (max 20 points)
            if 'Marketing Source' in row and pd.notna(row['Marketing Source']):
                source_leads = filtered_df[filtered_df['Marketing Source'] == row['Marketing Source']]
                if len(source_leads) > 0:
                    high_quality_rate = (
                        source_leads['Interest Level']
                        .str.contains('High|Medium', case=False, na=False)
                        .mean()
                    )
                    score += high_quality_rate * 20
            
            return score

        if 'Interest Level' in filtered_df.columns:
            filtered_df['Lead Quality Score'] = filtered_df.apply(calculate_lead_score, axis=1)

        # Overview metrics
        st.header("Overview Metrics")
        st.markdown("""
        These key performance indicators (KPIs) provide a snapshot of your marketing funnel's health. 
        Track total leads, high-interest prospects, demo progress, and the diversity of your marketing channels.
        """)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_leads = len(filtered_df)
            st.metric("Total Leads", f"{total_leads:,}")
        
        with col2:
            if 'Interest Level' in filtered_df.columns:
                # More inclusive definition of high interest
                high_interest = len(filtered_df[
                    filtered_df['Interest Level'].str.contains('High|Medium|Interested', case=False, na=False)
                ])
                high_interest_pct = (high_interest / total_leads * 100) if total_leads > 0 else 0
                st.metric("High Interest Leads", f"{high_interest:,} ({high_interest_pct:.1f}%)")
        
        with col3:
            if 'Demo Status' in filtered_df.columns:
                # Include both completed and scheduled demos
                completed_demos = len(filtered_df[
                    filtered_df['Demo Status'].str.contains('Complete|Schedule', case=False, na=False)
                ])
                demo_rate = (completed_demos / total_leads * 100) if total_leads > 0 else 0
                st.metric("Demo Progress", f"{completed_demos:,} ({demo_rate:.1f}%)")
        
        with col4:
            if 'Marketing Source' in filtered_df.columns:
                active_channels = filtered_df['Marketing Source'].nunique()
                st.metric("Active Marketing Channels", active_channels)

        # NEW: Data Statistics Section
        st.header("Data Statistics & Exploratory Analysis")
        st.markdown("""
        Explore detailed statistics and patterns in your marketing data to gain deeper insights.
        This section helps you understand data distributions, missing values, and relationships 
        between different variables in your dataset.
        """)
        
        # Generate statistics from the filtered dataset
        stats = generate_data_statistics(filtered_df)
        
        # Create tabs for different statistical views
        stats_tab1, stats_tab2, stats_tab3, stats_tab4 = st.tabs([
            "Data Overview", "Missing Values", "Column Distribution", "Key Correlations"
        ])
        
        with stats_tab1:
            st.subheader("Dataset Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Leads", stats['total_rows'])
                st.metric("Number of Columns", stats['total_columns'])
                
                # Show date range
                if 'date_stats' in stats and 'Lead created' in stats['date_stats']:
                    date_stats = stats['date_stats']['Lead created']
                    st.metric("Date Range (days)", date_stats['range_days'])
                    st.text(f"From: {date_stats['min_date'].date()} To: {date_stats['max_date'].date()}")
            
            with col2:
                # Show data types distribution
                dtypes = filtered_df.dtypes.value_counts()
                fig = px.pie(
                    names=dtypes.index.astype(str),
                    values=dtypes.values,
                    title="Column Data Types Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with stats_tab2:
            st.subheader("Missing Values Analysis")
            
            # Show missing values table
            if len(stats['missing_values']) > 0:
                # Create a bar chart of missing values
                missing_df = stats['missing_values'][stats['missing_values']['Missing Count'] > 0]
                
                if len(missing_df) > 0:
                    fig = px.bar(
                        missing_df,
                        x='Column',
                        y='Missing Percent',
                        title="Missing Values by Column (%)",
                        labels={'Missing Percent': 'Missing (%)'},
                        color='Missing Percent',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("No missing values in the filtered dataset!")
            else:
                st.success("No missing values in the filtered dataset!")
        
        with stats_tab3:
            st.subheader("Column Value Distributions")
            
            # Let user select a column to analyze
            if 'categorical_stats' in stats and len(stats['categorical_stats']) > 0:
                selected_column = st.selectbox(
                    "Select column to analyze:",
                    options=list(stats['categorical_stats'].keys())
                )
                
                if selected_column:
                    col_data = stats['categorical_stats'][selected_column]
                    
                    # Create a bar chart
                    fig = px.bar(
                        col_data,
                        x=selected_column,
                        y='Percentage',
                        title=f"{selected_column} Value Distribution",
                        color='Percentage',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(col_data, use_container_width=True)
            
            # Show numerical columns distribution if available
            if 'numerical_stats' in stats and len(stats['numerical_stats']) > 0:
                st.subheader("Numerical Columns Statistics")
                st.dataframe(stats['numerical_stats'], use_container_width=True)
                
                # Let user select a numerical column to visualize
                num_columns = list(stats['numerical_stats'].index)
                if num_columns:
                    selected_num_col = st.selectbox(
                        "Select numerical column to visualize:",
                        options=num_columns
                    )
                    
                    if selected_num_col and selected_num_col in filtered_df.columns:
                        # Create a histogram
                        fig = px.histogram(
                            filtered_df,
                            x=selected_num_col,
                            title=f"Distribution of {selected_num_col}",
                            color_discrete_sequence=['skyblue'],
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with stats_tab4:
            st.subheader("Key Correlations & Relationships")
            
            # Show correlations between numerical columns if available
            num_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
            
            if len(num_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = filtered_df[num_cols].corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdBu_r',
                    title="Correlation Matrix"
                )
                
                # Add text annotations manually
                for i in range(len(corr_matrix.index)):
                    for j in range(len(corr_matrix.columns)):
                        fig.add_annotation(
                            x=j,
                            y=i,
                            text=f"{corr_matrix.iloc[i, j]:.2f}",
                            showarrow=False,
                            font=dict(color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                        )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Column relationship exploration
                st.subheader("Explore Column Relationships")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("Select X axis:", num_cols)
                
                with col2:
                    y_col = st.selectbox("Select Y axis:", [c for c in num_cols if c != x_col])
                
                if x_col and y_col:
                    # Create scatter plot
                    fig = px.scatter(
                        filtered_df,
                        x=x_col,
                        y=y_col,
                        title=f"Relationship between {x_col} and {y_col}",
                        opacity=0.6
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numerical columns to calculate correlations. Need at least 2 numerical columns.")
            
            # Show categorical relationships with the target variable
            if 'Interest Level' in filtered_df.columns:
                st.subheader("Categorical Variables vs. Target")
                
                cat_cols = [c for c in filtered_df.select_dtypes(include=['object']).columns 
                           if c != 'Interest Level' and filtered_df[c].nunique() < 20]
                
                if cat_cols:
                    selected_cat_col = st.selectbox(
                        "Select categorical column:",
                        options=cat_cols
                    )
                    
                    if selected_cat_col:
                        # Calculate conversion rate by category
                        conversion_by_cat = filtered_df.groupby(selected_cat_col)['Interest Level'].apply(
                            lambda x: x.str.contains('High|Medium|Interested', case=False, na=False).mean() * 100
                        ).reset_index()
                        conversion_by_cat.columns = [selected_cat_col, 'Conversion Rate (%)']
                        conversion_by_cat = conversion_by_cat.sort_values('Conversion Rate (%)', ascending=False)
                        
                        # Create bar chart
                        fig = px.bar(
                            conversion_by_cat,
                            x=selected_cat_col,
                            y='Conversion Rate (%)',
                            title=f"Conversion Rate by {selected_cat_col}",
                            color='Conversion Rate (%)',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(conversion_by_cat, use_container_width=True)
                else:
                    st.info("No suitable categorical columns found for conversion analysis.")

        # After Overview metrics section, add new visualizations
        
        # Lead Funnel Analysis
        st.header("Lead Funnel Analysis")
        st.markdown("""
        The funnel visualization below demonstrates the journey of leads through your sales pipeline. 
        Each stage shows both absolute numbers and conversion rates, helping identify:
        - Initial lead capture effectiveness
        - Interest development success rate
        - Demo scheduling efficiency
        - Final demo completion performance

        💡 **Tip**: Look for significant drops between stages to identify areas needing improvement.
        """)
        
        if all(col in filtered_df.columns for col in ['Interest Level', 'Demo Status']):
            total_leads = len(filtered_df)
            # More inclusive interest level check
            interested = len(filtered_df[
                filtered_df['Interest Level'].str.contains('High|Medium|Interested', case=False, na=False)
            ])
            # More inclusive demo status check
            demo_scheduled = len(filtered_df[
                filtered_df['Demo Status'].str.contains('Schedule|Book|Pending', case=False, na=False)
            ])
            demo_completed = len(filtered_df[
                filtered_df['Demo Status'].str.contains('Complete|Done|Success', case=False, na=False)
            ])
            
            funnel_data = {
                'Stage': ['Total Leads', 'Interested', 'Demo Scheduled', 'Demo Completed'],
                'Count': [total_leads, interested, demo_scheduled, demo_completed]
            }
            
            fig = go.Figure(go.Funnel(
                y=funnel_data['Stage'],
                x=funnel_data['Count'],
                textinfo="value+percent initial",
                textposition="inside",
                textfont=dict(size=14)
            ))
            fig.update_layout(
                title_text="Lead Conversion Funnel",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add conversion rates
            col1, col2, col3 = st.columns(3)
            with col1:
                interest_rate = (interested / total_leads * 100) if total_leads > 0 else 0
                st.metric("Interest Rate", f"{interest_rate:.1f}%")
            with col2:
                demo_rate = (demo_scheduled / interested * 100) if interested > 0 else 0
                st.metric("Demo Schedule Rate", f"{demo_rate:.1f}%")
            with col3:
                completion_rate = (demo_completed / demo_scheduled * 100) if demo_scheduled > 0 else 0
                st.metric("Demo Completion Rate", f"{completion_rate:.1f}%")

        # Enhanced Time Analysis
        st.header("Enhanced Time Analysis")
        st.markdown("""
        Understanding temporal patterns in lead generation and conversion helps optimize your marketing timing and resource allocation.
        The analysis below provides insights into daily, weekly, and response time patterns.
        """)

        tab1, tab2, tab3 = st.tabs(["Daily Trends", "Weekly Patterns", "Response Time"])

        with tab1:
            st.markdown("""
            📈 **Daily Lead Generation Trends**
            - The gray line shows raw daily lead counts
            - Blue line (7-day MA) reveals weekly patterns
            - Red line (30-day MA) shows long-term trends
            
            Use this to:
            - Identify seasonal patterns
            - Spot unusual spikes or drops
            - Plan marketing campaign timing
            """)
            if 'Lead created' in filtered_df.columns:
                # Add moving average
                daily_leads = filtered_df.set_index('Lead created')['Lead Id'].resample('D').count().fillna(0)
                ma7 = daily_leads.rolling(window=7).mean()
                ma30 = daily_leads.rolling(window=30).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily_leads.index, y=daily_leads.values,
                                       name='Daily Leads', mode='lines',
                                       line=dict(color='lightgray')))
                fig.add_trace(go.Scatter(x=ma7.index, y=ma7.values,
                                       name='7-day MA', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=ma30.index, y=ma30.values,
                                       name='30-day MA', line=dict(color='red')))
                fig.update_layout(title="Lead Generation Trends with Moving Averages",
                                xaxis_title="Date",
                                yaxis_title="Number of Leads")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("""
            📅 **Weekly Lead Pattern Analysis**
            Understand how lead generation and quality vary across different days of the week.
            This helps optimize:
            - Marketing campaign scheduling
            - Sales team availability
            - Resource allocation
            """)
            # Add weekly pattern visualization
            if 'day_name' in filtered_df.columns:
                # Create base aggregation for lead counts
                weekly_leads = filtered_df.groupby('day_name').agg({
                    'Lead Id': 'count'
                }).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=weekly_leads['day_name'],
                    y=weekly_leads['Lead Id'],
                    name='Number of Leads',
                    marker_color='#1f77b4',
                    opacity=0.7
                ))
                
                # Only add quality score if it exists
                if 'Lead Quality Score' in filtered_df.columns:
                    quality_scores = filtered_df.groupby('day_name')['Lead Quality Score'].mean().reset_index()
                    fig.add_trace(go.Scatter(
                        x=quality_scores['day_name'],
                        y=quality_scores['Lead Quality Score'],
                        name='Avg Quality Score',
                        yaxis='y2',
                        line=dict(color='#d62728', width=3)
                    ))
                    
                    fig.update_layout(
                        yaxis2=dict(title="Average Quality Score", overlaying='y', side='right')
                    )
                
                fig.update_layout(
                    title=dict(
                        text="Weekly Lead Generation and Quality Patterns",
                        x=0.5,
                        xanchor='center',
                        font=dict(size=20)
                    ),
                    yaxis=dict(title="Number of Leads"),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=100)
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("""
            ⏱️ **Response Time Analysis**
            Track how quickly your team responds to leads and how this affects conversion rates.
            Key metrics include:
            - Average response time
            - Response time distribution
            - Impact on conversion rates
            """)
            # Add response time analysis
            if all(col in filtered_df.columns for col in ['Lead created', 'Lead Last Update time']):
                filtered_df['response_time'] = (filtered_df['Lead Last Update time'] - filtered_df['Lead created']).dt.total_seconds() / 3600  # in hours
                
                fig = px.histogram(
                    filtered_df[filtered_df['response_time'] < filtered_df['response_time'].quantile(0.95)],  # Remove outliers
                    x='response_time',
                    nbins=30,
                    color_discrete_sequence=['#2ca02c'],
                    title="Distribution of Lead Response Times"
                )
                fig.update_layout(
                    xaxis_title="Response Time (Hours)",
                    yaxis_title="Number of Leads",
                    showlegend=False,
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=100)
                )
                st.plotly_chart(fig, use_container_width=True)

        # Marketing Channel Performance Matrix
        st.header("Marketing Channel Performance Matrix")
        st.markdown("""
        This advanced visualization helps evaluate the effectiveness of different marketing channels using multiple key metrics:
        1. 🎯 **Conversion Rate** (y-axis): Percentage of leads showing high interest
        2. 🎥 **Demo Rate** (x-axis): Percentage of leads completing demos
        3. 📊 **Total Leads** (bubble size): Volume of leads generated
        4. 🔄 **Response Rate**: How quickly leads engage
        5. 💰 **Cost Efficiency**: Estimated value per marketing dollar

        **How to interpret**:
        - Bubbles in the top-right quadrant represent the best-performing channels
        - Larger bubbles indicate higher lead volume
        - Use this to optimize budget allocation across channels
        """)
        
        if all(col in filtered_df.columns for col in ['Marketing Source', 'Interest Level', 'Demo Status']):
            # Calculate metrics with more inclusive criteria
            channel_metrics = filtered_df.groupby('Marketing Source').agg({
                'Lead Id': 'count',
                'Interest Level': lambda x: (
                    x.str.contains('High|Medium|Interested', case=False, na=False).sum() / len(x) * 100
                ),
                'Demo Status': lambda x: (
                    x.str.contains('Complete|Schedule', case=False, na=False).sum() / len(x) * 100
                )
            }).reset_index()
            
            channel_metrics.columns = ['Marketing Source', 'Total Leads', 'Conversion Rate', 'Demo Rate']
            
            # Add Lead Quality Score if available
            if 'Lead Quality Score' in filtered_df.columns:
                quality_by_channel = filtered_df.groupby('Marketing Source')['Lead Quality Score'].mean().reset_index()
                channel_metrics = channel_metrics.merge(quality_by_channel, on='Marketing Source')
            
            # Add response time calculation if available
            if all(col in filtered_df.columns for col in ['Lead created', 'Lead Last Update time']):
                filtered_df['response_time'] = (filtered_df['Lead Last Update time'] - filtered_df['Lead created']).dt.total_seconds() / 3600
                response_by_channel = filtered_df.groupby('Marketing Source')['response_time'].median().reset_index()
                response_by_channel.columns = ['Marketing Source', 'Median Response Time (hours)']
                channel_metrics = channel_metrics.merge(response_by_channel, on='Marketing Source')
            
            # Add minimum size for better visibility
            min_size = max(50, channel_metrics['Total Leads'].max() * 0.1)
            channel_metrics['Bubble Size'] = channel_metrics['Total Leads'].apply(
                lambda x: max(x, min_size)
            )
            
            # Create performance matrix plot
            fig = px.scatter(
                channel_metrics,
                x='Demo Rate',
                y='Conversion Rate',
                size='Bubble Size',
                color='Marketing Source',
                hover_data={
                    'Marketing Source': True,
                    'Total Leads': True,
                    'Conversion Rate': ':.1f',
                    'Demo Rate': ':.1f',
                    'Lead Quality Score': ':.1f' if 'Lead Quality Score' in channel_metrics.columns else False,
                    'Median Response Time (hours)': ':.1f' if 'Median Response Time (hours)' in channel_metrics.columns else False,
                    'Bubble Size': False
                },
                labels={
                    'Demo Rate': 'Demo Completion Rate (%)',
                    'Conversion Rate': 'Lead Conversion Rate (%)'
                }
            )
            
            # Add quadrant lines
            x_mid = channel_metrics['Demo Rate'].median()
            y_mid = channel_metrics['Conversion Rate'].median()
            
            fig.add_shape(
                type="line", x0=x_mid, y0=0, x1=x_mid, y1=100,
                line=dict(color="Gray", width=1, dash="dash")
            )
            fig.add_shape(
                type="line", x0=0, y0=y_mid, x1=100, y1=y_mid,
                line=dict(color="Gray", width=1, dash="dash")
            )
            
            # Add quadrant annotations
            fig.add_annotation(
                x=x_mid/2, y=y_mid/2,
                text="Low Performance",
                showarrow=False,
                font=dict(size=12, color="red")
            )
            fig.add_annotation(
                x=x_mid/2, y=y_mid + (100-y_mid)/2,
                text="High Conversion, Low Demo",
                showarrow=False,
                font=dict(size=12, color="orange")
            )
            fig.add_annotation(
                x=x_mid + (100-x_mid)/2, y=y_mid/2,
                text="Low Conversion, High Demo",
                showarrow=False,
                font=dict(size=12, color="blue")
            )
            fig.add_annotation(
                x=x_mid + (100-x_mid)/2, y=y_mid + (100-y_mid)/2,
                text="High Performance",
                showarrow=False,
                font=dict(size=12, color="green")
            )
            
            fig.update_layout(
                title="Marketing Channel Performance Matrix",
                height=600,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Channel Performance Table with sorting
            st.markdown("### Detailed Channel Performance Metrics")
            
            # Add sort options
            sort_by = st.selectbox(
                "Sort channels by:",
                options=[
                    "Total Leads", 
                    "Conversion Rate", 
                    "Demo Rate", 
                    "Lead Quality Score" if "Lead Quality Score" in channel_metrics.columns else None,
                    "Median Response Time (hours)" if "Median Response Time (hours)" in channel_metrics.columns else None
                ],
                index=0
            )
            
            # Sort and display
            sorted_metrics = channel_metrics.sort_values(sort_by, ascending=False if sort_by != "Median Response Time (hours)" else True)
            st.dataframe(sorted_metrics.drop(columns=['Bubble Size']), use_container_width=True)
            
            # Marketing Channel Comparative Analysis
            st.subheader("Marketing Channel Comparative Analysis")
            
            # Select top channels for detailed comparison
            top_channels_count = min(5, len(channel_metrics))
            top_channels = channel_metrics.nlargest(top_channels_count, 'Total Leads')['Marketing Source'].tolist()
            
            # Create comparison metrics
            comparison_metrics = [
                "Conversion Rate",
                "Demo Rate",
                "Total Leads"
            ]
            
            if "Lead Quality Score" in channel_metrics.columns:
                comparison_metrics.append("Lead Quality Score")
            
            if "Median Response Time (hours)" in channel_metrics.columns:
                comparison_metrics.append("Median Response Time (hours)")
            
            # Select metrics to compare
            selected_metrics = st.multiselect(
                "Select metrics to compare:",
                options=comparison_metrics,
                default=comparison_metrics[:3]
            )
            
            if selected_metrics:
                # Create comparison dataframe
                comparison_data = channel_metrics[channel_metrics['Marketing Source'].isin(top_channels)]
                
                # Create separate charts for each metric
                for metric in selected_metrics:
                    fig = px.bar(
                        comparison_data,
                        x='Marketing Source',
                        y=metric,
                        title=f"{metric} by Marketing Channel",
                        color=metric,
                        color_continuous_scale='Viridis' if metric != "Median Response Time (hours)" else 'Viridis_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Lead Quality Score Distribution
        st.header("Lead Quality Score Distribution")
        st.markdown("""
        The Lead Quality Score is a composite metric (0-100) that helps predict lead conversion probability:

        **Score Components**:
        - 🎯 Interest Level (40 points)
          - High: 40 points
          - Medium: 30 points
          - Low: 20 points
          - Other: 10 points
        - 🎥 Demo Status (40 points)
          - Completed: 40 points
          - Scheduled: 30 points
          - Requested: 20 points
          - Other: 10 points
        - 📊 Source Quality (20 points)
          - Based on historical performance of the lead source

        **Using this visualization**:
        - Identify quality distribution patterns
        - Set appropriate quality thresholds
        - Monitor changes in lead quality over time
        """)
        
        if 'Interest Level' in filtered_df.columns:
            # Tabs for different quality score views
            quality_tab1, quality_tab2, quality_tab3 = st.tabs([
                "Quality Distribution", "Quality by Channel", "Quality Trends"
            ])
            
            with quality_tab1:
                # Create histogram with KDE overlay
                fig = px.histogram(
                    filtered_df,
                    x='Lead Quality Score',
                    nbins=20,
                    title="Distribution of Lead Quality Scores",
                    labels={
                        'Lead Quality Score': 'Quality Score (0-100)',
                        'count': 'Number of Leads'
                    },
                    color_discrete_sequence=['skyblue']
                )
                
                # Add distribution statistics
                q_mean = filtered_df['Lead Quality Score'].mean()
                q_median = filtered_df['Lead Quality Score'].median()
                q_std = filtered_df['Lead Quality Score'].std()
                
                fig.add_vline(x=q_mean, line_dash="solid", line_color="red",
                            annotation_text=f"Mean: {q_mean:.1f}", annotation_position="top right")
                fig.add_vline(x=q_median, line_dash="dash", line_color="green",
                            annotation_text=f"Median: {q_median:.1f}", annotation_position="top left")
                
                fig.update_layout(
                    showlegend=False,
                    height=400,
                    bargap=0.1
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show quality score statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Score", f"{q_mean:.1f}")
                col2.metric("Median Score", f"{q_median:.1f}")
                col3.metric("Min Score", f"{filtered_df['Lead Quality Score'].min():.1f}")
                col4.metric("Max Score", f"{filtered_df['Lead Quality Score'].max():.1f}")
                
                # Quality score segmentation
                st.subheader("Lead Quality Segmentation")
                
                # Create quality segments
                filtered_df['Quality Segment'] = pd.cut(
                    filtered_df['Lead Quality Score'],
                    bins=[0, 25, 50, 75, 100],
                    labels=['Low (0-25)', 'Medium (26-50)', 'High (51-75)', 'Premium (76-100)']
                )
                
                segment_counts = filtered_df['Quality Segment'].value_counts().reset_index()
                segment_counts.columns = ['Segment', 'Count']
                segment_counts['Percentage'] = (segment_counts['Count'] / segment_counts['Count'].sum() * 100).round(1)
                
                # Create pie chart
                fig = px.pie(
                    segment_counts,
                    values='Count',
                    names='Segment',
                    title="Lead Quality Segments",
                    color='Segment',
                    color_discrete_map={
                        'Low (0-25)': 'red',
                        'Medium (26-50)': 'orange',
                        'High (51-75)': 'lightgreen',
                        'Premium (76-100)': 'darkgreen'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with quality_tab2:
            # Quality Score by Channel
                avg_scores = filtered_df.groupby('Marketing Source')['Lead Quality Score'].agg(['mean', 'median', 'std', 'count']).round(2)
                avg_scores.columns = ['Average Score', 'Median Score', 'Std Deviation', 'Total Leads']
                
                # Sort by average score descending
                avg_scores = avg_scores.sort_values('Average Score', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    avg_scores.reset_index(),
                    x='Marketing Source',
                    y='Average Score',
                    error_y='Std Deviation',
                    title="Average Lead Quality Score by Channel",
                    color='Average Score',
                    color_continuous_scale='Viridis',
                    hover_data=['Median Score', 'Total Leads']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show table
                st.dataframe(avg_scores, use_container_width=True)
                
                # Add breakdown of quality segments by channel
                st.subheader("Quality Segments by Channel")
                
                # Calculate segment distribution by channel
                segment_by_channel = filtered_df.groupby(['Marketing Source', 'Quality Segment']).size().unstack().fillna(0)
                
                # Calculate percentages
                segment_pct = segment_by_channel.div(segment_by_channel.sum(axis=1), axis=0) * 100
                segment_pct = segment_pct.round(1)
                
                # Create stacked bar chart
                fig = px.bar(
                    segment_pct.reset_index(),
                    x='Marketing Source',
                    y=segment_pct.columns.tolist(),
                    title="Quality Segments Distribution by Channel (%)",
                    labels={'value': 'Percentage', 'variable': 'Quality Segment'},
                    color_discrete_map={
                        'Low (0-25)': 'red',
                        'Medium (26-50)': 'orange',
                        'High (51-75)': 'lightgreen',
                        'Premium (76-100)': 'darkgreen'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with quality_tab3:
                if 'Lead created' in filtered_df.columns:
                    st.subheader("Quality Score Trends Over Time")
                    
                    # Create time-based aggregation
                    filtered_df['month_year'] = filtered_df['Lead created'].dt.to_period('M').astype(str)
                    
                    quality_trend = filtered_df.groupby('month_year')['Lead Quality Score'].agg(['mean', 'count']).reset_index()
                    quality_trend.columns = ['Month', 'Average Quality Score', 'Lead Count']
                    
                    # Create line chart with lead count bars
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add bars for lead count
                    fig.add_trace(
                        go.Bar(
                            x=quality_trend['Month'],
                            y=quality_trend['Lead Count'],
                            name="Lead Count",
                            marker_color='lightblue',
                            opacity=0.7
                        ),
                        secondary_y=False
                    )
                    
                    # Add line for quality score
                    fig.add_trace(
                        go.Scatter(
                            x=quality_trend['Month'],
                            y=quality_trend['Average Quality Score'],
                            name="Avg Quality Score",
                            line=dict(color='darkred', width=3),
                            mode='lines+markers'
                        ),
                        secondary_y=True
                    )
                    
                    fig.update_layout(
                        title_text="Lead Quality Score Trend Over Time",
                        xaxis_title="Month",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    fig.update_yaxes(
                        title_text="Lead Count",
                        secondary_y=False
                    )
                    fig.update_yaxes(
                        title_text="Average Quality Score",
                        secondary_y=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add day of week quality patterns
                    st.subheader("Quality Score by Day of Week")
                    
                    dow_quality = filtered_df.groupby('day_name')['Lead Quality Score'].mean().reindex([
                        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
                    ]).reset_index()
                    
                    fig = px.bar(
                        dow_quality,
                        x='day_name',
                        y='Lead Quality Score',
                        title="Average Quality Score by Day of Week",
                        color='Lead Quality Score',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Time-based analysis requires 'Lead created' data. This column is not available in the filtered dataset.")

        # Model Performance
        st.header("Machine Learning Model Competitive Analysis")
        st.markdown("""
        Our advanced machine learning analysis compares multiple algorithms to find the optimal model for predicting lead interest.
        This comprehensive comparison helps identify which model provides the best balance of accuracy, precision, and computational efficiency.

        **Models Analyzed**:
        - Random Forest: Robust ensemble method resistant to overfitting
        - XGBoost: Gradient boosting algorithm optimized for performance
        - LightGBM: Highly efficient gradient boosting framework

        **Evaluation Metrics**:
        - Accuracy: Overall prediction correctness
        - Precision: Accuracy of positive predictions (high-interest leads)
        - Recall: Ability to identify all positive leads
        - F1 Score: Harmonic mean of precision and recall
        - Training & Inference Time: Computational efficiency
        
        **Advanced Analytics**:
        - ROC & PR Curves: Visual performance evaluation
        - Confusion Matrices: Detailed classification results
        - Feature Importance: Key drivers of lead interest
        """)
        
        if 'Interest Level' in filtered_df.columns:
            try:
                # Prepare features
                feature_cols = ['Lead Owner', 'Marketing Source', 'Creation Source']
                if 'hour_of_day' in filtered_df.columns:
                    feature_cols.extend(['hour_of_day', 'day_of_week'])
                if 'What do you do currently ?' in filtered_df.columns:
                    feature_cols.append('What do you do currently ?')
                
                # Verify we have all required columns
                missing_cols = [col for col in feature_cols if col not in filtered_df.columns]
                if not missing_cols:  # Only proceed if we have all columns
                    # Prepare data
                    X = filtered_df[feature_cols].copy()
                    
                    # Label encode categorical variables
                    le_dict = {}
                    for col in X.select_dtypes(include=['object']):
                        le_dict[col] = LabelEncoder()
                        X[col] = le_dict[col].fit_transform(X[col].fillna('Unknown'))
                    
                    # Prepare target variable with better handling of interest levels
                    def map_interest_level(x):
                        if pd.isna(x):
                            return 0
                        x = str(x).lower()
                        # More lenient mapping to ensure better class distribution
                        if any(level in x for level in ['high', 'medium', 'interested']):
                            return 1
                        return 0

                    y = filtered_df['Interest Level'].apply(map_interest_level)
                    
                    # Ensure minimum sample requirements
                    if len(y) >= 10:
                        # Check class distribution
                        class_counts = y.value_counts()
                        pos_samples = class_counts.get(1, 0)
                        neg_samples = class_counts.get(0, 0)
                        
                        # Display class distribution
                        st.subheader("Class Distribution")
                        dist_df = pd.DataFrame({
                            'Interest Level': ['Low/Other', 'High/Medium'],
                            'Count': [neg_samples, pos_samples],
                            'Percentage': [
                                f"{(neg_samples/len(y)*100):.1f}%",
                                f"{(pos_samples/len(y)*100):.1f}%"
                            ]
                        })
                        
                        # Create a pie chart of class distribution
                        fig = px.pie(
                            dist_df,
                            values='Count',
                            names='Interest Level',
                            title="Interest Level Distribution",
                            color_discrete_sequence=['#ff9999', '#66b3ff']
                        )
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.dataframe(dist_df, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # More lenient class balance check
                        if pos_samples > 0 and neg_samples > 0:
                            # Calculate class weights
                            class_weight = neg_samples / pos_samples if pos_samples > 0 else 1.0
                            
                            # Add warning for severe class imbalance
                            if min(pos_samples, neg_samples) < 5:
                                st.warning("Warning: Very few samples in one of the classes. Model performance may be unreliable.")
                            
                            # Train-test split with stratification
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y
                            )
                            
                            # Feature columns for display
                            feature_cols_display = feature_cols.copy()
                            
                            # Run ML model comparison
                            ml_results = ml_model_comparison(
                                X_train, X_test, y_train, y_test, 
                                feature_cols_display, class_weight
                            )
                            
                            # Display results only if we have any
                            if ml_results['results']:
                                # Create metric comparison table
                                metrics_df = pd.DataFrame({
                                    "Model": list(ml_results['results'].keys()),
                                    "Accuracy": [f"{ml_results['results'][model]['accuracy']:.3f}" for model in ml_results['results']],
                                    "Precision": [f"{ml_results['results'][model]['precision']:.3f}" for model in ml_results['results']],
                                    "Recall": [f"{ml_results['results'][model]['recall']:.3f}" for model in ml_results['results']],
                                    "F1 Score": [f"{ml_results['results'][model]['f1']:.3f}" for model in ml_results['results']],
                                    "Training Time (s)": [f"{ml_results['results'][model]['train_time']:.3f}" for model in ml_results['results']],
                                    "Inference Time (s)": [f"{ml_results['results'][model]['inference_time']:.3f}" for model in ml_results['results']]
                                })
                                
                                # Find best model for each metric
                                best_model = {}
                                for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
                                    best_val = metrics_df[metric].astype(float).max()
                                    best_model[metric] = metrics_df[metrics_df[metric].astype(float) == best_val]['Model'].values[0]
                                
                                # Display model comparison results
                                st.subheader("Model Performance Comparison")
                                
                                # Create metrics visualization
                                metrics_compare = pd.DataFrame({
                                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                                    "Random Forest": [
                                        float(metrics_df[metrics_df["Model"] == "Random Forest"]["Accuracy"].values[0]),
                                        float(metrics_df[metrics_df["Model"] == "Random Forest"]["Precision"].values[0]),
                                        float(metrics_df[metrics_df["Model"] == "Random Forest"]["Recall"].values[0]),
                                        float(metrics_df[metrics_df["Model"] == "Random Forest"]["F1 Score"].values[0])
                                    ],
                                    "XGBoost": [
                                        float(metrics_df[metrics_df["Model"] == "XGBoost"]["Accuracy"].values[0]),
                                        float(metrics_df[metrics_df["Model"] == "XGBoost"]["Precision"].values[0]),
                                        float(metrics_df[metrics_df["Model"] == "XGBoost"]["Recall"].values[0]),
                                        float(metrics_df[metrics_df["Model"] == "XGBoost"]["F1 Score"].values[0])
                                    ],
                                    "LightGBM": [
                                        float(metrics_df[metrics_df["Model"] == "LightGBM"]["Accuracy"].values[0]),
                                        float(metrics_df[metrics_df["Model"] == "LightGBM"]["Precision"].values[0]),
                                        float(metrics_df[metrics_df["Model"] == "LightGBM"]["Recall"].values[0]),
                                        float(metrics_df[metrics_df["Model"] == "LightGBM"]["F1 Score"].values[0])
                                    ]
                                })
                                
                                # Create radar chart for model comparison
                                fig = go.Figure()
                                
                                for model in ["Random Forest", "XGBoost", "LightGBM"]:
                                    fig.add_trace(go.Scatterpolar(
                                        r=metrics_compare[model].values,
                                        theta=metrics_compare["Metric"].values,
                                        fill='toself',
                                        name=model
                                    ))
                                
                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 1]
                                        )
                                    ),
                                    title="Model Performance Radar Chart",
                                    showlegend=True
                                )
                                
                                # Display radar chart and metrics table
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)
                                    
                                    # Show best model for each metric
                                    st.subheader("Best Model by Metric")
                                    for metric, model in best_model.items():
                                        st.markdown(f"**{metric}**: {model}")
                                
                                # Show ROC Curves
                                st.subheader("ROC Curves")
                                fig = go.Figure()
                                for name in ml_results['results']:
                                    if ml_results['results'][name]["roc"] is not None:
                                        fig.add_trace(go.Scatter(
                                            x=ml_results['results'][name]["roc"]["fpr"],
                                            y=ml_results['results'][name]["roc"]["tpr"],
                                            name=f"{name} (AUC={ml_results['results'][name]['roc']['auc']:.3f})"
                                        ))
                                
                                # Add diagonal reference line
                                fig.add_trace(go.Scatter(
                                    x=[0, 1],
                                    y=[0, 1],
                                    mode='lines',
                                    name='Reference',
                                    line=dict(dash='dash', color='gray'),
                                    showlegend=True
                                ))
                                
                                fig.update_layout(
                                    title="ROC Curves",
                                    xaxis_title="False Positive Rate",
                                    yaxis_title="True Positive Rate",
                                    showlegend=True,
                                    legend=dict(
                                        yanchor="bottom",
                                        y=0.01,
                                        xanchor="right",
                                        x=0.99
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show confusion matrices
                                st.subheader("Confusion Matrices")
                                
                                # Create tabs for each model
                                cm_tabs = st.tabs(list(ml_results['confusion_matrices'].keys()))
                                
                                for i, name in enumerate(ml_results['confusion_matrices']):
                                    with cm_tabs[i]:
                                        cm = ml_results['confusion_matrices'][name]
                                        
                                        # Calculate percentages for better interpretability
                                        cm_sum = np.sum(cm, axis=1, keepdims=True)
                                        cm_perc = cm / cm_sum.astype(float) * 100
                                        
                                        # Create heatmap with custom text
                                        fig = px.imshow(
                                            cm,
                                            labels=dict(x="Predicted", y="Actual"),
                                            x=["Low Interest", "High Interest"],
                                            y=["Low Interest", "High Interest"],
                                            color_continuous_scale="Blues",
                                            title=f"Confusion Matrix - {name}"
                                        )
                                        
                                        # Add text annotations manually
                                        for i in range(cm.shape[0]):
                                            for j in range(cm.shape[1]):
                                                fig.add_annotation(
                                                    x=j,
                                                    y=i,
                                                    text=f"{cm[i, j]} ({cm_perc[i, j]:.1f}%)",
                                                    showarrow=False,
                                                    font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                                                )
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                # Feature Importance
                                st.subheader("Feature Importance Analysis")
                                st.markdown("""
                                Feature importance helps identify which variables have the strongest influence on predicting lead interest.
                                This analysis is crucial for:
                                - Understanding key drivers of lead quality
                                - Optimizing marketing focus
                                - Improving data collection strategies
                                """)
                                
                                # Create tabs for each model
                                if ml_results['feature_importance']:
                                    fi_tabs = st.tabs(list(ml_results['feature_importance'].keys()))
                                    for i, name in enumerate(ml_results['feature_importance']):
                                        with fi_tabs[i]:
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                importance = ml_results['feature_importance'][name]['direct']
                                                fig = px.bar(
                                                    x=importance.values,
                                                    y=importance.index,
                                                    orientation='h',
                                                    title=f"Feature Importance - {name}",
                                                    labels={'x': 'Importance', 'y': 'Feature'},
                                                    color=importance.values,
                                                    color_continuous_scale='Viridis'
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                            
                                            with col2:
                                                if 'permutation' in ml_results['feature_importance'][name]:
                                                    perm_importance = ml_results['feature_importance'][name]['permutation']
                                                    fig = px.bar(
                                                        x=perm_importance.values,
                                                        y=perm_importance.index,
                                                        orientation='h',
                                                        title=f"Permutation Importance - {name}",
                                                        labels={'x': 'Importance', 'y': 'Feature'},
                                                        color=perm_importance.values,
                                                        color_continuous_scale='Viridis'
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Feature importance data not available for any model.")
                        else:
                            st.error("Need samples from both classes (High/Medium and Low) for model training. Please adjust your filters.")
                    else:
                        st.error("Not enough data points for model training. Please adjust your filters.")
            
            except Exception as e:
                st.error(f"Error in model training and evaluation: {str(e)}")
                st.info("Please check if your dataset contains the required columns and proper data types.")
                st.write("Error details:", str(e))

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please ensure your dataset contains the required columns and is properly formatted.")
    st.write("Error details:", str(e)) 