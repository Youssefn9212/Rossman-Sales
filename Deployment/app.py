from flask import Flask, request, render_template, jsonify, session, redirect, url_for, Response
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from datetime import datetime
import warnings
import logging
import tensorflow as tf
from functools import wraps
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json

# Suppress scikit-learn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Initialize global variables
model_xgb = None
model_cnnlstm = None
scaler = None

# Load models and scalers
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# In-memory storage for predictions
predictions_history = []

# Login credentials (in memory)
CREDENTIALS = {
    'username': 'Nakhla',
    'password': 'Nakhla'
}

# Global cache for the analysis DataFrame
analysis_df_cache = None
analysis_df_error = None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == CREDENTIALS['username'] and password == CREDENTIALS['password']:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

def load_models():
    """Load all required models and scalers"""
    global model_xgb, model_cnnlstm, scaler
    try:
        print("\n=== Loading Models ===")
        
        # Load XGBoost model
        xgb_path = os.path.join(MODEL_DIR, 'model_xgb.json')
        print(f"Loading XGBoost model from: {xgb_path}")
        model_xgb = xgb.Booster()
        model_xgb.load_model(xgb_path)
        print("XGBoost model loaded successfully")
        
        # Load CNN-LSTM model
        cnn_path = os.path.join(MODEL_DIR, 'model_cnnlstm.keras')
        print(f"Loading CNN-LSTM model from: {cnn_path}")
        model_cnnlstm = load_model(cnn_path)
        model_cnnlstm.compile(optimizer='adam', loss='mse')
        print("CNN-LSTM model loaded successfully")
        print("Model summary:")
        model_cnnlstm.summary()
        
        # Load scaler
        scaler_path = os.path.join(MODEL_DIR, 'cnn_scaler.pkl')
        print(f"\nLoading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully")
        print(f"Scaler n_features_in_: {scaler.n_features_in_}")
        if hasattr(scaler, 'feature_names_in_'):
            print(f"Scaler feature names: {scaler.feature_names_in_.tolist()}")
        
        print("\n✅ All models loaded successfully")
        return True
    except Exception as e:
        print(f"\n❌ Error loading models: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        print(traceback.format_exc())
        return False

def encode_store_type(store_type):
    """
    Encode store type according to the mapping:
    'a' -> 0, 'b' -> 1, 'c' -> 2, 'd' -> 3
    """
    store_type_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    if isinstance(store_type, pd.Series):
        return store_type.astype(str).str.lower().map(store_type_map).fillna(0)
    return store_type_map.get(str(store_type).lower(), 0)

def encode_assortment(assortment):
    """
    Encode assortment according to the mapping:
    'a' (basic) -> 0, 'b' (extra) -> 1, 'c' (extended) -> 2
    """
    assortment_map = {'a': 0, 'b': 1, 'c': 2}
    if isinstance(assortment, pd.Series):
        return assortment.astype(str).str.lower().map(assortment_map).fillna(0)
    return assortment_map.get(str(assortment).lower(), 0)

def encode_state_holiday(state_holiday):
    """
    Encode state holiday according to the mapping:
    '0' (no holiday) -> 0, 'a' (public holiday) -> 1,
    'b' (Easter holiday) -> 2, 'c' (Christmas) -> 3
    """
    holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
    if isinstance(state_holiday, pd.Series):
        return state_holiday.astype(str).str.lower().map(holiday_map).fillna(0)
    return holiday_map.get(str(state_holiday).lower(), 0)

def encode_promo_interval(interval_code):
    """
    Encode promo interval according to the mapping:
    'Jan,Apr,Jul,Oct' -> 1
    'Feb,May,Aug,Nov' -> 2
    'Mar,Jun,Sept,Dec' -> 3
    No Interval -> 0
    """
    promo_map = {
        0: 0,  # No interval
        1: 2,  # Feb,May,Aug,Nov
        2: 3,  # Mar,Jun,Sept,Dec
        3: 1   # Jan,Apr,Jul,Oct
    }
    if isinstance(interval_code, pd.Series):
        return interval_code.map(lambda x: promo_map.get(x, 0))
    return promo_map.get(interval_code, 0)

def apply_feature_engineering(df):
    """Apply feature engineering to input data"""
    df = df.copy()
    
    # Convert date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate DateInt (days since 2013-01-01)
    df['DateInt'] = (df['Date'] - pd.to_datetime("2013-01-01")).dt.days
    
    # Basic date features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['IsWeekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    
    # Handle categorical encodings
    df['StoreType'] = encode_store_type(df.get('StoreType', pd.Series(['a'] * len(df))))
    df['Assortment'] = encode_assortment(df.get('Assortment', pd.Series(['a'] * len(df))))
    df['StateHoliday'] = encode_state_holiday(df.get('StateHoliday', pd.Series(['0'] * len(df))))
    
    # Handle competition features
    df['CompetitionDistance'] = df.get('CompetitionDistance', pd.Series([0] * len(df)))
    df['CompetitionOpenSinceMonth'] = df.get('CompetitionOpenSinceMonth', pd.Series([1] * len(df)))
    df['CompetitionOpenSinceYear'] = df.get('CompetitionOpenSinceYear', pd.Series([2010] * len(df)))
    
    # If CompetitionDistance is 0, set competition opening dates to 0
    df.loc[df['CompetitionDistance'] == 0, ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']] = 0
    
    # Calculate competition duration
    df['CompetitionOpenSince'] = (
        12 * (df['Year'] - df['CompetitionOpenSinceYear']) +
        (df['Month'] - df['CompetitionOpenSinceMonth'])
    ).clip(lower=0)
    
    # Handle promo features
    df['Promo'] = df.get('Promo', pd.Series([0] * len(df)))
    df['Promo2'] = df.get('Promo2', pd.Series([0] * len(df)))
    df['Promo2SinceWeek'] = df.get('Promo2SinceWeek', pd.Series([0] * len(df)))
    df['Promo2SinceYear'] = df.get('Promo2SinceYear', pd.Series([0] * len(df)))
    
    # Calculate Promo2 duration only if store participates in Promo2
    df['Promo2Since'] = 0  # Default value
    promo2_mask = df['Promo2'] == 1
    df.loc[promo2_mask, 'Promo2Since'] = (
        52 * (df.loc[promo2_mask, 'Year'] - df.loc[promo2_mask, 'Promo2SinceYear']) +
        (df.loc[promo2_mask, 'WeekOfYear'] - df.loc[promo2_mask, 'Promo2SinceWeek'])
    ).clip(lower=0)
    
    # Handle PromoInterval
    # Convert string interval to numeric code
    promo_interval_map = {
        'Jan,Apr,Jul,Oct': 3,
        'Feb,May,Aug,Nov': 1,
        'Mar,Jun,Sept,Dec': 2,
        '': 0  # Empty string maps to 0
    }
    
    # Get PromoInterval with default as empty string
    promo_interval_series = df.get('PromoInterval', pd.Series([''] * len(df)))
    # Map string intervals to numeric codes
    numeric_intervals = promo_interval_series.map(lambda x: promo_interval_map.get(x, 0))
    # Encode the numeric intervals
    df['PromoInterval'] = encode_promo_interval(numeric_intervals)
    df['IsPromo2Month'] = 0
    
    # Only calculate IsPromo2Month if store participates in Promo2
    if df['Promo2'].iloc[0] == 1:
        # Map months to their respective promo intervals
        promo_months = {
            1: [1, 4, 7, 10],    # Jan,Apr,Jul,Oct
            2: [2, 5, 8, 11],    # Feb,May,Aug,Nov
            3: [3, 6, 9, 12]     # Mar,Jun,Sept,Dec
        }
        
        for interval_code, months in promo_months.items():
            df.loc[(df['PromoInterval'] == interval_code) & (df['Month'].isin(months)), 'IsPromo2Month'] = 1
    
    # Handle Open status
    df['Open'] = df.get('Open', pd.Series([1] * len(df)))  # Default to open
    if pd.isna(df['Open'].iloc[0]):
        df['Open'] = 0 if df['DayOfWeek'].iloc[0] == 7 else 1
    
    # Convert SchoolHoliday to int
    df['SchoolHoliday'] = df['SchoolHoliday'].astype(int)
    
    selected_columns = [
        'Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear',
        'PromoInterval', 'DateInt', 'Year', 'Month', 'Day', 'DayOfYear', 'WeekOfYear',
        'IsWeekend', 'Quarter', 'IsMonthStart', 'IsMonthEnd', 'CompetitionOpenSince',
        'Promo2Since', 'IsPromo2Month'
    ]
    
    return df[[col for col in selected_columns if col in df.columns]]

def prepare_cnn_input(df, window=14):
    """Prepare input data for CNN-LSTM model"""
    print("\n=== prepare_cnn_input ===")
    print("Input DataFrame shape:", df.shape)
    print("Input DataFrame columns:", df.columns.tolist())
    
    # Drop Date column and fill missing values
    df = df.drop(columns=['Date'])
    df = df.bfill()
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("Numeric columns:", numeric_cols.tolist())
    print("Number of numeric features:", len(numeric_cols))
    
    X_scaled = df[numeric_cols].copy()
    print("X_scaled shape:", X_scaled.shape)
    
    # Skip feature scaling since we don't have the feature scaler
    # The model should be robust enough to handle unscaled features
    X_scaled_array = X_scaled.values
    
    # For a single observation, repeat it to create the sequence
    if len(X_scaled_array) == 1:
        X_scaled_array = np.repeat(X_scaled_array, window, axis=0)
        X_seq = np.array([X_scaled_array])  # Shape will be (1, window, n_features)
    else:
        X_seq = [X_scaled_array[i-window:i] for i in range(window, len(X_scaled_array))]
        X_seq = np.array(X_seq)
    
    print("Final sequence shape:", X_seq.shape)
    return X_seq

def predict_sales(input_data):
    """Make sales prediction using ensemble model"""
    try:
        print("\n=== predict_sales ===")
        print("Input data:", input_data)
        
        # Convert input data to DataFrame and ensure proper column names
        df = pd.DataFrame([input_data])
        print("\nInput DataFrame shape:", df.shape)
        print("Input DataFrame columns:", df.columns.tolist())
        
        # Apply feature engineering
        df_fe = apply_feature_engineering(df)
        print("\nFeature engineered DataFrame shape:", df_fe.shape)
        print("Feature engineered columns:", df_fe.columns.tolist())
        
        # Prepare input for XGBoost
        test_xgb = df_fe.drop(columns=['Date'])
        print("\nXGBoost input shape:", test_xgb.shape)
        
        # Prepare input for CNN-LSTM
        X_cnn = prepare_cnn_input(df_fe)
        X_xgb = test_xgb.iloc[-len(X_cnn):].copy()
        
        # Align XGBoost input columns
        print("\nXGBoost feature names:", model_xgb.feature_names)
        for col in model_xgb.feature_names:
            if col not in X_xgb.columns:
                print(f"Adding missing column: {col}")
                X_xgb[col] = 0
        X_xgb = X_xgb[model_xgb.feature_names]
        print("Final XGBoost input shape:", X_xgb.shape)
        
        # Make predictions
        xgb_preds = model_xgb.predict(xgb.DMatrix(X_xgb))
        print("\nXGBoost prediction shape:", xgb_preds.shape)
        
        cnn_scaled_preds = model_cnnlstm.predict(X_cnn).flatten()
        print("CNN-LSTM prediction shape:", cnn_scaled_preds.shape)
        
        # Use scaler only for the final prediction (it's the target scaler)
        cnn_preds = scaler.inverse_transform(cnn_scaled_preds.reshape(-1, 1)).flatten()
        print("CNN-LSTM prediction after inverse transform shape:", cnn_preds.shape)
        
        # Ensemble predictions
        ensemble_preds = 0.5 * xgb_preds + 0.5 * cnn_preds
        print("\nEnsemble prediction:", ensemble_preds[-1])
        
        # Convert to actual sales value (reverse log1p transformation)
        final_sales = float(np.expm1(ensemble_preds[-1]))
        print("Final sales prediction:", final_sales)
        return final_sales
        
    except Exception as e:
        print("\nError in predict_sales:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise Exception(f"Error in prediction: {str(e)}")

@app.route('/')
@login_required
def home():
    """Render the home page"""
    return render_template('index.html', username=session.get('username'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """API endpoint for sales prediction"""
    try:
        # Get input data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        print("Received data:", data)  # Debug log
        
        # Define all possible input fields with their types and defaults
        required_fields = {
            'store': {'type': int, 'required': True},
            'date': {'type': str, 'required': True},
            'dayofweek': {'type': int, 'required': True},
            'open': {'type': int, 'required': True},
            'promo': {'type': int, 'required': True},
            'stateholiday': {'type': str, 'required': True},
            'schoolholiday': {'type': int, 'required': True},
            'storetype': {'type': str, 'required': True},
            'assortment': {'type': str, 'required': True},
            'competitiondistance': {'type': float, 'required': False, 'default': 0.0},
            'competitionopensincemonth': {'type': float, 'required': False, 'default': 0.0},
            'competitionopensinceyear': {'type': float, 'required': False, 'default': 0.0},
            'promo2': {'type': int, 'required': False, 'default': 0},
            'promo2sinceweek': {'type': float, 'required': False, 'default': 0.0},
            'promo2sinceyear': {'type': float, 'required': False, 'default': 0.0},
            'promointerval': {'type': str, 'required': False, 'default': ''}
        }
        
        # Validate and process input data
        processed_data = {}
        for field, config in required_fields.items():
            value = data.get(field.lower())
            
            # Check if required field is missing
            if config['required'] and value is None:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            
            # Use default if value is None
            if value is None:
                value = config['default']
            
            # Convert to correct type
            try:
                if config['type'] == str:
                    processed_data[field] = str(value)
                elif config['type'] == int:
                    processed_data[field] = int(float(value))  # Handle string numbers
                elif config['type'] == float:
                    processed_data[field] = float(value)
            except (ValueError, TypeError) as e:
                return jsonify({
                    'error': f'Invalid value for field {field}: {value}. Expected type: {config["type"].__name__}. Error: {str(e)}'
                }), 400
        
        print("Processed data:", processed_data)  # Debug log
        
        # Map the processed data to model format
        model_input = {
            'Store': processed_data['store'],
            'Date': processed_data['date'],
            'DayOfWeek': processed_data['dayofweek'],
            'Open': processed_data['open'],
            'Promo': processed_data['promo'],
            'StateHoliday': processed_data['stateholiday'],
            'SchoolHoliday': processed_data['schoolholiday'],
            'StoreType': processed_data['storetype'].lower(),  # Ensure lowercase
            'Assortment': processed_data['assortment'].lower(),  # Ensure lowercase
            'CompetitionDistance': processed_data['competitiondistance'],
            'CompetitionOpenSinceMonth': processed_data['competitionopensincemonth'],
            'CompetitionOpenSinceYear': processed_data['competitionopensinceyear'],
            'Promo2': processed_data['promo2'],
            'Promo2SinceWeek': processed_data['promo2sinceweek'],
            'Promo2SinceYear': processed_data['promo2sinceyear'],
            'PromoInterval': processed_data['promointerval']
        }
        
        print("Model input:", model_input)  # Debug log
        
        # Make prediction
        sales_prediction = predict_sales(model_input)
        
        # Store prediction in history
        prediction_record = {
            'store': processed_data['store'],
            'date': processed_data['date'],
            'predicted_sales': round(sales_prediction),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        predictions_history.append(prediction_record)
        
        return jsonify(prediction_record)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("Error details:", error_details)  # Debug log
        return jsonify({
            'error': str(e),
            'details': error_details
        }), 400

@app.route('/history')
@login_required
def history():
    """View prediction history"""
    return render_template('history.html', predictions=predictions_history)

def load_and_process_data():
    """Load and process the training data, with caching and error handling"""
    global analysis_df_cache, analysis_df_error
    if analysis_df_cache is not None:
        return analysis_df_cache
    try:
        # Use relative path instead of absolute path
        data_path = os.path.join(os.path.dirname(__file__), '..', 'Processed Data', 'train_final1.csv')
        print(f"[INFO] Loading data from: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
            
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        analysis_df_cache = df
        analysis_df_error = None
        print("[INFO] Analysis data loaded successfully.")
        print(f"[INFO] Data shape: {df.shape}")
        print(f"[INFO] Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        analysis_df_error = str(e)
        print(f"[ERROR] Failed to load analysis data: {e}")
        import traceback
        print(traceback.format_exc())
        return None

@app.route('/analysis')
@login_required
def analysis():
    """Sales Analysis page"""
    try:
        tab = request.args.get('tab', 'overview')
        df = load_and_process_data()
        
        if analysis_df_error:
            return render_template('analysis.html', tab=tab, error=analysis_df_error)
        if df is None:
            return render_template('analysis.html', tab=tab, error='Data could not be loaded.')
        
        print(f"[INFO] Rendering analysis tab: {tab}")
        print(f"[INFO] DataFrame shape: {df.shape}")
        
        if tab == 'overview':
            try:
                # Calculate overview metrics
                total_sales = df['Sales'].sum()
                avg_daily_sales = df.groupby('Date')['Sales'].sum().mean()
                unique_stores = df['Store'].nunique()
                promo_coverage = (df['Promo'].sum() / len(df)) * 100
                
                # Daily sales plot
                daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
                sales_plot = px.line(daily_sales, x='Date', y='Sales', 
                                   title='Total Daily Sales Over Time')
                sales_plot.update_layout(
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(size=12)
                )
                
                return render_template('analysis.html', 
                                     tab=tab,
                                     total_sales=total_sales,
                                     avg_daily_sales=avg_daily_sales,
                                     unique_stores=unique_stores,
                                     promo_coverage=promo_coverage,
                                     sales_plot=sales_plot.to_html(full_html=False))
            except Exception as e:
                print(f"[ERROR] Error in overview tab: {e}")
                return render_template('analysis.html', tab=tab, error=f'Error generating overview: {str(e)}')
        
        elif tab == 'time_trends':
            try:
                # Daily trends
                daily_avg = df.groupby('DayOfWeek')['Sales'].mean().reset_index()
                daily_plot = px.bar(daily_avg, x='DayOfWeek', y='Sales',
                                  title='Average Sales by Day of Week')
                daily_plot.update_layout(height=400)
                
                # Monthly trends
                monthly_avg = df.groupby('Month')['Sales'].mean().reset_index()
                monthly_plot = px.bar(monthly_avg, x='Month', y='Sales',
                                    title='Average Sales by Month')
                monthly_plot.update_layout(height=400)
                
                # Rolling average
                rolling_avg = df.groupby('Date')['Sales'].sum().rolling(7).mean().reset_index()
                rolling_plot = px.line(rolling_avg, x='Date', y='Sales',
                                     title='7-Day Rolling Average Sales')
                rolling_plot.update_layout(height=400)
                
                return render_template('analysis.html', 
                                     tab=tab,
                                     daily_plot=daily_plot.to_html(full_html=False),
                                     monthly_plot=monthly_plot.to_html(full_html=False),
                                     rolling_plot=rolling_plot.to_html(full_html=False))
            except Exception as e:
                print(f"[ERROR] Error in time_trends tab: {e}")
                return render_template('analysis.html', tab=tab, error=f'Error generating time trends: {str(e)}')
        
        elif tab == 'store_performance':
            # Sales by store type
            store_type_plot = px.box(df, x='StoreType', y='Sales',
                                   title='Sales Distribution by Store Type')
            store_type_plot.update_layout(height=400)
            
            # Top 10 stores
            top_stores = df.groupby('Store')['Sales'].mean().sort_values(ascending=False).head(10)
            top_stores_plot = px.bar(top_stores, title='Top 10 Stores by Average Sales')
            top_stores_plot.update_layout(height=400)
            
            # Top 5 stores daily sales
            top_5_stores = top_stores.head().index
            top_5_daily = df[df['Store'].isin(top_5_stores)].pivot(index='Date', columns='Store', values='Sales')
            top_5_plot = px.line(top_5_daily, title='Daily Sales for Top 5 Stores')
            top_5_plot.update_layout(height=400)
            
            return render_template('analysis.html',
                                 tab=tab,
                                 store_type_plot=store_type_plot.to_html(full_html=False),
                                 top_stores_plot=top_stores_plot.to_html(full_html=False),
                                 top_5_plot=top_5_plot.to_html(full_html=False))
        
        elif tab == 'promotions':
            # Promo vs non-promo
            promo_plot = px.box(df, x='Promo', y='Sales',
                              title='Sales Distribution: Promo vs Non-Promo Days')
            promo_plot.update_layout(height=400)
            
            # Sales by holiday type
            holiday_avg = df.groupby('StateHoliday')['Sales'].mean().reset_index()
            holiday_plot = px.bar(holiday_avg, x='StateHoliday', y='Sales',
                                title='Average Sales by State Holiday Type')
            holiday_plot.update_layout(height=400)
            
            # Promo2 comparison
            promo2_avg = df.groupby('Promo2')['Sales'].mean().reset_index()
            promo2_plot = px.bar(promo2_avg, x='Promo2', y='Sales',
                               title='Average Sales: Promo2 vs Non-Promo2 Stores')
            promo2_plot.update_layout(height=400)
            
            return render_template('analysis.html',
                                 tab=tab,
                                 promo_plot=promo_plot.to_html(full_html=False),
                                 holiday_plot=holiday_plot.to_html(full_html=False),
                                 promo2_plot=promo2_plot.to_html(full_html=False))
        
        elif tab == 'seasonality':
            stores = sorted(df['Store'].unique())
            selected_store = request.args.get('store', stores[0])
            
            store_data = df[df['Store'] == int(selected_store)]
            
            # Use pre-computed decomposition features
            trend_plot = px.line(store_data, x='Date', y='StoreTrend',
                               title=f'Store {selected_store} - Trend')
            trend_plot.update_layout(height=400)
            
            seasonal_cols = [col for col in df.columns if 'seasonal' in col.lower()]
            seasonal_data = store_data[['Date'] + seasonal_cols]
            seasonal_plot = px.line(seasonal_data, x='Date', y=seasonal_cols,
                                  title=f'Store {selected_store} - Seasonal Components')
            seasonal_plot.update_layout(height=400)
            
            residual_cols = [col for col in df.columns if 'resid' in col.lower()]
            residual_data = store_data[['Date'] + residual_cols]
            residual_plot = px.line(residual_data, x='Date', y=residual_cols,
                                  title=f'Store {selected_store} - Residuals')
            residual_plot.update_layout(height=400)
            
            return render_template('analysis.html',
                                 tab=tab,
                                 stores=stores,
                                 selected_store=selected_store,
                                 trend_plot=trend_plot.to_html(full_html=False),
                                 seasonal_plot=seasonal_plot.to_html(full_html=False),
                                 residual_plot=residual_plot.to_html(full_html=False))
        
        elif tab == 'export':
            stores = sorted(df['Store'].unique())
            return render_template('analysis.html',
                                 tab=tab,
                                 stores=stores)
        
        return render_template('analysis.html', tab=tab)
    except Exception as e:
        print(f"[ERROR] Error in analysis: {e}")
        return render_template('analysis.html', tab=tab, error=f'Error generating analysis: {str(e)}')

# Load models when the application starts
if not load_models():
    raise RuntimeError("Failed to load models. Application cannot start.")

if __name__ == '__main__':
    app.run(debug=True) 