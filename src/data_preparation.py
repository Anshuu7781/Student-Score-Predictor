"""
Data Preparation Module
Handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath='data/student_data.csv'):
    """Load student data from CSV file"""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully: {df.shape[0]} records")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File {filepath} not found")
        return None

def prepare_features(df):
    """Separate features and target variable"""
    X = df[['study_hours', 'previous_score', 'attendance', 'sleep_hours', 'extracurricular']]
    y = df['final_score']
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split data into train/test sets and scale features"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Data split: {len(X_train)} training, {len(X_test)} testing samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_data_statistics(df):
    """Get statistical summary of the dataset"""
    return df.describe()

def get_correlation_matrix(df):
    """Get correlation matrix"""
    return df.corr()
