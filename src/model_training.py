"""
Model Training Module
Handles model training and evaluation
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
import joblib
import os

def train_model(X_train, y_train):
    """Train Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✓ Model trained successfully")
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance on train and test sets"""
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Testing metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    metrics = {
        'train': {'r2': train_r2, 'rmse': train_rmse, 'mae': train_mae},
        'test': {'r2': test_r2, 'rmse': test_rmse, 'mae': test_mae}
    }
    
    print(f"\n=== Model Performance ===")
    print(f"Training R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}")
    print(f"Testing R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")
    
    return metrics

def save_model(model, scaler, model_path='models/model.pkl', scaler_path='models/scaler.pkl'):
    """Save trained model and scaler"""
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Scaler saved to {scaler_path}")

def load_model(model_path='models/model.pkl', scaler_path='models/scaler.pkl'):
    """Load trained model and scaler"""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("✓ Model and scaler loaded successfully")
        return model, scaler
    except FileNotFoundError:
        print("❌ Error: Model files not found. Please train the model first.")
        return None, None

def get_feature_importance(model, feature_names):
    """Get feature importance from trained model"""
    import pandas as pd
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=False)
    return importance_df