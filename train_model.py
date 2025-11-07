"""
Train and save the student score prediction model
Run this BEFORE starting the web app
"""

from src.data_preparation import load_data, prepare_features, split_and_scale_data
from src.model_training import train_model, evaluate_model, save_model, get_feature_importance

print("="*60)
print("TRAINING STUDENT SCORE PREDICTION MODEL")
print("="*60)

print("\n[1/4] Loading data...")
df = load_data('data/student_data.csv')

if df is None:
    print("❌ Error: Could not load data. Please run generate_data.py first!")
    exit()

print("\n[2/4] Preparing features...")
X, y = prepare_features(df)

print("\n[3/4] Splitting and scaling data...")
X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)

print("\n[4/4] Training model...")
model = train_model(X_train_scaled, y_train)

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)
metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)

print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)
importance_df = get_feature_importance(model, X.columns)
print(importance_df)

print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)
save_model(model, scaler)

print("\n" + "="*60)
print("✓ TRAINING COMPLETE!")
print("="*60)
print("\nYou can now run the web app with:")
print("  streamlit run app.py")
