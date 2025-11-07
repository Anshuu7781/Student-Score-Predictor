import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data for 100 students
n_students = 100

data = {
    'study_hours': np.random.randint(1, 10, n_students),
    'previous_score': np.random.randint(40, 100, n_students),
    'attendance': np.random.randint(60, 100, n_students),
    'sleep_hours': np.random.randint(4, 10, n_students),
    'extracurricular': np.random.randint(0, 5, n_students)
}

# Create DataFrame
df = pd.DataFrame(data)

# Create target variable (final_score) with some correlation to features
df['final_score'] = (
    df['study_hours'] * 3 +
    df['previous_score'] * 0.4 +
    df['attendance'] * 0.3 +
    df['sleep_hours'] * 1.5 -
    df['extracurricular'] * 0.5 +
    np.random.randint(-10, 10, n_students)
).clip(0, 100)

# Save the dataset
os.makedirs('data', exist_ok=True)
df.to_csv('data/student_data.csv', index=False)

print("✓ Dataset created and saved to data/student_data.csv")
print(f"✓ Total records: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())