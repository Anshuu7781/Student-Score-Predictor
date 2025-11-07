"""
Prediction Module
Handles score predictions for new students
"""

import numpy as np

def predict_score(model, scaler, study_hours, previous_score, attendance, sleep_hours, extracurricular):
    """
    Predict student's final score based on input features
    
    Parameters:
    -----------
    model : trained ML model
    scaler : fitted StandardScaler
    study_hours : float (1-10)
    previous_score : float (0-100)
    attendance : float (0-100)
    sleep_hours : float (4-10)
    extracurricular : float (0-5)
    
    Returns:
    --------
    predicted_score : float
    """
    input_data = np.array([[study_hours, previous_score, attendance, sleep_hours, extracurricular]])
    
    input_scaled = scaler.transform(input_data)
    
    predicted_score = model.predict(input_scaled)[0]
    
    predicted_score = max(0, min(100, predicted_score))
    
    return predicted_score

def get_performance_category(score):
    """Categorize performance based on predicted score"""
    if score >= 90:
        return "Outstanding", "🌟", "#28a745"
    elif score >= 80:
        return "Excellent", "🎉", "#20c997"
    elif score >= 70:
        return "Good", "👍", "#17a2b8"
    elif score >= 60:
        return "Average", "📚", "#ffc107"
    elif score >= 50:
        return "Below Average", "⚠️", "#fd7e14"
    else:
        return "Needs Improvement", "🚨", "#dc3545"

def get_recommendations(study_hours, previous_score, attendance, sleep_hours, extracurricular):
    """Provide personalized recommendations based on input"""
    recommendations = []
    
    if study_hours < 4:
        recommendations.append("📖 Increase study hours to at least 4-5 hours per day")
    
    if attendance < 75:
        recommendations.append("🎯 Improve attendance - aim for at least 80%")
    
    if sleep_hours < 6:
        recommendations.append("😴 Get more sleep - 7-8 hours is optimal for learning")
    
    if sleep_hours > 9:
        recommendations.append("⏰ Balance sleep time - too much sleep can reduce productivity")
    
    if extracurricular > 3:
        recommendations.append("⚖️ Balance extracurricular activities with study time")
    
    if previous_score < 60:
        recommendations.append("💪 Focus on strengthening fundamentals from previous courses")
    
    if len(recommendations) == 0:
        recommendations.append("✨ You're on the right track! Keep up the good work!")
    
    return recommendations
