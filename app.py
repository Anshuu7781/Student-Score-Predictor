"""
Student Score Prediction Web Application
Built with Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import from src
from src.data_preparation import load_data, get_data_statistics, get_correlation_matrix
from src.model_training import load_model, get_feature_importance
from src.prediction import predict_score, get_performance_category, get_recommendations

# Page configuration
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_resources():
    model, scaler = load_model()
    df = load_data()
    return model, scaler, df

model, scaler, df = load_resources()

# Header
st.title("🎓 Student Score Prediction System")
st.markdown("### Predict student performance using Machine Learning")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Navigation")
    page = st.radio("Go to", ["🏠 Home", "🔮 Predict Score", "📈 Data Analytics",])
    
    st.markdown("---")
    st.markdown("### 📚 Model Info")
    st.info("**Model:** Linear Regression\n\n**Accuracy:** ~78% R² Score")

# HOME PAGE
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to Student Score Predictor! 👋")
        st.markdown("""
        This application uses **Machine Learning** to predict student final exam scores based on various factors:
        
        - 📖 **Study Hours** per day
        - 📝 **Previous Exam Score**
        - 🎯 **Attendance Percentage**
        - 😴 **Sleep Hours** per day
        - 🎨 **Extracurricular Activities**
        
        ### How it works:
        1. Navigate to **Predict Score** page
        2. Enter student information
        3. Get instant score prediction with recommendations
        
        ### Features:
        ✅ Real-time predictions  
        ✅ Performance analysis  
        ✅ Personalized recommendations  
        ✅ Data visualizations  
        """)
    
    with col2:
        st.image("https://img.icons8.com/clouds/400/student-male.png", width=300)
    
    st.markdown("---")
    
    # Quick Stats
    st.header("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Average Score", f"{df['final_score'].mean():.1f}")
    with col3:
        st.metric("Highest Score", f"{df['final_score'].max():.1f}")
    with col4:
        st.metric("Lowest Score", f"{df['final_score'].min():.1f}")

# PREDICTION PAGE
elif page == "🔮 Predict Score":
    st.header("🔮 Predict Student Score")
    st.markdown("Enter student information to get score prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        study_hours = st.slider("📖 Study Hours per Day", 1, 10, 5, help="Average daily study hours")
        previous_score = st.slider("📝 Previous Exam Score", 0, 100, 70, help="Score from last exam")
        attendance = st.slider("🎯 Attendance Percentage", 0, 100, 80, help="Overall attendance rate")
    
    with col2:
        sleep_hours = st.slider("😴 Sleep Hours per Day", 4, 10, 7, help="Average daily sleep hours")
        extracurricular = st.slider("🎨 Extracurricular Activities", 0, 5, 2, help="Number of activities")
    
    st.markdown("---")
    
    if st.button("🎯 Predict Score", use_container_width=True):
        # Make prediction
        predicted_score = predict_score(
            model, scaler, study_hours, previous_score, 
            attendance, sleep_hours, extracurricular
        )
        
        category, emoji, color = get_performance_category(predicted_score)
        
        # Display prediction
        st.markdown(f"""
        <div style='background-color: {color}; padding: 30px; border-radius: 15px; text-align: center;'>
            <h1 style='color: white; margin: 0;'>{emoji} Predicted Score: {predicted_score:.2f}</h1>
            <h3 style='color: white; margin: 10px 0 0 0;'>Performance Level: {category}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("💡 Personalized Recommendations")
        recommendations = get_recommendations(
            study_hours, previous_score, attendance, sleep_hours, extracurricular
        )
        
        for rec in recommendations:
            st.info(rec)
        
        # Comparison Chart
        st.markdown("---")
        st.subheader("📊 How You Compare")
        
        comparison_data = pd.DataFrame({
            'Metric': ['Study Hours', 'Previous Score', 'Attendance', 'Sleep Hours'],
            'Your Values': [study_hours, previous_score, attendance, sleep_hours],
            'Dataset Average': [
                df['study_hours'].mean(),
                df['previous_score'].mean(),
                df['attendance'].mean(),
                df['sleep_hours'].mean()
            ]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Your Values', x=comparison_data['Metric'], 
                             y=comparison_data['Your Values'], marker_color='#4CAF50'))
        fig.add_trace(go.Bar(name='Dataset Average', x=comparison_data['Metric'], 
                             y=comparison_data['Dataset Average'], marker_color='#2196F3'))
        
        fig.update_layout(barmode='group', height=400, title="Your Input vs Dataset Average")
        st.plotly_chart(fig, use_container_width=True)

# DATA ANALYTICS PAGE
elif page == "📈 Data Analytics":
    st.header("📈 Data Analytics & Insights")
    
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "🎯 Feature Importance"])
    
    with tab1:
        st.subheader("Score Distribution")
        fig = px.histogram(df, x='final_score', nbins=20, 
                          title='Distribution of Final Scores',
                          color_discrete_sequence=['#4CAF50'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='study_hours', y='final_score',
                           title='Study Hours vs Final Score',
                           trendline='ols',
                           color_discrete_sequence=['#2196F3'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='previous_score', y='final_score',
                           title='Previous Score vs Final Score',
                           trendline='ols',
                           color_discrete_sequence=['#FF9800'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Matrix")
        corr_matrix = get_correlation_matrix(df)
        
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       color_continuous_scale='RdBu_r',
                       title='Feature Correlation Heatmap')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(" **Insight:** Features with higher correlation (closer to 1) have stronger influence on final scores")
    
    with tab3:
        st.subheader("Feature Importance")
        importance_df = get_feature_importance(model, df.columns[:-1])
        
        fig = px.bar(importance_df, x='Coefficient', y='Feature',
                    orientation='h',
                    title='Feature Coefficients (Importance)',
                    color='Coefficient',
                    color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(importance_df, use_container_width=True)
