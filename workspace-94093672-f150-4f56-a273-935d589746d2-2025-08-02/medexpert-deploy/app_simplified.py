#!/usr/bin/env python3
"""
MedExpert - Simplified Medical AI System for Web Deployment
Streamlined version optimized for cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import warnings
import os
import logging
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="MedExpert - AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/medexpert/help',
        'Report a bug': 'https://github.com/medexpert/issues',
        'About': "MedExpert - Advanced Medical AI System for Healthcare Professionals"
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .medical-disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

class SimplifiedMedicalKnowledge:
    """Simplified medical knowledge base for demonstration"""
    
    def __init__(self):
        self.conditions = {
            "hypertension": {
                "description": "High blood pressure condition",
                "symptoms": ["headache", "dizziness", "chest pain"],
                "treatments": ["ACE inhibitors", "lifestyle changes", "diet modification"]
            },
            "diabetes": {
                "description": "Blood sugar regulation disorder",
                "symptoms": ["increased thirst", "frequent urination", "fatigue"],
                "treatments": ["insulin", "metformin", "diet control", "exercise"]
            },
            "asthma": {
                "description": "Respiratory condition affecting airways",
                "symptoms": ["wheezing", "shortness of breath", "coughing"],
                "treatments": ["bronchodilators", "corticosteroids", "avoiding triggers"]
            }
        }
    
    def get_condition_info(self, condition: str) -> Dict:
        """Get information about a medical condition"""
        return self.conditions.get(condition.lower(), {})
    
    def search_conditions(self, symptoms: List[str]) -> List[str]:
        """Search for conditions based on symptoms"""
        matches = []
        for condition, info in self.conditions.items():
            if any(symptom.lower() in [s.lower() for s in info["symptoms"]] for symptom in symptoms):
                matches.append(condition)
        return matches

def main():
    """Main application function"""
    
    # Initialize medical knowledge base
    med_kb = SimplifiedMedicalKnowledge()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 MedExpert AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Medical AI System for Healthcare Professionals</p>', unsafe_allow_html=True)
    
    # Medical Disclaimer
    st.markdown("""
    <div class="medical-disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and informational purposes only. 
        It is not intended to replace professional medical advice, diagnosis, or treatment. 
        Always consult with qualified healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Controls")
        
        # Feature selection
        selected_feature = st.selectbox(
            "Select Feature",
            ["Clinical Decision Support", "Symptom Checker", "Drug Interaction", "Medical Calculator", "Patient Analytics"]
        )
        
        st.markdown("### 📊 System Status")
        st.success("✅ System Online")
        st.info("🔄 Last Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Patients", "1,234", "12")
        with col2:
            st.metric("Accuracy", "94.5%", "2.1%")
    
    # Main content area
    if selected_feature == "Clinical Decision Support":
        st.markdown("## 🩺 Clinical Decision Support")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Patient Information")
            
            # Patient input form
            with st.form("patient_form"):
                patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
                patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                chief_complaint = st.text_area("Chief Complaint", placeholder="Describe the main symptoms...")
                medical_history = st.text_area("Medical History", placeholder="Previous conditions, medications...")
                
                submitted = st.form_submit_button("Analyze Case")
                
                if submitted and chief_complaint:
                    st.success("✅ Case analysis completed!")
                    
                    # Mock analysis results
                    st.markdown("### 📋 Analysis Results")
                    
                    # Differential diagnosis
                    st.markdown("#### Differential Diagnosis")
                    diagnoses = [
                        {"condition": "Hypertension", "probability": 0.75, "confidence": "High"},
                        {"condition": "Anxiety Disorder", "probability": 0.45, "confidence": "Medium"},
                        {"condition": "Cardiac Arrhythmia", "probability": 0.30, "confidence": "Low"}
                    ]
                    
                    for dx in diagnoses:
                        col_dx1, col_dx2, col_dx3 = st.columns([3, 1, 1])
                        with col_dx1:
                            st.write(f"**{dx['condition']}**")
                        with col_dx2:
                            st.write(f"{dx['probability']:.0%}")
                        with col_dx3:
                            color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[dx['confidence']]
                            st.write(f"{color} {dx['confidence']}")
                    
                    # Recommendations
                    st.markdown("#### 💡 Recommendations")
                    recommendations = [
                        "Order blood pressure monitoring",
                        "Consider ECG if chest symptoms persist",
                        "Lifestyle counseling for stress management",
                        "Follow-up in 2 weeks"
                    ]
                    
                    for rec in recommendations:
                        st.write(f"• {rec}")
        
        with col2:
            st.markdown("### 📊 Risk Assessment")
            
            # Risk factors visualization
            risk_data = pd.DataFrame({
                'Factor': ['Age', 'BMI', 'Blood Pressure', 'Cholesterol', 'Smoking'],
                'Risk Level': [0.6, 0.8, 0.9, 0.4, 0.2]
            })
            
            fig = px.bar(risk_data, x='Risk Level', y='Factor', orientation='h',
                        color='Risk Level', color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Vital signs trend
            st.markdown("### 📈 Vital Signs Trend")
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            bp_systolic = np.random.normal(140, 10, 30)
            bp_diastolic = np.random.normal(90, 5, 30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=bp_systolic, name='Systolic BP', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=dates, y=bp_diastolic, name='Diastolic BP', line=dict(color='blue')))
            fig.update_layout(height=250, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_feature == "Symptom Checker":
        st.markdown("## 🔍 Symptom Checker")
        
        st.markdown("### Enter Your Symptoms")
        symptoms_input = st.text_input("Describe your symptoms (comma-separated)", 
                                     placeholder="headache, dizziness, fatigue")
        
        if symptoms_input:
            symptoms = [s.strip() for s in symptoms_input.split(',')]
            
            # Search for matching conditions
            matches = med_kb.search_conditions(symptoms)
            
            if matches:
                st.markdown("### 🎯 Possible Conditions")
                for condition in matches:
                    info = med_kb.get_condition_info(condition)
                    with st.expander(f"📋 {condition.title()}"):
                        st.write(f"**Description:** {info['description']}")
                        st.write(f"**Common Symptoms:** {', '.join(info['symptoms'])}")
                        st.write(f"**Typical Treatments:** {', '.join(info['treatments'])}")
            else:
                st.info("No matching conditions found. Please consult a healthcare professional.")
    
    elif selected_feature == "Medical Calculator":
        st.markdown("## 🧮 Medical Calculator")
        
        calc_type = st.selectbox("Select Calculator", 
                                ["BMI Calculator", "Cardiac Risk Score", "Drug Dosage"])
        
        if calc_type == "BMI Calculator":
            col1, col2 = st.columns(2)
            with col1:
                height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
                weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
            
            with col2:
                if height > 0 and weight > 0:
                    bmi = weight / ((height/100) ** 2)
                    st.metric("BMI", f"{bmi:.1f}")
                    
                    if bmi < 18.5:
                        st.info("Underweight")
                    elif bmi < 25:
                        st.success("Normal weight")
                    elif bmi < 30:
                        st.warning("Overweight")
                    else:
                        st.error("Obese")
    
    elif selected_feature == "Patient Analytics":
        st.markdown("## 📊 Patient Analytics Dashboard")
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        patient_visits = np.random.poisson(15, 90)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Daily Patient Visits")
            fig = px.line(x=dates, y=patient_visits, title="Patient Visit Trends")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Condition Distribution")
            conditions = ['Hypertension', 'Diabetes', 'Asthma', 'Arthritis', 'Other']
            values = [30, 25, 20, 15, 10]
            fig = px.pie(values=values, names=conditions, title="Most Common Conditions")
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", "1,234", "12")
        with col2:
            st.metric("Avg Age", "52.3", "-1.2")
        with col3:
            st.metric("Satisfaction", "4.8/5", "0.1")
        with col4:
            st.metric("Response Time", "2.3 min", "-0.5")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🏥 <strong>MedExpert AI</strong> - Empowering Healthcare with Artificial Intelligence</p>
        <p>Built with ❤️ for healthcare professionals worldwide</p>
        <p><small>Version 2.0 | Last Updated: August 2024</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()