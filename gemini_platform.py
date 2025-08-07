import os
import streamlit as st
from streamlit_option_menu import option_menu
import google.generativeai as genai
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import uuid
from datetime import datetime
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Optional imports
try:
    from streamlit_lottie import st_lottie
    import requests
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="🚀 Advanced Gemini AI Platform", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data Classes
@dataclass
class PromptResult:
    prompt: str
    output: str
    technique: str
    length: int
    time_taken: float
    timestamp: datetime
    trace_id: Optional[str] = None

@dataclass
class PromptAnalysis:
    clarity_score: float
    specificity_score: float
    context_score: float
    structure_score: float
    completeness_score: float
    overall_score: float
    suggestions: List[str]
    improved_prompt: str

# Enhanced CSS
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
    .main .block-container {
        max-width: none !important;
        padding: 1rem 2rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 4s ease-in-out infinite;
    }
    
    .hero-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: float 3s ease-in-out infinite;
    }
    
    .technique-card {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.4s ease;
    }
    
    .technique-card:hover {
        transform: translateY(-8px);
    }
    
    .response-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Load Lottie Animation
@st.cache_data
def load_lottie_url(url: str):
    if not LOTTIE_AVAILABLE:
        return None
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# Initialize Session State
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'prompt_results' not in st.session_state:
    st.session_state.prompt_results = []
if 'prompt_analyses' not in st.session_state:
    st.session_state.prompt_analyses = []

# Load CSS
load_css()

# Load animations
ai_animation = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

# Hero Section
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if ai_animation and LOTTIE_AVAILABLE:
        st_lottie(ai_animation, height=200, key="hero_left")

with col2:
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🚀 Advanced Gemini AI Platform</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Professional AI Platform</h3>
        <p>Advanced prompting techniques • Real-time analytics • Professional evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    if ai_animation and LOTTIE_AVAILABLE:
        st_lottie(ai_animation, height=200, key="hero_right")

# Sidebar Configuration
with st.sidebar:
    st.markdown("## 🔑 Configuration")
    
    api_key = st.text_input("🔐 Google API Key", type="password")
    
    if api_key:
        if not st.session_state.api_configured:
            try:
                os.environ["GOOGLE_API_KEY"] = api_key
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                test = model.generate_content("Hello")
                st.session_state.api_configured = True
                st.success("✅ Connected!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.success("🚀 Active")

if st.session_state.api_configured:
    # Navigation Menu
    selected = option_menu(
        menu_title="🎛️ AI Platform",
        options=["🎯 Prompting", "📊 Analyzer", "📈 Analytics"],
        icons=["target", "graph-up", "bar-chart"],
        default_index=0,
        orientation="horizontal"
    )

    if selected == "🎯 Prompting":
        st.markdown("## 🎯 Advanced Prompting")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="technique-card">', unsafe_allow_html=True)
            
            technique = st.selectbox("Select Technique", [
                "Zero-shot", "Chain of Thought", "Role-based"
            ])
            
            base_prompt = st.text_area("Enter prompt:", height=100)
            
            if technique == "Role-based":
                role = st.selectbox("Role", ["business_analyst", "data_scientist", "teacher"])
            
            if st.button("🚀 Execute", use_container_width=True):
                if base_prompt:
                    with st.spinner("Processing..."):
                        try:
                            if technique == "Zero-shot":
                                final_prompt = f"Task: {base_prompt}\n\nProvide a comprehensive response:"
                            elif technique == "Chain of Thought":
                                final_prompt = f"Task: {base_prompt}\n\nLet's think step by step:\n1. Understanding\n2. Analysis\n3. Solution"
                            elif technique == "Role-based":
                                final_prompt = f"You are an expert {role}.\n\n{base_prompt}\n\nYour expert response:"
                            
                            start_time = time.time()
                            model = genai.GenerativeModel("gemini-1.5-flash")
                            response = model.generate_content(final_prompt)
                            time_taken = round(time.time() - start_time, 2)
                            
                            result = PromptResult(
                                prompt=final_prompt,
                                output=response.text,
                                technique=technique,
                                length=len(response.text),
                                time_taken=time_taken,
                                timestamp=datetime.now()
                            )
                            st.session_state.prompt_results.append(result)
                            
                            st.markdown("### 🎯 Results:")
                            st.markdown(f'<div class="response-container">{result.output}</div>', unsafe_allow_html=True)
                            
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Time", f"{result.time_taken}s")
                            col_b.metric("Length", result.length)
                            col_c.metric("Words", len(result.output.split()))
                            
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                else:
                    st.warning("⚠️ Enter a prompt.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎛️ Guide")
            st.info("Select a technique and enter your prompt for enhanced AI responses.")

    elif selected == "📊 Analyzer":
        st.markdown("## 📊 Prompt Analyzer")
        
        prompt_to_analyze = st.text_area("Enter prompt to analyze:", height=150)
        
        if st.button("🔍 Analyze", use_container_width=True):
            if prompt_to_analyze:
                # Simple analysis
                clarity_score = min(10.0, max(1.0, 5.0 + len(prompt_to_analyze.split()) * 0.05))
                specificity_score = min(10.0, max(1.0, 5.0 + (2.0 if any(char.isdigit() for char in prompt_to_analyze) else 0)))
                context_score = min(10.0, max(1.0, 5.0 + (3.0 if len(prompt_to_analyze) > 50 else 0)))
                structure_score = min(10.0, max(1.0, 5.0 + (2.0 if '?' in prompt_to_analyze else 0)))
                completeness_score = min(10.0, max(1.0, 5.0 + (1.0 if len(prompt_to_analyze.split()) > 10 else 0)))
                
                overall_score = (clarity_score + specificity_score + context_score + structure_score + completeness_score) / 5
                
                col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
                col_s1.metric("Clarity", f"{clarity_score:.1f}")
                col_s2.metric("Specificity", f"{specificity_score:.1f}")
                col_s3.metric("Context", f"{context_score:.1f}")
                col_s4.metric("Structure", f"{structure_score:.1f}")
                col_s5.metric("Complete", f"{completeness_score:.1f}")
                
                color = "green" if overall_score >= 8 else "orange" if overall_score >= 6 else "red"
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: {color}33; border-radius: 15px;">
                    <h3 style="color: {color};">Overall: {overall_score:.1f}/10</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Radar chart
                categories = ['Clarity', 'Specificity', 'Context', 'Structure', 'Completeness']
                scores = [clarity_score, specificity_score, context_score, structure_score, completeness_score]
                
                fig = go.Figure(data=go.Scatterpolar(r=scores, theta=categories, fill='toself'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("⚠️ Enter a prompt.")

    elif selected == "📈 Analytics":
        st.markdown("## �� Analytics")
        
        if not st.session_state.prompt_results:
            st.info("No data available. Use prompting techniques to generate data!")
        else:
            data = []
            for result in st.session_state.prompt_results:
                data.append({
                    'timestamp': result.timestamp,
                    'technique': result.technique,
                    'time_taken': result.time_taken,
                    'length': result.length
                })
            
            df = pd.DataFrame(data)
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Total Prompts", len(df))
            col_m2.metric("Avg Time", f"{df['time_taken'].mean():.2f}s")
            col_m3.metric("Total Length", f"{df['length'].sum():,}")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                technique_counts = df['technique'].value_counts()
                fig_pie = px.pie(values=technique_counts.values, names=technique_counts.index, title='Technique Usage')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                fig_line = px.line(df, x='timestamp', y='time_taken', color='technique', title='Response Times')
                st.plotly_chart(fig_line, use_container_width=True)
            
            st.dataframe(df, use_container_width=True)

else:
    st.warning("⚠️ Please configure your Google API key to access the platform.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 20px;'>
    <h4>🚀 Advanced Gemini AI Platform</h4>
    <p><strong>Professional AI Prompting & Analytics Solution</strong></p>
    <p><em>Streamlit • Google AI • Plotly • Lottie</em></p>
</div>
""", unsafe_allow_html=True)
