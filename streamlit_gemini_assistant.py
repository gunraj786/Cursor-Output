import os
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
import json
import time
import base64
import concurrent.futures
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import uuid
from datetime import datetime
import numpy as np

# Optional Langfuse Integration
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

# ------------------ Page Setup ------------------
st.set_page_config(
    page_title="🚀 Gemini AI Assistant", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Enhanced CSS ------------------
def load_css():
    st.markdown("""
    <style>
    .main .block-container {
        max-width: none !important;
        padding: 1rem 2rem !important;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin: 1.5rem 0;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .response-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ------------------ Langfuse Integration ------------------
class LangfuseTracker:
    def __init__(self):
        self.langfuse = None
        self.is_configured = False
        
    def configure(self, public_key, secret_key, host="https://cloud.langfuse.com"):
        if LANGFUSE_AVAILABLE:
            try:
                os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
                os.environ["LANGFUSE_SECRET_KEY"] = secret_key
                os.environ["LANGFUSE_HOST"] = host
                self.langfuse = Langfuse()
                self.is_configured = True
                return True
            except Exception as e:
                st.error(f"Langfuse configuration failed: {e}")
                return False
        return False
    
    def run_prompt_with_logging(self, prompt_text: str, trace_name: str, model_name="gemini-1.5-flash"):
        start_time = time.time()
        
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt_text)
            result_text = response.text
            
            # Log to Langfuse if configured
            if self.is_configured and self.langfuse:
                try:
                    trace = self.langfuse.trace(
                        name=trace_name,
                        input={"prompt": prompt_text},
                        output={"response": result_text}
                    )
                    trace_id = trace.id
                except:
                    trace_id = None
            else:
                trace_id = None
                
        except Exception as e:
            result_text = f"Error: {e}"
            trace_id = None
        
        end_time = time.time()
        
        return {
            "prompt": prompt_text,
            "output": result_text,
            "length": len(result_text),
            "input_tokens": len(prompt_text.split()),
            "output_tokens": len(result_text.split()) if 'result_text' in locals() else 0,
            "time_taken": round(end_time - start_time, 2),
            "trace_id": trace_id,
            "timestamp": datetime.now()
        }

# ------------------ Helper Functions ------------------
def create_download_link(text, filename):
    """Create a download link for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}" style="color: white; text-decoration: none; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); padding: 8px 16px; border-radius: 8px; font-weight: 600;">📥 Download</a>'
    return href

def display_hero_section():
    st.markdown('<h1 class="main-header">🚀 Gemini AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Advanced AI-Powered Workspace</h4>
        <p>Experience Google's Gemini with parallel processing, analytics, and professional tools.</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------ Initialize Session State ------------------
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'langfuse_tracker' not in st.session_state:
    st.session_state.langfuse_tracker = LangfuseTracker()
if 'prompt_results' not in st.session_state:
    st.session_state.prompt_results = []

# ------------------ Load CSS ------------------
load_css()

# ------------------ Hero Section ------------------
display_hero_section()

# ------------------ Sidebar Configuration ------------------
with st.sidebar:
    st.markdown("## 🔑 Configuration")
    
    # Google API Key
    api_key = st.text_input("🔐 Google API Key", type="password", 
                           help="Get your API key from Google AI Studio")
    
    # Langfuse Configuration (Optional)
    if LANGFUSE_AVAILABLE:
        st.markdown("### 📊 Langfuse Tracking (Optional)")
        langfuse_enabled = st.checkbox("Enable Langfuse", value=False)
        
        if langfuse_enabled:
            langfuse_public = st.text_input("🔑 Public Key", type="password")
            langfuse_secret = st.text_input("🔐 Secret Key", type="password")
            
            if langfuse_public and langfuse_secret:
                if st.button("🔗 Connect"):
                    if st.session_state.langfuse_tracker.configure(langfuse_public, langfuse_secret):
                        st.success("✅ Connected!")
                    else:
                        st.error("❌ Failed to connect")
    
    # API Key Configuration
    if api_key:
        if not st.session_state.api_configured:
            try:
                os.environ["GOOGLE_API_KEY"] = api_key
                genai.configure(api_key=api_key)
                
                # Test the API key
                model = genai.GenerativeModel("gemini-1.5-flash")
                test_response = model.generate_content("Hello")
                
                st.session_state.api_configured = True
                st.success("✅ API Configured!")
                
            except Exception as e:
                st.error(f"❌ API Error: {str(e)}")
                st.session_state.api_configured = False
        else:
            st.success("✅ API Active")
    
    # Session Stats
    if st.session_state.prompt_results:
        st.markdown("### 📈 Session Stats")
        total_prompts = len(st.session_state.prompt_results)
        avg_time = np.mean([r['time_taken'] for r in st.session_state.prompt_results])
        total_tokens = sum([r['input_tokens'] + r['output_tokens'] for r in st.session_state.prompt_results])
        
        st.metric("Prompts", total_prompts)
        st.metric("Avg Time", f"{avg_time:.2f}s")
        st.metric("Tokens", total_tokens)

if st.session_state.api_configured:
    # ------------------ Navigation Menu ------------------
    selected = option_menu(
        menu_title="🎛️ AI Tools",
        options=[
            "🧠 Chat Assistant", 
            "⚡ Batch Processing",
            "🖼️ Vision Analysis",
            "📊 Analytics", 
            "🔗 LangChain Tools"
        ],
        icons=["chat", "lightning", "image", "graph-up", "link"],
        menu_icon="robot",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#667eea", "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "2px",
                "padding": "10px",
                "border-radius": "10px",
                "background-color": "rgba(255,255,255,0.08)"
            },
            "nav-link-selected": {"background-color": "#667eea"},
        }
    )

    # ------------------ Chat Assistant ------------------
    if selected == "🧠 Chat Assistant":
        st.markdown("## 💬 AI Chat Assistant")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            
            # Prompt templates
            template_choice = st.selectbox(
                "🎯 Prompt Template",
                [
                    "Custom",
                    "Business Analysis",
                    "Creative Writing",
                    "Technical Help", 
                    "Problem Solving",
                    "Code Review"
                ]
            )
            
            template_prompts = {
                "Business Analysis": "Analyze the following business scenario: ",
                "Creative Writing": "Write a creative story about: ",
                "Technical Help": "Explain the following technical concept: ",
                "Problem Solving": "Help me solve this problem: ",
                "Code Review": "Review this code for best practices: "
            }
            
            if template_choice != "Custom":
                prompt = st.text_area(
                    "✏️ Your Prompt:", 
                    value=template_prompts.get(template_choice, ""),
                    height=150
                )
            else:
                prompt = st.text_area("✏️ Your Prompt:", height=150)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                generate_btn = st.button("🚀 Generate", use_container_width=True)
            with col_btn2:
                clear_btn = st.button("🗑️ Clear", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ⚙️ Settings")
            temperature = st.slider("🌡️ Creativity", 0.0, 1.0, 0.7, 0.1)
            
            if prompt:
                word_count = len(prompt.split())
                st.metric("Words", word_count)
                st.metric("Characters", len(prompt))
        
        if generate_btn and prompt:
            with st.spinner("🔄 Generating response..."):
                try:
                    result = st.session_state.langfuse_tracker.run_prompt_with_logging(
                        prompt, f"chat-{len(st.session_state.prompt_results)}"
                    )
                    
                    st.session_state.prompt_results.append(result)
                    
                    st.markdown("### 🎯 Response:")
                    st.markdown(f'<div class="response-container">{result["output"]}</div>', 
                              unsafe_allow_html=True)
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Time", f"{result['time_taken']}s")
                    col_b.metric("Length", result['length'])
                    col_c.metric("Tokens", f"{result['input_tokens'] + result['output_tokens']}")
                    
                    # Download
                    st.markdown(
                        create_download_link(result["output"], "response.txt"), 
                        unsafe_allow_html=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        if clear_btn:
            st.rerun()

    # ------------------ Batch Processing ------------------
    elif selected == "⚡ Batch Processing":
        st.markdown("## ⚡ Batch Processing")
        
        st.markdown("""
        <div class="info-box">
            <h4>🚀 Process Multiple Prompts</h4>
            <p>Run multiple prompts simultaneously for efficiency.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input methods
        input_method = st.radio("Input Method:", ["Manual", "File Upload"])
        
        prompts_to_process = []
        
        if input_method == "Manual":
            prompt_text = st.text_area(
                "Enter prompts (one per line):",
                height=200,
                placeholder="What is AI?\nExplain quantum computing\nWrite a short poem"
            )
            prompts_to_process = [p.strip() for p in prompt_text.split('\n') if p.strip()]
            
        else:
            uploaded_file = st.file_uploader("Upload text file", type=['txt'])
            if uploaded_file:
                content = str(uploaded_file.read(), "utf-8")
                prompts_to_process = [p.strip() for p in content.split('\n') if p.strip()]
                st.success(f"📁 Loaded {len(prompts_to_process)} prompts")
        
        if prompts_to_process:
            st.markdown(f"### 📊 Ready to process {len(prompts_to_process)} prompts")
            
            max_workers = st.slider("🔧 Concurrent Workers", 1, 5, min(3, len(prompts_to_process)))
            
            if st.button("🚀 Start Processing", use_container_width=True):
                start_time = time.time()
                results = []
                progress_bar = st.progress(0)
                
                def process_prompt(prompt_data):
                    index, prompt = prompt_data
                    try:
                        return st.session_state.langfuse_tracker.run_prompt_with_logging(
                            prompt, f"batch-{index}"
                        )
                    except Exception as e:
                        return {
                            'prompt': prompt,
                            'output': f"Error: {str(e)}",
                            'error': True
                        }
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_prompt = {
                        executor.submit(process_prompt, (i, prompt)): (i, prompt) 
                        for i, prompt in enumerate(prompts_to_process)
                    }
                    
                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_prompt):
                        result = future.result()
                        results.append(result)
                        if not result.get('error', False):
                            st.session_state.prompt_results.append(result)
                        
                        completed += 1
                        progress_bar.progress(completed / len(prompts_to_process))
                
                total_time = time.time() - start_time
                
                # Results
                st.markdown("### 🎯 Results")
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Total Time", f"{total_time:.2f}s")
                col_s2.metric("Avg per Prompt", f"{total_time/len(results):.2f}s")
                col_s3.metric("Success", f"{len([r for r in results if not r.get('error', False)])}/{len(results)}")
                
                # Show results
                for i, result in enumerate(results):
                    if not result.get('error', False):
                        with st.expander(f"📝 Prompt {i+1}: {result['prompt'][:50]}..."):
                            st.markdown(result['output'])
                            st.markdown(
                                create_download_link(result['output'], f"result_{i+1}.txt"),
                                unsafe_allow_html=True
                            )

    # ------------------ Vision Analysis ------------------
    elif selected == "🖼️ Vision Analysis":
        st.markdown("## 🖼️ Vision Analysis")
        
        st.markdown("""
        <div class="info-box">
            <h4>👁️ Image Understanding</h4>
            <p>Upload images for AI-powered analysis and description.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_image = st.file_uploader(
                "📷 Upload Image",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp']
            )
            
            analysis_prompt = st.text_area(
                "✏️ What do you want to know about the image?",
                placeholder="Describe this image, identify objects, analyze composition...",
                height=100
            )
            
            analysis_type = st.selectbox(
                "🎯 Analysis Type",
                ["General Description", "Detailed Analysis", "Object Detection", "Text Extraction"]
            )
        
        with col2:
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                st.markdown("### 📊 Image Info")
                st.metric("Size", f"{image.size[0]}×{image.size[1]}")
                st.metric("Format", image.format)
                st.metric("Mode", image.mode)
        
        if st.button("🔍 Analyze Image", use_container_width=True):
            if uploaded_image and analysis_prompt:
                with st.spinner("👁️ Analyzing image..."):
                    try:
                        image = Image.open(uploaded_image)
                        
                        analysis_prompts = {
                            "General Description": "Describe this image in detail.",
                            "Detailed Analysis": "Provide a comprehensive analysis including composition, colors, objects, and setting.",
                            "Object Detection": "Identify and list all objects visible in this image.",
                            "Text Extraction": "Extract and transcribe all text visible in this image."
                        }
                        
                        full_prompt = f"{analysis_prompts[analysis_type]}\n\nUser request: {analysis_prompt}"
                        
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content([full_prompt, image])
                        
                        result_data = {
                            "prompt": full_prompt,
                            "output": response.text,
                            "length": len(response.text),
                            "input_tokens": len(full_prompt.split()),
                            "output_tokens": len(response.text.split()),
                            "time_taken": 0,
                            "trace_id": None,
                            "timestamp": datetime.now()
                        }
                        
                        st.session_state.prompt_results.append(result_data)
                        
                        st.markdown("### 🎯 Analysis Result:")
                        st.markdown(f'<div class="response-container">{response.text}</div>', 
                                  unsafe_allow_html=True)
                        
                        st.markdown(
                            create_download_link(response.text, "image_analysis.txt"),
                            unsafe_allow_html=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Analysis error: {str(e)}")
            else:
                st.warning("⚠️ Please upload an image and provide analysis instructions.")

    # ------------------ Analytics ------------------
    elif selected == "📊 Analytics":
        st.markdown("## 📊 Analytics Dashboard")
        
        if not st.session_state.prompt_results:
            st.markdown("""
            <div class="info-box">
                <h4>📈 No Data Available</h4>
                <p>Use other tools first to generate analytics data!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            df = pd.DataFrame(st.session_state.prompt_results)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['total_tokens'] = df['input_tokens'] + df['output_tokens']
            
            # Overview metrics
            st.markdown("### 📈 Session Overview")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            col_m1.metric("Total Prompts", len(df))
            col_m2.metric("Avg Response Time", f"{df['time_taken'].mean():.2f}s")
            col_m3.metric("Total Tokens", f"{df['total_tokens'].sum():,}")
            col_m4.metric("Avg Length", f"{df['length'].mean():.0f}")
            
            # Visualizations
            st.markdown("### 📊 Performance Charts")
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                # Response time chart
                fig_time = px.line(
                    df.reset_index(), x='index', y='time_taken',
                    title='Response Time Trend',
                    labels={'index': 'Prompt Number', 'time_taken': 'Time (seconds)'}
                )
                fig_time.update_traces(line_color='#667eea')
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col_c2:
                # Token usage
                fig_tokens = px.scatter(
                    df, x='input_tokens', y='output_tokens',
                    size='time_taken', color='length',
                    title='Token Usage Pattern',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_tokens, use_container_width=True)
            
            # Data table
            st.markdown("### 📋 Detailed Results")
            display_df = df[['timestamp', 'prompt', 'time_taken', 'length', 'total_tokens']].copy()
            display_df['prompt'] = display_df['prompt'].str[:100] + '...'
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    'timestamp': 'Time',
                    'prompt': st.column_config.TextColumn('Prompt', width='large'),
                    'time_taken': st.column_config.NumberColumn('Time (s)', format='%.2f'),
                    'length': 'Length',
                    'total_tokens': 'Tokens'
                }
            )
            
            # Export
            if st.button("📊 Export CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )

    # ------------------ LangChain Tools ------------------
    elif selected == "🔗 LangChain Tools":
        st.markdown("## 🔗 LangChain Integration")
        
        st.markdown("""
        <div class="info-box">
            <h4>⚡ Advanced Prompt Engineering</h4>
            <p>Use LangChain's powerful templating and chain capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.7
            )
            
            template_category = st.selectbox(
                "Template Category:",
                ["Business Analysis", "Content Creation", "Educational", "Custom"]
            )
            
            langchain_templates = {
                "Business Analysis": """
Analyze the following business scenario:

Company: {company_name}
Industry: {industry}  
Challenge: {challenge}

Provide:
1. Situation Analysis
2. Key Challenges
3. Recommendations
4. Implementation Plan

Analysis:""",
                
                "Content Creation": """
Create {content_type} content:

Topic: {topic}
Audience: {audience}
Tone: {tone}
Length: {length}

Requirements:
- Engaging and relevant
- Clear structure
- Professional quality

Content:""",
                
                "Educational": """
Explain {topic} for {level} level:

Learning Objectives: {objectives}
Key Concepts: {concepts}
Examples Needed: {examples}

Create comprehensive educational content.

Explanation:"""
            }
            
            if template_category in langchain_templates:
                template_text = langchain_templates[template_category]
                st.code(template_text, language='text')
                
                # Extract variables
                import re
                variables = re.findall(r'{(\w+)}', template_text)
                
                st.markdown("### 📝 Template Variables")
                variable_values = {}
                
                cols = st.columns(2)
                for i, var in enumerate(variables):
                    with cols[i % 2]:
                        variable_values[var] = st.text_input(
                            f"{var.replace('_', ' ').title()}:",
                            key=f"lc_var_{var}"
                        )
                
                if st.button("🚀 Generate with LangChain"):
                    if all(variable_values.values()):
                        with st.spinner("🔄 Processing with LangChain..."):
                            try:
                                prompt_template = ChatPromptTemplate.from_template(template_text)
                                chain = prompt_template | llm | StrOutputParser()
                                
                                start_time = time.time()
                                result = chain.invoke(variable_values)
                                end_time = time.time()
                                
                                result_data = {
                                    "prompt": prompt_template.format(**variable_values),
                                    "output": result,
                                    "length": len(result),
                                    "input_tokens": len(prompt_template.format(**variable_values).split()),
                                    "output_tokens": len(result.split()),
                                    "time_taken": round(end_time - start_time, 2),
                                    "trace_id": None,
                                    "timestamp": datetime.now()
                                }
                                
                                st.session_state.prompt_results.append(result_data)
                                
                                st.markdown("### 🎯 LangChain Result:")
                                st.markdown(f'<div class="response-container">{result}</div>', 
                                          unsafe_allow_html=True)
                                
                                col_r1, col_r2, col_r3 = st.columns(3)
                                col_r1.metric("Time", f"{result_data['time_taken']}s")
                                col_r2.metric("Length", result_data['length'])
                                col_r3.metric("Tokens", f"{result_data['input_tokens'] + result_data['output_tokens']}")
                                
                                st.markdown(
                                    create_download_link(result, "langchain_result.txt"),
                                    unsafe_allow_html=True
                                )
                                
                            except Exception as e:
                                st.error(f"❌ LangChain error: {str(e)}")
                    else:
                        st.warning("⚠️ Please fill all template variables.")
            
            else:  # Custom template
                custom_template = st.text_area(
                    "Custom Template (use {variable_name} for variables):",
                    height=200,
                    placeholder="Hello {name}, welcome to {company}! Today we'll discuss {topic}."
                )
                
                if custom_template:
                    import re
                    variables = re.findall(r'{(\w+)}', custom_template)
                    
                    if variables:
                        st.markdown("### 📝 Variables Found")
                        variable_values = {}
                        
                        for var in variables:
                            variable_values[var] = st.text_input(f"{var}:", key=f"custom_{var}")
                        
                        if st.button("🚀 Generate Custom"):
                            if all(variable_values.values()):
                                with st.spinner("🔄 Processing..."):
                                    try:
                                        prompt_template = ChatPromptTemplate.from_template(custom_template)
                                        chain = prompt_template | llm | StrOutputParser()
                                        
                                        result = chain.invoke(variable_values)
                                        
                                        st.markdown("### 🎯 Custom Result:")
                                        st.markdown(f'<div class="response-container">{result}</div>', 
                                                  unsafe_allow_html=True)
                                        
                                    except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")
                    else:
                        st.info("💡 Add variables using {variable_name} syntax")
        
        except Exception as e:
            st.error(f"❌ LangChain setup error: {str(e)}")
            st.info("💡 Make sure your Google API key is configured correctly.")

else:
    st.warning("⚠️ Please configure your Google API key in the sidebar to get started.")
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. **Enter Key**: Paste your API key in the sidebar
    3. **Start Creating**: Use any of the AI tools available!
    
    ### ✨ Features Available:
    - 🧠 **Chat Assistant**: Interactive AI conversations
    - ⚡ **Batch Processing**: Process multiple prompts simultaneously  
    - 🖼️ **Vision Analysis**: Analyze and describe images
    - 📊 **Analytics**: Track your usage and performance
    - 🔗 **LangChain Tools**: Advanced prompt engineering
    """)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 Gemini AI Assistant | Built with Streamlit & Google AI</p>
</div>
""", unsafe_allow_html=True)