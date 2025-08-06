import os
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import google.generativeai as genai
import requests
import time
import base64
import concurrent.futures
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import uuid
from datetime import datetime
import numpy as np
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Langfuse Integration
LANGFUSE_AVAILABLE = False
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    pass

# Page Setup
st.set_page_config(
    page_title="🚀 Advanced Gemini AI Assistant", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
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
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient 3s ease-in-out infinite;
    }
    
    .technique-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
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
    
    .warning-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .langfuse-trace {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
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
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    """, unsafe_allow_html=True)

# Data Classes
@dataclass
class PromptResult:
    prompt: str
    output: str
    technique: str
    length: int
    input_tokens: int
    output_tokens: int
    time_taken: float
    trace_id: Optional[str]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

# Langfuse Tracker
class LangfuseTracker:
    def __init__(self):
        self.langfuse = None
        self.is_configured = False
        self.session_id = str(uuid.uuid4())
        
    def configure(self, public_key: str, secret_key: str, host: str = "https://cloud.langfuse.com"):
        if not LANGFUSE_AVAILABLE:
            return False
            
        try:
            os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
            os.environ["LANGFUSE_SECRET_KEY"] = secret_key
            os.environ["LANGFUSE_HOST"] = host
            
            self.langfuse = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
            self.langfuse.auth_check()
            self.is_configured = True
            return True
        except:
            return False
    
    def run_prompt_with_tracing(self, prompt_text: str, technique: str, **kwargs):
        start_time = time.time()
        trace_id = None
        
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt_text)
            result_text = response.text
            
            end_time = time.time()
            time_taken = round(end_time - start_time, 2)
            
            input_tokens = len(prompt_text.split())
            output_tokens = len(result_text.split())
            
            if self.is_configured and self.langfuse:
                try:
                    trace = self.langfuse.trace(
                        name=f"{technique}_execution",
                        session_id=self.session_id,
                        input={"prompt": prompt_text, "technique": technique},
                        output={"response": result_text},
                        metadata={"time_taken": time_taken, "input_tokens": input_tokens, "output_tokens": output_tokens, **kwargs}
                    )
                    trace_id = trace.id
                except:
                    pass
            
            return PromptResult(
                prompt=prompt_text,
                output=result_text,
                technique=technique,
                length=len(result_text),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                time_taken=time_taken,
                trace_id=trace_id,
                timestamp=datetime.now(),
                metadata=kwargs
            )
            
        except Exception as e:
            return PromptResult(
                prompt=prompt_text,
                output=f"Error: {str(e)}",
                technique=technique,
                length=0,
                input_tokens=len(prompt_text.split()),
                output_tokens=0,
                time_taken=round(time.time() - start_time, 2),
                trace_id=None,
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

# Prompting Techniques
class PromptingTechniques:
    @staticmethod
    def zero_shot(base_prompt: str) -> str:
        return f"""Task: {base_prompt}

Provide a comprehensive response based on your knowledge."""

    @staticmethod
    def few_shot(base_prompt: str, examples: List[Dict[str, str]]) -> str:
        examples_text = "\n\n".join([f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}" for i, ex in enumerate(examples)])
        return f"""Task: {base_prompt}

Here are examples:

{examples_text}

Now complete the task following the same pattern:"""

    @staticmethod
    def chain_of_thought(base_prompt: str) -> str:
        return f"""Task: {base_prompt}

Let's think through this step-by-step:
1. First, understand what's being asked
2. Break down into smaller parts
3. Work through each part systematically
4. Verify the reasoning
5. Provide the complete answer

Step by step:"""

    @staticmethod
    def react_prompting(base_prompt: str, max_iterations: int = 3) -> str:
        return f"""Task: {base_prompt}

Use ReAct framework with {max_iterations} iterations:

ITERATION 1:
Thought: [Reasoning about current state]
Action: [What to do next]
Observation: [What you discover]

ITERATION 2:
Thought: [Continue reasoning]
Action: [Next action]
Observation: [New findings]

ITERATION 3:
Thought: [Final reasoning]
Action: [Final action]
Observation: [Final findings]

ANSWER: [Complete solution]"""

    @staticmethod
    def role_based_prompting(base_prompt: str, role: str) -> str:
        roles = {
            "business_analyst": "You are a senior business analyst with 10+ years experience.",
            "data_scientist": "You are an expert data scientist with deep ML knowledge.",
            "teacher": "You are an experienced educator skilled at explaining complex topics.",
            "consultant": "You are a management consultant helping solve business problems."
        }
        role_desc = roles.get(role, f"You are an expert {role}.")
        return f"""{role_desc}

{base_prompt}

Provide your expert response including:
1. Professional assessment
2. Key considerations
3. Recommended approach
4. Potential risks
5. Best practices"""

# Helper Functions
def create_download_link(text: str, filename: str) -> str:
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}" style="color: white; text-decoration: none; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); padding: 8px 16px; border-radius: 8px; font-weight: 600;">📥 Download</a>'

def display_hero_section():
    st.markdown('<h1 class="main-header">🚀 Advanced Gemini AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Professional AI Prompting & Evaluation Platform</h4>
        <p>Master advanced prompting techniques with comprehensive analytics and evaluation methods.</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize Session State
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'langfuse_tracker' not in st.session_state:
    st.session_state.langfuse_tracker = LangfuseTracker()
if 'prompt_results' not in st.session_state:
    st.session_state.prompt_results = []

# Load CSS and Hero
load_css()
display_hero_section()

# Sidebar Configuration
with st.sidebar:
    st.markdown("## 🔑 Configuration")
    
    api_key = st.text_input("🔐 Google API Key", type="password", help="Get your API key from Google AI Studio")
    
    if LANGFUSE_AVAILABLE:
        st.markdown("### 📊 Langfuse Tracing")
        langfuse_enabled = st.checkbox("Enable Langfuse Tracing", value=False)
        
        if langfuse_enabled:
            langfuse_public = st.text_input("🔑 Langfuse Public Key", type="password")
            langfuse_secret = st.text_input("🔐 Langfuse Secret Key", type="password")
            
            if langfuse_public and langfuse_secret:
                if st.button("🔗 Connect to Langfuse"):
                    if st.session_state.langfuse_tracker.configure(langfuse_public, langfuse_secret):
                        st.success("✅ Langfuse Connected!")
                    else:
                        st.error("❌ Failed to connect")
    
    if api_key:
        if not st.session_state.api_configured:
            try:
                os.environ["GOOGLE_API_KEY"] = api_key
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                test_response = model.generate_content("Hello")
                st.session_state.api_configured = True
                st.success("✅ Google API Configured!")
            except Exception as e:
                st.error(f"❌ API Error: {str(e)}")
        else:
            st.success("✅ Google API Active")
    
    if st.session_state.prompt_results:
        st.markdown("### 📈 Session Stats")
        st.metric("Total Prompts", len(st.session_state.prompt_results))
        avg_time = np.mean([r.time_taken for r in st.session_state.prompt_results])
        st.metric("Avg Time", f"{avg_time:.2f}s")

if st.session_state.api_configured:
    # Navigation Menu
    selected = option_menu(
        menu_title="🎛️ Advanced AI Techniques",
        options=["🎯 Prompting Techniques", "🔍 Evaluation Methods", "⚡ Batch Processing", "🖼️ Vision Analysis", "📊 Analytics Dashboard"],
        icons=["target", "search", "lightning", "image", "graph-up"],
        menu_icon="robot",
        default_index=0,
        orientation="horizontal"
    )

    # PROMPTING TECHNIQUES SECTION
    if selected == "🎯 Prompting Techniques":
        st.markdown("## 🎯 Advanced Prompting Techniques")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="technique-card">', unsafe_allow_html=True)
            
            technique = st.selectbox("🧠 Select Technique", [
                "Zero-shot Prompting", "Few-shot Prompting", "Chain of Thought (CoT)", 
                "ReAct (Reasoning & Acting)", "Role-based Prompting"
            ])
            
            base_prompt = st.text_area("✏️ Enter your prompt:", height=100, placeholder="Enter your question or task...")
            
            # Technique-specific parameters
            if technique == "Few-shot Prompting":
                st.markdown("**📚 Examples:**")
                num_examples = st.number_input("Number of examples", min_value=1, max_value=3, value=2)
                examples = []
                for i in range(num_examples):
                    col_ex1, col_ex2 = st.columns(2)
                    with col_ex1:
                        example_input = st.text_input(f"Example {i+1} Input:", key=f"ex_input_{i}")
                    with col_ex2:
                        example_output = st.text_input(f"Example {i+1} Output:", key=f"ex_output_{i}")
                    if example_input and example_output:
                        examples.append({"input": example_input, "output": example_output})
            
            elif technique == "ReAct (Reasoning & Acting)":
                max_iterations = st.slider("Max Iterations", min_value=2, max_value=5, value=3)
            
            elif technique == "Role-based Prompting":
                role = st.selectbox("Select Role", ["business_analyst", "data_scientist", "teacher", "consultant"])
            
            if st.button("🚀 Execute Technique", use_container_width=True):
                if base_prompt:
                    with st.spinner(f"🔄 Executing {technique}..."):
                        try:
                            if technique == "Zero-shot Prompting":
                                final_prompt = PromptingTechniques.zero_shot(base_prompt)
                            elif technique == "Few-shot Prompting" and examples:
                                final_prompt = PromptingTechniques.few_shot(base_prompt, examples)
                            elif technique == "Chain of Thought (CoT)":
                                final_prompt = PromptingTechniques.chain_of_thought(base_prompt)
                            elif technique == "ReAct (Reasoning & Acting)":
                                final_prompt = PromptingTechniques.react_prompting(base_prompt, max_iterations)
                            elif technique == "Role-based Prompting":
                                final_prompt = PromptingTechniques.role_based_prompting(base_prompt, role)
                            else:
                                final_prompt = base_prompt
                            
                            result = st.session_state.langfuse_tracker.run_prompt_with_tracing(
                                final_prompt, technique.replace(" ", "_").lower()
                            )
                            
                            st.session_state.prompt_results.append(result)
                            
                            st.markdown("### 🎯 Results:")
                            st.markdown(f'<div class="response-container">{result.output}</div>', unsafe_allow_html=True)
                            
                            col_a, col_b, col_c, col_d = st.columns(4)
                            col_a.metric("Time", f"{result.time_taken}s")
                            col_b.metric("Length", result.length)
                            col_c.metric("Input Tokens", result.input_tokens)
                            col_d.metric("Output Tokens", result.output_tokens)
                            
                            if result.trace_id:
                                st.markdown(f"🔗 [View Trace](https://cloud.langfuse.com/traces/{result.trace_id})")
                            
                            st.markdown(create_download_link(result.output, f"{technique.lower().replace(' ', '_')}_result.txt"), unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a prompt.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎛️ Technique Guide")
            technique_descriptions = {
                "Zero-shot Prompting": "Direct instruction without examples.",
                "Few-shot Prompting": "Learning from examples provided.",
                "Chain of Thought (CoT)": "Step-by-step reasoning process.",
                "ReAct (Reasoning & Acting)": "Iterative reasoning and action.",
                "Role-based Prompting": "Expert perspective responses."
            }
            if technique in technique_descriptions:
                st.info(technique_descriptions[technique])

    # EVALUATION METHODS SECTION
    elif selected == "🔍 Evaluation Methods":
        st.markdown("## 🔍 Advanced Evaluation Methods")
        
        st.markdown('<div class="technique-card">', unsafe_allow_html=True)
        
        evaluation_method = st.selectbox("🔬 Select Method", [
            "Chain of Verification", "Prompt Quality Assessment", "Response Comparison"
        ])
        
        if evaluation_method == "Chain of Verification":
            st.markdown("**🔗 Fact-checking with verification**")
            
            if st.session_state.prompt_results:
                use_existing = st.checkbox("Use existing result", value=False)
                if use_existing:
                    result_options = [f"{r.technique} - {r.prompt[:50]}..." for r in st.session_state.prompt_results]
                    selected_idx = st.selectbox("Select result:", range(len(result_options)), format_func=lambda x: result_options[x])
                    selected_result = st.session_state.prompt_results[selected_idx]
                    original_prompt = selected_result.prompt
                    original_response = selected_result.output
                    st.text_area("Original Prompt:", value=original_prompt, disabled=True)
                    st.text_area("Original Response:", value=original_response, disabled=True)
                else:
                    original_prompt = st.text_area("Original Prompt:")
                    original_response = st.text_area("Original Response:")
            else:
                original_prompt = st.text_area("Original Prompt:")
                original_response = st.text_area("Original Response:")
            
            if st.button("🔍 Verify Response"):
                if original_prompt and original_response:
                    verification_prompt = f"""Original: {original_response}

Perform fact-checking:
1. Identify key claims
2. Verify each claim
3. Rate accuracy (1-10)
4. Provide corrected version if needed"""
                    
                    result = st.session_state.langfuse_tracker.run_prompt_with_tracing(
                        verification_prompt, "verification"
                    )
                    st.markdown("### ✅ Verification Results:")
                    st.markdown(f'<div class="response-container">{result.output}</div>', unsafe_allow_html=True)
        
        elif evaluation_method == "Response Comparison":
            if len(st.session_state.prompt_results) >= 2:
                st.markdown("**⚖️ Compare responses**")
                result_options = [f"{r.technique} - {r.prompt[:50]}..." for r in st.session_state.prompt_results]
                
                col1, col2 = st.columns(2)
                with col1:
                    result1_idx = st.selectbox("First Result:", range(len(result_options)), format_func=lambda x: result_options[x])
                with col2:
                    result2_idx = st.selectbox("Second Result:", range(len(result_options)), format_func=lambda x: result_options[x])
                
                if result1_idx != result2_idx and st.button("⚖️ Compare"):
                    result1 = st.session_state.prompt_results[result1_idx]
                    result2 = st.session_state.prompt_results[result2_idx]
                    
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.markdown(f"**{result1.technique}**")
                        st.markdown(f'<div class="response-container">{result1.output[:300]}...</div>', unsafe_allow_html=True)
                        st.metric("Length", result1.length)
                        st.metric("Time", f"{result1.time_taken}s")
                    
                    with col_r2:
                        st.markdown(f"**{result2.technique}**")
                        st.markdown(f'<div class="response-container">{result2.output[:300]}...</div>', unsafe_allow_html=True)
                        st.metric("Length", result2.length)
                        st.metric("Time", f"{result2.time_taken}s")
            else:
                st.info("Generate at least 2 results to compare responses.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # BATCH PROCESSING SECTION
    elif selected == "⚡ Batch Processing":
        st.markdown("## ⚡ Advanced Batch Processing")
        
        st.markdown("""
        <div class="info-box">
            <h4>🚀 High-Performance Concurrent Processing</h4>
            <p>Process multiple prompts simultaneously with real-time progress tracking.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            st.markdown('<div class="technique-card">', unsafe_allow_html=True)
            
            batch_method = st.radio("📝 Input Method:", ["Manual Entry", "Upload File"])
            
            prompts_to_process = []
            
            if batch_method == "Manual Entry":
                batch_text = st.text_area("Enter prompts (one per line):", height=200, 
                    placeholder="Write a summary of AI\nExplain quantum computing\nCreate a business plan")
                prompts_to_process = [p.strip() for p in batch_text.split('\n') if p.strip()]
            
            elif batch_method == "Upload File":
                uploaded_file = st.file_uploader("📁 Upload text file", type=['txt'])
                if uploaded_file:
                    content = str(uploaded_file.read(), "utf-8")
                    prompts_to_process = [p.strip() for p in content.split('\n') if p.strip()]
                    st.success(f"✅ Loaded {len(prompts_to_process)} prompts")
            
            if prompts_to_process:
                st.markdown(f"### 🎯 Batch Configuration ({len(prompts_to_process)} prompts)")
                
                col_config1, col_config2 = st.columns(2)
                with col_config1:
                    technique_for_batch = st.selectbox("🧠 Apply Technique:", 
                        ["Zero-shot", "Chain of Thought", "Role-based"])
                    max_workers = st.slider("🔧 Workers", 1, 8, min(4, len(prompts_to_process)))
                
                with col_config2:
                    if technique_for_batch == "Role-based":
                        role_for_batch = st.selectbox("👤 Role:", ["business_analyst", "data_scientist", "teacher"])
                    timeout_seconds = st.slider("⏱️ Timeout (sec)", 30, 180, 60)
                
                if st.button("🚀 Start Batch Processing", use_container_width=True):
                    
                    def process_batch_prompt(prompt_data):
                        index, prompt = prompt_data
                        try:
                            if technique_for_batch == "Zero-shot":
                                final_prompt = PromptingTechniques.zero_shot(prompt)
                            elif technique_for_batch == "Chain of Thought":
                                final_prompt = PromptingTechniques.chain_of_thought(prompt)
                            elif technique_for_batch == "Role-based":
                                final_prompt = PromptingTechniques.role_based_prompting(prompt, role_for_batch)
                            else:
                                final_prompt = prompt
                            
                            result = st.session_state.langfuse_tracker.run_prompt_with_tracing(
                                final_prompt, f"batch_{technique_for_batch.lower()}", batch_index=index
                            )
                            result.metadata = result.metadata or {}
                            result.metadata['batch_index'] = index
                            return result
                        except Exception as e:
                            return PromptResult(
                                prompt=prompt, output=f"Error: {str(e)}", technique=f"batch_{technique_for_batch.lower()}",
                                length=0, input_tokens=len(prompt.split()), output_tokens=0, time_taken=0,
                                trace_id=None, timestamp=datetime.now(), metadata={'error': True, 'batch_index': index}
                            )
                    
                    start_time = time.time()
                    results = []
                    
                    progress_bar = st.progress(0)
                    status_container = st.empty()
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_prompt = {
                            executor.submit(process_batch_prompt, (i, prompt)): (i, prompt)
                            for i, prompt in enumerate(prompts_to_process)
                        }
                        
                        completed = 0
                        for future in concurrent.futures.as_completed(future_to_prompt, timeout=timeout_seconds):
                            try:
                                result = future.result()
                                results.append(result)
                                st.session_state.prompt_results.append(result)
                                completed += 1
                                
                                progress = completed / len(prompts_to_process)
                                progress_bar.progress(progress)
                                status_container.text(f"✅ Completed: {completed}/{len(prompts_to_process)}")
                            except Exception as e:
                                st.error(f"❌ Task failed: {str(e)}")
                    
                    total_time = time.time() - start_time
                    
                    st.markdown("### 🎯 Batch Results")
                    
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    col_s1.metric("Total Time", f"{total_time:.2f}s")
                    col_s2.metric("Avg per Prompt", f"{total_time/len(results):.2f}s")
                    col_s3.metric("Success Rate", f"{len([r for r in results if not r.metadata.get('error', False)])}/{len(results)}")
                    col_s4.metric("Total Tokens", sum([r.input_tokens + r.output_tokens for r in results]))
                    
                    # Performance chart
                    if results:
                        times = [r.time_taken for r in results if not r.metadata.get('error', False)]
                        if times:
                            fig = px.bar(x=range(len(times)), y=times, title="📊 Batch Performance",
                                       labels={'x': 'Prompt Index', 'y': 'Response Time (s)'})
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Individual results
                    for result in sorted(results, key=lambda x: x.metadata.get('batch_index', 0)):
                        if not result.metadata.get('error', False):
                            with st.expander(f"📝 Prompt {result.metadata.get('batch_index', 0) + 1}: {result.prompt[:60]}..."):
                                st.markdown(f"**Response:**\n{result.output}")
                                col_r1, col_r2, col_r3 = st.columns(3)
                                col_r1.metric("Time", f"{result.time_taken:.2f}s")
                                col_r2.metric("Length", result.length)
                                col_r3.metric("Tokens", f"{result.input_tokens + result.output_tokens}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ⚡ Batch Guide")
            st.info("""
            **Features:**
            - Concurrent processing
            - Real-time progress
            - Performance analytics
            - Error handling
            - Results export
            """)
            
            if prompts_to_process:
                st.markdown("### 📊 Preview")
                st.write(f"**Prompts:** {len(prompts_to_process)}")
                st.write(f"**Est. Time:** {len(prompts_to_process) * 3:.0f}s")

    # VISION ANALYSIS SECTION
    elif selected == "🖼️ Vision Analysis":
        st.markdown("## 🖼️ Advanced Vision Analysis")
        
        st.markdown("""
        <div class="info-box">
            <h4>🎨 Multimodal AI with Vision Techniques</h4>
            <p>Upload images and apply sophisticated analysis with expert perspectives.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            st.markdown('<div class="technique-card">', unsafe_allow_html=True)
            
            uploaded_images = st.file_uploader("📷 Upload Images", type=['png', 'jpg', 'jpeg', 'gif', 'webp'], accept_multiple_files=True)
            
            if uploaded_images:
                st.markdown(f"### 📸 Uploaded Images ({len(uploaded_images)})")
                
                cols = st.columns(min(3, len(uploaded_images)))
                for i, img_file in enumerate(uploaded_images[:3]):
                    with cols[i % 3]:
                        image = Image.open(img_file)
                        st.image(image, caption=f"Image {i+1}", use_column_width=True)
                
                if len(uploaded_images) > 3:
                    st.info(f"📁 {len(uploaded_images) - 3} more images uploaded")
            
            st.markdown("### 🔧 Analysis Configuration")
            
            analysis_technique = st.selectbox("🧠 Analysis Technique:", [
                "Standard Description", "Chain of Thought Analysis", "Role-based Expert Analysis",
                "Technical Assessment", "Creative Interpretation"
            ])
            
            analysis_focus = st.multiselect("🎯 Focus Areas:", [
                "Objects & People", "Colors & Composition", "Text Extraction",
                "Emotional Tone", "Technical Quality", "Artistic Elements"
            ], default=["Objects & People", "Colors & Composition"])
            
            custom_prompt = st.text_area("✏️ Additional Instructions:", height=80,
                placeholder="Any specific questions about the images...")
            
            col_set1, col_set2 = st.columns(2)
            with col_set1:
                detail_level = st.select_slider("📊 Detail Level:", options=["Brief", "Standard", "Comprehensive"], value="Standard")
            
            with col_set2:
                if analysis_technique == "Role-based Expert Analysis":
                    expert_role = st.selectbox("👤 Expert Role:", ["photographer", "designer", "marketer", "art_critic"])
            
            if uploaded_images and st.button("🔍 Analyze Images", use_container_width=True):
                
                def create_vision_prompt(technique, focus_areas, custom_instructions, detail_level, role=None):
                    base_prompts = {
                        "Standard Description": "Provide a comprehensive description of this image.",
                        "Chain of Thought Analysis": "Analyze step-by-step: 1) First impression 2) Details 3) Context 4) Assessment",
                        "Role-based Expert Analysis": f"As a professional {role}, analyze this image from your expert perspective.",
                        "Technical Assessment": "Analyze technical aspects: quality, composition, lighting, techniques.",
                        "Creative Interpretation": "Provide creative interpretation: mood, story, symbolism, artistic elements."
                    }
                    
                    prompt = base_prompts[technique] + "\n\n"
                    
                    if focus_areas:
                        prompt += "Focus on:\n" + "\n".join([f"- {area}" for area in focus_areas]) + "\n\n"
                    
                    if custom_instructions:
                        prompt += f"Additional: {custom_instructions}\n\n"
                    
                    prompt += f"Provide {detail_level.lower()} analysis with clear insights."
                    return prompt
                
                for i, img_file in enumerate(uploaded_images):
                    st.markdown(f"### 📷 Analysis {i+1}: {img_file.name}")
                    
                    with st.spinner(f"🔄 Analyzing image {i+1}..."):
                        try:
                            image = Image.open(img_file)
                            
                            vision_prompt = create_vision_prompt(
                                analysis_technique, analysis_focus, custom_prompt, detail_level,
                                expert_role if analysis_technique == "Role-based Expert Analysis" else None
                            )
                            
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            response = model.generate_content([vision_prompt, image])
                            
                            result = PromptResult(
                                prompt=vision_prompt, output=response.text,
                                technique=f"vision_{analysis_technique.lower().replace(' ', '_')}",
                                length=len(response.text), input_tokens=len(vision_prompt.split()),
                                output_tokens=len(response.text.split()), time_taken=0,
                                trace_id=None, timestamp=datetime.now(),
                                metadata={'image_name': img_file.name, 'analysis_technique': analysis_technique}
                            )
                            
                            st.session_state.prompt_results.append(result)
                            
                            st.markdown(f'<div class="response-container">{response.text}</div>', unsafe_allow_html=True)
                            
                            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                            col_info1.metric("Size", f"{image.size[0]}×{image.size[1]}")
                            col_info2.metric("Format", image.format)
                            col_info3.metric("Mode", image.mode)
                            col_info4.metric("Response Length", len(response.text))
                            
                            st.markdown(create_download_link(response.text, f"vision_analysis_{i+1}.txt"), unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ Error analyzing image {i+1}: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎨 Vision Guide")
            
            technique_guide = {
                "Standard Description": "Comprehensive overview of image content.",
                "Chain of Thought Analysis": "Step-by-step systematic analysis.",
                "Role-based Expert Analysis": "Professional domain expert perspective.",
                "Technical Assessment": "Technical quality and composition evaluation.",
                "Creative Interpretation": "Artistic and creative analysis."
            }
            
            if 'analysis_technique' in locals():
                st.info(technique_guide.get(analysis_technique, "Advanced vision analysis"))
            
            st.markdown("### 📊 Supported")
            st.write("• PNG, JPG, JPEG, GIF, WebP")
            st.write("• Multiple images")
            st.write("• Expert analysis roles")
            st.write("• Custom instructions")

    # ANALYTICS DASHBOARD SECTION
    elif selected == "📊 Analytics Dashboard":
        st.markdown("## 📊 Professional Analytics Dashboard")
        
        if not st.session_state.prompt_results:
            st.markdown("""
            <div class="info-box">
                <h4>📈 Generate Data for Analytics</h4>
                <p>Use prompting techniques, batch processing, or vision analysis to generate comprehensive analytics!</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Analytics Preview")
            col_prev1, col_prev2 = st.columns(2)
            with col_prev1:
                st.markdown("""
                **📈 Performance Metrics:**
                - Response time trends
                - Token usage patterns
                - Efficiency analysis
                - Success rates
                """)
            
            with col_prev2:
                st.markdown("""
                **🎯 Technique Analysis:**
                - Usage distribution
                - Performance comparison
                - Time series analysis
                - Cost optimization
                """)
        else:
            # Convert to DataFrame
            results_data = []
            for result in st.session_state.prompt_results:
                results_data.append({
                    'timestamp': result.timestamp,
                    'technique': result.technique,
                    'prompt': result.prompt[:100] + "..." if len(result.prompt) > 100 else result.prompt,
                    'time_taken': result.time_taken,
                    'length': result.length,
                    'input_tokens': result.input_tokens,
                    'output_tokens': result.output_tokens,
                    'total_tokens': result.input_tokens + result.output_tokens,
                    'trace_id': result.trace_id,
                    'has_trace': result.trace_id is not None,
                    'hour': result.timestamp.hour,
                    'efficiency': result.length / result.time_taken if result.time_taken > 0 else 0
                })
            
            df = pd.DataFrame(results_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Executive Summary
            st.markdown("### 📈 Executive Summary")
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            
            col_m1.metric("Total Prompts", len(df))
            col_m2.metric("Avg Response Time", f"{df['time_taken'].mean():.2f}s")
            col_m3.metric("Total Tokens", f"{df['total_tokens'].sum():,}")
            col_m4.metric("Traced Prompts", df['has_trace'].sum())
            col_m5.metric("Avg Efficiency", f"{df['efficiency'].mean():.0f} chars/s")
            
            # Performance Analytics
            st.markdown("### 📊 Performance Analytics")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Response time trends
                fig_time = px.line(df, x='timestamp', y='time_taken', color='technique',
                                 title='⏱️ Response Time Trends', labels={'time_taken': 'Response Time (s)'})
                fig_time.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col_chart2:
                # Token usage pattern
                fig_tokens = px.scatter(df, x='input_tokens', y='output_tokens', size='time_taken', color='technique',
                                      title='🎯 Token Usage Pattern', hover_data=['length', 'time_taken'])
                fig_tokens.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
                st.plotly_chart(fig_tokens, use_container_width=True)
            
            # Technique Analysis
            col_tech1, col_tech2 = st.columns(2)
            
            with col_tech1:
                # Usage distribution
                technique_counts = df['technique'].value_counts()
                fig_pie = px.pie(values=technique_counts.values, names=technique_counts.index,
                               title='🧠 Technique Usage Distribution')
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_tech2:
                # Performance comparison
                technique_stats = df.groupby('technique')['time_taken'].mean().reset_index()
                fig_bar = px.bar(technique_stats, x='technique', y='time_taken',
                               title='⚡ Avg Response Time by Technique', color='time_taken')
                fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400, xaxis={'tickangle': 45})
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Advanced Analytics
            st.markdown("### 🔍 Advanced Analytics")
            
            col_adv1, col_adv2, col_adv3 = st.columns(3)
            
            with col_adv1:
                # Efficiency distribution
                fig_eff = px.box(df, x='technique', y='efficiency', title='📈 Efficiency Distribution', color='technique')
                fig_eff.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300, xaxis={'tickangle': 45})
                st.plotly_chart(fig_eff, use_container_width=True)
            
            with col_adv2:
                # Hourly activity
                hourly_activity = df.groupby('hour').size()
                fig_hourly = px.bar(x=hourly_activity.index, y=hourly_activity.values,
                                  title='🕐 Activity by Hour', labels={'x': 'Hour', 'y': 'Prompts'})
                fig_hourly.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with col_adv3:
                # Length vs Time correlation
                fig_corr = px.scatter(df, x='length', y='time_taken', trendline='ols',
                                    title='🔗 Length vs Time', color='technique')
                fig_corr.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Performance Matrix
            st.markdown("### 📋 Performance Matrix")
            performance_matrix = df.groupby('technique').agg({
                'time_taken': ['mean', 'std', 'min', 'max'],
                'length': ['mean', 'std'],
                'total_tokens': ['mean', 'sum'],
                'has_trace': 'sum'
            }).round(2)
            
            performance_matrix.columns = ['Avg Time', 'Time StdDev', 'Min Time', 'Max Time',
                                        'Avg Length', 'Length StdDev', 'Avg Tokens', 'Total Tokens', 'Traces']
            st.dataframe(performance_matrix, use_container_width=True)
            
            # Detailed Results
            st.markdown("### 🔍 Detailed Results")
            
            col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
            
            with col_filter1:
                technique_filter = st.multiselect("Filter Technique:", df['technique'].unique(), default=df['technique'].unique())
            
            with col_filter2:
                min_time = st.number_input("Min Time (s):", 0.0, float(df['time_taken'].max()), 0.0)
                max_time = st.number_input("Max Time (s):", 0.0, float(df['time_taken'].max()), float(df['time_taken'].max()))
            
            with col_filter3:
                min_length = st.number_input("Min Length:", 0, int(df['length'].max()), 0)
                max_length = st.number_input("Max Length:", 0, int(df['length'].max()), int(df['length'].max()))
            
            with col_filter4:
                show_traced_only = st.checkbox("Traced Only")
                sort_by = st.selectbox("Sort by:", ['timestamp', 'time_taken', 'length', 'total_tokens'])
            
            # Apply filters
            filtered_df = df[
                (df['technique'].isin(technique_filter)) &
                (df['time_taken'] >= min_time) & (df['time_taken'] <= max_time) &
                (df['length'] >= min_length) & (df['length'] <= max_length)
            ]
            
            if show_traced_only:
                filtered_df = filtered_df[filtered_df['has_trace'] == True]
            
            filtered_df = filtered_df.sort_values(sort_by, ascending=False)
            
            st.dataframe(filtered_df[['timestamp', 'technique', 'prompt', 'time_taken', 'length', 'input_tokens', 'output_tokens', 'total_tokens']], use_container_width=True)
            
            # Export Options
            st.markdown("### 💾 Export Analytics")
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                if st.button("📊 Export Full Dataset"):
                    csv_data = df.to_csv(index=False)
                    st.download_button("Download CSV", csv_data, f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            
            with col_export2:
                if st.button("📈 Export Performance"):
                    summary_data = performance_matrix.to_csv()
                    st.download_button("Download Summary", summary_data, f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            
            with col_export3:
                if st.button("🗑️ Clear Data"):
                    if st.button("⚠️ Confirm Clear", type="primary"):
                        st.session_state.prompt_results = []
                        st.rerun()

else:
    st.warning("⚠️ Please configure your Google API key in the sidebar to get started.")
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. **Enter Key**: Paste your API key in the sidebar
    3. **Optional**: Configure Langfuse for advanced tracing
    4. **Start Experimenting**: Use advanced prompting techniques!
    
    ### ✨ Professional Features Available:
    
    #### 🎯 **Prompting Techniques:**
    - **Zero-shot**: Direct instruction without examples
    - **Few-shot**: Learning from provided examples  
    - **Chain of Thought**: Step-by-step reasoning
    - **ReAct**: Reasoning and Acting framework
    - **Role-based**: Expert perspective prompting
    
    #### 🔍 **Evaluation Methods:**
    - **Chain of Verification**: Systematic fact-checking
    - **Response Comparison**: Side-by-side analysis
    
    #### ⚡ **Batch Processing:**
    - **Concurrent Execution**: Process multiple prompts simultaneously
    - **Real-time Progress**: Live tracking with analytics
    
    #### 🖼️ **Vision Analysis:**
    - **Multi-Image Support**: Analyze multiple images
    - **Expert Perspectives**: Professional analysis roles
    
    #### 📊 **Analytics Dashboard:**
    - **Interactive Charts**: Performance trends and insights
    - **Advanced Filtering**: Multi-criteria exploration
    - **Export Options**: Professional reporting
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 Advanced Gemini AI Assistant | Professional Prompting & Evaluation Platform</p>
    <p>Built with Streamlit, Google AI & Langfuse | Enterprise-Ready Analytics</p>
</div>
""", unsafe_allow_html=True)