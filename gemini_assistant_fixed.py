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
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Enhanced Langfuse Integration with proper error handling
LANGFUSE_AVAILABLE = False
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
    print("✅ Langfuse available")
except ImportError:
    print("⚠️ Langfuse not available - install with: pip install langfuse")

# ------------------ Page Setup ------------------
st.set_page_config(
    page_title="🚀 Advanced Gemini AI Assistant", 
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
    
    .technique-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
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
    
    .reasoning-step {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    .evaluation-result {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
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
    
    .tree-node {
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #667eea;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Hide Streamlit elements */
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

# ------------------ Data Classes ------------------
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
    evaluation_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

# ------------------ Enhanced Langfuse Integration ------------------
class AdvancedLangfuseTracker:
    def __init__(self):
        self.langfuse = None
        self.is_configured = False
        self.session_id = str(uuid.uuid4())
        
    def configure(self, public_key: str, secret_key: str, host: str = "https://cloud.langfuse.com"):
        if not LANGFUSE_AVAILABLE:
            st.error("❌ Langfuse not installed. Install with: pip install langfuse")
            return False
            
        try:
            # Set environment variables
            os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
            os.environ["LANGFUSE_SECRET_KEY"] = secret_key
            os.environ["LANGFUSE_HOST"] = host
            
            # Initialize Langfuse
            self.langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            
            # Test connection
            try:
                self.langfuse.auth_check()
                self.is_configured = True
                return True
            except Exception as auth_error:
                st.error(f"❌ Langfuse authentication failed: {auth_error}")
                return False
                
        except Exception as e:
            st.error(f"❌ Langfuse configuration failed: {e}")
            return False
    
    def run_prompt_with_tracing(self, prompt_text: str, technique: str, model_name: str = "gemini-1.5-flash", **kwargs):
        """Enhanced prompt execution with proper Langfuse tracing"""
        start_time = time.time()
        trace_id = None
        
        try:
            # Execute with Gemini
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt_text)
            result_text = response.text
            
            end_time = time.time()
            time_taken = round(end_time - start_time, 2)
            
            # Calculate tokens (approximation)
            input_tokens = len(prompt_text.split())
            output_tokens = len(result_text.split())
            
            # Create Langfuse trace if configured
            if self.is_configured and self.langfuse and LANGFUSE_AVAILABLE:
                try:
                    # Create trace
                    trace = self.langfuse.trace(
                        name=f"{technique}_execution",
                        session_id=self.session_id,
                        input={"prompt": prompt_text, "technique": technique},
                        output={"response": result_text},
                        metadata={
                            "model": model_name,
                            "time_taken": time_taken,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "technique": technique,
                            **kwargs
                        }
                    )
                    
                    # Create generation
                    generation = self.langfuse.generation(
                        name=f"gemini_{technique}",
                        model=model_name,
                        input=prompt_text,
                        output=result_text,
                        usage={
                            "input": input_tokens,
                            "output": output_tokens,
                            "total": input_tokens + output_tokens
                        },
                        metadata={"technique": technique},
                        trace_id=trace.id
                    )
                    
                    trace_id = trace.id
                    
                except Exception as langfuse_error:
                    st.warning(f"⚠️ Langfuse tracing failed: {langfuse_error}")
                    trace_id = None
            
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
            end_time = time.time()
            error_result = PromptResult(
                prompt=prompt_text,
                output=f"Error: {str(e)}",
                technique=technique,
                length=0,
                input_tokens=len(prompt_text.split()),
                output_tokens=0,
                time_taken=round(end_time - start_time, 2),
                trace_id=None,
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
            
            # Log error to Langfuse if configured
            if self.is_configured and self.langfuse and LANGFUSE_AVAILABLE:
                try:
                    self.langfuse.trace(
                        name=f"{technique}_error",
                        session_id=self.session_id,
                        input={"prompt": prompt_text},
                        output={"error": str(e)},
                        metadata={"technique": technique}
                    )
                except:
                    pass  # Ignore Langfuse errors during error logging
            
            return error_result

# ------------------ Prompting Techniques ------------------
class PromptingTechniques:
    
    @staticmethod
    def zero_shot(base_prompt: str) -> str:
        """Zero-shot prompting - direct instruction without examples"""
        return f"""Task: {base_prompt}

Provide a comprehensive response based on your knowledge without any examples or prior context."""

    @staticmethod
    def few_shot(base_prompt: str, examples: List[Dict[str, str]]) -> str:
        """Few-shot prompting with examples"""
        examples_text = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(examples)
        ])
        
        return f"""Task: {base_prompt}

Here are some examples:

{examples_text}

Now, please complete the following task following the same pattern:"""

    @staticmethod
    def chain_of_thought(base_prompt: str) -> str:
        """Chain of Thought prompting for step-by-step reasoning"""
        return f"""Task: {base_prompt}

Please think through this step-by-step:

1. First, let me understand what is being asked...
2. Let me break this down into smaller parts...
3. Now I'll work through each part systematically...
4. Let me verify my reasoning...
5. Finally, I'll provide the complete answer...

Let's work through this step by step:"""

    @staticmethod
    def react_prompting(base_prompt: str, max_iterations: int = 3) -> str:
        """ReAct (Reasoning and Acting) prompting"""
        return f"""You are an AI assistant that uses the ReAct (Reasoning and Acting) framework.

Task: {base_prompt}

For this task, use exactly {max_iterations} iterations of the following format:

ITERATION 1:
Thought: [Your reasoning about the current state and what you need to do]
Action: [What specific action or analysis you will perform]
Observation: [What you observe or discover from your action]
Assessment: [Evaluate your progress and plan next steps]

ITERATION 2:
Thought: [Continue reasoning based on previous observations]
Action: [Next action to take]
Observation: [New observations]
Assessment: [Updated assessment]

ITERATION 3:
Thought: [Final reasoning incorporating all previous observations]
Action: [Final action or synthesis]
Observation: [Final observations]
Assessment: [Complete evaluation]

FINAL ANSWER:
[Provide your comprehensive final answer based on all iterations]

Begin your ReAct analysis:"""

    @staticmethod
    def tree_of_thoughts(base_prompt: str, num_branches: int = 3) -> str:
        """Tree of Thoughts prompting for exploring multiple reasoning paths"""
        return f"""Task: {base_prompt}

Use Tree of Thoughts approach to explore {num_branches} different reasoning paths:

BRANCH 1 - Approach A:
Reasoning: [First approach to solving this problem]
Analysis: [Detailed analysis using this approach]
Conclusion: [What this approach suggests]

BRANCH 2 - Approach B:
Reasoning: [Second approach to solving this problem]
Analysis: [Detailed analysis using this approach]
Conclusion: [What this approach suggests]

BRANCH 3 - Approach C:
Reasoning: [Third approach to solving this problem]
Analysis: [Detailed analysis using this approach]
Conclusion: [What this approach suggests]

SYNTHESIS:
Comparison: [Compare the three approaches]
Best Elements: [Identify the best elements from each branch]
Integrated Solution: [Combine the best insights into a final solution]

Final Answer: [Your comprehensive answer incorporating insights from all branches]

Begin your Tree of Thoughts analysis:"""

    @staticmethod
    def role_based_prompting(base_prompt: str, role: str, expertise_level: str = "expert") -> str:
        """Role-based prompting with specific persona"""
        role_descriptions = {
            "business_analyst": "You are a senior business analyst with 10+ years of experience in strategic planning and market analysis.",
            "data_scientist": "You are an expert data scientist with deep knowledge in machine learning, statistics, and data analysis.",
            "software_engineer": "You are a senior software engineer with expertise in system design, coding best practices, and architecture.",
            "marketing_expert": "You are a marketing professional with extensive experience in digital marketing, brand strategy, and customer acquisition.",
            "financial_advisor": "You are a certified financial advisor with expertise in investment strategies, risk management, and financial planning.",
            "teacher": "You are an experienced educator skilled at explaining complex concepts in simple, understandable terms.",
            "consultant": "You are a management consultant with experience helping organizations solve complex business problems.",
            "researcher": "You are an academic researcher with expertise in conducting thorough analysis and presenting findings."
        }
        
        role_desc = role_descriptions.get(role, f"You are an {expertise_level} {role.replace('_', ' ')} with extensive experience in your field.")
        
        return f"""{role_desc}

Given your expertise and experience, please address the following:

{base_prompt}

Provide your response from the perspective of your professional role, including:
1. Your professional assessment
2. Key considerations from your field
3. Recommended approach or solution
4. Potential risks or challenges
5. Best practices to follow

Your expert response:"""

    @staticmethod
    def instruction_based(base_prompt: str, instructions: List[str]) -> str:
        """Instruction-based prompting with specific guidelines"""
        instructions_text = "\n".join([f"- {instruction}" for instruction in instructions])
        
        return f"""Task: {base_prompt}

Please follow these specific instructions:
{instructions_text}

Your response following the above instructions:"""

    @staticmethod
    def parallel_prompting(base_prompt: str, perspectives: List[str]) -> str:
        """Parallel prompting from multiple perspectives"""
        perspectives_text = "\n".join([f"{i+1}. {perspective}" for i, perspective in enumerate(perspectives)])
        
        return f"""Task: {base_prompt}

Please analyze this from the following {len(perspectives)} different perspectives:

{perspectives_text}

For each perspective, provide:
- Analysis from that viewpoint
- Key insights or concerns
- Recommendations

Then provide a synthesis that considers all perspectives.

Multi-perspective analysis:"""

# ------------------ Evaluation Methods ------------------
class EvaluationMethods:
    
    @staticmethod
    def chain_of_verification(original_response: str, original_prompt: str) -> str:
        """Chain of Verification (CoVe) for fact-checking responses"""
        return f"""Original Question: {original_prompt}

Original Response: {original_response}

Now, please perform Chain of Verification (CoVe):

STEP 1 - IDENTIFY CLAIMS:
List all factual claims made in the original response:
1. [Claim 1]
2. [Claim 2]
3. [Claim 3]

STEP 2 - GENERATE VERIFICATION QUESTIONS:
For each claim, create a verification question:
1. [Verification question for claim 1]
2. [Verification question for claim 2]
3. [Verification question for claim 3]

STEP 3 - ANSWER VERIFICATION QUESTIONS:
Answer each verification question independently:
1. [Answer to verification question 1]
2. [Answer to verification question 2]
3. [Answer to verification question 3]

STEP 4 - FINAL VERIFICATION:
Based on the verification answers, assess the original response:
- Accurate claims: [List verified claims]
- Inaccurate claims: [List any inaccuracies found]
- Uncertain claims: [List claims that couldn't be verified]

STEP 5 - REVISED RESPONSE:
Provide a corrected and improved version of the original response:

[Improved response incorporating verification results]

Confidence Score: [Rate confidence from 1-10 based on verification]"""

    @staticmethod
    def prompt_quality_assessment(prompt: str) -> str:
        """Assess the quality of a prompt"""
        return f"""Analyze the following prompt for quality and effectiveness:

PROMPT TO EVALUATE: "{prompt}"

ASSESSMENT CRITERIA:

1. CLARITY (1-10):
   Score: [X/10]
   Analysis: [How clear and unambiguous is the prompt?]

2. SPECIFICITY (1-10):
   Score: [X/10]
   Analysis: [How specific are the instructions?]

3. CONTEXT (1-10):
   Score: [X/10]
   Analysis: [Is sufficient context provided?]

4. STRUCTURE (1-10):
   Score: [X/10]
   Analysis: [Is the prompt well-structured?]

5. COMPLETENESS (1-10):
   Score: [X/10]
   Analysis: [Does it include all necessary information?]

OVERALL SCORE: [Total/50]

STRENGTHS:
- [List prompt strengths]

WEAKNESSES:
- [List areas for improvement]

IMPROVED VERSION:
[Provide an enhanced version of the prompt]

EXPLANATION OF IMPROVEMENTS:
[Explain what was changed and why]"""

# ------------------ Helper Functions ------------------
def create_download_link(text: str, filename: str) -> str:
    """Create a download link for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}" style="color: white; text-decoration: none; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); padding: 8px 16px; border-radius: 8px; font-weight: 600;">📥 Download</a>'
    return href

def display_hero_section():
    st.markdown('<h1 class="main-header">🚀 Advanced Gemini AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Professional AI Prompting & Evaluation Platform</h4>
        <p>Master advanced prompting techniques: Zero-shot, Few-shot, Chain-of-Thought, ReAct, Tree of Thoughts, and comprehensive evaluation methods with Langfuse tracing.</p>
    </div>
    """, unsafe_allow_html=True)

def format_reasoning_output(output_text: str, technique: str) -> str:
    """Format different reasoning outputs for better display"""
    if "ReAct" in technique:
        formatted = output_text.replace("ITERATION", '<div class="reasoning-step"><strong>🔄 ITERATION')
        formatted = formatted.replace("Thought:", '</strong><br><strong>💭 Thought:</strong>')
        formatted = formatted.replace("Action:", '<br><strong>⚡ Action:</strong>')
        formatted = formatted.replace("Observation:", '<br><strong>👁️ Observation:</strong>')
        formatted = formatted.replace("Assessment:", '<br><strong>📊 Assessment:</strong>')
        formatted = formatted.replace("FINAL ANSWER:", '</div><div class="evaluation-result"><strong>✨ FINAL ANSWER:</strong>')
        formatted += '</div>'
    elif "Tree of Thoughts" in technique:
        formatted = output_text.replace("BRANCH", '<div class="tree-node"><strong>🌳 BRANCH')
        formatted = formatted.replace("SYNTHESIS:", '</div><div class="evaluation-result"><strong>🔗 SYNTHESIS:</strong>')
        formatted += '</div>'
    elif "Chain of Verification" in technique:
        formatted = output_text.replace("STEP", '<div class="evaluation-result"><strong>✅ STEP')
        formatted += '</div>'
    else:
        formatted = output_text
    
    return formatted

# ------------------ Initialize Session State ------------------
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'langfuse_tracker' not in st.session_state:
    st.session_state.langfuse_tracker = AdvancedLangfuseTracker()
if 'prompt_results' not in st.session_state:
    st.session_state.prompt_results = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []

# ------------------ Load CSS ------------------
load_css()

# ------------------ Hero Section ------------------
display_hero_section()

# ------------------ Enhanced Sidebar Configuration ------------------
with st.sidebar:
    st.markdown("## 🔑 Configuration")
    
    # Google API Key
    api_key = st.text_input("🔐 Google API Key", type="password", 
                           help="Get your API key from Google AI Studio")
    
    # Enhanced Langfuse Configuration
    if LANGFUSE_AVAILABLE:
        st.markdown("### 📊 Langfuse Tracing")
        langfuse_enabled = st.checkbox("Enable Langfuse Tracing", value=False)
        
        if langfuse_enabled:
            st.markdown("""
            <div class="info-box">
                <small><strong>Get Langfuse Keys:</strong><br>
                1. Visit <a href="https://cloud.langfuse.com" target="_blank">cloud.langfuse.com</a><br>
                2. Create account & project<br>
                3. Go to Settings → API Keys</small>
            </div>
            """, unsafe_allow_html=True)
            
            langfuse_public = st.text_input("🔑 Langfuse Public Key", type="password")
            langfuse_secret = st.text_input("🔐 Langfuse Secret Key", type="password")
            langfuse_host = st.text_input("🌐 Langfuse Host", value="https://cloud.langfuse.com")
            
            if langfuse_public and langfuse_secret:
                if st.button("🔗 Connect to Langfuse"):
                    with st.spinner("Connecting to Langfuse..."):
                        if st.session_state.langfuse_tracker.configure(langfuse_public, langfuse_secret, langfuse_host):
                            st.success("✅ Langfuse Connected Successfully!")
                            st.markdown(f"""
                            <div class="success-box">
                                <strong>🎯 Tracing Active</strong><br>
                                Session ID: <code>{st.session_state.langfuse_tracker.session_id[:8]}...</code><br>
                                All prompts will be traced in Langfuse dashboard.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("❌ Failed to connect to Langfuse")
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Langfuse Not Available</strong><br>
            Install with: <code>pip install langfuse</code><br>
            Then restart the application.
        </div>
        """, unsafe_allow_html=True)
    
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
                st.success("✅ Google API Configured!")
                
            except Exception as e:
                st.error(f"❌ API Error: {str(e)}")
                st.session_state.api_configured = False
        else:
            st.success("✅ Google API Active")
            
            # Show Langfuse status
            if st.session_state.langfuse_tracker.is_configured:
                st.markdown("""
                <div class="langfuse-trace">
                    <strong>📊 Langfuse Tracing Active</strong><br>
                    <small>All interactions are being traced</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Enhanced Session Stats
    if st.session_state.prompt_results:
        st.markdown("### 📈 Session Analytics")
        total_prompts = len(st.session_state.prompt_results)
        avg_time = np.mean([r.time_taken for r in st.session_state.prompt_results])
        total_tokens = sum([r.input_tokens + r.output_tokens for r in st.session_state.prompt_results])
        
        # Technique distribution
        techniques = [r.technique for r in st.session_state.prompt_results]
        technique_counts = pd.Series(techniques).value_counts()
        
        st.metric("Total Prompts", total_prompts)
        st.metric("Avg Time", f"{avg_time:.2f}s")
        st.metric("Total Tokens", total_tokens)
        
        if len(technique_counts) > 0:
            st.markdown("**Techniques Used:**")
            for technique, count in technique_counts.head(5).items():
                st.write(f"• {technique}: {count}")

if st.session_state.api_configured:
    # ------------------ Enhanced Navigation Menu ------------------
    selected = option_menu(
        menu_title="🎛️ Advanced AI Techniques",
        options=[
            "🎯 Prompting Techniques", 
            "🔍 Evaluation Methods",
            "⚡ Batch Processing",
            "🖼️ Vision Analysis",
            "📊 Analytics Dashboard"
        ],
        icons=["target", "search", "lightning", "image", "graph-up"],
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

    # ------------------ Prompting Techniques Section ------------------
    if selected == "🎯 Prompting Techniques":
        st.markdown("## 🎯 Advanced Prompting Techniques")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="technique-card">', unsafe_allow_html=True)
            
            # Technique Selection
            technique = st.selectbox(
                "🧠 Select Prompting Technique",
                [
                    "Zero-shot Prompting",
                    "Few-shot Prompting", 
                    "Chain of Thought (CoT)",
                    "ReAct (Reasoning & Acting)",
                    "Tree of Thoughts (ToT)",
                    "Role-based Prompting",
                    "Instruction-based Prompting",
                    "Parallel Prompting"
                ]
            )
            
            # Base prompt input
            base_prompt = st.text_area(
                "✏️ Enter your base prompt:",
                height=100,
                placeholder="Enter the main question or task you want the AI to address..."
            )
            
            # Technique-specific parameters
            if technique == "Few-shot Prompting":
                st.markdown("**📚 Examples for Few-shot Learning:**")
                num_examples = st.number_input("Number of examples", min_value=1, max_value=5, value=2)
                
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
            
            elif technique == "Tree of Thoughts (ToT)":
                num_branches = st.slider("Number of Branches", min_value=2, max_value=5, value=3)
            
            elif technique == "Role-based Prompting":
                role = st.selectbox(
                    "Select Role",
                    ["business_analyst", "data_scientist", "software_engineer", 
                     "marketing_expert", "financial_advisor", "teacher", "consultant", "researcher"]
                )
                expertise_level = st.selectbox("Expertise Level", ["expert", "senior", "experienced"])
            
            elif technique == "Instruction-based Prompting":
                st.markdown("**📋 Specific Instructions:**")
                num_instructions = st.number_input("Number of instructions", min_value=1, max_value=8, value=3)
                
                instructions = []
                for i in range(num_instructions):
                    instruction = st.text_input(f"Instruction {i+1}:", key=f"instruction_{i}")
                    if instruction:
                        instructions.append(instruction)
            
            elif technique == "Parallel Prompting":
                st.markdown("**🔄 Multiple Perspectives:**")
                num_perspectives = st.number_input("Number of perspectives", min_value=2, max_value=6, value=3)
                
                perspectives = []
                for i in range(num_perspectives):
                    perspective = st.text_input(f"Perspective {i+1}:", key=f"perspective_{i}")
                    if perspective:
                        perspectives.append(perspective)
            
            # Generate button
            if st.button("🚀 Execute Technique", use_container_width=True):
                if base_prompt:
                    with st.spinner(f"🔄 Executing {technique}..."):
                        try:
                            # Generate the appropriate prompt based on technique
                            if technique == "Zero-shot Prompting":
                                final_prompt = PromptingTechniques.zero_shot(base_prompt)
                            elif technique == "Few-shot Prompting" and examples:
                                final_prompt = PromptingTechniques.few_shot(base_prompt, examples)
                            elif technique == "Chain of Thought (CoT)":
                                final_prompt = PromptingTechniques.chain_of_thought(base_prompt)
                            elif technique == "ReAct (Reasoning & Acting)":
                                final_prompt = PromptingTechniques.react_prompting(base_prompt, max_iterations)
                            elif technique == "Tree of Thoughts (ToT)":
                                final_prompt = PromptingTechniques.tree_of_thoughts(base_prompt, num_branches)
                            elif technique == "Role-based Prompting":
                                final_prompt = PromptingTechniques.role_based_prompting(base_prompt, role, expertise_level)
                            elif technique == "Instruction-based Prompting" and instructions:
                                final_prompt = PromptingTechniques.instruction_based(base_prompt, instructions)
                            elif technique == "Parallel Prompting" and perspectives:
                                final_prompt = PromptingTechniques.parallel_prompting(base_prompt, perspectives)
                            else:
                                st.warning("⚠️ Please fill in all required parameters for the selected technique.")
                                final_prompt = None
                            
                            if final_prompt:
                                # Execute with enhanced tracing
                                result = st.session_state.langfuse_tracker.run_prompt_with_tracing(
                                    final_prompt, 
                                    technique.replace(" ", "_").lower(),
                                    base_prompt=base_prompt,
                                    technique_params={"technique": technique}
                                )
                                
                                st.session_state.prompt_results.append(result)
                                
                                # Display results
                                st.markdown("### 🎯 Results:")
                                
                                # Show formatted output
                                formatted_output = format_reasoning_output(result.output, technique)
                                st.markdown(f'<div class="response-container">{formatted_output}</div>', 
                                          unsafe_allow_html=True)
                                
                                # Metrics
                                col_a, col_b, col_c, col_d = st.columns(4)
                                col_a.metric("Time", f"{result.time_taken}s")
                                col_b.metric("Length", result.length)
                                col_c.metric("Input Tokens", result.input_tokens)
                                col_d.metric("Output Tokens", result.output_tokens)
                                
                                # Langfuse trace link
                                if result.trace_id:
                                    langfuse_url = f"https://cloud.langfuse.com/traces/{result.trace_id}"
                                    st.markdown(f"""
                                    <div class="langfuse-trace">
                                        <strong>📊 Langfuse Trace Created</strong><br>
                                        <a href="{langfuse_url}" target="_blank">🔗 View in Langfuse Dashboard</a><br>
                                        <small>Trace ID: {result.trace_id}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Download option
                                st.markdown(
                                    create_download_link(
                                        f"Technique: {technique}\nBase Prompt: {base_prompt}\n\nResult:\n{result.output}",
                                        f"{technique.lower().replace(' ', '_')}_result.txt"
                                    ), 
                                    unsafe_allow_html=True
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Error executing technique: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a base prompt.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎛️ Technique Guide")
            
            technique_descriptions = {
                "Zero-shot Prompting": "Direct instruction without examples. Best for general knowledge tasks.",
                "Few-shot Prompting": "Learning from examples. Provide 2-5 examples for pattern recognition.",
                "Chain of Thought (CoT)": "Step-by-step reasoning. Excellent for complex problem-solving.",
                "ReAct (Reasoning & Acting)": "Iterative reasoning and action. Great for multi-step analysis.",
                "Tree of Thoughts (ToT)": "Explore multiple reasoning paths. Ideal for complex decisions.",
                "Role-based Prompting": "Leverage specific expertise. Get professional perspective responses.",
                "Instruction-based Prompting": "Detailed guidelines. Ensure specific format and requirements.",
                "Parallel Prompting": "Multiple perspectives. Compare different viewpoints on the same topic."
            }
            
            if technique in technique_descriptions:
                st.info(technique_descriptions[technique])
            
            # Show technique statistics
            if st.session_state.prompt_results:
                technique_stats = pd.Series([r.technique for r in st.session_state.prompt_results]).value_counts()
                if len(technique_stats) > 0:
                    st.markdown("### 📊 Usage Stats")
                    for tech, count in technique_stats.head(5).items():
                        st.write(f"• {tech}: {count}")

    # ------------------ Evaluation Methods Section ------------------
    elif selected == "🔍 Evaluation Methods":
        st.markdown("## 🔍 Advanced Evaluation Methods")
        
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            st.markdown('<div class="technique-card">', unsafe_allow_html=True)
            
            evaluation_method = st.selectbox(
                "🔬 Select Evaluation Method",
                [
                    "Chain of Verification (CoVe)",
                    "Prompt Quality Assessment",
                    "Response Comparison"
                ]
            )
            
            if evaluation_method == "Chain of Verification (CoVe)":
                st.markdown("**🔗 Chain of Verification for Fact-Checking**")
                
                # Option to use existing result or input new content
                use_existing = st.checkbox("Use existing result from session", value=False)
                
                if use_existing and st.session_state.prompt_results:
                    result_options = [f"{r.technique} - {r.prompt[:50]}..." for r in st.session_state.prompt_results]
                    selected_result_idx = st.selectbox("Select result to verify:", range(len(result_options)), format_func=lambda x: result_options[x])
                    selected_result = st.session_state.prompt_results[selected_result_idx]
                    
                    original_prompt = selected_result.prompt
                    original_response = selected_result.output
                    
                    st.text_area("Original Prompt:", value=original_prompt, height=100, disabled=True)
                    st.text_area("Original Response:", value=original_response, height=150, disabled=True)
                else:
                    original_prompt = st.text_area("Original Prompt:", height=100)
                    original_response = st.text_area("Original Response to Verify:", height=150)
                
                if st.button("🔍 Run Chain of Verification", use_container_width=True):
                    if original_prompt and original_response:
                        with st.spinner("🔄 Running Chain of Verification..."):
                            try:
                                cove_prompt = EvaluationMethods.chain_of_verification(original_response, original_prompt)
                                
                                result = st.session_state.langfuse_tracker.run_prompt_with_tracing(
                                    cove_prompt,
                                    "chain_of_verification",
                                    original_prompt=original_prompt,
                                    original_response=original_response,
                                    evaluation_method="CoVe"
                                )
                                
                                st.session_state.evaluation_results.append(result)
                                
                                # Display verification results
                                st.markdown("### ✅ Verification Results:")
                                formatted_output = format_reasoning_output(result.output, "Chain of Verification")
                                st.markdown(f'<div class="evaluation-result">{formatted_output}</div>', 
                                          unsafe_allow_html=True)
                                
                                # Extract confidence score if available
                                confidence_match = re.search(r'Confidence Score:.*?(\d+)', result.output)
                                if confidence_match:
                                    confidence = int(confidence_match.group(1))
                                    st.metric("Confidence Score", f"{confidence}/10")
                                
                            except Exception as e:
                                st.error(f"❌ Verification error: {str(e)}")
                    else:
                        st.warning("⚠️ Please provide both original prompt and response.")
            
            elif evaluation_method == "Prompt Quality Assessment":
                st.markdown("**📊 Prompt Quality Evaluation**")
                
                prompt_to_assess = st.text_area("Prompt to Assess:", height=150)
                
                if st.button("📊 Assess Prompt Quality", use_container_width=True):
                    if prompt_to_assess:
                        with st.spinner("📊 Analyzing prompt quality..."):
                            try:
                                assessment_prompt = EvaluationMethods.prompt_quality_assessment(prompt_to_assess)
                                
                                result = st.session_state.langfuse_tracker.run_prompt_with_tracing(
                                    assessment_prompt,
                                    "prompt_quality_assessment",
                                    assessed_prompt=prompt_to_assess,
                                    evaluation_method="Quality Assessment"
                                )
                                
                                st.session_state.evaluation_results.append(result)
                                
                                st.markdown("### 📊 Quality Assessment:")
                                st.markdown(f'<div class="evaluation-result">{result.output}</div>', 
                                          unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"❌ Assessment error: {str(e)}")
                    else:
                        st.warning("⚠️ Please provide a prompt to assess.")
            
            elif evaluation_method == "Response Comparison":
                st.markdown("**⚖️ Compare Multiple Responses**")
                
                if len(st.session_state.prompt_results) >= 2:
                    st.markdown("Select two results to compare:")
                    
                    result_options = [f"{r.technique} - {r.prompt[:50]}..." for r in st.session_state.prompt_results]
                    
                    col_comp1, col_comp2 = st.columns(2)
                    with col_comp1:
                        result1_idx = st.selectbox("First Result:", range(len(result_options)), format_func=lambda x: result_options[x])
                    with col_comp2:
                        result2_idx = st.selectbox("Second Result:", range(len(result_options)), format_func=lambda x: result_options[x])
                    
                    if result1_idx != result2_idx:
                        result1 = st.session_state.prompt_results[result1_idx]
                        result2 = st.session_state.prompt_results[result2_idx]
                        
                        if st.button("⚖️ Compare Responses", use_container_width=True):
                            st.markdown("### ⚖️ Response Comparison:")
                            
                            col_r1, col_r2 = st.columns(2)
                            
                            with col_r1:
                                st.markdown(f"**{result1.technique}**")
                                st.markdown(f'<div class="response-container">{result1.output[:500]}...</div>', unsafe_allow_html=True)
                                st.metric("Length", result1.length)
                                st.metric("Time", f"{result1.time_taken}s")
                            
                            with col_r2:
                                st.markdown(f"**{result2.technique}**")
                                st.markdown(f'<div class="response-container">{result2.output[:500]}...</div>', unsafe_allow_html=True)
                                st.metric("Length", result2.length)
                                st.metric("Time", f"{result2.time_taken}s")
                    else:
                        st.warning("⚠️ Please select two different results to compare.")
                else:
                    st.info("💡 You need at least 2 results in your session to compare responses. Try some prompting techniques first!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔬 Evaluation Guide")
            
            evaluation_descriptions = {
                "Chain of Verification (CoVe)": "Systematic fact-checking by generating verification questions and cross-checking claims.",
                "Prompt Quality Assessment": "Evaluate prompts across multiple criteria: clarity, specificity, context, structure.",
                "Response Comparison": "Side-by-side comparison of different responses to identify strengths and weaknesses."
            }
            
            if evaluation_method in evaluation_descriptions:
                st.info(evaluation_descriptions[evaluation_method])

    # ------------------ Batch Processing Section ------------------
    elif selected == "⚡ Batch Processing":
        st.markdown("## ⚡ Parallel Batch Processing")
        
        st.markdown("""
        <div class="info-box">
            <h4>🚀 High-Performance Concurrent Processing</h4>
            <p>Process multiple prompts simultaneously using advanced techniques with real-time progress tracking.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            st.markdown('<div class="technique-card">', unsafe_allow_html=True)
            
            # Batch input method
            batch_method = st.radio(
                "📝 Input Method:",
                ["Manual Entry", "Upload File", "Template Generator"]
            )
            
            prompts_to_process = []
            
            if batch_method == "Manual Entry":
                batch_text = st.text_area(
                    "Enter prompts (one per line):",
                    height=200,
                    placeholder="Write a summary of artificial intelligence\nExplain quantum computing basics\nCreate a business plan outline\nAnalyze market trends in tech"
                )
                prompts_to_process = [p.strip() for p in batch_text.split('\n') if p.strip()]
            
            elif batch_method == "Upload File":
                uploaded_file = st.file_uploader("📁 Upload text file", type=['txt'], 
                                                help="Upload a .txt file with one prompt per line")
                if uploaded_file:
                    content = str(uploaded_file.read(), "utf-8")
                    prompts_to_process = [p.strip() for p in content.split('\n') if p.strip()]
                    st.success(f"✅ Loaded {len(prompts_to_process)} prompts from file")
            
            elif batch_method == "Template Generator":
                template_category = st.selectbox(
                    "🎯 Template Category:",
                    ["Business Analysis", "Educational Content", "Creative Writing", "Technical Reviews"]
                )
                
                templates = {
                    "Business Analysis": [
                        "Analyze market opportunities for {industry}",
                        "Create competitive analysis for {company_type}",
                        "Develop pricing strategy for {product}",
                        "Assess market risks in {sector}"
                    ],
                    "Educational Content": [
                        "Explain {concept} to beginners",
                        "Create study guide for {subject}",
                        "Compare {topic1} vs {topic2}",
                        "List key principles of {field}"
                    ],
                    "Creative Writing": [
                        "Write a story about {theme}",
                        "Create character description for {character_type}",
                        "Develop plot outline for {genre}",
                        "Write dialogue between {characters}"
                    ],
                    "Technical Reviews": [
                        "Review code quality for {language}",
                        "Analyze system architecture for {system_type}",
                        "Evaluate security for {application}",
                        "Optimize performance for {platform}"
                    ]
                }
                
                if template_category in templates:
                    template_list = templates[template_category]
                    
                    # Extract unique variables
                    all_vars = set()
                    for template in template_list:
                        import re
                        vars_found = re.findall(r'{(\w+)}', template)
                        all_vars.update(vars_found)
                    
                    st.markdown("**📋 Fill Template Variables:**")
                    variables = {}
                    for var in sorted(all_vars):
                        variables[var] = st.text_input(f"{var.replace('_', ' ').title()}:", key=f"batch_var_{var}")
                    
                    if st.button("🎯 Generate Batch Prompts"):
                        if all(variables.values()):
                            prompts_to_process = []
                            for template in template_list:
                                try:
                                    formatted_prompt = template.format(**variables)
                                    prompts_to_process.append(formatted_prompt)
                                except KeyError as e:
                                    st.warning(f"Missing variable: {e}")
                            
                            if prompts_to_process:
                                st.success(f"✅ Generated {len(prompts_to_process)} prompts")
                        else:
                            st.warning("⚠️ Please fill all template variables")
            
            # Batch processing configuration
            if prompts_to_process:
                st.markdown(f"### 🎯 Batch Configuration ({len(prompts_to_process)} prompts)")
                
                col_config1, col_config2 = st.columns(2)
                with col_config1:
                    technique_for_batch = st.selectbox(
                        "🧠 Apply Technique to All:",
                        ["Zero-shot", "Chain of Thought", "ReAct", "Role-based", "Instruction-based"]
                    )
                    max_workers = st.slider("🔧 Concurrent Workers", 1, 8, min(4, len(prompts_to_process)))
                
                with col_config2:
                    role_for_batch = st.selectbox(
                        "👤 Role (if Role-based selected):",
                        ["business_analyst", "data_scientist", "teacher", "consultant"]
                    ) if technique_for_batch == "Role-based" else None
                    
                    timeout_seconds = st.slider("⏱️ Timeout per Prompt (seconds)", 30, 180, 60)
                
                if st.button("🚀 Start Batch Processing", use_container_width=True):
                    
                    def process_batch_prompt(prompt_data):
                        index, prompt = prompt_data
                        try:
                            # Apply selected technique
                            if technique_for_batch == "Zero-shot":
                                final_prompt = PromptingTechniques.zero_shot(prompt)
                            elif technique_for_batch == "Chain of Thought":
                                final_prompt = PromptingTechniques.chain_of_thought(prompt)
                            elif technique_for_batch == "ReAct":
                                final_prompt = PromptingTechniques.react_prompting(prompt, 3)
                            elif technique_for_batch == "Role-based":
                                final_prompt = PromptingTechniques.role_based_prompting(prompt, role_for_batch)
                            elif technique_for_batch == "Instruction-based":
                                final_prompt = PromptingTechniques.instruction_based(prompt, ["Be comprehensive", "Use examples", "Be clear"])
                            else:
                                final_prompt = prompt
                            
                            result = st.session_state.langfuse_tracker.run_prompt_with_tracing(
                                final_prompt,
                                f"batch_{technique_for_batch.lower()}",
                                batch_index=index,
                                batch_technique=technique_for_batch
                            )
                            result.metadata = result.metadata or {}
                            result.metadata['batch_index'] = index
                            result.metadata['batch_technique'] = technique_for_batch
                            return result
                            
                        except Exception as e:
                            return PromptResult(
                                prompt=prompt,
                                output=f"Error: {str(e)}",
                                technique=f"batch_{technique_for_batch.lower()}",
                                length=0,
                                input_tokens=len(prompt.split()),
                                output_tokens=0,
                                time_taken=0,
                                trace_id=None,
                                timestamp=datetime.now(),
                                metadata={'error': True, 'batch_index': index}
                            )
                    
                    # Execute batch processing
                    start_time = time.time()
                    results = []
                    
                    progress_bar = st.progress(0)
                    status_container = st.empty()
                    results_container = st.container()
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all tasks
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
                                
                                # Update progress
                                progress = completed / len(prompts_to_process)
                                progress_bar.progress(progress)
                                status_container.text(f"✅ Completed: {completed}/{len(prompts_to_process)}")
                                
                            except Exception as e:
                                st.error(f"❌ Task failed: {str(e)}")
                    
                    total_time = time.time() - start_time
                    
                    # Display batch results
                    with results_container:
                        st.markdown("### 🎯 Batch Processing Results")
                        
                        # Summary metrics
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        col_s1.metric("Total Time", f"{total_time:.2f}s")
                        col_s2.metric("Avg per Prompt", f"{total_time/len(results):.2f}s")
                        col_s3.metric("Success Rate", f"{len([r for r in results if not r.metadata.get('error', False)])}/{len(results)}")
                        col_s4.metric("Total Tokens", sum([r.input_tokens + r.output_tokens for r in results]))
                        
                        # Individual results in expandable sections
                        for result in sorted(results, key=lambda x: x.metadata.get('batch_index', 0)):
                            if not result.metadata.get('error', False):
                                with st.expander(f"📝 Prompt {result.metadata.get('batch_index', 0) + 1}: {result.prompt[:60]}..."):
                                    st.markdown(f"**Response:**\n{result.output}")
                                    
                                    col_r1, col_r2, col_r3 = st.columns(3)
                                    col_r1.metric("Time", f"{result.time_taken:.2f}s")
                                    col_r2.metric("Length", result.length)
                                    col_r3.metric("Tokens", f"{result.input_tokens + result.output_tokens}")
                                    
                                    if result.trace_id:
                                        st.markdown(f"🔗 [View Trace](https://cloud.langfuse.com/traces/{result.trace_id})")
                        
                        # Batch download
                        if results:
                            combined_results = "\n\n" + "="*80 + "\n\n".join([
                                f"PROMPT {r.metadata.get('batch_index', 0)+1}:\n{r.prompt}\n\nTECHNIQUE: {r.technique}\n\nRESPONSE:\n{r.output}"
                                for r in sorted(results, key=lambda x: x.metadata.get('batch_index', 0))
                            ])
                            
                            st.markdown(
                                create_download_link(
                                    combined_results,
                                    f"batch_results_{technique_for_batch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                ),
                                unsafe_allow_html=True
                            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ⚡ Batch Processing Guide")
            
            st.info("""
            **🎯 Key Features:**
            - Concurrent processing up to 8 prompts
            - Apply any technique to entire batch
            - Real-time progress tracking
            - Automatic error handling
            - Comprehensive results export
            """)
            
            if prompts_to_process:
                st.markdown("### 📊 Batch Preview")
                st.write(f"**Total Prompts:** {len(prompts_to_process)}")
                st.write(f"**Estimated Time:** {len(prompts_to_process) * 3:.0f}-{len(prompts_to_process) * 8:.0f}s")
                
                st.markdown("**Sample Prompts:**")
                for i, prompt in enumerate(prompts_to_process[:3]):
                    st.write(f"{i+1}. {prompt[:50]}...")
                if len(prompts_to_process) > 3:
                    st.write(f"... and {len(prompts_to_process) - 3} more")

    # ------------------ Vision Analysis Section ------------------
    elif selected == "🖼️ Vision Analysis":
        st.markdown("## 🖼️ Advanced Vision Analysis")
        
        st.markdown("""
        <div class="info-box">
            <h4>🎨 Multimodal AI with Advanced Vision Techniques</h4>
            <p>Upload images and apply sophisticated analysis techniques with detailed insights and downloadable reports.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            st.markdown('<div class="technique-card">', unsafe_allow_html=True)
            
            # Image upload
            uploaded_images = st.file_uploader(
                "📷 Upload Images",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
                accept_multiple_files=True,
                help="Upload one or more images for analysis"
            )
            
            if uploaded_images:
                st.markdown(f"### 📸 Uploaded Images ({len(uploaded_images)})")
                
                # Display thumbnails
                cols = st.columns(min(3, len(uploaded_images)))
                for i, img_file in enumerate(uploaded_images[:3]):
                    with cols[i % 3]:
                        image = Image.open(img_file)
                        st.image(image, caption=f"Image {i+1}", use_column_width=True)
                
                if len(uploaded_images) > 3:
                    st.info(f"📁 {len(uploaded_images) - 3} more images uploaded")
            
            # Analysis configuration
            st.markdown("### 🔧 Analysis Configuration")
            
            analysis_technique = st.selectbox(
                "🧠 Vision Analysis Technique:",
                [
                    "Standard Description",
                    "Chain of Thought Analysis", 
                    "Role-based Expert Analysis",
                    "Detailed Technical Assessment",
                    "Creative Interpretation",
                    "Comparative Analysis"
                ]
            )
            
            analysis_focus = st.multiselect(
                "🎯 Analysis Focus Areas:",
                [
                    "Objects & People",
                    "Colors & Composition", 
                    "Text Extraction (OCR)",
                    "Emotional Tone",
                    "Technical Quality",
                    "Artistic Elements",
                    "Business Applications",
                    "Accessibility Description"
                ],
                default=["Objects & People", "Colors & Composition"]
            )
            
            # Custom prompt
            custom_prompt = st.text_area(
                "✏️ Additional Instructions (Optional):",
                height=80,
                placeholder="Any specific questions or aspects you want the AI to focus on..."
            )
            
            # Analysis settings
            col_set1, col_set2 = st.columns(2)
            with col_set1:
                detail_level = st.select_slider(
                    "📊 Detail Level:",
                    options=["Brief", "Standard", "Comprehensive"],
                    value="Standard"
                )
            
            with col_set2:
                if analysis_technique == "Role-based Expert Analysis":
                    expert_role = st.selectbox(
                        "👤 Expert Role:",
                        ["photographer", "designer", "marketer", "security_analyst", "art_critic", "accessibility_expert"]
                    )
            
            # Process images
            if uploaded_images and st.button("🔍 Analyze Images", use_container_width=True):
                
                def create_vision_prompt(technique, focus_areas, custom_instructions, detail_level, role=None):
                    base_analysis = {
                        "Standard Description": "Provide a comprehensive description of this image.",
                        "Chain of Thought Analysis": "Analyze this image step-by-step: 1) First impression 2) Detailed observation 3) Context analysis 4) Final assessment",
                        "Role-based Expert Analysis": f"As a professional {role}, analyze this image from your expert perspective.",
                        "Detailed Technical Assessment": "Provide a technical analysis including image quality, composition, lighting, and photographic techniques.",
                        "Creative Interpretation": "Provide a creative interpretation including mood, story, symbolism, and artistic elements.",
                        "Comparative Analysis": "Analyze this image by comparing and contrasting its elements, identifying patterns and relationships."
                    }
                    
                    focus_instructions = {
                        "Objects & People": "Identify and describe all objects, people, and their relationships.",
                        "Colors & Composition": "Analyze color palette, composition rules, visual balance, and aesthetic elements.",
                        "Text Extraction (OCR)": "Extract and transcribe all visible text, signs, and written content.",
                        "Emotional Tone": "Assess the emotional impact, mood, and psychological elements conveyed.",
                        "Technical Quality": "Evaluate image quality, resolution, lighting, focus, and technical aspects.",
                        "Artistic Elements": "Analyze artistic techniques, style, visual storytelling, and creative elements.",
                        "Business Applications": "Identify potential business uses, marketing value, and commercial applications.",
                        "Accessibility Description": "Create detailed accessibility descriptions for visually impaired users."
                    }
                    
                    prompt = base_analysis[technique] + "\n\n"
                    
                    if focus_areas:
                        prompt += "Focus specifically on:\n"
                        for area in focus_areas:
                            prompt += f"- {focus_instructions[area]}\n"
                        prompt += "\n"
                    
                    if custom_instructions:
                        prompt += f"Additional requirements: {custom_instructions}\n\n"
                    
                    prompt += f"Provide a {detail_level.lower()} analysis with clear structure and actionable insights."
                    
                    return prompt
                
                # Process each image
                for i, img_file in enumerate(uploaded_images):
                    st.markdown(f"### 📷 Analysis {i+1}: {img_file.name}")
                    
                    with st.spinner(f"🔄 Analyzing image {i+1}..."):
                        try:
                            # Load image
                            image = Image.open(img_file)
                            
                            # Create analysis prompt
                            vision_prompt = create_vision_prompt(
                                analysis_technique, 
                                analysis_focus, 
                                custom_prompt, 
                                detail_level,
                                expert_role if analysis_technique == "Role-based Expert Analysis" else None
                            )
                            
                            # Use Gemini Vision
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            response = model.generate_content([vision_prompt, image])
                            
                            # Create result
                            result = PromptResult(
                                prompt=vision_prompt,
                                output=response.text,
                                technique=f"vision_{analysis_technique.lower().replace(' ', '_')}",
                                length=len(response.text),
                                input_tokens=len(vision_prompt.split()),
                                output_tokens=len(response.text.split()),
                                time_taken=0,  # Gemini doesn't provide timing
                                trace_id=None,
                                timestamp=datetime.now(),
                                metadata={
                                    'image_name': img_file.name,
                                    'analysis_technique': analysis_technique,
                                    'focus_areas': analysis_focus,
                                    'detail_level': detail_level,
                                    'image_size': f"{image.size[0]}×{image.size[1]}",
                                    'image_format': image.format
                                }
                            )
                            
                            # Log to Langfuse if configured
                            if st.session_state.langfuse_tracker.is_configured:
                                try:
                                    result.trace_id = st.session_state.langfuse_tracker.langfuse.trace(
                                        name=f"vision_analysis_{i+1}",
                                        session_id=st.session_state.langfuse_tracker.session_id,
                                        input={"prompt": vision_prompt, "image": img_file.name},
                                        output={"response": response.text},
                                        metadata=result.metadata
                                    ).id
                                except:
                                    pass
                            
                            st.session_state.prompt_results.append(result)
                            
                            # Display results
                            st.markdown(f'<div class="response-container">{response.text}</div>', unsafe_allow_html=True)
                            
                            # Image metadata
                            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                            col_info1.metric("Size", f"{image.size[0]}×{image.size[1]}")
                            col_info2.metric("Format", image.format)
                            col_info3.metric("Mode", image.mode)
                            col_info4.metric("Response Length", len(response.text))
                            
                            # Download option
                            st.markdown(
                                create_download_link(
                                    f"Image: {img_file.name}\nTechnique: {analysis_technique}\nFocus: {', '.join(analysis_focus)}\n\nAnalysis:\n{response.text}",
                                    f"vision_analysis_{img_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                ),
                                unsafe_allow_html=True
                            )
                            
                            if result.trace_id:
                                st.markdown(f"🔗 [View Trace](https://cloud.langfuse.com/traces/{result.trace_id})")
                            
                        except Exception as e:
                            st.error(f"❌ Error analyzing image {i+1}: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎨 Vision Analysis Guide")
            
            technique_guide = {
                "Standard Description": "Comprehensive overview of image content and context.",
                "Chain of Thought Analysis": "Step-by-step systematic analysis with reasoning.",
                "Role-based Expert Analysis": "Professional perspective from domain experts.",
                "Detailed Technical Assessment": "Technical evaluation of image quality and composition.",
                "Creative Interpretation": "Artistic and creative analysis of visual elements.",
                "Comparative Analysis": "Comparative examination of elements and relationships."
            }
            
            if 'analysis_technique' in locals():
                st.info(technique_guide.get(analysis_technique, "Advanced vision analysis technique"))
            
            st.markdown("### 📊 Supported Formats")
            st.write("• PNG, JPG, JPEG")
            st.write("• GIF, WebP")
            st.write("• Max size: 20MB")
            st.write("• Multiple images supported")
            
            if uploaded_images:
                st.markdown("### 📁 Upload Summary")
                total_size = sum([img.size for img in uploaded_images]) / (1024*1024)  # MB
                st.write(f"**Images:** {len(uploaded_images)}")
                st.write(f"**Total Size:** {total_size:.1f} MB")
                
                for i, img in enumerate(uploaded_images[:3]):
                    img_obj = Image.open(img)
                    st.write(f"{i+1}. {img.name} ({img_obj.size[0]}×{img_obj.size[1]})")
    
    elif selected == "📊 Analytics Dashboard":
        st.markdown("## 📊 Comprehensive Analytics Dashboard")
        
        if not st.session_state.prompt_results:
            st.markdown("""
            <div class="info-box">
                <h4>📈 No Data Available</h4>
                <p>Use the prompting techniques and evaluation methods to generate analytics data!</p>
            </div>
            """, unsafe_allow_html=True)
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
            
            # Enhanced Overview metrics with trends
            st.markdown("### 📈 Session Overview")
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            
            # Calculate trends (last 5 vs previous)
            recent_results = df.tail(5) if len(df) >= 5 else df
            previous_results = df.head(len(df)-5) if len(df) > 5 else pd.DataFrame()
            
            avg_time_trend = recent_results['time_taken'].mean() - previous_results['time_taken'].mean() if len(previous_results) > 0 else 0
            avg_length_trend = recent_results['length'].mean() - previous_results['length'].mean() if len(previous_results) > 0 else 0
            
            col_m1.metric(
                "Total Prompts", 
                len(df),
                delta=f"+{len(recent_results)}" if len(recent_results) > 0 else None
            )
            col_m2.metric(
                "Avg Response Time", 
                f"{df['time_taken'].mean():.2f}s",
                delta=f"{avg_time_trend:+.2f}s" if avg_time_trend != 0 else None
            )
            col_m3.metric(
                "Total Tokens", 
                f"{df['total_tokens'].sum():,}",
                delta=f"+{recent_results['total_tokens'].sum()}"
            )
            col_m4.metric(
                "Traced Prompts", 
                df['has_trace'].sum(),
                delta=f"{df['has_trace'].sum()/len(df)*100:.0f}% traced"
            )
            col_m5.metric(
                "Avg Efficiency", 
                f"{df['efficiency'].mean():.0f} chars/s",
                delta=f"{avg_length_trend:+.0f} chars" if avg_length_trend != 0 else None
            )
            
            # Advanced Visualizations
            st.markdown("### 📊 Performance Analytics")
            
            # Create comprehensive dashboard with subplots
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            
            # Time series analysis
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Response time over time
                fig_time = px.line(
                    df, 
                    x='timestamp', 
                    y='time_taken',
                    color='technique',
                    title='⏱️ Response Time Trends',
                    labels={'time_taken': 'Response Time (s)', 'timestamp': 'Time'}
                )
                fig_time.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col_chart2:
                # Token usage scatter
                fig_tokens = px.scatter(
                    df, 
                    x='input_tokens', 
                    y='output_tokens',
                    size='time_taken',
                    color='technique',
                    title='🎯 Token Usage Pattern',
                    labels={'input_tokens': 'Input Tokens', 'output_tokens': 'Output Tokens'},
                    hover_data=['length', 'time_taken']
                )
                fig_tokens.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig_tokens, use_container_width=True)
            
            # Technique analysis
            col_tech1, col_tech2 = st.columns(2)
            
            with col_tech1:
                # Technique usage pie chart
                technique_counts = df['technique'].value_counts()
                fig_pie = px.pie(
                    values=technique_counts.values,
                    names=technique_counts.index,
                    title='🧠 Technique Usage Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_tech2:
                # Technique performance comparison
                technique_stats = df.groupby('technique').agg({
                    'time_taken': 'mean',
                    'length': 'mean',
                    'total_tokens': 'mean'
                }).reset_index()
                
                fig_comparison = px.bar(
                    technique_stats,
                    x='technique',
                    y='time_taken',
                    title='⚡ Avg Response Time by Technique',
                    labels={'time_taken': 'Avg Response Time (s)', 'technique': 'Technique'},
                    color='time_taken',
                    color_continuous_scale='viridis'
                )
                fig_comparison.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    xaxis={'tickangle': 45}
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Advanced analytics
            st.markdown("### 🔍 Advanced Analytics")
            
            col_adv1, col_adv2, col_adv3 = st.columns(3)
            
            with col_adv1:
                # Efficiency analysis
                fig_efficiency = px.box(
                    df,
                    x='technique',
                    y='efficiency',
                    title='📈 Efficiency Distribution (chars/sec)',
                    color='technique'
                )
                fig_efficiency.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    xaxis={'tickangle': 45}
                )
                st.plotly_chart(fig_efficiency, use_container_width=True)
            
            with col_adv2:
                # Hourly activity
                hourly_activity = df.groupby('hour').size()
                fig_hourly = px.bar(
                    x=hourly_activity.index,
                    y=hourly_activity.values,
                    title='🕐 Activity by Hour',
                    labels={'x': 'Hour of Day', 'y': 'Number of Prompts'},
                    color=hourly_activity.values,
                    color_continuous_scale='blues'
                )
                fig_hourly.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with col_adv3:
                # Length vs Time correlation
                fig_correlation = px.scatter(
                    df,
                    x='length',
                    y='time_taken',
                    trendline='ols',
                    title='🔗 Length vs Response Time',
                    labels={'length': 'Response Length', 'time_taken': 'Time (s)'},
                    color='technique'
                )
                fig_correlation.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300
                )
                st.plotly_chart(fig_correlation, use_container_width=True)
            
            # Comprehensive performance matrix
            st.markdown("### 📋 Performance Matrix")
            
            performance_matrix = df.groupby('technique').agg({
                'time_taken': ['mean', 'std', 'min', 'max'],
                'length': ['mean', 'std'],
                'total_tokens': ['mean', 'sum'],
                'has_trace': 'sum'
            }).round(2)
            
            performance_matrix.columns = [
                'Avg Time', 'Time StdDev', 'Min Time', 'Max Time',
                'Avg Length', 'Length StdDev', 'Avg Tokens', 'Total Tokens', 'Traces'
            ]
            
            st.dataframe(performance_matrix, use_container_width=True)
            
            # Detailed results with advanced filtering
            st.markdown("### 🔍 Detailed Results")
            
            # Advanced filters
            col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
            
            with col_filter1:
                technique_filter = st.multiselect(
                    "Filter by Technique:",
                    options=df['technique'].unique(),
                    default=df['technique'].unique()
                )
            
            with col_filter2:
                min_time = st.number_input("Min Response Time (s):", 0.0, float(df['time_taken'].max()), 0.0)
                max_time = st.number_input("Max Response Time (s):", 0.0, float(df['time_taken'].max()), float(df['time_taken'].max()))
            
            with col_filter3:
                min_length = st.number_input("Min Length:", 0, int(df['length'].max()), 0)
                max_length = st.number_input("Max Length:", 0, int(df['length'].max()), int(df['length'].max()))
            
            with col_filter4:
                show_traced_only = st.checkbox("Show Only Traced")
                sort_by = st.selectbox("Sort by:", ['timestamp', 'time_taken', 'length', 'total_tokens'])
            
            # Apply filters
            filtered_df = df[
                (df['technique'].isin(technique_filter)) &
                (df['time_taken'] >= min_time) &
                (df['time_taken'] <= max_time) &
                (df['length'] >= min_length) &
                (df['length'] <= max_length)
            ]
            
            if show_traced_only:
                filtered_df = filtered_df[filtered_df['has_trace'] == True]
            
            filtered_df = filtered_df.sort_values(sort_by, ascending=False)
            
            # Enhanced display columns
            display_columns = [
                'timestamp', 'technique', 'prompt', 'time_taken', 
                'length', 'input_tokens', 'output_tokens', 'total_tokens', 'trace_id'
            ]
            
            st.dataframe(
                filtered_df[display_columns],
                use_container_width=True,
                column_config={
                    'timestamp': st.column_config.DatetimeColumn('Time', format='MM/DD/YY HH:mm'),
                    'technique': 'Technique',
                    'prompt': st.column_config.TextColumn('Prompt', width='large'),
                    'time_taken': st.column_config.NumberColumn('Time (s)', format='%.2f'),
                    'length': st.column_config.NumberColumn('Length'),
                    'input_tokens': st.column_config.NumberColumn('Input Tokens'),
                    'output_tokens': st.column_config.NumberColumn('Output Tokens'),
                    'total_tokens': st.column_config.NumberColumn('Total Tokens'),
                    'trace_id': st.column_config.LinkColumn(
                        'Trace',
                        display_text="View Trace",
                        help="Click to view in Langfuse"
                    )
                }
            )
            
            # Export options
            st.markdown("### 💾 Export Analytics")
            col_export1, col_export2, col_export3, col_export4 = st.columns(4)
            
            with col_export1:
                if st.button("📊 Export Full Dataset"):
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        f"gemini_analytics_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
            
            with col_export2:
                if st.button("📈 Export Performance Summary"):
                    summary_data = performance_matrix.to_csv()
                    st.download_button(
                        "Download Summary",
                        summary_data,
                        f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
            
            with col_export3:
                if st.button("🎯 Export Filtered Results"):
                    filtered_csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        "Download Filtered",
                        filtered_csv,
                        f"filtered_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
            
            with col_export4:
                if st.button("🗑️ Clear Session Data"):
                    if st.button("⚠️ Confirm Clear", type="primary"):
                        st.session_state.prompt_results = []
                        st.session_state.evaluation_results = []
                        st.rerun()

else:
    st.warning("⚠️ Please configure your Google API key in the sidebar to get started.")
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. **Enter Key**: Paste your API key in the sidebar
    3. **Optional**: Configure Langfuse for advanced tracing
    4. **Start Experimenting**: Use advanced prompting techniques!
    
    ### ✨ Advanced Features Available:
    
    #### 🎯 **Prompting Techniques:**
    - **Zero-shot**: Direct instruction without examples
    - **Few-shot**: Learning from provided examples  
    - **Chain of Thought**: Step-by-step reasoning
    - **ReAct**: Reasoning and Acting framework
    - **Tree of Thoughts**: Multiple reasoning paths
    - **Role-based**: Expert perspective prompting
    - **Instruction-based**: Detailed guideline following
    - **Parallel**: Multiple perspective analysis
    
    #### 🔍 **Evaluation Methods:**
    - **Chain of Verification**: Systematic fact-checking
    - **Prompt Quality Assessment**: Comprehensive prompt evaluation
    - **Response Comparison**: Side-by-side analysis
    """)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 Advanced Gemini AI Assistant | Professional Prompting & Evaluation Platform</p>
    <p>Built with Streamlit, Google AI, LangChain & Langfuse</p>
</div>
""", unsafe_allow_html=True)