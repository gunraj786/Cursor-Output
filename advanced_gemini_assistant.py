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
import asyncio
from dataclasses import dataclass

# Enhanced Langfuse Integration
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    # Create dummy decorator when Langfuse is not available
    def observe():
        def decorator(func):
            return func
        return decorator

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
    
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
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
        if LANGFUSE_AVAILABLE:
            try:
                os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
                os.environ["LANGFUSE_SECRET_KEY"] = secret_key
                os.environ["LANGFUSE_HOST"] = host
                
                self.langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                
                # Test connection
                self.langfuse.auth_check()
                self.is_configured = True
                return True
            except Exception as e:
                st.error(f"Langfuse configuration failed: {e}")
                return False
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
            if self.is_configured and self.langfuse:
                try:
                    trace = self.langfuse.trace(
                        name=f"{technique}_execution",
                        session_id=self.session_id,
                        input={"prompt": prompt_text, "technique": technique, "model": model_name},
                        output={"response": result_text},
                        metadata={
                            "time_taken": time_taken,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "technique": technique,
                            **kwargs
                        }
                    )
                    
                    # Create generation within trace
                    generation = trace.generation(
                        name=f"gemini_{technique}",
                        model=model_name,
                        input=prompt_text,
                        output=result_text,
                        usage={
                            "input": input_tokens,
                            "output": output_tokens,
                            "total": input_tokens + output_tokens
                        },
                        metadata={"technique": technique}
                    )
                    
                    trace_id = trace.id
                except Exception as langfuse_error:
                    print(f"Langfuse tracing error: {langfuse_error}")
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
            
            if self.is_configured and self.langfuse:
                try:
                    self.langfuse.trace(
                        name=f"{technique}_error",
                        session_id=self.session_id,
                        input={"prompt": prompt_text},
                        output={"error": str(e)},
                        metadata={"technique": technique}
                    )
                except Exception as langfuse_error:
                    print(f"Langfuse error logging failed: {langfuse_error}")
            
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
            "business_analyst": f"You are a senior business analyst with 10+ years of experience in strategic planning and market analysis.",
            "data_scientist": f"You are an expert data scientist with deep knowledge in machine learning, statistics, and data analysis.",
            "software_engineer": f"You are a senior software engineer with expertise in system design, coding best practices, and architecture.",
            "marketing_expert": f"You are a marketing professional with extensive experience in digital marketing, brand strategy, and customer acquisition.",
            "financial_advisor": f"You are a certified financial advisor with expertise in investment strategies, risk management, and financial planning.",
            "teacher": f"You are an experienced educator skilled at explaining complex concepts in simple, understandable terms.",
            "consultant": f"You are a management consultant with experience helping organizations solve complex business problems.",
            "researcher": f"You are an academic researcher with expertise in conducting thorough analysis and presenting findings."
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
...

STEP 2 - GENERATE VERIFICATION QUESTIONS:
For each claim, create a verification question:
1. [Verification question for claim 1]
2. [Verification question for claim 2]
3. [Verification question for claim 3]
...

STEP 3 - ANSWER VERIFICATION QUESTIONS:
Answer each verification question independently:
1. [Answer to verification question 1]
2. [Answer to verification question 2]
3. [Answer to verification question 3]
...

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
    def self_consistency_check(prompt: str, num_samples: int = 3) -> str:
        """Self-consistency evaluation across multiple reasoning paths"""
        return f"""Task: {prompt}

Generate {num_samples} independent reasoning paths for this problem:

REASONING PATH 1:
[Complete independent reasoning and answer]

REASONING PATH 2:
[Complete independent reasoning and answer]

REASONING PATH 3:
[Complete independent reasoning and answer]

CONSISTENCY ANALYSIS:
- Common elements across all paths: [List similarities]
- Differences between paths: [List differences]
- Most reliable elements: [Identify most consistent parts]
- Areas of uncertainty: [Identify inconsistent parts]

FINAL ANSWER:
[Synthesized answer based on most consistent elements]

Confidence Level: [High/Medium/Low based on consistency]"""

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
    if technique == "ReAct":
        formatted = output_text.replace("ITERATION", '<div class="reasoning-step"><strong>🔄 ITERATION')
        formatted = formatted.replace("Thought:", '</strong><br><strong>💭 Thought:</strong>')
        formatted = formatted.replace("Action:", '<br><strong>⚡ Action:</strong>')
        formatted = formatted.replace("Observation:", '<br><strong>👁️ Observation:</strong>')
        formatted = formatted.replace("Assessment:", '<br><strong>📊 Assessment:</strong>')
        formatted = formatted.replace("FINAL ANSWER:", '</div><div class="evaluation-result"><strong>✨ FINAL ANSWER:</strong>')
        formatted += '</div>'
    elif technique == "Tree of Thoughts":
        formatted = output_text.replace("BRANCH", '<div class="tree-node"><strong>🌳 BRANCH')
        formatted = formatted.replace("SYNTHESIS:", '</div><div class="evaluation-result"><strong>🔗 SYNTHESIS:</strong>')
        formatted += '</div>'
    elif technique == "Chain of Verification":
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
            Install with: <code>pip install langfuse</code>
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
            for technique, count in technique_counts.items():
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
            "📊 Analytics Dashboard",
            "🧪 Prompt Testing Lab"
        ],
        icons=["target", "search", "lightning", "image", "graph-up", "flask"],
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
                                    metadata={
                                        "base_prompt": base_prompt,
                                        "technique_params": locals()
                                    }
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
                    "Self-Consistency Check", 
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
                                    metadata={
                                        "original_prompt": original_prompt,
                                        "original_response": original_response,
                                        "evaluation_method": "CoVe"
                                    }
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
            
            elif evaluation_method == "Self-Consistency Check":
                st.markdown("**🔄 Self-Consistency Evaluation**")
                
                consistency_prompt = st.text_area("Prompt to Check for Consistency:", height=100)
                num_samples = st.slider("Number of Reasoning Paths", min_value=2, max_value=5, value=3)
                
                if st.button("🔄 Run Consistency Check", use_container_width=True):
                    if consistency_prompt:
                        with st.spinner("🔄 Generating multiple reasoning paths..."):
                            try:
                                consistency_check_prompt = EvaluationMethods.self_consistency_check(consistency_prompt, num_samples)
                                
                                result = st.session_state.langfuse_tracker.run_prompt_with_tracing(
                                    consistency_check_prompt,
                                    "self_consistency_check",
                                    metadata={
                                        "original_prompt": consistency_prompt,
                                        "num_samples": num_samples,
                                        "evaluation_method": "Self-Consistency"
                                    }
                                )
                                
                                st.session_state.evaluation_results.append(result)
                                
                                st.markdown("### 🔄 Consistency Analysis:")
                                st.markdown(f'<div class="evaluation-result">{result.output}</div>', 
                                          unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"❌ Consistency check error: {str(e)}")
                    else:
                        st.warning("⚠️ Please provide a prompt to check for consistency.")
            
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
                                    metadata={
                                        "assessed_prompt": prompt_to_assess,
                                        "evaluation_method": "Quality Assessment"
                                    }
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
                "Self-Consistency Check": "Generate multiple reasoning paths and compare consistency across different approaches.",
                "Prompt Quality Assessment": "Evaluate prompts across multiple criteria: clarity, specificity, context, structure.",
                "Response Comparison": "Side-by-side comparison of different responses to identify strengths and weaknesses."
            }
            
            if evaluation_method in evaluation_descriptions:
                st.info(evaluation_descriptions[evaluation_method])
            
            # Show evaluation statistics
            if st.session_state.evaluation_results:
                st.markdown("### 📊 Evaluation Stats")
                eval_stats = pd.Series([r.technique for r in st.session_state.evaluation_results]).value_counts()
                for method, count in eval_stats.items():
                    st.write(f"• {method}: {count}")

    # ------------------ Enhanced Batch Processing ------------------
    elif selected == "⚡ Batch Processing":
        st.markdown("## ⚡ Advanced Batch Processing")
        
        st.markdown("""
        <div class="info-box">
            <h4>🚀 Multi-Technique Batch Processing</h4>
            <p>Apply different prompting techniques to multiple prompts simultaneously with comprehensive tracing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Batch configuration
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            batch_technique = st.selectbox(
                "🧠 Select Technique for Batch",
                ["Zero-shot Prompting", "Chain of Thought (CoT)", "ReAct (Reasoning & Acting)", 
                 "Role-based Prompting", "Instruction-based Prompting"]
            )
            
            max_workers = st.slider("🔧 Concurrent Workers", 1, 5, 3)
        
        with col_config2:
            if batch_technique == "Role-based Prompting":
                batch_role = st.selectbox("Role for all prompts", 
                    ["business_analyst", "data_scientist", "software_engineer", "marketing_expert"])
            elif batch_technique == "Instruction-based Prompting":
                batch_instructions = st.text_area("Instructions for all prompts (one per line):", height=100)
                batch_instructions = [inst.strip() for inst in batch_instructions.split('\n') if inst.strip()]
        
        # Input methods
        input_method = st.radio("📝 Input Method:", ["Manual Entry", "File Upload"])
        
        prompts_to_process = []
        
        if input_method == "Manual Entry":
            prompt_text = st.text_area(
                "Enter prompts (one per line):",
                height=200,
                placeholder="Analyze market trends for AI startups\nExplain quantum computing benefits\nCreate a marketing strategy for SaaS products\nDevelop a risk assessment framework"
            )
            prompts_to_process = [p.strip() for p in prompt_text.split('\n') if p.strip()]
        else:
            uploaded_file = st.file_uploader("Upload text file with prompts", type=['txt'])
            if uploaded_file:
                content = str(uploaded_file.read(), "utf-8")
                prompts_to_process = [p.strip() for p in content.split('\n') if p.strip()]
                st.success(f"📁 Loaded {len(prompts_to_process)} prompts")
        
        if prompts_to_process:
            st.markdown(f"### 📊 Ready to process {len(prompts_to_process)} prompts with {batch_technique}")
            
            if st.button("🚀 Start Batch Processing", use_container_width=True):
                start_time = time.time()
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def process_batch_prompt(prompt_data):
                    index, base_prompt = prompt_data
                    try:
                        # Generate technique-specific prompt
                        if batch_technique == "Zero-shot Prompting":
                            final_prompt = PromptingTechniques.zero_shot(base_prompt)
                        elif batch_technique == "Chain of Thought (CoT)":
                            final_prompt = PromptingTechniques.chain_of_thought(base_prompt)
                        elif batch_technique == "ReAct (Reasoning & Acting)":
                            final_prompt = PromptingTechniques.react_prompting(base_prompt, 3)
                        elif batch_technique == "Role-based Prompting":
                            final_prompt = PromptingTechniques.role_based_prompting(base_prompt, batch_role)
                        elif batch_technique == "Instruction-based Prompting" and batch_instructions:
                            final_prompt = PromptingTechniques.instruction_based(base_prompt, batch_instructions)
                        else:
                            final_prompt = base_prompt
                        
                        return st.session_state.langfuse_tracker.run_prompt_with_tracing(
                            final_prompt, 
                            f"batch_{batch_technique.lower().replace(' ', '_')}",
                            metadata={
                                "batch_index": index,
                                "base_prompt": base_prompt,
                                "batch_technique": batch_technique
                            }
                        )
                    except Exception as e:
                        return PromptResult(
                            prompt=base_prompt,
                            output=f"Error: {str(e)}",
                            technique=batch_technique,
                            length=0,
                            input_tokens=0,
                            output_tokens=0,
                            time_taken=0,
                            trace_id=None,
                            timestamp=datetime.now(),
                            metadata={"error": str(e)}
                        )
                
                # Parallel execution
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_prompt = {
                        executor.submit(process_batch_prompt, (i, prompt)): (i, prompt) 
                        for i, prompt in enumerate(prompts_to_process)
                    }
                    
                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_prompt):
                        result = future.result()
                        results.append(result)
                        st.session_state.prompt_results.append(result)
                        
                        completed += 1
                        progress = completed / len(prompts_to_process)
                        progress_bar.progress(progress)
                        status_text.text(f"✅ Completed: {completed}/{len(prompts_to_process)}")
                
                total_time = time.time() - start_time
                
                # Display batch results
                st.markdown("### 🎯 Batch Processing Results")
                
                # Summary metrics
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                successful_results = [r for r in results if not r.output.startswith("Error:")]
                
                col_s1.metric("Total Time", f"{total_time:.2f}s")
                col_s2.metric("Avg per Prompt", f"{total_time/len(results):.2f}s")
                col_s3.metric("Success Rate", f"{len(successful_results)}/{len(results)}")
                col_s4.metric("Total Tokens", sum([r.input_tokens + r.output_tokens for r in successful_results]))
                
                # Individual results with enhanced display
                for i, result in enumerate(results):
                    if not result.output.startswith("Error:"):
                        with st.expander(f"📝 {batch_technique} Result {i+1}: {result.prompt[:60]}..."):
                            formatted_output = format_reasoning_output(result.output, batch_technique)
                            st.markdown(formatted_output, unsafe_allow_html=True)
                            
                            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                            col_r1.metric("Time", f"{result.time_taken:.2f}s")
                            col_r2.metric("Length", result.length)
                            col_r3.metric("Tokens", f"{result.input_tokens + result.output_tokens}")
                            
                            if result.trace_id:
                                col_r4.markdown(f"[📊 Trace]({f'https://cloud.langfuse.com/traces/{result.trace_id}'})")
                            
                            st.markdown(
                                create_download_link(result.output, f"batch_result_{i+1}.txt"),
                                unsafe_allow_html=True
                            )
                
                # Batch export
                if successful_results:
                    combined_results = "\n\n" + "="*80 + "\n\n".join([
                        f"PROMPT {i+1} ({batch_technique}):\n{r.prompt}\n\nRESPONSE:\n{r.output}\n\nMETRICS:\nTime: {r.time_taken}s | Tokens: {r.input_tokens + r.output_tokens} | Length: {r.length}"
                        for i, r in enumerate(successful_results)
                    ])
                    
                    st.markdown(
                        create_download_link(
                            combined_results,
                            f"batch_{batch_technique.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        ),
                        unsafe_allow_html=True
                    )

    # ------------------ Enhanced Vision Analysis ------------------
    elif selected == "🖼️ Vision Analysis":
        st.markdown("## 🖼️ Advanced Vision Analysis")
        
        st.markdown("""
        <div class="info-box">
            <h4>👁️ Multi-Modal AI Analysis</h4>
            <p>Upload images and apply advanced prompting techniques for comprehensive visual understanding.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_images = st.file_uploader(
                "📷 Upload Images",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
                accept_multiple_files=True
            )
            
            if uploaded_images:
                # Vision analysis technique
                vision_technique = st.selectbox(
                    "🧠 Vision Analysis Technique",
                    ["Standard Analysis", "Chain of Thought Analysis", "Role-based Analysis", "Detailed Evaluation"]
                )
                
                # Technique-specific parameters
                if vision_technique == "Role-based Analysis":
                    vision_role = st.selectbox(
                        "Analysis Perspective",
                        ["data_scientist", "business_analyst", "marketing_expert", "designer", "researcher"]
                    )
                
                analysis_prompt = st.text_area(
                    "✏️ What do you want to analyze in the image(s)?",
                    placeholder="Analyze the business metrics, identify trends, extract insights, describe the visual elements...",
                    height=100
                )
                
                analysis_type = st.selectbox(
                    "🎯 Analysis Focus",
                    ["Comprehensive Analysis", "Data Extraction", "Visual Elements", "Business Insights", "Technical Assessment"]
                )
        
        with col2:
            if uploaded_images:
                st.markdown("### 📸 Uploaded Images")
                for i, img in enumerate(uploaded_images[:3]):
                    image = Image.open(img)
                    st.image(image, caption=f"Image {i+1}", use_column_width=True)
                    st.write(f"**Size:** {image.size[0]}×{image.size[1]}")
                    st.write(f"**Format:** {image.format}")
        
        if uploaded_images and st.button("🔍 Analyze Images", use_container_width=True):
            if analysis_prompt:
                with st.spinner("👁️ Analyzing images with advanced techniques..."):
                    try:
                        for i, uploaded_file in enumerate(uploaded_images):
                            st.markdown(f"### 📷 Analysis {i+1}: {uploaded_file.name}")
                            
                            image = Image.open(uploaded_file)
                            
                            # Build analysis prompt based on technique
                            analysis_focus_prompts = {
                                "Comprehensive Analysis": "Provide a comprehensive analysis covering all aspects of this image including visual elements, data, patterns, and insights.",
                                "Data Extraction": "Extract all data, numbers, text, and quantitative information visible in this image. Organize the findings clearly.",
                                "Visual Elements": "Analyze the visual design elements: colors, layout, typography, composition, and aesthetic choices.",
                                "Business Insights": "Focus on business-relevant insights: trends, performance metrics, opportunities, and strategic implications.",
                                "Technical Assessment": "Provide technical analysis: image quality, data visualization effectiveness, and presentation standards."
                            }
                            
                            base_analysis = f"{analysis_focus_prompts[analysis_type]}\n\nSpecific request: {analysis_prompt}"
                            
                            # Apply selected technique
                            if vision_technique == "Standard Analysis":
                                final_prompt = base_analysis
                            elif vision_technique == "Chain of Thought Analysis":
                                final_prompt = PromptingTechniques.chain_of_thought(base_analysis)
                            elif vision_technique == "Role-based Analysis":
                                final_prompt = PromptingTechniques.role_based_prompting(base_analysis, vision_role)
                            elif vision_technique == "Detailed Evaluation":
                                final_prompt = f"""Perform a detailed evaluation of this image:

1. INITIAL OBSERVATION:
{base_analysis}

2. DETAILED ANALYSIS:
- Visual Elements: [Analyze composition, colors, layout]
- Content Analysis: [Examine text, data, information]
- Quality Assessment: [Evaluate clarity, readability, effectiveness]
- Context Understanding: [Interpret purpose, audience, message]

3. INSIGHTS AND RECOMMENDATIONS:
- Key Findings: [Main discoveries]
- Actionable Insights: [Practical implications]
- Improvement Suggestions: [How to enhance or build upon this]

Provide comprehensive analysis:"""
                            
                            # Execute with Gemini Vision
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            response = model.generate_content([final_prompt, image])
                            
                            # Create result with enhanced metadata
                            result = PromptResult(
                                prompt=final_prompt,
                                output=response.text,
                                technique=f"Vision_{vision_technique.replace(' ', '_')}",
                                length=len(response.text),
                                input_tokens=len(final_prompt.split()),
                                output_tokens=len(response.text.split()),
                                time_taken=0,  # Vision API doesn't provide timing
                                trace_id=None,
                                timestamp=datetime.now(),
                                metadata={
                                    "image_name": uploaded_file.name,
                                    "image_size": f"{image.size[0]}x{image.size[1]}",
                                    "image_format": image.format,
                                    "analysis_type": analysis_type,
                                    "vision_technique": vision_technique
                                }
                            )
                            
                            st.session_state.prompt_results.append(result)
                            
                            # Display enhanced results
                            if vision_technique == "Chain of Thought Analysis":
                                formatted_output = format_reasoning_output(response.text, "Chain of Thought")
                                st.markdown(formatted_output, unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="response-container">{response.text}</div>', 
                                          unsafe_allow_html=True)
                            
                            # Enhanced metrics
                            col_i1, col_i2, col_i3, col_i4 = st.columns(4)
                            col_i1.metric("Response Length", len(response.text))
                            col_i2.metric("Image Size", f"{image.size[0]}×{image.size[1]}")
                            col_i3.metric("Format", image.format)
                            col_i4.metric("Analysis Type", analysis_type)
                            
                            # Download with enhanced filename
                            st.markdown(
                                create_download_link(
                                    f"Image: {uploaded_file.name}\nTechnique: {vision_technique}\nAnalysis: {analysis_type}\n\n{response.text}",
                                    f"vision_{vision_technique.lower().replace(' ', '_')}_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                ),
                                unsafe_allow_html=True
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Vision analysis error: {str(e)}")
            else:
                st.warning("⚠️ Please provide analysis instructions.")

    # ------------------ Enhanced Analytics Dashboard ------------------
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
            # Convert to DataFrame with enhanced data
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
                    'has_trace': result.trace_id is not None
                })
            
            df = pd.DataFrame(results_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Enhanced overview metrics
            st.markdown("### 📈 Session Overview")
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            
            col_m1.metric("Total Prompts", len(df))
            col_m2.metric("Avg Response Time", f"{df['time_taken'].mean():.2f}s")
            col_m3.metric("Total Tokens", f"{df['total_tokens'].sum():,}")
            col_m4.metric("Techniques Used", df['technique'].nunique())
            col_m5.metric("Traced Prompts", df['has_trace'].sum())
            
            # Technique performance analysis
            st.markdown("### 🎯 Technique Performance")
            
            col_perf1, col_perf2 = st.columns(2)
            
            with col_perf1:
                # Technique usage distribution
                technique_counts = df['technique'].value_counts()
                fig_techniques = px.pie(
                    values=technique_counts.values,
                    names=technique_counts.index,
                    title="Technique Usage Distribution"
                )
                fig_techniques.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_techniques, use_container_width=True)
            
            with col_perf2:
                # Average response time by technique
                technique_performance = df.groupby('technique').agg({
                    'time_taken': 'mean',
                    'length': 'mean',
                    'total_tokens': 'mean'
                }).round(2)
                
                fig_performance = px.bar(
                    x=technique_performance.index,
                    y=technique_performance['time_taken'],
                    title="Average Response Time by Technique",
                    labels={'x': 'Technique', 'y': 'Time (seconds)'}
                )
                fig_performance.update_xaxis(tickangle=45)
                st.plotly_chart(fig_performance, use_container_width=True)
            
            # Advanced visualizations
            st.markdown("### 📊 Advanced Analytics")
            
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                # Time series of usage
                df['hour'] = df['timestamp'].dt.hour
                hourly_usage = df.groupby('hour').size()
                
                fig_hourly = px.line(
                    x=hourly_usage.index,
                    y=hourly_usage.values,
                    title="Usage Pattern by Hour",
                    labels={'x': 'Hour of Day', 'y': 'Number of Prompts'}
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with col_adv2:
                # Token efficiency analysis
                fig_efficiency = px.scatter(
                    df, x='input_tokens', y='output_tokens',
                    size='time_taken', color='technique',
                    title='Token Efficiency by Technique',
                    hover_data=['length']
                )
                st.plotly_chart(fig_efficiency, use_container_width=True)
            
            # Technique comparison table
            st.markdown("### 📋 Technique Comparison")
            
            comparison_stats = df.groupby('technique').agg({
                'time_taken': ['mean', 'min', 'max'],
                'length': ['mean', 'min', 'max'],
                'total_tokens': ['mean', 'sum'],
                'has_trace': 'sum'
            }).round(2)
            
            comparison_stats.columns = ['Avg Time', 'Min Time', 'Max Time', 
                                      'Avg Length', 'Min Length', 'Max Length',
                                      'Avg Tokens', 'Total Tokens', 'Traced']
            
            st.dataframe(comparison_stats, use_container_width=True)
            
            # Detailed results with enhanced filtering
            st.markdown("### 🔍 Detailed Results")
            
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            with col_filter1:
                technique_filter = st.multiselect(
                    "Filter by Technique",
                    options=df['technique'].unique(),
                    default=df['technique'].unique()
                )
            with col_filter2:
                min_time = st.number_input("Min Response Time (s)", 0.0, df['time_taken'].max(), 0.0)
            with col_filter3:
                show_traced_only = st.checkbox("Show Only Traced Prompts")
            
            # Apply filters
            filtered_df = df[
                (df['technique'].isin(technique_filter)) & 
                (df['time_taken'] >= min_time)
            ]
            
            if show_traced_only:
                filtered_df = filtered_df[filtered_df['has_trace']]
            
            # Enhanced display
            display_columns = ['timestamp', 'technique', 'prompt', 'time_taken', 'length', 'total_tokens', 'trace_id']
            
            st.dataframe(
                filtered_df[display_columns],
                use_container_width=True,
                column_config={
                    'timestamp': st.column_config.DatetimeColumn('Time'),
                    'technique': 'Technique',
                    'prompt': st.column_config.TextColumn('Prompt', width='large'),
                    'time_taken': st.column_config.NumberColumn('Time (s)', format='%.2f'),
                    'length': 'Length',
                    'total_tokens': 'Tokens',
                    'trace_id': st.column_config.LinkColumn(
                        'Langfuse Trace',
                        display_text="View Trace"
                    )
                }
            )
            
            # Enhanced export options
            st.markdown("### 💾 Export Options")
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            with col_exp1:
                if st.button("📊 Export Full Dataset"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"advanced_gemini_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
            
            with col_exp2:
                if st.button("📈 Export Summary Report"):
                    summary_report = f"""Advanced Gemini AI Assistant - Session Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SESSION OVERVIEW:
- Total Prompts: {len(df)}
- Unique Techniques: {df['technique'].nunique()}
- Average Response Time: {df['time_taken'].mean():.2f}s
- Total Tokens Used: {df['total_tokens'].sum():,}
- Traced Prompts: {df['has_trace'].sum()}

TECHNIQUE PERFORMANCE:
{comparison_stats.to_string()}

TOP PERFORMING TECHNIQUES:
{df.groupby('technique')['time_taken'].mean().sort_values().head().to_string()}
"""
                    st.download_button(
                        "Download Report",
                        summary_report,
                        f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain"
                    )
            
            with col_exp3:
                if st.button("🧹 Clear Session Data"):
                    st.session_state.prompt_results = []
                    st.session_state.evaluation_results = []
                    st.rerun()

    # ------------------ Prompt Testing Lab ------------------
    elif selected == "🧪 Prompt Testing Lab":
        st.markdown("## 🧪 Advanced Prompt Testing Laboratory")
        
        st.markdown("""
        <div class="info-box">
            <h4>🔬 Systematic Prompt Engineering</h4>
            <p>Test and compare multiple prompting approaches systematically with comprehensive evaluation metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="technique-card">', unsafe_allow_html=True)
            
            # Test configuration
            st.markdown("### ⚙️ Test Configuration")
            
            base_task = st.text_area(
                "📝 Base Task/Question:",
                height=100,
                placeholder="Enter the core task you want to test with different prompting approaches..."
            )
            
            # Select techniques to compare
            st.markdown("### 🎯 Techniques to Test:")
            techniques_to_test = st.multiselect(
                "Select techniques for comparison:",
                [
                    "Zero-shot Prompting",
                    "Few-shot Prompting",
                    "Chain of Thought (CoT)",
                    "ReAct (Reasoning & Acting)",
                    "Tree of Thoughts (ToT)",
                    "Role-based Prompting",
                    "Instruction-based Prompting"
                ],
                default=["Zero-shot Prompting", "Chain of Thought (CoT)", "ReAct (Reasoning & Acting)"]
            )
            
            # Test parameters
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                test_role = st.selectbox("Role (for role-based prompting):", 
                    ["business_analyst", "data_scientist", "consultant", "researcher"])
                run_evaluation = st.checkbox("Run automatic evaluation", value=True)
            
            with col_param2:
                num_iterations = st.number_input("Iterations per technique", min_value=1, max_value=3, value=1)
                include_verification = st.checkbox("Include Chain of Verification", value=False)
            
            # Few-shot examples (if selected)
            if "Few-shot Prompting" in techniques_to_test:
                st.markdown("### 📚 Few-shot Examples:")
                example1_input = st.text_input("Example 1 Input:")
                example1_output = st.text_input("Example 1 Output:")
                example2_input = st.text_input("Example 2 Input:")
                example2_output = st.text_input("Example 2 Output:")
                
                few_shot_examples = []
                if example1_input and example1_output:
                    few_shot_examples.append({"input": example1_input, "output": example1_output})
                if example2_input and example2_output:
                    few_shot_examples.append({"input": example2_input, "output": example2_output})
            
            # Instructions (if selected)
            if "Instruction-based Prompting" in techniques_to_test:
                st.markdown("### 📋 Instructions:")
                test_instructions = st.text_area("Instructions (one per line):", height=80)
                test_instructions = [inst.strip() for inst in test_instructions.split('\n') if inst.strip()]
            
            # Run comprehensive test
            if st.button("🚀 Run Comprehensive Test", use_container_width=True):
                if base_task and techniques_to_test:
                    with st.spinner("🧪 Running comprehensive prompt testing..."):
                        test_results = []
                        
                        for technique in techniques_to_test:
                            st.write(f"Testing: {technique}")
                            
                            for iteration in range(num_iterations):
                                try:
                                    # Generate appropriate prompt
                                    if technique == "Zero-shot Prompting":
                                        final_prompt = PromptingTechniques.zero_shot(base_task)
                                    elif technique == "Few-shot Prompting" and few_shot_examples:
                                        final_prompt = PromptingTechniques.few_shot(base_task, few_shot_examples)
                                    elif technique == "Chain of Thought (CoT)":
                                        final_prompt = PromptingTechniques.chain_of_thought(base_task)
                                    elif technique == "ReAct (Reasoning & Acting)":
                                        final_prompt = PromptingTechniques.react_prompting(base_task, 3)
                                    elif technique == "Tree of Thoughts (ToT)":
                                        final_prompt = PromptingTechniques.tree_of_thoughts(base_task, 3)
                                    elif technique == "Role-based Prompting":
                                        final_prompt = PromptingTechniques.role_based_prompting(base_task, test_role)
                                    elif technique == "Instruction-based Prompting" and test_instructions:
                                        final_prompt = PromptingTechniques.instruction_based(base_task, test_instructions)
                                    else:
                                        continue
                                    
                                    # Execute with tracing
                                    result = st.session_state.langfuse_tracker.run_prompt_with_tracing(
                                        final_prompt,
                                        f"test_{technique.lower().replace(' ', '_')}",
                                        metadata={
                                            "base_task": base_task,
                                            "test_iteration": iteration + 1,
                                            "test_session": datetime.now().isoformat()
                                        }
                                    )
                                    
                                    test_results.append(result)
                                    st.session_state.prompt_results.append(result)
                                    
                                    # Run evaluation if requested
                                    if run_evaluation:
                                        eval_prompt = EvaluationMethods.prompt_quality_assessment(final_prompt)
                                        eval_result = st.session_state.langfuse_tracker.run_prompt_with_tracing(
                                            eval_prompt,
                                            "quality_assessment",
                                            metadata={"evaluated_technique": technique}
                                        )
                                        st.session_state.evaluation_results.append(eval_result)
                                    
                                    # Run verification if requested
                                    if include_verification:
                                        verify_prompt = EvaluationMethods.chain_of_verification(result.output, base_task)
                                        verify_result = st.session_state.langfuse_tracker.run_prompt_with_tracing(
                                            verify_prompt,
                                            "verification",
                                            metadata={"verified_technique": technique}
                                        )
                                        st.session_state.evaluation_results.append(verify_result)
                                
                                except Exception as e:
                                    st.error(f"Error testing {technique}: {str(e)}")
                        
                        # Display comprehensive results
                        if test_results:
                            st.markdown("## 🎯 Test Results Comparison")
                            
                            # Results summary table
                            results_summary = []
                            for result in test_results:
                                results_summary.append({
                                    'Technique': result.technique,
                                    'Response Time': f"{result.time_taken:.2f}s",
                                    'Length': result.length,
                                    'Input Tokens': result.input_tokens,
                                    'Output Tokens': result.output_tokens,
                                    'Total Tokens': result.input_tokens + result.output_tokens,
                                    'Trace ID': result.trace_id[:8] + "..." if result.trace_id else "None"
                                })
                            
                            summary_df = pd.DataFrame(results_summary)
                            st.dataframe(summary_df, use_container_width=True)
                            
                            # Detailed results
                            st.markdown("### 📋 Detailed Responses")
                            for i, result in enumerate(test_results):
                                with st.expander(f"{result.technique} - Response {i+1}"):
                                    formatted_output = format_reasoning_output(result.output, result.technique)
                                    st.markdown(formatted_output, unsafe_allow_html=True)
                                    
                                    if result.trace_id:
                                        st.markdown(f"🔗 [View Langfuse Trace](https://cloud.langfuse.com/traces/{result.trace_id})")
                            
                            # Performance comparison chart
                            st.markdown("### 📊 Performance Comparison")
                            
                            col_chart1, col_chart2 = st.columns(2)
                            
                            with col_chart1:
                                # Response time comparison
                                fig_time = px.bar(
                                    summary_df, x='Technique', y=[float(t.replace('s', '')) for t in summary_df['Response Time']],
                                    title="Response Time Comparison"
                                )
                                fig_time.update_xaxis(tickangle=45)
                                st.plotly_chart(fig_time, use_container_width=True)
                            
                            with col_chart2:
                                # Token usage comparison
                                fig_tokens = px.bar(
                                    summary_df, x='Technique', y='Total Tokens',
                                    title="Token Usage Comparison"
                                )
                                fig_tokens.update_xaxis(tickangle=45)
                                st.plotly_chart(fig_tokens, use_container_width=True)
                            
                            # Export test results
                            test_report = f"""Prompt Testing Lab Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BASE TASK: {base_task}

TECHNIQUES TESTED: {', '.join(techniques_to_test)}

RESULTS SUMMARY:
{summary_df.to_string(index=False)}

DETAILED RESPONSES:
{'='*50}
"""
                            
                            for result in test_results:
                                test_report += f"\n\nTECHNIQUE: {result.technique}\nRESPONSE:\n{result.output}\n\nMETRICS:\nTime: {result.time_taken}s | Length: {result.length} | Tokens: {result.input_tokens + result.output_tokens}\n{'='*50}"
                            
                            st.markdown(
                                create_download_link(
                                    test_report,
                                    f"prompt_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                ),
                                unsafe_allow_html=True
                            )
                        
                else:
                    st.warning("⚠️ Please provide a base task and select at least one technique to test.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔬 Testing Guide")
            
            st.info("""
            **Systematic Testing Approach:**
            
            1. **Define Base Task**: Clear, specific question or problem
            2. **Select Techniques**: Choose 2-5 techniques to compare
            3. **Configure Parameters**: Set role, examples, instructions
            4. **Run Tests**: Execute all techniques systematically
            5. **Analyze Results**: Compare performance and quality
            6. **Document Findings**: Export comprehensive report
            """)
            
            st.markdown("### 🎯 Best Practices")
            
            st.markdown("""
            - **Consistent Base Task**: Use the same core question for all techniques
            - **Multiple Iterations**: Run 2-3 iterations for reliability
            - **Enable Tracing**: Track all experiments in Langfuse
            - **Include Evaluation**: Use automatic quality assessment
            - **Document Results**: Export reports for future reference
            """)
            
            # Quick test templates
            st.markdown("### 📝 Quick Test Templates")
            
            test_templates = {
                "Business Analysis": "Analyze the market opportunity for AI-powered customer service solutions in the healthcare industry.",
                "Technical Problem": "Design a scalable architecture for a real-time data processing system that handles 1 million events per second.",
                "Creative Task": "Develop a comprehensive marketing campaign for a sustainable fashion brand targeting Gen Z consumers.",
                "Research Question": "Evaluate the potential impact of quantum computing on current cybersecurity protocols and recommend mitigation strategies."
            }
            
            for template_name, template_text in test_templates.items():
                if st.button(f"Use {template_name} Template", key=f"template_{template_name}"):
                    st.session_state.base_task = template_text

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
    - **Self-Consistency**: Multiple reasoning path comparison
    - **Prompt Quality Assessment**: Comprehensive prompt evaluation
    - **Response Comparison**: Side-by-side analysis
    
    #### 🧪 **Advanced Features:**
    - **Batch Processing**: Multi-technique parallel execution
    - **Vision Analysis**: Advanced image understanding
    - **Comprehensive Analytics**: Performance tracking and insights
    - **Prompt Testing Lab**: Systematic technique comparison
    - **Langfuse Integration**: Professional tracing and monitoring
    """)

# ------------------ Enhanced Footer ------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 Advanced Gemini AI Assistant | Professional Prompting & Evaluation Platform</p>
    <p>Built with Streamlit, Google AI, LangChain & Langfuse | Advanced Prompting Techniques Included</p>
</div>
""", unsafe_allow_html=True)