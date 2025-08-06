import os
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import google.generativeai as genai
import time
import base64
import concurrent.futures
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import uuid
from datetime import datetime
import numpy as np
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    from streamlit_lottie import st_lottie
    import requests
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False
    st.warning("Lottie animations not available. Install with: pip install streamlit-lottie")

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="🚀 Ultimate Gemini AI Platform", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Lottie Animations
@st.cache_data
def load_lottie_url(url: str):
    if not LOTTIE_AVAILABLE:
        return None
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

# Animation URLs
ai_animation = load_lottie_url("https://lottie.host/4f3c5c4d-2c4a-4c3a-8b7e-1f2d3e4f5a6b/9Kj8H7G6F5.json")
analytics_animation = load_lottie_url("https://lottie.host/8a9b0c1d-2e3f-4g5h-6i7j-8k9l0m1n2o3p/4A5B6C7D8E.json")
processing_animation = load_lottie_url("https://lottie.host/1f2e3d4c-5b6a-7g8h-9i0j-1k2l3m4n5o6p/7F8G9H0I1J.json")

# Enhanced CSS with Animations
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
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
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #f093fb, #ffeaa7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 4s ease-in-out infinite;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        animation: float 3s ease-in-out infinite;
    }
    
    .technique-card {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .technique-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .technique-card:hover::before {
        left: 100%;
    }
    
    .technique-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.4);
    }
    
    .response-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: slide-up 0.6s ease-out;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
        animation: pulse-glow 2s ease-in-out infinite alternate;
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
        animation: bounce-in 0.8s ease-out;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(255, 236, 210, 0.4);
    }
    
    .langfuse-trace {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 8px 25px rgba(168, 237, 234, 0.3);
        animation: slide-in-left 0.8s ease-out;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.4s ease;
        width: 100%;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transition: width 0.6s, height 0.6s, top 0.6s, left 0.6s;
        transform: translate(-50%, -50%);
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
        top: 50%;
        left: 50%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes slide-up {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes slide-in-left {
        from { transform: translateX(-30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse-glow {
        from { box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3); }
        to { box-shadow: 0 8px 35px rgba(240, 147, 251, 0.6); }
    }
    
    @keyframes bounce-in {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.05); }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
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
    quality_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

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

# Advanced Langfuse Tracker with Enhanced Analytics
class AdvancedLangfuseTracker:
    def __init__(self):
        self.langfuse = None
        self.is_configured = False
        self.session_id = str(uuid.uuid4())
        self.trace_count = 0
        
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
        except Exception as e:
            st.error(f"Langfuse configuration failed: {e}")
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
                    self.trace_count += 1
                    trace = self.langfuse.trace(
                        name=f"{technique}_execution_{self.trace_count}",
                        session_id=self.session_id,
                        input={"prompt": prompt_text, "technique": technique},
                        output={"response": result_text},
                        metadata={
                            "time_taken": time_taken,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "technique": technique,
                            "trace_number": self.trace_count,
                            **kwargs
                        }
                    )
                    
                    generation = self.langfuse.generation(
                        name=f"gemini_{technique}",
                        model="gemini-1.5-flash",
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
                except Exception as e:
                    st.warning(f"Langfuse tracing failed: {e}")
            
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

# Advanced Prompting Techniques
class AdvancedPromptingTechniques:
    @staticmethod
    def zero_shot(base_prompt: str) -> str:
        return f"""Task: {base_prompt}

Provide a comprehensive and well-structured response based on your knowledge. Include relevant details and examples where appropriate."""

    @staticmethod
    def few_shot(base_prompt: str, examples: List[Dict[str, str]]) -> str:
        examples_text = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(examples)
        ])
        
        return f"""Task: {base_prompt}

Here are some examples to guide your response:

{examples_text}

Now, please complete the following task following the same pattern and style:"""

    @staticmethod
    def chain_of_thought(base_prompt: str) -> str:
        return f"""Task: {base_prompt}

Let's approach this systematically using step-by-step reasoning:

1. **Understanding**: First, let me clearly understand what is being asked
2. **Analysis**: Break down the problem into smaller, manageable components
3. **Processing**: Work through each component methodically
4. **Synthesis**: Combine insights from each step
5. **Verification**: Check the reasoning for consistency and accuracy
6. **Conclusion**: Provide a comprehensive final answer

Let me work through this step by step:"""

    @staticmethod
    def react_prompting(base_prompt: str, max_iterations: int = 3) -> str:
        return f"""Task: {base_prompt}

I'll use the ReAct (Reasoning and Acting) framework to solve this systematically:

ITERATION 1:
🤔 Thought: [Analyze the current situation and determine what needs to be done]
⚡ Action: [Describe the specific action or analysis to perform]
👁️ Observation: [Record what is discovered or learned from the action]
📊 Assessment: [Evaluate progress and plan next steps]

ITERATION 2:
🤔 Thought: [Build upon previous observations and continue reasoning]
⚡ Action: [Take the next logical action based on new understanding]
👁️ Observation: [Document new findings and insights]
📊 Assessment: [Assess current progress and adjust strategy if needed]

ITERATION 3:
🤔 Thought: [Final reasoning incorporating all previous observations]
⚡ Action: [Execute final action or synthesis]
👁️ Observation: [Record final observations and insights]
📊 Assessment: [Complete evaluation of the solution]

🎯 FINAL ANSWER:
[Provide comprehensive final answer based on all iterations]

Beginning ReAct analysis:"""

    @staticmethod
    def tree_of_thoughts(base_prompt: str, num_branches: int = 3) -> str:
        branches = ["A", "B", "C", "D", "E"][:num_branches]
        branch_text = ""
        
        for i, branch in enumerate(branches):
            branch_text += f"""
🌳 BRANCH {branch} - Approach {i+1}:
💡 Reasoning: [Describe the {i+1} approach to solving this problem]
🔍 Analysis: [Detailed analysis using this specific approach]
📝 Findings: [Key insights and conclusions from this branch]
⭐ Strengths: [Advantages of this approach]
⚠️ Limitations: [Potential weaknesses or constraints]
"""
        
        return f"""Task: {base_prompt}

Using Tree of Thoughts methodology to explore {num_branches} different reasoning paths:
{branch_text}

🔗 SYNTHESIS:
📊 Comparison: [Compare and contrast all {num_branches} approaches]
💎 Best Elements: [Identify the strongest insights from each branch]
🎯 Integrated Solution: [Combine the best elements into a comprehensive solution]
⚖️ Trade-offs: [Discuss any trade-offs in the final approach]

🏆 FINAL ANSWER:
[Provide the optimal solution incorporating insights from all branches]

Beginning Tree of Thoughts analysis:"""

    @staticmethod
    def role_based_prompting(base_prompt: str, role: str, expertise_level: str = "expert") -> str:
        role_descriptions = {
            "business_analyst": "You are a senior business analyst with 15+ years of experience in strategic planning, market analysis, and business process optimization.",
            "data_scientist": "You are an expert data scientist with deep knowledge in machine learning, statistical analysis, and data-driven decision making.",
            "software_engineer": "You are a senior software engineer with expertise in system architecture, best practices, and scalable solution design.",
            "marketing_expert": "You are a marketing professional with extensive experience in digital marketing, brand strategy, and customer acquisition.",
            "financial_advisor": "You are a certified financial advisor with expertise in investment strategies, risk management, and financial planning.",
            "teacher": "You are an experienced educator skilled at explaining complex concepts clearly and engagingly.",
            "consultant": "You are a management consultant with experience helping organizations solve complex business problems.",
            "researcher": "You are an academic researcher with expertise in conducting thorough analysis and presenting findings."
        }
        
        role_desc = role_descriptions.get(role, f"You are an {expertise_level} {role.replace('_', ' ')} with extensive experience in your field.")
        
        return f"""{role_desc}

Given your professional expertise and experience, please address the following:

{base_prompt}

Provide your response from your professional perspective, including:

🎯 **Professional Assessment**: Your expert evaluation of the situation
🔍 **Key Considerations**: Important factors from your field of expertise
💡 **Recommended Approach**: Your suggested solution or strategy
⚠️ **Potential Risks**: Challenges or obstacles to consider
✅ **Best Practices**: Industry standards and proven methods
📊 **Success Metrics**: How to measure effectiveness
🔮 **Future Considerations**: Long-term implications and trends

Your expert response:"""

    @staticmethod
    def instruction_based(base_prompt: str, instructions: List[str]) -> str:
        instructions_text = "\n".join([f"✓ {instruction}" for instruction in instructions])
        
        return f"""Task: {base_prompt}

Please follow these specific instructions carefully:

{instructions_text}

Ensure your response adheres to all the above guidelines while providing comprehensive and valuable information."""

    @staticmethod
    def parallel_prompting(base_prompt: str, perspectives: List[str]) -> str:
        perspectives_text = "\n".join([f"{i+1}. **{perspective}**" for i, perspective in enumerate(perspectives)])
        
        return f"""Task: {base_prompt}

Please analyze this from the following {len(perspectives)} different perspectives:

{perspectives_text}

For each perspective, provide:
🔍 **Analysis**: Detailed examination from that viewpoint
💡 **Key Insights**: Important discoveries or concerns
📋 **Recommendations**: Specific suggestions from that perspective
⚖️ **Trade-offs**: Benefits and limitations of this viewpoint

🔗 **SYNTHESIS**:
After analyzing from all perspectives, provide:
- **Common Themes**: Areas of agreement across perspectives
- **Conflicting Views**: Where perspectives disagree and why
- **Integrated Recommendation**: A balanced approach considering all viewpoints
- **Implementation Strategy**: How to address different stakeholder concerns

Multi-perspective analysis:"""

# Advanced Prompt Analyzer
class AdvancedPromptAnalyzer:
    @staticmethod
    def analyze_prompt(prompt: str) -> PromptAnalysis:
        # Scoring criteria (1-10 scale)
        clarity_score = AdvancedPromptAnalyzer._analyze_clarity(prompt)
        specificity_score = AdvancedPromptAnalyzer._analyze_specificity(prompt)
        context_score = AdvancedPromptAnalyzer._analyze_context(prompt)
        structure_score = AdvancedPromptAnalyzer._analyze_structure(prompt)
        completeness_score = AdvancedPromptAnalyzer._analyze_completeness(prompt)
        
        overall_score = (clarity_score + specificity_score + context_score + structure_score + completeness_score) / 5
        
        suggestions = AdvancedPromptAnalyzer._generate_suggestions(prompt, {
            'clarity': clarity_score,
            'specificity': specificity_score,
            'context': context_score,
            'structure': structure_score,
            'completeness': completeness_score
        })
        
        improved_prompt = AdvancedPromptAnalyzer._improve_prompt(prompt, suggestions)
        
        return PromptAnalysis(
            clarity_score=clarity_score,
            specificity_score=specificity_score,
            context_score=context_score,
            structure_score=structure_score,
            completeness_score=completeness_score,
            overall_score=overall_score,
            suggestions=suggestions,
            improved_prompt=improved_prompt
        )
    
    @staticmethod
    def _analyze_clarity(prompt: str) -> float:
        score = 5.0  # Base score
        
        # Check for clear language
        if len(prompt.split()) > 10:
            score += 1.0
        
        # Check for ambiguous words
        ambiguous_words = ['maybe', 'perhaps', 'might', 'could', 'somewhat', 'kind of']
        ambiguous_count = sum(1 for word in ambiguous_words if word in prompt.lower())
        score -= ambiguous_count * 0.5
        
        # Check for question clarity
        if '?' in prompt:
            score += 1.0
        
        # Check for specific action words
        action_words = ['analyze', 'explain', 'describe', 'compare', 'evaluate', 'create', 'generate']
        if any(word in prompt.lower() for word in action_words):
            score += 1.0
        
        return min(10.0, max(1.0, score))
    
    @staticmethod
    def _analyze_specificity(prompt: str) -> float:
        score = 5.0
        
        # Check for specific details
        if any(char.isdigit() for char in prompt):
            score += 1.0
        
        # Check for specific examples or constraints
        specific_indicators = ['for example', 'specifically', 'exactly', 'precisely', 'in particular']
        if any(indicator in prompt.lower() for indicator in specific_indicators):
            score += 1.5
        
        # Check for vague terms
        vague_terms = ['thing', 'stuff', 'something', 'anything', 'everything']
        vague_count = sum(1 for term in vague_terms if term in prompt.lower())
        score -= vague_count * 0.8
        
        return min(10.0, max(1.0, score))
    
    @staticmethod
    def _analyze_context(prompt: str) -> float:
        score = 5.0
        
        # Check for background information
        context_indicators = ['background', 'context', 'situation', 'scenario', 'given that']
        if any(indicator in prompt.lower() for indicator in context_indicators):
            score += 2.0
        
        # Check for domain specification
        if len(prompt.split()) > 20:
            score += 1.0
        
        return min(10.0, max(1.0, score))
    
    @staticmethod
    def _analyze_structure(prompt: str) -> float:
        score = 5.0
        
        # Check for numbered lists or bullet points
        if re.search(r'\d+\.|\•|\-', prompt):
            score += 1.5
        
        # Check for clear sections
        if '\n' in prompt:
            score += 1.0
        
        # Check for logical flow
        if len(prompt.split('.')) > 2:
            score += 1.0
        
        return min(10.0, max(1.0, score))
    
    @staticmethod
    def _analyze_completeness(prompt: str) -> float:
        score = 5.0
        
        # Check for output format specification
        format_words = ['format', 'structure', 'organize', 'present', 'display']
        if any(word in prompt.lower() for word in format_words):
            score += 1.5
        
        # Check for constraints or requirements
        constraint_words = ['must', 'should', 'require', 'need', 'include']
        if any(word in prompt.lower() for word in constraint_words):
            score += 1.0
        
        # Check for expected length
        length_indicators = ['brief', 'detailed', 'comprehensive', 'summary', 'in-depth']
        if any(indicator in prompt.lower() for indicator in length_indicators):
            score += 1.0
        
        return min(10.0, max(1.0, score))
    
    @staticmethod
    def _generate_suggestions(prompt: str, scores: Dict[str, float]) -> List[str]:
        suggestions = []
        
        if scores['clarity'] < 7:
            suggestions.append("Add more specific action words (analyze, explain, describe)")
            suggestions.append("Remove ambiguous language and be more direct")
        
        if scores['specificity'] < 7:
            suggestions.append("Include specific examples or constraints")
            suggestions.append("Add numerical requirements or measurable criteria")
        
        if scores['context'] < 7:
            suggestions.append("Provide more background information")
            suggestions.append("Explain the situation or scenario clearly")
        
        if scores['structure'] < 7:
            suggestions.append("Use numbered lists or bullet points for clarity")
            suggestions.append("Break down complex requests into smaller parts")
        
        if scores['completeness'] < 7:
            suggestions.append("Specify the desired output format")
            suggestions.append("Include any constraints or requirements")
        
        return suggestions
    
    @staticmethod
    def _improve_prompt(prompt: str, suggestions: List[str]) -> str:
        improved = f"""**Enhanced Prompt:**

{prompt}

**Additional Specifications:**
- Provide a well-structured response with clear sections
- Include specific examples where relevant
- Ensure comprehensive coverage of the topic
- Use professional language and formatting

**Expected Output:**
- Clear introduction and conclusion
- Logical flow of information
- Actionable insights where applicable
"""
        return improved

# Helper Functions
def create_download_link(text: str, filename: str) -> str:
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}" style="color: white; text-decoration: none; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); padding: 12px 24px; border-radius: 12px; font-weight: 600; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);">📥 Download Results</a>'

def display_hero_section():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if ai_animation and LOTTIE_AVAILABLE:
            st_lottie(ai_animation, height=200, key="hero_animation")
    
    with col2:
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="main-header">🚀 Ultimate Gemini AI Platform</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <h3>🎯 Professional AI Prompting & Analytics Platform</h3>
            <p>Master advanced prompting techniques with comprehensive analytics, real-time monitoring, and professional-grade evaluation methods.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if analytics_animation and LOTTIE_AVAILABLE:
            st_lottie(analytics_animation, height=200, key="analytics_animation")

# Initialize Session State
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'langfuse_tracker' not in st.session_state:
    st.session_state.langfuse_tracker = AdvancedLangfuseTracker()
if 'prompt_results' not in st.session_state:
    st.session_state.prompt_results = []
if 'prompt_analyses' not in st.session_state:
    st.session_state.prompt_analyses = []

# Load CSS and Display Hero
load_css()
display_hero_section()

# Enhanced Sidebar Configuration
with st.sidebar:
    st.markdown("## 🔑 Advanced Configuration")
    
    # API Configuration
    api_key = st.text_input("🔐 Google API Key", type="password", 
                           help="Get your API key from Google AI Studio")
    
    # Enhanced Langfuse Configuration
    if LANGFUSE_AVAILABLE:
        st.markdown("### 📊 Langfuse Analytics")
        langfuse_enabled = st.checkbox("Enable Advanced Tracing", value=False)
        
        if langfuse_enabled:
            st.markdown("""
            <div class="info-box">
                <small><strong>🎯 Professional Monitoring</strong><br>
                Track all interactions with advanced analytics</small>
            </div>
            """, unsafe_allow_html=True)
            
            langfuse_public = st.text_input("🔑 Langfuse Public Key", type="password")
            langfuse_secret = st.text_input("🔐 Langfuse Secret Key", type="password")
            langfuse_host = st.text_input("🌐 Langfuse Host", value="https://cloud.langfuse.com")
            
            if langfuse_public and langfuse_secret:
                if st.button("🔗 Connect to Langfuse"):
                    with st.spinner("Establishing connection..."):
                        if st.session_state.langfuse_tracker.configure(langfuse_public, langfuse_secret, langfuse_host):
                            st.markdown("""
                            <div class="success-box">
                                <strong>✅ Connected Successfully!</strong><br>
                                Advanced tracing is now active
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("❌ Connection failed")
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Langfuse Not Available</strong><br>
            Install with: <code>pip install langfuse</code><br>
            Then restart the application.
        </div>
        """, unsafe_allow_html=True)
    
    # API Key Setup
    if api_key:
        if not st.session_state.api_configured:
            try:
                os.environ["GOOGLE_API_KEY"] = api_key
                genai.configure(api_key=api_key)
                
                # Test the API key
                model = genai.GenerativeModel("gemini-1.5-flash")
                test_response = model.generate_content("Hello")
                
                st.session_state.api_configured = True
                st.markdown("""
                <div class="success-box">
                    <strong>✅ Google AI Connected!</strong><br>
                    Platform ready for advanced operations
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ API Configuration Error: {str(e)}")
                st.session_state.api_configured = False
        else:
            st.markdown("""
            <div class="success-box">
                <strong>🚀 Platform Active</strong><br>
                All systems operational
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.langfuse_tracker.is_configured:
                st.markdown(f"""
                <div class="langfuse-trace">
                    <strong>📊 Advanced Analytics Active</strong><br>
                    Session: {st.session_state.langfuse_tracker.session_id[:8]}...<br>
                    Traces: {st.session_state.langfuse_tracker.trace_count}
                </div>
                """, unsafe_allow_html=True)
    
    # Session Statistics
    if st.session_state.prompt_results:
        st.markdown("### 📈 Session Analytics")
        
        total_prompts = len(st.session_state.prompt_results)
        avg_time = np.mean([r.time_taken for r in st.session_state.prompt_results])
        total_tokens = sum([r.input_tokens + r.output_tokens for r in st.session_state.prompt_results])
        success_rate = len([r for r in st.session_state.prompt_results if "Error:" not in r.output]) / total_prompts * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Prompts", total_prompts)
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col2:
            st.metric("Avg Time", f"{avg_time:.2f}s")
            st.metric("Total Tokens", f"{total_tokens:,}")

if st.session_state.api_configured:
    # Enhanced Navigation Menu
    selected = option_menu(
        menu_title="🎛️ Ultimate AI Platform",
        options=[
            "🎯 Prompting Techniques", 
            "🔍 Evaluation Methods",
            "📊 Prompt Analyzer",
            "⚡ Batch Processing", 
            "🖼️ Vision Analysis",
            "📈 Analytics Dashboard"
        ],
        icons=["target", "search", "graph-up-arrow", "lightning", "image", "bar-chart"],
        menu_icon="robot",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#667eea", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "5px",
                "padding": "12px",
                "border-radius": "15px",
                "background-color": "rgba(255,255,255,0.1)",
                "backdrop-filter": "blur(10px)",
                "border": "1px solid rgba(255,255,255,0.2)"
            },
            "nav-link-selected": {
                "background": "linear-gradient(45deg, #667eea 0%, #764ba2 100%)",
                "box-shadow": "0 8px 25px rgba(102, 126, 234, 0.3)"
            },
        }
    )
    
    # PROMPTING TECHNIQUES SECTION
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
                        if processing_animation and LOTTIE_AVAILABLE:
                            st_lottie(processing_animation, height=100, key="processing")
                        
                        try:
                            # Generate the appropriate prompt based on technique
                            if technique == "Zero-shot Prompting":
                                final_prompt = AdvancedPromptingTechniques.zero_shot(base_prompt)
                            elif technique == "Few-shot Prompting" and examples:
                                final_prompt = AdvancedPromptingTechniques.few_shot(base_prompt, examples)
                            elif technique == "Chain of Thought (CoT)":
                                final_prompt = AdvancedPromptingTechniques.chain_of_thought(base_prompt)
                            elif technique == "ReAct (Reasoning & Acting)":
                                final_prompt = AdvancedPromptingTechniques.react_prompting(base_prompt, max_iterations)
                            elif technique == "Tree of Thoughts (ToT)":
                                final_prompt = AdvancedPromptingTechniques.tree_of_thoughts(base_prompt, num_branches)
                            elif technique == "Role-based Prompting":
                                final_prompt = AdvancedPromptingTechniques.role_based_prompting(base_prompt, role, expertise_level)
                            elif technique == "Instruction-based Prompting" and instructions:
                                final_prompt = AdvancedPromptingTechniques.instruction_based(base_prompt, instructions)
                            elif technique == "Parallel Prompting" and perspectives:
                                final_prompt = AdvancedPromptingTechniques.parallel_prompting(base_prompt, perspectives)
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
                                st.markdown(f'<div class="response-container">{result.output}</div>', 
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

    # PROMPT ANALYZER SECTION
    elif selected == "📊 Prompt Analyzer":
        st.markdown("## 📊 Intelligent Prompt Analyzer")
        
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Professional Prompt Quality Assessment</h4>
            <p>Analyze and optimize your prompts with advanced scoring algorithms and improvement suggestions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="technique-card">', unsafe_allow_html=True)
            
            prompt_to_analyze = st.text_area("✏️ Enter prompt to analyze:", height=150, 
                placeholder="Enter your prompt here for comprehensive quality analysis...")
            
            if st.button("🔍 Analyze Prompt Quality", use_container_width=True):
                if prompt_to_analyze:
                    with st.spinner("🔄 Analyzing prompt quality..."):
                        analysis = AdvancedPromptAnalyzer.analyze_prompt(prompt_to_analyze)
                        st.session_state.prompt_analyses.append(analysis)
                        
                        st.markdown("### 📊 Quality Assessment Results")
                        
                        # Score visualization
                        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
                        col_s1.metric("🎯 Clarity", f"{analysis.clarity_score:.1f}/10")
                        col_s2.metric("🔍 Specificity", f"{analysis.specificity_score:.1f}/10")
                        col_s3.metric("📋 Context", f"{analysis.context_score:.1f}/10")
                        col_s4.metric("🏗️ Structure", f"{analysis.structure_score:.1f}/10")
                        col_s5.metric("✅ Completeness", f"{analysis.completeness_score:.1f}/10")
                        
                        # Overall score with color coding
                        overall_color = "green" if analysis.overall_score >= 8 else "orange" if analysis.overall_score >= 6 else "red"
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: {overall_color}33; border-radius: 15px; margin: 1rem 0;">
                            <h3 style="color: {overall_color};">Overall Score: {analysis.overall_score:.1f}/10</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Radar chart for scores
                        categories = ['Clarity', 'Specificity', 'Context', 'Structure', 'Completeness']
                        scores = [analysis.clarity_score, analysis.specificity_score, analysis.context_score, 
                                analysis.structure_score, analysis.completeness_score]
                        
                        fig = go.Figure(data=go.Scatterpolar(
                            r=scores,
                            theta=categories,
                            fill='toself',
                            name='Prompt Quality'
                        ))
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                            showlegend=False,
                            title="Prompt Quality Radar Chart"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Suggestions
                        if analysis.suggestions:
                            st.markdown("### 💡 Improvement Suggestions")
                            for i, suggestion in enumerate(analysis.suggestions, 1):
                                st.markdown(f"**{i}.** {suggestion}")
                        
                        # Improved prompt
                        st.markdown("### ✨ Enhanced Prompt")
                        st.markdown(f'<div class="response-container">{analysis.improved_prompt}</div>', unsafe_allow_html=True)
                        
                        st.markdown(create_download_link(
                            f"Original Prompt: {prompt_to_analyze}\n\nAnalysis Results:\n{analysis.improved_prompt}",
                            "prompt_analysis.txt"
                        ), unsafe_allow_html=True)
                        
                else:
                    st.warning("⚠️ Please enter a prompt to analyze.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔍 Analysis Guide")
            st.info("""
            **Quality Metrics:**
            - **Clarity**: How clear and unambiguous
            - **Specificity**: Level of detail and precision
            - **Context**: Background information provided
            - **Structure**: Organization and flow
            - **Completeness**: All necessary information included
            """)
            
            if st.session_state.prompt_analyses:
                st.markdown("### 📈 Analysis History")
                for i, analysis in enumerate(st.session_state.prompt_analyses[-3:], 1):
                    st.write(f"**Analysis {i}**: {analysis.overall_score:.1f}/10")

    # ANALYTICS DASHBOARD SECTION
    elif selected == "📈 Analytics Dashboard":
        st.markdown("## 📈 Professional Analytics Dashboard")
        
        if not st.session_state.prompt_results:
            st.markdown("""
            <div class="info-box">
                <h4>📈 Generate Data for Analytics</h4>
                <p>Use prompting techniques, batch processing, or vision analysis to generate comprehensive analytics data!</p>
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
            
            # Enhanced Overview metrics with trends
            st.markdown("### 📈 Executive Summary")
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
            st.markdown("### 🔍 Deep Analytics")
            
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
            st.markdown("### 🔍 Detailed Results Explorer")
            
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
                        st.session_state.prompt_analyses = []
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
    
    #### 🎯 **Advanced Prompting Techniques:**
    - **Zero-shot**: Direct instruction without examples
    - **Few-shot**: Learning from provided examples  
    - **Chain of Thought**: Step-by-step reasoning
    - **ReAct**: Reasoning and Acting framework
    - **Tree of Thoughts**: Multiple reasoning paths
    - **Role-based**: Expert perspective prompting
    - **Instruction-based**: Detailed guideline following
    - **Parallel**: Multiple perspective analysis
    
    #### 🔍 **Advanced Evaluation Methods:**
    - **Chain of Verification**: Systematic fact-checking
    - **Prompt Quality Assessment**: Comprehensive prompt evaluation
    - **Response Comparison**: Side-by-side analysis
    
    #### 📊 **Intelligent Prompt Analyzer:**
    - **Real-time quality assessment**: 5-dimensional scoring system
    - **Professional scoring system**: Clarity, Specificity, Context, Structure, Completeness
    - **Automated improvement suggestions**: AI-powered recommendations
    - **Performance optimization**: Enhanced prompt generation
    
    #### ⚡ **High-Performance Batch Processing:**
    - **Concurrent multi-prompt execution**: Process up to 8 prompts simultaneously
    - **Real-time progress monitoring**: Live tracking with performance analytics
    - **Performance analytics**: Comprehensive metrics and insights
    
    #### 🖼️ **Professional Vision Analysis:**
    - **Multi-modal AI capabilities**: Advanced image analysis
    - **Expert perspective analysis**: Professional domain insights
    - **Comprehensive image insights**: Technical and creative assessments
    
    #### 📈 **Enterprise Analytics Dashboard:**
    - **10+ interactive visualizations**: Professional Plotly charts
    - **Real-time performance metrics**: Live session analytics
    - **Advanced filtering & export**: Professional reporting capabilities
    
    #### 🔗 **Langfuse Integration:**
    - **Professional-grade tracing**: Track all AI interactions
    - **Advanced observability features**: Detailed execution analytics
    - **Session-based performance tracking**: Comprehensive monitoring
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 2rem;'>
    <h4>🚀 Ultimate Gemini AI Platform</h4>
    <p>Professional-Grade AI Prompting & Analytics Solution</p>
    <p><strong>Built with:</strong> Streamlit • Google AI • Langfuse • Plotly • Lottie</p>
    <p><em>Enterprise-Ready • Real-time Analytics • Advanced Techniques</em></p>
</div>
""", unsafe_allow_html=True)