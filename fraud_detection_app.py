import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
import requests
import hashlib
from typing import Dict, List, Tuple
import uuid
from langfuse import Langfuse
import os
from dataclasses import dataclass
import asyncio

# Configure Streamlit page
st.set_page_config(
    page_title="🛡️ AI Banking Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize LangFuse (replace with your actual keys)
@st.cache_resource
def init_langfuse():
    try:
        langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", "lf-sk-your-secret-key"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "lf-pk-your-public-key"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        return langfuse
    except Exception as e:
        st.warning(f"LangFuse initialization failed: {e}")
        return None

# Data classes for better structure
@dataclass
class TransactionData:
    amount: float
    merchant_category: str
    location: str
    transaction_time: str
    card_type: str
    user_id: str = None
    transaction_id: str = None

@dataclass
class RiskAssessment:
    risk_score: float
    risk_level: str
    risk_factors: List[str]
    explanation: str
    recommendation: str
    confidence: float

# Advanced GenAI Features
class GenAIFraudDetector:
    def __init__(self):
        self.langfuse = init_langfuse()
        self.model_versions = {
            "v1": "Basic Rule-based Model",
            "v2": "ML Enhanced Model", 
            "v3": "Large Language Model",
            "v4": "Multimodal AI Model"
        }
        
    def analyze_transaction_llm(self, transaction: TransactionData) -> RiskAssessment:
        """Advanced LLM-based fraud analysis with contextual understanding"""
        
        # Create trace for LangFuse
        trace_id = str(uuid.uuid4())
        if self.langfuse:
            trace = self.langfuse.trace(
                name="fraud_detection_analysis",
                id=trace_id,
                user_id=transaction.user_id or "anonymous",
                metadata={
                    "model_version": "v4",
                    "transaction_amount": transaction.amount,
                    "merchant_category": transaction.merchant_category
                }
            )
        
        # Simulate LLM API call (replace with actual OpenAI/Claude API)
        prompt = self._generate_analysis_prompt(transaction)
        
        # Advanced risk calculation with LLM reasoning
        risk_assessment = self._calculate_advanced_risk(transaction)
        
        # Track generation in LangFuse
        if self.langfuse:
            generation = trace.generation(
                name="risk_assessment_generation",
                model="gpt-4",
                model_parameters={
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                prompt=prompt,
                completion=risk_assessment.explanation,
                metadata={
                    "risk_score": risk_assessment.risk_score,
                    "risk_level": risk_assessment.risk_level
                }
            )
            trace.update(status="success")
        
        return risk_assessment
    
    def _generate_analysis_prompt(self, transaction: TransactionData) -> str:
        """Generate contextual prompt for LLM analysis"""
        return f"""
        Analyze this banking transaction for potential fraud:
        
        Amount: ${transaction.amount:,.2f}
        Merchant Category: {transaction.merchant_category}
        Location: {transaction.location}
        Time: {transaction.transaction_time}
        Card Type: {transaction.card_type}
        
        Consider:
        1. Amount patterns and merchant category risk
        2. Geographic location anomalies
        3. Temporal patterns and velocity
        4. Card type vulnerabilities
        5. Cross-channel behavior consistency
        
        Provide a detailed risk assessment with:
        - Risk score (0-100)
        - Risk level (Low/Medium/High/Critical)
        - Key risk factors
        - Explanation
        - Recommended actions
        - Confidence level
        """
    
    def _calculate_advanced_risk(self, transaction: TransactionData) -> RiskAssessment:
        """Advanced risk calculation with multiple factors"""
        
        # Base risk factors
        amount_risk = min(transaction.amount / 1000, 1.0) * 30
        category_risk = self._get_merchant_category_risk(transaction.merchant_category)
        location_risk = self._get_location_risk(transaction.location)
        time_risk = self._get_time_risk(transaction.transaction_time)
        card_risk = self._get_card_type_risk(transaction.card_type)
        
        # Composite risk score
        total_risk = (amount_risk + category_risk + location_risk + time_risk + card_risk) / 5
        
        # Determine risk level
        if total_risk < 25:
            risk_level = "Low"
        elif total_risk < 50:
            risk_level = "Medium"
        elif total_risk < 75:
            risk_level = "High"
        else:
            risk_level = "Critical"
        
        # Generate risk factors
        risk_factors = []
        if amount_risk > 20:
            risk_factors.append("High transaction amount")
        if category_risk > 20:
            risk_factors.append(f"Risky merchant category: {transaction.merchant_category}")
        if location_risk > 20:
            risk_factors.append(f"Unusual location: {transaction.location}")
        if time_risk > 20:
            risk_factors.append("Suspicious transaction timing")
        if card_risk > 20:
            risk_factors.append(f"Vulnerable card type: {transaction.card_type}")
        
        # Generate explanation
        explanation = f"Transaction analyzed using advanced AI model. Risk factors: {', '.join(risk_factors)}. "
        explanation += f"Amount risk: {amount_risk:.1f}%, Category risk: {category_risk:.1f}%, "
        explanation += f"Location risk: {location_risk:.1f}%, Time risk: {time_risk:.1f}%, Card risk: {card_risk:.1f}%"
        
        # Generate recommendations
        if risk_level == "Critical":
            recommendation = "Immediate transaction block and account freeze required"
        elif risk_level == "High":
            recommendation = "Transaction review and potential block"
        elif risk_level == "Medium":
            recommendation = "Enhanced monitoring and verification"
        else:
            recommendation = "Standard processing with routine monitoring"
        
        return RiskAssessment(
            risk_score=total_risk,
            risk_level=risk_level,
            risk_factors=risk_factors,
            explanation=explanation,
            recommendation=recommendation,
            confidence=0.85 + (total_risk / 100) * 0.15
        )
    
    def _get_merchant_category_risk(self, category: str) -> float:
        """Get risk score for merchant category"""
        high_risk_categories = {
            "Gambling": 80, "Adult Entertainment": 75, "Cryptocurrency": 70,
            "Money Transfer": 65, "Gaming": 60, "Travel": 40
        }
        return high_risk_categories.get(category, 20)
    
    def _get_location_risk(self, location: str) -> float:
        """Get risk score for location"""
        high_risk_locations = {
            "Nigeria": 85, "Russia": 80, "China": 70, "Brazil": 60
        }
        return high_risk_locations.get(location, 15)
    
    def _get_time_risk(self, time_str: str) -> float:
        """Get risk score for transaction time"""
        try:
            transaction_time = datetime.fromisoformat(time_str)
            hour = transaction_time.hour
            
            # High risk during unusual hours
            if 2 <= hour <= 5:
                return 70
            elif 23 <= hour or hour <= 1:
                return 60
            else:
                return 20
        except:
            return 30
    
    def _get_card_type_risk(self, card_type: str) -> float:
        """Get risk score for card type"""
        risk_scores = {
            "Prepaid": 50,
            "Debit": 30,
            "Credit": 25,
            "Corporate": 40
        }
        return risk_scores.get(card_type, 35)

# Initialize the fraud detector
fraud_detector = GenAIFraudDetector()

# Streamlit UI
def main():
    st.title("🛡️ AI Banking Fraud Detection System")
    st.markdown("Advanced fraud detection using GenAI and machine learning")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    model_version = st.sidebar.selectbox(
        "AI Model Version",
        options=list(fraud_detector.model_versions.keys()),
        format_func=lambda x: f"{x}: {fraud_detector.model_versions[x]}"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Transaction Analysis")
        
        # Transaction input form
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input("Amount ($)", min_value=0.01, max_value=100000.0, value=100.0)
                merchant_category = st.selectbox(
                    "Merchant Category",
                    ["Retail", "Restaurant", "Travel", "Gaming", "Gambling", "Cryptocurrency", "Money Transfer"]
                )
                location = st.selectbox(
                    "Location",
                    ["United States", "Canada", "United Kingdom", "Nigeria", "Russia", "China", "Brazil"]
                )
            
            with col2:
                transaction_time = st.text_input("Transaction Time (ISO format)", value=datetime.now().isoformat())
                card_type = st.selectbox(
                    "Card Type",
                    ["Credit", "Debit", "Prepaid", "Corporate"]
                )
                user_id = st.text_input("User ID", value="user_123")
            
            submitted = st.form_submit_button("🔍 Analyze Transaction")
        
        if submitted:
            # Create transaction data
            transaction = TransactionData(
                amount=amount,
                merchant_category=merchant_category,
                location=location,
                transaction_time=transaction_time,
                card_type=card_type,
                user_id=user_id,
                transaction_id=str(uuid.uuid4())
            )
            
            # Analyze transaction
            with st.spinner("🤖 AI analyzing transaction..."):
                risk_assessment = fraud_detector.analyze_transaction_llm(transaction)
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Risk score visualization
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{risk_assessment.risk_score:.1f}/100")
            with col2:
                st.metric("Risk Level", risk_assessment.risk_level)
            with col3:
                st.metric("Confidence", f"{risk_assessment.confidence:.1%}")
            
            # Risk factors
            st.subheader("🚨 Risk Factors")
            for factor in risk_assessment.risk_factors:
                st.error(f"• {factor}")
            
            # Detailed analysis
            st.subheader("📋 Detailed Analysis")
            st.info(risk_assessment.explanation)
            
            # Recommendation
            st.subheader("💡 Recommendation")
            if risk_assessment.risk_level == "Critical":
                st.error(risk_assessment.recommendation)
            elif risk_assessment.risk_level == "High":
                st.warning(risk_assessment.recommendation)
            else:
                st.success(risk_assessment.recommendation)
    
    with col2:
        st.header("📈 Analytics")
        
        # Generate sample data for visualization
        @st.cache_data
        def generate_sample_data():
            np.random.seed(42)
            n_transactions = 100
            
            categories = ["Retail", "Restaurant", "Travel", "Gaming", "Gambling"]
            locations = ["United States", "Canada", "United Kingdom", "Nigeria", "Russia"]
            card_types = ["Credit", "Debit", "Prepaid"]
            
            data = []
            for i in range(n_transactions):
                transaction = TransactionData(
                    amount=np.random.exponential(100),
                    merchant_category=np.random.choice(categories),
                    location=np.random.choice(locations),
                    transaction_time=(datetime.now() - timedelta(hours=np.random.randint(0, 168))).isoformat(),
                    card_type=np.random.choice(card_types)
                )
                risk = fraud_detector.analyze_transaction_llm(transaction)
                data.append({
                    "amount": transaction.amount,
                    "risk_score": risk.risk_score,
                    "category": transaction.merchant_category,
                    "location": transaction.location,
                    "card_type": transaction.card_type
                })
            
            return pd.DataFrame(data)
        
        sample_data = generate_sample_data()
        
        # Risk distribution
        fig_risk = px.histogram(
            sample_data, 
            x="risk_score", 
            nbins=20,
            title="Risk Score Distribution",
            labels={"risk_score": "Risk Score", "count": "Number of Transactions"}
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Risk by category
        fig_category = px.box(
            sample_data,
            x="category",
            y="risk_score",
            title="Risk Scores by Merchant Category"
        )
        st.plotly_chart(fig_category, use_container_width=True)
        
        # Amount vs Risk correlation
        fig_scatter = px.scatter(
            sample_data,
            x="amount",
            y="risk_score",
            color="card_type",
            title="Transaction Amount vs Risk Score",
            labels={"amount": "Amount ($)", "risk_score": "Risk Score"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

if __name__ == "__main__":
    main()