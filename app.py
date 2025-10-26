import streamlit as st
import random
import requests
from datetime import datetime

# ⚠️ IMPORTANT: These imports must point to your simplified files (db_utils and db_models)
# If you don't want logging at all, you can remove these and related code.
from db_utils import initialize_database, log_post_analysis 

# --- API CLIENT AND UTILITY STUBS ---
# URL must match the host/port of your Python Flask/FastAPI service (e.g., model_service/ensemble_api.py)
API_URL = "http://127.0.0.1:5001/predict_sentiment" 

def clean_text_for_analysis(text):
    """Placeholder for text cleaning before API call."""
    return text

class APIModelClient:
    """Client to interact with the external Python model API."""
    def predict(self, text):
        try:
            response = requests.post(API_URL, json={'text': text}, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            sentiment = data.get('sentiment')
            confidence = data.get('confidence')
            
            if sentiment is None or confidence is None:
                 raise ValueError("API response missing 'sentiment' or 'confidence'.")
                 
            return str(sentiment), float(confidence)

        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot reach model service at {API_URL}. Is the Python service running?")
        except requests.exceptions.Timeout:
            raise ConnectionError("Model service timed out.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API Request failed: {e}")

def load_ensemble_models():
    """Initializes the API client connection."""
    return APIModelClient()

# --- END API CLIENT STUBS ---


# Configure page
st.set_page_config(
    page_title="Mental Health Post Risk Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (Retained for styling)
st.markdown("""
<style>
    .main-header { text-align: center; color: #2E7D32; font-size: 2.5rem; margin-bottom: 2rem; font-weight: 600; }
    .risk-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; margin-left: 10px; }
    .risk-no { background-color: #C8E6C9; color: #2E7D32; }
    .risk-low { background-color: #FFF3E0; color: #F57C00; }
    .risk-moderate { background-color: #FFE0B2; color: #E65100; }
    .risk-high { background-color: #FFCDD2; color: #C62828; }
    .confidence-score { display: inline-block; padding: 4px 8px; border-radius: 12px; font-size: 0.75rem; background-color: #E0E0E0; color: #424242; margin-left: 8px; }
    .disclaimer { background-color: #FFF9C4; padding: 1rem; border-left: 4px solid #FBC02D; margin: 1rem 0; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initializes only the model and database status."""
    if 'ensemble_model' not in st.session_state:
        st.session_state.ensemble_model = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = initialize_database()


def get_risk_badge_html(risk_level, confidence=None):
    """Generates HTML for risk badge and confidence score."""
    risk_class = f"risk-{risk_level.replace(' ', '').lower()}"
    badge_html = f'<span class="risk-badge {risk_class}">{risk_level.upper()}</span>'
    
    if confidence is not None:
        confidence_pct = confidence * 100
        confidence_html = f'<span class="confidence-score">Confidence: {confidence_pct:.1f}%</span>'
        return badge_html + confidence_html
    
    return badge_html

def get_supportive_response(risk_level):
    """Generates appropriate informative text based on risk level."""
    responses = {
        'no risk': ["This post shows a generally positive or neutral tone regarding mental health. Continue to foster a safe and supportive environment."],
        'low': ["The content suggests minor distress or general low mood. While not immediate risk, ongoing support and check-ins are advisable."],
        'moderate': ["The post contains clear signs of distress or challenging emotions, indicating a moderate level of risk. This warrants attention and referral to resources."],
        'high': ["🚨 HIGH RISK CONTENT DETECTED. The language used suggests immediate or severe mental health concern. Crisis resources should be provided immediately."]
    }
    return random.choice(responses.get(risk_level, responses['moderate']))

def add_risk_based_resources(risk_level):
    """Display resources based on detected risk level."""
    if risk_level == 'high':
        st.markdown("""### 🚨 CRITICAL RESOURCE ALERT 🚨...""", unsafe_allow_html=True)
    elif risk_level == 'moderate':
        st.markdown("""### 💙 Recommended Support...""", unsafe_allow_html=True)
    elif risk_level == 'low':
        st.markdown("""### 🌱 General Wellness Resources...""", unsafe_allow_html=True)
    elif risk_level == 'no risk':
        st.markdown("""### ✅ Postivity Detected...""", unsafe_allow_html=True)


def main():
    """Main application function for single-post analysis"""
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">🧠 Mental Health Post Risk Detector</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Important Disclaimer:</strong> This tool uses an ensemble ML model...
    </div>
    """, unsafe_allow_html=True)
    
    # Check DB status once at the top
    if not st.session_state.db_initialized:
         st.warning("⚠️ **Database Logging is Disabled.** Ensure the `DATABASE_URL` environment variable is set if auditing is required.")

    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Initializing Model Service Connection..."):
            try:
                st.session_state.ensemble_model = load_ensemble_models()
                st.session_state.models_loaded = True
                st.success("✅ Model service connection successful!")
            except Exception as e:
                st.error(f"❌ Error connecting to model service: {str(e)}")
                st.info("Please ensure the Python model API is running on `http://127.0.0.1:5001/`.")
    
    # Main Analysis Interface
    st.markdown("---")
    st.subheader("📝 Text Input")
        
    user_input = st.text_area(
        "Paste the social media post (e.g., Reddit, X, etc.) here:",
        height=200,
        placeholder="e.g., 'I just can't take it anymore, everything feels pointless and heavy.'",
        key="analysis_input"
    )

    if st.button("Analyze Post Risk", use_container_width=True, key="submit_analysis") and user_input.strip():
        
        risk_level = 'unknown'
        confidence = 0.0
        
        if not st.session_state.models_loaded or not st.session_state.ensemble_model:
            st.error("Model service is not available. Please try again later.")
            return

        cleaned_input = clean_text_for_analysis(user_input)
        
        with st.spinner("Analyzing risk level..."):
            try:
                # Call the prediction method (which internally calls the API)
                risk_level, confidence = st.session_state.ensemble_model.predict(cleaned_input)
                risk_level = risk_level.replace(' ', '').lower() 
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                risk_level = 'moderate' 

        # 3. Log the Analysis (Audit Log)
        if st.session_state.db_initialized:
            success = log_post_analysis(
                content=user_input, 
                risk_level=risk_level, 
                confidence=confidence
            )
            if not success:
                 st.warning("⚠️ Warning: Could not log analysis to the database.")
        
        # 4. Display Results
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔍 Assessment Results")
        st.markdown(f"**Calculated Risk Level:** {get_risk_badge_html(risk_level, confidence)}", unsafe_allow_html=True)
        
        if risk_level != 'unknown':
            supportive_text = get_supportive_response(risk_level)
            st.info(supportive_text)
            st.markdown("---")
            add_risk_based_resources(risk_level)
        else:
            st.warning("Could not reliably determine risk level.")
            
    st.markdown("---")

if __name__ == "__main__":
    main()