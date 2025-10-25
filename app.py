import streamlit as st
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import os
import json
import csv
import io
from model_utils import MentalHealthEnsemble, load_ensemble_models

from text_preprocessing import clean_text_for_analysis
from db_utils import (initialize_database, generate_session_id, save_message, 
                      load_conversation, get_all_sessions, delete_conversation,
                      get_conversation_stats)
import time

# Configure page
st.set_page_config(
    page_title="Mental Health Support Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Custom CSS for mental health themed styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E7D32;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 10px;
        background-color: #F8F9FA;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #E3F2FD;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        margin-left: 20%;
        text-align: right;
    }
    
    .bot-message {
        background-color: #E8F5E9;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        margin-right: 20%;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 10px;
    }
    
    .risk-no { background-color: #C8E6C9; color: #2E7D32; }
    .risk-low { background-color: #FFF3E0; color: #F57C00; }
    .risk-moderate { background-color: #FFE0B2; color: #E65100; }
    .risk-high { background-color: #FFCDD2; color: #C62828; }
    
    .confidence-score {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        background-color: #E0E0E0;
        color: #424242;
        margin-left: 8px;
    }
    
    .disclaimer {
        background-color: #FFF9C4;
        padding: 1rem;
        border-left: 4px solid #FBC02D;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'ensemble_model' not in st.session_state:
        st.session_state.ensemble_model = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_session_id()
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = initialize_database()

def get_risk_badge_html(risk_level, confidence=None):
    """Generate HTML for risk level badge with optional confidence score"""
    risk_class = f"risk-{risk_level.replace(' ', '').lower()}"
    badge_html = f'<span class="risk-badge {risk_class}">{risk_level.upper()}</span>'
    
    if confidence is not None:
        confidence_pct = confidence * 100
        confidence_html = f'<span class="confidence-score">Confidence: {confidence_pct:.1f}%</span>'
        return badge_html + confidence_html
    
    return badge_html

def display_chat_history():
    """Display chat messages with risk assessments"""
    if st.session_state.messages:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            if message['role'] == 'user':
                risk_level = message.get('risk_level', 'unknown')
                confidence = message.get('confidence', None)
                risk_badge = get_risk_badge_html(risk_level, confidence)
                st.markdown(f'''
                    <div class="user-message">
                        <strong>You:</strong> {message['content']}
                        {risk_badge}
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div class="bot-message">
                        <strong>Assistant:</strong> {message['content']}
                    </div>
                ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def get_supportive_response(risk_level):
    """Generate appropriate supportive response based on risk level"""
    responses = {
        'no risk': [
            "It's wonderful to hear such positive thoughts! Keep nurturing this positive mindset.",
            "Your message radiates positivity. That's a great outlook to maintain!",
            "I'm glad you're feeling good. Continue with whatever is working well for you!"
        ],
        'low': [
            "I hear you, and it's okay to have these feelings sometimes. Would you like to talk more about what's on your mind?",
            "Thank you for sharing. Remember that it's normal to have ups and downs. How can I support you today?",
            "I appreciate you opening up. Sometimes talking through our thoughts can be helpful."
        ],
        'moderate': [
            "I can sense you're going through a challenging time. Please know that you're not alone, and these feelings are valid.",
            "It sounds like things are difficult right now. Would it help to talk through what you're experiencing?",
            "Thank you for trusting me with your feelings. Managing difficult emotions takes strength."
        ],
        'high': [
            "I'm concerned about how you're feeling right now. Please remember that you matter and there are people who want to help.",
            "Your feelings are important, and I want you to know that support is available. Consider reaching out to a mental health professional.",
            "I hear your pain, and I'm worried about you. Please consider contacting a crisis helpline if you're having thoughts of self-harm."
        ]
    }
    
    import random
    return random.choice(responses.get(risk_level, responses['moderate']))

def add_risk_based_resources(risk_level):
    """Display resources based on detected risk level"""
    if risk_level == 'high':
        st.markdown("""
        ### 🚨 Crisis Resources
        
        **If you're in immediate danger or having thoughts of self-harm:**
        - **Crisis Text Line:** Text HOME to 741741
        - **National Suicide Prevention Lifeline:** 988 or 1-800-273-8255
        - **Emergency Services:** 911
        
        **Additional Support:**
        - **NAMI (National Alliance on Mental Illness):** 1-800-950-NAMI (6264)
        - **SAMHSA National Helpline:** 1-800-662-4357
        
        **Remember:** You are not alone. These feelings are temporary, and help is available.
        """)
    
    elif risk_level == 'moderate':
        st.markdown("""
        ### 💙 Support Resources
        
        **Professional Help:**
        - **Psychology Today Therapist Finder:** [Find a therapist near you](https://www.psychologytoday.com/us/therapists)
        - **BetterHelp Online Therapy:** Accessible, affordable counseling
        - **NAMI Helpline:** 1-800-950-NAMI (6264)
        
        **Self-Care Strategies:**
        - Practice mindfulness and meditation
        - Maintain regular sleep schedule
        - Stay connected with supportive friends/family
        - Engage in physical activity
        
        **Educational Resources:**
        - **NAMI:** Learn about mental health conditions
        - **Mental Health America:** Screening tools and information
        """)
    
    elif risk_level == 'low':
        st.markdown("""
        ### 🌱 Wellness Resources
        
        **Mental Wellness Tips:**
        - Maintain a consistent sleep schedule
        - Practice gratitude journaling
        - Stay physically active
        - Connect with loved ones
        
        **Helpful Apps:**
        - Headspace or Calm for meditation
        - Moodpath for mood tracking
        - Sanvello for stress management
        
        **Educational Content:**
        - TED Talks on mental wellness
        - Podcasts about mental health
        - Mental Health America resources
        """)
    
    elif risk_level == 'no risk':
        st.markdown("""
        ### ✨ Maintain Your Wellness
        
        **Keep Up the Good Work:**
        - Continue your positive habits
        - Share your positivity with others
        - Help someone who might be struggling
        
        **Growth Opportunities:**
        - Learn new coping strategies
        - Explore mindfulness practices
        - Build resilience for future challenges
        
        **Stay Informed:**
        - Mental Health First Aid training
        - Become a mental health advocate
        """)
    
def add_crisis_resources():
    """Display crisis resources for high-risk situations - legacy function"""
    add_risk_based_resources('high')

def export_conversation_to_csv():
    """Export conversation history to CSV format"""
    if not st.session_state.messages:
        return None
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Timestamp', 'Role', 'Message', 'Risk Level', 'Confidence'])
    
    # Write data
    for msg in st.session_state.messages:
        timestamp = msg.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
        role = msg.get('role', '')
        content = msg.get('content', '')
        risk_level = msg.get('risk_level', '') if msg.get('role') == 'user' else ''
        confidence = f"{msg.get('confidence', 0):.2%}" if msg.get('role') == 'user' and 'confidence' in msg else ''
        
        writer.writerow([timestamp, role, content, risk_level, confidence])
    
    return output.getvalue()

def export_conversation_to_json():
    """Export conversation history to JSON format"""
    if not st.session_state.messages:
        return None
    
    # Convert messages to JSON-serializable format
    export_data = []
    for msg in st.session_state.messages:
        msg_copy = msg.copy()
        # Convert datetime to string
        if 'timestamp' in msg_copy:
            msg_copy['timestamp'] = msg_copy['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        export_data.append(msg_copy)
    
    return json.dumps(export_data, indent=2)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Sidebar for session management
    with st.sidebar:
        st.title("📚 Session Management")
        
        # Current session info
        st.write(f"**Current Session:** {st.session_state.session_id[:8]}...")
        
        # New session button
        if st.button("🆕 Start New Session", use_container_width=True):
            st.session_state.session_id = generate_session_id()
            st.session_state.messages = []
            st.rerun()
        
        # Load previous session
        st.markdown("---")
        st.subheader("Previous Sessions")
        
        if st.session_state.db_initialized:
            sessions = get_all_sessions()
            if sessions:
                for sess in sessions[:10]:  # Show last 10 sessions
                    session_display = f"{sess['session_id'][:8]}... ({sess['updated_at'].strftime('%m/%d %H:%M')})"
                    if st.button(session_display, key=f"load_{sess['session_id']}", use_container_width=True):
                        st.session_state.session_id = sess['session_id']
                        st.session_state.messages = load_conversation(sess['session_id'])
                        st.rerun()
            else:
                st.info("No previous sessions found")
        else:
            st.warning("Database not available")
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Mental Health Support Assistant</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Important Disclaimer:</strong> This is an AI assistant designed to provide supportive conversation 
        and basic mental health resources. It is not a substitute for professional mental health care, therapy, 
        or medical advice. If you're experiencing a mental health crisis, please contact emergency services 
        or a crisis helpline immediately.
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading mental health analysis models... This may take a moment."):
            try:
                st.session_state.ensemble_model = load_ensemble_models()
                st.session_state.models_loaded = True
                st.success("✅ Models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                st.info("The application will continue with limited functionality.")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display chat history
        display_chat_history()
        
        # Input area
        with st.form(key="message_form", clear_on_submit=True):
            user_input = st.text_area(
                "Share your thoughts or feelings:",
                height=100,
                placeholder="I'm here to listen. Feel free to share what's on your mind..."
            )
            submit_button = st.form_submit_button("Send Message")
        
        if submit_button and user_input.strip():
            # Process user message
            cleaned_input = clean_text_for_analysis(user_input)
            
            # Analyze risk level
            risk_level = 'unknown'
            confidence = 0.0
            
            if st.session_state.models_loaded and st.session_state.ensemble_model:
                try:
                    risk_level, confidence = st.session_state.ensemble_model.predict(cleaned_input)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    risk_level = 'moderate'  # Default to moderate for safety
            
            # Add user message to history
            st.session_state.messages.append({
                'role': 'user',
                'content': user_input,
                'risk_level': risk_level,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            # Save user message to database
            if st.session_state.db_initialized:
                save_message(
                    st.session_state.session_id,
                    'user',
                    user_input,
                    risk_level,
                    confidence
                )
            
            # Generate bot response
            bot_response = get_supportive_response(risk_level)
            st.session_state.messages.append({
                'role': 'assistant',
                'content': bot_response,
                'timestamp': datetime.now()
            })
            
            # Save bot message to database
            if st.session_state.db_initialized:
                save_message(
                    st.session_state.session_id,
                    'assistant',
                    bot_response
                )
            
            # Rerun to update display
            st.rerun()
    
    with col2:
        st.subheader("📊 Analysis Summary")
        
        # Add tabs for current session vs analytics
        tab1, tab2 = st.tabs(["Current Session", "Analytics"])
        
        with tab1:
            if st.session_state.messages:
                # Filter user messages for analysis
                user_messages = [msg for msg in st.session_state.messages if msg['role'] == 'user']
                
                if user_messages:
                    # Risk level distribution
                    risk_levels = [msg.get('risk_level', 'unknown') for msg in user_messages]
                    risk_counts = pd.Series(risk_levels).value_counts()
                    
                    st.write("**Risk Level Distribution:**")
                    for risk, count in risk_counts.items():
                        percentage = (count / len(user_messages)) * 100
                        st.write(f"- {risk.title()}: {count} ({percentage:.1f}%)")
                    
                    # Latest assessment
                    if user_messages:
                        latest_msg = user_messages[-1]
                        st.write("**Latest Assessment:**")
                        st.write(f"Risk Level: **{latest_msg.get('risk_level', 'unknown').title()}**")
                        if 'confidence' in latest_msg:
                            st.write(f"Confidence: {latest_msg['confidence']:.2%}")
            else:
                st.info("Start a conversation to see analysis")
        
        with tab2:
            st.write("**Session Analytics**")
            if st.session_state.db_initialized:
                stats = get_conversation_stats(st.session_state.session_id)
                if stats:
                    st.metric("Total Messages", stats['total_messages'])
                    st.metric("Avg Confidence", f"{stats['average_confidence']:.2%}")
                    
                    if stats['risk_distribution']:
                        st.write("**Risk Distribution:**")
                        for risk, count in stats['risk_distribution'].items():
                            st.write(f"- {risk.title()}: {count}")
                else:
                    st.info("No analytics available for current session")
            else:
                st.warning("Database required for analytics")
        
        # Risk-based resources
        if st.session_state.messages:
            user_messages = [msg for msg in st.session_state.messages if msg['role'] == 'user']
            if user_messages:
                latest_risk = user_messages[-1].get('risk_level', 'unknown')
                if latest_risk != 'unknown':
                    st.markdown("---")
                    add_risk_based_resources(latest_risk)
        
        # Export and clear buttons
        st.markdown("---")
        st.write("**Conversation Actions:**")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.session_state.messages:
                csv_data = export_conversation_to_csv()
                if csv_data:
                    st.download_button(
                        label="📥 Export CSV",
                        data=csv_data,
                        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        with col_export2:
            if st.session_state.messages:
                json_data = export_conversation_to_json()
                if json_data:
                    st.download_button(
                        label="📥 Export JSON",
                        data=json_data,
                        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
