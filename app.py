import os
import streamlit as st
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import random
#my test
# ✅ Set Streamlit Page Config - Must be the first Streamlit command
st.set_page_config(
    page_title="DataSage - AI Data Science Tutor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure API Keys
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
UNSPLASH_API_KEY = st.secrets["UNSPLASH_API_KEY"]


# Initialize Gemini Flash
try:
    genai.configure(api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Error initializing Gemini Flash model: {e}")
    st.stop()

# Function to fetch 3D chatbot/AI images from Unsplash with improved queries
def get_unsplash_image():
    # List of specific queries to find 3D chatbot images
    queries = [
        "3d robot assistant",
        "3d digital assistant",
        "3d chatbot render",
        "3d ai virtual assistant",
        "3d hologram ai",
        "futuristic 3d ai"
    ]
    
    query = random.choice(queries)
    url = f"https://api.unsplash.com/photos/random?query={query}&client_id={UNSPLASH_API_KEY}"
    try:
        response = requests.get(url).json()
        return response.get("urls", {}).get("regular", "")
    except Exception as e:
        st.error(f"Error fetching image: {e}")
        return ""

# Custom CSS for unique UI
def load_css():
    # Glass morphism styles + floating elements + custom backgrounds
    st.markdown("""
    <style>
    /* Main body styling */
    .main {
        background: linear-gradient(135deg, #051937, #004d7a, #008793, #00bf72);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    /* Glass morphism for containers */
    div.css-1r6slb0.e1tzin5v2 {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Floating animation for the chat responses */
    .chat-bubble {
        animation: float 6s ease-in-out infinite;
        transform-origin: center;
    }
    
    @keyframes float {
        0% {transform: translateY(0px);}
        50% {transform: translateY(-10px);}
        100% {transform: translateY(0px);}
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 50px !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        padding: 15px 25px !important;
        font-size: 16px !important;
        background: rgba(255, 255, 255, 0.1) !important;
        color: black !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput>div>div>input:focus {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Custom button styling */
    .stButton button {
        border-radius: 50px !important;
        background: linear-gradient(45deg, #FF512F, #DD2476) !important;
        color: white !important;
        font-weight: bold !important;
        padding: 10px 30px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Custom header styling */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif !important;
        background: linear-gradient(to right, #FF512F, #DD2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    /* Dark mode styling */
    .dark-mode {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    
    /* Chat bubbles */
    .user-bubble {
        background: rgba(0, 123, 255, 0.2);
        border-radius: 20px 20px 0 20px;
        padding: 15px;
        margin: 10px 0;
        align-self: flex-end;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .ai-bubble {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px 20px 20px 0;
        padding: 15px;
        margin: 10px 0;
        align-self: flex-start;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load Google Fonts
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# Load CSS
load_css()

# Sidebar with glass morphism effect
with st.sidebar:
    st.markdown("<h2>⚙️ Console</h2>", unsafe_allow_html=True)
    
    # Profile selector with avatars
    profiles = {
        "Beginner": "👶",
        "Intermediate": "👨‍💻",
        "Advanced": "🧙‍♂️"
    }
    
    profile_options = [f"{emoji} {name}" for name, emoji in profiles.items()]
    selected_profile = st.selectbox("Your Expertise Level", profile_options, index=1)
    
    # Visual skill meter
    st.markdown("<h3>Skills Focus</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        python_level = st.slider("Python", 0, 100, 70)
        stats_level = st.slider("Statistics", 0, 100, 60)
    
    with col2:
        ml_level = st.slider("Machine Learning", 0, 100, 50)
        viz_level = st.slider("Visualization", 0, 100, 40)
    
    # Create visual progress bars
    def create_progress_bar(value, color):
        return f"""
        <div style="width:100%; background-color:rgba(255,255,255,0.1); border-radius:10px; height:10px; margin-bottom:10px;">
            <div style="width:{value}%; background-color:{color}; height:10px; border-radius:10px;"></div>
        </div>
        """
    
    st.markdown(f"""
    <div style="margin-top:20px;">
        <p>Python {create_progress_bar(python_level, '#36d7b7')}</p>
        <p>Statistics {create_progress_bar(stats_level, '#9b59b6')}</p>
        <p>Machine Learning {create_progress_bar(ml_level, '#3498db')}</p>
        <p>Visualization {create_progress_bar(viz_level, '#f39c12')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme settings
    st.markdown("<h3>🎨 Theme Settings</h3>", unsafe_allow_html=True)
    theme_options = ["Cosmic Gradient", "Dark Elegant", "Light Minimal", "Matrix Vibes"]
    selected_theme = st.selectbox("Select Theme", theme_options)
    
    # Dark Mode Toggle with sun/moon icons
    dark_mode = st.checkbox("🌙 Dark Mode")
    if dark_mode:
        st.markdown("""
        <style>
        body {
            color: white;
        }
        .dark-mode-active {
            display: block;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Language selection with flags
    st.markdown("<h3>🌍 Language</h3>", unsafe_allow_html=True)
    languages = {
        "English": "🇺🇸", 
        "Hindi": "🇮🇳", 
        "Spanish": "🇪🇸", 
        "French": "🇫🇷", 
        "German": "🇩🇪"
    }
    
    lang_options = [f"{emoji} {lang}" for lang, emoji in languages.items()]
    selected_lang = st.selectbox("Choose Language", lang_options)
    
    if selected_lang != "🇺🇸 English":
        st.info("Translation feature coming soon! 🚀")

# Main content area
# Get 3D AI image
image_url = get_unsplash_image()

# Split into columns for asymmetrical layout
col1, col2 = st.columns([2, 1])

with col1:
    # Main header with animation
    st.markdown("""
    <div style="text-align:center; animation: fadeIn 1.5s ease-in-out;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0;">DataSage 🧠</h1>
        <p style="font-size: 1.5rem; opacity: 0.8;">Your AI Data Science Tutor</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Define Prompt Template
    prompt_template = PromptTemplate(
        input_variables=["question", "profile"],
        template="""You are a friendly AI tutor specializing in Data Science. If a user asks about non-data science topics, politely inform them that you only answer data science-related questions.

        User's expertise level: {profile}
        
        **User Question:** {question}

        **AI Tutor Response:**
        """
    )

    # LangChain LLMChain
    qa_chain = prompt_template | llm

    def get_response(question, profile):
        try:
            response = qa_chain.invoke({"question": question, "profile": profile})

            # Handle response based on its structure
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, dict):
                content = response.get("content", "")
                if content:
                    return content.strip()
                
                ai_response = response.get("response", "")
                if ai_response:
                    return ai_response.strip()

            response_str = str(response)
            if "content=" in response_str:
                clean_content = response_str.split("content=")[1].strip('"')
                if "additional_kwargs" in clean_content:
                    clean_content = clean_content.split("additional_kwargs")[0].strip()
                return clean_content.strip()

            return response_str
        except Exception as e:
            return f"⚠️ An error occurred: {e}"

with col2:
    # Display the 3D AI image with a floating effect
    if image_url:
        st.markdown("""
        <div class="chat-bubble" style="text-align:center;">
        """, unsafe_allow_html=True)
        st.image(image_url, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Custom chat input area with gradient and animation
st.markdown("""
<div style="max-width: 800px; margin: 30px auto; position: relative;">
    <div style="position: absolute; top: -35px; left: 20px; background: linear-gradient(45deg, #FF512F, #DD2476); border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
        <span style="color: white; font-size: 20px;">💬</span>
    </div>
</div>
""", unsafe_allow_html=True)

# User input with a floating effect
user_question = st.text_input("", placeholder="Ask me anything about Data Science...")
send_button = st.button("Send 🚀")

# Chat history with glass morphism bubbles
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

profile = selected_profile.split(" ")[1] if " " in selected_profile else selected_profile

if user_question and send_button:
    response = get_response(user_question, profile)
    
    # Add to chat history
    st.session_state.chat_history.append({
        "question": user_question,
        "answer": response
    })

# Display chat history with enhanced bubble design
st.markdown("<h3>Conversation</h3>", unsafe_allow_html=True)

if st.session_state.chat_history:
    for i, chat in enumerate(st.session_state.chat_history):
        # User message
        st.markdown(f"""
        <div class="user-bubble">
            <strong>You:</strong> {chat["question"]}
        </div>
        """, unsafe_allow_html=True)
        
        # AI response with floating animation
        st.markdown(f"""
        <div class="ai-bubble chat-bubble">
            <strong>DataSage:</strong> {chat["answer"]}
        </div>
        """, unsafe_allow_html=True)
else:
    # Welcome message with suggestions
    st.markdown("""
    <div class="ai-bubble chat-bubble">
        <strong>DataSage:</strong> 👋 Hi there! I'm your AI Data Science tutor. Ask me anything about Python, statistics, machine learning, or data visualization!
        
        <p>Try asking questions like:</p>
        <ul>
            <li>How do I handle missing data in pandas?</li>
            <li>Explain random forest algorithm</li>
            <li>What's the difference between classification and regression?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer with visual elements
st.markdown("""
<div style="position: fixed; bottom: 0; left: 0; width: 100%; background: rgba(0,0,0,0.1); backdrop-filter: blur(10px); padding: 10px; text-align: center; font-size: 12px; color: rgba(255,255,255,0.7);">
    <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
        <div>🧠 Powered by Gemini 2.0</div>
        <div>🖼️ Images via Unsplash</div>
        <div>🚀 Made with Streamlit</div>
    </div>
</div>
""", unsafe_allow_html=True)
