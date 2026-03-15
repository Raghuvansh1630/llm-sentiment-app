import streamlit as st
from transformers import pipeline

# --- 1. SET UP THE APP ---
st.set_page_config(page_title="LLM Sentiment Analyzer", layout="centered")
st.title("🤖 LLM Sentiment Analysis")
st.write("This dashboard uses a pre-trained Large Language Model (Hugging Face) to understand the context of your text.")

# --- 2. LOAD THE LLM ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

with st.spinner("Loading the LLM into memory..."):
    sentiment_analyzer = load_model()

# --- 3. APP SETTINGS (NEW FEATURE) ---
st.divider()
st.subheader("⚙️ App Settings")
st.write("Adjust how confident the model needs to be to make a definitive prediction.")

# The Slider Element
threshold = st.slider(
    "Confidence Threshold", 
    min_value=0.50, 
    max_value=1.00, 
    value=0.70, 
    help="If the model's confidence is below this number, the result will be marked as 'Neutral/Unsure'."
)

# --- 4. APP INTERFACE ---
st.divider()
st.subheader("Test the Model")

user_input = st.text_area(
    "Type a sentence or review here:", 
    "The food was okay, I guess."
)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Run the LLM
        result = sentiment_analyzer(user_input)[0]
        
        # Extract data
        label = result['label']
        confidence = result['score']
        
        # NEW LOGIC: Check against the slider threshold
        if confidence < threshold:
            st.warning(f"**Prediction:** NEUTRAL / UNSURE 🟡")
            st.info(f"**Model Confidence:** {confidence:.2%} (Below your {threshold:.2%} threshold)")
        elif label == "POSITIVE":
            st.success(f"**Prediction:** {label} 🟢")
            st.info(f"**Model Confidence:** {confidence:.2%}")
        else:
            st.error(f"**Prediction:** {label} 🔴")
            st.info(f"**Model Confidence:** {confidence:.2%}")
            
    else:
        st.warning("Please enter some text to analyze.")
