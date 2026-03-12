import streamlit as st
from transformers import pipeline

# --- 1. SET UP THE APP ---
st.set_page_config(page_title="LLM Sentiment Analyzer", layout="centered")
st.title("🤖 LLM Sentiment Analysis")
st.write("This dashboard uses a pre-trained Large Language Model (Hugging Face) to understand the context of your text.")

# --- 2. LOAD THE LLM ---
# @st.cache_resource ensures the heavy model is only downloaded and loaded into memory ONCE.
@st.cache_resource
def load_model():
    # By default, this loads a distilled BERT model fine-tuned for sentiment analysis
    return pipeline("sentiment-analysis")

with st.spinner("Loading the LLM into memory... (This takes a moment on startup)"):
    sentiment_analyzer = load_model()

# --- 3. APP INTERFACE ---
st.divider()
st.subheader("Test the Model")

# User Input
user_input = st.text_area(
    "Type a sentence or review here:", 
    "The food was a bit cold, but the waiters were so incredibly sweet!"
)

# Prediction Button
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Run the LLM
        result = sentiment_analyzer(user_input)[0]
        
        # Extract data
        label = result['label']
        confidence = result['score']
        
        # Display Results
        if label == "POSITIVE":
            st.success(f"**Prediction:** {label} 🟢")
            st.info(f"**Model Confidence:** {confidence:.2%}")
        else:
            st.error(f"**Prediction:** {label} 🔴")
            st.info(f"**Model Confidence:** {confidence:.2%}")
            
    else:
        st.warning("Please enter some text to analyze.")

# --- 4. HOW IT WORKS ---
st.divider()
with st.expander("How is this different from the first version?"):
    st.write("""
    * **No Training Data Needed:** We didn't have to provide a list of hardcoded sentences. The model was pre-trained on massive datasets.
    * **Context Aware:** Unlike `CountVectorizer` which just counts words, this Transformer model understands sentence structure. Try giving it sarcastic text or mixed reviews!
    """)