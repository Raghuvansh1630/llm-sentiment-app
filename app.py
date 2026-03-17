import streamlit as st
import pandas as pd
from transformers import pipeline

# --- 1. SET UP THE APP ---
st.set_page_config(page_title="LLM Sentiment Analyzer", layout="wide")
st.title("🤖 LLM Sentiment Analysis")
st.write("Use a pre-trained Large Language Model (Hugging Face) to understand text context.")

# --- 2. LOAD THE LLM ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

with st.spinner("Loading the LLM into memory..."):
    sentiment_analyzer = load_model()

# --- 3. APP SETTINGS (SIDEBAR) ---
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.50, 
    max_value=1.00, 
    value=0.70, 
    help="If confidence is below this, the result is 'Neutral/Unsure'."
)

# --- 4. TABS SETUP ---
tab1, tab2 = st.tabs(["📝 Single Sentence", "📂 Bulk CSV Upload"])

# --- TAB 1: SINGLE SENTENCE ---
with tab1:
    st.subheader("Test a Single Sentence")
    user_input = st.text_area("Type a sentence or review here:", "The product is decent, but shipping took forever.")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            result = sentiment_analyzer(user_input)[0]
            label, confidence = result['label'], result['score']
            
            if confidence < threshold:
                st.warning(f"**Prediction:** NEUTRAL / UNSURE 🟡 (Confidence: {confidence:.2%})")
            elif label == "POSITIVE":
                st.success(f"**Prediction:** {label} 🟢 (Confidence: {confidence:.2%})")
            else:
                st.error(f"**Prediction:** {label} 🔴 (Confidence: {confidence:.2%})")
        else:
            st.warning("Please enter some text.")

# --- TAB 2: BULK UPLOAD ---
with tab2:
    st.subheader("Analyze an Entire Dataset")
    st.write("Upload a CSV file. The model will analyze the first 100 rows to prevent server crashes.")
    
    # 1. The File Uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # 2. Read the CSV using Pandas
        df = pd.read_csv(uploaded_file)
        
        # Limit to 100 rows so the free server doesn't time out
        df = df.head(100) 
        
        st.write("Preview of your data:")
        st.dataframe(df.head())
        
        # 3. Let the user pick which column has the text
        text_column = st.selectbox("Which column contains the text you want to analyze?", df.columns)
        
        if st.button("Analyze Entire CSV"):
            with st.spinner("The LLM is reading your data... This takes a few seconds."):
                
                # Convert the chosen column to a list of strings
                texts = df[text_column].astype(str).tolist()
                
                # Run the LLM on all rows at once
                results = sentiment_analyzer(texts)
                
                # Add the results as new columns in the dataframe
                df['LLM_Sentiment'] = [res['label'] for res
