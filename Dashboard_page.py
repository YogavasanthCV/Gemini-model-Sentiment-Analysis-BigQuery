import streamlit as st
from google.cloud import bigquery
import pandas as pd
import google.generativeai as genai
import os

# Configure the Google Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Google Gemini model and get improvement suggestions
def get_gemini_response(text):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"Based on the following negative feedback: '{text}', how can the agent improve to achieve a positive score? What are the steps to take for improvisation?"
    response = model.generate_content(prompt)
    return response.text

# Initialize BigQuery client
client = bigquery.Client()

# Define the BigQuery SQL query
query = """
SELECT conversation_id, ml_generate_text_llm_result, text, prompt
FROM `support-agent-rating.sentiment_analysis_dataset.review_sentiment_analysis`
"""

# Fetch data from BigQuery
def fetch_data():
    query_job = client.query(query)  # Make an API request.
    return query_job.result().to_dataframe()

# Analyze negative sentiments and get improvement suggestions
def analyze_negative_sentiments(data):
    negative_data = data[data['ml_generate_text_llm_result'].str.contains("Negative", case=False, na=False)]
    suggestions = []

    for _, row in negative_data.iterrows():
        text = row['text']
        suggestion = get_gemini_response(text)
        suggestions.append({
            "Conversation ID": row['conversation_id'],
            "Text": text,
            "Suggestion": suggestion
        })
    
    return pd.DataFrame(suggestions)
##---------------------------------------------------------------

st.title("Sentiment Analysis Dashboard")

# Fetch data
data = fetch_data()

# Login Page
st.sidebar.header("Login")
entered_conversation_id = st.sidebar.text_input("Enter Conversation ID")

# Validate Conversation ID
if entered_conversation_id in data['conversation_id'].values:
    st.header(f"Conversation Details for ID: {entered_conversation_id}")
    
    # Filter data based on entered conversation_id
    filtered_data = data[data['conversation_id'] == entered_conversation_id]
    
    if not filtered_data.empty:
        st.write(filtered_data)
        
        # Analyze and display improvement suggestions
        suggestions_df = analyze_negative_sentiments(filtered_data)
        if not suggestions_df.empty:
            st.subheader("Improvement Suggestions")
            st.write(suggestions_df)
        else:
            st.write("No negative sentiments found for this conversation.")
    else:
        st.write("No data available for the entered Conversation ID.")
else:
    st.write("Please enter a valid Conversation ID to view the details.")
