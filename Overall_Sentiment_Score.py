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
SELECT conversation_id , ml_generate_text_llm_result, text, prompt,   
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
            "Text": text,
            "Suggestion": suggestion
        })
    
    return pd.DataFrame(suggestions)

# Streamlit app layout
st.title("Sentiment Analysis Results")

# Fetch data
data = fetch_data()

# Create a sidebar for filters
st.sidebar.header("Filter Options")
sentiment_filter = st.sidebar.selectbox(
    "Select Sentiment",
    options=["All", "Positive", "Negative", "Neutral"]
)

# Filter data based on user selection
if sentiment_filter != "All":
    filtered_data = data[data['ml_generate_text_llm_result'].str.contains(sentiment_filter, case=False, na=False)]
else:
    filtered_data = data

# Display the filtered data
st.write(filtered_data)

# Analyze and display improvement suggestions if negative sentiment is selected
if sentiment_filter == "Negative":
    st.subheader("Improvement Suggestions")
    suggestions_df = analyze_negative_sentiments(filtered_data)
    st.write(suggestions_df)
