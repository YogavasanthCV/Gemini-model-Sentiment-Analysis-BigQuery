import streamlit as st
from google.cloud import bigquery

# Initialize BigQuery client
client = bigquery.Client()

# Define the BigQuery SQL query
query = """
SELECT ml_generate_text_llm_result, text, prompt,   
FROM `support-agent-rating.sentiment_analysis_dataset.review_sentiment_analysis`
"""

# Fetch data from BigQuery
def fetch_data():
    query_job = client.query(query)  # Make an API request.
    return query_job.result().to_dataframe()

# Streamlit app layout
st.title("Sentiment Analysis Results")

# Fetch and display data
data = fetch_data()
st.write(data)
