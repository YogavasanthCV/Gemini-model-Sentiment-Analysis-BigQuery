from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("akshayjambhulkar/telecom-conversational-support-chat-pre-processed-with-agent")

# Convert dataset to DataFrame
df = pd.DataFrame(dataset['train'])  # Use the appropriate split

# Save DataFrame to Parquet file
df.to_parquet('C:/vasanth/Sentiment Analysis/RAW data/conversation_dataset_file.parquet')
