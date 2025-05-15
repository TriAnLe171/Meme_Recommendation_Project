from transformers import pipeline
from transformers import AutoTokenizer
import pandas as pd
import csv
import os
import torch

model_name = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = tokenizer.model_max_length

df = pd.read_csv('#REPLACE WITH INPUT .CSV FILE',header=None,names=["Filename", "Extracted Text + Title"])

emotion_labels = [
    "joy", "anticipation", "disgust", "sadness", "anger", 
    "optimism", "surprise", "pessimism", "fear", "trust", "love"
]
sentiment_labels = ["neutral", "positive", "negative"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe_1 = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", top_k=None)
pipe_2 = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None)

def get_sentiment(text):
    text = text[:max_length]  
    result_emotions = pipe_1(text)[0]  
    result_sentiment = pipe_2(text)[0]

    scores = {label: 0.0 for label in emotion_labels + sentiment_labels}

    scores.update({item['label']: item['score'] for item in result_emotions})  
    scores.update({item['label']: item['score'] for item in result_sentiment})  

    return [scores[label] for label in (emotion_labels + sentiment_labels)]  

output_file = "#REPLACE WITH OUTPUT .CSV FILE"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    header = ["Filename", "Extracted Text + Title"] + emotion_labels + sentiment_labels
    writer.writerow(header)
    for index, row in df.iterrows():
        try:
            filename, caption = row["Filename"], str(row["Extracted Text + Title"])
            sentiment_scores = get_sentiment(caption)
            
            writer.writerow([filename, caption] + sentiment_scores)

            if index % 100 == 0:  
                print(f"Processed {index}/{len(df)} rows...")
        except Exception as e:
            # save error file
            error_file = "error_rows.csv"
            with open(error_file, "w", newline="", encoding="utf-8") as ef:
                error_writer = csv.writer(ef)
                error_writer.writerow(["Filename", "Extracted Text + Title", "Error"])
                error_writer.writerow([filename, caption, str(e)])
            print(f"Error processing row {index}: {e}")
