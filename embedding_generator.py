from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import csv
import os
import torch
import numpy as np

torch.cuda.empty_cache()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load dataset
df = pd.read_csv("CSV_for_embeddings/template_topic_noUsage.csv", header=None, names=["Filename", "Extracted Text"])
print(df.shape)
# get the first and second columns
# docs = df.iloc[:, 1].values
docs = df["Extracted Text"].astype(str).values
print(docs.shape)
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Generate embeddings
embeddings = embedding_model.encode(docs, convert_to_tensor=True)
np.save("embeddings/template_topic_noUsage.npy", embeddings.cpu().numpy())
print(embeddings.shape)