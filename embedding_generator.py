from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import csv
import os
import torch
import numpy as np

torch.cuda.empty_cache()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

df = pd.read_csv("CSV_for_embeddings/template_topic_noUsage.csv", header=None, names=["Filename", "Extracted Text"])
print(df.shape)
docs = df["Extracted Text"].astype(str).values
print(docs.shape)
embedding_model = SentenceTransformer("all-mpnet-base-v2")

embeddings = embedding_model.encode(docs, convert_to_tensor=True)
np.save("embeddings/template_topic_noUsage.npy", embeddings.cpu().numpy())
print(embeddings.shape)