import torch
import gc
import os
import pandas as pd
from transformers import pipeline
from datasets import Dataset
from tqdm.auto import tqdm

device = 0 if torch.cuda.is_available() else -1
batch_size = 64

i = 177407

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device,
    batch_size=batch_size,
    truncation=True,
    max_length=512,  
    padding='max_length'
)

df = pd.read_csv("full_context_300.csv", names=["Filename", "full_context"])
if i > 0:
    # drop rows before resume_index and reset pandas index
    df = df.iloc[i:].reset_index(drop=True)
    print(f"Resuming processing from row {i} (total rows now {len(df)})")    
dataset = Dataset.from_pandas(df)

candidate_labels = [
    "Comparison or Contrast",
    "Before and After Transformation",
    "Hyperbole or Exaggeration",
    "Punchline or Joke",
    "Wordplay or Pun",
    "Absurd or Random Humor",
    "Sarcasm or Irony",
    "Self-Deprecation",
    "Parody or Spoof",
    "Political Satire",
    "Social Commentary",
    "Media or Brand Critique",
    "Relationship Commentary",
    "Work or School Frustration",
    "Emotional Frustration",
    "Joy or Excitement",
    "Sadness or Disappointment",
    "Anxiety or Stress",
    "Confusion or Disbelief",
    "Common Life Experience",
    "Current Events or Pop Culture",
    "Celebrity Mockery",
    "Trolling or Baiting",
    "Reaction or Reply Meme",
    "Popular Meme Format",
    "Existential or Ethical Reflection",
    "Product or Brand Promotion",
    "Holiday or Event Participation"
]

print(len(candidate_labels), "candidate labels")

threshold = 0.7
output_file = "usage_results.csv"

# open(output_file, 'w').close()

results_buffer = []

def classify_batch(batch):
    results = classifier(batch["full_context"], candidate_labels, multi_label=True)
    for filename, res in zip(batch["Filename"], results):
        labels = [label for label, score in zip(res["labels"], res["scores"]) if score > threshold]
        if not labels:
            max_idx = res["scores"].index(max(res["scores"]))
            labels = [res["labels"][max_idx]]
            #log the filename and the label <0.7
            with open("under_0.7.txt", "a") as f:
                f.write(f"{filename}: {res['labels'][max_idx]} - {res['scores'][max_idx]}\n")
        results_buffer.append((filename, "|".join(labels)))

    if len(results_buffer) >= 100:
        df_out = pd.DataFrame(results_buffer, columns=["Filename", "Usage"])
        df_out.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
        results_buffer.clear()

    torch.cuda.empty_cache()
    gc.collect()

    return batch

_ = dataset.map(
    classify_batch,
    batched=True,
    batch_size=batch_size,
    desc=f"🚀 Classifying in batches of {batch_size}"
)

if results_buffer:
    df_out = pd.DataFrame(results_buffer, columns=["Filename", "Usage"])
    df_out.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))

print("✅ Done. Results saved to:", output_file)