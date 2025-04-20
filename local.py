import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import imagehash
from PIL import Image
import shutil

from sklearn.metrics.pairwise import cosine_similarity

df_filename_noTemplate_local = pd.read_csv("filenames/filenames_noTemplate_local.csv", names=["Filename"])
df_filename_noTemplate_global = pd.read_csv("filenames/filenames_noTemplate_global.csv", names=["Filename"])
df_filename_withTemplate = pd.read_csv("filenames/filenames_withTemplate.csv", names=["Filename"])

meme_dataset_path = "/home/hsdslab/Documents/Meme_project_TriAn/Meme_Recommendation_Final/images"
meme_template_path = "/home/hsdslab/Documents/Meme_project_TriAn/Meme_Recommendation_Final/all_images_IMGFlip_templates"

text_model = SentenceTransformer("all-mpnet-base-v2")

sentiment_categories = ["joy", "anticipation", "disgust", "sadness", "anger", "optimism", "surprise", "pessimism", "fear", "trust", "love", "neutral", "positive", "negative"]

def get_image_path(meme_index, need_template, search_by="Global"):
    """Get the image path of a meme given its index."""
    if need_template:
        fileName = df_filename_withTemplate["Filename"][meme_index]
        meme_path = meme_template_path
    else:
        if search_by == "Local":
            fileName = df_filename_noTemplate_local["Filename"][meme_index]
        else:  # Global
            fileName = df_filename_noTemplate_global["Filename"][meme_index]
        meme_path = meme_dataset_path
    return os.path.join(meme_path, fileName)

def normalize_scores(scores):
    """Normalize similarity scores between 0 and 1."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(scores.reshape(-1, 1)).flatten()


def compute_image_hash(image_path):
    if image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
        try:
            with Image.open(image_path) as img:
                img_hash = imagehash.phash(img)
                return str(img_hash)
        except Exception as e:
            with open("error_log/error_log.txt", "a") as log_file:
                log_file.write(f"Error processing {image_path}: {e}\n")
                return None


def get_similar_memes(topics = None, need_template=None, usages=None, search_by="Global", sentiment_preference='joy', top_n=20,top_n_template=10):
    """Recommend top-N memes based on text input similarity."""
    if topics and usages:
        topic_embedding = text_model.encode(topics).reshape(1, -1)

        if need_template:
            df_mapping = pd.read_csv("meme_to_template_map_updated.csv", names=["Filename", "Template"])
            embeddings_path = "embeddings/template_topic_usage.npy"

            current_filenames = df_mapping["Filename"].tolist()
            current_mapping = df_mapping.set_index("Filename")["Template"].to_dict()
        else:
            if search_by == "Local":
                embeddings_path = "embeddings/noTemplate_topicLocal_usage.npy"
                current_filenames = df_filename_noTemplate_local["Filename"].tolist()
            else:
                embeddings_path = "embeddings/noTemplate_topicGlobal_usage.npy"
                current_filenames = df_filename_noTemplate_global["Filename"].tolist()

            current_mapping = None  

    elif topics and not usages:
        if need_template:
            df_mapping = pd.read_csv("meme_to_template_map_updated.csv", names=["Filename", "Template"])
            embeddings_path = "embeddings/template_topic_noUsage.npy"

            current_filenames = df_mapping["Filename"].tolist()
            current_mapping = df_mapping.set_index("Filename")["Template"].to_dict()
        else:
            if search_by == "Local":
                embeddings_path = "embeddings/noTemplate_topicLocal_noUsage.npy"
                current_filenames = df_filename_noTemplate_local["Filename"].tolist()
            else:
                embeddings_path = "embeddings/noTemplate_topicGlobal_noUsage.npy"
                current_filenames = df_filename_noTemplate_global["Filename"].tolist()
            current_mapping = None

    elif not topics and usages:
        if need_template:
            df_mapping = pd.read_csv("meme_to_template_map_updated.csv", names=["Filename", "Template"])
            embeddings_path = "embeddings/template_noTopic_usage.npy"

            current_filenames = df_mapping["Filename"].tolist()
            current_mapping = df_mapping.set_index("Filename")["Template"].to_dict()
        else:
            embeddings_path = "embeddings/noTemplate_noTopic_usage.npy"
            current_filenames = df_filename_noTemplate_global["Filename"].tolist()
            current_mapping = None
    else:
        return sentiment_based_recommendations(need_template, search_by,
                                                sentiment_preference,
                                                top_n, top_n_template)
    if topics:                                            
        topic_emb = text_model.encode(topics).reshape(1, -1)
        embeddings = np.load(embeddings_path).astype(np.float32)
        sims = cosine_similarity(topic_emb, embeddings).flatten()
        sims = normalize_scores(sims)
        idxs = np.argsort(sims)[::-1]
    else:
        if usages:
            usage_emb = text_model.encode(usages).reshape(1, -1)
            embeddings = np.load(embeddings_path).astype(np.float32)
            sims = cosine_similarity(usage_emb, embeddings).flatten()
            sims = normalize_scores(sims)
            idxs = np.argsort(sims)[::-1]
        else:
            return []

    recs = [current_filenames[i] for i in idxs]

    if need_template:
        recs = [current_mapping[f] for f in recs]

    return _filter_and_copy(recs, need_template=need_template, limit=top_n_template if need_template else top_n, results_folder="results")

def sentiment_based_recommendations(need_template, search_by, sentiment_preference, top_n, top_n_template):
    if not sentiment_preference:
        return []

    df_sentiments = pd.read_csv("sentiment_analysis_scores_300.csv")
    sentiment_lookup = df_sentiments.set_index("Filename")

    if need_template:
        df_mapping = pd.read_csv("meme_to_template_map_updated.csv", names=["Filename", "Template"])
        mapping = df_mapping.set_index("Filename")["Template"].to_dict()

        sentiment_scores = sentiment_lookup[sentiment_preference].dropna()
        sorted_filenames = sentiment_scores.sort_values(ascending=False).index.tolist()

        templs = [mapping.get(fn) for fn in sorted_filenames if fn in mapping]

        return _filter_and_copy(templs, need_template=True, limit=top_n_template, results_folder="results")

    else:
        files = (df_filename_noTemplate_local["Filename"].tolist())

        valid_scores = sentiment_lookup.loc[
            sentiment_lookup.index.intersection(files),
            sentiment_preference
            ].dropna().sort_values(ascending=False)

        sorted_files = valid_scores.index.tolist()

        return _filter_and_copy(sorted_files, need_template=False, limit=top_n, results_folder="results")


def _filter_and_copy(items, need_template, limit, results_folder):
    uniq, seen, saved = [], set(), []
    for key in items:
        idx = _get_index_for(key, need_template)
        img_path = get_image_path(idx, need_template)
        h = compute_image_hash(img_path)
        if h and h not in seen:
            seen.add(h)
            uniq.append(key)
            saved.append(img_path)
            if len(uniq) >= limit:
                break
    for p in saved:
        shutil.copy(p, results_folder)

    return [os.path.basename(p) for p in saved]

def _get_index_for(key, need_template):
    if need_template:
        df = df_filename_withTemplate
    else:
        df = df_filename_noTemplate_global

    matches = df.index[df["Filename"] == key].tolist()
    if matches:
        return matches[0]
    else:
        return None