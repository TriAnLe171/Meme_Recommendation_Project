import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import imagehash
from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity

df_filename_noTemplate_local = pd.read_csv("filenames_noTemplate_local.csv", names=["Filename"])
df_filename_noTemplate_global = pd.read_csv("filenames_noTemplate_global.csv", names=["Filename"])
df_filename_withTemplate = pd.read_csv("filenames_withTemplate.csv", names=["Filename"])

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
            #log error to a file
            with open("error_log.txt", "a") as log_file:
                log_file.write(f"Error processing {image_path}: {e}\n")
                return None


def get_similar_memes(user_input, need_template=None,search_by="Global", sentiment_preference=None, top_n=10,top_n_template=5):
    """Recommend top-N memes based on text input similarity."""
    user_embedding = text_model.encode(user_input).reshape(1, -1)

    # Use appropriate embeddings and dataset
    if need_template:
        df_mapping = pd.read_csv("meme_to_template_map_updated.csv", names=["Filename", "Template"])
        embeddings_template = np.load("embeddings/templates.npy").astype(np.float32)

        cosine_sim = cosine_similarity(user_embedding, embeddings_template)
        cosine_sim = normalize_scores(cosine_sim)

        current_filenames = df_mapping["Filename"].tolist()
        current_mapping = df_mapping.set_index("Filename")["Template"].to_dict()
    else:
        # Load local or global embeddings based on user choice
        if search_by == "Local":
            embeddings_local = np.load("embeddings/local298.npy").astype(np.float32)
            cosine_sim_local = cosine_similarity(user_embedding, embeddings_local)
            cosine_sim_local = normalize_scores(cosine_sim_local)
            cosine_sim = cosine_sim_local
            current_filenames = df_filename_noTemplate_local["Filename"].tolist()
        else:
            embeddings_global = np.load("embeddings/global300.npy").astype(np.float32)
            cosine_sim_global = cosine_similarity(user_embedding, embeddings_global)
            cosine_sim_global = normalize_scores(cosine_sim_global)
            cosine_sim = cosine_sim_global
            current_filenames = df_filename_noTemplate_global["Filename"].tolist()

        current_mapping = None  

    similar_indices = np.argsort(cosine_sim)[::-1]

    # sentiment preference
    if sentiment_preference:
        df_sentiments = pd.read_csv("sentiment_analysis_scores_300.csv") 
        sentiment_lookup = df_sentiments.set_index("Filename")
        filtered_indices = []
        for idx in similar_indices:
            filename = df_filename_withTemplate["Filename"][idx] if need_template else current_filenames[idx]
            if filename in sentiment_lookup.index:
                sentiment_score = sentiment_lookup.loc[filename].get(sentiment_preference, 0)
                if sentiment_score > 0.7:
                    filtered_indices.append(idx)
    
        similar_indices = filtered_indices
    
    recommended_memes = [current_filenames[idx] for idx in similar_indices]

    # If template mode, map filenames to templates
    if need_template:
        recommended_templates = [current_mapping.get(meme) for meme in recommended_memes]

        unique_templates = []
        unique_template_harshes = set()
        for template in recommended_templates:
            image_path = get_image_path(current_filenames.index(template), need_template)
            image_hash = compute_image_hash(image_path)
            if image_hash not in unique_template_harshes:
                unique_template_harshes.add(image_hash)
                unique_templates.append(template)
            if len(unique_templates) >= top_n_template:
                break

        return unique_templates
    else:
        unique_meme_hashes = set()
        unique_meme_recommendations = []
        for meme in recommended_memes:
            image_path = get_image_path(current_filenames.index(meme), need_template, search_by)
            image_hash = compute_image_hash(image_path)
            if image_hash not in unique_meme_hashes:
                unique_meme_hashes.add(image_hash)
                unique_meme_recommendations.append(meme)
            if len(unique_meme_recommendations) >= top_n:
                break
        
        return unique_meme_recommendations