import matplotlib.pyplot as plt
from PIL import Image
import local
import os
import shutil
import torch
import Gemini_agents as agents

results_folder = "test_images"
error_log_path = "error_log.txt"
os.makedirs(results_folder, exist_ok=True)  

def init_new_folders():
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)  
    os.makedirs(results_folder, exist_ok=True) 

    if os.path.exists(error_log_path):
        os.remove(error_log_path)

def save_meme(image_path, meme_fileName):
    image = Image.open(image_path)
    meme_fileName = meme_fileName.replace(".jpg", "")
    save_path = os.path.join(results_folder, f"{meme_fileName}.png")
    image.save(save_path)

def main(user_query):
    torch.cuda.empty_cache()
    # init_new_folders()

    processed = agents.process_user_input(user_query)
    need_template      = processed["need_template"]
    has_topic          = processed["has_topics"]
    has_usage          = processed["has_usages"]
    details            = processed.get("details") or {}

    topics_str         = " ".join(details.get("topics", [])) if has_topic else None
    usages_str         = "|".join(details.get("usages", []))   if has_usage else None
    search_by          = details.get("recommendation_focus", "Global").capitalize() if has_topic else None
    sentiment_preference = details.get("sentiment_preference", "neutral") if not has_topic and not has_usage else None

    recommended_memes = local.get_similar_memes(
        topics=topics_str,
        need_template=need_template,
        usages=usages_str,
        search_by=search_by,
        sentiment_preference=sentiment_preference,
        top_n=20,
        top_n_template=10
    )

    image_paths = []
    if isinstance(recommended_memes, list) and len(recommended_memes) > 0:
        if need_template:
            df = local.df_filename_withTemplate 
        else:
            df = (local.df_filename_noTemplate_local 
                    if search_by == "Local" 
                    else local.df_filename_noTemplate_global)
        for recommended_meme in recommended_memes:
            idx = df[df["Filename"] == recommended_meme].index[0]
            image_path = local.get_image_path(idx, need_template, search_by if not need_template else None)
            save_meme(image_path, recommended_meme)
            image_paths.append(os.path.join(results_folder, recommended_meme.replace(".jpg", "") + ".png"))
    else:
        return []

    return image_paths

# if __name__ == "__main__":
#     main("funny reaction memes")