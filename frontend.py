import matplotlib.pyplot as plt
from PIL import Image
import local
import os
import shutil
import torch

results_folder = "results"
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

def main():
    torch.cuda.empty_cache()
    init_new_folders()
    user_query = input("Enter a meme description or topic: ")
    mode = input("Do you want to search using meme templates? (y/n): ").strip().lower()
    need_template = mode == 'y'

    sentiment_options = [
        "joy", "neutral", "anticipation", "disgust", "anger", "sadness", 
        "fear", "negative", "positive", "love", "optimism", "surprised", "None"
    ]

    print("Select a sentiment preference:")
    for i, sentiment in enumerate(sentiment_options, 1):
        print(f"{i}. {sentiment}")
    try:
        sentiment_choice = int(input("Enter the number corresponding to your choice (1-13): "))
        sentiment_preference = sentiment_options[sentiment_choice - 1]
    except:
        print("Invalid choice. Proceeding without sentiment preference.")
        sentiment_preference = None

    if need_template == False:
        search_mode = input("Search by Local or Global? (L/G): ").strip().upper()
        search_by = "Local" if search_mode == 'L' else "Global"
        recommended_memes = local.get_similar_memes(
            user_query, need_template=need_template, search_by=search_by, sentiment_preference=sentiment_preference
        )   
    else:
        recommended_memes = local.get_similar_memes(
            user_query, need_template=need_template, sentiment_preference=sentiment_preference
        )

    for i, result in enumerate(recommended_memes):
        if need_template:
            print(f"Recommended Template {i+1}: {result}")
            meme_entry = local.df_filename_withTemplate[local.df_filename_withTemplate["Filename"] == result]
            index = meme_entry.index[0]
        else:
            print(f"Recommended Meme {i+1}: {result}")
            meme_entry = local.df_filename_noTemplate_local[local.df_filename_noTemplate_local["Filename"] == result] if search_by == "Local" else local.df_filename_noTemplate_global[local.df_filename_noTemplate_global["Filename"] == result]
            index = meme_entry.index[0]

        image_path = local.get_image_path(index, need_template, search_by if not need_template else None)
        save_meme(image_path, result)

if __name__ == "__main__":
    main()