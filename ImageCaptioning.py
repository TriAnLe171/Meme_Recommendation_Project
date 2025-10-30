from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import csv

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to('cuda')
input_dir = "###"

image_files = [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png"))]

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to('cuda')
    
    with torch.no_grad():
        output = model.generate(**inputs)
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

i=0
output_file = "caption_IMGFlip_for_recommendation.csv"
with open(output_file ,"a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    for image_file in image_files[i:]:
        image_path = os.path.join(input_dir, image_file)
        caption = generate_caption(image_path)
        writer.writerow([image_file, caption])
        print(f"Processed {image_file}: {caption}")
print(f"All captions are saved in '{output_file}'.")
