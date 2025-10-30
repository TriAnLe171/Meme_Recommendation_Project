import easyocr
import os

reader = easyocr.Reader(['en'], gpu=True)

def extract_text_easyocr(image_path):
    """Extracts text from an image using EasyOCR."""
    results = reader.readtext(image_path)
    return " ".join([result[1] for result in results])  

cwd = os.getcwd()
image_dir = os.path.join(cwd, "all_images_IMGFlip_templates")
output_file = "template_extracted_text.csv"
filename_file = "delete.csv"

image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]

with open(output_file, "a", encoding="utf-8") as text_file, open(filename_file, "a", encoding="utf-8") as name_file:
    for i, file in enumerate(image_files):
        image_path = os.path.join(image_dir, file)
        print(f"Processing {i+1}/{len(image_files)}: {image_path}")

        try:
            extracted_text = extract_text_easyocr(image_path)
            name_file.write(f"{file}\n")  
            text_file.write(f"{file},{extracted_text}\n")  
        except Exception as e:
            print(f"Error processing {file}: {e}")