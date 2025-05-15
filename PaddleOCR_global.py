import cv2
import os
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="en")  

input_dir = "# REPLACE WITH INPUT DIRECTORY"  
output_dir = "# REPLACE WITH OUTPUT DIRECTORY"  

os.makedirs(output_dir, exist_ok=True)
image_files = [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png"))]
exceptions = {}
i = 0

for file_name in image_files[i:]:
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)

    print(f"Processing {i+1}/{len(image_files)}: {file_name}")
    try:
        image = cv2.imread(input_path)

        results = ocr.ocr(input_path, cls=True)
        
        for line in results:
            if line:
                for word_info in line:
                    text = word_info[1][0] 
                    # confidence = word_info[1][1]  # OCR confidence score
                    bbox = word_info[0]  

                    if text not in exceptions:  
                        x1, y1 = map(int, bbox[0])
                        x2, y2 = map(int, bbox[2])

                        if x2 > x1 and y2 > y1:
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

                            # Blur the detected text region
                            # roi = image[y1:y2, x1:x2]
                            # blurred_roi = cv2.GaussianBlur(roi, (55, 55), 0)
                            # image[y1:y2, x1:x2] = blurred_roi

        cv2.imwrite(output_path, image)

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        
    i+=1
        

print("Processing complete! All blurred images are saved in 'GlobalContext'.")