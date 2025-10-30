# MEMEMATCH: AI-Powered Meme Recommendation System - Tri An Le

![MEMEMATCH Logo](UI_images/memematch_96.png)

**MEMEMATCH** is an AI-powered tool that recommends the perfect meme or meme template for any situation. Whether you're feeling sarcastic, sad, or celebratory — just type a request or upload an image, and MEMEMATCH will do the rest. It leverages cutting-edge natural language processing, computer vision, and large language models to deliver contextually relevant and emotionally resonant meme suggestions.

---

## 🚀 Features

- 🔍 **Natural Language Meme Search**  
  Describe what you want (e.g., “Give me a sarcastic Zoom fatigue meme”), and get spot-on meme or template suggestions.

- 🖼️ **Image-Based Search**  
  Upload a meme to find visually or contextually similar ones using OCR and image captioning.

- 🧠 **LLM-Powered Query Understanding**  
  Google Gemini models extract your intent, usage context, and topics to improve recommendations.

- 🌐 **Multimodal AI Pipelines**  
  Combines NLP, computer vision, and zero-shot classification using:  
  - SentenceTransformers & Gemini LLMs (embeddings generation, semantic search, query understanding, and topic modeling)  
  - BART & RoBERTa (usage classification & emotion detection)
  - EasyOCR & PaddleOCR (text from memes)  
  - BLIP (image captioning)

- 💬 **Semantic & Sentiment Search**  
  Matches memes based on topics, use cases (e.g., roast, celebration), and emotional tone.

- 💻 **Modern Web Interface and Android Application**

---

## 🧰 Tech Stack

- **Android:** Kotlin, Jetpack Compose, Firebase (Auth, Firestore), Retrofit, Kotlin Coroutines
- **Backend:** FastAPI, Google Gemini API, Hugging Face Transformers, SentenceTransformers, BERTopic, EasyOCR, PaddleOCR, BLIP, scikit-learn, pandas, NumPy  
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Uvicorn, pyngrok (for local tunneling)  
- **Data:** Meme/meme template datasets, precomputed embeddings, usage/emotion labels  

---

## 📦 Project Structure

```
Meme_Recommendation_Final/
│
├── server.py                # FastAPI backend server
├── Gemini_agents.py         # Gemini LLM prompt engineering, topic modeling, and query parsing
├── prompts.py               # Prompt templates for Gemini
├── local.py                 # Meme retrieval and similarity logic
├── usage.py                 # Zero-shot meme usage classification
├── sentiment_analysis.py    # Meme sentiment/emotion scoring
├── ImageCaptioning.py       # BLIP-based image captioning
├── embedding_generator.py   # Embedding generation for memes/templates
├── PaddleOCR_global.py      # PaddleOCR for masking text regions
├── EasyOCR_local.py         # EasyOCR for textual content extraction
├── index.html               # Main frontend UI
├── UI_images/               # App icons, profile images
├── results/                 # Folder for serving recommended memes
├── zipped_CSV_files         # Reddit memes' local context, meme-template mappings, sentiment scores, usages
├── recommendation_filepaths # Returns filepaths for recommended memes
├── test_labels_generator.py # Semi-automated relevance labels (2,1,0) generator for queries and recommended memes
├── test_meme.csv            # Test results of meme queries
├── test_template.csv        # Test results of meme template queries
├── evaluation.ipynb         # Evaluation of text-to-image model performance
├── EDA_Reddit_local.ipynb   # Exploratory Data Analysis of Local Context of Reddit memes
├── requirements.txt         # Dependencies version
```

---

## ⚡ Quick Start

- Download the **Android application** at https://drive.google.com/file/d/1wZ1ORq2wbjOP-TfdhMHnRdoSeN1UgZvR/view or access the **website** at [https://hugely-climbing-moray.ngrok-free.app/](https://hugely-climbing-moray.ngrok-free.app/).

---

## 🧠 How It Works

1. **User Query:**  
   User enters a text prompt or uploads an image.

2. **Intent Parsing:**  
   Gemini LLMs analyze the query to extract topics, usage, and intent.

3. **Meme Retrieval:**  
   - **Text Queries:** Performs semantic search using topic and usage embeddings derived from the user's prompt.  
   - **Image Queries:** Applies OCR or BLIP-based captioning to extract textual context, which is then used for semantic retrieval.  
   - **Vague or Emotional Prompts:** Recommend memes that align with the desired emotional tone utilizing sentiment and emotion scores.

4. **Result Delivery:**  
   Top memes/templates are copied to the results folder and served to the frontend.

---

## 📝 Example Queries

- "Give me a Spongebob meme about taxes and US-China relations."
- "I want a meme template for political satire."
- "Upload a meme and find similar ones."
- "Give me a meme for cheering up."

---

## Setup and Run Instructions

- Create a Python environment with the dependencies listed in requirements.txt.
- For Jupyter Notebooks, unzip the files in "zipped_CSV_files" folder and extract to the current working directory (/Meme_Recommendation_Final).
- For Python files, except for "server.py", they are implementations of the methodologies that I used in this project, please feel free to adjust it for your own purposes. You will need to create a "API_keys.py" and input your Gemini API key "key_gemini = # YOUR_API_KEY" there.
- For index.html, feel free to use it as reference for your web interface.
- **Please notice**: Without the meme dataset, the framework cannot operate. Please contact the author for the access to the dataset.

## 👤 About the Author

**Tri An Le**  
Natural Language Processing Enthusiast
- Email: [triandole@gmail.com](mailto:triandole@gmail.com)  
- LinkedIn: [https://www.linkedin.com/in/trianle/](https://www.linkedin.com/in/trianle/)

---

## 📄 Research Paper

For an in-depth explanation of the methods and results behind MEMEMATCH, see the research paper:

- [MemeMatch: Context-Aware Multimodal Meme Retrieval and Recommendation via Structured Semantic Understanding](https://github.com/TriAnLe171/Meme_Recommendation_Project/blob/main/paper.pdf)

---

## 📄 License

© 2025 MEMEMATCH. All rights reserved.

---

## ⭐️ Contributing

Pull requests and suggestions are welcome! Please open an issue or contact the author.
