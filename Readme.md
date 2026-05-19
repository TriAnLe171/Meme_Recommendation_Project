# MemeMatch: A Large-Scale Dual-Context Multimodal Dataset and Retrieval System for Internet Memes

![MemeMatch Logo](UI_images/memematch_96.png)

**MemeMatch** is a large-scale multimodal dataset and retrieval framework for studying internet memes. The project introduces a dual-context representation that separates the **local context** of a meme, including user-added overlay text and post titles, from the **global context** of the underlying visual template or base image. This structure enables more precise analysis of how meme meaning emerges from the interaction between text, image, emotion, topic, and communicative intent.

The MemeMatch corpus was built from nearly one million image-with-text memes collected from Reddit’s r/Memes and ImgFlip. After cleaning, deduplication, and preprocessing, the released dataset contains approximately **301K memes** spanning **2,083 meme templates**. Each meme is enriched with transformer-based annotations, including sentiment and emotion vectors, BERTopic-derived topics, and zero-shot usage-intent labels.

This repository contains the code, data-processing pipeline, annotation framework, exploratory analysis notebooks, and retrieval system associated with the paper:

> **MemeMatch: A Large-Scale Dual-Context Multimodal Dataset and Retrieval System for Internet Memes**  
> Do Tri An Le, Donát Ákos Köller, Qixin Deng, Roland Molontay  
> Accepted to ICWSM 2026

---

## Overview

Internet memes are compact multimodal artifacts that combine image, text, humor, emotion, and cultural context. However, computational meme analysis is difficult because the same visual template can express very different meanings depending on the overlaid text, while the same textual idea can appear across different visual forms.

MemeMatch addresses this challenge through a **dual-context framework**:

- **Local context:** the user-added textual layer of the meme, extracted from OCR and combined with post titles.
- **Global context:** the underlying visual or template-level meaning, obtained by masking overlaid text and captioning the remaining image.

By separating these two layers, MemeMatch supports research on how memes encode affect, topics, usage intent, cultural references, and community-specific meanings.

---

## Main Contributions

This repository supports the following contributions from the paper:

1. **Large-scale meme dataset**  
   A curated corpus of approximately 301K memes from Reddit’s r/Memes and ImgFlip, covering 2,083 common meme templates.

2. **Dual-context meme representation**  
   A structured representation that separates user-specific local text from template-level global visual semantics.

3. **Rich semantic annotations**  
   Each meme is annotated with:
   - 14-dimensional sentiment and emotion vectors
   - BERTopic-derived local and global topics
   - Zero-shot usage-intent labels
   - Auxiliary features such as local text length

4. **Context-aware retrieval system**  
   A retrieval framework that supports natural-language and image-based meme search using precomputed case-based embeddings and an LLM-based query parser.

5. **Exploratory analysis of meme culture**  
   Analysis of temporal upload patterns, affective differences between local and global contexts, topic distributions, and common meme usage categories.

---

## Dataset Summary

The MemeMatch dataset combines two complementary sources:

| Source | Description |
|---|---|
| Reddit r/Memes | Organic memes shared in an online community, collected from 2018 to 2023 with metadata such as title, timestamp, and score |
| ImgFlip | Template-based memes with explicit template labels |

After preprocessing, the dataset contains:

| Component | Count |
|---|---:|
| Reddit memes | 146,991 |
| ImgFlip memes | 153,792 |
| Unique templates | 2,083 |
| Total curated memes | ~301K |

---

## Dual-Context Framework

MemeMatch processes each meme through two synchronized branches.

### Local Context

The local context captures the user-added meaning of a meme.

It is constructed from:

- OCR-extracted overlay text using EasyOCR
- Reddit post titles when available
- Template-string filtering to remove repeated artifacts, watermarks, and non-informative tokens

This context is useful for studying meme messages, jokes, emotions, and situational meanings.

### Global Context

The global context captures the reusable visual substrate of a meme.

It is constructed by:

- Detecting text regions with PaddleOCR
- Masking user-added text
- Captioning the remaining image using BLIP

This context is useful for studying visual templates, recurring image motifs, and template-level semantics.

---

## Annotation Pipeline

MemeMatch enriches both local and global contexts with multiple semantic signals.

### Sentiment and Emotion

Each meme receives a 14-dimensional affect vector for both local and global contexts:

- 11 emotion categories
- 3 sentiment polarity scores: positive, neutral, negative

The models are based on RoBERTa models fine-tuned on Twitter data.

### Topic Modeling

MemeMatch uses BERTopic to extract themes separately from local and global contexts.

- Local context topics capture written meme content, cultural references, and time-sensitive themes.
- Global context topics capture visual setups, recurring templates, objects, characters, and scenes.

### Usage-Intent Labels

MemeMatch applies zero-shot classification with BART-MNLI to infer communicative usage categories, such as:

- Sarcasm or Irony
- Parody or Spoof
- Reaction or Reply Meme
- Confusion or Disbelief
- Wordplay or Pun
- Emotional Frustration
- Self-Deprecation
- Media or Brand Critique

---

## Data Schema

Each meme is represented using core metadata and derived annotations.

### Core Metadata

| Field | Description |
|---|---|
| `filename` | Unique image filename |
| `created_utc` | Reddit upload timestamp, when available |
| `score` | Reddit upvote score at crawl time |

### Derived Annotations

| Field | Description |
|---|---|
| `local_context` | Cleaned OCR text plus title |
| `global_context` | BLIP caption of masked meme image |
| `text_length` | Character count of local context |
| `sentiment_local[14]` | Local-context emotion and sentiment vector |
| `sentiment_global[14]` | Global-context emotion and sentiment vector |
| `topic_local` | BERTopic label for local context |
| `topic_global` | BERTopic label for global context |
| `topic_score_local` | Local topic confidence |
| `topic_score_global` | Global topic confidence |
| `usage_labels` | Zero-shot usage-intent labels |

---

## Retrieval System

### Natural-Language Retrieval

A user query is parsed into structured retrieval attributes:

- Search scope: meme or template
- Topic: subject, entity, or concept
- Usage intent: humor, complaint, reaction, motivation, critique, etc.

The system then routes the query to the appropriate precomputed embedding collection and retrieves the most relevant memes using cosine similarity.

### Image-Based Retrieval

For uploaded meme images, MemeMatch supports two retrieval modes:

1. **Global image-context retrieval**  
   Masks text, captions the visual template, and retrieves visually or template-similar memes.

2. **Local text-context retrieval**  
   Extracts overlay text using OCR and retrieves memes with similar textual meaning.

---

## Repository Structure

```text
Meme_Recommendation_Final/
│
├── server.py                # FastAPI backend server
├── Gemini_agents.py         # LLM-based query parsing and topic/intent extraction
├── prompts.py               # Prompt templates
├── local.py                 # Meme retrieval and similarity logic
├── usage.py                 # Zero-shot usage classification
├── sentiment_analysis.py    # Sentiment and emotion scoring
├── ImageCaptioning.py       # BLIP-based image captioning
├── embedding_generator.py   # Embedding generation for memes and templates
├── PaddleOCR_global.py      # Text-region detection and masking for global context
├── EasyOCR_local.py         # OCR extraction for local context
├── index.html               # Web interface
├── UI_images/               # App icons and interface images
├── results/                 # Folder for serving retrieved memes
├── zipped_CSV_files/        # CSV files required for retrieval system
├── recommendation_filepaths # File paths for retrieved memes
├── test_labels_generator.py # Semi-automated relevance label generation
├── test_meme.csv            # Meme retrieval test results
├── test_template.csv        # Template retrieval test results
├── evaluation.ipynb         # Retrieval evaluation notebook
├── EDA_local.ipynb          # Exploratory analysis notebook for local context
├── EDA_global.ipynb         # Exploratory analysis notebook for global context
├── requirements.txt         # Python dependencies
```

---

## Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

For Jupyter notebooks, unzip the files in the `zipped_CSV_files/` folder and extract them into the main project directory:

```text
/Meme_Recommendation_Final
```

For scripts that use Gemini models, create an `API_keys.py` file:

```python
key_gemini = "YOUR_GEMINI_API_KEY"
```

---

## Retrieval Demo

- MemeMatch Retrieval System (Web): [https://hugely-climbing-moray.ngrok-free.app](https://hugely-climbing-moray.ngrok-free.app)
- MemeMatch Android App (Download): [https://drive.google.com/drive/u/2/home](https://drive.google.com/file/d/1wZ1ORq2wbjOP-TfdhMHnRdoSeN1UgZvR/view?usp=sharing)
- Demo Video: [Youtube video here](https://youtu.be/j9r0e2kqzwI)

## Dataset Access

See: https://doi.org/10.34740/kaggle/dsv/14510064

---

## Applications

MemeMatch is designed for research in:

- Computational social science
- Social media analysis
- Multimodal machine learning
- Digital culture and meme studies
- Computational humor
- Affective computing
- Media literacy
- Meme retrieval
- Content moderation research

---

## Limitations

MemeMatch has several important limitations:

- The current dataset focuses primarily on English-language memes from Reddit’s r/Memes and ImgFlip.
- Automated annotations are probabilistic and should be treated as semantic signals rather than ground-truth labels.
- OCR and captioning may fail on stylized fonts, dense overlays, low-quality images, or culturally specific references.
- Reddit data and pretrained models may encode social, linguistic, and platform-specific biases.
- The dataset covers 2018 to 2023 and does not include newer meme trends unless updated.

---

## Citation

**Coming soon**

---

## Paper

For the full methodology, dataset description, exploratory analysis, retrieval design, and limitations, see:

[MemeMatch: A Large-Scale Dual-Context Multimodal Dataset and Retrieval System for Internet Memes](https://github.com/TriAnLe171/MemeMatch-v1.0/blob/main/paper.pdf)

---

## Author

**Do Tri An Le**  
Department of Mathematics and Computer Science, Wabash College  
Incoming Ph.D. Student in Computer Science, University of Houston  

- Email: [triandole@gmail.com](mailto:triandole@gmail.com)
- Homepage: [https://trianle171.github.io/](https://trianle171.github.io/)
- LinkedIn: [https://www.linkedin.com/in/trianle/](https://www.linkedin.com/in/trianle/)

---

## License

© 2026 MemeMatch. All rights reserved.

---

## Contributing

Pull requests, issues, and suggestions are welcome. Please open an issue or contact the author for questions about the dataset, code, or research use.
