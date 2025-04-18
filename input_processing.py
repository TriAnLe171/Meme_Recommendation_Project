import re
import google.generativeai as genai
import json
import GeminiAPI

# Initialize Gemini with your API key
genai.configure(api_key=GeminiAPI.API)  # 🔒 Replace with your actual API key

# Create a Gemini model
model = genai.GenerativeModel("gemini-2.0-flash-lite")


def build_prompt_template_decision(user_input):
    prompt = f"""
You are a meme-recommendation assistant. 

Analyze the user's request and answer this question:

    "Is the user explicitly requesting meme templates or formats (e.g., specific meme images or formats)?"

Respond with **only one word**: `yes` or `no`. 

Use the following rules:
• Respond `yes` if the user clearly mentions a meme *template*, *format*, or *type of meme* (e.g., 'Distracted Boyfriend', 'Drakeposting').
• Respond `no` if the user only refers to *text*, *message*, *topic* without mentioning templates or if the request is unclear, ambiguous.
• If uncertain, make your best guess based on the language of the request.

User request:
\"\"\"{user_input}\"\"\"
"""
    print("\n[DEBUG] Prompt for template decision:\n", prompt)
    return prompt


def predict_template_decision(user_input):
    prompt = build_prompt_template_decision(user_input)
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip().lower()
        print("\n[DEBUG] Raw Gemini response for template decision:\n", answer)
        if "yes" in answer:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error during Gemini prediction: {e}")
        return None

def build_prompt_topic_presence(user_input):
    prompt = f"""
You are a meme-recommendation assistant. 

Analyze the user's request and answer this question:

    "Does the request mention specific topics or keywords (such as names, events, issues, or objects)?"

Respond with **only one word**: `yes` or `no`.

Use the following rules:
• Respond `yes` if the user clearly mentions specific topics or keywords (e.g., 'taxes', 'Elon Musk', 'climate change', 'graduation').
• Respond `no` if the request is vague, general, or does not mention any specific topics.
• If uncertain, make your best guess based on the language of the request.

User request:
\"\"\"{user_input}\"\"\"
"""
    print("\n[DEBUG] Prompt for topic presence decision:\n", prompt)
    return prompt

def predict_topic_presence(user_input):
    prompt = build_prompt_topic_presence(user_input)
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip().lower()
        print("\n[DEBUG] Gemini response for topic presence:", answer)
        if "yes" in answer:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error during topic presence prediction: {e}")
        return None



def build_prompt_details(user_input, wants_template, has_topics):
    if wants_template and has_topics:
        prompt = f"""
You are a meme-recommendation assistant. Analyze the following user request and return a JSON object with exactly these fields:

    1. "topics": A list of specific and relevant subjects, keywords, or themes the user is interested in (e.g., people, events, themes).
    2. "usages": A list of the intended purposes or usages of the memes (e.g., humor, comparison, political satire).

Return only the JSON object in the format below. If a field is unclear or missing, make a best guess based on the request.

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
  "topics": ["tax", "US-China relations"],
  "usages": ["Humor", "Political Satire"]
  }}
"""


    elif not wants_template and has_topics:
        prompt = f"""
You are a meme-recommendation assistant. Analyze the following user request and return a JSON object with exactly these three fields:

1. "topics": A list of specific and relevant subjects, keywords, or themes the user is interested in (e.g., people, themes, events).
2. "recommendation_focus": Specify the focus of the recommendation — either "local" (based on the text or message in the meme) or "global" (based on the overall meme format or template).
3. "usages": A list of the intended purposes or usages for the memes (e.g., humor, comparison, political satire).

Return only the JSON object in the format below. If a field is unclear or missing, make a best guess based on the request.

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
  "topics": ["tax", "US-China relations"],
  "recommendation_focus": "global",
  "usages": ["Humor", "Political Satire"]
}}
"""


    elif not has_topics:
        prompt = f"""
You are a meme-recommendation assistant. The user did not mention any specific topics in their request.

Your task is to analyze the overall tone or emotional intent behind the user's request and return a JSON object with exactly one field:

• "sentiment_preference": a list containing a single emotion or tone that best matches the user's intent.

Choose **only one** from the following list:
["joy", "neutral", "anticipation", "disgust", "anger", "sadness", 
 "fear", "negative", "positive", "love", "optimism", "surprised"]

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
  "sentiment_preference": ["joy"]
}}
"""
    print("\n[DEBUG] Prompt for extracting details:\n", prompt)
    return prompt


def predict_input_details(user_input, wants_template,has_topics):
    prompt = build_prompt_details(user_input, wants_template, has_topics)
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()

        print("\n[DEBUG] Raw Gemini response for input details:\n", answer)

        match = re.search(r"\{.*\}", answer, re.DOTALL)
        if match:
            json_str = match.group(0).strip()
            json_str = json_str.replace("'", '"')  # Ensure quotes are JSON-compatible

            print("\n[DEBUG] Extracted JSON string:\n", json_str)

            processed_input = json.loads(json_str)
            print("\n[DEBUG] Parsed JSON object:\n", processed_input)
            return processed_input
        else:
            print("[ERROR] No valid dictionary found in the response.")
            return None
    except Exception as e:
        print(f"[ERROR] Error during Gemini prediction (input details): {e}")
        return None



def process_user_input(user_input):
    print("\n========== Processing New Input ==========\n")
    print("[INFO] User input:\n", user_input)

    want_template = predict_template_decision(user_input)
    has_topics = predict_topic_presence(user_input)
    if want_template is None:
        print("[ERROR] Failed to determine if templates are wanted.")
        return None
    if has_topics is None:
        print("[ERROR] Failed to determine if topics are present.")
        return None

    details = predict_input_details(user_input, wants_template=want_template,has_topics=has_topics)
    result = {
        "wants_template": want_template,
        "has_topics": has_topics,
        "details": details
    }

    print("\n[FINAL RESULT] Processed Output:\n", result)
    return result


if __name__ == "__main__":
    user_input = "I wants sad memes."
    process_user_input(user_input)