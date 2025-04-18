import re
import google.generativeai as genai
import json
import GeminiAPI

genai.configure(api_key=GeminiAPI.key)  # 🔒 Replace with your actual API key

model = genai.GenerativeModel("gemini-2.0-flash")


def build_prompt_template_decision(user_input):
    prompt = f"""
You are a meme-recommendation assistant. 

Analyze the user's request and answer this question:

    "Is the user explicitly requesting meme templates or formats (e.g., specific meme images or recognizable formats)?"

Respond with **only one word**: `yes` or `no`. 

Use the following rules:
• Respond `yes` if the user clearly mentions a meme *template*, *format*, or *type of meme* (e.g., 'Distracted Boyfriend', 'Drakeposting', 'Two Buttons').
• Respond `no` if the user only refers to *text*, *message*, *topic* without mentioning templates or if the request is unclear, ambiguous.
• If uncertain, make your best guess based on the language of the request.

User request:
\"\"\"{user_input}\"\"\"
"""
    return prompt


def predict_template_decision(user_input):
    prompt = build_prompt_template_decision(user_input)
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip().lower()
        # print("\n[DEBUG] Raw Gemini response for template decision:\n", answer)
        if "yes" in answer:
            return True
        else:
            return False
    except Exception as e:
        return None

def build_prompt_topic_presence(user_input):
    prompt = f"""
You are a meme-recommendation assistant. 

Analyze the user's request and answer this question:

    "Does the request mention specific topics or keywords for the recommended memes (such as names, events, issues, or objects)?"

Respond with **only one word**: `yes` or `no`.

Use the following rules:
• Respond `yes` if the user clearly mentions specific topics or keywords (e.g., 'taxes', 'Elon Musk', 'climate change', 'graduation').
• Respond `no` if the request is vague, general, or does not mention any specific topics.
• If uncertain, make your best guess based on the language of the request.

User request:
\"\"\"{user_input}\"\"\"
"""
    return prompt

def predict_topic_presence(user_input):
    prompt = build_prompt_topic_presence(user_input)
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip().lower()
        # print("\n[DEBUG] Gemini response for topic presence:", answer)
        if "yes" in answer:
            return True
        else:
            return False
    except Exception as e:
        return None

def build_prompt_usage_decision(user_input):
    prompt = f"""
You are a meme-recommendation assistant.

Analyze the user's request and answer this question:

    "Does the user explicitly or implicitly mention the intended purposes or usages for the recommended memes (e.g., for humor, sarcasm, comparison, political commentary, venting)?"

Respond with **only one word**: `yes` or `no`.

Use the following rules:
• Respond `yes` if the user clearly mentions the intended purposes or usages for the recommended memes (e.g., 'to compare', 'for humor', 'for political commentary', 'to express frustration').
• Respond `no` if the request is vague, general, or does not mention any specific purpose or usage.
• If uncertain, use your best judgment based on the user's language and context.

User request:
\"\"\"{user_input}\"\"\"
"""
    return prompt

def predict_usage_decision(user_input):
    prompt = build_prompt_usage_decision(user_input)
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip().lower()
        # print("\n[DEBUG] Gemini response for usage decision:", answer)
        if "yes" in answer:
            return True
        else:
            return False
    except Exception as e:
        return None



def build_prompt_details(user_input, need_template, has_topics, has_usages):
    if need_template and has_topics and has_usages: 
        #example: "Give me a Spongebob meme template about taxes and US-China relations for humor and political satire."
        prompt = f"""
You are a meme-recommendation assistant. 

Analyze the following user request and return a JSON object with exactly these fields:

    1. "topics": A list of specific and relevant subjects, keywords, or themes the user is interested in (e.g., people, events, themes).
    2. "usages": A list of the intended purposes or usages for the recommended memes (e.g., humor, comparison, political satire).

Return only the JSON object in the format below. If a field is unclear or missing, use your best judgment based on the user's language and context.

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
  "topics": ["tax", "US-China relations"],
  "usages": ["Humor", "Political Satire"]
  }}
"""


    elif not need_template and has_topics and has_usages: 
        #example: "Give me a Spongebob meme about taxes and US-China relations for humor and political satire."
        prompt = f"""
You are a meme-recommendation assistant. 

Analyze the following user request and return a JSON object with exactly these three fields:

1. "topics": A list of specific and relevant subjects, keywords, or themes the user is interested in (e.g., people, themes, events).
2. "recommendation_focus": Specify the focus of the recommendation — either "local" (based on the text or message in the meme) or "global" (based on the overall meme format or template).
3. "usages": A list of the intended purposes or usages for the recommended memes (e.g., humor, comparison, political satire).

Return only the JSON object in the format below. If a field is unclear or missing, use your best judgment based on the user's language and context.

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
  "topics": ["tax", "US-China relations"],
  "recommendation_focus": "global",
  "usages": ["Humor", "Political Satire"]
}}
"""

    elif not has_topics and has_usages:
        #example: "Give me a meme for humor and political satire."
        prompt = f"""
You are a meme-recommendation assistant. The user did not mention any specific topics in their request.

Your task is to analyze the intended purposes or usages for the recommended memes and return a JSON object with exactly one field:

• "usages": A list of the intended purposes or usages for the recommended memes (e.g., humor, comparison, political satire).

Return only the JSON object in the format below. If the purpose is not explicitly stated, use your best judgment based on the user's language and context.

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
    "usages": ["Humor", "Political Satire"]
}}
"""

    elif not need_template and has_topics and not has_usages:
        #example: "Give me a Spongebob meme about taxes and US-China relations."
        prompt = f"""
You are a meme-recommendation assistant. The user did not mention any specific purposes or usages for the recommended memes in their request.

Analyze the following user request and return a JSON object with exactly these fields:

1. "topics": A list of specific and relevant subjects, keywords, or themes the user is interested in (e.g., people, themes, events).
2. "recommendation_focus": Specify the focus of the recommendation — either "local" (based on the text or message in the meme) or "global" (based on the overall meme format or template).

Return only the JSON object in the format below. If a field is unclear or missing, use your best judgment based on the user's language and context.

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
    "topics": ["tax", "US-China relations"],
    "recommendation_focus": "local"
}}
"""

    elif need_template and has_topics and not has_usages:
        #example: "Give me a Spongebob meme template about taxes and US-China relations."
        prompt = f"""
You are a meme-recommendation assistant. The user did not mention any specific purposes or usages for the recommended memes in their request.

Your task is to analyze the subjects, keywords, or themes of the recommended memes and return a JSON object with exactly one field:

• "topics": A list of specific and relevant subjects, keywords, or themes the user is interested in (e.g., people, themes, events).

Return only the JSON object in the format below. If a field is unclear or missing, use your best judgment based on the user's language and context.

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
    "topics": ["tax", "US-China relations"]
}}
"""

    elif not has_topics and not has_usages:
        #example: "Give me a meme."
        prompt = f"""
You are a meme-recommendation assistant. The user did not mention any specific topics or intended usages in their request.

Your task is to analyze the overall tone or emotional intent behind the user's request and return a JSON object with exactly one field:

• "sentiment_preference": a single emotion or tone that best matches the user's intent.

Choose **only one** from the following list:
["joy", "neutral", "anticipation", "disgust", "anger", "sadness",
    "fear", "negative", "positive", "love", "optimism", "surprised"]

Return only the JSON object in the format below. If the overall tone or emotional intent is unclear or missing, use your best judgment based on the user's language and context.

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
    "sentiment_preference": "joy"
}}
"""

    return prompt


def predict_input_details(user_input, need_template,has_topics,has_usages):
    prompt = build_prompt_details(user_input, need_template, has_topics, has_usages)
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()

        # print("\n[DEBUG] Raw Gemini response for input details:\n", answer)

        match = re.search(r"\{.*\}", answer, re.DOTALL)
        if match:
            json_str = match.group(0).strip()
            json_str = json_str.replace("'", '"')  # Ensure quotes are JSON-compatible

            # print("\n[DEBUG] Extracted JSON string:\n", json_str)

            processed_input = json.loads(json_str)
            # print("\n[DEBUG] Parsed JSON object:\n", processed_input)
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

    need_template = predict_template_decision(user_input)
    has_topics = predict_topic_presence(user_input)
    has_usages = predict_usage_decision(user_input)
    if need_template is None:
        print("[ERROR] Failed to determine if templates are wanted.")
        return None
    if has_topics is None:
        print("[ERROR] Failed to determine if topics are present.")
        return None
    if has_usages is None:
        print("[ERROR] Failed to determine if usages are present.")
        return None

    details = predict_input_details(user_input, need_template=need_template,has_topics=has_topics,has_usages=has_usages)
    result = {
        "need_template": need_template,
        "has_topics": has_topics,
        "has_usages": has_usages,
        "details": details
    }

    print("\n[FINAL RESULT] Processed Output:\n", result)
    return result


if __name__ == "__main__":
    user_input = "Give me a Leonardo DiCaprio meme about Russian-Ukraine war."
    process_user_input(user_input)