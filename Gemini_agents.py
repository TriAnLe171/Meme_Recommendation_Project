import re
import google.generativeai as genai
import json
import GeminiAPI
from prompts import (
    build_prompt_template_decision,
    build_prompt_topic_presence,
    build_prompt_usage_decision,
    build_prompt_details,
)

genai.configure(api_key=GeminiAPI.key)  # 🔒 Replace with your actual API key

model = genai.GenerativeModel("gemini-2.0-flash") #gemini-2.5-pro-exp-03-25

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


def predict_input_details(user_input, need_template,has_topics,has_usages):
    prompt = build_prompt_details(user_input, need_template, has_topics, has_usages)
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        # print("\n[DEBUG] Raw Gemini response for input details:\n", answer)

        # Prefer content inside code block if present
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", answer, re.DOTALL)
        if not match:
            match = re.search(r"\{.*\}", answer, re.DOTALL)
        if not match:
            print("[ERROR] No valid dictionary found in the response.")
            return None

        json_str = match.group(1).strip() if match.lastindex == 1 else match.group(0).strip()
        json_str = json_str.replace("'", '"')
        # print("\n[DEBUG] Extracted JSON string:\n", json_str)

        return json.loads(json_str)

    except Exception as e:
        print(f"[ERROR] Failed to parse Gemini response: {e}")
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


# if __name__ == "__main__":
#     user_input = "Give me a meme to cheer me up after a long day of hard work."
#     process_user_input(user_input)