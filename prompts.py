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

def build_prompt_details(user_input, need_template, has_topics, has_usages):
# Add a rule to ignore emotional or comparative mentions in topics
    common_instruction = ('''
Use the following rule when extracting "topics":
Include only specific people, characters, events, or themes the user explicitly wants memes about.
Do NOT include context-setting phrases (like 'bad day'), unrelated preferences (e.g., liking or disliking a subject), or secondary mentions not requested as the main focus.
''')

    if need_template and has_topics and has_usages:
        prompt = f"""
You are a meme-recommendation assistant.

{common_instruction}

Analyze the following user request and return a JSON object with exactly these fields:

1. "topics": A list of specific and relevant subjects, keywords, or themes the user explicitly wants the meme to be about (e.g., people, events, themes).
2. "usages": A list of the intended purposes or usages for the recommended memes (e.g., humor, comparison, political satire).

Return only the JSON object in the format below. Use your best judgment to identify the user's intent, but only extract what's directly relevant.

---

Example:

User request:
\"\"\"My day has been really bad, I want a meme that can cheer me up. I like Spongebob but not as much as minions, so give me minion meme templates.\"\"\"

Expected output:
{{
  "topics": ["Minions"],
  "usages": ["Humor", "cheering up"]
}}

---

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
  "topics": ["..."],
  "usages": ["..."]
}}
"""

    elif not need_template and has_topics and has_usages: 
        #example: "Give me a Spongebob meme about taxes and US-China relations for humor and political satire."
        prompt = f"""
You are a meme-recommendation assistant. 

{common_instruction}

Analyze the following user request and return a JSON object with exactly these three fields:

1. "topics": A list of specific and relevant subjects, keywords, or themes the user explicitly wants the meme to be about (e.g., people, events, themes).
2. "recommendation_focus": Specify the focus of the recommendation — either "local" or "global".
    - Use "local" if the user cares most about the *message, emotion, or text* in the meme.
      Example: "Give me a meme about college finals that cheers me up" → local
    - Use "global" if the user cares about a *specific meme format, structure, or visual style*.
      Example: "Give me a Minions meme" or "Give me a Spongebob template" → global
3. "usages": A list of the intended purposes or usages for the recommended memes (e.g., humor, comparison, political satire).

Return only the JSON object in the format below. Use your best judgment to identify the user's intent, but only extract what's directly relevant.

---

Example:

User request:
\"\"\"My day has been really bad, I want a meme that can cheer me up. I like Spongebob but not as much as minions, so give me minion memes.\"\"\"

Expected output:
{{
  "topics": ["Minions"],
  "recommendation_focus": "global",
  "usages": ["Humor", "cheering up"]
}}

---

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
  "topics": ["..."],
  "recommendation_focus": "local" or "global",
  "usages": ["..."]
}}
"""

    elif not has_topics and has_usages:
        #example: "Give me a meme for humor and political satire."
        #example: "Give me a meme template for humor." //need to improve
        prompt = f"""
You are a meme-recommendation assistant. The user did not mention any specific topics or themes in their request.

Your task is to identify the **intended purposes or usages** for the recommended memes and return a JSON object with exactly one field:

1. "usages": A list of the intended purposes or usages for the recommended memes (e.g., humor, comparison, political satire).

Ignore vague expressions or unrelated details, and focus only on what the user clearly wants the meme to achieve.

Return only the JSON object in the format below. Use your best judgment to identify the user's intent, but only extract what's directly relevant.

---

Example:

User request:
\"\"\"I'm in the mood for something funny or something that makes fun of politicians.\"\"\"

Expected output:
{{
  "usages": ["Humor", "Political Satire"]
}}

---

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
  "usages": ["..."]
}}
"""

    elif not need_template and has_topics and not has_usages:
        #example: "Give me a Spongebob meme about taxes and US-China relations."
        prompt = f"""
You are a meme-recommendation assistant.

{common_instruction}

Analyze the following user request and return a JSON object with exactly these two fields:

1. "topics": A list of specific and relevant subjects, keywords, or themes the user explicitly wants the meme to be about (e.g., people, events, themes).
2. "recommendation_focus": Specify the focus of the recommendation — either "local" or "global".
    - Use "local" if the user cares most about the *message, emotion, or text* in the meme.
      Example: "Give me a meme about college finals that cheers me up" → local
    - Use "global" if the user cares about a *specific meme format, structure, or visual style*.
      Example: "Give me a Minions meme" or "Give me a Spongebob template" → global

Return only the JSON object in the format below. Use your best judgment to identify the user's intent, but only extract what's directly relevant.

---

Example:

User request:
\"\"\"I saw a funny Spongebob meme once about economic stuff. Can I get one about inflation and how rent is insane now?\"\"\"

Expected output:
{{
  "topics": ["inflation", "rising rent"],
  "recommendation_focus": "local"
}}

---

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
  "topics": ["..."],
  "recommendation_focus": "local" or "global"
}}
"""

    elif need_template and has_topics and not has_usages:
        #example: "Give me a Spongebob meme template about taxes and US-China relations." #need to improve
        prompt = f"""
You are a meme-recommendation assistant. The user is requesting a meme *template* and has mentioned some specific topics, but did not provide any explicit usage or purpose.

{common_instruction}

Your task is to analyze the request and return a JSON object with exactly one field:

1. "topics": A list of specific and relevant subjects, keywords, or themes the user explicitly wants the meme to be about (e.g., people, events, themes).

---

Example:

User request:
\"\"\"I've had a long week and could really use a laugh. I love old Spongebob memes, but I want something fresh — maybe a meme *template* about inflation and how everything's so expensive now, like rent and groceries.\"\"\"

Expected output:
{{
  "topics": ["inflation", "rising rent", "grocery prices"]
}}

---

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
  "topics": ["..."]
}}
"""

    elif not has_topics and not has_usages:
        #example: "Give me a meme."
        #example: "Give me a meme template."
        prompt = f"""
You are a meme-recommendation assistant. The user did not mention any specific topics or intended usages in their request.

Your task is to infer the **emotional tone or sentiment** behind the user's request and return a JSON object with exactly one field:

1. "sentiment_preference": A single emotion or tone that best captures the user's likely mood or intent.

Select **only one** from the following controlled list:
["joy", "neutral", "anticipation", "disgust", "anger", "sadness",
 "fear", "negative", "positive", "love", "optimism", "surprised"]

Use your best judgment, but avoid making assumptions. If the request is too vague, default to **"neutral"**.

Return only the JSON object in the format below.

---

Example:

User request:
\"\"\"Give me a meme template.\"\"\"

Expected output:
{{
  "sentiment_preference": "neutral"
}}

---

User request:
\"\"\"{user_input}\"\"\"

Expected output format:
{{
  "sentiment_preference": "..."
}}
"""

    return prompt