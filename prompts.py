def build_prompt_template_decision(user_input):
    prompt = f"""
You are a meme-recommendation assistant. 

Analyze the user's request and answer this question:

    "Is the user explicitly requesting meme templates or formats (e.g., specific meme images or recognizable formats)?"

Respond with **only one word**: `yes` or `no`. 

Use the following rules:
• Respond `yes` if the user clearly uses the word *template*, *format*, or phrases like “type of meme” or “structure” that indicate they want a blank meme template or image format.
• Do NOT respond `yes` just because the user names a meme (e.g., 'Drakeposting' or 'Two Buttons') — they must explicitly request it as a *template* or *format*.
• Respond `no` if the user is asking for meme content (e.g., "Give me a Drakeposting meme") or does not clearly indicate interest in a blank format/template.

---

Example:

User request:
\"\"\"I want Leonardo Dicaprio memes.\"\"\"

Expected answer:
no

---

User request:
\"\"\"{user_input}\"\"\"
"""
    return prompt


def build_prompt_topic_presence(user_input):
    prompt = f"""
You are a meme-recommendation assistant. 

Analyze the user's request and answer this question:

    "Does the request mention specific topics or keywords for the recommended memes (such as names, events, issues, objects, or recognizable meme formats)?"

Respond with **only one word**: `yes` or `no`.

Use the following rules:
• Respond `yes` if the user is clearly interested in specific topics, keywords, or well-known meme formats (e.g., 'college students', 'taxes', 'Elon Musk', 'graduation', 'climate change', 'Two Buttons', 'Distracted Boyfriend').
• Respond `no` if the request is vague, general, or does not mention any specific subjects or formats.
• If uncertain, make your best guess based on the language of the request.

---

Example:

User request:
\"\"\"Give me Two Buttons meme templates.\"\"\"

Expected answer:
yes

---

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
1. Include only specific people, characters, events, or themes the user explicitly wants memes about.
2. Do NOT include context-setting phrases (like 'bad day'), unrelated preferences (e.g., liking or disliking a subject), or secondary mentions not requested as the main focus.
3. Recognizable meme formats (e.g., "Two Buttons", "Distracted Boyfriend") should also be treated as valid topics.
''')

    if need_template and has_topics and has_usages:
        prompt = f"""
You are a meme-recommendation assistant.

{common_instruction}

Analyze the following user request and return a JSON object with exactly these fields:

1. "topics": A list of specific and relevant subjects the user explicitly wants the meme to be about. This includes people, events, concepts, keywords, themes, or well-known meme formats (e.g., "Drakeposting", "Two Buttons").
2. "usages": A list of the intended purposes or usages for the recommended memes (e.g., humor, comparison, political satire).

Return only the JSON object in the format below. Use your best judgment to identify the user's intent, but only extract what's directly relevant.

---

Example:

User request:
\"\"\"My day has been really bad, I want a meme that can cheer me up. I really like the movie Despicable Me, give me Minions meme templates.\"\"\"

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

1. "topics": A list of specific and relevant subjects the user explicitly wants the meme to be about. This includes people, events, concepts, keywords, themes, or well-known meme formats (e.g., "Drakeposting", "Two Buttons").
2. "recommendation_focus": Specify **exactly one** of `"local"` or `"global"`.  
    • **Global** if **any** of the following conditions is met (case-insensitive):  
      1. A topic exactly equals or contains any **known meme format or template**:  
        ["Two Buttons", "Distracted Boyfriend", "Drakeposting", "Expanding Brain", "Gru's Plan", …]  
      2. A topic exactly equals or contains any **franchise/character** commonly memed:  
        ["Minions", "Despicable Me", "Spongebob", "Superman", "Batman", "Marvel", "DC", …]  
    • **Local** otherwise (i.e. none of the above apply—user cares about the message/text, not a pre‑existing image set).  
    • If multiple topics are listed and at least one triggers **Global**, the final answer must be **Global**.  
    • You must return **only** `"local"` or `"global"`.  
    Examples:
    - Request: “Give me Superman memes about teamwork.”  
      → topics include “Superman” (franchise) ⇒ recommendation_focus = "global"  
    - Request: “Make a meme about Monday motivation.”  
      → no formats, no templates, no franchises ⇒ recommendation_focus = "local"
3. "usages": A list of the intended purposes or usages for the recommended memes (e.g., humor, comparison, political satire).

Return only the JSON object in the format below. Use your best judgment to identify the user's intent, but only extract what's directly relevant.

---

Example:

User request:
\"\"\"My day has been really bad, I want a meme that can cheer me up. I really like the movie Despicable Me, give me some Minions memes.\"\"\"

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

1. "topics": A list of specific and relevant subjects the user explicitly wants the meme to be about. This includes people, events, concepts, keywords, themes, or well-known meme formats (e.g., "Drakeposting", "Two Buttons").
2. "recommendation_focus": Specify **exactly one** of `"local"` or `"global"`.  
    • **Global** if **any** of the following conditions is met (case-insensitive):  
      1. A topic exactly equals or contains any **known meme format or template**:  
        ["Two Buttons", "Distracted Boyfriend", "Drakeposting", "Expanding Brain", "Gru's Plan", …]  
      2. A topic exactly equals or contains any **franchise/character** commonly memed:  
        ["Minions", "Despicable Me", "Spongebob", "Superman", "Batman", "Marvel", "DC", …]  
    • **Local** otherwise (i.e. none of the above apply—user cares about the message/text, not a pre-existing image set).  
    • If multiple topics are listed and at least one triggers **Global**, the final answer must be **Global**.  
    • You must return **only** `"local"` or `"global"`.  
    Examples:
    - Request: “Give me Superman memes about teamwork.”  
      → topics include “Superman” (franchise) ⇒ recommendation_focus = "global"  
    - Request: “Make a meme about Monday motivation.”  
      → no formats, no templates, no franchises ⇒ recommendation_focus = "local"

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

1. "topics": A list of specific and relevant subjects the user explicitly wants the meme to be about. This includes people, events, concepts, keywords, themes, or well-known meme formats (e.g., "Drakeposting", "Two Buttons").

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