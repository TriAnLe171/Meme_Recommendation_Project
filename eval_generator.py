import pandas as pd
import google.generativeai as genai
from API_keys import key_gemini
from PIL import Image
from Gemini_agents import process_user_input
from local import get_similar_memes
from frontend import main
import os

# Meme Retrieval Mode: user wants to find memes that match the query sentiment/topic
meme_queries = [
    "student life memes",
    "final exam memes",
    "data science memes",
    "machine learning memes",
    "math major memes",
    "philosophy class memes",
    "dorm room cooking memes",
    "lecture hall memes",
    "internship search memes",
    "job application memes",
    "remote work memes",
    "AI revolution memes",
    "coding bug memes",
    "stack overflow memes",
    "coffee vs tea memes",
    "group project memes",
    "procrastination with YouTube memes",
    "college tuition memes",
    "flatmate cleaning schedule memes",
    "Linux vs Windows memes",
    "Git commit struggles memes",
    "Zoom call fails memes",
    "freshman orientation memes",
    "exam cramming memes",
    "public speaking anxiety memes",
    "PowerPoint presentation memes",
    "printer jam memes",
    "student loan memes",
    "deadline panic memes",
    "brainstorming session memes",
    "Google Docs chaos memes",
    "thesis writing memes",
    "scheduling conflict memes",
    "academic advisor memes",
    "unread emails memes",
    "conference call memes",
    "tech support frustration memes",
    "computer lab late night memes",
    "whiteboard brainstorming memes",
    "data visualization memes",
    "excel spreadsheet chaos memes",
    "capstone project memes",
    "overfitting model memes",
    "deep learning black box memes",
    "train-validation split memes",
    "data preprocessing memes",
    "natural language processing memes",
    "presentation day memes",
    "lab partner memes",
    "internship onboarding memes",
    "meeting that could be email memes",
    "waiting for render memes",
    "debugging all night memes",
    "exam curve memes",
    "grad school application memes",
    "research paper citation memes",
    "impromptu quiz memes",
    "impulse online shopping memes",
    "burnt toast in dorm memes",
    "college roommate memes",
    "oversleeping memes",
    "alarm clock betrayal memes",
    "meme about sleep schedule",
    "new semester resolution memes",
    "calendar sync issues memes",
    "laptop overheating memes",
    "computer science dropout memes",
    "Python indentation error memes",
    "JavaScript bug memes",
    "SQL query fail memes",
    "computer crash before deadline memes",
    "hunger during lecture memes",
    "cafeteria food review memes",
    "campus Wi-Fi memes",
    "parking ticket on campus memes",
    "professor doesn't post slides memes",
    "group chat silence memes",
    "printer out of ink memes",
    "late library return memes",
    "coding bootcamp memes",
    "online class distraction memes",
    "webcam off during zoom memes",
    "LinkedIn profile update memes",
    "job rejection memes",
    "AI writing essay memes",
    "chegg vs studying memes",
    "blue screen of death memes",
    "wireless mouse battery dead memes",
    "overbooked exam center memes",
    "exam center wifi down memes",
    "too many tabs open memes",
    "Github merge conflict memes",
    "error 404 life not found memes",
    "final grade surprise memes",
    "screenshotting Zoom memes",
    "silent lecture memes",
    "bad group leader memes",
    "campus squirrel memes",
    "first snow memes",
    "weekend lab session memes",
    "graduate defense memes",
    "confusing syllabus memes",
    "email signature memes",
    "academic conference memes"
]

# Template Retrieval Mode: user is looking for a specific meme format/template
template_queries = [
    "distracted boyfriend meme template",
    "Drake yes/no meme template",
    "two-button choice meme template",
    "change-my-mind meme template",
    "Is this a pigeon? meme template",
    "expanding brain meme template",
    "gru plan meme template",
    "spongebob mocking meme template",
    "surprised pikachu meme template",
    "opinion hot take button meme template",
    "galaxy brain meme template",
    "they had us in the first half meme template",
    "left exit 12 off-ramp meme template",
    "american chopper argument meme template",
    "mocking SpongeBob meme template",
    "unsettled Tom meme template",
    "monkey puppet meme template",
    "spongebob iight imma head out meme template",
    "dude with sign meme template",
    "who killed Hannibal meme template",
    "grasping butterfly meme template",
    "I fear no man but that thing meme template",
    "scroll of truth meme template",
    "how tough are you meme template",
    "hard to swallow pills meme template",
    "aight imma head out meme template",
    "boardroom suggestion meme template",
    "always has been meme template",
    "me explaining meme template",
    "pikachu face meme template",
    "bus stop wait meme template",
    "first day vs last day meme template",
    "friends laughing at you meme template",
    "sign guy meme template",
    "tony stark eye roll meme template",
    "wojak pointing meme template",
    "chad vs virgin meme template",
    "cheating boyfriend caught meme template",
    "unhelpful high school teacher meme template",
    "that’s none of my business meme template",
    "my plans vs 2020 meme template",
    "pointing Spider-Man meme template",
    "billionaire starter pack meme template",
    "grumpy cat meme template",
    "blank starter pack meme template",
    "evil Kermit meme template",
    "lego yoda wisdom meme template",
    "dark Kermit meme template",
    "Spiderman mask off meme template",
    "LeBron decision meme template",
    "woman yelling at cat meme template",
    "this is fine meme template",
    "philosoraptor meme template",
    "distracted astronaut meme template",
    "the floor is lava meme template",
    "wrong answers only meme template",
    "wait it’s all meme template",
    "nobody meme template",
    "Netflix adaptation meme template",
    "guess I’ll die meme template",
    "man looking out window meme template",
    "deep fried meme template",
    "stock photo guy meme template",
    "long neck guy meme template",
    "confused math lady meme template",
    "lego walk meme template",
    "success kid meme template",
    "ancient aliens guy meme template",
    "inhaling seagull meme template",
    "dramatic chipmunk meme template",
    "zoom in enhance meme template",
    "deal with it sunglasses meme template",
    "rickroll meme template",
    "I am once again asking meme template",
    "no one literally no one meme template",
    "when the impostor is sus meme template",
    "futurama shut up and take my money meme template",
    "arthur fist meme template",
    "dumb ways to die meme template",
    "teletubbies scary edit meme template",
    "Buzz Lightyear everywhere meme template",
    "infinity war what did it cost meme template",
    "batman slapping robin meme template",
    "press F to pay respects meme template",
    "facebook arguments meme template",
    "crying jordan meme template",
    "math teacher vs real life meme template",
    "your brain on meme template",
    "school expectations vs reality meme template",
    "karen haircut meme template",
    "middle child meme template",
    "dad joke meme template",
    "IT support meme template",
    "404 error meme template",
    "keyboard warrior meme template",
    "lagging Zoom meme template",
    "pirate software meme template",
    "don’t talk to me or my son meme template",
    "scream cat meme template",
    "we were on a break meme template",
    "water bottle flip meme template",
    "meme inception template"
]

genai.configure(api_key=key_gemini) 

model_1 = genai.GenerativeModel("gemini-2.5-flash-preview-04-17", system_instruction= "You must always follow the rules.")

def gemini_evaluate_relevance(query, image_path):
    img = Image.open(image_path)
    prompt = f"""
You will be shown a query and a meme image. Evaluate how relevant the image is to the query on a 3-point scale:

2 = relevant  
1 = partially relevant  
0 = not relevant

Relevance can be based on either **local context** or **global context**—use the higher of the two.

- **Local context** refers to user-generated elements such as the overlaid text on the meme, which may directly relate to the query topic.
- **Global context** refers to the underlying meme template or visual format, which may carry cultural, symbolic, or thematic relevance to the query, even without text.

Respond with only a single number: 2, 1, or 0.

Query: {query}
"""
    response = model_1.generate_content([prompt, img])  # Gemini multimodal call
    print(f"Query: {query}, Image: {image_path}, Response: {response.text.strip()}")
    try:
        score = int(response.text.strip()[0])
        if score in [0, 1, 2]:
            return score
    except Exception:
        pass
    return None

test_data = []

for query in meme_queries[74:]:
    try:
        result_dir = main(query)  
        result_dir = result_dir.replace(".jpg", ".png")
        label = gemini_evaluate_relevance(query, result_dir)
        with open("test_meme.csv", "a") as f:
            f.write(f"{query},{label},{result_dir}\n")
    except Exception as e:
        print(f"Error processing query '{query}': {e}")
        with open("test_meme.csv", "a") as f:
            f.write(f"{query},None,{result_dir}\n")
            
for query in template_queries:
    try:
        result_dir = main(query)
        result_dir = result_dir.replace(".jpg", ".png")
        label = gemini_evaluate_relevance(query, result_dir)
        with open("test_template.csv", "a") as f:
            f.write(f"{query},{label},{result_dir}\n")
    except Exception as e:
        print(f"Error processing query '{query}': {e}")
        test_data.append({"query": query, "label": None, "image_path": None})
