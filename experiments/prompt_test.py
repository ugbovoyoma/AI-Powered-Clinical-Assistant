from transformers import pipeline


clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "fever + cough"

labels = ["flu","malaria","common cold","bronchitis","asthma","food poisoning"]

res = clf(text, candidate_labels=labels, multi_label=True)

sorted(list(zip(res["labels"], res["scores"])), key=lambda x: x[1], reverse=True)[:2]


from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
# The client is configured to use OpenRouter's API endpoint with the provided key.

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Create a chat completion request
messages = [
    {"role": "system", "content": "Answer ONLY with 1–2 likely conditions, separated by commas. No explanations."},
    {"role": "user", "content": "fever + cough"},
]

resp = client.chat.completions.create(
    model="openai/gpt-oss-20b:free",
    messages=messages,
    temperature=0.2,
    max_tokens=40,
    extra_headers={"HTTP-Referer": "https://local", "X-Title": "Clinical Assistant Demo"},
)

# Prefer visible content; fall back to reasoning if content is empty
choice = resp.choices[0]
content = getattr(getattr(choice, "message", None), "content", "") or ""

if content.strip():
    print(content.strip())
else:
    # Fallback: pick 1–2 conditions from the reasoning text
    reasoning = getattr(choice.message, "reasoning", "") or ""
    CANDIDATES = [
        "Influenza", "COVID-19", "Common cold", "Bronchitis", "Pneumonia",
        "Upper respiratory infection"
    ]
    picked = []
    rl = reasoning.lower()
    for c in CANDIDATES:
        if c.lower() in rl and c not in picked:
            picked.append(c)
        if len(picked) == 2:
            break
    if not picked:
        picked = ["Influenza", "COVID-19"]  # sane default
    print(", ".join(picked))