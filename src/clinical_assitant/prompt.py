
import os
from openai import OpenAI

os.environ["OPENROUTER_API_KEY"]  = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


from typing import Dict

#  Model and system setup
# MODEL: which LLM to use
# SYSTEM: defines the assistant’s role and behavior (cautious, not a doctor, always recommends seeing a clinician)


MODEL = "openai/gpt-oss-20b:free"

SYSTEM = (
    "You are a cautious medical assistant. "
    "Explain likely differentials, what to rule out, and when to escalate. "
    "You are not a doctor and always advise seeing a clinician for diagnosis."
)

# Few-shot exemplars
# Provide sample interactions (user + assistant) so the model learns the expected style.
# These are strings in message format (role + content).

FEW_SHOTS = [
    {
        "role": "user",
        "content": (
            "Question: Provide differentials and monitoring advice.\n"
            "Structured input:\n"
            '{"symptoms": "sore throat, fever", "labs": "rapid strep negative"}'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Top considerations: viral pharyngitis, early bacterial infection. "
            "Ask about cough, exposure, rash. Recommend fluids, rest, antipyretics. "
            "Escalate if breathing difficulty, dehydration, or persistent high fever."
        ),
    },
]

import json

# Helper: convert structured input into messages
# Takes a dict of structured input + a user question.
# Builds a message list containing system prompt, few-shots, and user query.

def to_messages(struct_input: dict, user_question: str):
    """All content fields are strings; we stringify the structured input."""
    msgs = [{"role": "system", "content": SYSTEM}]
    msgs.extend(FEW_SHOTS)
    msgs.append({
        "role": "user",
        "content": (
            f"Question: {user_question}\n"
            "Structured input:\n"
            f"{json.dumps(struct_input, ensure_ascii=False)}"
        ),
    })
    return msgs


# Main consultation function
# Prepares messages, sends them to the model, and returns the assistant’s response.
# Includes error handling with detailed debug printing if something goes wrong.


def consult(struct_input: dict, question: str, temperature: float = 0.3, max_tokens: int = 400):
    msgs = to_messages(struct_input, question)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        # Helpful debug output if anything else is off
        print("Request failed:", type(e).__name__, "-", str(e))
        print("\nMessages payload that was sent:")
        for m in msgs:
            print(m["role"].upper()+":", (m["content"][:400] + ("..." if len(m["content"])>400 else "")))
        raise

# Example usage
# Provide structured symptoms/lab data and ask what differentials to consider.
structured = {"symptoms": "fever, cough, sore throat", "labs": "WBC mildly elevated"}
print(consult(structured, "What differentials should I consider and what should I monitor?"))


#Temperature Sweep (see how creativity affects answers)

question = "Draft a short patient-friendly explanation of possible causes and next steps."
struct_input = {"symptoms": "fever", "labs": "high WBC"}

for t in [0.0, 0.3, 0.7, 1.0]:
    print(f"\n=== temperature = {t} ===")
    print(consult(struct_input, question, temperature=t, max_tokens=200))



