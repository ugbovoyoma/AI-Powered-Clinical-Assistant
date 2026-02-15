
import json, re
from pydantic import BaseModel, Field, ValidationError


class DiagnosisOutput(BaseModel):
    symptoms: list[str]
    reasoning: str
    diagnosis: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_sources: list[str]

def _extract_json(blob: str) -> str:
    """Pull the first {...} JSON object out of the model's reply."""
    m = re.search(r"\{[\s\S]*\}", blob)
    return m.group(0) if m else blob

# Tiny helper to map cosine score [-1..1] → confidence [0..1]
def _score_to_conf(score: float) -> float:
    raw = (float(score) + 1.0) / 2.0
    return float(np.clip(raw, 0.0, 1.0))

SCHEMA_EXAMPLE = {
    "symptoms": ["fever", "chills"],
    "reasoning": "Fever + chills suggests malaria vs influenza; context supports malaria.",
    "diagnosis": "Malaria",
    "confidence": 0.72,
    "supporting_sources": ["json:12:Malaria", "json:31:Dengue"]
}

AGENT_SYSTEM = (
    "You are a clinical reasoning assistant. Return ONLY valid JSON matching the schema. "
    "Use ONLY the provided Context; if uncertain, set diagnosis to 'uncertain' and lower confidence."
)

def _consult_json(symptoms_list: list[str], hits: list[dict]) -> DiagnosisOutput:
    # Build context + sources from retrieve() hits
    context, sources = _build_context(hits)

    user_prompt = (
        "Schema (example):\n" + json.dumps(SCHEMA_EXAMPLE, indent=2) +
        "\n\nContext:\n" + context +
        "\n\nTask: Using ONLY the Context, output STRICT JSON per the Schema. Do not add prose."
    )

    # If no API key, produce a simple, honest fallback so the notebook still runs.
    if not OPENROUTER_API_KEY:
        top = hits[0] if hits else None
        best_label = top["label"] if top else "uncertain"
        conf = _score_to_conf(top["score"]) if top else 0.2
        return DiagnosisOutput(
            symptoms=symptoms_list,
            reasoning="Heuristic fallback using the top retrieved chunk.",
            diagnosis=best_label,
            confidence=conf,
            supporting_sources=sources,
        )

    # Call the model
    payload = {
        "model": MODEL_NAME_FOR_CHAT,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://local",
        "X-Title": "llm_agent",
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"].strip()
    blob = _extract_json(raw)

    # Validate JSON with Pydantic
    try:
        return DiagnosisOutput.model_validate_json(blob)
    except ValidationError:
        data = json.loads(blob)
        for k in SCHEMA_EXAMPLE:
            data.setdefault(k, SCHEMA_EXAMPLE[k])
        return DiagnosisOutput(**data)

def diagnose(symptoms: list[str], k: int = 6) -> DiagnosisOutput:
    """
    End-to-end agent for one case:
      - Build a natural language query from symptoms
      - Retrieve top-K evidence from the KB
      - Ask the model to produce STRICT JSON (or fallback if no key)
    """
    text = ", ".join(symptoms)
    query = f"Symptoms: {text}. Consider differentials and distinguishing features."
    hits = retrieve(query, top_k=k)
    return _consult_json(symptoms, hits)


print(diagnose(["fever", "chills"]).model_dump())
print(diagnose(["abdominal pain", "diarrhea", "headache"]).model_dump())