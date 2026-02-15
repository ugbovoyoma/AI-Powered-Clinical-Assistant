import faiss
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from advanced_prompts import to_messages, consult, SYSTEM

# Turn dataset into vectors, index with FAISS, retrieve top-K similar
class DiseaseKB:
    """
    Minimal knowledge base:
      - keeps a list of disease entries (name + text fields)
      - builds embeddings using Sentence-Transformers
      - indexes with FAISS for fast similarity search
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.entries: List[Dict] = []
        self.index = None  
        self.matrix = None
        
    def add_entries(self, rows: List[Dict[str, str]]):
        """
        rows: list of dicts with keys like:
          - 'disease' (required)
          - 'symptoms' (string)
          - 'description' (string)
        """
        self.entries.extend(rows)

    def _entry_text(self, row: Dict[str, str]) -> str:
        """Concatenate fields to form a single searchable text string."""
        parts = [
            f"Disease: {row.get('disease','')}",
            f"Symptoms: {row.get('symptoms','')}",
            f"Description: {row.get('description','')}",
        ]
        return " | ".join(parts)

    def build(self):
        """Encode all entries and create a FAISS index."""
        texts = [self._entry_text(r) for r in self.entries]
        if not texts:
            raise ValueError("No entries to index. Add entries first.")
        emb = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.matrix = emb.astype(np.float32)
        dim = self.matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.matrix)

    def query(self, text: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Return the top_k most similar entries to the input text.
        Output: list of (entry_dict, score) pairs sorted by score desc.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() after adding entries.")
        q = self.encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q, top_k)
        results = []
        for i, s in zip(idxs[0], scores[0]):
            results.append((self.entries[int(i)], float(s)))
        return results
    
EXAMPLE_DISEASES = [
    {"disease": "Influenza",
     "symptoms": "fever, cough, sore throat, muscle aches, fatigue",
     "description": "Acute viral respiratory illness; often seasonal; sudden onset of fever and myalgia."},
    {"disease": "Malaria",
     "symptoms": "fever (often periodic), chills, sweats, headache, malaise",
     "description": "Parasitic infection via mosquitoes; consider travel/residence in endemic areas."},
    {"disease": "Typhoid fever",
     "symptoms": "prolonged fever, abdominal pain, constipation or diarrhea, headache",
     "description": "Systemic Salmonella Typhi; suspect contaminated food/water exposure."},
    {"disease": "COVID-19",
     "symptoms": "fever, cough, shortness of breath, loss of taste or smell, fatigue",
     "description": "Viral respiratory infection; severity varies; guided by exposure/testing."},
    {"disease": "Dengue",
     "symptoms": "high fever, severe headache, retro-orbital pain, myalgia, rash",
     "description": "Mosquito-borne viral illness; travel to tropical/subtropical regions; watch for warning signs."},
    {"disease": "Pneumonia",
     "symptoms": "fever, productive cough, pleuritic chest pain, dyspnea",
     "description": "Infection of lung parenchyma; bacterial/viral; focal consolidation may support."},
]


def build_and_demo_search():
    """
    Build the FAISS index from the example dataset,
    then run a demo query: 'malaria symptoms' and print top 3 matches.
    """
    kb = DiseaseKB()
    kb.add_entries(EXAMPLE_DISEASES)
    kb.build()

    user_query = "malaria symptoms"  # <-- your test input
    results = kb.query(user_query, top_k=3)

    print("\n--- Knowledge Search: Top 3 similar diseases ---")
    for rank, (entry, score) in enumerate(results, start=1):
        print(f"{rank}. {entry['disease']} (score: {score:.3f})")
        print(f"   symptoms: {entry.get('symptoms','')}")
        print(f"   note: {entry.get('description','')}\n")




if __name__ == "__main__":

    # Prompt section demo
    struct_input = {"symptoms": "fever", "labs": "high WBC"}
    question = "Draft a short patient-friendly explanation of possible causes and next steps."
    for t in [0.0, 0.3, 0.7, 1.0]:
      print(f"\n=== temperature = {t} ===")
      print(consult(struct_input, question, temperature=t, max_tokens=200))


    # Retrieval section demo
    build_and_demo_search()

