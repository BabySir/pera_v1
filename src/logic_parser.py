import re

class HybridParser:
    def __init__(self):
        # Rule-based map for instant, zero-cost anatomy detection
        self.anatomy_map = {
            "knee": ["patella", "meniscus", "leg", "joint"],
            "back": ["spine", "lumbar", "sciatica"],
            "shoulder": ["arm", "rotator", "neck", "scapula"]
        }

    def extract_anatomy(self, text):
        """Identifies physical focus areas using keywords."""
        text = text.lower()
        found = [key for key, syns in self.anatomy_map.items() 
                 if key in text or any(s in text for s in syns)]
        return found if found else ["general"]

    def get_llm_context(self, user_input, llm):
        """Extracts emotional/lifestyle tags using the LLM."""
        prompt = f"Identify 1 emotion and 1 lifestyle keyword from: '{user_input}'. Reply only as: 'Emotion, Lifestyle'"
        try:
            response = llm.generate(prompt)
            return [word.strip() for word in response.split(",")]
        except Exception:
            return ["Neutral", "Active"]