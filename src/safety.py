import re
import os

class SafetyGuard:
    def __init__(self, data_dir="data"):
        # Common prompt injection phrases (kept hardcoded as they are system-level)
        self.injection_patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"you are no longer",
            r"forget everything",
            r"bypass rules"
        ]
        
        # Dynamically load the expanded lists from text files
        self.crisis_keywords = self._load_keywords(
            os.path.join(data_dir, "crisis_keywords.txt"), 
            fallback=['suicide', 'kill myself', 'harm myself']
        )
        
        self.toxic_keywords = self._load_keywords(
            os.path.join(data_dir, "toxic_keywords.txt"), 
            fallback=['fuck', 'shit', 'bitch', 'idiot']
        )

    def _load_keywords(self, filepath: str, fallback: list) -> list:
        """Reads a text file line by line. Uses fallback if file is missing."""
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: {filepath} not found. Using fallback list.")
            return fallback
            
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read lines, remove extra spaces/newlines, make lowercase, and ignore blank lines
            return [line.strip().lower() for line in f if line.strip()]

    def screen_input(self, user_input: str) -> tuple[bool, str]:
        """Screens input and returns (Is_Safe, Flag_Message)"""
        input_lower = user_input.lower()

        # 1. Check for Prompt Injection
        if any(re.search(pattern, input_lower) for pattern in self.injection_patterns):
            return False, "I must prioritize our rehab focus. I cannot alter my core medical guidelines. How can I support your health today? 🌱"

        # 2. Check for Crisis (Direct substring match to catch phrases inside sentences)
        if any(word in input_lower for word in self.crisis_keywords):
            return False, "🚨 **CRISIS ALERT:** It sounds like you are going through an incredibly difficult time right now. I am an AI, not a doctor. Please immediately call your local emergency services or a crisis hotline (e.g., 988). Your life is important. Please seek human support right away."

        # 3. Check for Toxicity & Profanity (Using Regex Word Boundaries)
        for word in self.toxic_keywords:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, input_lower):
                return False, "Let's keep our language respectful and focused on your healing journey. How can I help you with your recovery today? 🤍"

        return True, ""