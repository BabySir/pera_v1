import json
import os
from src.logic_parser import HybridParser
from src.vector_manager import EvolutionManager

class PeraBrain:
    def __init__(self, nano_llm, db_path="./vector_db", history_path="data/patient_history.json"):
        self.llm = nano_llm
        self.parser = HybridParser()
        self.evolution_mgr = EvolutionManager(db_path)
        self.history_path = history_path
        self.milestone_path = "data/milestones.json"
        
        # Ensure data directory exists and milestone file is initialized
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.milestone_path):
            with open(self.milestone_path, 'w') as f:
                json.dump([], f)

    def _save_milestone(self, win_summary):
        """Saves achievements to a persistent log."""
        milestones = []
        if os.path.exists(self.milestone_path):
            with open(self.milestone_path, 'r') as f:
                try:
                    milestones = json.load(f)
                except json.JSONDecodeError:
                    milestones = []
        milestones.append(win_summary)
        with open(self.milestone_path, 'w') as f:
            json.dump(milestones, f, indent=4)

    def get_progress_data(self):
        """Calculates progress % and identifies the current Ninja Rank."""
        if not os.path.exists(self.milestone_path):
            return 0, "Academy Student 🎓"
        
        with open(self.milestone_path, 'r') as f:
            try:
                count = len(json.load(f))
            except json.JSONDecodeError:
                count = 0
        
        # Mastery logic: 10 wins = 100%
        progress_pct = min(count * 10, 100)
        if count == 0: badge = "Academy Student 🎓"
        elif count < 3: badge = "Genin 🍃"
        elif count < 7: badge = "Chunin 🎴"
        else: badge = "Hokage 🏔️"
        
        return progress_pct, badge

    def perform_morning_ritual(self, energy_level, gratitude_note, profile):
        """Calibrates tone and provides a motivational anchor."""
        if gratitude_note:
            self._save_milestone(f"Gratitude: {gratitude_note}")
            
        # Tone calibration based on energy level
        tone = "gentle and soothing" if energy_level <= 2 else "high-energy and 'Believe it!'"
        goal = profile.get('mental', {}).get('goal', 'your recovery')
        
        prompt = f"Persona: Motherly PeRA. Tone: {tone}. Goal: {goal}. Gratitude: {gratitude_note}. Task: Greet them and give a motivational goal reminder."
        return self.llm.generate_response(prompt)[0] # Returns only the response text

    def generate_response(self, user_input, user_profile):
        """Chat logic with Emergency Protocol Intercept."""
        
        # 1. Check for Critical Comfort Levels from Step 1
        comfort_level = str(user_profile.get('physical', {}).get('comfort', ''))
        focus_area = user_profile.get('physical', {}).get('focus', '')
        
        # Define what triggers an emergency
        is_critical = "Severe" in comfort_level or comfort_level in ["9", "10", "1 ⛈️"]
        
        if is_critical:
            # 🛑 Emergency Intercept: Force RAG to pull acute/flare-up protocols
            emergency_keywords = [focus_area, "acute", "severe", "flare up", "cravings", "management"]
            emergency_tip = self.evolution_mgr.query_knowledge(emergency_keywords)
            
            prompt = f"""
            Persona: Medical PeRA (Urgent Care Mode). 
            The user is reporting CRITICAL discomfort ({comfort_level}) regarding {focus_area}.
            Medical Protocol to deliver: {emergency_tip}
            Task: Stop standard chat. Deliver a VERY SHORT, single grounding paragraph (under 50 words). 
            Focus only on immediate comfort and the absolute first step of the protocol. 
            Do not ask complex questions. Format the text simply so it is easily read by a screen reader.
            """
            return self.llm.generate_response(prompt)[0]

        # 🌱 2. Standard Chat Logic (if not critical)
        anatomy = self.parser.extract_anatomy(user_input)
        context_tags = self.parser.get_llm_context(user_input, self.llm)
        medical_tip = self.evolution_mgr.query_knowledge(anatomy + context_tags)
        
        prompt = self._build_prompt(user_input, user_profile, medical_tip)
        response, _, _ = self.llm.generate_response(prompt)
        return response

    def _build_prompt(self, user_input, profile, tip):
        goal = profile.get('mental', {}).get('goal', 'recovery')
        return f"Persona: PeRA. Goal: {goal}. Medical Tip: {tip}. User Input: {user_input}"