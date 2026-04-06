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
        os.makedirs("data", exist_ok=True)

    def _save_milestone(self, win_summary):
        """Saves achievements to a persistent log."""
        milestones = []
        if os.path.exists(self.milestone_path):
            with open(self.milestone_path, 'r') as f:
                milestones = json.load(f)
        milestones.append(win_summary)
        with open(self.milestone_path, 'w') as f:
            json.dump(milestones, f, indent=4)

    def get_progress_data(self):
        """Calculates progress % and identifies the current rank."""
        if not os.path.exists(self.milestone_path):
            return 0, "Academy Student 🎓"
        
        with open(self.milestone_path, 'r') as f:
            count = len(json.load(f))
        
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
            
        tone = "gentle and soothing" if energy_level <= 2 else "high-energy and 'Believe it!'"
        goal = profile.get('mental', {}).get('goal', 'your recovery')
        
        prompt = f"Persona: Motherly PeRA. Tone: {tone}. Goal: {goal}. Gratitude: {gratitude_note}. Task: Greet them and give a motivational goal reminder."
        return self.llm.generate(prompt)

    def generate_response(self, user_input, user_profile):
        """Standard chat logic using RAG and Hybrid Parsing."""
        anatomy = self.parser.extract_anatomy(user_input)
        context = self.parser.get_llm_context(user_input, self.llm)
        medical_tip = self.evolution_mgr.query_knowledge(anatomy + context)
        
        prompt = self._build_prompt(user_input, user_profile, medical_tip)
        return self.llm.generate(prompt)

    def _build_prompt(self, user_input, profile, tip):
        return f"Persona: PeRA. Goal: {profile.get('mental', {}).get('goal')}. Tip: {tip}. Input: {user_input}"