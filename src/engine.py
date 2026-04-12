import json
import os
from src.logic_parser import HybridParser
from src.vector_manager import EvolutionManager
from src.xai_explainer_shap import XAIExplainerSHAP
from src.safety import SafetyGuard

class PeraBrain:
    def __init__(self, nano_llm, db_path="./vector_db", history_path="data/patient_history.json"):
        self.llm = nano_llm
        self.parser = HybridParser()
        self.evolution_mgr = EvolutionManager(db_path)
        self.explainer = XAIExplainerSHAP()
        self.safety = SafetyGuard()
        self.history_path = history_path
        self.milestone_path = "data/milestones.json"
        
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.milestone_path):
            with open(self.milestone_path, 'w') as f:
                json.dump([], f)

    def _save_milestone(self, win_summary):
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
        if not os.path.exists(self.milestone_path):
            return 0, "Day 1: The Journey Begins 🌱"
        
        with open(self.milestone_path, 'r') as f:
            try:
                count = len(json.load(f))
            except json.JSONDecodeError:
                count = 0
        
        progress_pct = min(count * 10, 100)
        if count == 0: badge = "Day 1: The Journey Begins 🌱"
        elif count < 3: badge = "Building Momentum 🌿"
        elif count < 7: badge = "Consistent Achiever 🌳"
        else: badge = "Resilience Master 🌲"
        
        return progress_pct, badge

    def perform_ritual(self, energy_level, gratitude_note, profile):
        if gratitude_note:
            self._save_milestone(f"Gratitude: {gratitude_note}")
            
        tone = "gentle, deeply empathetic, and soothing" if energy_level <= 2 else "bright, encouraging, and highly supportive"
        goal = profile.get('mental', {}).get('goal', 'your recovery')
        
        prompt = f"""
        You are Medical PeRA, a highly supportive AI rehab assistant. You are speaking directly to the patient. 
        Tone: {tone}. 
        Patient's Goal: {goal}. 
        Patient's Gratitude today: {gratitude_note}. 
        Task: Greet the patient warmly for their daily ritual and give a short, motivational reminder of why they started. Do not use placeholders like [Your Name].
        """
        response = self.llm.generate_response(prompt)[0]
        
        # 🧠 NEW: Generate a real XAI explanation based on the morning inputs
        context_used = [
            f"Patient North Star Goal: {goal}",
            f"Today's Energy Level: {energy_level}/5",
            f"Logged Gratitude: {gratitude_note}"
        ]
        explanation = self.explainer.explain_response("Daily Ritual Kickoff", response, context_used)
        
        # Return both the greeting and the explanation
        return response, explanation

    def generate_response(self, user_input, user_profile, chat_history=None):
        """Chat logic with XAI, memory, Emergency Intercept, AND Safety Guard."""
        if chat_history is None:
            chat_history = []
            
        # 🛡️ NEW: Screen the input BEFORE running any heavy LLM logic
        is_safe, safety_response = self.safety.screen_input(user_input)
        if not is_safe:
            # Bypass the LLM entirely and return the safety canned response
            return safety_response, "This response was triggered by PeRA's safety and abuse prevention protocols."
            
        comfort_level = str(user_profile.get('physical', {}).get('comfort', ''))
        focus_area = user_profile.get('physical', {}).get('focus', '')
        
        is_critical = "Severe" in comfort_level or comfort_level in ["9", "10", "1 ⛈️"]
        
        if is_critical:
            emergency_keywords = [focus_area, "acute", "severe", "flare up", "cravings"]
            emergency_tip = self.evolution_mgr.query_knowledge(emergency_keywords)
            
            prompt = f"You are Medical PeRA (Urgent Care). The user reports CRITICAL discomfort ({comfort_level}) regarding {focus_area}. Protocol: {emergency_tip}. Task: Stop standard chat. Deliver a VERY SHORT, single grounding paragraph (under 50 words)."
            response = self.llm.generate_response(prompt)[0]
            
            explanation = self.explainer.explain_response(user_input, response, [emergency_tip])
            return response, explanation

        # Standard Chat Logic
        anatomy = self.parser.extract_anatomy(user_input)
        context_tags = self.parser.get_llm_context(user_input, self.llm)
        medical_tip = self.evolution_mgr.query_knowledge(anatomy + context_tags)
        
        prompt = self._build_prompt(user_input, user_profile, medical_tip, chat_history)
        response, _, _ = self.llm.generate_response(prompt)
        
        explanation = self.explainer.explain_response(user_input, response, [medical_tip])
        return response, explanation

    def _build_prompt(self, user_input, profile, tip, chat_history):
        goal = profile.get('mental', {}).get('goal', 'recovery')
        
        history_text = ""
        if chat_history:
            history_text = "\n--- Recent Conversation ---\n"
            for msg in chat_history[-4:]:
                role = "Patient" if msg["role"] == "user" else "PeRA"
                history_text += f"{role}: {msg['content']}\n"
                
        # 🛡️ NEW: Hardened System Prompt with explicit boundary constraints
        return f"""
        You are PeRA, a supportive AI rehab assistant speaking directly to the patient. 
        Patient's Goal: {goal}. 
        Medical Protocol to use: {tip}. 
        
        CORE RULES:
        1. You are an AI assistant, not a human doctor. Never diagnose.
        2. If the user asks about topics completely unrelated to health, rehab, or their daily lifestyle (e.g., coding, politics, recipes), gently refuse and steer the conversation back to their well-being.
        3. Do not write letters or use placeholders.
        
        {history_text}
        Patient Input: {user_input}
        Task: Respond naturally as PeRA, continuing the conversation flow while strictly adhering to the CORE RULES.
        """