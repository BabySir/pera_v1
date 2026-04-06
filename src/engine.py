from src.logic_parser import HybridParser
from src.vector_manager import EvolutionManager

class PeraBrain:
    def __init__(self, nano_llm, db_path="./vector_db"):
        self.llm = nano_llm
        self.parser = HybridParser()
        self.evolution_mgr = EvolutionManager(db_path)
        
    def generate_response(self, user_input, user_profile):
        # 1. Parsing: Rule-based + Neural context
        anatomy = self.parser.extract_anatomy(user_input)
        context = self.parser.get_llm_context(user_input, self.llm)
        
        # 2. Retrieval: Smart Search in the Vector DB
        keywords = anatomy + context
        medical_tip = self.evolution_mgr.query_knowledge(keywords)
        
        # 3. Generation: LoRA-style Empathetic Response
        prompt = self._build_prompt(user_input, user_profile, medical_tip)
        return self.llm.generate(prompt)

    def _build_prompt(self, user_input, profile, tip):
        return f"""
        Persona: Warm, motherly rehab assistant named PeRA.
        User Goal: {profile.get('goal', 'wellness')}
        Medical Tip to include: {tip}
        User Input: "{user_input}"
        
        Task: 
        1. Validate feelings warmly.
        2. Incorporate the medical tip naturally.
        3. End with encouragement toward their goal.
        """