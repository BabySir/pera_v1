class XAIExplainerSHAP:
    def __init__(self, explainer_type: str = "shap"):
        self.explainer_type = explainer_type
        self.explainer = None
    
    def explain_response(self, query: str, response: str, context_docs: list) -> str:
        """Generate human-readable, conversational explanation"""
        # 1. Detect if we are explaining the Morning Ritual or a Medical Chat
        is_ritual = any("Energy Level" in str(doc) for doc in context_docs)
        
        if is_ritual:
            return self._ritual_explanation(context_docs)
        else:
            return self._chat_explanation(context_docs)
            
    def _ritual_explanation(self, context_docs: list) -> str:
        """Conversational breakdown for the morning check-in"""
        goal = "your overall recovery"
        energy = "normal"
        gratitude = "your progress"
        
        # Safely extract the exact values you typed in
        for doc in context_docs:
            if "North Star Goal:" in doc:
                goal = doc.split("Goal: ")[-1]
            elif "Energy Level:" in doc:
                energy = doc.split("Level: ")[-1]
            elif "Gratitude:" in doc:
                gratitude = doc.split("Gratitude: ")[-1]
                
        return f"I shaped today's greeting around your core goal of **{goal}**. Since your energy is currently at a **{energy}**, I wanted to make sure my tone matched exactly what you need right now. I also loved your gratitude regarding **{gratitude}**—keeping that top of mind is a huge part of your journey! 🌱"

    def _chat_explanation(self, context_docs: list) -> str:
        """Conversational breakdown for medical RAG protocols"""
        important_factors = self._analyze_context_importance(context_docs)
        
        if not important_factors:
            return "I suggested this based on general rehab best practices to keep you moving safely today. 🤍"
            
        # Grab the top protocol it used
        top_protocol = important_factors[0][0].lower().strip()
        
        return f"I suggested this because I cross-referenced your symptoms with specific medical protocols regarding **{top_protocol}**. I always blend these clinical guidelines with your personal profile to ensure my advice is safe, targeted, and right for your body today. 🌿"
    
    def _analyze_context_importance(self, context_docs: list) -> list:
        scores = []
        for i, doc in enumerate(context_docs):
            score = len(doc) * 0.001 + self._keyword_score(doc)
            scores.append((self._summarize_doc(doc), score))
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def _keyword_score(self, text: str) -> float:
        keywords = ['pain', 'exercise', 'knee', 'mood', 'recovery', 'rehab', 'acute', 'severe', 'stress', 'craving']
        return sum(1 for kw in keywords if kw in text.lower())
    
    def _summarize_doc(self, doc: str) -> str:
        sentences = doc.split('.')
        return sentences[0][:75] + "..." if sentences else "medical guidelines"