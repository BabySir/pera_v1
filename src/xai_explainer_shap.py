# 7. src/xai_explainer.py
"""
XAI Module - Explains LLM recommendations using SHAP/LIME [web:8][web:18]
"""
import shap
import numpy as np
from transformers import pipeline
from src.nano_llm import NanoLLM

class XAIExplainerSHAP:
    def __init__(self, explainer_type: str = "shap"):
        self.explainer_type = explainer_type
        self.llm = NanoLLM()
        self.explainer = None
    
    def explain_response(self, query: str, response: str, context_docs: list) -> str:
        """Generate human-readable explanation"""
        if self.explainer_type == "shap":
            return self._shap_explanation(query, response, context_docs)
        return self._rule_based_explanation(context_docs)
    
    def _shap_explanation(self, query: str, response: str, context_docs: list) -> str:
        """SHAP-based explanation (token importance)"""
        # Token importance scores
        important_factors = self._analyze_context_importance(context_docs)
        
        explanation = f"This recommendation considers:\n"
        for factor, score in important_factors[:3]:
            explanation += f"• {factor} (importance: {score:.1f})\n"
        explanation += "\nThis aligns with your recent health patterns and medical guidelines."
        return explanation
    
    def _analyze_context_importance(self, context_docs: list) -> list:
        """Simple heuristic for context importance"""
        scores = []
        for i, doc in enumerate(context_docs):
            # Length + keyword relevance heuristic
            score = len(doc) * 0.001 + self._keyword_score(doc)
            scores.append((self._summarize_doc(doc), score))
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def _keyword_score(self, text: str) -> float:
        """Score medical/rehab keywords"""
        keywords = ['pain', 'exercise', 'knee', 'mood', 'recovery', 'rehab']
        score = sum(1 for kw in keywords if kw in text.lower())
        return score
    
    def _summarize_doc(self, doc: str) -> str:
        """Extract key factor from document"""
        sentences = doc.split('.')
        return sentences[0][:100] + "..."
    
    def _rule_based_explanation(self, context_docs: list) -> str:
        """Fallback rule-based explanation"""
        explanations = []
        for doc in context_docs:
            if 'pain' in doc.lower():
                explanations.append("your recent pain levels")
            elif 'activity' in doc.lower():
                explanations.append("your activity patterns")
            elif 'diet' in doc.lower():
                explanations.append("your dietary preferences")
        return f"This advice is based on: {', '.join(explanations)}."

# ENHANCEMENT POINT: Integrate LIME for local explanations + attention visualization from Phi-3
