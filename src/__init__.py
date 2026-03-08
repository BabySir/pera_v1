"""
e-Rehabilitation Assistant - Core Package
MTech AI/ML Project - All components importable
"""
__version__ = "1.0.0"
__author__ = "SAgar Sharma"

from .personalization import PatientDataManager
from .rag_retriever import RAGRetriever
from .nano_llm import NanoLLM
from .xai_explainer_lime import XAIExplainerLIME
from xai_explainer_shap import XAIExplainerSHAP
from .app import main as run_app

__all__ = [
    "PatientDataManager",
    "RAGRetriever", 
    "NanoLLM",
    "XAIExplainer",
    "run_app"
]
