# src/nano_llm.py
import yaml
from llama_cpp import Llama
from src.rag_retriever import RAGRetriever

class NanoLLM:
    def __init__(self, config_path: str = "./config.yaml"):
        self.config = self._load_config(config_path)
        
        # Load the GGUF model using llama.cpp
        self.model = Llama(
            model_path=f"./models/{self.config['nano_llm']['gguf_file']}",
            n_ctx=self.config['nano_llm']['max_length'],
            n_threads=4, # Adjust this based on your laptop's CPU cores
            verbose=False
        )
        
        self.rag = RAGRetriever()
    
    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_response(self, query: str, patient_id: str = "P001"):
        """Generates response and a lightweight explanation"""
        context = self.rag.retrieve(query, patient_id)
        context_str = "\n\n".join(context)
        
        system_prompt = """You are an empathetic e-Rehabilitation Assistant. 
Use simple language, provide clear exercise instructions, and reference patient history."""
        
        # Prompt format for Phi-3
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\nPatient Context: {context_str}\n\nQuery: {query}\n<|end|>\n<|assistant|>"
        
        output = self.model(
            prompt,
            max_tokens=512,
            temperature=self.config['nano_llm']['temperature'],
            stop=["<|end|>"]
        )
        
        response = output['choices'][0]['text'].strip()
        
        # Prompt-Based XAI (Replaces heavy SHAP/LIME calculations)
        explanation_prompt = f"<|user|>\nBased on this context: {context_str}\nExplain briefly why you gave this recommendation: {response}\n<|end|>\n<|assistant|>"
        explanation_output = self.model(explanation_prompt, max_tokens=150, temperature=0.3, stop=["<|end|>"])
        explanation = explanation_output['choices'][0]['text'].strip()
        
        return response, explanation, context