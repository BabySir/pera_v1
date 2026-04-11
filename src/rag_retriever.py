import os
import json
import yaml
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.personalization import PatientDataManager

class RAGRetriever:
    def __init__(self, config_path: str = "./config.yaml"):
        self.config = self._load_config(config_path)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config['rag']['embedding_model'])
        self.patient_manager = PatientDataManager()
        self.vectorstore = self._init_vectorstore()
    
    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_vectorstore(self):
        chroma_path = self.config['storage']['chroma_path']
        return Chroma(
            collection_name="e_rehab",
            embedding_function=self.embeddings,
            persist_directory=chroma_path
        )
    
    def retrieve(self, query: str, patient_id: str = "P001", k: int = 3) -> List[str]:
        """Retrieve relevant medical context and combine with patient history."""
        patient_context = self.patient_manager.get_patient_context(patient_id)
        
        retrieved_texts = [f"CRITICAL PATIENT DATA:\n{patient_context}"]
        
        if self.vectorstore:
            relevant_docs = self.vectorstore.similarity_search(query, k=k)
            retrieved_texts.extend([doc.page_content for doc in relevant_docs])
            
        return retrieved_texts