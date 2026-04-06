import os
import json
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.personalization import PatientDataManager

class RAGRetriever:
    def __init__(self, config_path: str = "./config.yaml"):
        self.config = self._load_config(config_path)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['rag']['embedding_model']
        )
        self.vectorstore = None
        self.patient_manager = PatientDataManager()
        self._init_vectorstore()
    
    def _load_config(self, path):
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_vectorstore(self):
        """Initialize ChromaDB with JSON medical docs"""
        chroma_path = self.config['storage']['chroma_path']
        
        # Load medical guidelines
        documents = self._load_medical_docs()
        
        self.vectorstore = Chroma(
            collection_name="e_rehab",
            embedding_function=self.embeddings,
            persist_directory=chroma_path
        )
        
        if documents:
            self.vectorstore.add_documents(documents)
    
    def _load_medical_docs(self):
        """Load JSON medical guidelines and convert to LangChain Documents"""
        docs = []
        json_path = "./data/medical_knowledge.json"
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)

            for item in knowledge_data:
                # Use .get() to safely handle the keys in your specific JSON
                section = item.get('section', 'General Guidance')
                content_body = item.get('content', 'No content available.')
                evidence = item.get('evidence_level', 'N/A')
                tags = ", ".join(item.get('keywords', []))

                # Build a clean string for the LLM to read
                content = f"Topic: {section}\n"
                content += f"Guideline: {content_body}\n"
                content += f"Evidence Level: {evidence}\n"
                content += f"Keywords: {tags}"

                doc = Document(
                    page_content=content, 
                    metadata={'source': section, 'type': 'medical_guideline'}
                )
                docs.append(doc)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['rag']['chunk_size'],
                chunk_overlap=self.config['rag']['chunk_overlap']
            )
            docs = splitter.split_documents(docs)
            
        return docs
    
    def retrieve(self, query: str, patient_id: str = "P001", k: int = 5) -> List[str]:
        """Retrieve relevant context for query and ALWAYS include patient profile"""
        patient_context = self.patient_manager.get_patient_context(patient_id)
        augmented_query = f"Patient context: {patient_context}\n\nQuery: {query}"
        
        retrieved_texts = []
        # Hardcode the patient profile at the top so the AI never misses it
        retrieved_texts.append(f"CRITICAL PATIENT DATA:\n{patient_context}")
        
        if self.vectorstore:
            relevant_docs = self.vectorstore.similarity_search(augmented_query, k=k)
            retrieved_texts.extend([doc.page_content for doc in relevant_docs])
            
        return retrieved_texts