# 5. src/rag_retriever.py
"""
RAG Pipeline using ChromaDB + Sentence Transformers
Optimized for medical documents + patient data [web:9][web:25]
"""
import langchain_community 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community import document_loaders
from langchain_community import vectorstores

from langchain_community import Chroma
from langchain_community import PyPDFLoader

from langchain_huggingface import HuggingFaceEmbeddings
from src.personalization import PatientDataManager
import os
from typing import List, Dict

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
        """Initialize ChromaDB with medical docs + patient data"""
        chroma_path = self.config['storage']['chroma_path']
        
        # Load medical guidelines + patient data
        documents = self._load_medical_docs()
        patient_docs = self._create_patient_documents()
        all_docs = documents + patient_docs
        
        self.vectorstore = Chroma(
            collection_name="e_rehab",
            embedding_function=self.embeddings,
            persist_directory=chroma_path
        )
        self.vectorstore.add_documents(all_docs)
        self.vectorstore.persist()
    
    def _load_medical_docs(self):
        """Load PDF medical guidelines"""
        docs = []
        pdf_path = "./data/medical_guidelines.pdf"  # Add your PDF here
        if os.path.exists(pdf_path):
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['rag']['chunk_size'],
                chunk_overlap=self.config['rag']['chunk_overlap']
            )
            docs = splitter.split_documents(docs)
        return docs
    
    def _create_patient_documents(self):
        """Convert all patient data to documents"""
        docs = []
        for patient_id, context in self.patient_manager.patients.items():
            patient_context = self.patient_manager.get_patient_context(patient_id)
            doc = {
                'page_content': patient_context,
                'metadata': {'source': f'patient_{patient_id}', 'type': 'personal'}
            }
            docs.append(doc)
        return docs
    
    def retrieve(self, query: str, patient_id: str = "P001", k: int = 5) -> List[str]:
        """Retrieve relevant context for query + patient"""
        # Augment query with patient context
        patient_context = self.patient_manager.get_patient_context(patient_id)
        augmented_query = f"Patient context: {patient_context}\n\nQuery: {query}"
        
        if self.vectorstore:
            relevant_docs = self.vectorstore.similarity_search(augmented_query, k=k)
            return [doc.page_content for doc in relevant_docs]
        return [patient_context]

# ENHANCEMENT POINT: Add hybrid search (BM25 + semantic) for better medical retrieval
