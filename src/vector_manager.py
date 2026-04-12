import chromadb
import json
import os

class EvolutionManager:
    def __init__(self, db_path="./vector_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("rehab_knowledge")

    def query_knowledge(self, keywords):
        """Retrieves the most relevant tip from the database."""
        if not keywords:
            return "Focus on gentle consistency."
            
        if isinstance(keywords, list):
            query_text = " ".join(keywords)
        else:
            query_text = str(keywords)
            
        results = self.collection.query(query_texts=[query_text], n_results=1)
        
        if results['documents'] and results['documents'][0]:
            return results['documents'][0][0]
        return "Focus on gentle consistency."

    def add_new_guideline(self, text, metadata):
        """Ingests a single new guideline to update the system's knowledge."""
        doc_id = f"id_{hash(text)}"
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"Added single guideline: {metadata.get('section', 'Unknown')}")

    def ingest_knowledge_base(self, json_path="data/medical_knowledge.json"):
        """Bulk ingests the comprehensive clinical dataset from a JSON file."""
        if not os.path.exists(json_path):
            print(f"Knowledge base file not found at {json_path}")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        for index, item in enumerate(data):
            content = f"Topic: {item.get('section', '')}\nGuideline: {item.get('content', '')}\nEvidence: {item.get('evidence_level', '')}"
            documents.append(content)
            
            keyword_string = ", ".join(item.get('keywords', []))
            metadatas.append({"section": item.get('section', ''), "keywords": keyword_string})
            ids.append(f"clinical_rule_{index}")
        
        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"Successfully ingested {len(documents)} clinical guidelines into PeRA's brain! 🧠")