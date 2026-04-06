import chromadb

class EvolutionManager:
    def __init__(self, db_path="./vector_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("rehab_knowledge")

    def query_knowledge(self, keywords):
        """Searches the database using the hybrid keywords."""
        query_text = " ".join(keywords)
        results = self.collection.query(query_texts=[query_text], n_results=1)
        return results['documents'][0] if results['documents'] else "Focus on gentle consistency."

    def add_new_guideline(self, text, metadata):
        """Logic for system updation with new external data."""
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[f"id_{hash(text)}"]
        )