# 4. src/personalization.py
"""
Patient Data Handler - Loads and processes personal health records
ENHANCEMENT: Integrate with Google BigQuery for your cloud workflow [memory: user GCP pref]
"""
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

class PatientDataManager:
    def __init__(self, data_path: str = "./data/sample_patient_data.json"):
        self.data_path = data_path
        self.patients = self._load_data()
    
    def _load_data(self) -> Dict[str, Dict]:
        """Load patient JSON data into memory"""
        with open(self.data_path, 'r') as f:
            return {p['patient_id']: p for p in json.load(f)}
    
    def get_patient_context(self, patient_id: str) -> str:
        """Convert patient data to RAG-friendly context string"""
        if patient_id not in self.patients:
            return "No patient data available."
        
        patient = self.patients[patient_id]
        context = f"""
Patient Profile:
- Name: {patient['name']}
- Age: {patient['age']}
- Primary Condition: {patient['condition']}
- Lifestyle: {patient['lifestyle']}
- Diet: {patient['diet']}
- Medical History: {patient['medical_history']}
- Recent Status: {self._summarize_recent_logs(patient_id)}
- Goals: {', '.join(patient['goals'])}
        """
        return context.strip()
    
    def _summarize_recent_logs(self, patient_id: str) -> str:
        """Summarize recent pain/activity logs dynamically"""
        patient = self.patients[patient_id]
        if 'pain_log' not in patient or not patient['pain_log']:
            return "No recent logs"
        
        recent = patient['pain_log'][-3:]  # Last 3 entries
        
        # Dynamically find the key that ends with '_pain' (e.g., knee_pain, back_pain)
        pain_key = next((k for k in recent[0].keys() if k.endswith('_pain')), 'pain_level')
        
        avg_pain = sum(log.get(pain_key, 0) for log in recent) / len(recent)
        avg_mood = sum(log.get('mood', 0) for log in recent) / len(recent)
        
        formatted_pain_name = pain_key.replace('_', ' ')
        return f"Average {formatted_pain_name}: {avg_pain:.1f}/10, Mood: {avg_mood:.1f}/10"

# ENHANCEMENT POINT: Add real-time data ingestion from wearables (Fitbit API, Google Fit)
