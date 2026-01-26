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
        """Summarize recent pain/activity logs"""
        patient = self.patients[patient_id]
        if 'pain_log' not in patient or not patient['pain_log']:
            return "No recent logs"
        
        recent = patient['pain_log'][-3:]  # Last 3 entries
        summary = f"Average knee pain: {sum(log['knee_pain'] for log in recent)/len(recent):.1f}/10, "
        summary += f"Mood: {sum(log['mood'] for log in recent)/len(recent):.1f}/10"
        return summary

# ENHANCEMENT POINT: Add real-time data ingestion from wearables (Fitbit API, Google Fit)
