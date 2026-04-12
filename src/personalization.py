import json
import os
from typing import Dict

class PatientDataManager:
    def __init__(self, data_path: str = "./data/sample_patient_data.json"):
        self.data_path = data_path
        self.patients = self._load_data()
    
    def _load_data(self) -> Dict[str, Dict]:
        """Resiliently load patient JSON data into memory."""
        if not os.path.exists(self.data_path):
            return {}
        with open(self.data_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # Handle it whether it's saved correctly as a List or accidentally as a Dict
                if isinstance(data, list):
                    return {p.get('patient_id', 'unknown'): p for p in data if isinstance(p, dict)}
                elif isinstance(data, dict):
                    return {data.get('patient_id', 'unknown'): data}
                return {}
            except json.JSONDecodeError:
                return {}
    
    def get_patient_context(self, patient_id: str) -> str:
        if patient_id not in self.patients:
            return "No specific patient history found. Proceed with general motherly care."
        
        p = self.patients[patient_id]
        
        physical = p.get('physical', {})
        lifestyle = p.get('lifestyle', {})
        mental = p.get('mental', {})

        # Safe-Access Pattern
        if isinstance(lifestyle, dict):
            work_info = lifestyle.get('work', 'Not specified')
        else:
            work_info = lifestyle if lifestyle else 'Not specified'
            
        context = f"""
Patient Profile Context:
- Focus Area: {physical.get('focus', 'General')if isinstance(physical, dict) else physical}
- Current Comfort: {physical.get('comfort', 'Normal')if isinstance(physical, dict) else physical}
- Lifestyle/Work: {lifestyle.get('work', 'Not specified')if isinstance(lifestyle, dict) else lifestyle}
- Energy Level: {lifestyle.get('sleep', 'Average')if isinstance(lifestyle, dict) else lifestyle}
- Mindset: {mental.get('mindset', 'Determined')if isinstance(mental, dict) else mental}
- North Star Goal: {mental.get('goal', 'Recovery') if isinstance(mental, dict) else mental}
        """
        return context.strip()