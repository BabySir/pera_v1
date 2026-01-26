# e-Rehabilitation Assistant 🏥🤖

**Complete Personalized e-Rehabilitation Assistant using Nano-LLM + RAG + LoRA + XAI**  
*M.Tech AI/ML Project - Ready for deployment on Google Cloud / Edge devices*

[![Streamlit](https://img.shields.io/badge/Streamlit-FF6B35?logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FF4B4B?logo=huggingface&logoColor=white)](https://huggingface.co)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-3DDC84?logo=chromadb&logoColor=white)](https://docs.trychroma.com)

A production-ready AI-powered rehabilitation assistant that combines **Nano-LLM (Phi-3)**, **RAG**, **LoRA fine-tuning**, and **XAI explainability** for personalized patient care.

**Author**: Sagar Sharma  
**Memory**: ~45MB (LoRA) + 7GB (quantized base)  
**Target**: Edge devices, Google Cloud Run, Jupyter workflows

## ✨ Features

- **Personalized Care**: Patient-specific recommendations using medical history & pain logs
- **RAG-Powered**: Real-time retrieval from medical guidelines + patient data
- **Nano-LLM**: Phi-3 mini (4K) + LoRA fine-tuning (~50MB adapter weights)
- **XAI Explainable**: SHAP/LIME for recommendation transparency
- **Production Ready**: Streamlit UI, ChromaDB vector store, Google Cloud deployable
- **CBT/MI Style**: Cognitive Behavioral Therapy & Motivational Interviewing responses

## 📁 Project Structure

e-Rehabilitation_Assistant/
├── README.md # 📖 Setup instructions
├── requirements.txt # 📦 Dependencies
├── config.yaml # ⚙️ Hyperparameters
├── data/ # 🗂️ Patient data + medical knowledge
│ ├── sample_patient_data.json
│ └── medical_knowledge.json
├── models/ # 🤖 Fine-tuned models [~7.5GB]
│ ├── finetuned_phi/ # LoRA adapter (~50MB)
│ └── phi-3-mini-4k-instruct/ # Base model cache
├── vector_db/ # 🗃️ ChromaDB storage [~50MB]
│ └── chroma/
├── src/ # 🎯 Source code
│ ├── init.py
│ ├── rag_retriever.py # RAG pipeline
│ ├── nano_llm.py # Phi-2 mini + LoRA
│ ├── xai_explainer.py # SHAP/LIME explanations
│ ├── personalization.py # Patient data handler
│ └── app.py # Streamlit UI
└── notebooks/ # 📓 Jupyter demos
├── 01_full_pipeline.ipynb
├── 02_finetune_demo.ipynb
└── 03_rag_demo.ipynb


## 🚀 Quick Start (5 minutes)

### 1. Clone & Setup
```bash
git clone <your-repo>
cd e-Rehabilitation_Assistant
pip install -r requirements.txt
2. Populate Everything (One Command)
bash
chmod +x populate_data.sh
./populate_data.sh
✅ Creates: models/, vector_db/, data/ automatically

3. Launch App
bash
streamlit run src/app.py
Visit http://localhost:8501 🎉

🔧 Quick Demos
Fine-tuning Demo
python
# notebooks/02_finetune_demo.ipynb
from src.nano_llm import NanoLLM

llm = NanoLLM()
training_data = [
    {"prompt": "I feel sad and my knee hurts when I walk.", 
     "response": "I'm sorry to hear... What small movement feels possible right now?"}
]
llm.fine_tune(training_data)  # ✅ Saves to models/finetuned_phi/
RAG Demo
python
# notebooks/03_rag_demo.ipynb
from src.rag_retriever import RAGRetriever

rag = RAGRetriever()  # ✅ Creates vector_db/chroma/
docs = rag.retrieve("knee pain exercises for IT worker", "P001")
📊 Directory Status After Setup
Directory	Size	Contents
models/	~7.5GB	LoRA weights + Phi-3 base
vector_db/	~50MB	ChromaDB + FAISS embeddings
data/	~1KB	Patient JSON + medical knowledge
🧹 Clean/Reset Commands
bash
# 🔄 Fresh start (delete everything)
rm -rf models/ vector_db/
./populate_data.sh

# 🗑️ Reset only vector DB
rm -rf vector_db/
python -c "from src.rag_retriever import RAGRetriever; RAGRetriever()"

# 💾 Backup models
tar -czf models_backup.tar.gz models/
🔍 Monitoring
Add to any notebook:

python
import os
print("models/ size:", sum(os.path.getsize(f'models/{f}') for f in os.listdir('models/') if os.path.isfile(f'models/{f}')) / 1e9, "GB")
print("vector_db/ exists:", os.path.exists('vector_db/chroma'))
☁️ Deployment
Google Cloud Run
bash
gcloud run deploy rehab-assistant \
  --source . \
  --port 8501 \
  --memory 8Gi \
  --cpu 2
Edge Devices
RAM: 8GB minimum

Storage: 8GB (models)

GPU: Optional (CPU inference works)

📚 Sample Data
Patient P001 (Sedentary IT worker, 45yo):

json
{
  "patient_id": "P001",
  "condition": "knee osteoarthritis",
  "lifestyle": "sedentary IT worker",
  "pain_log": [{"date": "2025-12-20", "knee_pain": 7, "mood": 4}]
}
🔬 Technical Stack
Component	Technology	Purpose
LLM	Phi-3 mini 4K + LoRA	Personalized responses
RAG	ChromaDB + FAISS	Medical knowledge retrieval
XAI	SHAP/LIME	Explainable recommendations
UI	Streamlit	Patient/clinician interface
Data	JSON + SQLite	Patient records
🤝 Contributing
Fork the repo

Add your patient data to data/

Fine-tune with your domain data

Submit PR!

📄 License
MIT License - Free for research & clinical use.

🎉 What's Next?
 Add voice input (Whisper)

 Multi-language support

 Wearable integration (Fitbit)

 Clinical trial validation