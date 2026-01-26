# 6. src/nano_llm.py
"""
Nano-LLM with LoRA fine-tuning for personalization
Uses Phi-3-mini (3.8B) - optimal for edge deployment [web:52][web:2]
"""
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from src.rag_retriever import RAGRetriever
import yaml

class NanoLLM:
    def __init__(self, config_path: str = "./config.yaml"):
        self.config = self._load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = self.config['nano_llm']['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization for edge deployment
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, self.lora_config)
        
        self.rag = RAGRetriever()
    
    def _load_config(self, path):
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_response(self, query: str, patient_id: str = "P001") -> str:
        """RAG + Fine-tuned LLM generation"""
        # Retrieve relevant context
        context = self.rag.retrieve(query, patient_id)
        context_str = "\n\n".join(context)
        
        # CBT/MI empathetic system prompt
        system_prompt = """You are an empathetic e-Rehabilitation Assistant using CBT and Motivational Interviewing techniques.
Use simple language, ask open-ended questions, provide clear exercise instructions, and reference patient history.
Always explain your reasoning. End with a question to engage the patient."""
        
        prompt = f"""<|system|>
{system_prompt}

<|user|>
Patient Context: {context_str}

Query: {query}
<|end|>

<|assistant|>"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.config['nano_llm']['temperature'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("<|assistant|>")[-1].strip()
    
    def fine_tune(self, training_data: List[Dict]):
        """Fine-tune with LoRA on patient interaction data"""
        # Convert to instruction format
        def format_example(example):
            return f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}<|end|>"
        
        dataset = Dataset.from_list([{'text': format_example(d)} for d in training_data])
        tokenized_dataset = dataset.map(self._tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir="./models/finetuned_phi",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=10,
            save_steps=100,
            fp16=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
        )
        
        trainer.train()
        self.model.save_pretrained("./models/finetuned_phi")

    def _tokenize_function(self, examples):
        return self.tokenizer(examples['text'], truncation=True, max_length=2048)

# ENHANCEMENT POINT: Add continual learning - incremental LoRA updates without full retraining
