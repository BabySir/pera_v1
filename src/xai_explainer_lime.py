import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional

class XAIExplainerLIME:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str = "cuda"):
        """
        Initializes the Explainable AI (XAI) module for Phi-3.
        
        Args:
            model: The loaded Phi-3 model (or LoRA adapted version).
            tokenizer: The tokenizer associated with the model.
            device: Device to run the inference on ("cuda" or "cpu").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # LIME is traditionally for classification. For causal LLMs, 
        # we frame it as the probability of generating a specific target sequence.
        self.lime_explainer = LimeTextExplainer(class_names=["Other", "Target Phrase"])

    def explain_with_lime(self, input_context: str, target_phrase: str, num_samples: int = 150):
        """
        Uses LIME to explain which parts of the `input_context` strongly 
        influence the generation of the `target_phrase`.
        
        Args:
            input_context: The prompt + RAG context fed into the model.
            target_phrase: The specific output phrase you want to explain.
            num_samples: Number of perturbed samples for LIME.
            
        Returns:
            LIME Explanation object that can be visualized using exp.show_in_notebook() 
            or saved as HTML using exp.save_to_file()
        """
        print(f"Generating LIME explanation for target: '{target_phrase}'")
        
        def predictor(texts: List[str]) -> np.ndarray:
            scores = []
            target_ids = self.tokenizer(target_phrase, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            target_first_token = target_ids[0, 0]
            
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    # Generate logits for the next token based on perturbed input
                    outputs = self.model(**inputs)
                    
                # Extract logits for the very last token in the sequence
                next_token_logits = outputs.logits[0, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Probability of generating the first token of our target phrase
                prob_target = probs[target_first_token].item()
                
                # LIME requires a multi-class probability output summing to 1
                scores.append([1.0 - prob_target, prob_target])
                
            return np.array(scores)

        # Generate LIME explanation using the custom predictor
        explanation = self.lime_explainer.explain_instance(
            input_context,
            predictor,
            num_features=10,
            num_samples=num_samples
        )
        return explanation

    def visualize_phi3_attention(self, prompt: str, layer_idx: int = -1, head_idx: int = 0) -> Optional[np.ndarray]:
        """
        Extracts and visualizes the self-attention weights from Phi-3 to see exactly 
        which tokens the model is focusing on during processing.
        
        Args:
            prompt: The input string.
            layer_idx: Which transformer layer to inspect (-1 for the last layer).
            head_idx: Which attention head to inspect.
            
        Returns:
            The raw attention matrix as a numpy array.
        """
        print(f"Visualizing Phi-3 Attention (Layer: {layer_idx}, Head: {head_idx})")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Crucial: Instruct the model to return attention weights
            outputs = self.model(**inputs, output_attentions=True)
            
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            print("Warning: Model config does not support output_attentions. "
                  "Please load the model with config.output_attentions=True")
            return None
            
        # Attention shape: (batch_size, num_heads, seq_len, seq_len)
        # We index into the specific layer and head
        attention_matrix = outputs.attentions[layer_idx][0, head_idx].cpu().numpy()
        
        # Map token IDs back to human-readable strings for the plot axes
        token_ids = inputs.input_ids[0].cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # Clean up special subword markers (like ' ' used in SentencePiece/Phi tokenizers)
        cleaned_tokens = [t.replace(' ', ' ').replace('Ġ', ' ') for t in tokens]

        # Generate Heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            attention_matrix,
            xticklabels=cleaned_tokens,
            yticklabels=cleaned_tokens,
            cmap="magma",
            cbar_kws={'label': 'Attention Weight (Probability)'}
        )
        
        plt.title(f"Phi-3 Cross-Token Attention Map\n(Layer {layer_idx}, Head {head_idx})")
        plt.xlabel("Key Tokens (Being attended to)")
        plt.ylabel("Query Tokens (Paying attention)")
        
        # Improve label readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return attention_matrix

    def export_lime_html(self, explanation, filename: str = "lime_explanation.html"):
        """Saves the LIME explanation as an HTML file for Streamlit/dashboard embedding."""
        explanation.save_to_file(filename)
        print(f"LIME explanation saved to {filename}")